"""
WebSocket listener for Decloud events
Real-time subscription to program logs
"""
import asyncio
import json
import base64
import struct
from typing import Callable, Optional, Any
from dataclasses import dataclass
from enum import Enum

import websockets
from rich.console import Console

from config import PROGRAM_ID, config, DATASET_ID_TO_NAME

console = Console()


WS_ENDPOINTS = {
    "devnet": "wss://api.devnet.solana.com",
    "mainnet": "wss://api.mainnet-beta.solana.com",
    "testnet": "wss://api.testnet.solana.com",
}


class EventType(Enum):
    ROUND_CREATED = "RoundCreated"
    GRADIENT_SUBMITTED = "GradientSubmitted"
    PREVALIDATED = "Prevalidated"
    POSTVALIDATED = "Postvalidated"
    ROUND_FINALIZED = "RoundFinalized"


EVENT_DISCRIMINATORS = {
    bytes([16, 19, 68, 117, 87, 198, 7, 124]): EventType.ROUND_CREATED,
    bytes([165, 33, 17, 57, 235, 165, 150, 201]): EventType.GRADIENT_SUBMITTED,
    bytes([139, 133, 194, 202, 88, 229, 189, 30]): EventType.PREVALIDATED,
    bytes([189, 201, 251, 120, 36, 69, 198, 209]): EventType.POSTVALIDATED,
    bytes([43, 187, 17, 193, 36, 241, 48, 82]): EventType.ROUND_FINALIZED,
}


@dataclass
class RoundCreatedEvent:
    round_id: int
    creator: str
    dataset_id: int
    dataset: str
    reward_amount: int


@dataclass
class GradientSubmittedEvent:
    round_id: int
    trainer: str
    gradient_cid: str
    gradients_count: int


@dataclass
class RoundFinalizedEvent:
    round_id: int


class EventParser:
    """Parse events from program logs"""
    
    @staticmethod
    def parse_log_data(data_b64: str) -> Optional[Any]:
        try:
            data = base64.b64decode(data_b64)
            if len(data) < 8:
                return None
            
            discriminator = data[:8]
            event_type = EVENT_DISCRIMINATORS.get(bytes(discriminator))
            
            if not event_type:
                return None
            
            payload = data[8:]
            
            if event_type == EventType.ROUND_CREATED:
                return EventParser._parse_round_created(payload)
            elif event_type == EventType.ROUND_FINALIZED:
                return EventParser._parse_round_finalized(payload)
            
            return None
        except:
            return None
    
    @staticmethod
    def _parse_round_created(data: bytes) -> Optional[RoundCreatedEvent]:
        try:
            offset = 0
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            creator = base64.b64encode(data[offset:offset+32]).decode()
            offset += 32
            
            dataset_id = data[offset]
            offset += 1
            
            reward_amount = struct.unpack("<Q", data[offset:offset+8])[0]
            
            dataset = DATASET_ID_TO_NAME.get(dataset_id, f"Unknown({dataset_id})")
            
            return RoundCreatedEvent(
                round_id=round_id,
                creator=creator,
                dataset_id=dataset_id,
                dataset=dataset,
                reward_amount=reward_amount,
            )
        except:
            return None
    
    @staticmethod
    def _parse_round_finalized(data: bytes) -> Optional[RoundFinalizedEvent]:
        try:
            round_id = struct.unpack("<Q", data[:8])[0]
            return RoundFinalizedEvent(round_id=round_id)
        except:
            return None


class DecloudWebSocket:
    """WebSocket listener for Decloud events"""
    
    def __init__(self):
        self.ws_url = WS_ENDPOINTS.get(config.network, WS_ENDPOINTS["devnet"])
        self.program_id = PROGRAM_ID
        self.running = False
        self.subscription_id = None
        self.ws = None
        
        # Event handlers
        self.on_round_created: Optional[Callable] = None
        self.on_round_finalized: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
    
    async def connect(self) -> bool:
        try:
            self.ws = await websockets.connect(
                self.ws_url,
                ping_interval=30,
                ping_timeout=10,
            )
            console.print(f"[green]✓ WebSocket connected[/green]")
            return True
        except Exception as e:
            console.print(f"[red]✗ WebSocket failed: {e}[/red]")
            return False
    
    async def subscribe(self) -> bool:
        if not self.ws:
            return False
        
        subscribe_msg = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "logsSubscribe",
            "params": [
                {"mentions": [self.program_id]},
                {"commitment": "confirmed"}
            ]
        }
        
        await self.ws.send(json.dumps(subscribe_msg))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if "result" in data:
            self.subscription_id = data["result"]
            console.print(f"[green]✓ Subscribed to events[/green]")
            return True
        return False
    
    async def disconnect(self):
        self.running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
    
    def _parse_notification(self, msg: dict) -> Optional[Any]:
        try:
            if msg.get("method") != "logsNotification":
                return None
            
            logs = msg.get("params", {}).get("result", {}).get("value", {}).get("logs", [])
            
            for log in logs:
                if log.startswith("Program data: "):
                    data_b64 = log[14:]
                    event = EventParser.parse_log_data(data_b64)
                    if event:
                        return event
            return None
        except:
            return None
    
    async def _handle_event(self, event: Any):
        try:
            if isinstance(event, RoundCreatedEvent):
                if self.on_round_created:
                    if asyncio.iscoroutinefunction(self.on_round_created):
                        await self.on_round_created(event)
                    else:
                        self.on_round_created(event)
            
            elif isinstance(event, RoundFinalizedEvent):
                if self.on_round_finalized:
                    if asyncio.iscoroutinefunction(self.on_round_finalized):
                        await self.on_round_finalized(event)
                    else:
                        self.on_round_finalized(event)
        except Exception as e:
            console.print(f"[red]Event handler error: {e}[/red]")
    
    async def listen(self):
        self.running = True
        
        while self.running:
            try:
                if not self.ws:
                    if not await self.connect():
                        await asyncio.sleep(5)
                        continue
                    if not await self.subscribe():
                        await asyncio.sleep(5)
                        continue
                
                try:
                    msg_raw = await asyncio.wait_for(self.ws.recv(), timeout=60)
                    msg = json.loads(msg_raw)
                    event = self._parse_notification(msg)
                    if event:
                        await self._handle_event(event)
                except asyncio.TimeoutError:
                    continue
                
            except websockets.exceptions.ConnectionClosed:
                console.print("[yellow]WebSocket disconnected, reconnecting...[/yellow]")
                self.ws = None
                await asyncio.sleep(2)
            except Exception as e:
                console.print(f"[red]WebSocket error: {e}[/red]")
                await asyncio.sleep(5)
    
    def stop(self):
        self.running = False
