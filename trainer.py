"""
Core Decloud Trainer - Event-driven training
"""
import asyncio
import time
from pathlib import Path
from typing import Optional, Dict, Any, Set
from dataclasses import dataclass
from collections import defaultdict
from rich.console import Console
from rich.table import Table

from config import config, GRADIENTS_DIR
from ipfs_client import ipfs_client
from pinata_client import pinata_client
from solana_client import SolanaClient, RoundInfo
from websocket_listener import DecloudWebSocket, RoundCreatedEvent, RoundFinalizedEvent
from training import train_round, TrainingResult

console = Console()


@dataclass
class TrainerStats:
    """Trainer statistics"""
    rounds_trained: int = 0
    gradients_submitted: int = 0
    total_improvement: float = 0
    errors: int = 0
    uptime_start: float = 0


class DeCloudTrainer:
    """
    Decloud Trainer - Event-driven training and submission
    """
    
    def __init__(self, private_key: str):
        self.solana = SolanaClient.from_private_key(private_key)
        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.running = False
        self.ws: Optional[DecloudWebSocket] = None
        
        self.stats = TrainerStats()
        
        # Track what we've trained
        self.trained_rounds: Set[int] = set()
        
        # Work queue
        self.training_queue: asyncio.Queue = asyncio.Queue()
        
        console.print(f"[green]✓ Trainer initialized[/green]")
        console.print(f"[dim]  Wallet: {self.solana.pubkey}[/dim]")
        console.print(f"[dim]  Device: {self.device}[/dim]")
        console.print(f"[dim]  Network: {config.network}[/dim]")
    
    def get_balance(self) -> float:
        return self.solana.get_balance() / 1e9
    
    # ═══════════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════════
    
    async def train_and_submit(self, round_id: int) -> bool:
        """Train and submit gradient for a round"""
        
        if round_id in self.trained_rounds:
            console.print(f"[dim]Round {round_id} already trained[/dim]")
            return False
        
        round_info = self.solana.get_round(round_id)
        if not round_info:
            console.print(f"[red]Round {round_id} not found[/red]")
            return False
        
        if round_info.status != "Active":
            console.print(f"[dim]Round {round_id} not active[/dim]")
            return False
        
        # Check reward threshold
        reward_sol = round_info.reward_amount / 1e9
        if reward_sol < config.min_reward:
            console.print(f"[dim]Round {round_id}: reward {reward_sol:.4f} SOL < min {config.min_reward}[/dim]")
            return False
        
        # Check if we have dataset
        dataset_path = config.get_dataset_path(round_info.dataset)
        if not dataset_path:
            console.print(f"[dim]Round {round_id}: no local dataset for {round_info.dataset}[/dim]")
            return False
        
        # Check prevalidation exists
        if round_info.pre_count == 0:
            console.print(f"[dim]Round {round_id}: waiting for prevalidation[/dim]")
            return False
        
        # Check if already submitted
        if self.solana.has_submitted_gradient(round_id):
            self.trained_rounds.add(round_id)
            console.print(f"[dim]Round {round_id}: already submitted[/dim]")
            return False
        
        console.print(f"\n[cyan]⚡ Training for Round #{round_id}[/cyan]")
        console.print(f"[dim]   Dataset: {round_info.dataset}[/dim]")
        console.print(f"[dim]   Reward: {reward_sol:.4f} SOL[/dim]")
        
        try:
            # Download base model from IPFS
            console.print(f"[dim]  Downloading base model...[/dim]")
            model_path = await ipfs_client.download_model_package(round_info.model_cid)
            if not model_path:
                console.print(f"[red]Failed to download model[/red]")
                return False
            
            # Load config
            import json
            with open(model_path / "config.json", "r") as f:
                model_config = json.load(f)
            
            # Train
            result = train_round(
                round_id=round_id,
                model_config=model_config,
                head_weights_path=model_path / "head.safetensors",
                embeddings_path=model_path / "embeddings.safetensors",
                dataset_path=dataset_path,
            )
            
            if not result.success:
                console.print(f"[red]Training failed: {result.error}[/red]")
                self.stats.errors += 1
                return False
            
            console.print(f"[green]✓ Training complete! Improvement: {result.improvement:+.2f}%[/green]")
            
            # Upload to IPFS via Pinata
            console.print(f"[dim]  Uploading gradient to IPFS...[/dim]")
            gradient_cid = await pinata_client.upload_gradient_package(result.gradient_dir, round_id)
            
            if not gradient_cid:
                console.print(f"[red]Failed to upload to IPFS[/red]")
                return False
            
            console.print(f"[green]✓ Uploaded: {gradient_cid}[/green]")
            
            # Submit to blockchain
            console.print(f"[dim]  Submitting to blockchain...[/dim]")
            tx = self.solana.submit_gradient(round_id, gradient_cid)
            
            self.trained_rounds.add(round_id)
            self.stats.rounds_trained += 1
            self.stats.gradients_submitted += 1
            self.stats.total_improvement += result.improvement
            
            console.print(f"[green]✓ Gradient submitted![/green]")
            console.print(f"[dim]   TX: {tx}[/dim]")
            console.print(f"[dim]   CID: {gradient_cid}[/dim]")
            
            return True
            
        except Exception as e:
            self.stats.errors += 1
            console.print(f"[red]Error: {e}[/red]")
            return False
    
    # ═══════════════════════════════════════════════════════════════
    # Event Handlers
    # ═══════════════════════════════════════════════════════════════
    
    async def on_round_created(self, event: RoundCreatedEvent):
        """Handle new round"""
        reward_sol = event.reward_amount / 1e9
        
        console.print(f"\n[yellow]📢 New Round #{event.round_id}[/yellow]")
        console.print(f"[dim]   Dataset: {event.dataset}[/dim]")
        console.print(f"[dim]   Reward: {reward_sol:.4f} SOL[/dim]")
        
        # Check if we can train
        if reward_sol < config.min_reward:
            console.print(f"[dim]   ⏭ Skipping (reward too low)[/dim]")
            return
        
        if not config.can_train(event.dataset):
            console.print(f"[dim]   ⏭ Skipping (no local dataset)[/dim]")
            return
        
        # Queue for training (wait for prevalidation)
        await self.training_queue.put(event.round_id)
    
    async def on_round_finalized(self, event: RoundFinalizedEvent):
        """Handle round finalized"""
        console.print(f"\n[green]🏁 Round #{event.round_id} finalized![/green]")
        
        if event.round_id in self.trained_rounds:
            console.print(f"[yellow]   💰 You have rewards! Run: decloud-trainer claim {event.round_id}[/yellow]")
    
    # ═══════════════════════════════════════════════════════════════
    # Workers
    # ═══════════════════════════════════════════════════════════════
    
    async def training_worker(self):
        """Worker that processes training queue"""
        while self.running:
            try:
                round_id = await asyncio.wait_for(self.training_queue.get(), timeout=1.0)
                
                # Wait a bit for prevalidation
                await asyncio.sleep(5)
                
                await self.train_and_submit(round_id)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                console.print(f"[red]Training worker error: {e}[/red]")
    
    # ═══════════════════════════════════════════════════════════════
    # Initial Scan
    # ═══════════════════════════════════════════════════════════════
    
    async def scan_existing_rounds(self):
        """Scan existing rounds on startup"""
        console.print("\n[cyan]🔍 Scanning existing rounds...[/cyan]")
        
        rounds = self.solana.get_active_rounds()
        queued = 0
        
        for round_info in rounds:
            reward_sol = round_info.reward_amount / 1e9
            
            if reward_sol < config.min_reward:
                continue
            
            if not config.can_train(round_info.dataset):
                continue
            
            if self.solana.has_submitted_gradient(round_info.id):
                self.trained_rounds.add(round_info.id)
                continue
            
            if round_info.pre_count > 0:  # Has prevalidation
                await self.training_queue.put(round_info.id)
                queued += 1
        
        console.print(f"[green]✓ Found {queued} rounds to train[/green]")
    
    # ═══════════════════════════════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════════════════════════════
    
    async def start(self):
        """Start the trainer"""
        self.running = True
        self.stats.uptime_start = time.time()
        
        self.show_status()
        
        if not config.dataset_paths:
            console.print("\n[red]✗ No datasets configured![/red]")
            console.print("[dim]Run: decloud-trainer dataset set <name> <path>[/dim]")
            return
        
        if not config.has_pinata():
            console.print("\n[red]✗ Pinata not configured![/red]")
            console.print("[dim]Run: decloud-trainer setup[/dim]")
            return
        
        console.print("\n[cyan]🔌 Connecting to WebSocket...[/cyan]")
        self.ws = DecloudWebSocket()
        
        self.ws.on_round_created = self.on_round_created
        self.ws.on_round_finalized = self.on_round_finalized
        
        if not await self.ws.connect():
            console.print("[red]✗ WebSocket connection failed[/red]")
            return
        
        if not await self.ws.subscribe():
            console.print("[red]✗ Subscription failed[/red]")
            return
        
        await self.scan_existing_rounds()
        
        console.print("\n[green]🚀 Trainer running! Listening for rounds...[/green]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")
        
        workers = [
            asyncio.create_task(self.ws.listen()),
            asyncio.create_task(self.training_worker()),
        ]
        
        try:
            await asyncio.gather(*workers)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await self.ws.disconnect()
    
    def stop(self):
        self.running = False
    
    # ═══════════════════════════════════════════════════════════════
    # Rewards
    # ═══════════════════════════════════════════════════════════════
    
    def claim_reward(self, round_id: int) -> Dict[str, Any]:
        """Claim trainer reward"""
        round_info = self.solana.get_round(round_id)
        if not round_info:
            return {"error": "Round not found"}
        
        if round_info.status != "Finalized":
            return {"error": f"Round not finalized: {round_info.status}"}
        
        gradient = self.solana.get_gradient(round_id, self.solana.pubkey)
        if not gradient:
            return {"error": "No gradient submitted for this round"}
        
        if gradient.reward_claimed:
            return {"error": "Already claimed"}
        
        try:
            tx = self.solana.claim_trainer(round_id)
            return {"success": True, "tx": tx}
        except Exception as e:
            return {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════
    # Status
    # ═══════════════════════════════════════════════════════════════
    
    def show_status(self):
        """Display trainer status"""
        try:
            balance = self.get_balance()
        except:
            balance = -1
        
        table = Table(title="🏋️ Trainer Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Wallet", str(self.solana.pubkey))
        table.add_row("Balance", f"{balance:.4f} SOL" if balance >= 0 else "Error")
        table.add_row("Network", config.network)
        table.add_row("Device", self.device)
        table.add_row("Min Reward", f"{config.min_reward} SOL")
        table.add_row("Datasets Configured", str(len(config.dataset_paths)))
        table.add_row("Rounds Trained", str(self.stats.rounds_trained))
        table.add_row("Avg Improvement", f"{self.stats.total_improvement / max(1, self.stats.rounds_trained):.2f}%")
        table.add_row("Pinata", "✓ Configured" if config.has_pinata() else "✗ Not configured")
        
        console.print(table)
        
        if config.dataset_paths:
            console.print("\n[cyan]Configured Datasets:[/cyan]")
            for name, path in config.dataset_paths.items():
                console.print(f"  {name} → {path}")
    
    def show_rounds(self, limit: int = 10):
        """Display active rounds"""
        rounds = self.solana.get_active_rounds()
        
        table = Table(title=f"Active Rounds ({len(rounds)} total)")
        table.add_column("ID", style="cyan")
        table.add_column("Dataset", style="yellow")
        table.add_column("Reward", style="green")
        table.add_column("Pre", style="blue")
        table.add_column("Gradients", style="magenta")
        table.add_column("Can Train", style="white")
        
        for round_info in rounds[:limit]:
            reward_sol = round_info.reward_amount / 1e9
            can_train = config.can_train(round_info.dataset) and reward_sol >= config.min_reward
            submitted = self.solana.has_submitted_gradient(round_info.id)
            
            if submitted:
                status = "[green]✓ submitted[/green]"
            elif can_train:
                status = "[yellow]⏳ ready[/yellow]"
            else:
                status = "[dim]⏭ skip[/dim]"
            
            table.add_row(
                str(round_info.id),
                round_info.dataset,
                f"{reward_sol:.4f}",
                str(round_info.pre_count),
                str(round_info.gradients_count),
                status,
            )
        
        console.print(table)
