"""
Solana client for interacting with Decloud contract
Trainer operations: submit_gradient, claim_trainer
"""
import struct
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import base58
import time

from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import ID as SYSTEM_PROGRAM_ID
from solders.instruction import Instruction, AccountMeta
from solders.transaction import Transaction
from solders.message import Message
from solana.rpc.api import Client
from solana.rpc.commitment import Confirmed
from solana.rpc.types import TxOpts

from config import PROGRAM_ID, TREASURY, config, DATASET_ID_TO_NAME


@dataclass
class RoundInfo:
    """Round information from blockchain"""
    id: int
    creator: str
    model_cid: str
    dataset: str
    dataset_id: int
    reward_amount: int
    created_at: int
    status: str
    pre_count: int
    pre_accuracy_sum: int
    gradients_count: int
    total_validations: int
    total_improvement: int
    bump: int
    vault_bump: int


@dataclass
class GradientInfo:
    """Gradient information"""
    round_id: int
    trainer: str
    cid: str
    post_count: int
    post_accuracy_sum: int
    improvement: int
    reward_claimed: bool


class SolanaClient:
    """Client for Decloud Solana program - Trainer operations"""
    
    DISCRIMINATORS = {
        "submit_gradient": bytes([52, 174, 224, 247, 246, 136, 203, 99]),
        "claim_trainer": bytes([55, 24, 189, 180, 35, 249, 70, 75]),
    }
    
    # Account discriminators for parsing validation
    ACCOUNT_DISCRIMINATORS = {
        "Round": bytes([87, 127, 165, 51, 73, 78, 116, 174]),
        "Gradient": bytes([173, 254, 210, 185, 231, 180, 152, 152]),
    }
    
    def __init__(self, keypair: Optional[Keypair] = None):
        self.program_id = Pubkey.from_string(PROGRAM_ID)
        self.treasury = Pubkey.from_string(TREASURY)
        self.client = Client(config.rpc_url)
        self.keypair = keypair
    
    @classmethod
    def from_private_key(cls, private_key: str) -> "SolanaClient":
        """Create client from base58 private key"""
        secret = base58.b58decode(private_key)
        keypair = Keypair.from_bytes(secret)
        return cls(keypair)
    
    @property
    def pubkey(self) -> Optional[Pubkey]:
        return self.keypair.pubkey() if self.keypair else None
    
    def get_balance(self) -> int:
        """Get SOL balance in lamports"""
        if not self.pubkey:
            return 0
        response = self.client.get_balance(self.pubkey, commitment=Confirmed)
        return response.value
    
    # ═══════════════════════════════════════════════════════════════
    # PDA Derivation
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_counter_pda(self) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address([b"round_counter"], self.program_id)
    
    def get_round_pda(self, round_id: int) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address(
            [b"round", round_id.to_bytes(8, "little")],
            self.program_id
        )
    
    def get_vault_pda(self, round_id: int) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address(
            [b"vault", round_id.to_bytes(8, "little")],
            self.program_id
        )
    
    def get_gradient_pda(self, round_id: int, trainer: Pubkey) -> Tuple[Pubkey, int]:
        return Pubkey.find_program_address(
            [b"gradient", round_id.to_bytes(8, "little"), bytes(trainer)],
            self.program_id
        )
    
    # ═══════════════════════════════════════════════════════════════
    # Read Operations
    # ═══════════════════════════════════════════════════════════════
    
    def get_round_count(self) -> int:
        pda, _ = self.get_round_counter_pda()
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return 0
        
        data = response.value.data
        count = struct.unpack("<Q", data[8:16])[0]
        return count
    
    def get_round(self, round_id: int) -> Optional[RoundInfo]:
        pda, _ = self.get_round_pda(round_id)
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return None
        
        data = bytes(response.value.data)
        return self._parse_round(data)
    
    def _parse_round(self, data: bytes) -> Optional[RoundInfo]:
        """Parse round data. Returns None if parsing fails (old/incompatible format)."""
        try:
            # Verify discriminator
            discriminator = data[:8]
            if discriminator != self.ACCOUNT_DISCRIMINATORS.get("Round"):
                return None
            
            offset = 8
            
            id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            creator = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            model_cid_bytes = data[offset:offset+64]
            offset += 64
            model_cid_len = data[offset]
            offset += 1
            model_cid = model_cid_bytes[:model_cid_len].decode("utf-8", errors="ignore")
            
            dataset_id = data[offset]
            offset += 1
            dataset = DATASET_ID_TO_NAME.get(dataset_id, f"Unknown({dataset_id})")
            
            reward_amount = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            created_at = struct.unpack("<q", data[offset:offset+8])[0]
            offset += 8
            
            status_id = data[offset]
            offset += 1
            status_map = {0: "Active", 1: "Finalized", 2: "Cancelled"}
            status = status_map.get(status_id, "Unknown")
            
            pre_count = data[offset]
            offset += 1
            
            pre_accuracy_sum = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            gradients_count = data[offset]
            offset += 1
            
            total_validations = struct.unpack("<H", data[offset:offset+2])[0]
            offset += 2
            
            total_improvement = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            bump = data[offset]
            offset += 1
            
            vault_bump = data[offset]
            
            return RoundInfo(
                id=id, creator=creator, model_cid=model_cid, dataset=dataset,
                dataset_id=dataset_id, reward_amount=reward_amount, created_at=created_at,
                status=status, pre_count=pre_count, pre_accuracy_sum=pre_accuracy_sum,
                gradients_count=gradients_count, total_validations=total_validations,
                total_improvement=total_improvement, bump=bump, vault_bump=vault_bump,
            )
        except Exception:
            # Failed to parse - likely old/incompatible format
            return None
    
    def get_gradient(self, round_id: int, trainer: Pubkey) -> Optional[GradientInfo]:
        pda, _ = self.get_gradient_pda(round_id, trainer)
        response = self.client.get_account_info(pda, commitment=Confirmed)
        
        if response.value is None:
            return None
        
        data = bytes(response.value.data)
        return self._parse_gradient(data)
    
    def _parse_gradient(self, data: bytes) -> Optional[GradientInfo]:
        """Parse gradient data. Returns None if parsing fails (old/incompatible format)."""
        try:
            # Verify discriminator
            discriminator = data[:8]
            if discriminator != self.ACCOUNT_DISCRIMINATORS.get("Gradient"):
                return None
            
            offset = 8
            
            round_id = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            trainer = base58.b58encode(data[offset:offset+32]).decode()
            offset += 32
            
            cid_bytes = data[offset:offset+64]
            offset += 64
            cid_len = data[offset]
            offset += 1
            cid = cid_bytes[:cid_len].decode("utf-8", errors="ignore")
            
            post_count = data[offset]
            offset += 1
            
            post_accuracy_sum = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            improvement = struct.unpack("<Q", data[offset:offset+8])[0]
            offset += 8
            
            reward_claimed = bool(data[offset])
            
            return GradientInfo(
                round_id=round_id, trainer=trainer, cid=cid,
                post_count=post_count, post_accuracy_sum=post_accuracy_sum,
                improvement=improvement, reward_claimed=reward_claimed,
            )
        except Exception:
            # Failed to parse - likely old/incompatible format
            return None
    
    def get_active_rounds(self) -> List[RoundInfo]:
        """Get all active rounds"""
        rounds = []
        count = self.get_round_count()
        
        for i in range(count):
            round_info = self.get_round(i)
            if round_info and round_info.status == "Active":
                rounds.append(round_info)
        
        return rounds
    
    def has_submitted_gradient(self, round_id: int) -> bool:
        """Check if we already submitted gradient"""
        if not self.keypair:
            return False
        gradient = self.get_gradient(round_id, self.keypair.pubkey())
        return gradient is not None
    
    # ═══════════════════════════════════════════════════════════════
    # Write Operations
    # ═══════════════════════════════════════════════════════════════
    
    def _send_transaction(self, instruction: Instruction) -> str:
        """Send transaction with retry on blockhash error"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        for attempt in range(3):
            try:
                recent_blockhash = self.client.get_latest_blockhash(commitment=Confirmed).value.blockhash
                
                message = Message.new_with_blockhash(
                    [instruction],
                    self.keypair.pubkey(),
                    recent_blockhash
                )
                
                tx = Transaction.new_unsigned(message)
                tx.sign([self.keypair], recent_blockhash)
                
                response = self.client.send_transaction(
                    tx,
                    opts=TxOpts(skip_preflight=False, preflight_commitment=Confirmed)
                )
                return str(response.value)
            
            except Exception as e:
                if "Blockhash not found" in str(e) and attempt < 2:
                    time.sleep(1)
                    continue
                raise
        
        raise Exception("Failed after 3 attempts")
    
    def submit_gradient(self, round_id: int, gradient_cid: str) -> str:
        """Submit gradient for a round"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        gradient_pda, _ = self.get_gradient_pda(round_id, self.keypair.pubkey())
        
        # Build instruction data
        data = self.DISCRIMINATORS["submit_gradient"]
        data += struct.pack("<Q", round_id)
        
        # String: 4 bytes length + utf8 bytes
        cid_bytes = gradient_cid.encode("utf-8")
        data += struct.pack("<I", len(cid_bytes))
        data += cid_bytes
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=True),
            AccountMeta(gradient_pda, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)
    
    def claim_trainer(self, round_id: int) -> str:
        """Claim trainer reward"""
        if not self.keypair:
            raise ValueError("No keypair configured")
        
        round_pda, _ = self.get_round_pda(round_id)
        gradient_pda, _ = self.get_gradient_pda(round_id, self.keypair.pubkey())
        vault_pda, _ = self.get_vault_pda(round_id)
        
        data = self.DISCRIMINATORS["claim_trainer"]
        data += struct.pack("<Q", round_id)
        
        accounts = [
            AccountMeta(round_pda, is_signer=False, is_writable=False),
            AccountMeta(gradient_pda, is_signer=False, is_writable=True),
            AccountMeta(vault_pda, is_signer=False, is_writable=True),
            AccountMeta(self.treasury, is_signer=False, is_writable=True),
            AccountMeta(self.keypair.pubkey(), is_signer=True, is_writable=True),
            AccountMeta(SYSTEM_PROGRAM_ID, is_signer=False, is_writable=False),
        ]
        
        instruction = Instruction(self.program_id, data, accounts)
        return self._send_transaction(instruction)