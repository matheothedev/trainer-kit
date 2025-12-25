"""
DECLOUD Trainer Kit
===================

Train models and earn rewards on DECLOUD.

Usage:
    from decloud_trainer import Trainer
    
    trainer = Trainer()
    trainer.login("your_private_key")
    
    # Join a round
    trainer.join_round(42)
    
    # Train and submit
    trainer.train_and_submit(42)
    
    # Claim reward
    trainer.claim_reward(42)
"""

from .config import Config
from .trainer import Trainer, RoundInfo
from .ipfs import IPFSClient
from .training import Trainer as LocalTrainer, SimpleCNN

__version__ = "0.1.0"
__all__ = [
    "Trainer",
    "Config",
    "RoundInfo",
    "IPFSClient",
    "LocalTrainer",
    "SimpleCNN",
]
