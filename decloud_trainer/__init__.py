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

Custom Datasets:
    from decloud_trainer.custom_data import CustomDataset, CustomImageFolder
    
    # From folder structure
    dataset = CustomImageFolder("./my_images", input_size=224)
    
    # From tensors
    dataset = CustomDataset(X_tensor, y_tensor)
    
    # From CSV
    dataset = CustomDataset.from_csv("data.csv", target_column="label")
"""

from .config import Config
from .trainer import Trainer, RoundInfo
from .ipfs import IPFSClient
from .training import Trainer as LocalTrainer, SimpleCNN, detect_architecture

# Custom dataset support
from .custom_data import (
    CustomDataset,
    CustomImageFolder,
    CustomTextDataset,
    CustomAudioDataset,
    CustomTimeSeriesDataset,
    create_custom_config,
    load_custom_dataset,
)

__version__ = "0.2.0"
__all__ = [
    # Core
    "Trainer",
    "Config",
    "RoundInfo",
    "IPFSClient",
    "LocalTrainer",
    "SimpleCNN",
    "detect_architecture",
    # Custom datasets
    "CustomDataset",
    "CustomImageFolder",
    "CustomTextDataset",
    "CustomAudioDataset",
    "CustomTimeSeriesDataset",
    "create_custom_config",
    "load_custom_dataset",
]