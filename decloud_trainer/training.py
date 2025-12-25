"""
Training Module
===============

Train models and compute gradients.
"""

import os
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Simple CNN for image classification (CIFAR-10, MNIST, etc.)"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ═══════════════════════════════════════════════════════════════
# DATASET LOADING
# ═══════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    "cifar10": {
        "class": "CIFAR10",
        "num_classes": 10,
        "in_channels": 3,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "cifar100": {
        "class": "CIFAR100",
        "num_classes": 100,
        "in_channels": 3,
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "mnist": {
        "class": "MNIST",
        "num_classes": 10,
        "in_channels": 1,
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "fashionmnist": {
        "class": "FashionMNIST",
        "num_classes": 10,
        "in_channels": 1,
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]) if TORCHVISION_AVAILABLE else None,
    },
}


def get_dataset(name: str, data_dir: str = "./data", train: bool = True) -> DataLoader:
    """
    Get dataset by name.
    
    Args:
        name: Dataset name (cifar10, mnist, etc.)
        data_dir: Directory to store/load data
        train: Use training set (True) or test set (False)
    
    Returns:
        DataLoader
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for dataset loading")
    
    name_lower = name.lower().replace("_", "").replace("-", "")
    
    if name_lower not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(DATASET_CONFIG.keys())}")
    
    config = DATASET_CONFIG[name_lower]
    dataset_class = getattr(torchvision.datasets, config["class"])
    
    dataset = dataset_class(
        root=data_dir,
        train=train,
        download=True,
        transform=config["transform"]
    )
    
    return dataset, config


def create_dataloader(
    dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 2
) -> DataLoader:
    """Create DataLoader from dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

class Trainer:
    """
    Local trainer for federated learning.
    
    Trains model on local data and computes gradients.
    """
    
    def __init__(
        self,
        device: str = "auto",
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
    
    def load_model(self, model_path: str, dataset_name: str) -> nn.Module:
        """
        Load model from file.
        
        Args:
            model_path: Path to model state dict
            dataset_name: Dataset name to determine model architecture
        
        Returns:
            Loaded model
        """
        name_lower = dataset_name.lower().replace("_", "").replace("-", "")
        config = DATASET_CONFIG.get(name_lower, {"num_classes": 10, "in_channels": 3})
        
        model = SimpleCNN(
            num_classes=config["num_classes"],
            in_channels=3  # We convert all to 3 channels
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def train(
        self,
        model: nn.Module,
        dataset_name: str,
        epochs: int = 1,
        data_dir: str = "./data",
        max_batches: Optional[int] = None,
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Train model on local data.
        
        Args:
            model: Model to train
            dataset_name: Dataset to use
            epochs: Number of epochs
            data_dir: Data directory
            max_batches: Limit batches per epoch (for testing)
        
        Returns:
            Tuple of (trained_model, gradients)
        """
        # Save original weights
        original_state = copy.deepcopy(model.state_dict())
        
        # Load dataset
        dataset, config = get_dataset(dataset_name, data_dir, train=True)
        dataloader = create_dataloader(dataset, self.batch_size)
        
        # Training setup
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Stats
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_samples += len(target)
                
                if batch_idx % 50 == 0:
                    print(f"\r   Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}", end="", flush=True)
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            accuracy = 100.0 * epoch_correct / epoch_samples
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"\r   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        # Compute gradients (difference from original weights)
        gradients = {}
        trained_state = model.state_dict()
        
        for key in original_state:
            gradients[key] = trained_state[key] - original_state[key]
        
        # Final stats
        final_accuracy = 100.0 * total_correct / total_samples
        print(f"   ✓ Training complete! Final accuracy: {final_accuracy:.2f}%")
        
        return model, gradients
    
    def save_gradients(self, gradients: Dict[str, torch.Tensor], output_path: str):
        """Save gradients to file"""
        torch.save(gradients, output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"   ✓ Gradients saved: {size_kb:.1f} KB")
    
    def evaluate(
        self,
        model: nn.Module,
        dataset_name: str,
        data_dir: str = "./data"
    ) -> float:
        """
        Evaluate model on test set.
        
        Returns:
            Accuracy (0-100)
        """
        dataset, config = get_dataset(dataset_name, data_dir, train=False)
        dataloader = create_dataloader(dataset, self.batch_size, shuffle=False)
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += len(target)
        
        accuracy = 100.0 * correct / total
        return accuracy
