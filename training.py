"""
Training logic for Decloud Trainer
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from safetensors.torch import save_file as save_safetensors
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import config, GRADIENTS_DIR
from model_loader import ModelPackage, DynamicHead

console = Console()


@dataclass
class TrainingResult:
    """Result of training"""
    success: bool
    gradient_dir: Optional[Path] = None
    initial_accuracy: float = 0
    final_accuracy: float = 0
    improvement: float = 0
    epochs_trained: int = 0
    error: Optional[str] = None


class DatasetLoader:
    """Load training data - LABELS only from local, EMBEDDINGS from IPFS"""
    
    @staticmethod
    def load_labels_from_path(dataset_path: str) -> np.ndarray:
        """
        Load ONLY labels from local path
        
        Expected: labels_test.npy or similar
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        labels = None
        
        # Try different naming conventions
        for label_name in ["labels_test.npy", "test_labels.npy", "y_test.npy", "labels.npy"]:
            if (path / label_name).exists():
                labels = np.load(path / label_name)
                break
        
        # Try subdirectory
        if labels is None and (path / "test").exists():
            for name in ["labels.npy", "y.npy", "targets.npy"]:
                if (path / "test" / name).exists():
                    labels = np.load(path / "test" / name)
                    break
        
        if labels is None:
            raise FileNotFoundError(f"Labels not found in {dataset_path}")
        
        return labels
    
    @staticmethod
    def create_dataloader(
        embeddings: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader"""
        dataset = TensorDataset(
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class Trainer:
    """Training engine"""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> float:
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, labels in dataloader:
                embeddings = embeddings.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(embeddings)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total * 100
    
    def train_epoch(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train one epoch, return loss"""
        model.train()
        total_loss = 0
        
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
    ) -> Tuple[nn.Module, float, float]:
        """
        Train model
        Returns: (trained_model, initial_accuracy, final_accuracy)
        """
        model = model.to(self.device)
        
        # Initial evaluation
        initial_acc = self.evaluate(model, test_loader)
        console.print(f"[dim]  Initial accuracy: {initial_acc:.2f}%[/dim]")
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
        ) as progress:
            task = progress.add_task("Training...", total=epochs)
            
            for epoch in range(epochs):
                loss = self.train_epoch(model, train_loader, optimizer, criterion)
                acc = self.evaluate(model, test_loader)
                progress.update(task, advance=1, description=f"Epoch {epoch+1}: loss={loss:.4f} acc={acc:.2f}%")
        
        # Final evaluation
        final_acc = self.evaluate(model, test_loader)
        console.print(f"[green]  Final accuracy: {final_acc:.2f}%[/green]")
        console.print(f"[green]  Improvement: {final_acc - initial_acc:+.2f}%[/green]")
        
        return model, initial_acc, final_acc


def adapt_model_to_input_dim(model_config: Dict[str, Any], input_dim: int, num_classes: int) -> Dict[str, Any]:
    """
    Adapt model config to match input dimension
    Creates new config with correct input_dim for first layer
    """
    new_config = json.loads(json.dumps(model_config))  # Deep copy
    head_config = new_config.get("head", new_config)
    layers = head_config.get("layers", [])
    
    if not layers:
        # No layers defined - create simple MLP
        head_config["layers"] = [
            {"type": "Linear", "params": {"in_features": input_dim, "out_features": 256}},
            {"type": "ReLU", "params": {}},
            {"type": "Dropout", "params": {"p": 0.2}},
            {"type": "Linear", "params": {"in_features": 256, "out_features": num_classes}},
        ]
    else:
        # Update first Linear layer's in_features
        for layer in layers:
            if layer.get("type") == "Linear" and "in_features" in layer.get("params", {}):
                layer["params"]["in_features"] = input_dim
                break
        
        # Update last Linear layer's out_features to match num_classes
        for layer in reversed(layers):
            if layer.get("type") == "Linear" and "out_features" in layer.get("params", {}):
                layer["params"]["out_features"] = num_classes
                break
        
        # Fix intermediate BatchNorm dimensions if needed
        prev_out = input_dim
        for layer in layers:
            params = layer.get("params", {})
            layer_type = layer.get("type")
            
            if layer_type == "Linear":
                if "in_features" not in params:
                    params["in_features"] = prev_out
                prev_out = params.get("out_features", prev_out)
            elif layer_type in ["BatchNorm1d", "LayerNorm"]:
                if "num_features" in params:
                    params["num_features"] = prev_out
                if "normalized_shape" in params:
                    params["normalized_shape"] = prev_out
    
    if "head" in new_config:
        new_config["head"] = head_config
    else:
        new_config = head_config
    
    return new_config


def train_round(
    round_id: int,
    model_config: Dict[str, Any],
    head_weights_path: Path,
    embeddings_path: Path,
    dataset_path: str,
) -> TrainingResult:
    """
    Train a model for a round
    
    EMBEDDINGS come from IPFS (embeddings_path)
    LABELS come from local dataset_path
    
    Args:
        round_id: Round ID
        model_config: Model configuration from IPFS
        head_weights_path: Path to head.safetensors from IPFS
        embeddings_path: Path to embeddings.safetensors from IPFS
        dataset_path: Local path containing labels
    
    Returns:
        TrainingResult with gradient directory
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(device)
    
    try:
        # 1. Load EMBEDDINGS from IPFS package
        console.print(f"[dim]  Loading embeddings from IPFS...[/dim]")
        from safetensors.torch import load_file as load_safetensors
        
        ipfs_data = load_safetensors(str(embeddings_path))
        embeddings = ipfs_data["embeddings"].numpy()
        
        console.print(f"[dim]  Embeddings: {embeddings.shape}[/dim]")
        
        # 2. Load LABELS from local dataset
        console.print(f"[dim]  Loading labels from {dataset_path}...[/dim]")
        labels = DatasetLoader.load_labels_from_path(dataset_path)
        
        console.print(f"[dim]  Labels: {labels.shape}[/dim]")
        
        # Validate shapes match
        if len(embeddings) != len(labels):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(labels)} labels")
        
        input_dim = embeddings.shape[1]
        num_classes = len(np.unique(labels))
        
        console.print(f"[dim]  Samples: {len(embeddings)}, Dim: {input_dim}, Classes: {num_classes}[/dim]")
        
        # 3. Build model
        console.print(f"[dim]  Building model...[/dim]")
        adapted_config = adapt_model_to_input_dim(model_config, input_dim, num_classes)
        head_config = adapted_config.get("head", adapted_config)
        model = DynamicHead(head_config)
        
        # Load pretrained weights if available
        if head_weights_path.exists():
            try:
                state_dict = load_safetensors(str(head_weights_path))
                model.load_state_dict(state_dict, strict=False)
                console.print(f"[dim]  Loaded pretrained weights[/dim]")
            except:
                console.print(f"[dim]  Starting with fresh weights[/dim]")
        
        # 4. Create dataloader (embeddings + labels)
        dataloader = DatasetLoader.create_dataloader(
            embeddings, labels,
            batch_size=config.training_batch_size,
            shuffle=True
        )
        
        # 5. Train
        console.print(f"[dim]  Training for {config.training_epochs} epochs...[/dim]")
        trained_model, initial_acc, final_acc = trainer.train(
            model,
            dataloader,
            dataloader,  # Use same for train/test (we only have test embeddings)
            epochs=config.training_epochs,
            lr=config.learning_rate,
        )
        
        # 6. Save gradient package
        gradient_dir = GRADIENTS_DIR / f"round_{round_id}"
        gradient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(gradient_dir / "config.json", "w") as f:
            json.dump(adapted_config, f, indent=2)
        
        # Save trained weights
        save_safetensors(trained_model.state_dict(), str(gradient_dir / "head.safetensors"))
        
        # Save same embeddings (for validators to verify)
        
        return TrainingResult(
            success=True,
            gradient_dir=gradient_dir,
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            improvement=final_acc - initial_acc,
            epochs_trained=config.training_epochs,
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(success=False, error=str(e))