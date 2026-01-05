"""
Custom Dataset Support for DECLOUD Trainer
==========================================

Allows trainers to use their own local datasets instead of downloading standard ones.

Usage:
    from decloud_trainer.custom_data import CustomDataset, CustomImageFolder
    
    # Option 1: From folder structure (like ImageFolder)
    dataset = CustomImageFolder(
        root="./my_data/train",
        input_size=224,
        num_classes=10
    )
    
    # Option 2: From tensors
    dataset = CustomDataset(
        X=my_images,  # tensor [N, C, H, W]
        y=my_labels,  # tensor [N]
        num_classes=10
    )
    
    # Option 3: From numpy arrays
    dataset = CustomDataset.from_numpy(X_np, y_np)
    
    # Option 4: From CSV file
    dataset = CustomDataset.from_csv("data.csv", target_column="label")
"""

import os
from typing import Optional, Tuple, List, Callable, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class CustomDataset(Dataset):
    """
    Custom dataset from tensors or arrays.
    
    Args:
        X: Input features (tensor or numpy array)
        y: Labels (tensor or numpy array)
        num_classes: Number of classes (auto-detected if None)
        transform: Optional transform to apply
        task: Type of task ('classification', 'regression')
    """
    
    def __init__(
        self,
        X: Union[torch.Tensor, np.ndarray],
        y: Union[torch.Tensor, np.ndarray],
        num_classes: int = None,
        transform: Callable = None,
        task: str = "classification",
    ):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        
        self.X = X
        self.y = y.long() if task == "classification" else y.float()
        self.transform = transform
        self.task = task
        self.num_classes = num_classes or int(y.max().item() + 1)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    @classmethod
    def from_numpy(cls, X: np.ndarray, y: np.ndarray, **kwargs) -> "CustomDataset":
        """Create dataset from numpy arrays"""
        return cls(X, y, **kwargs)
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        target_column: str,
        feature_columns: List[str] = None,
        delimiter: str = ",",
        **kwargs
    ) -> "CustomDataset":
        """
        Create dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target/label column
            feature_columns: List of feature column names (all except target if None)
            delimiter: CSV delimiter
        """
        import csv
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            rows = list(reader)
        
        if not feature_columns:
            feature_columns = [k for k in rows[0].keys() if k != target_column]
        
        X = np.array([[float(row[col]) for col in feature_columns] for row in rows])
        y = np.array([float(row[target_column]) for row in rows])
        
        return cls(X, y, **kwargs)
    
    @classmethod
    def from_tensors(cls, X: torch.Tensor, y: torch.Tensor, **kwargs) -> "CustomDataset":
        """Create dataset from PyTorch tensors"""
        return cls(X, y, **kwargs)


class CustomImageFolder(Dataset):
    """
    Custom image dataset from folder structure.
    
    Expected folder structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                img4.jpg
    
    Args:
        root: Root directory path
        input_size: Target image size (will be resized)
        num_classes: Number of classes (auto-detected if None)
        extensions: Valid image extensions
        transform: Custom transform (overrides default)
    """
    
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    def __init__(
        self,
        root: str,
        input_size: int = 224,
        num_classes: int = None,
        extensions: set = None,
        transform: Callable = None,
    ):
        self.root = Path(root)
        self.input_size = input_size
        self.extensions = extensions or self.VALID_EXTENSIONS
        
        # Find all classes (subdirectories)
        self.classes = sorted([
            d.name for d in self.root.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.num_classes = num_classes or len(self.classes)
        
        # Find all images
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))
        
        # Setup transform
        if transform:
            self.transform = transform
        else:
            from torchvision import transforms
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        print(f"📁 CustomImageFolder loaded:")
        print(f"   Root: {self.root}")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Samples: {len(self.samples)}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CustomTextDataset(Dataset):
    """
    Custom text dataset for classification.
    
    Args:
        texts: List of text strings
        labels: List of integer labels
        max_length: Maximum sequence length
        tokenizer: Custom tokenizer (simple whitespace tokenizer if None)
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        max_length: int = 256,
        vocab_size: int = 30000,
        tokenizer: Callable = None,
    ):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.num_classes = max(labels) + 1
        
        # Build simple vocabulary if no tokenizer provided
        if tokenizer is None:
            self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
            idx = 2
            for text in texts:
                for word in text.lower().split():
                    if word not in self.word_to_idx and idx < vocab_size:
                        self.word_to_idx[word] = idx
                        idx += 1
            self.tokenizer = self._simple_tokenize
        else:
            self.tokenizer = tokenizer
    
    def _simple_tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenizer"""
        tokens = []
        for word in text.lower().split()[:self.max_length]:
            idx = self.word_to_idx.get(word, 1)  # 1 = UNK
            tokens.append(idx)
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(0)  # 0 = PAD
        
        return tokens[:self.max_length]
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        tokens = self.tokenizer(text)
        
        return torch.tensor(tokens, dtype=torch.long), label
    
    @classmethod
    def from_csv(
        cls,
        filepath: str,
        text_column: str,
        label_column: str,
        **kwargs
    ) -> "CustomTextDataset":
        """Load from CSV file"""
        import csv
        
        texts = []
        labels = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[text_column])
                labels.append(int(row[label_column]))
        
        return cls(texts, labels, **kwargs)


class CustomAudioDataset(Dataset):
    """
    Custom audio dataset from folder or file list.
    
    Args:
        audio_paths: List of audio file paths
        labels: List of integer labels
        sample_rate: Target sample rate
        duration: Target duration in seconds
    """
    
    def __init__(
        self,
        audio_paths: List[str],
        labels: List[int],
        sample_rate: int = 16000,
        duration: float = 1.0,
    ):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.num_classes = max(labels) + 1
        
        try:
            import torchaudio
            self.torchaudio = torchaudio
        except ImportError:
            raise ImportError("torchaudio required for audio datasets")
    
    def __len__(self) -> int:
        return len(self.audio_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.audio_paths[idx]
        label = self.labels[idx]
        
        waveform, sr = self.torchaudio.load(path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = self.torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Pad or truncate
        if waveform.shape[1] < self.num_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.num_samples - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.num_samples]
        
        # Convert to mel spectrogram
        mel = self.torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=64,
        )(waveform)
        
        return mel, label


class CustomTimeSeriesDataset(Dataset):
    """
    Custom time series dataset.
    
    Args:
        sequences: Tensor of shape [N, seq_len, features] or [N, seq_len]
        labels: Labels for classification or target values for regression
        task: 'classification' or 'regression'
    """
    
    def __init__(
        self,
        sequences: Union[torch.Tensor, np.ndarray],
        labels: Union[torch.Tensor, np.ndarray],
        task: str = "classification",
    ):
        if isinstance(sequences, np.ndarray):
            sequences = torch.from_numpy(sequences).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        
        # Ensure [N, features, seq_len] format for Conv1d
        if sequences.dim() == 2:
            sequences = sequences.unsqueeze(1)  # [N, 1, seq_len]
        elif sequences.dim() == 3 and sequences.shape[1] > sequences.shape[2]:
            sequences = sequences.permute(0, 2, 1)  # [N, features, seq_len]
        
        self.sequences = sequences
        self.labels = labels.long() if task == "classification" else labels.float()
        self.task = task
        self.num_classes = int(labels.max().item() + 1) if task == "classification" else 1
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIG FOR CUSTOM DATA
# ═══════════════════════════════════════════════════════════════════════════════

def create_custom_config(
    num_classes: int,
    task: str = "image",
    in_channels: int = 3,
    input_dim: int = None,
    input_size: int = 224,
    architecture: str = None,
) -> dict:
    """
    Create a dataset config for custom data.
    
    Args:
        num_classes: Number of classes
        task: Task type ('image', 'text', 'tabular', 'audio', 'timeseries')
        in_channels: Number of input channels (for images)
        input_dim: Input dimension (for tabular/graph data)
        input_size: Input size (for images)
        architecture: Preferred architecture (auto-selected if None)
    
    Returns:
        Dataset config dict
    """
    
    # Auto-select architecture
    if architecture is None:
        arch_map = {
            "image": "resnet18" if input_size >= 64 else "simple_cnn",
            "text": "text_cnn",
            "tabular": "mlp",
            "audio": "audio_cnn",
            "timeseries": "timeseries_cnn",
            "graph": "graph_mlp",
        }
        architecture = arch_map.get(task, "simple_cnn")
    
    return {
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_dim": input_dim,
        "input_size": input_size,
        "default_arch": architecture,
        "task": task,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_custom_dataset(
    path: str,
    dataset_type: str = "auto",
    **kwargs
) -> Tuple[Dataset, dict]:
    """
    Quick loader for custom datasets.
    
    Args:
        path: Path to data (folder, CSV, etc.)
        dataset_type: Type of dataset ('image_folder', 'csv', 'tensors', 'auto')
        **kwargs: Additional arguments for the dataset
    
    Returns:
        (dataset, config) tuple
    """
    path = Path(path)
    
    # Auto-detect type
    if dataset_type == "auto":
        if path.is_dir():
            dataset_type = "image_folder"
        elif path.suffix.lower() == ".csv":
            dataset_type = "csv"
        elif path.suffix.lower() in (".pt", ".pth"):
            dataset_type = "tensors"
        else:
            raise ValueError(f"Cannot auto-detect dataset type for: {path}")
    
    if dataset_type == "image_folder":
        input_size = kwargs.get("input_size", 224)
        dataset = CustomImageFolder(str(path), input_size=input_size)
        config = create_custom_config(
            num_classes=dataset.num_classes,
            task="image",
            input_size=input_size,
        )
    
    elif dataset_type == "csv":
        target_col = kwargs.get("target_column", "label")
        text_col = kwargs.get("text_column")
        
        if text_col:
            dataset = CustomTextDataset.from_csv(str(path), text_col, target_col)
            config = create_custom_config(num_classes=dataset.num_classes, task="text")
        else:
            dataset = CustomDataset.from_csv(str(path), target_col)
            config = create_custom_config(
                num_classes=dataset.num_classes,
                task="tabular",
                input_dim=dataset.X.shape[1],
            )
    
    elif dataset_type == "tensors":
        data = torch.load(path)
        if isinstance(data, dict):
            X, y = data["X"], data["y"]
        else:
            X, y = data
        dataset = CustomDataset(X, y)
        config = create_custom_config(
            num_classes=dataset.num_classes,
            task="tabular" if X.dim() == 2 else "image",
            input_dim=X.shape[1] if X.dim() == 2 else None,
            in_channels=X.shape[1] if X.dim() == 4 else 3,
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return dataset, config


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Custom Dataset Examples")
    print("=" * 50)
    
    # Example 1: Random tensor data
    print("\n1. Creating dataset from random tensors...")
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))
    dataset = CustomDataset(X, y, num_classes=10)
    print(f"   Samples: {len(dataset)}, Classes: {dataset.num_classes}")
    
    # Example 2: Tabular data
    print("\n2. Creating tabular dataset...")
    X_tab = np.random.randn(100, 13)
    y_tab = np.random.randint(0, 3, 100)
    dataset = CustomDataset.from_numpy(X_tab, y_tab)
    print(f"   Samples: {len(dataset)}, Classes: {dataset.num_classes}")
    
    # Example 3: Text data
    print("\n3. Creating text dataset...")
    texts = ["hello world", "this is a test", "machine learning"]
    labels = [0, 1, 0]
    dataset = CustomTextDataset(texts, labels)
    print(f"   Samples: {len(dataset)}, Classes: {dataset.num_classes}")
    
    print("\n✓ All examples completed!")