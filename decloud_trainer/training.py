"""
Training Module - Extended
==========================

Train models and compute gradients.
Supports multiple architectures and datasets.
"""

import os
import copy
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torchaudio
    import torchaudio.transforms as audio_transforms
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════

class SimpleCNN(nn.Module):
    """Simple CNN for small images (32x32) - CIFAR, MNIST"""
    
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


class MediumCNN(nn.Module):
    """Medium CNN for larger images (64x64 - 128x128)"""
    
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SimpleMLP(nn.Module):
    """MLP for tabular data"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AudioCNN(nn.Module):
    """CNN for audio spectrograms"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class TextCNN(nn.Module):
    """Simple CNN for text classification (as fallback without transformers)"""
    
    def __init__(self, vocab_size: int = 30000, embed_dim: int = 128, 
                 num_classes: int = 2, max_len: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 100, kernel_size=k) for k in [3, 4, 5]
        ])
        self.fc = nn.Linear(300, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        conv_outs = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ═══════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Model configuration"""
    architecture: str
    num_classes: int
    in_channels: int = 3
    input_size: int = 32
    pretrained: bool = False
    extra_args: Dict = None


def create_model(config: ModelConfig) -> nn.Module:
    """Create model from config"""
    arch = config.architecture.lower()
    nc = config.num_classes
    
    # Simple CNNs
    if arch == "simple_cnn":
        return SimpleCNN(num_classes=nc, in_channels=config.in_channels)
    
    elif arch == "medium_cnn":
        return MediumCNN(num_classes=nc, in_channels=config.in_channels)
    
    # ResNet family
    elif arch == "resnet18":
        model = models.resnet18(pretrained=config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, nc)
        if config.in_channels != 3:
            model.conv1 = nn.Conv2d(config.in_channels, 64, 7, stride=2, padding=3, bias=False)
        return model
    
    elif arch == "resnet34":
        model = models.resnet34(pretrained=config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, nc)
        if config.in_channels != 3:
            model.conv1 = nn.Conv2d(config.in_channels, 64, 7, stride=2, padding=3, bias=False)
        return model
    
    elif arch == "resnet50":
        model = models.resnet50(pretrained=config.pretrained)
        model.fc = nn.Linear(model.fc.in_features, nc)
        if config.in_channels != 3:
            model.conv1 = nn.Conv2d(config.in_channels, 64, 7, stride=2, padding=3, bias=False)
        return model
    
    # EfficientNet
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=config.pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, nc)
        return model
    
    elif arch == "efficientnet_b1":
        model = models.efficientnet_b1(pretrained=config.pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, nc)
        return model
    
    elif arch == "efficientnet_b2":
        model = models.efficientnet_b2(pretrained=config.pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, nc)
        return model
    
    # MobileNet
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=config.pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, nc)
        return model
    
    elif arch == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(pretrained=config.pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, nc)
        return model
    
    elif arch == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(pretrained=config.pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, nc)
        return model
    
    # VGG
    elif arch == "vgg11":
        model = models.vgg11(pretrained=config.pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, nc)
        return model
    
    elif arch == "vgg16":
        model = models.vgg16(pretrained=config.pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, nc)
        return model
    
    # DenseNet
    elif arch == "densenet121":
        model = models.densenet121(pretrained=config.pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, nc)
        return model
    
    # Vision Transformer (ViT)
    elif arch == "vit_b_16":
        model = models.vit_b_16(pretrained=config.pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, nc)
        return model
    
    elif arch == "vit_b_32":
        model = models.vit_b_32(pretrained=config.pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, nc)
        return model
    
    # Tabular
    elif arch == "mlp":
        input_dim = config.extra_args.get("input_dim", 10) if config.extra_args else 10
        return SimpleMLP(input_dim=input_dim, num_classes=nc)
    
    # Audio
    elif arch == "audio_cnn":
        return AudioCNN(num_classes=nc)
    
    # Text
    elif arch == "text_cnn":
        vocab_size = config.extra_args.get("vocab_size", 30000) if config.extra_args else 30000
        return TextCNN(vocab_size=vocab_size, num_classes=nc)
    
    # Transformers (text)
    elif arch == "distilbert" and TRANSFORMERS_AVAILABLE:
        return DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=nc
        )
    
    elif arch == "bert" and TRANSFORMERS_AVAILABLE:
        return AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=nc
        )
    
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ═══════════════════════════════════════════════════════════════
# DATASET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    # ─────────────────────────────────────────────────────────────
    # IMAGE CLASSIFICATION - Small (32x32)
    # ─────────────────────────────────────────────────────────────
    "cifar10": {
        "class": "CIFAR10",
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "default_arch": "simple_cnn",
        "recommended_archs": ["simple_cnn", "resnet18", "mobilenet_v2"],
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "cifar100": {
        "class": "CIFAR100",
        "num_classes": 100,
        "in_channels": 3,
        "input_size": 32,
        "default_arch": "resnet18",
        "recommended_archs": ["resnet18", "resnet34", "efficientnet_b0"],
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "mnist": {
        "class": "MNIST",
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 32,
        "default_arch": "simple_cnn",
        "recommended_archs": ["simple_cnn"],
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "fashionmnist": {
        "class": "FashionMNIST",
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 32,
        "default_arch": "simple_cnn",
        "recommended_archs": ["simple_cnn", "resnet18"],
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "emnist": {
        "class": "EMNIST",
        "num_classes": 47,
        "in_channels": 1,
        "input_size": 32,
        "default_arch": "simple_cnn",
        "recommended_archs": ["simple_cnn", "resnet18"],
        "split": "balanced",
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3332,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "kmnist": {
        "class": "KMNIST",
        "num_classes": 10,
        "in_channels": 1,
        "input_size": 32,
        "default_arch": "simple_cnn",
        "recommended_archs": ["simple_cnn"],
        "transform": transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.1918,), (0.3483,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "svhn": {
        "class": "SVHN",
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 32,
        "default_arch": "simple_cnn",
        "recommended_archs": ["simple_cnn", "resnet18"],
        "split": "train",
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    
    # ─────────────────────────────────────────────────────────────
    # IMAGE CLASSIFICATION - Medium (64x64 - 224x224)
    # ─────────────────────────────────────────────────────────────
    "food101": {
        "class": "Food101",
        "num_classes": 101,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "resnet50",
        "recommended_archs": ["resnet50", "efficientnet_b2", "vit_b_16"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "flowers102": {
        "class": "Flowers102",
        "num_classes": 102,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "resnet34",
        "recommended_archs": ["resnet34", "resnet50", "efficientnet_b1"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "stanforddogs": {
        "class": "StanfordDogs",
        "num_classes": 120,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "resnet50",
        "recommended_archs": ["resnet50", "efficientnet_b2"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "oxfordpets": {
        "class": "OxfordIIITPet",
        "num_classes": 37,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "resnet34",
        "recommended_archs": ["resnet34", "mobilenet_v3_large"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "eurosat": {
        "class": "EuroSAT",
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 64,
        "default_arch": "medium_cnn",
        "recommended_archs": ["medium_cnn", "resnet18"],
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3444, 0.3803, 0.4078), (0.2027, 0.1366, 0.1148))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "caltech101": {
        "class": "Caltech101",
        "num_classes": 101,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "resnet34",
        "recommended_archs": ["resnet34", "resnet50"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "caltech256": {
        "class": "Caltech256",
        "num_classes": 257,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "resnet50",
        "recommended_archs": ["resnet50", "efficientnet_b2"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "stl10": {
        "class": "STL10",
        "num_classes": 10,
        "in_channels": 3,
        "input_size": 96,
        "default_arch": "medium_cnn",
        "recommended_archs": ["medium_cnn", "resnet18"],
        "split": "train",
        "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    
    # ─────────────────────────────────────────────────────────────
    # MEDICAL IMAGES
    # ─────────────────────────────────────────────────────────────
    "chestxray": {
        "class": "ChestXray",
        "num_classes": 2,
        "in_channels": 1,
        "input_size": 224,
        "default_arch": "resnet18",
        "recommended_archs": ["resnet18", "densenet121"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.229,))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "skincancer": {
        "class": "SkinCancer",
        "num_classes": 7,
        "in_channels": 3,
        "input_size": 224,
        "default_arch": "efficientnet_b1",
        "recommended_archs": ["efficientnet_b1", "resnet50"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "braintumor": {
        "class": "BrainTumor",
        "num_classes": 4,
        "in_channels": 1,
        "input_size": 224,
        "default_arch": "resnet34",
        "recommended_archs": ["resnet34", "efficientnet_b0"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.229,))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "malaria": {
        "class": "Malaria",
        "num_classes": 2,
        "in_channels": 3,
        "input_size": 128,
        "default_arch": "medium_cnn",
        "recommended_archs": ["medium_cnn", "resnet18"],
        "transform": transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "bloodcells": {
        "class": "BloodCells",
        "num_classes": 4,
        "in_channels": 3,
        "input_size": 128,
        "default_arch": "medium_cnn",
        "recommended_archs": ["medium_cnn", "resnet18"],
        "transform": transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    "covidxray": {
        "class": "CovidXray",
        "num_classes": 3,
        "in_channels": 1,
        "input_size": 224,
        "default_arch": "resnet18",
        "recommended_archs": ["resnet18", "densenet121"],
        "transform": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.229,))
        ]) if TORCHVISION_AVAILABLE else None,
    },
    
    # ─────────────────────────────────────────────────────────────
    # TABULAR DATA
    # ─────────────────────────────────────────────────────────────
    "iris": {
        "class": "Iris",
        "num_classes": 3,
        "input_dim": 4,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "wine": {
        "class": "Wine",
        "num_classes": 3,
        "input_dim": 13,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "diabetes": {
        "class": "Diabetes",
        "num_classes": 2,
        "input_dim": 8,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "breastcancer": {
        "class": "BreastCancer",
        "num_classes": 2,
        "input_dim": 30,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "californiahousing": {
        "class": "CaliforniaHousing",
        "num_classes": 1,
        "input_dim": 8,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular_regression",
    },
    "adultincome": {
        "class": "AdultIncome",
        "num_classes": 2,
        "input_dim": 14,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "bankmarketing": {
        "class": "BankMarketing",
        "num_classes": 2,
        "input_dim": 16,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "creditdefault": {
        "class": "CreditDefault",
        "num_classes": 2,
        "input_dim": 23,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "titanic": {
        "class": "Titanic",
        "num_classes": 2,
        "input_dim": 7,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    "heartdisease": {
        "class": "HeartDisease",
        "num_classes": 2,
        "input_dim": 13,
        "default_arch": "mlp",
        "recommended_archs": ["mlp"],
        "task": "tabular",
    },
    
    # ─────────────────────────────────────────────────────────────
    # TEXT CLASSIFICATION
    # ─────────────────────────────────────────────────────────────
    "imdb": {
        "class": "IMDB",
        "num_classes": 2,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "text_cnn"],
        "task": "text",
        "max_length": 256,
    },
    "sst2": {
        "class": "SST2",
        "num_classes": 2,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "text_cnn"],
        "task": "text",
        "max_length": 128,
    },
    "agnews": {
        "class": "AGNews",
        "num_classes": 4,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "text_cnn"],
        "task": "text",
        "max_length": 256,
    },
    "yelpreviewfull": {
        "class": "YelpReviewFull",
        "num_classes": 5,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "bert"],
        "task": "text",
        "max_length": 256,
    },
    "amazonpolarity": {
        "class": "AmazonPolarity",
        "num_classes": 2,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert"],
        "task": "text",
        "max_length": 256,
    },
    "dbpedia": {
        "class": "DBpedia",
        "num_classes": 14,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "bert"],
        "task": "text",
        "max_length": 256,
    },
    "rottentomatoes": {
        "class": "RottenTomatoes",
        "num_classes": 2,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "text_cnn"],
        "task": "text",
        "max_length": 128,
    },
    "smsspam": {
        "class": "SMSSpam",
        "num_classes": 2,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "text_cnn"],
        "task": "text",
        "max_length": 128,
    },
    "hatespeech": {
        "class": "HateSpeech",
        "num_classes": 3,
        "default_arch": "distilbert",
        "recommended_archs": ["distilbert", "bert"],
        "task": "text",
        "max_length": 128,
    },
    
    # ─────────────────────────────────────────────────────────────
    # AUDIO
    # ─────────────────────────────────────────────────────────────
    "speechcommands": {
        "class": "SpeechCommands",
        "num_classes": 35,
        "default_arch": "audio_cnn",
        "recommended_archs": ["audio_cnn"],
        "task": "audio",
        "sample_rate": 16000,
    },
    "gtzan": {
        "class": "GTZAN",
        "num_classes": 10,
        "default_arch": "audio_cnn",
        "recommended_archs": ["audio_cnn", "resnet18"],
        "task": "audio",
        "sample_rate": 22050,
    },
    "esc50": {
        "class": "ESC50",
        "num_classes": 50,
        "default_arch": "audio_cnn",
        "recommended_archs": ["audio_cnn"],
        "task": "audio",
        "sample_rate": 44100,
    },
    "urbansound8k": {
        "class": "UrbanSound8K",
        "num_classes": 10,
        "default_arch": "audio_cnn",
        "recommended_archs": ["audio_cnn"],
        "task": "audio",
        "sample_rate": 22050,
    },
}


def get_dataset_config(name: str) -> Dict:
    """Get dataset configuration by name"""
    name_lower = name.lower().replace("_", "").replace("-", "")
    
    if name_lower not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(DATASET_CONFIG.keys())}")
    
    return DATASET_CONFIG[name_lower]


def get_dataset(name: str, data_dir: str = "./data", train: bool = True):
    """
    Get dataset by name.
    
    Args:
        name: Dataset name
        data_dir: Directory to store/load data
        train: Use training set (True) or test set (False)
    
    Returns:
        Tuple of (dataset, config)
    """
    if not TORCHVISION_AVAILABLE:
        raise ImportError("torchvision is required for dataset loading")
    
    config = get_dataset_config(name)
    task = config.get("task", "image")
    
    # Handle different task types
    if task == "tabular":
        return _load_tabular_dataset(name, config, train)
    elif task == "text":
        return _load_text_dataset(name, config, train)
    elif task == "audio":
        return _load_audio_dataset(name, config, data_dir, train)
    else:
        return _load_image_dataset(name, config, data_dir, train)


def _load_image_dataset(name: str, config: Dict, data_dir: str, train: bool):
    """Load image dataset"""
    dataset_class_name = config["class"]
    
    # Special handling for some datasets
    if dataset_class_name == "EMNIST":
        dataset = torchvision.datasets.EMNIST(
            root=data_dir,
            split=config.get("split", "balanced"),
            train=train,
            download=True,
            transform=config["transform"]
        )
    elif dataset_class_name == "SVHN":
        split = "train" if train else "test"
        dataset = torchvision.datasets.SVHN(
            root=data_dir,
            split=split,
            download=True,
            transform=config["transform"]
        )
    elif dataset_class_name == "STL10":
        split = "train" if train else "test"
        dataset = torchvision.datasets.STL10(
            root=data_dir,
            split=split,
            download=True,
            transform=config["transform"]
        )
    elif hasattr(torchvision.datasets, dataset_class_name):
        dataset_class = getattr(torchvision.datasets, dataset_class_name)
        try:
            dataset = dataset_class(
                root=data_dir,
                train=train,
                download=True,
                transform=config["transform"]
            )
        except TypeError:
            # Some datasets don't have train parameter
            split = "train" if train else "test"
            dataset = dataset_class(
                root=data_dir,
                split=split,
                download=True,
                transform=config["transform"]
            )
    else:
        raise ValueError(f"Dataset class not found: {dataset_class_name}")
    
    return dataset, config


def _load_tabular_dataset(name: str, config: Dict, train: bool):
    """Load tabular dataset from sklearn"""
    from sklearn import datasets as sklearn_datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    name_lower = name.lower()
    
    if name_lower == "iris":
        data = sklearn_datasets.load_iris()
    elif name_lower == "wine":
        data = sklearn_datasets.load_wine()
    elif name_lower == "breastcancer":
        data = sklearn_datasets.load_breast_cancer()
    elif name_lower == "diabetes":
        data = sklearn_datasets.load_diabetes()
        data.target = (data.target > data.target.mean()).astype(int)
    else:
        raise ValueError(f"Tabular dataset not implemented: {name}")
    
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if train:
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
    else:
        X_tensor = torch.FloatTensor(X_test)
        y_tensor = torch.LongTensor(y_test)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    return dataset, config


def _load_text_dataset(name: str, config: Dict, train: bool):
    """Load text dataset (requires datasets library)"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("'datasets' library required for text datasets")
    
    name_lower = name.lower()
    
    dataset_map = {
        "imdb": ("imdb", None),
        "sst2": ("glue", "sst2"),
        "agnews": ("ag_news", None),
        "yelpreviewfull": ("yelp_review_full", None),
        "amazonpolarity": ("amazon_polarity", None),
        "dbpedia": ("dbpedia_14", None),
        "rottentomatoes": ("rotten_tomatoes", None),
    }
    
    if name_lower not in dataset_map:
        raise ValueError(f"Text dataset not implemented: {name}")
    
    hf_name, subset = dataset_map[name_lower]
    split = "train" if train else "test"
    
    if subset:
        ds = load_dataset(hf_name, subset, split=split, trust_remote_code=True)
    else:
        ds = load_dataset(hf_name, split=split, trust_remote_code=True)
    
    return ds, config


def _load_audio_dataset(name: str, config: Dict, data_dir: str, train: bool):
    """Load audio dataset"""
    if not TORCHAUDIO_AVAILABLE:
        raise ImportError("torchaudio required for audio datasets")
    
    name_lower = name.lower()
    
    if name_lower == "speechcommands":
        dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_dir,
            download=True,
            subset=None
        )
    else:
        raise ValueError(f"Audio dataset not implemented: {name}")
    
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
        num_workers=num_workers,
        pin_memory=True
    )


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def detect_architecture(state_dict: dict) -> tuple:
    """
    Detect model architecture from state_dict keys.
    
    Returns:
        (architecture_name, num_classes, in_channels)
    """
    keys = set(state_dict.keys())
    key_str = " ".join(keys)
    
    # Check for ResNet patterns
    if "layer1.0.conv1.weight" in keys and "layer4.1.conv2.weight" in keys:
        # Determine ResNet variant by layer4 structure
        if "layer4.2.conv1.weight" in keys:
            arch = "resnet50"  # Has 3 blocks in layer4
        elif "layer3.5.conv1.weight" in keys:
            arch = "resnet34"  # Has 6 blocks in layer3
        else:
            arch = "resnet18"  # Basic
        
        # Get num_classes from fc layer
        num_classes = state_dict["fc.weight"].shape[0]
        # Get in_channels from first conv
        in_channels = state_dict["conv1.weight"].shape[1]
        print(f"   🔍 Detected architecture: {arch} (classes={num_classes}, channels={in_channels})")
        return arch, num_classes, in_channels
    
    # Check for EfficientNet
    if "features.0.0.weight" in keys and "classifier.1.weight" in keys:
        num_classes = state_dict["classifier.1.weight"].shape[0]
        in_channels = state_dict["features.0.0.weight"].shape[1]
        # Detect variant by features depth
        if "features.8.0.block.0.0.weight" in keys:
            arch = "efficientnet_b2"
        elif "features.7.0.block.0.0.weight" in keys:
            arch = "efficientnet_b1"
        else:
            arch = "efficientnet_b0"
        print(f"   🔍 Detected architecture: {arch} (classes={num_classes})")
        return arch, num_classes, in_channels
    
    # Check for MobileNetV2
    if "features.0.0.weight" in keys and "classifier.1.weight" in keys and "features.18" in key_str:
        num_classes = state_dict["classifier.1.weight"].shape[0]
        in_channels = state_dict["features.0.0.weight"].shape[1]
        print(f"   🔍 Detected architecture: mobilenet_v2 (classes={num_classes})")
        return "mobilenet_v2", num_classes, in_channels
    
    # Check for MobileNetV3
    if "features.0.0.weight" in keys and "classifier.3.weight" in keys:
        num_classes = state_dict["classifier.3.weight"].shape[0]
        in_channels = state_dict["features.0.0.weight"].shape[1]
        if state_dict["features.0.0.weight"].shape[0] == 16:
            arch = "mobilenet_v3_small"
        else:
            arch = "mobilenet_v3_large"
        print(f"   🔍 Detected architecture: {arch} (classes={num_classes})")
        return arch, num_classes, in_channels
    
    # Check for VGG
    if "features.0.weight" in keys and "classifier.6.weight" in keys:
        num_classes = state_dict["classifier.6.weight"].shape[0]
        in_channels = state_dict["features.0.weight"].shape[1]
        # Count conv layers to determine variant
        conv_count = sum(1 for k in keys if k.startswith("features.") and k.endswith(".weight") and "bias" not in k)
        if conv_count >= 16:
            arch = "vgg16"
        else:
            arch = "vgg11"
        print(f"   🔍 Detected architecture: {arch} (classes={num_classes})")
        return arch, num_classes, in_channels
    
    # Check for DenseNet
    if "features.conv0.weight" in keys and "classifier.weight" in keys:
        num_classes = state_dict["classifier.weight"].shape[0]
        in_channels = state_dict["features.conv0.weight"].shape[1]
        print(f"   🔍 Detected architecture: densenet121 (classes={num_classes})")
        return "densenet121", num_classes, in_channels
    
    # Check for ViT
    if "encoder.layers.encoder_layer_0.self_attention.in_proj_weight" in keys:
        num_classes = state_dict["heads.head.weight"].shape[0]
        if "encoder.layers.encoder_layer_11" in key_str:
            if state_dict["conv_proj.weight"].shape[2] == 16:
                arch = "vit_b_16"
            else:
                arch = "vit_b_32"
        else:
            arch = "vit_b_16"
        print(f"   🔍 Detected architecture: {arch} (classes={num_classes})")
        return arch, num_classes, 3
    
    # Check for MediumCNN
    if "features.0.weight" in keys and "classifier.1.weight" in keys and "features.4.weight" in keys:
        num_classes = state_dict["classifier.4.weight"].shape[0]
        in_channels = state_dict["features.0.weight"].shape[1]
        print(f"   🔍 Detected architecture: medium_cnn (classes={num_classes})")
        return "medium_cnn", num_classes, in_channels
    
    # Check for SimpleCNN (our default)
    if "conv1.weight" in keys and "conv2.weight" in keys and "fc1.weight" in keys and "fc2.weight" in keys:
        num_classes = state_dict["fc2.weight"].shape[0]
        in_channels = state_dict["conv1.weight"].shape[1]
        print(f"   🔍 Detected architecture: simple_cnn (classes={num_classes})")
        return "simple_cnn", num_classes, in_channels
    
    # Check for SimpleMLP
    if "model.0.weight" in keys and "model.0.bias" in keys:
        # Find last linear layer
        last_layer = max([k for k in keys if k.startswith("model.") and k.endswith(".weight")])
        num_classes = state_dict[last_layer].shape[0]
        input_dim = state_dict["model.0.weight"].shape[1]
        print(f"   🔍 Detected architecture: mlp (classes={num_classes}, input_dim={input_dim})")
        return "mlp", num_classes, input_dim
    
    # Check for AudioCNN
    if "conv.0.weight" in keys and "fc.0.weight" in keys:
        num_classes = state_dict["fc.3.weight"].shape[0]
        print(f"   🔍 Detected architecture: audio_cnn (classes={num_classes})")
        return "audio_cnn", num_classes, 1
    
    # Check for TextCNN
    if "embedding.weight" in keys and "convs.0.weight" in keys:
        num_classes = state_dict["fc.weight"].shape[0]
        vocab_size = state_dict["embedding.weight"].shape[0]
        print(f"   🔍 Detected architecture: text_cnn (classes={num_classes}, vocab={vocab_size})")
        return "text_cnn", num_classes, vocab_size
    
    # Check for DistilBERT/BERT (transformers)
    if "distilbert.embeddings.word_embeddings.weight" in keys:
        num_classes = state_dict["classifier.weight"].shape[0]
        print(f"   🔍 Detected architecture: distilbert (classes={num_classes})")
        return "distilbert", num_classes, 0
    
    if "bert.embeddings.word_embeddings.weight" in keys:
        num_classes = state_dict["classifier.weight"].shape[0]
        print(f"   🔍 Detected architecture: bert (classes={num_classes})")
        return "bert", num_classes, 0
    
    # Unknown - return None
    print(f"   ⚠️ Could not detect architecture from state_dict keys")
    print(f"   Keys sample: {list(keys)[:10]}")
    return None, None, None


class Trainer:
    """
    Local trainer for federated learning.
    Supports multiple model architectures and datasets.
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
        print(f"   Device: {self.device}")
    
    def load_model(self, model_path: str, dataset_name: str = None, 
                   architecture: str = None) -> nn.Module:
        """
        Load model from file with automatic architecture detection.
        
        If architecture is not specified, it will be detected from state_dict.
        """
        # Load state_dict first to detect architecture
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Try to detect architecture if not specified
        if architecture is None:
            detected_arch, detected_classes, detected_channels = detect_architecture(state_dict)
            
            if detected_arch:
                architecture = detected_arch
                num_classes = detected_classes
                in_channels = detected_channels if detected_channels else 3
            else:
                # Fallback to dataset config
                config = get_dataset_config(dataset_name) if dataset_name else {}
                architecture = config.get("default_arch", "simple_cnn")
                num_classes = config.get("num_classes", 10)
                in_channels = config.get("in_channels", 3)
        else:
            # Use provided architecture but try to get num_classes from state_dict
            config = get_dataset_config(dataset_name) if dataset_name else {}
            _, detected_classes, detected_channels = detect_architecture(state_dict)
            num_classes = detected_classes or config.get("num_classes", 10)
            in_channels = detected_channels if detected_channels else config.get("in_channels", 3)
        
        # Handle special case for MLP input_dim
        extra_args = None
        if architecture == "mlp":
            input_dim = state_dict["model.0.weight"].shape[1]
            extra_args = {"input_dim": input_dim}
        elif architecture == "text_cnn":
            vocab_size = state_dict["embedding.weight"].shape[0]
            extra_args = {"vocab_size": vocab_size}
        
        # Get input_size from dataset config if available
        config = get_dataset_config(dataset_name) if dataset_name else {}
        input_size = config.get("input_size", 32)
        
        model_config = ModelConfig(
            architecture=architecture,
            num_classes=num_classes,
            in_channels=in_channels,
            input_size=input_size,
            extra_args=extra_args
        )
        
        model = create_model(model_config)
        model.load_state_dict(state_dict)
        model.to(self.device)
        
        return model
    
    def create_model(self, dataset_name: str, architecture: str = None, 
                     pretrained: bool = False) -> nn.Module:
        """Create a new model for dataset."""
        config = get_dataset_config(dataset_name)
        arch = architecture or config.get("default_arch", "simple_cnn")
        
        model_config = ModelConfig(
            architecture=arch,
            num_classes=config.get("num_classes", 10),
            in_channels=config.get("in_channels", 3),
            input_size=config.get("input_size", 32),
            pretrained=pretrained,
            extra_args={"input_dim": config.get("input_dim", 10)} if config.get("task") == "tabular" else None
        )
        
        model = create_model(model_config)
        model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Model: {arch}")
        print(f"   Parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        return model
    
    def train(
        self,
        model: nn.Module,
        dataset_name: str,
        epochs: int = 1,
        data_dir: str = "./data",
        max_batches: Optional[int] = None,
        augment: bool = True,
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Train model on local data."""
        original_state = copy.deepcopy(model.state_dict())
        
        config = get_dataset_config(dataset_name)
        task = config.get("task", "image")
        
        if task == "text":
            return self._train_text(model, dataset_name, config, epochs, max_batches, original_state)
        elif task == "tabular":
            dataset, _ = get_dataset(dataset_name, data_dir, train=True)
            dataloader = create_dataloader(dataset, self.batch_size)
        else:
            dataset, _ = get_dataset(dataset_name, data_dir, train=True)
            dataloader = create_dataloader(dataset, self.batch_size)
        
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
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
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = output.argmax(dim=1)
                epoch_correct += pred.eq(target).sum().item()
                epoch_samples += len(target)
                
                if batch_idx % 50 == 0:
                    print(f"\r   Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}", end="", flush=True)
            
            scheduler.step()
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
            
            accuracy = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0
            avg_loss = epoch_loss / (batch_idx + 1) if batch_idx >= 0 else 0
            print(f"\r   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        gradients = {}
        trained_state = model.state_dict()
        
        for key in original_state:
            gradients[key] = trained_state[key] - original_state[key]
        
        final_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        print(f"   ✓ Training complete! Final accuracy: {final_accuracy:.2f}%")
        
        return model, gradients
    
    def _train_text(self, model, dataset_name, config, epochs, max_batches, original_state):
        """Train text model with transformers"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for text training")
        
        from transformers import DistilBertTokenizer, AdamW
        
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        dataset, _ = get_dataset(dataset_name, train=True)
        
        max_length = config.get("max_length", 256)
        
        model.train()
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        batch_size = min(self.batch_size, 16)
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx in range(0, len(dataset), batch_size):
                if max_batches and batch_idx // batch_size >= max_batches:
                    break
                
                batch = dataset[batch_idx:batch_idx + batch_size]
                
                if "text" in batch:
                    texts = batch["text"]
                elif "sentence" in batch:
                    texts = batch["sentence"]
                else:
                    texts = batch[list(batch.keys())[0]]
                
                labels = batch.get("label", batch.get("labels", [0] * len(texts)))
                
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                labels = torch.tensor(labels).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                pred = outputs.logits.argmax(dim=1)
                epoch_correct += pred.eq(labels).sum().item()
                epoch_samples += len(labels)
                
                if (batch_idx // batch_size) % 10 == 0:
                    print(f"\r   Batch {batch_idx // batch_size}, Loss: {loss.item():.4f}", end="", flush=True)
            
            accuracy = 100.0 * epoch_correct / epoch_samples if epoch_samples > 0 else 0
            avg_loss = epoch_loss / max(1, epoch_samples // batch_size)
            print(f"\r   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        gradients = {}
        trained_state = model.state_dict()
        
        for key in original_state:
            gradients[key] = trained_state[key] - original_state[key]
        
        final_accuracy = 100.0 * total_correct / total_samples if total_samples > 0 else 0
        print(f"   ✓ Training complete! Final accuracy: {final_accuracy:.2f}%")
        
        return model, gradients
    
    def save_gradients(self, gradients: Dict[str, torch.Tensor], output_path: str):
        """Save gradients to file"""
        cpu_gradients = {k: v.cpu() for k, v in gradients.items()}
        torch.save(cpu_gradients, output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✓ Gradients saved: {size_mb:.2f} MB")
    
    def save_model(self, model: nn.Module, output_path: str):
        """Save model state dict"""
        torch.save(model.state_dict(), output_path)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"   ✓ Model saved: {size_mb:.2f} MB")
    
    def evaluate(
        self,
        model: nn.Module,
        dataset_name: str,
        data_dir: str = "./data"
    ) -> float:
        """Evaluate model on test set."""
        config = get_dataset_config(dataset_name)
        task = config.get("task", "image")
        
        if task == "text":
            return self._evaluate_text(model, dataset_name, config)
        
        dataset, _ = get_dataset(dataset_name, data_dir, train=False)
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
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy
    
    def _evaluate_text(self, model, dataset_name, config) -> float:
        """Evaluate text model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required")
        
        from transformers import DistilBertTokenizer
        
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        dataset, _ = get_dataset(dataset_name, train=False)
        
        model.eval()
        correct = 0
        total = 0
        max_length = config.get("max_length", 256)
        batch_size = 16
        
        with torch.no_grad():
            for batch_idx in range(0, min(len(dataset), 1000), batch_size):
                batch = dataset[batch_idx:batch_idx + batch_size]
                
                if "text" in batch:
                    texts = batch["text"]
                elif "sentence" in batch:
                    texts = batch["sentence"]
                else:
                    texts = batch[list(batch.keys())[0]]
                
                labels = batch.get("label", batch.get("labels", [0] * len(texts)))
                
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                labels = torch.tensor(labels).to(self.device)
                
                outputs = model(**inputs)
                pred = outputs.logits.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += len(labels)
        
        accuracy = 100.0 * correct / total if total > 0 else 0
        return accuracy


# ═══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def list_supported_datasets() -> Dict[str, Dict]:
    """List all supported datasets with their configurations"""
    result = {}
    for name, config in DATASET_CONFIG.items():
        result[name] = {
            "num_classes": config.get("num_classes", "N/A"),
            "task": config.get("task", "image"),
            "default_arch": config.get("default_arch", "simple_cnn"),
            "recommended_archs": config.get("recommended_archs", []),
            "input_size": config.get("input_size", "N/A"),
        }
    return result


def list_supported_architectures() -> List[str]:
    """List all supported model architectures"""
    return [
        "simple_cnn", "medium_cnn",
        "resnet18", "resnet34", "resnet50",
        "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
        "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
        "vgg11", "vgg16",
        "densenet121",
        "vit_b_16", "vit_b_32",
        "mlp",
        "audio_cnn",
        "text_cnn", "distilbert", "bert",
    ]


def get_model_size(architecture: str, num_classes: int = 10) -> int:
    """Get approximate model size in parameters"""
    sizes = {
        "simple_cnn": 100_000,
        "medium_cnn": 2_000_000,
        "resnet18": 11_700_000,
        "resnet34": 21_800_000,
        "resnet50": 25_600_000,
        "efficientnet_b0": 5_300_000,
        "efficientnet_b1": 7_800_000,
        "efficientnet_b2": 9_200_000,
        "mobilenet_v2": 3_500_000,
        "mobilenet_v3_small": 2_500_000,
        "mobilenet_v3_large": 5_500_000,
        "vgg11": 132_000_000,
        "vgg16": 138_000_000,
        "densenet121": 8_000_000,
        "vit_b_16": 86_000_000,
        "vit_b_32": 88_000_000,
        "mlp": 50_000,
        "audio_cnn": 500_000,
        "text_cnn": 4_000_000,
        "distilbert": 66_000_000,
        "bert": 110_000_000,
    }
    return sizes.get(architecture.lower(), 0)