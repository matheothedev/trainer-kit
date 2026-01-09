"""
Dynamic model loader for any architecture
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors


class DynamicHead(nn.Module):
    """Dynamic head that builds from config"""
    
    SUPPORTED_LAYERS = {
        "Linear": nn.Linear,
        "Conv1d": nn.Conv1d,
        "Conv2d": nn.Conv2d,
        "BatchNorm1d": nn.BatchNorm1d,
        "BatchNorm2d": nn.BatchNorm2d,
        "LayerNorm": nn.LayerNorm,
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "SiLU": nn.SiLU,
        "Tanh": nn.Tanh,
        "Sigmoid": nn.Sigmoid,
        "Softmax": nn.Softmax,
        "Dropout": nn.Dropout,
        "Dropout2d": nn.Dropout2d,
        "Flatten": nn.Flatten,
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()
        self._build_from_config(config)
    
    def _build_from_config(self, config: Dict[str, Any]):
        layers_config = config.get("layers", [])
        
        for layer_cfg in layers_config:
            layer_type = layer_cfg.get("type")
            params = layer_cfg.get("params", {})
            
            if layer_type not in self.SUPPORTED_LAYERS:
                raise ValueError(f"Unsupported layer: {layer_type}")
            
            layer_cls = self.SUPPORTED_LAYERS[layer_type]
            
            if layer_type in ["ReLU", "GELU", "SiLU", "Tanh", "Sigmoid", "Flatten"]:
                layer = layer_cls()
            else:
                layer = layer_cls(**params)
            
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ModelPackage:
    """Loaded model package"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        head: nn.Module,
        embeddings: Optional[torch.Tensor] = None,
    ):
        self.config = config
        self.head = head
        self.embeddings = embeddings
        self.device = "cpu"
    
    def to(self, device: str) -> "ModelPackage":
        self.device = device
        self.head = self.head.to(device)
        if self.embeddings is not None:
            self.embeddings = self.embeddings.to(device)
        return self
    
    def eval(self) -> "ModelPackage":
        self.head.eval()
        return self
    
    def train_mode(self) -> "ModelPackage":
        self.head.train()
        return self
    
    def save_head(self, path: Path):
        """Save head weights to safetensors"""
        save_safetensors(self.head.state_dict(), str(path))
    
    def save_embeddings(self, embeddings: torch.Tensor, path: Path):
        """Save embeddings to safetensors"""
        save_safetensors({"embeddings": embeddings}, str(path))


class ModelLoader:
    """Loads model packages"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir
    
    def load_from_directory(self, path: Path) -> ModelPackage:
        """Load model from directory"""
        config_path = path / "config.json"
        head_path = path / "head.safetensors"
        embeddings_path = path / "embeddings.safetensors"
        
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        head_config = config.get("head", config)
        head = DynamicHead(head_config)
        
        if head_path.exists():
            state_dict = load_safetensors(str(head_path))
            head.load_state_dict(state_dict, strict=False)
        
        embeddings = None
        if embeddings_path.exists():
            embeddings_dict = load_safetensors(str(embeddings_path))
            embeddings = embeddings_dict.get("embeddings")
        
        return ModelPackage(config, head, embeddings)
    
    def create_head_from_config(self, config: Dict[str, Any]) -> DynamicHead:
        """Create a new head from config"""
        head_config = config.get("head", config)
        return DynamicHead(head_config)


model_loader = ModelLoader()
