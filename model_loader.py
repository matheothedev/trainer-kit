"""
Dynamic model loader for any architecture
Supports: Classification (head + embeddings) and LLM (full model)
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
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
    """Loaded model package (Classification)"""

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
        self.is_llm = False
    
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


class LLMModelPackage:
    """Loaded LLM model package with full model and tokenizer"""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        tokenizer: Any,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cpu"
        self.is_llm = True
        self.dataset_name = config.get("dataset", {}).get("name", "unknown")
        self.num_classes = config.get("dataset", {}).get("num_classes", 2)

    def to(self, device: str) -> "LLMModelPackage":
        self.device = device
        self.model = self.model.to(device)
        return self

    def eval(self) -> "LLMModelPackage":
        self.model.eval()
        return self

    def train_mode(self) -> "LLMModelPackage":
        self.model.train()
        return self

    def save_model(self, path: Path):
        """Save full model weights"""
        state_dict = {}
        for key, tensor in self.model.state_dict().items():
            if tensor.dtype in [torch.float32, torch.float64]:
                state_dict[key] = tensor.half().clone()
            else:
                state_dict[key] = tensor.clone()
        save_safetensors(state_dict, str(path))

    def save_tokenizer(self, path: Path):
        """Save tokenizer"""
        self.tokenizer.save_pretrained(path)


class ModelLoader:
    """Loads model packages - supports both Classification and LLM"""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir

    def load_from_directory(self, path: Path) -> ModelPackage | LLMModelPackage:
        """Load model from directory. Auto-detects format."""
        config_path = path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {path}")

        with open(config_path, "r") as f:
            config = json.load(f)

        # Check package type
        pkg_type = config.get("type", "classification")

        if pkg_type == "llm_full":
            return self._load_llm_package(path, config)
        else:
            return self._load_classification_package(path, config)

    def _load_classification_package(self, path: Path, config: Dict[str, Any]) -> ModelPackage:
        """Load classification model (embeddings + head)"""
        head_path = path / "head.safetensors"
        embeddings_path = path / "embeddings.safetensors"

        head_config = config.get("head", config)
        head = DynamicHead(head_config)

        if head_path.exists():
            state_dict = load_safetensors(str(head_path))
            head.load_state_dict(state_dict, strict=False)

        embeddings = None
        if embeddings_path.exists():
            embeddings_dict = load_safetensors(str(embeddings_path))
            embeddings = embeddings_dict.get("embeddings")

        pkg = ModelPackage(config, head, embeddings)
        pkg.is_llm = False
        return pkg

    def _load_llm_package(self, path: Path, config: Dict[str, Any]) -> LLMModelPackage:
        """Load full LLM model"""
        model_path = path / "model.safetensors"
        tokenizer_path = path / "tokenizer"

        if not model_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {path}")

        # Get model architecture from config
        model_info = config.get("model", {})
        model_source = model_info.get("source", "gpt2")

        # Load model architecture from HuggingFace
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

        # If tokenizer not in package, download from HuggingFace
        if not tokenizer_path.exists():
            print(f"Tokenizer not found locally, downloading from HuggingFace: {model_source}")
            tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
            tokenizer.save_pretrained(tokenizer_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load config to create empty model
        hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(hf_config)

        # Load weights from safetensors
        state_dict = load_safetensors(str(model_path))

        # Convert fp16 weights back to model dtype
        for key in state_dict:
            if state_dict[key].dtype == torch.float16:
                state_dict[key] = state_dict[key].float()

        model.load_state_dict(state_dict, strict=False)

        return LLMModelPackage(config, model, tokenizer)

    def create_head_from_config(self, config: Dict[str, Any]) -> DynamicHead:
        """Create a new head from config"""
        head_config = config.get("head", config)
        return DynamicHead(head_config)


model_loader = ModelLoader()
