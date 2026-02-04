"""
Training logic for Decloud Trainer
Supports both Classification (embeddings) and LLM (full model) training
"""
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from safetensors.torch import save_file as save_safetensors
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import config, GRADIENTS_DIR
from model_loader import ModelPackage, DynamicHead

console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Dataset Loader - Supports multiple formats
# ═══════════════════════════════════════════════════════════════════════════════

class LLMDatasetLoader:
    """
    Load LLM training data from various formats
    Supports: JSON, JSONL, CSV, TXT, Parquet
    """

    SUPPORTED_FORMATS = [".json", ".jsonl", ".csv", ".txt", ".parquet", ".tsv"]

    @classmethod
    def load_from_path(
        cls,
        dataset_path: str,
        text_column: str = "text",
        label_column: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Load dataset from path (file or directory)

        Returns:
            {
                "texts": List[str],
                "labels": Optional[List[int]],
                "format": str,
                "num_samples": int
            }
        """
        path = Path(dataset_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        # If directory, find data files
        if path.is_dir():
            data_file = cls._find_data_file(path)
            if not data_file:
                raise FileNotFoundError(f"No data file found in {dataset_path}")
            path = data_file

        # Load based on format
        ext = path.suffix.lower()

        if ext == ".json":
            data = cls._load_json(path, text_column, label_column)
        elif ext == ".jsonl":
            data = cls._load_jsonl(path, text_column, label_column)
        elif ext == ".csv":
            data = cls._load_csv(path, text_column, label_column)
        elif ext == ".tsv":
            data = cls._load_csv(path, text_column, label_column, delimiter="\t")
        elif ext == ".txt":
            data = cls._load_txt(path)
        elif ext == ".parquet":
            data = cls._load_parquet(path, text_column, label_column)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Apply limit
        if limit and len(data["texts"]) > limit:
            data["texts"] = data["texts"][:limit]
            if data["labels"]:
                data["labels"] = data["labels"][:limit]

        data["num_samples"] = len(data["texts"])
        data["format"] = ext

        return data

    @classmethod
    def _find_data_file(cls, directory: Path) -> Optional[Path]:
        """Find data file in directory"""
        # Priority order for file names
        priority_names = [
            "train", "training", "data", "dataset",
            "train_data", "training_data"
        ]

        for name in priority_names:
            for ext in cls.SUPPORTED_FORMATS:
                candidate = directory / f"{name}{ext}"
                if candidate.exists():
                    return candidate

        # Fall back to any supported file
        for ext in cls.SUPPORTED_FORMATS:
            files = list(directory.glob(f"*{ext}"))
            if files:
                return files[0]

        return None

    @classmethod
    def _load_json(cls, path: Path, text_col: str, label_col: Optional[str]) -> Dict:
        """Load JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # Try common keys
            for key in ["data", "samples", "items", "train", "examples"]:
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
            else:
                items = [data]
        else:
            raise ValueError("Invalid JSON structure")

        return cls._extract_texts_labels(items, text_col, label_col)

    @classmethod
    def _load_jsonl(cls, path: Path, text_col: str, label_col: Optional[str]) -> Dict:
        """Load JSONL (JSON Lines) file"""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        return cls._extract_texts_labels(items, text_col, label_col)

    @classmethod
    def _load_csv(
        cls, path: Path, text_col: str, label_col: Optional[str], delimiter: str = ","
    ) -> Dict:
        """Load CSV/TSV file"""
        items = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                items.append(row)

        return cls._extract_texts_labels(items, text_col, label_col)

    @classmethod
    def _load_txt(cls, path: Path) -> Dict:
        """Load plain text file (one sample per line or full text)"""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if it's line-based or full text
        lines = [l.strip() for l in content.split("\n") if l.strip()]

        if len(lines) > 1 and all(len(l) < 1000 for l in lines):
            # Line-based dataset
            texts = lines
        else:
            # Full text - split into chunks
            texts = cls._chunk_text(content, chunk_size=512)

        return {"texts": texts, "labels": None}

    @classmethod
    def _load_parquet(cls, path: Path, text_col: str, label_col: Optional[str]) -> Dict:
        """Load Parquet file"""
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            items = df.to_dict("records")
            return cls._extract_texts_labels(items, text_col, label_col)
        except ImportError:
            raise ImportError("pandas and pyarrow required for parquet files")

    @classmethod
    def _extract_texts_labels(
        cls, items: List[Dict], text_col: str, label_col: Optional[str]
    ) -> Dict:
        """Extract texts and labels from list of dicts"""
        texts = []
        labels = []

        # Auto-detect text column if not found
        text_keys = [text_col, "text", "content", "input", "sentence", "question", "prompt"]
        label_keys = [label_col, "label", "labels", "target", "class", "category"] if label_col else []

        for item in items:
            # Find text
            text = None
            for key in text_keys:
                if key and key in item:
                    text = str(item[key])
                    break

            if text is None:
                # Use first string value
                for v in item.values():
                    if isinstance(v, str) and len(v) > 10:
                        text = v
                        break

            if text:
                texts.append(text)

                # Find label
                label = None
                for key in label_keys:
                    if key and key in item:
                        label = item[key]
                        break

                labels.append(label)

        # Convert labels to int if possible
        if labels and labels[0] is not None:
            try:
                labels = [int(l) if l is not None else 0 for l in labels]
            except (ValueError, TypeError):
                # Keep as-is for text labels
                pass
        else:
            labels = None

        return {"texts": texts, "labels": labels}

    @classmethod
    def _chunk_text(cls, text: str, chunk_size: int = 512) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks


class LLMTextDataset(Dataset):
    """PyTorch Dataset for LLM text training"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),  # For causal LM
        }


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
    model_path: Optional[Path] = None,
) -> TrainingResult:
    """
    Train a model for a round
    Auto-detects model type (Classification vs LLM)

    Args:
        round_id: Round ID
        model_config: Model configuration from IPFS
        head_weights_path: Path to head.safetensors from IPFS (for classification)
        embeddings_path: Path to embeddings.safetensors from IPFS (for classification)
        dataset_path: Local path containing training data
        model_path: Full path to model directory (for LLM)

    Returns:
        TrainingResult with gradient directory
    """
    # Check model type
    model_type = model_config.get("type", "classification")

    if model_type == "llm_full":
        return train_llm_round(round_id, model_config, model_path, dataset_path)
    else:
        return train_classification_round(
            round_id, model_config, head_weights_path, embeddings_path, dataset_path
        )


def train_classification_round(
    round_id: int,
    model_config: Dict[str, Any],
    head_weights_path: Path,
    embeddings_path: Path,
    dataset_path: str,
) -> TrainingResult:
    """
    Train a CLASSIFICATION model (embeddings + head)

    EMBEDDINGS come from IPFS (embeddings_path)
    LABELS come from local dataset_path (NPY format)
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


def train_llm_round(
    round_id: int,
    model_config: Dict[str, Any],
    model_path: Path,
    dataset_path: str,
) -> TrainingResult:
    """
    Train a FULL LLM model

    Model comes from IPFS (model.safetensors + tokenizer/)
    Training data comes from local dataset_path (JSON/JSONL/CSV/TXT/Parquet)
    """
    if not config.allow_llm:
        return TrainingResult(success=False, error="LLM training disabled. Set allow_llm=true")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        from transformers import TrainingArguments, Trainer as HFTrainer
        from safetensors.torch import load_file as load_safetensors

        # 1. Load model from IPFS
        console.print(f"[dim]  Loading LLM model...[/dim]")

        model_info = model_config.get("model", {})
        model_source = model_info.get("source", "gpt2")

        # Load config and create model
        hf_config = AutoConfig.from_pretrained(model_source, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(hf_config)

        # Load weights
        weights_path = model_path / "model.safetensors"
        if weights_path.exists():
            state_dict = load_safetensors(str(weights_path))
            # Convert fp16 to fp32 if needed
            for key in state_dict:
                if state_dict[key].dtype == torch.float16:
                    state_dict[key] = state_dict[key].float()
            model.load_state_dict(state_dict, strict=False)
            console.print(f"[dim]  Loaded model weights[/dim]")

        # Load tokenizer
        tokenizer_path = model_path / "tokenizer"
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = model.to(device)

        # 2. Load training data (any format)
        console.print(f"[dim]  Loading training data from {dataset_path}...[/dim]")
        data = LLMDatasetLoader.load_from_path(dataset_path)

        console.print(f"[dim]  Loaded {data['num_samples']} samples ({data['format']})[/dim]")

        # 3. Create dataset
        train_dataset = LLMTextDataset(
            texts=data["texts"],
            tokenizer=tokenizer,
            max_length=512,
        )

        # 4. Evaluate initial perplexity
        console.print(f"[dim]  Computing initial perplexity...[/dim]")
        initial_ppl = compute_perplexity(model, tokenizer, data["texts"][:100], device)
        console.print(f"[dim]  Initial perplexity: {initial_ppl:.2f}[/dim]")

        # 5. Train
        console.print(f"[dim]  Training for {config.training_epochs} epochs...[/dim]")

        gradient_dir = GRADIENTS_DIR / f"round_{round_id}"
        gradient_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(gradient_dir / "checkpoints"),
            num_train_epochs=config.training_epochs,
            per_device_train_batch_size=config.training_batch_size,
            learning_rate=config.learning_rate,
            logging_steps=10,
            save_strategy="no",
            report_to="none",
            fp16=torch.cuda.is_available(),
        )

        hf_trainer = HFTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        hf_trainer.train()

        # 6. Evaluate final perplexity
        console.print(f"[dim]  Computing final perplexity...[/dim]")
        final_ppl = compute_perplexity(model, tokenizer, data["texts"][:100], device)
        console.print(f"[dim]  Final perplexity: {final_ppl:.2f}[/dim]")

        # Convert perplexity to accuracy-like metric
        initial_acc = max(0, 100 - initial_ppl)
        final_acc = max(0, 100 - final_ppl)
        improvement = final_acc - initial_acc

        console.print(f"[green]  Improvement: {improvement:+.2f}% (ppl: {initial_ppl:.2f} → {final_ppl:.2f})[/green]")

        # 7. Save gradient package (full trained model)

        # Save config
        new_config = model_config.copy()
        new_config["training"] = {
            "initial_perplexity": initial_ppl,
            "final_perplexity": final_ppl,
            "epochs": config.training_epochs,
        }
        with open(gradient_dir / "config.json", "w") as f:
            json.dump(new_config, f, indent=2)

        # Save trained model weights
        console.print(f"[dim]  Saving trained model...[/dim]")
        state_dict = model.state_dict()
        # Convert to fp16 for storage
        for key in state_dict:
            if state_dict[key].dtype == torch.float32:
                state_dict[key] = state_dict[key].half()
        save_safetensors(state_dict, str(gradient_dir / "model.safetensors"))

        # Copy tokenizer
        import shutil
        tokenizer_dest = gradient_dir / "tokenizer"
        if tokenizer_dest.exists():
            shutil.rmtree(tokenizer_dest)
        shutil.copytree(tokenizer_path, tokenizer_dest)

        return TrainingResult(
            success=True,
            gradient_dir=gradient_dir,
            initial_accuracy=initial_acc,
            final_accuracy=final_acc,
            improvement=improvement,
            epochs_trained=config.training_epochs,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return TrainingResult(success=False, error=str(e))


def compute_perplexity(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 4,
) -> float:
    """Compute perplexity on texts"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            outputs = model(**inputs, labels=inputs["input_ids"])

            loss = outputs.loss.item()
            num_tokens = inputs["attention_mask"].sum().item()

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity
