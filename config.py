"""
Configuration management for Decloud Trainer
"""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Paths
HOME_DIR = Path.home()
CONFIG_DIR = HOME_DIR / ".decloud-trainer"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHE_DIR = CONFIG_DIR / "cache"
MODELS_CACHE = CACHE_DIR / "models"
GRADIENTS_DIR = CONFIG_DIR / "gradients"

# Solana
PROGRAM_ID = "HvQ8c3BBCsibJpH74UDXbvqEidJymzFYyGjNjRn7MYwC"
TREASURY = "FzuCxi65QyFXAGbHcXB28RXqyBZSZ5KXLQxeofx1P9K2"

# RPC Endpoints
RPC_ENDPOINTS = {
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com",
    "testnet": "https://api.testnet.solana.com",
}

# WebSocket endpoints
WS_ENDPOINTS = {
    "devnet": "wss://api.devnet.solana.com",
    "mainnet": "wss://api.mainnet-beta.solana.com",
    "testnet": "wss://api.testnet.solana.com",
}

# IPFS Gateways for reading
IPFS_GATEWAYS = [
    "https://gateway.pinata.cloud/ipfs/",
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://dweb.link/ipfs/",
]

# Dataset enum mapping (must match contract)
DATASETS = {
    "Cifar10": 0, "Cifar100": 1, "Mnist": 2, "FashionMnist": 3, "Emnist": 4,
    "Kmnist": 5, "Food101": 6, "Flowers102": 7, "StanfordDogs": 8, "StanfordCars": 9,
    "OxfordPets": 10, "CatsVsDogs": 11, "Eurosat": 12, "Svhn": 13, "Caltech101": 14,
    "Caltech256": 15, "Imdb": 16, "Sst2": 17, "Sst5": 18, "YelpReviews": 19,
    "AmazonPolarity": 20, "RottenTomatoes": 21, "FinancialSentiment": 22, "TweetSentiment": 23,
    "AgNews": 24, "Dbpedia": 25, "YahooAnswers": 26, "TwentyNewsgroups": 27,
    "SmsSpam": 28, "HateSpeech": 29, "CivilComments": 30, "Toxicity": 31,
    "ClincIntent": 32, "Banking77": 33, "SnipsIntent": 34, "Conll2003": 35,
    "Wnut17": 36, "Squad": 37, "SquadV2": 38, "TriviaQa": 39, "BoolQ": 40,
    "CommonsenseQa": 41, "Stsb": 42, "Mrpc": 43, "Qqp": 44, "Snli": 45,
    "Mnli": 46, "CnnDailymail": 47, "Xsum": 48, "Samsum": 49, "SpeechCommands": 50,
    "Librispeech": 51, "CommonVoice": 52, "Gtzan": 53, "Esc50": 54, "Urbansound8k": 55,
    "Nsynth": 56, "Ravdess": 57, "CremaD": 58, "Iemocap": 59, "Iris": 60,
    "Wine": 61, "Diabetes": 62, "BreastCancer": 63, "CaliforniaHousing": 64,
    "AdultIncome": 65, "BankMarketing": 66, "CreditDefault": 67, "Titanic": 68,
    "HeartDisease": 69, "ChestXray": 70, "SkinCancer": 71, "DiabeticRetinopathy": 72,
    "BrainTumor": 73, "Malaria": 74, "BloodCells": 75, "CovidXray": 76,
    "PubmedQa": 77, "MedQa": 78, "Electricity": 79, "Weather": 80, "StockPrices": 81,
    "EcgHeartbeat": 82, "CodeSearchNet": 83, "Humaneval": 84, "Mbpp": 85,
    "Spider": 86, "Cora": 87, "Citeseer": 88, "Qm9": 89, "NslKdd": 90,
    "CreditCardFraud": 91, "Phishing": 92, "Movielens1m": 93, "Movielens100k": 94,
    "Xnli": 95, "AmazonReviewsMulti": 96, "Sberquad": 97,
}

DATASET_ID_TO_NAME = {v: k for k, v in DATASETS.items()}


class Config:
    """Trainer configuration"""
    
    def __init__(self):
        # Wallet
        self.private_key: Optional[str] = None
        self.network: str = "mainnet"
        
        # Pinata
        self.pinata_api_key: Optional[str] = None
        self.pinata_secret_key: Optional[str] = None
        self.pinata_jwt: Optional[str] = None  # Alternative to api_key + secret
        
        # Training settings
        self.min_reward: float = 0.01  # Minimum reward in SOL to participate
        self.max_concurrent_training: int = 1
        self.training_epochs: int = 5
        self.training_batch_size: int = 32
        self.learning_rate: float = 0.001
        
        # Dataset paths mapping: dataset_name -> local_path
        self.dataset_paths: Dict[str, str] = {}
        
        # Auto mode
        self.auto_train: bool = True
        self.poll_interval: int = 30
        
        self._ensure_dirs()
        self._load()
    
    def _ensure_dirs(self):
        """Create necessary directories"""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_CACHE.mkdir(parents=True, exist_ok=True)
        GRADIENTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load(self):
        """Load config from file"""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.private_key = data.get("private_key")
                self.network = data.get("network", "devnet")
                self.pinata_api_key = data.get("pinata_api_key")
                self.pinata_secret_key = data.get("pinata_secret_key")
                self.pinata_jwt = data.get("pinata_jwt")
                self.min_reward = data.get("min_reward", 0.01)
                self.max_concurrent_training = data.get("max_concurrent_training", 1)
                self.training_epochs = data.get("training_epochs", 5)
                self.training_batch_size = data.get("training_batch_size", 32)
                self.learning_rate = data.get("learning_rate", 0.001)
                self.dataset_paths = data.get("dataset_paths", {})
                self.auto_train = data.get("auto_train", True)
    
    def save(self):
        """Save config to file"""
        data = {
            "private_key": self.private_key,
            "network": self.network,
            "pinata_api_key": self.pinata_api_key,
            "pinata_secret_key": self.pinata_secret_key,
            "pinata_jwt": self.pinata_jwt,
            "min_reward": self.min_reward,
            "max_concurrent_training": self.max_concurrent_training,
            "training_epochs": self.training_epochs,
            "training_batch_size": self.training_batch_size,
            "learning_rate": self.learning_rate,
            "dataset_paths": self.dataset_paths,
            "auto_train": self.auto_train,
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)
    
    @property
    def rpc_url(self) -> str:
        return RPC_ENDPOINTS.get(self.network, RPC_ENDPOINTS["devnet"])
    
    @property
    def ws_url(self) -> str:
        return WS_ENDPOINTS.get(self.network, WS_ENDPOINTS["devnet"])
    
    def has_pinata(self) -> bool:
        """Check if Pinata is configured"""
        return bool(self.pinata_jwt or (self.pinata_api_key and self.pinata_secret_key))
    
    def get_dataset_path(self, dataset_name: str) -> Optional[str]:
        """Get local path for dataset"""
        return self.dataset_paths.get(dataset_name)
    
    def set_dataset_path(self, dataset_name: str, path: str):
        """Set local path for dataset"""
        self.dataset_paths[dataset_name] = path
        self.save()
    
    def remove_dataset_path(self, dataset_name: str):
        """Remove dataset path mapping"""
        if dataset_name in self.dataset_paths:
            del self.dataset_paths[dataset_name]
            self.save()
    
    def can_train(self, dataset_name: str) -> bool:
        """Check if we can train on this dataset"""
        return dataset_name in self.dataset_paths


# Global config instance
config = Config()
