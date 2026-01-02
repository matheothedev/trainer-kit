"""
DECLOUD Trainer Configuration
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field


NETWORKS = {
    "devnet": {
        "rpc_url": "https://api.devnet.solana.com",
        "ws_url": "wss://api.devnet.solana.com",
    },
    "mainnet-beta": {
        "rpc_url": "https://api.mainnet-beta.solana.com",
        "ws_url": "wss://api.mainnet-beta.solana.com",
    },
}

DEFAULT_PROGRAM_ID = "HvQ8c3BBCsibJpH74UDXbvqEidJymzFYyGjNjRn7MYwC"


@dataclass
class TrainingRule:
    """
    Rule for automatic training.
    
    Example:
        rule = TrainingRule(
            name="cifar-expert",
            accept_datasets=["cifar10", "cifar100"],  # Which datasets to accept
            train_dataset="cifar10",                   # Which dataset to train on
            epochs=2,
            min_reward=0.01,
        )
    """
    name: str                                    # Rule name
    accept_datasets: List[str] = field(default_factory=list)  # Which creator datasets to accept (empty = all)
    train_dataset: Optional[str] = None          # Which dataset to train on (None = same as creator requested)
    train_path: Optional[str] = None             # Custom path to data
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.001
    max_batches: int = 0                         # 0 = all
    min_reward: float = 0.0                      # Minimum SOL
    max_reward: float = 0.0                      # 0 = no limit
    enabled: bool = True
    
    def matches(self, creator_dataset: str, reward_lamports: int) -> bool:
        """Check if round matches this rule"""
        if not self.enabled:
            return False
        
        reward_sol = reward_lamports / 1e9
        
        # Check reward
        if self.min_reward > 0 and reward_sol < self.min_reward:
            return False
        if self.max_reward > 0 and reward_sol > self.max_reward:
            return False
        
        # Check dataset
        if self.accept_datasets:
            if creator_dataset.lower() not in [d.lower() for d in self.accept_datasets]:
                return False
        
        return True
    
    def get_training_config(self, creator_dataset: str) -> dict:
        """Get training config"""
        return {
            "dataset": self.train_dataset or creator_dataset,
            "path": self.train_path,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_batches": self.max_batches,
        }
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingRule":
        return cls(**data)


# Pre-made rule templates
RULE_TEMPLATES = {
    "image-all": TrainingRule(
        name="image-all",
        accept_datasets=["cifar10", "cifar100", "mnist", "fashionmnist", "emnist", "kmnist", "svhn"],
        epochs=1,
    ),
    "mnist-family": TrainingRule(
        name="mnist-family",
        accept_datasets=["mnist", "fashionmnist", "emnist", "kmnist"],
        train_dataset="mnist",  # Train all on mnist
        epochs=1,
    ),
    "cifar-family": TrainingRule(
        name="cifar-family",
        accept_datasets=["cifar10", "cifar100"],
        train_dataset="cifar10",
        epochs=2,
    ),
    "text-all": TrainingRule(
        name="text-all",
        accept_datasets=["imdb", "sst2", "agnews"],
        epochs=2,
    ),
    "high-reward": TrainingRule(
        name="high-reward",
        accept_datasets=[],  # Any
        min_reward=0.1,
        epochs=3,
    ),
    "quick-test": TrainingRule(
        name="quick-test",
        accept_datasets=["cifar10", "mnist"],
        epochs=1,
        max_batches=50,
    ),
}


@dataclass
class Config:
    """Trainer configuration"""
    
    network: str = "mainnet-beta"
    rpc_url: str = "https://api.mainnet-beta.solana.com"
    ws_url: str = "wss://api.mainnet-beta.solana.com"
    program_id: str = DEFAULT_PROGRAM_ID
    
    # Paths
    data_dir: str = "~/.decloud-trainer"
    idl_path: str = ""
    
    # Auth
    private_key: Optional[str] = None
    keypair_path: Optional[str] = None
    
    # IPFS
    ipfs_gateway: str = "https://ipfs.io/ipfs/"
    pinata_api_key: Optional[str] = None
    pinata_secret_key: Optional[str] = None
    
    # === RULES ===
    # List of rules (checked in order, first match is used)
    rules: List[Dict] = field(default_factory=list)
    
    # === FALLBACK settings (if no rules or nothing matched) ===
    # Accept any datasets?
    accept_any: bool = False
    
    # Which datasets to ACCEPT (empty = all)
    datasets: List[str] = field(default_factory=list)
    
    # Your LOCAL dataset to train on (None = same as creator requested)
    train_dataset: Optional[str] = None
    
    # Default training settings
    default_epochs: int = 1
    default_batch_size: int = 32
    default_learning_rate: float = 0.001
    default_max_batches: int = 0
    
    # Reward filters (global)
    min_reward: float = 0.0
    max_reward: float = 0.0
    
    # Automation
    max_concurrent_rounds: int = 1
    auto_train: bool = True
    auto_claim: bool = True
    poll_interval: int = 5
    @property
    def batch_size(self):
        return self.default_batch_size
    @batch_size.setter
    def batch_size(self, value):
        self.default_batch_size = value

    @property
    def epochs(self):
        return self.default_epochs

    @epochs.setter
    def epochs(self, value):
        self.default_epochs = value

    @property
    def learning_rate(self):
        return self.default_learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.default_learning_rate = value

    @property
    def max_batches(self):
        return self.default_max_batches

    @max_batches.setter
    def max_batches(self, value):
        self.default_max_batches = value

    def __post_init__(self):
        self.data_dir = os.path.expanduser(self.data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "gradients"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "datasets"), exist_ok=True)
        
        if not self.idl_path:
            self.idl_path = self._find_idl()
        self.idl_path = self.idl_path.replace("\\", "/")
        
        self.pinata_api_key = self.pinata_api_key or os.getenv("PINATA_API_KEY")
        self.pinata_secret_key = self.pinata_secret_key or os.getenv("PINATA_SECRET_KEY")
    
    def _find_idl(self) -> str:
        package_dir = os.path.dirname(os.path.abspath(__file__))
        kit_dir = os.path.dirname(package_dir)
        
        search_paths = [
            os.path.join(kit_dir, "idl.json"),
            os.path.join(package_dir, "idl.json"),
            os.path.join(os.getcwd(), "idl.json"),
            os.path.expanduser("~/.decloud/idl.json"),
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        return os.path.join(kit_dir, "idl.json")
    
    def get_rules(self) -> List[TrainingRule]:
        """Get rules list as objects"""
        return [TrainingRule.from_dict(r) for r in self.rules]
    
    def add_rule(self, rule: TrainingRule):
        """Add a rule"""
        # Remove if already exists with same name
        self.rules = [r for r in self.rules if r.get("name") != rule.name]
        self.rules.append(rule.to_dict())
    
    def remove_rule(self, name: str) -> bool:
        """Remove rule by name"""
        old_len = len(self.rules)
        self.rules = [r for r in self.rules if r.get("name") != name]
        return len(self.rules) < old_len
    
    def add_template(self, template_name: str) -> bool:
        """Add rule from template"""
        if template_name not in RULE_TEMPLATES:
            return False
        self.add_rule(RULE_TEMPLATES[template_name])
        return True
    
    def find_matching_rule(self, creator_dataset: str, reward_lamports: int) -> Optional[TrainingRule]:
        """Find first matching rule for round"""
        for rule_dict in self.rules:
            rule = TrainingRule.from_dict(rule_dict)
            if rule.matches(creator_dataset, reward_lamports):
                return rule
        return None
    
    def matches_round(self, dataset: str, reward_lamports: int) -> bool:
        """Check if round matches our settings"""
        reward_sol = reward_lamports / 1e9
        
        # Global filters
        if self.min_reward > 0 and reward_sol < self.min_reward:
            return False
        if self.max_reward > 0 and reward_sol > self.max_reward:
            return False
        
        # Check global dataset filter
        if self.datasets:
            if dataset.lower() not in [d.lower() for d in self.datasets]:
                return False
        
        # Check rules
        if self.rules:
            rule = self.find_matching_rule(dataset, reward_lamports)
            if rule:
                return True
            # If rules exist but nothing matched
            return self.accept_any
        
        # No rules - accept based on dataset filter or accept_any
        return True
    
    def get_training_config(self, creator_dataset: str, reward_lamports: int) -> dict:
        """Get training config for round"""
        rule = self.find_matching_rule(creator_dataset, reward_lamports)
        
        if rule:
            return rule.get_training_config(creator_dataset)
        
        # Fallback defaults - use global train_dataset if set
        return {
            "dataset": self.train_dataset or creator_dataset,
            "path": None,
            "epochs": self.default_epochs,
            "batch_size": self.default_batch_size,
            "learning_rate": self.default_learning_rate,
            "max_batches": 0,
        }
    
    @classmethod
    def load(cls) -> "Config":
        config_path = os.path.expanduser("~/.decloud-trainer/config.json")
        
        if os.path.exists(config_path):
            with open(config_path) as f:
                data = json.load(f)
                return cls(**data)
        
        return cls()
    
    def save(self):
        config_path = os.path.join(self.data_dir, "config.json")
        
        data = {}
        for k, v in asdict(self).items():
            if v is not None:
                data[k] = v
        
        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        
        return config_path
    
    def print_settings(self):
        """Print current settings"""
        print("  ⚙️  Settings:")
        print(f"     Accept any:   {self.accept_any}")
        print(f"     Min reward:   {self.min_reward} SOL")
        print(f"     Max reward:   {self.max_reward if self.max_reward > 0 else 'unlimited'} SOL")
        print(f"     Max parallel: {self.max_concurrent_rounds}")
        print(f"     Auto-train:   {self.auto_train}")
        print(f"     Auto-claim:   {self.auto_claim}")
        
        if self.rules:
            print(f"\n  📋 Rules ({len(self.rules)}):")
            for r in self.get_rules():
                status = "✓" if r.enabled else "✗"
                datasets = ", ".join(r.accept_datasets) if r.accept_datasets else "any"
                train_on = r.train_dataset or "(same)"
                print(f"     [{status}] {r.name}")
                print(f"         Accept: {datasets}")
                print(f"         Train:  {train_on}, {r.epochs} epochs")
                if r.min_reward > 0:
                    print(f"         Min:    {r.min_reward} SOL")
    
    @staticmethod
    def list_templates():
        """List available rule templates"""
        print("\n  📋 Available Templates:")
        print("  " + "─" * 55)
        for name, rule in RULE_TEMPLATES.items():
            datasets = ", ".join(rule.accept_datasets) if rule.accept_datasets else "any"
            train_on = rule.train_dataset or "(same as creator)"
            print(f"  {name}")
            print(f"      Accept:  {datasets}")
            print(f"      Train:   {train_on}")
            print(f"      Epochs:  {rule.epochs}")
            if rule.min_reward > 0:
                print(f"      Min reward: {rule.min_reward} SOL")
            if rule.max_batches > 0:
                print(f"      Max batches: {rule.max_batches}")
            print()