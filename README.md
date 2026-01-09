# Decloud Trainer Kit

Train models and earn rewards on Solana federated learning network.

## Features

- 🔐 Wallet + Pinata IPFS integration
- 📡 WebSocket real-time round detection
- 🏋️ Automatic training and submission
- 💰 Easy reward claiming
- ⚙️ Configurable training parameters

## Installation

```bash
cd decloud-trainer-kit

# Virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Setup

```bash
decloud-trainer setup
```

Prompts for:
- Solana wallet private key
- Network (devnet/mainnet)
- Pinata JWT or API keys
- Training parameters

### 2. Configure Datasets

```bash
# Set path for each dataset you want to train on
decloud-trainer dataset set Cifar10 /path/to/cifar10_data
decloud-trainer dataset set Mnist /path/to/mnist_data

# View configured datasets
decloud-trainer dataset list
```

### 3. Start Training

```bash
decloud-trainer start
```

## Commands

### Setup & Config

| Command | Description |
|---------|-------------|
| `setup` | Interactive setup wizard |
| `status` | Show trainer status |
| `balance` | Show wallet balance |
| `network -n <net>` | Change network |

### Dataset Configuration

| Command | Description |
|---------|-------------|
| `dataset list` | Show configured datasets |
| `dataset set <n> <path>` | Set dataset path |
| `dataset remove <n>` | Remove dataset |
| `dataset available` | Show all datasets |

### Training Settings

| Command | Description |
|---------|-------------|
| `settings show` | Show current settings |
| `settings set min_reward 0.05` | Set minimum reward |
| `settings set epochs 10` | Set training epochs |
| `settings set batch_size 64` | Set batch size |
| `settings set lr 0.0001` | Set learning rate |

### Training

| Command | Description |
|---------|-------------|
| `start` | Start auto-training (WebSocket) |
| `rounds` | Show active rounds |
| `train <round_id>` | Manually train for round |
| `info <round_id>` | Show round details |

### Rewards

| Command | Description |
|---------|-------------|
| `claim <round_id>` | Claim reward |

## Dataset Format

Your local dataset should contain numpy arrays:

```
my_dataset/
├── embeddings_train.npy   # (N, embedding_dim) float32
├── labels_train.npy       # (N,) int64
├── embeddings_test.npy    # (M, embedding_dim) float32
└── labels_test.npy        # (M,) int64
```

Alternative naming:
- `X_train.npy`, `y_train.npy`
- `train/embeddings.npy`, `train/labels.npy`

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW                            │
└─────────────────────────────────────────────────────────────┘

1. WebSocket detects RoundCreated event
   ↓
2. Check: reward >= min_reward? dataset configured?
   ↓
3. Download base model from IPFS
   ↓
4. Load YOUR local training data
   ↓
5. Train head model (fine-tune)
   ↓
6. Upload trained model to Pinata/IPFS
   ↓
7. Submit gradient CID to blockchain
   ↓
8. Wait for validators + round finalization
   ↓
9. Claim reward!
```

## Configuration

Config file: `~/.decloud-trainer/config.json`

```json
{
  "private_key": "...",
  "network": "devnet",
  "pinata_jwt": "...",
  "min_reward": 0.01,
  "training_epochs": 5,
  "training_batch_size": 32,
  "learning_rate": 0.001,
  "dataset_paths": {
    "Cifar10": "/home/user/data/cifar10",
    "Mnist": "/home/user/data/mnist"
  }
}
```

## Pinata Setup

1. Go to https://app.pinata.cloud/keys
2. Create new API key
3. Copy JWT token
4. Use in `decloud-trainer setup`

## Tips

- Use GPU for faster training: `CUDA_VISIBLE_DEVICES=0`
- Set higher `min_reward` to only train for profitable rounds
- Keep embeddings small (~1000 test samples) for faster upload

## License

MIT
