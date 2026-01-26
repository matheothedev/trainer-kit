# Decloud Trainer Kit

Train models and earn rewards on Solana federated learning network.

## Features

- 🔐 Wallet + Lighthouse IPFS integration
- 📡 WebSocket real-time round detection
- 🏋️ Automatic training and submission
- ⭐ Trainer rating system
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
- Training parameters

Lighthouse Storage API key is created automatically from your Solana wallet.

### 2. Create Trainer Profile

```bash
# Required before training - creates on-chain profile
decloud-trainer create-profile

# View your profile
decloud-trainer profile
```

### 3. Configure Datasets

```bash
# Set path for each dataset you want to train on
decloud-trainer dataset set Cifar10 /path/to/cifar10_data
decloud-trainer dataset set Mnist /path/to/mnist_data

# View configured datasets
decloud-trainer dataset list
```

### 4. Start Training

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

### Profile Management

| Command | Description |
|---------|-------------|
| `create-profile` | Create trainer profile (required) |
| `profile` | Show your profile and rating |

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

## Trainer Rating System

New trainers start at **5.00 ★** rating.

- If your training makes the model worse (post accuracy < pre accuracy), your rating is slashed by **0.01 ★**
- Creators can set minimum rating requirements for their rounds
- Maintain quality training to keep your rating high

**Check if you can participate in a round:**
```bash
decloud-trainer info <round_id>
# Shows min rating requirement vs your rating
```

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
2. Check: reward >= min_reward? dataset configured? rating OK?
   ↓
3. Download base model from IPFS
   ↓
4. Load YOUR local training data
   ↓
5. Train head model (fine-tune)
   ↓
6. Upload trained model to Lighthouse/IPFS
   ↓
7. Submit gradient CID to blockchain
   ↓
8. Wait for validators + round finalization
   ↓
9. Claim reward! (rating updated based on performance)
```

## Configuration

Config file: `~/.decloud-trainer/config.json`

```json
{
  "private_key": "...",
  "network": "devnet",
  "lighthouse_api_key": "...",
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

## Lighthouse Storage

Lighthouse API key is created automatically during setup using your Solana wallet for authentication. No separate account needed!

## Tips

- **Create profile first!** You can't submit gradients without a profile
- Use GPU for faster training: `CUDA_VISIBLE_DEVICES=0`
- Set higher `min_reward` to only train for profitable rounds
- Keep embeddings small (~1000 test samples) for faster upload
- Maintain quality training to preserve your rating

## License

MIT