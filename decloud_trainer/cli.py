#!/usr/bin/env python3
"""
DECLOUD Trainer CLI
===================

Command-line interface for training and submitting gradients.

Usage:
    decloud-trainer list              # List available rounds
    decloud-trainer join <id>         # Join a round
    decloud-trainer train <id>        # Train and submit gradients
    decloud-trainer auto              # Auto-trainer mode
    decloud-trainer claim <id>        # Claim reward
"""

import os
import sys
import argparse
from typing import Optional

from .config import Config, TrainingRule, RULE_TEMPLATES
from .trainer import Trainer
from .ipfs import setup_pinata_interactive


def get_private_key(args) -> str:
    """Get private key from args, env, config, or keypair file"""
    # 1. From command line
    if hasattr(args, 'private_key') and args.private_key:
        return args.private_key
    
    # 2. From env
    key = os.getenv("DECLOUD_PRIVATE_KEY") or os.getenv("SOLANA_PRIVATE_KEY")
    if key:
        return key
    
    # 3. From config
    config = Config.load()
    if config.private_key:
        return config.private_key
    
    # 4. From keypair file
    if hasattr(args, 'keypair') and args.keypair:
        keypair_path = os.path.expanduser(args.keypair)
    elif config.keypair_path:
        keypair_path = os.path.expanduser(config.keypair_path)
    else:
        keypair_path = os.path.expanduser("~/.config/solana/id.json")
    
    if os.path.exists(keypair_path):
        import subprocess
        
        # Convert JSON array to base58
        script = f'''
const fs = require("fs");
const bs58 = require("bs58");
const secret = JSON.parse(fs.readFileSync("{keypair_path}"));
console.log(bs58.encode(Buffer.from(secret)));
'''
        result = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    
    print("❌ No private key found!")
    print("   Options:")
    print("   1. decloud-trainer config --set private_key=YOUR_KEY")
    print("   2. decloud-trainer -k YOUR_KEY <command>")
    print("   3. export DECLOUD_PRIVATE_KEY=YOUR_KEY")
    print("   4. Use default Solana keypair (~/.config/solana/id.json)")
    sys.exit(1)


def cmd_list(args):
    """List available rounds"""
    config = Config.load()
    trainer = Trainer(config)
    
    print("═" * 70)
    print("  📋 Available Rounds")
    print("═" * 70)
    
    status_filter = args.status if hasattr(args, 'status') else None
    rounds = trainer.list_rounds(status_filter=status_filter)
    
    if not rounds:
        print("\n  No rounds found.")
        return
    
    print()
    for r in rounds:
        status_icons = {
            "waitingValidator": "⏳",
            "waitingTrainers": "👥",
            "training": "🏋️",
            "validating": "🔬",
            "completed": "✅",
            "expired": "⏰",
            "cancelled": "❌",
        }
        icon = status_icons.get(r.status, "❓")
        reward = r.reward_amount / 1e9
        
        print(f"  {icon} #{r.id:3d} | {r.status:18s} | {r.dataset:15s} | {reward:.4f} SOL | {r.trainers_count} trainers")
    
    print()
    
    # Hint
    joinable = [r for r in rounds if r.status == "waitingTrainers"]
    if joinable:
        print(f"  💡 Tip: Join with `decloud-trainer join {joinable[0].id}`")


def cmd_join(args):
    """Join a round"""
    config = Config.load()
    trainer = Trainer(config)
    
    private_key = get_private_key(args)
    pubkey, balance = trainer.login(private_key)
    
    print(f"  Wallet:  {pubkey[:20]}...")
    print(f"  Balance: {balance:.4f} SOL")
    print()
    
    # Get round info
    round_info = trainer.get_round(args.round_id)
    
    print(f"  Round #{args.round_id}:")
    print(f"    Dataset: {round_info.dataset}")
    print(f"    Reward:  {round_info.reward_amount / 1e9:.4f} SOL")
    print(f"    Status:  {round_info.status}")
    print()
    
    if round_info.status != "waitingTrainers":
        print(f"  ❌ Cannot join: round is in '{round_info.status}' status")
        print(f"     Only 'waitingTrainers' rounds can be joined.")
        sys.exit(1)
    
    try:
        tx = trainer.join_round(args.round_id)
        print(f"  ✅ Joined! TX: {tx[:40]}...")
        print()
        print(f"  Next: Wait for training to start, then run:")
        print(f"        decloud-trainer train {args.round_id}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        sys.exit(1)


def cmd_train(args):
    """Train and submit gradients"""
    config = Config.load()
    trainer = Trainer(config)
    
    private_key = get_private_key(args)
    pubkey, balance = trainer.login(private_key)
    
    print("═" * 60)
    print(f"  🏋️ DECLOUD Trainer")
    print("═" * 60)
    print(f"  Wallet:  {pubkey[:20]}...")
    print(f"  Balance: {balance:.4f} SOL")
    print("═" * 60)
    
    try:
        trainer.train_and_submit(
            round_id=args.round_id,
            epochs=args.epochs,
            max_batches=args.max_batches,
        )
    except Exception as e:
        print(f"\n  ❌ Error: {e}")
        sys.exit(1)


def cmd_auto(args):
    """Start daemon mode"""
    config = Config.load()
    
    # Apply template if specified
    if hasattr(args, 'preset') and args.preset:
        if args.preset in RULE_TEMPLATES:
            config.add_template(args.preset)
            print(f"  📋 Using template: {args.preset}")
        else:
            print(f"  ❌ Unknown template: {args.preset}")
            print(f"  Available: {', '.join(RULE_TEMPLATES.keys())}")
            return
    
    # Quick overrides for simple cases
    if args.dataset:
        # Create a temporary rule
        datasets = [d.strip().lower() for d in args.dataset.split(",")]
        quick_rule = TrainingRule(
            name="_cli_override",
            accept_datasets=datasets,
            epochs=args.epochs if args.epochs else config.default_epochs,
        )
        config.rules = [quick_rule.to_dict()]
    
    if args.min_reward:
        config.min_reward = args.min_reward
    if args.max_reward:
        config.max_reward = args.max_reward
    if args.epochs and not args.dataset:
        config.default_epochs = args.epochs
    if args.max_parallel:
        config.max_concurrent_rounds = args.max_parallel
    if args.no_auto_train:
        config.auto_train = False
    if args.no_auto_claim:
        config.auto_claim = False
    
    # If no rules, set accept_any
    if not config.rules and not args.dataset:
        config.accept_any = True
    
    trainer = Trainer(config)
    
    private_key = get_private_key(args)
    trainer.login(private_key)
    
    trainer.start_daemon()


def cmd_config(args):
    """Configure trainer settings"""
    config = Config.load()
    
    if args.show:
        print("═" * 60)
        print("  ⚙️  DECLOUD Trainer Configuration")
        print("═" * 60)
        print()
        print("  Wallet:")
        if config.private_key:
            print(f"    Key: {config.private_key[:8]}...{config.private_key[-4:]}")
        elif config.keypair_path:
            print(f"    Keypair: {config.keypair_path}")
        else:
            print(f"    Keypair: ~/.config/solana/id.json (default)")
        print()
        print("  Global filters:")
        print(f"    Accept any:    {config.accept_any}")
        print(f"    Min reward:    {config.min_reward} SOL")
        print(f"    Max reward:    {config.max_reward if config.max_reward > 0 else 'unlimited'} SOL")
        print()
        print("  Automation:")
        print(f"    Max parallel:  {config.max_concurrent_rounds}")
        print(f"    Auto-train:    {config.auto_train}")
        print(f"    Auto-claim:    {config.auto_claim}")
        print(f"    Poll interval: {config.poll_interval}s")
        print()
        print("  IPFS:")
        print(f"    Pinata: {'✓ configured' if config.pinata_api_key else '✗ not set'}")
        print()
        
        rules = config.get_rules()
        if rules:
            print(f"  📋 Training Rules ({len(rules)}):")
            print("  " + "─" * 50)
            for i, r in enumerate(rules, 1):
                status = "✓" if r.enabled else "✗"
                accept = ", ".join(r.accept_datasets) if r.accept_datasets else "any"
                train = r.train_dataset or "(same)"
                print(f"  {i}. [{status}] {r.name}")
                print(f"       Accept: {accept}")
                print(f"       Train:  {train} ({r.epochs} epochs)")
                if r.min_reward > 0:
                    print(f"       Min:    {r.min_reward} SOL")
                if r.train_path:
                    print(f"       Path:   {r.train_path}")
            print()
        else:
            print("  📋 No training rules configured")
            print("     Add with: decloud-trainer config --add-template image-all")
            print("     Or:       decloud-trainer config --add-rule")
            print()
        
        return
    
    if args.templates:
        Config.list_templates()
        return
    
    if args.add_template:
        if config.add_template(args.add_template):
            config.save()
            print(f"  ✅ Added template: {args.add_template}")
        else:
            print(f"  ❌ Unknown template: {args.add_template}")
            print(f"  Available: {', '.join(RULE_TEMPLATES.keys())}")
        return
    
    if args.add_rule:
        print("  📝 Create new rule")
        print("  " + "─" * 40)
        
        name = input("  Name: ").strip()
        if not name:
            print("  ❌ Name required")
            return
        
        print("  Accept datasets (comma-separated, empty=any):")
        accept = input("  > ").strip()
        accept_list = [d.strip().lower() for d in accept.split(",") if d.strip()] if accept else []
        
        print("  Train on dataset (empty=same as creator):")
        train = input("  > ").strip().lower() or None
        
        print("  Custom path to data (empty=download):")
        path = input("  > ").strip() or None
        
        epochs = input("  Epochs [1]: ").strip()
        epochs = int(epochs) if epochs else 1
        
        min_rew = input("  Min reward SOL [0]: ").strip()
        min_rew = float(min_rew) if min_rew else 0.0
        
        rule = TrainingRule(
            name=name,
            accept_datasets=accept_list,
            train_dataset=train,
            train_path=path,
            epochs=epochs,
            min_reward=min_rew,
        )
        
        config.add_rule(rule)
        config.save()
        print(f"\n  ✅ Rule '{name}' added!")
        return
    
    if args.remove_rule:
        if config.remove_rule(args.remove_rule):
            config.save()
            print(f"  ✅ Rule '{args.remove_rule}' removed")
        else:
            print(f"  ❌ Rule '{args.remove_rule}' not found")
        return
    
    if args.clear_rules:
        config.rules = []
        config.save()
        print("  ✅ All rules cleared")
        return
    
    if args.set:
        key, value = args.set.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Type conversion
        if key in ("min_reward", "max_reward", "default_learning_rate"):
            value = float(value)
        elif key in ("default_epochs", "default_batch_size", "max_concurrent_rounds", "poll_interval"):
            value = int(value)
        elif key in ("auto_train", "auto_claim", "accept_any"):
            value = value.lower() in ("true", "1", "yes")
        
        if hasattr(config, key):
            setattr(config, key, value)
            config.save()
            
            if key == "private_key":
                print(f"  ✅ Set {key} = {value[:8]}...{value[-4:]}")
            else:
                print(f"  ✅ Set {key} = {value}")
        else:
            print(f"  ❌ Unknown setting: {key}")
        return
    
    if args.reset:
        new_config = Config()
        new_config.save()
        print("  ✅ Config reset to defaults")
        return
    
    # Default: show help
    print("  Usage:")
    print("    decloud-trainer config --show              # View config")
    print("    decloud-trainer config --templates         # List templates")
    print("    decloud-trainer config --add-template X    # Add template")
    print("    decloud-trainer config --add-rule          # Create custom rule")
    print("    decloud-trainer config --remove-rule NAME  # Remove rule")
    print("    decloud-trainer config --set key=value     # Set option")
    print("    decloud-trainer config --reset             # Reset all")


def cmd_setup(args):
    """Interactive setup"""
    print("═" * 60)
    print("  🛠️  DECLOUD Trainer Setup")
    print("═" * 60)
    
    config = Config.load()
    
    # 0. Wallet
    print("\n0️⃣  Wallet (private key)")
    print("   Options:")
    print("   [1] Use Solana CLI keypair (~/.config/solana/id.json)")
    print("   [2] Enter private key (base58)")
    print("   [3] Skip (use env variable later)")
    
    wallet_choice = input("   > ").strip()
    
    if wallet_choice == "1":
        config.keypair_path = "~/.config/solana/id.json"
        config.private_key = None
        print("   ✓ Will use Solana CLI keypair")
    elif wallet_choice == "2":
        pk = input("   Enter base58 private key: ").strip()
        if pk:
            config.private_key = pk
            config.keypair_path = None
            print(f"   ✓ Saved: {pk[:8]}...{pk[-4:]}")
    
    # 1. Datasets to ACCEPT
    print("\n1️⃣  Which datasets do you want to ACCEPT?")
    print("   Leave empty to accept ALL rounds")
    print("   Or enter comma-separated list (e.g. cifar10,mnist,imdb)")
    datasets_input = input("   > ").strip()
    
    if datasets_input:
        config.datasets = [d.strip().lower() for d in datasets_input.split(",")]
    else:
        config.datasets = []
    
    # 2. Your LOCAL dataset to train on
    print("\n2️⃣  Your LOCAL dataset to train on?")
    print("   This is YOUR data that you'll use for training")
    print("   Available: cifar10, cifar100, mnist, fashionmnist")
    print("   Leave empty to use same as creator requested")
    train_dataset = input("   > ").strip().lower()
    
    if train_dataset:
        config.train_dataset = train_dataset
        print(f"   ✓ Will train on: {train_dataset}")
    else:
        config.train_dataset = None
        print("   ✓ Will use creator's dataset")
    
    # 3. Rewards
    print("\n3️⃣  Minimum reward (SOL)?")
    print("   Enter 0 for no minimum")
    min_reward = input("   > ").strip()
    config.min_reward = float(min_reward) if min_reward else 0.0
    
    print("\n4️⃣  Maximum reward (SOL)?")
    print("   Enter 0 for no maximum (unlimited)")
    max_reward = input("   > ").strip()
    config.max_reward = float(max_reward) if max_reward else 0.0
    
    # 4. Training
    print("\n5️⃣  Epochs per round?")
    epochs = input("   [1] > ").strip()
    config.epochs = int(epochs) if epochs else 1
    
    print("\n6️⃣  Max parallel rounds?")
    max_parallel = input("   [1] > ").strip()
    config.max_concurrent_rounds = int(max_parallel) if max_parallel else 1
    
    # 5. IPFS
    print("\n7️⃣  Setup IPFS (Pinata)?")
    print("   Required for uploading gradients!")
    print("   [y/n]")
    if input("   > ").strip().lower() == 'y':
        api_key, secret_key = setup_pinata_interactive()
        if api_key:
            config.pinata_api_key = api_key
            config.pinata_secret_key = secret_key
    
    # Save
    path = config.save()
    
    print("\n" + "═" * 60)
    print("  ✅ Setup complete!")
    print("═" * 60)
    print(f"  Config saved to: {path}")
    print()
    print("  Start daemon with:")
    print("    decloud-trainer start")
    print()


def cmd_claim(args):
    """Claim reward"""
    config = Config.load()
    trainer = Trainer(config)
    
    private_key = get_private_key(args)
    pubkey, balance = trainer.login(private_key)
    
    print(f"  Wallet: {pubkey[:20]}...")
    print(f"  Claiming reward for round #{args.round_id}...")
    
    try:
        # Check round status
        round_info = trainer.get_round(args.round_id)
        
        if round_info.status != "completed":
            print(f"  ❌ Cannot claim: round is '{round_info.status}', not 'completed'")
            sys.exit(1)
        
        tx = trainer.claim_reward(args.round_id)
        print(f"  ✅ Claimed! TX: {tx[:40]}...")
        
        # Check new balance
        _, new_balance = trainer.login(private_key)
        print(f"  New balance: {new_balance:.4f} SOL (+{new_balance - balance:.4f})")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        sys.exit(1)


def cmd_status(args):
    """Check round status"""
    config = Config.load()
    trainer = Trainer(config)
    
    round_info = trainer.get_round(args.round_id)
    
    print("═" * 60)
    print(f"  Round #{round_info.id}")
    print("═" * 60)
    
    status_icons = {
        "waitingValidator": "⏳",
        "waitingTrainers": "👥",
        "training": "🏋️",
        "validating": "🔬",
        "completed": "✅",
        "expired": "⏰",
        "cancelled": "❌",
    }
    icon = status_icons.get(round_info.status, "❓")
    
    print(f"  Status:      {icon} {round_info.status}")
    print(f"  Dataset:     {round_info.dataset}")
    print(f"  Reward:      {round_info.reward_amount / 1e9:.4f} SOL")
    print(f"  Model CID:   {round_info.model_cid}")
    print(f"  Trainers:    {round_info.trainers_count}")
    print(f"  Submissions: {round_info.submissions_count}")
    
    if round_info.validator != "11111111111111111111111111111111":
        print(f"  Validator:   {round_info.validator[:20]}...")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="DECLOUD Trainer - Train models and earn rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  decloud-trainer setup                    # Interactive setup
  decloud-trainer start                    # Start daemon (main command!)
  decloud-trainer start --dataset cifar10  # Only cifar10 rounds
  decloud-trainer start --min-reward 0.01  # Min 0.01 SOL reward
  
  decloud-trainer list                     # List available rounds
  decloud-trainer join 42                  # Join round #42 manually
  decloud-trainer train 42                 # Train manually
  decloud-trainer claim 42                 # Claim reward
  
  decloud-trainer config --show            # View settings
  decloud-trainer config --set epochs=3    # Change setting
        """
    )
    
    # Global args
    parser.add_argument("-k", "--private-key", help="Wallet private key (base58)")
    parser.add_argument("--keypair", help="Path to Solana keypair JSON")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Setup
    setup_parser = subparsers.add_parser("setup", help="Interactive setup")
    
    # Start (main command - daemon)
    start_parser = subparsers.add_parser("start", help="Start trainer daemon")
    start_parser.add_argument("--preset", help="Use a preset (image-all, mnist-only, etc)")
    start_parser.add_argument("--dataset", help="Filter by datasets (comma-separated)")
    start_parser.add_argument("--min-reward", type=float, help="Minimum reward in SOL")
    start_parser.add_argument("--max-reward", type=float, help="Maximum reward in SOL")
    start_parser.add_argument("--epochs", type=int, help="Training epochs")
    start_parser.add_argument("--max-parallel", type=int, help="Max parallel rounds")
    start_parser.add_argument("--no-auto-train", action="store_true", help="Don't auto-train (just join)")
    start_parser.add_argument("--no-auto-claim", action="store_true", help="Don't auto-claim rewards")
    
    # Config
    config_parser = subparsers.add_parser("config", help="View/edit configuration")
    config_parser.add_argument("--show", action="store_true", help="Show current config")
    config_parser.add_argument("--templates", action="store_true", help="List rule templates")
    config_parser.add_argument("--add-template", help="Add rule from template")
    config_parser.add_argument("--add-rule", action="store_true", help="Create custom rule (interactive)")
    config_parser.add_argument("--remove-rule", help="Remove rule by name")
    config_parser.add_argument("--clear-rules", action="store_true", help="Remove all rules")
    config_parser.add_argument("--set", help="Set a value (key=value)")
    config_parser.add_argument("--reset", action="store_true", help="Reset to defaults")
    
    # List
    list_parser = subparsers.add_parser("list", help="List available rounds")
    list_parser.add_argument("--status", help="Filter by status")
    
    # Join
    join_parser = subparsers.add_parser("join", help="Join a round")
    join_parser.add_argument("round_id", type=int, help="Round ID")
    
    # Train
    train_parser = subparsers.add_parser("train", help="Train and submit gradients")
    train_parser.add_argument("round_id", type=int, help="Round ID")
    train_parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    train_parser.add_argument("--max-batches", type=int, help="Max batches per epoch (for testing)")
    
    # Auto (deprecated, alias for start)
    auto_parser = subparsers.add_parser("auto", help="[deprecated] Use 'start' instead")
    auto_parser.add_argument("--dataset", help="Filter by datasets")
    auto_parser.add_argument("--min-reward", type=float, default=0, help="Minimum reward in SOL")
    auto_parser.add_argument("--max-reward", type=float, help="Maximum reward in SOL")
    auto_parser.add_argument("--epochs", type=int, help="Training epochs")
    auto_parser.add_argument("--max-parallel", type=int, help="Max parallel rounds")
    auto_parser.add_argument("--no-auto-train", action="store_true")
    auto_parser.add_argument("--no-auto-claim", action="store_true")
    
    # Claim
    claim_parser = subparsers.add_parser("claim", help="Claim reward")
    claim_parser.add_argument("round_id", type=int, help="Round ID")
    
    # Status
    status_parser = subparsers.add_parser("status", help="Check round status")
    status_parser.add_argument("round_id", type=int, help="Round ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    commands = {
        "setup": cmd_setup,
        "start": cmd_auto,  # start is the main command now
        "auto": cmd_auto,   # deprecated alias
        "config": cmd_config,
        "list": cmd_list,
        "join": cmd_join,
        "train": cmd_train,
        "claim": cmd_claim,
        "status": cmd_status,
    }
    
    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()