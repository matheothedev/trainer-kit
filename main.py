#!/usr/bin/env python3
"""
Decloud Trainer CLI
Train models and submit gradients for federated learning rounds
"""
import sys
import asyncio
import getpass
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from config import config, DATASETS
from trainer import DeCloudTrainer
from pinata_client import pinata_client

console = Console()


def get_trainer() -> DeCloudTrainer:
    """Get trainer instance"""
    if not config.private_key:
        console.print("[red]No private key configured. Run 'decloud-trainer setup' first.[/red]")
        sys.exit(1)
    return DeCloudTrainer(config.private_key)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """
    Decloud Trainer CLI
    
    Train models and earn rewards on Solana federated learning.
    """
    pass


# ═══════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════

@cli.command()
def setup():
    """Interactive setup wizard"""
    console.print("\n[bold cyan]=== Decloud Trainer Setup ===[/bold cyan]\n")
    
    # Private key
    console.print("[yellow]Enter your Solana wallet private key (base58)[/yellow]")
    private_key = getpass.getpass("Private Key: ")
    
    if not private_key:
        console.print("[red]Private key required[/red]")
        return
    
    try:
        from solana_client import SolanaClient
        client = SolanaClient.from_private_key(private_key)
        console.print(f"[green]✓ Wallet: {client.pubkey}[/green]")
    except Exception as e:
        console.print(f"[red]Invalid private key: {e}[/red]")
        return
    
    # Network
    console.print("\n[yellow]Select network:[/yellow]")
    for i, net in enumerate(["devnet", "mainnet", "testnet"], 1):
        console.print(f"  {i}. {net}")
    
    network_choice = Prompt.ask("Network", choices=["1", "2", "3"], default="1")
    network = ["devnet", "mainnet", "testnet"][int(network_choice) - 1]
    
    # Pinata
    console.print("\n[yellow]Pinata API for IPFS uploads[/yellow]")
    console.print("[dim]Get keys at: https://app.pinata.cloud/keys[/dim]")
    
    use_jwt = Confirm.ask("Use JWT token (recommended)?", default=True)
    
    if use_jwt:
        pinata_jwt = getpass.getpass("Pinata JWT: ")
        config.pinata_jwt = pinata_jwt if pinata_jwt else None
    else:
        api_key = Prompt.ask("Pinata API Key")
        secret_key = getpass.getpass("Pinata Secret Key: ")
        config.pinata_api_key = api_key if api_key else None
        config.pinata_secret_key = secret_key if secret_key else None
    
    # Save
    config.private_key = private_key
    config.network = network
    config.save()
    
    console.print(f"\n[green]✓ Configuration saved![/green]")
    
    # Test Pinata
    if config.has_pinata():
        console.print("[dim]Testing Pinata connection...[/dim]")
        if pinata_client.test_authentication_sync():
            console.print("[green]✓ Pinata connected![/green]")
        else:
            console.print("[red]✗ Pinata authentication failed[/red]")
    
    # Training settings
    console.print("\n[yellow]Training settings (press Enter for defaults):[/yellow]")
    
    min_reward = Prompt.ask("Minimum reward (SOL)", default="0.01")
    epochs = Prompt.ask("Training epochs", default="5")
    batch_size = Prompt.ask("Batch size", default="32")
    lr = Prompt.ask("Learning rate", default="0.001")
    
    config.min_reward = float(min_reward)
    config.training_epochs = int(epochs)
    config.training_batch_size = int(batch_size)
    config.learning_rate = float(lr)
    config.save()
    
    console.print("\n[cyan]Next steps:[/cyan]")
    console.print("  1. Configure datasets: [bold]decloud-trainer dataset set Cifar10 /path/to/data[/bold]")
    console.print("  2. Start training: [bold]decloud-trainer start[/bold]")


@cli.command()
@click.option("--network", "-n", type=click.Choice(["devnet", "mainnet", "testnet"]))
def network(network):
    """Change or show network"""
    if network:
        config.network = network
        config.save()
        console.print(f"[green]✓ Network: {network}[/green]")
    else:
        console.print(f"Network: [cyan]{config.network}[/cyan]")


# ═══════════════════════════════════════════════════════════════
# Dataset Configuration
# ═══════════════════════════════════════════════════════════════

@cli.group()
def dataset():
    """Dataset path configuration"""
    pass


@dataset.command("list")
def dataset_list():
    """List configured datasets"""
    if not config.dataset_paths:
        console.print("[yellow]No datasets configured[/yellow]")
        console.print("[dim]Run: decloud-trainer dataset set <n> <path>[/dim]")
        return
    
    table = Table(title="Configured Datasets")
    table.add_column("Dataset", style="cyan")
    table.add_column("Path", style="white")
    table.add_column("Exists", style="green")
    
    for name, path in config.dataset_paths.items():
        exists = "✓" if Path(path).exists() else "✗"
        table.add_row(name, path, exists)
    
    console.print(table)


@dataset.command("set")
@click.argument("name")
@click.argument("path")
def dataset_set(name, path):
    """Set path for a dataset"""
    if name not in DATASETS:
        console.print(f"[red]Unknown dataset: {name}[/red]")
        console.print(f"[dim]Available: {', '.join(list(DATASETS.keys())[:10])}...[/dim]")
        return
    
    path_obj = Path(path)
    if not path_obj.exists():
        console.print(f"[yellow]Warning: Path does not exist: {path}[/yellow]")
    
    config.set_dataset_path(name, str(path_obj.absolute()))
    console.print(f"[green]✓ {name} → {path}[/green]")


@dataset.command("remove")
@click.argument("name")
def dataset_remove(name):
    """Remove dataset configuration"""
    if name not in config.dataset_paths:
        console.print(f"[yellow]Dataset {name} not configured[/yellow]")
        return
    
    config.remove_dataset_path(name)
    console.print(f"[green]✓ Removed {name}[/green]")


@dataset.command("available")
def dataset_available():
    """Show available datasets"""
    console.print("[bold]Available Datasets:[/bold]")
    configured = set(config.dataset_paths.keys())
    
    categories = {
        "Image": ["Cifar10", "Cifar100", "Mnist", "FashionMnist", "Food101"],
        "Text": ["Imdb", "Sst2", "AgNews", "YelpReviews"],
        "Tabular": ["Iris", "Wine", "BreastCancer", "Diabetes"],
    }
    
    for cat, datasets in categories.items():
        console.print(f"\n[cyan]{cat}:[/cyan]")
        for ds in datasets:
            status = "[green]✓[/green]" if ds in configured else "[dim]○[/dim]"
            console.print(f"  {status} {ds}")


# ═══════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════

@cli.group()
def settings():
    """Training settings"""
    pass


@settings.command("show")
def settings_show():
    """Show current settings"""
    table = Table(title="Training Settings")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Min Reward", f"{config.min_reward} SOL")
    table.add_row("Epochs", str(config.training_epochs))
    table.add_row("Batch Size", str(config.training_batch_size))
    table.add_row("Learning Rate", str(config.learning_rate))
    table.add_row("Max Concurrent", str(config.max_concurrent_training))
    
    console.print(table)


@settings.command("set")
@click.argument("key")
@click.argument("value")
def settings_set(key, value):
    """Set a training setting"""
    mapping = {
        "min_reward": ("min_reward", float),
        "epochs": ("training_epochs", int),
        "batch_size": ("training_batch_size", int),
        "lr": ("learning_rate", float),
        "learning_rate": ("learning_rate", float),
    }
    
    if key not in mapping:
        console.print(f"[red]Unknown setting: {key}[/red]")
        console.print(f"[dim]Available: {', '.join(mapping.keys())}[/dim]")
        return
    
    attr, type_fn = mapping[key]
    setattr(config, attr, type_fn(value))
    config.save()
    console.print(f"[green]✓ {key} = {value}[/green]")


# ═══════════════════════════════════════════════════════════════
# Training Commands
# ═══════════════════════════════════════════════════════════════

@cli.command()
def start():
    """Start trainer (WebSocket real-time)"""
    trainer = get_trainer()
    
    if not config.dataset_paths:
        console.print("\n[yellow]No datasets configured![/yellow]")
        console.print("[dim]Run: decloud-trainer dataset set <n> <path>[/dim]")
        return
    
    if not config.has_pinata():
        console.print("\n[yellow]Pinata not configured![/yellow]")
        console.print("[dim]Run: decloud-trainer setup[/dim]")
        return
    
    try:
        asyncio.run(trainer.start())
    except KeyboardInterrupt:
        trainer.stop()


@cli.command()
def status():
    """Show trainer status"""
    trainer = get_trainer()
    trainer.show_status()


@cli.command()
@click.option("--limit", "-l", default=10)
def rounds(limit):
    """Show active rounds"""
    trainer = get_trainer()
    trainer.show_rounds(limit=limit)


@cli.command()
@click.argument("round_id", type=int)
def train(round_id):
    """Manually train for a specific round"""
    trainer = get_trainer()
    console.print(f"[cyan]Training for round {round_id}...[/cyan]")
    result = asyncio.run(trainer.train_and_submit(round_id))
    
    if result:
        console.print("[green]✓ Success![/green]")
    else:
        console.print("[red]✗ Failed[/red]")


# ═══════════════════════════════════════════════════════════════
# Rewards
# ═══════════════════════════════════════════════════════════════

@cli.command("claim")
@click.argument("round_id", type=int)
def claim_reward(round_id):
    """Claim reward from finalized round"""
    trainer = get_trainer()
    console.print(f"[cyan]Claiming from round {round_id}...[/cyan]")
    
    result = trainer.claim_reward(round_id)
    
    if result.get("success"):
        console.print(f"[green]✓ Reward claimed![/green]")
        console.print(f"[dim]TX: {result['tx']}[/dim]")
    else:
        console.print(f"[red]✗ {result.get('error')}[/red]")


@cli.command("balance")
def show_balance():
    """Show wallet balance"""
    trainer = get_trainer()
    try:
        balance = trainer.get_balance()
        console.print(f"Balance: [green]{balance:.6f} SOL[/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ═══════════════════════════════════════════════════════════════
# Info
# ═══════════════════════════════════════════════════════════════

@cli.command("info")
@click.argument("round_id", type=int)
def round_info(round_id):
    """Show round details"""
    trainer = get_trainer()
    
    info = trainer.solana.get_round(round_id)
    if not info:
        console.print(f"[red]Round {round_id} not found[/red]")
        return
    
    table = Table(title=f"Round {round_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Dataset", info.dataset)
    table.add_row("Reward", f"{info.reward_amount / 1e9:.4f} SOL")
    table.add_row("Status", info.status)
    table.add_row("Pre-validators", str(info.pre_count))
    table.add_row("Trainers", str(info.gradients_count))
    table.add_row("Model CID", info.model_cid[:40] + "...")
    
    console.print(table)
    
    # Check our submission
    submitted = trainer.solana.has_submitted_gradient(round_id)
    console.print(f"\nYour status: {'[green]✓ Submitted[/green]' if submitted else '[dim]Not submitted[/dim]'}")


def main():
    cli()


if __name__ == "__main__":
    main()
