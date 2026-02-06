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
from lighthouse_client import LighthouseClient, get_lighthouse_client, init_lighthouse_client

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
    
    # Network — always mainnet
    network = "mainnet"
    console.print(f"[green]✓ Network: {network}[/green]")

    # Create Lighthouse API key automatically from Solana private key
    console.print("\n[yellow]Creating Lighthouse Storage API key...[/yellow]")
    console.print("[dim]Using your Solana wallet for IPFS uploads[/dim]")
    
    lighthouse_api_key = None
    try:
        lighthouse_api_key = LighthouseClient.create_api_key_from_private_key(
            private_key, 
            key_name=f"decloud-trainer-{str(client.pubkey)[:8]}"
        )
        console.print(f"[green]✓ Lighthouse API key created![/green]")
    except Exception as e:
        console.print(f"[red]✗ Failed to create Lighthouse API key: {e}[/red]")
        console.print("[dim]You can create it manually later[/dim]")
    
    # Save
    config.private_key = private_key
    config.network = network
    config.lighthouse_api_key = lighthouse_api_key
    config.save()
    
    console.print(f"\n[green]✓ Configuration saved![/green]")
    
    # Test Lighthouse
    if config.has_lighthouse():
        console.print("[dim]Testing Lighthouse connection...[/dim]")
        lighthouse = init_lighthouse_client(config.lighthouse_api_key)
        if lighthouse.test_authentication_sync():
            console.print("[green]✓ Lighthouse Storage connected![/green]")
        else:
            console.print("[red]✗ Lighthouse authentication failed[/red]")
    
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
def network():
    """Show current network (always mainnet)"""
    console.print(f"Network: [cyan]{config.network}[/cyan]")
    console.print(f"RPC: [dim]{config.rpc_url}[/dim]")


@cli.group()
def rpc():
    """RPC endpoint configuration"""
    pass


@rpc.command("set")
@click.argument("url")
def rpc_set(url):
    """Set custom RPC endpoint"""
    config.custom_rpc = url
    config.save()
    console.print(f"[green]✓ Custom RPC set: {url}[/green]")


@rpc.command("reset")
def rpc_reset():
    """Reset to default RPC (based on network)"""
    config.custom_rpc = None
    config.save()
    console.print(f"[green]✓ Reset to default RPC[/green]")
    console.print(f"[dim]Using: {config.rpc_url}[/dim]")


@rpc.command("show")
def rpc_show():
    """Show current RPC endpoint"""
    if config.custom_rpc:
        console.print(f"Custom RPC: [cyan]{config.custom_rpc}[/cyan]")
    else:
        console.print(f"Default RPC ({config.network}): [cyan]{config.rpc_url}[/cyan]")


@cli.command("allow-llm")
@click.option("--enable/--disable", default=None, help="No longer needed")
def allow_llm(enable):
    """LLM models now use the same flow as classification (no special flag needed)"""
    console.print("[green]LLM models now use the same training flow as classification.[/green]")
    console.print("[dim]No special flag needed. Just set up your dataset and train.[/dim]")


@cli.command("privacy-noise")
@click.option("--enable/--disable", default=None, help="Enable or disable privacy noise")
@click.option("--scale", type=float, help="Noise scale (e.g., 0.001 = 0.1%)")
def privacy_noise(enable, scale):
    """Configure privacy noise for weight submissions

    Privacy noise adds small random perturbations to model weights
    before uploading, making it harder to reverse-engineer training data.
    """
    if enable is not None:
        config.noise_enabled = enable
        config.save()
        status = "enabled" if enable else "disabled"
        console.print(f"[green]✓ Privacy noise {status}[/green]")

    if scale is not None:
        if scale < 0 or scale > 0.1:
            console.print("[yellow]Warning: scale should be between 0 and 0.1 (0-10%)[/yellow]")
        config.noise_scale = scale
        config.save()
        console.print(f"[green]✓ Noise scale set to {scale} ({scale*100:.2f}%)[/green]")

    if enable is None and scale is None:
        # Show current status
        status = "✓ Enabled" if config.noise_enabled else "✗ Disabled"
        console.print(f"Privacy noise: [cyan]{status}[/cyan]")
        console.print(f"Noise scale: [cyan]{config.noise_scale} ({config.noise_scale*100:.2f}%)[/cyan]")
        console.print("\n[dim]Usage:[/dim]")
        console.print("[dim]  decloud-trainer privacy-noise --enable[/dim]")
        console.print("[dim]  decloud-trainer privacy-noise --disable[/dim]")
        console.print("[dim]  decloud-trainer privacy-noise --scale 0.002[/dim]")


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
    table.add_row("Privacy Noise", f"{'✓ Enabled' if config.noise_enabled else '✗ Disabled'} (scale={config.noise_scale})")

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
        "noise_scale": ("noise_scale", float),
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
    
    if not config.has_lighthouse():
        console.print("\n[yellow]Lighthouse not configured![/yellow]")
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
# Profile Commands
# ═══════════════════════════════════════════════════════════════

@cli.command("create-profile")
def create_profile():
    """Create trainer profile (required for training)"""
    trainer = get_trainer()
    
    if trainer.solana.has_trainer_profile():
        console.print("[yellow]Profile already exists![/yellow]")
        profile = trainer.get_profile()
        if profile:
            console.print(f"[dim]Rating: {profile.rating/100:.2f} ★[/dim]")
        return
    
    console.print("[cyan]Creating trainer profile...[/cyan]")
    result = trainer.create_profile()
    
    if result.get("success"):
        console.print(f"[green]✓ Profile created![/green]")
        console.print(f"[dim]TX: {result['tx']}[/dim]")
        console.print(f"[dim]Initial rating: 5.00 ★[/dim]")
    else:
        console.print(f"[red]✗ {result.get('error')}[/red]")


@cli.command("profile")
def show_profile():
    """Show your trainer profile"""
    trainer = get_trainer()
    profile = trainer.get_profile()
    
    if not profile:
        console.print("[yellow]No profile found![/yellow]")
        console.print("[dim]Run: decloud-trainer create-profile[/dim]")
        return
    
    table = Table(title="🏋️ Trainer Profile")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Wallet", profile.trainer[:20] + "...")
    table.add_row("Rating", f"{profile.rating/100:.2f} ★")
    table.add_row("Total Submissions", str(profile.total_submissions))
    table.add_row("Successful", str(profile.successful_submissions))
    table.add_row("Slashed", str(profile.slashed_count))
    
    if profile.total_submissions > 0:
        success_rate = profile.successful_submissions / profile.total_submissions * 100
        table.add_row("Success Rate", f"{success_rate:.1f}%")
    
    console.print(table)


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
    
    profile = trainer.get_profile()
    my_rating = profile.rating if profile else 0
    
    table = Table(title=f"Round {round_id}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Dataset", info.dataset)
    table.add_row("Reward", f"{info.reward_amount / 1e9:.4f} SOL")
    table.add_row("Min Rating", f"{info.min_trainer_rating/100:.2f} ★")
    table.add_row("Status", info.status)
    table.add_row("Pre-validators", str(info.pre_count))
    table.add_row("Trainers", str(info.gradients_count))
    table.add_row("Model CID", info.model_cid[:40] + "...")
    
    console.print(table)
    
    # Check our submission
    submitted = trainer.solana.has_submitted_gradient(round_id)
    console.print(f"\nYour status: {'[green]✓ Submitted[/green]' if submitted else '[dim]Not submitted[/dim]'}")
    
    if not submitted and profile:
        if my_rating >= info.min_trainer_rating:
            console.print(f"[green]✓ Your rating ({my_rating/100:.2f}★) meets requirement[/green]")
        else:
            console.print(f"[red]✗ Your rating ({my_rating/100:.2f}★) below required ({info.min_trainer_rating/100:.2f}★)[/red]")


def main():
    cli()


if __name__ == "__main__":
    main()