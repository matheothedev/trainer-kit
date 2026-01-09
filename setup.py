#!/usr/bin/env python3
from setuptools import setup

setup(
    name="decloud-trainer",
    version="1.0.0",
    description="Decloud Trainer - Train and earn on Solana federated learning",
    python_requires=">=3.9",
    py_modules=[
        "main",
        "config",
        "ipfs_client",
        "pinata_client",
        "solana_client",
        "websocket_listener",
        "model_loader",
        "training",
        "trainer",
    ],
    install_requires=[
        "solana>=0.30.0",
        "solders>=0.18.0",
        "torch>=2.0.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "aiohttp>=3.9.0",
        "requests>=2.31.0",
        "websockets>=12.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "base58>=2.1.0",
    ],
    entry_points={
        "console_scripts": [
            "decloud-trainer=main:main",
        ],
    },
)