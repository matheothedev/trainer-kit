"""
IPFS client for fetching model packages
"""
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import Dict, Optional, List

from config import IPFS_GATEWAYS, MODELS_CACHE


class IPFSClient:
    """
    IPFS client with gateway fallback for downloading
    """
    
    def __init__(self, gateways: List[str] = IPFS_GATEWAYS, timeout: int = 120):
        self.gateways = gateways
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.cache_dir = MODELS_CACHE
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def fetch_file(self, cid: str, filename: str = "") -> Optional[bytes]:
        """Fetch a single file from IPFS"""
        path = f"{cid}/{filename}" if filename else cid
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            for gateway in self.gateways:
                url = f"{gateway}{path}"
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            return await response.read()
                except Exception:
                    continue
        
        return None
    
    async def fetch_json(self, cid: str, filename: str = "config.json") -> Optional[Dict]:
        """Fetch and parse JSON file"""
        data = await self.fetch_file(cid, filename)
        if data:
            return json.loads(data.decode("utf-8"))
        return None
    
    async def fetch_model_package(self, cid: str) -> Optional[Dict[str, bytes]]:
        """
        Fetch complete model package.
        Auto-detects format:
          - Classification: config.json + head.safetensors + embeddings.safetensors (optional)
          - LLM Full: config.json + model.safetensors + tokenizer/*
        """
        result = {}

        # Always fetch config first
        config_data = await self.fetch_file(cid, "config.json")
        if config_data is None:
            print("Failed to fetch config.json")
            return None
        result["config.json"] = config_data

        # Check package type
        config = json.loads(config_data.decode("utf-8"))
        pkg_type = config.get("type", "classification")

        if pkg_type == "llm_full":
            # LLM package: model.safetensors + tokenizer/
            model_data = await self.fetch_file(cid, "model.safetensors")
            if model_data is None:
                print("Failed to fetch model.safetensors")
                return None
            result["model.safetensors"] = model_data

            # Fetch tokenizer files
            tokenizer_files = [
                "tokenizer/tokenizer_config.json",
                "tokenizer/vocab.json",
                "tokenizer/merges.txt",
                "tokenizer/special_tokens_map.json",
                "tokenizer/tokenizer.json",
            ]
            for tf in tokenizer_files:
                tf_data = await self.fetch_file(cid, tf)
                if tf_data:
                    result[tf] = tf_data
        else:
            # Classification package: head.safetensors + embeddings (optional)
            head_data = await self.fetch_file(cid, "head.safetensors")
            if head_data is None:
                print("Failed to fetch head.safetensors")
                return None
            result["head.safetensors"] = head_data

            # embeddings.safetensors is optional
            emb_data = await self.fetch_file(cid, "embeddings.safetensors")
            if emb_data:
                result["embeddings.safetensors"] = emb_data

        return result

    async def download_model_package(self, cid: str) -> Optional[Path]:
        """Download and cache model package"""
        cache_path = self.cache_dir / cid

        if cache_path.exists() and (cache_path / "config.json").exists():
            return cache_path

        package = await self.fetch_model_package(cid)
        if package is None:
            return None

        cache_path.mkdir(parents=True, exist_ok=True)
        for filename, data in package.items():
            file_path = cache_path / filename
            # Create subdirectories if needed (e.g., tokenizer/)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(data)

        return cache_path
    
    def download_model_package_sync(self, cid: str) -> Optional[Path]:
        """Sync wrapper"""
        return asyncio.run(self.download_model_package(cid))
    
    def is_cached(self, cid: str) -> bool:
        """Check if model is cached"""
        cache_path = self.cache_dir / cid
        return cache_path.exists() and (cache_path / "config.json").exists()


# Global instance
ipfs_client = IPFSClient()
