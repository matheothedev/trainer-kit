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
        """Fetch complete model package"""
        required_files = ["config.json", "head.safetensors"]
        optional_files = ["embeddings.safetensors"]
        
        result = {}
        
        for filename in required_files:
            data = await self.fetch_file(cid, filename)
            if data is None:
                return None
            result[filename] = data
        
        for filename in optional_files:
            data = await self.fetch_file(cid, filename)
            if data:
                result[filename] = data
        
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
            with open(cache_path / filename, "wb") as f:
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
