"""
Pinata client for uploading model packages to IPFS
"""
import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import shutil
import requests

from config import config


PINATA_API_URL = "https://api.pinata.cloud"
PINATA_GATEWAY = "https://gateway.pinata.cloud/ipfs/"


class PinataClient:
    """
    Pinata IPFS client for uploading files
    """
    
    def __init__(self):
        self.api_url = PINATA_API_URL
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if config.pinata_jwt:
            return {
                "Authorization": f"Bearer {config.pinata_jwt}"
            }
        elif config.pinata_api_key and config.pinata_secret_key:
            return {
                "pinata_api_key": config.pinata_api_key,
                "pinata_secret_key": config.pinata_secret_key,
            }
        else:
            raise ValueError("Pinata not configured. Run 'decloud-trainer setup'")
    
    def test_authentication_sync(self) -> bool:
        """Test if authentication works"""
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{self.api_url}/data/testAuthentication",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def test_authentication(self) -> bool:
        """Async wrapper"""
        return self.test_authentication_sync()
    
    def upload_file_sync(self, file_path: Path, name: Optional[str] = None) -> Optional[str]:
        """
        Upload a single file to Pinata
        Returns IPFS CID or None on failure
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        headers = self._get_headers()
        
        with open(file_path, 'rb') as f:
            files = {
                'file': (name or file_path.name, f)
            }
            
            data = {
                'pinataMetadata': json.dumps({"name": name or file_path.name}),
                'pinataOptions': json.dumps({"cidVersion": 1})
            }
            
            response = requests.post(
                f"{self.api_url}/pinning/pinFileToIPFS",
                headers=headers,
                files=files,
                data=data,
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("IpfsHash")
            else:
                print(f"Pinata upload error: {response.text}")
                return None
    
    async def upload_file(self, file_path: Path, name: Optional[str] = None) -> Optional[str]:
        """Async wrapper"""
        return self.upload_file_sync(file_path, name)
    
    def upload_directory_sync(self, dir_path: Path, name: Optional[str] = None) -> Optional[str]:
        """
        Upload a directory to Pinata
        Returns IPFS CID (folder CID) or None on failure
        """
        if not dir_path.exists() or not dir_path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        
        headers = self._get_headers()
        folder_name = name or dir_path.name
        
        # Collect all files
        files_list = []
        file_handles = []
        
        try:
            for file_path in sorted(dir_path.rglob("*")):
                if file_path.is_file():
                    relative_path = file_path.relative_to(dir_path)
                    f = open(file_path, 'rb')
                    file_handles.append(f)
                    # Format: ('file', (path, file_handle, content_type))
                    files_list.append(
                        ('file', (f"{folder_name}/{relative_path}", f, 'application/octet-stream'))
                    )
            
            if not files_list:
                print("No files to upload")
                return None
            
            data = {
                'pinataMetadata': json.dumps({"name": folder_name}),
                'pinataOptions': json.dumps({"cidVersion": 1, "wrapWithDirectory": False})
            }
            
            response = requests.post(
                f"{self.api_url}/pinning/pinFileToIPFS",
                headers=headers,
                files=files_list,
                data=data,
                timeout=300
            )
            
            if response.status_code == 200:
                return response.json().get("IpfsHash")
            else:
                print(f"Pinata upload error: {response.text}")
                return None
        
        finally:
            # Close all file handles
            for f in file_handles:
                f.close()
    
    async def upload_directory(self, dir_path: Path, name: Optional[str] = None) -> Optional[str]:
        """Async wrapper"""
        return self.upload_directory_sync(dir_path, name)
    
    def upload_model_package_sync(
        self,
        config_dict: Dict[str, Any],
        head_weights_path: Path,
        embeddings_path: Path,
        name: Optional[str] = None
    ) -> Optional[str]:
        """
        Upload complete model package (config.json, head.safetensors, embeddings.safetensors)
        Returns IPFS CID
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "model_package"
            tmp_path.mkdir()
            
            with open(tmp_path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
            
            shutil.copy(head_weights_path, tmp_path / "head.safetensors")
            shutil.copy(embeddings_path, tmp_path / "embeddings.safetensors")
            
            return self.upload_directory_sync(tmp_path, name)
    
    async def upload_model_package(
        self,
        config_dict: Dict[str, Any],
        head_weights_path: Path,
        embeddings_path: Path,
        name: Optional[str] = None
    ) -> Optional[str]:
        """Async wrapper"""
        return self.upload_model_package_sync(config_dict, head_weights_path, embeddings_path, name)
    
    def upload_gradient_package_sync(self, gradient_dir: Path, round_id: int) -> Optional[str]:
        """Upload gradient package for a training round"""
        return self.upload_directory_sync(gradient_dir, name=f"gradient_round_{round_id}")
    
    async def upload_gradient_package(self, gradient_dir: Path, round_id: int) -> Optional[str]:
        """Async wrapper"""
        return self.upload_gradient_package_sync(gradient_dir, round_id)
    
    def get_gateway_url(self, cid: str) -> str:
        """Get gateway URL for CID"""
        return f"{PINATA_GATEWAY}{cid}"


# Global instance
pinata_client = PinataClient()