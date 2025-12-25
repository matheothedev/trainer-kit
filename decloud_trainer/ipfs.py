"""
IPFS Module for Trainer
=======================

Download models, upload gradients.
"""

import os
import hashlib
import requests
from typing import Optional
from dataclasses import dataclass


GATEWAYS = [
    "https://ipfs.io/ipfs/",
    "https://cloudflare-ipfs.com/ipfs/",
    "https://gateway.pinata.cloud/ipfs/",
    "https://dweb.link/ipfs/",
    "https://w3s.link/ipfs/",
]


@dataclass
class UploadResult:
    cid: str
    size: int
    provider: str


class IPFSClient:
    """IPFS client for downloading models and uploading gradients"""
    
    def __init__(
        self,
        local_store: str = "./data/ipfs_local",
        pinata_api_key: Optional[str] = None,
        pinata_secret_key: Optional[str] = None,
    ):
        self.local_store = os.path.expanduser(local_store)
        os.makedirs(self.local_store, exist_ok=True)
        
        self.pinata_api_key = pinata_api_key or os.getenv("PINATA_API_KEY")
        self.pinata_secret_key = pinata_secret_key or os.getenv("PINATA_SECRET_KEY")
    
    def download(self, cid: str, output_path: str, timeout: int = 120) -> bool:
        """
        Download file from IPFS.
        
        Checks local store first, then tries public gateways.
        """
        # Check local store first
        local_path = os.path.join(self.local_store, cid)
        if os.path.exists(local_path):
            with open(local_path, 'rb') as src:
                with open(output_path, 'wb') as dst:
                    dst.write(src.read())
            return True
        
        # Try gateways
        for gateway in GATEWAYS:
            try:
                url = f"{gateway}{cid}"
                response = requests.get(url, timeout=timeout)
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Cache locally
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    
                    return True
            except Exception:
                continue
        
        return False
    
    def upload(self, file_path: str) -> UploadResult:
        """
        Upload file to IPFS.
        
        Uses Pinata if configured, otherwise local store.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        
        # Try Pinata first
        if self.pinata_api_key and self.pinata_secret_key:
            try:
                cid = self._upload_pinata(file_path)
                return UploadResult(cid=cid, size=file_size, provider="pinata")
            except Exception as e:
                print(f"   ⚠️ Pinata failed: {e}, using local store")
        
        # Fallback to local store (for testing)
        cid = self._upload_local(file_path)
        return UploadResult(cid=cid, size=file_size, provider="local")
    
    def _upload_pinata(self, file_path: str) -> str:
        """Upload to Pinata"""
        url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
        
        filename = os.path.basename(file_path)
        
        with open(file_path, 'rb') as f:
            response = requests.post(
                url,
                files={'file': (filename, f)},
                headers={
                    'pinata_api_key': self.pinata_api_key,
                    'pinata_secret_api_key': self.pinata_secret_key
                },
                timeout=300
            )
        
        if response.status_code != 200:
            raise RuntimeError(f"Pinata error: {response.text}")
        
        return response.json()['IpfsHash']
    
    def _upload_local(self, file_path: str) -> str:
        """Upload to local store (for testing)"""
        with open(file_path, "rb") as f:
            content = f.read()
        
        # Generate fake CID from hash
        hash_hex = hashlib.sha256(content).hexdigest()
        cid = "Qm" + hash_hex[:44]
        
        # Save to local store
        local_path = os.path.join(self.local_store, cid)
        with open(local_path, "wb") as f:
            f.write(content)
        
        return cid
    
    def is_valid_cid(self, cid: str) -> bool:
        """Check if CID is valid format"""
        if len(cid) < 46 or len(cid) > 64:
            return False
        return cid.startswith('Qm') or cid.startswith('ba')


def setup_pinata_interactive():
    """Interactive Pinata setup"""
    print("\n   📌 Pinata Setup")
    print("   Get free API keys at: https://app.pinata.cloud/keys")
    print()
    
    api_key = input("   API Key: ").strip()
    secret_key = input("   Secret Key: ").strip()
    
    if not api_key or not secret_key:
        return "", ""
    
    # Test connection
    try:
        response = requests.get(
            "https://api.pinata.cloud/data/testAuthentication",
            headers={
                'pinata_api_key': api_key,
                'pinata_secret_api_key': secret_key
            }
        )
        if response.status_code == 200:
            print("   ✓ Pinata connected!")
            return api_key, secret_key
        else:
            print(f"   ❌ Authentication failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
    
    return "", ""
