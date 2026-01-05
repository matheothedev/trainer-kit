"""
DECLOUD Trainer - Main Module
==============================

Join rounds, train models, submit gradients.
"""

import os
import subprocess
import json
from datetime import datetime
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .config import Config
from .ipfs import IPFSClient
from .training import Trainer as LocalTrainer, SimpleCNN


@dataclass
class RoundInfo:
    """Round information"""
    id: int
    creator: str
    model_cid: str
    dataset: str
    reward_amount: int
    status: str
    validator: str
    trainers_count: int
    submissions_count: int
    join_deadline: int
    submit_deadline: int


class Trainer:
    """
    DECLOUD Trainer - join rounds and contribute gradients.
    
    Usage:
        trainer = Trainer()
        trainer.login("your_private_key")
        
        # List available rounds
        rounds = trainer.list_rounds()
        
        # Join a round
        trainer.join_round(round_id)
        
        # Train and submit
        trainer.train_and_submit(round_id)
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config.load()
        self.ipfs = IPFSClient(
            local_store=os.path.join(self.config.data_dir, "ipfs"),
            pinata_api_key=self.config.pinata_api_key,
            pinata_secret_key=self.config.pinata_secret_key,
        )
        self.local_trainer = LocalTrainer(
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
        )
        
        self._private_key: Optional[str] = None
        self._public_key: Optional[str] = None
    
    def login(self, private_key: str) -> Tuple[str, float]:
        """Login with private key"""
        self._private_key = private_key
        
        script = f'''
const {{ Keypair, Connection, LAMPORTS_PER_SOL }} = require("@solana/web3.js");
const bs58 = require("bs58");

const keypair = Keypair.fromSecretKey(bs58.decode("{private_key}"));
const conn = new Connection("{self.config.rpc_url}", "confirmed");

(async () => {{
    const balance = await conn.getBalance(keypair.publicKey);
    console.log(JSON.stringify({{
        publicKey: keypair.publicKey.toBase58(),
        balance: balance
    }}));
}})();
'''
        result = self._run_node(script)
        data = json.loads(result)
        
        self._public_key = data["publicKey"]
        balance_sol = data["balance"] / 1e9
        
        return self._public_key, balance_sol
    
    def _run_node(self, script: str) -> str:
        """Run Node.js script"""
        result = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Node.js error: {result.stderr}")
        
        return result.stdout.strip()
    
    def list_rounds(self, status_filter: Optional[str] = None) -> List[RoundInfo]:
        """
        List available rounds.
        
        Args:
            status_filter: Filter by status (waitingTrainers, training, etc.)
        """
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey }} = require("@solana/web3.js");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));

(async () => {{
    const provider = new anchor.AnchorProvider(conn, {{ publicKey: PublicKey.default }}, {{}});
    const program = new anchor.Program(idl, provider);
    
    const rounds = await program.account.round.all();
    
    const result = rounds.map(r => ({{
        id: r.account.id.toNumber(),
        creator: r.account.creator.toBase58(),
        modelCid: r.account.modelCid,
        dataset: Object.keys(r.account.dataset)[0],
        rewardAmount: r.account.rewardAmount.toNumber(),
        status: Object.keys(r.account.status)[0],
        validator: r.account.validator.toBase58(),
        trainersCount: r.account.trainersCount,
        submissionsCount: r.account.submissionsCount,
        joinDeadline: r.account.joinDeadline.toNumber(),
        submitDeadline: r.account.submitDeadline.toNumber(),
    }}));
    
    console.log(JSON.stringify(result));
}})();
'''
        result = self._run_node(script)
        data = json.loads(result)
        
        rounds = []
        for r in data:
            info = RoundInfo(
                id=r["id"],
                creator=r["creator"],
                model_cid=r["modelCid"],
                dataset=r["dataset"],
                reward_amount=r["rewardAmount"],
                status=r["status"],
                validator=r["validator"],
                trainers_count=r["trainersCount"],
                submissions_count=r["submissionsCount"],
                join_deadline=r["joinDeadline"],
                submit_deadline=r["submitDeadline"],
            )
            
            if status_filter is None or info.status == status_filter:
                rounds.append(info)
        
        return sorted(rounds, key=lambda r: r.id, reverse=True)
    
    def get_round(self, round_id: int) -> RoundInfo:
        """Get specific round info"""
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey }} = require("@solana/web3.js");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));

(async () => {{
    const provider = new anchor.AnchorProvider(conn, {{ publicKey: PublicKey.default }}, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const round = await program.account.round.fetch(roundPda);
    
    console.log(JSON.stringify({{
        id: round.id.toNumber(),
        creator: round.creator.toBase58(),
        modelCid: round.modelCid,
        dataset: Object.keys(round.dataset)[0],
        rewardAmount: round.rewardAmount.toNumber(),
        status: Object.keys(round.status)[0],
        validator: round.validator.toBase58(),
        trainersCount: round.trainersCount,
        submissionsCount: round.submissionsCount,
        joinDeadline: round.joinDeadline.toNumber(),
        submitDeadline: round.submitDeadline.toNumber(),
    }}));
}})();
'''
        result = self._run_node(script)
        r = json.loads(result)
        
        return RoundInfo(
            id=r["id"],
            creator=r["creator"],
            model_cid=r["modelCid"],
            dataset=r["dataset"],
            reward_amount=r["rewardAmount"],
            status=r["status"],
            validator=r["validator"],
            trainers_count=r["trainersCount"],
            submissions_count=r["submissionsCount"],
            join_deadline=r["joinDeadline"],
            submit_deadline=r["submitDeadline"],
        )
    
    def join_round(self, round_id: int) -> str:
        """Join a training round"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair, SystemProgram }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const trainer = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(trainer);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [submissionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("submission"), roundId.toArrayLike(Buffer, "le", 8), trainer.publicKey.toBuffer()],
        programId
    );
    
    const tx = await program.methods
        .joinRound(roundId)
        .accounts({{
            round: roundPda,
            submission: submissionPda,
            trainer: trainer.publicKey,
            systemProgram: SystemProgram.programId,
        }})
        .signers([trainer])
        .rpc();
    
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def submit_gradients(self, round_id: int, gradient_cid: str) -> str:
        """Submit gradients to blockchain"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const trainer = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

(async () => {{
    const wallet = new anchor.Wallet(trainer);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [submissionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("submission"), roundId.toArrayLike(Buffer, "le", 8), trainer.publicKey.toBuffer()],
        programId
    );
    
    const tx = await program.methods
        .submitGradients(roundId, "{gradient_cid}")
        .accounts({{
            round: roundPda,
            submission: submissionPda,
            trainer: trainer.publicKey,
        }})
        .signers([trainer])
        .rpc();
    
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def claim_reward(self, round_id: int) -> str:
        """Claim reward after round completion"""
        if not self._private_key:
            raise RuntimeError("Not logged in")
        
        # Get referrer from config, default to TREASURY if not set
        referrer_js = "TREASURY"
        if self.config.referrer:
            referrer_js = f'new PublicKey("{self.config.referrer}")'
        
        idl_path = self.config.idl_path
        script = f'''
const anchor = require("@coral-xyz/anchor");
const {{ Connection, PublicKey, Keypair, SystemProgram }} = require("@solana/web3.js");
const bs58 = require("bs58");
const fs = require("fs");

const conn = new Connection("{self.config.rpc_url}", "confirmed");
const programId = new PublicKey("{self.config.program_id}");
const idl = JSON.parse(fs.readFileSync("{idl_path}"));
const trainer = Keypair.fromSecretKey(bs58.decode("{self._private_key}"));

const TREASURY = new PublicKey("FzuCxi65QyFXAGbHcXB28RXqyBZSZ5KXLQxeofx1P9K2");

(async () => {{
    const wallet = new anchor.Wallet(trainer);
    const provider = new anchor.AnchorProvider(conn, wallet, {{}});
    const program = new anchor.Program(idl, provider);
    
    const roundId = new anchor.BN({round_id});
    
    const [roundPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("round"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    const [submissionPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("submission"), roundId.toArrayLike(Buffer, "le", 8), trainer.publicKey.toBuffer()],
        programId
    );
    const [vaultPda] = PublicKey.findProgramAddressSync(
        [Buffer.from("vault"), roundId.toArrayLike(Buffer, "le", 8)],
        programId
    );
    
    const referrer = {referrer_js};
    
    const tx = await program.methods
        .claimReward(roundId)
        .accounts({{
            round: roundPda,
            submission: submissionPda,
            vault: vaultPda,
            treasury: TREASURY,
            referrer: referrer,
            trainer: trainer.publicKey,
            systemProgram: SystemProgram.programId,
        }})
        .signers([trainer])
        .rpc();
    
    console.log(tx);
}})();
'''
        return self._run_node(script)
    
    def train_and_submit(
        self,
        round_id: int,
        epochs: int = 1,
        max_batches: Optional[int] = None,
    ) -> str:
        """
        Full training flow:
        1. Download model from IPFS
        2. Train on local data
        3. Upload gradients to IPFS
        4. Submit to blockchain
        """
        print(f"\n📥 Step 1: Fetching round info...")
        round_info = self.get_round(round_id)
        
        print(f"   Round #{round_id}: {round_info.dataset}")
        print(f"   Model CID: {round_info.model_cid}")
        print(f"   Reward: {round_info.reward_amount / 1e9:.4f} SOL")
        
        # Download model
        print(f"\n📥 Step 2: Downloading model from IPFS...")
        model_path = os.path.join(self.config.data_dir, "models", f"model_{round_id}.pt")
        
        if not self.ipfs.download(round_info.model_cid, model_path):
            raise RuntimeError(f"Failed to download model: {round_info.model_cid}")
        
        print(f"   ✓ Model downloaded: {os.path.getsize(model_path) / 1024:.1f} KB")
        
        # Load model
        print(f"\n🔧 Step 3: Loading model...")
        model = self.local_trainer.load_model(model_path, round_info.dataset)
        print(f"   ✓ Model loaded on {self.local_trainer.device}")
        
        # Train
        print(f"\n🏋️ Step 4: Training on {round_info.dataset}...")
        model, gradients = self.local_trainer.train(
            model=model,
            dataset_name=round_info.dataset,
            epochs=epochs,
            data_dir=os.path.join(self.config.data_dir, "datasets"),
            max_batches=max_batches,
        )
        
        # Save gradients
        print(f"\n💾 Step 5: Saving gradients...")
        gradients_path = os.path.join(
            self.config.data_dir, 
            "gradients", 
            f"gradients_{round_id}_{self._public_key[:8]}.pt"
        )
        self.local_trainer.save_gradients(gradients, gradients_path)
        
        # Upload to IPFS
        print(f"\n📤 Step 6: Uploading gradients to IPFS...")
        upload_result = self.ipfs.upload(gradients_path)
        print(f"   ✓ Uploaded! CID: {upload_result.cid}")
        
        # Submit to blockchain
        print(f"\n⛓️ Step 7: Submitting to blockchain...")
        tx = self.submit_gradients(round_id, upload_result.cid)
        print(f"   ✓ Submitted! TX: {tx[:40]}...")
        
        print(f"\n{'═' * 60}")
        print(f"  ✅ Training complete!")
        print(f"  Gradient CID: {upload_result.cid}")
        print(f"  TX: {tx}")
        print(f"{'═' * 60}\n")
        
        return tx
    
    def auto_train(
        self,
        dataset_filter: Optional[str] = None,
        min_reward: float = 0,
        epochs: int = 1,
    ):
        """
        DEPRECATED: Use start_daemon() instead.
        """
        print("⚠️  auto_train() is deprecated. Use start_daemon() instead.")
        self.start_daemon()
    
    def start_daemon(self):
        """
        Start the training daemon.
        
        Monitors for new rounds with validators, automatically joins,
        trains, and submits gradients.
        """
        import time
        from datetime import datetime
        
        config = self.config
        
        print("\n" + "═" * 60)
        print("  🤖 DECLOUD Trainer Daemon")
        print("═" * 60)
        print(f"  Wallet: {self._public_key[:20]}...")
        print()
        config.print_settings()
        print("═" * 60)
        print("\n  👀 Listening for rounds with validators... (Ctrl+C to stop)\n")
        
        # Track our active rounds
        active_rounds: set = set()
        joined_rounds: set = set()  # Rounds we've joined (even if completed)
        pending_training: set = set()  # Joined, waiting for training phase
        
        while True:
            try:
                now = datetime.now().strftime("%H:%M:%S")
                
                # Fetch all rounds
                all_rounds = self.list_rounds()
                
                # Process each round
                for r in all_rounds:
                    # Skip if already processed
                    if r.id in joined_rounds:
                        # Check if we need to train
                        if r.id in pending_training and r.status == "training":
                            pending_training.discard(r.id)
                            active_rounds.add(r.id)
                            
                            if config.auto_train:
                                print(f"\n[{now}] 🏋️ Round #{r.id} started training!")
                                try:
                                    self.train_and_submit(
                                        round_id=r.id,
                                        epochs=config.epochs,
                                        max_batches=config.max_batches if config.max_batches > 0 else None,
                                    )
                                    active_rounds.discard(r.id)
                                except Exception as e:
                                    print(f"   ❌ Training failed: {e}")
                        
                        # Check if completed (for auto-claim)
                        if r.status == "completed" and config.auto_claim:
                            if r.id in active_rounds or r.id in pending_training:
                                print(f"\n[{now}] 💰 Round #{r.id} completed! Claiming reward...")
                                try:
                                    tx = self.claim_reward(r.id)
                                    print(f"   ✅ Claimed! TX: {tx[:30]}...")
                                except Exception as e:
                                    print(f"   ⚠️ Claim failed: {e}")
                                
                                active_rounds.discard(r.id)
                                pending_training.discard(r.id)
                        
                        continue
                    
                    # === NEW ROUND DETECTION ===
                    
                    # Only interested in waitingTrainers (has validator)
                    if r.status != "waitingTrainers":
                        continue
                    
                    # Check if matches our filters
                    if not config.matches_round(r.dataset, r.reward_amount):
                        continue
                    
                    # Check concurrent limit
                    current_active = len(active_rounds) + len(pending_training)
                    if current_active >= config.max_concurrent_rounds:
                        continue
                    
                    # === JOIN THE ROUND ===
                    reward_sol = r.reward_amount / 1e9
                    
                    print(f"\n[{now}] 🎯 NEW ROUND #{r.id}")
                    print(f"         Dataset: {r.dataset}")
                    print(f"         Reward:  {reward_sol:.4f} SOL")
                    print(f"         Trainers: {r.trainers_count}")
                    
                    try:
                        tx = self.join_round(r.id)
                        print(f"   ✅ Joined! TX: {tx[:30]}...")
                        
                        joined_rounds.add(r.id)
                        pending_training.add(r.id)
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "already joined" in error_msg.lower() or "already exists" in error_msg.lower():
                            print(f"   ℹ️ Already joined this round")
                            joined_rounds.add(r.id)
                        else:
                            print(f"   ❌ Join failed: {e}")
                
                # Status line
                status_parts = []
                if pending_training:
                    status_parts.append(f"⏳ waiting: {len(pending_training)}")
                if active_rounds:
                    status_parts.append(f"🏋️ training: {len(active_rounds)}")
                
                status = " | ".join(status_parts) if status_parts else "idle"
                print(f"\r  [{now}] {status}   ", end="", flush=True)
                
                time.sleep(config.poll_interval)
                
            except KeyboardInterrupt:
                print("\n\n👋 Daemon stopped")
                break
            except Exception as e:
                print(f"\n   ⚠️ Error: {e}")
                time.sleep(config.poll_interval)