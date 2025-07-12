# ddp_setup.py
import os
import socket
import torch.distributed as dist
import torch

def find_free_port() -> int:
    """Find a free port on localhost for distributed training."""
    print("Finding free port...")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))  # Let the OS assign a free port
        return s.getsockname()[1]

def ddp_setup(rank: int, world_size: int) -> None:
    print(f"[GPU {rank}] DDP setup starting...")

    """Set up DistributedDataParallel environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", str(find_free_port()))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"[DDP] Rank {rank}/{world_size} initialized with MASTER_ADDR={os.environ['MASTER_ADDR']}, PORT={os.environ['MASTER_PORT']}")

    torch.cuda.set_device(rank)

def ddp_cleanup() -> None:
    """Clean up the DistributedDataParallel environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("[DDP] Process group destroyed.")
