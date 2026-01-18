"""
Distributed training module for Hanabi.
Enables training with GPU on a remote machine while running game simulation locally.
"""

from rl_hanabi.training.distributed.gpu_client import GPUClient, GPUClientConfig
from rl_hanabi.training.distributed.gpu_server import GPUServer, GPUTrainer

__all__ = [
    "GPUClient",
    "GPUClientConfig", 
    "GPUServer",
    "GPUTrainer",
]
