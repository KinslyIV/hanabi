"""Training module for Hanabi self-play."""

from rl_hanabi.training.game_simulator import (
    GameSimulator,
    GameConfig,
    GameResult,
    Transition,
    sample_game_config,
)
from rl_hanabi.training.mcts_simulator import (
    MCTSGameSimulator,
    run_mcts_self_play,
)
from rl_hanabi.training.data_collection import (
    ReplayBuffer,
    HanabiDataset,
    StreamingHanabiDataset,
    BatchedTransition,
    create_dataloader,
)
from rl_hanabi.training.trainer import (
    HanabiTrainer,
    log_game_metrics,
    init_wandb,
)

# Distributed training imports
from rl_hanabi.training.distributed import (
    GPUClient,
    GPUClientConfig,
    GPUServer,
    GPUTrainer,
)

__all__ = [
    # Game simulation
    "GameSimulator",
    "GameConfig",
    "GameResult",
    "Transition",
    "sample_game_config",
    # MCTS simulation
    "MCTSGameSimulator",
    "run_mcts_self_play",
    # Data collection
    "ReplayBuffer",
    "HanabiDataset",
    "StreamingHanabiDataset",
    "BatchedTransition",
    "create_dataloader",
    # Training
    "HanabiTrainer",
    "log_game_metrics",
    "init_wandb",
    # Distributed training
    "GPUClient",
    "GPUClientConfig",
    "GPUServer",
    "GPUTrainer",
]
