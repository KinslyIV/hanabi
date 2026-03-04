# Hanabi Token-Based Training

This refactor trims the training loop to a token-based, AlphaZero-style pipeline using the new `HLETokenizer` and `ActionDecoder`. The training path no longer depends on belief state or MCTS and keeps distributed self-play (CPU) separate from GPU training.

## Key decisions
- **State encoding:** `HLETokenizer` builds the token sequence as `[life, info, previous_action] + fireworks + hands + discard`. Discard tokens are truncated from the front if needed to fit `context_size`.
- **Action selection:** `ActionDecoder` outputs per-card logits for `[discard, play, clue-color, clue-rank]`. Clue logits are aggregated by **max** over target-player cards that share the same color/rank.
- **Reward:** only the **final normalized score** is used as the policy gradient reward.
- **Previous action token:** the previous move token is passed as the action token; the first move uses the pad token.
- **Removed from training:** belief state tracking and MCTS data collection.

## Local training
```bash
python rl_hanabi/training/train.py \
  --checkpoint-dir checkpoints \
  --num-iterations 50 \
  --games-per-iteration 100 \
  --train-steps-per-iteration 500
```

## Distributed training
Start the GPU server on the GPU machine:
```bash
python rl_hanabi/training/distributed/gpu_server.py \
  --host 0.0.0.0 --port 5555 \
  --checkpoint-dir checkpoints
```

Run the coordinator (self-play) on the CPU machine:
```bash
python rl_hanabi/training/distributed/coordinator.py \
  --gpu-host <gpu-host> --gpu-port 5555 \
  --checkpoint-dir checkpoints
```

Checkpoints are written to `checkpoint_latest.pt` and `checkpoint_iter_*.pt` so training can be stopped and resumed safely.
