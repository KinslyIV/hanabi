#!/usr/bin/env python3
"""
Run the Belief MCTS Bot on hanab.live.

This script starts the bot with the belief-integrated MCTS that uses
neural network policy and value heads combined with belief tracking.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_hanabi.bot.belief_mcts_bot import BeliefMCTSBot


def main():
    parser = argparse.ArgumentParser(description="Run Belief MCTS Bot on hanab.live")
    
    # Connection arguments
    parser.add_argument("--url", type=str, default="wss://hanab.live/ws",
                        help="WebSocket URL")
    parser.add_argument("--username", type=str, required=True,
                        help="Bot username")
    parser.add_argument("--password", type=str, required=True,
                        help="Bot password")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    
    # MCTS arguments
    parser.add_argument("--time-ms", type=int, default=2000,
                        help="MCTS time budget per move in milliseconds")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Fixed number of MCTS simulations")
    parser.add_argument("--c-puct", type=float, default=1.4,
                        help="PUCT exploration constant")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for action selection")
    
    # Hardware
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cpu or cuda)")
    
    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Create and run bot
    bot = BeliefMCTSBot(
        checkpoint_path=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        time_limit_ms=args.time_ms,
        c_puct=args.c_puct,
        temperature=args.temperature,
        num_simulations=args.simulations,
        device=args.device,
    )
    
    print(f"Starting Belief MCTS Bot...")
    print(f"  Time limit: {args.time_ms}ms")
    print(f"  C_PUCT: {args.c_puct}")
    print(f"  Temperature: {args.temperature}")
    
    try:
        bot.run(
            url=args.url,
            username=args.username,
            password=args.password,
        )
    except KeyboardInterrupt:
        print("\nBot stopped by user")
    except Exception as e:
        print(f"Bot error: {e}")
        raise


if __name__ == "__main__":
    main()
