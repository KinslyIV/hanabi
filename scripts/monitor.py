#!/usr/bin/env python3
"""
Monitor distributed training status.
Can connect to the GPU server to get status, or read local state.

Usage:
    python scripts/monitor.py --gpu-host <laptop-ip> [--gpu-port PORT] [--auth-token TOKEN]
    python scripts/monitor.py --local  # Read local training_state.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_hanabi.training.distributed.gpu_client import GPUClient, GPUClientConfig


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def print_local_status(checkpoint_dir: str = "checkpoints") -> None:
    """Print status from local training state file."""
    state_file = Path(checkpoint_dir) / "training_state.json"
    
    print("\n" + "="*50)
    print("Local Training State")
    print("="*50)
    
    if not state_file.exists():
        print("No training state found.")
        print(f"Expected file: {state_file}")
        return
    
    with open(state_file, 'r') as f:
        state = json.load(f)
    
    print(f"Iteration:         {state.get('iteration', 0)}")
    print(f"Total Games:       {state.get('total_games', 0)}")
    print(f"Total Train Steps: {state.get('total_train_steps', 0)}")
    print(f"Best Avg Score:    {state.get('best_avg_score', 0):.2f}")
    
    sim_config = state.get('simulation_config', {})
    if sim_config:
        print(f"\nSimulation Config:")
        print(f"  Temperature: {sim_config.get('temperature', 'N/A')}")
        print(f"  Epsilon:     {sim_config.get('epsilon', 'N/A')}")
    
    # Check for checkpoints
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir_path.glob("checkpoint_*.pt"))
    if checkpoints:
        print(f"\nCheckpoints ({len(checkpoints)} found):")
        for cp in sorted(checkpoints)[-5:]:  # Show last 5
            stat = cp.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size_mb = stat.st_size / (1024 * 1024)
            print(f"  {cp.name}: {size_mb:.1f}MB, {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


async def print_gpu_status(
    host: str,
    port: int = 5555,
    auth_token: str | None = None,
) -> None:
    """Connect to GPU server and print status."""
    print("\n" + "="*50)
    print(f"GPU Server Status ({host}:{port})")
    print("="*50)
    
    config = GPUClientConfig(
        host=host,
        port=port,
        auth_token=auth_token,
        initial_retry_delay=1.0,
        max_retry_delay=5.0,
        max_connect_attempts=3,
        connection_timeout=10.0,
    )
    
    client = GPUClient(config)
    
    try:
        print("Connecting...")
        await client.connect()
        
        if not client.is_connected:
            print("Failed to connect to GPU server")
            return
        
        # Get ping
        ping_result = await client.ping(timeout=5.0)
        print(f"Status: Connected ✓")
        print(f"Uptime: {format_duration(ping_result.get('uptime', 0))}")
        print(f"Global Step: {ping_result.get('global_step', 0)}")
        print(f"Requests Handled: {ping_result.get('requests_handled', 0)}")
        
        # Get detailed stats
        stats = await client.get_stats()
        print(f"\nDetailed Stats:")
        print(f"  Device: {stats.get('device', 'N/A')}")
        print(f"  Total Training Time: {format_duration(stats.get('total_training_time', 0))}")
        
        # Connection stats
        cs = client.stats
        print(f"\nConnection Stats:")
        print(f"  Connect Attempts: {cs.connect_attempts}")
        print(f"  Successful: {cs.successful_connects}")
        print(f"  Failed: {cs.failed_connects}")
        print(f"  Requests Sent: {cs.requests_sent}")
        print(f"  Requests Failed: {cs.requests_failed}")
        print(f"  Bytes Sent: {cs.bytes_sent / 1024:.1f} KB")
        print(f"  Bytes Received: {cs.bytes_received / 1024:.1f} KB")
        if cs.requests_sent > 0:
            avg_latency = cs.total_latency / cs.requests_sent
            print(f"  Avg Latency: {avg_latency*1000:.1f} ms")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        await client.disconnect()


async def continuous_monitor(
    host: str,
    port: int = 5555,
    auth_token: str | None = None,
    interval: float = 5.0,
) -> None:
    """Continuously monitor the GPU server."""
    import os
    
    config = GPUClientConfig(
        host=host,
        port=port,
        auth_token=auth_token,
        initial_retry_delay=1.0,
        max_retry_delay=30.0,
        connection_timeout=10.0,
    )
    
    client = GPUClient(config)
    last_step = 0
    last_time = asyncio.get_event_loop().time()
    
    try:
        await client.connect()
        
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            print("="*60)
            print(f"Hanabi Distributed Training Monitor - {datetime.now().strftime('%H:%M:%S')}")
            print("="*60)
            
            if client.is_connected:
                try:
                    stats = await client.get_stats()
                    current_step = stats.get('global_step', 0)
                    current_time = asyncio.get_event_loop().time()
                    
                    # Calculate steps per second
                    time_diff = current_time - last_time
                    step_diff = current_step - last_step
                    steps_per_sec = step_diff / time_diff if time_diff > 0 else 0
                    
                    print(f"Status: ✓ Connected to {host}:{port}")
                    print(f"Device: {stats.get('device', 'N/A')}")
                    print(f"")
                    print(f"Global Step: {current_step:,}")
                    print(f"Steps/sec: {steps_per_sec:.2f}")
                    print(f"Uptime: {format_duration(stats.get('uptime', 0))}")
                    print(f"Training Time: {format_duration(stats.get('total_training_time', 0))}")
                    print(f"Requests: {stats.get('requests_handled', 0):,}")
                    
                    last_step = current_step
                    last_time = current_time
                    
                except Exception as e:
                    print(f"Status: ⚠ Error fetching stats: {e}")
            else:
                print(f"Status: ✗ Disconnected from {host}:{port}")
                print("Reconnecting...")
            
            print(f"\nPress Ctrl+C to exit")
            await asyncio.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        await client.disconnect()


def main():
    parser = argparse.ArgumentParser(description='Monitor distributed training')
    parser.add_argument('--gpu-host', type=str, help='GPU server hostname/IP')
    parser.add_argument('--gpu-port', type=int, default=5555, help='GPU server port')
    parser.add_argument('--auth-token', type=str, help='Authentication token')
    parser.add_argument('--local', action='store_true', help='Show local state only')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--watch', action='store_true', help='Continuous monitoring')
    parser.add_argument('--interval', type=float, default=5.0, help='Watch interval in seconds')
    
    args = parser.parse_args()
    
    if args.local:
        print_local_status(args.checkpoint_dir)
    elif args.gpu_host:
        if args.watch:
            asyncio.run(continuous_monitor(
                args.gpu_host,
                args.gpu_port,
                args.auth_token,
                args.interval,
            ))
        else:
            asyncio.run(print_gpu_status(
                args.gpu_host,
                args.gpu_port,
                args.auth_token,
            ))
            print_local_status(args.checkpoint_dir)
    else:
        print("Please specify --gpu-host <ip> or --local")
        print("Use --watch for continuous monitoring")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
