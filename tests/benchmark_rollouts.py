"""
Benchmark script to test rollout performance for different parallelization strategies.

Tests:
1. Sequential rollouts
2. Thread-parallel rollouts (ThreadPoolExecutor)
3. Process-parallel rollouts (ProcessPoolExecutor)

At different game states:
- Early game (just started)
- Mid game (some cards played)
- Late game (near end)

Note: The HLE (Hanabi Learning Environment) uses CFFI objects that cannot be pickled
for process parallelism. For process-parallel rollouts, we create fresh game states
in each worker process.
"""

import time
import numpy as np
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from hanabi_learning_environment import pyhanabi
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.mcts.convention_rollout import ConventionRolloutPolicy


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    strategy: str
    game_stage: str
    num_rollouts: int
    total_time_ms: float
    avg_time_per_rollout_ms: float
    avg_score: float
    num_workers: int = 1


# Game configuration for serialization
DEFAULT_GAME_PARAMS = {
    "players": 2,
    "colors": 5,
    "ranks": 5,
    "hand_size": 5,
    "max_information_tokens": 8,
    "max_life_tokens": 3,
    "random_start_player": False,
}


def create_game_state(seed: int = -1) -> HLEGameState:
    """Create a fresh game state. seed=-1 means random."""
    params = {
        "players": 2,
        "colors": 5,
        "ranks": 5,
        "hand_size": 5,
        "max_information_tokens": 8,
        "max_life_tokens": 3,
        "seed": seed,
        "random_start_player": False,
        "observation_type": pyhanabi.AgentObservationType.CARD_KNOWLEDGE,
    }
    game = pyhanabi.HanabiGame(params)
    state = game.new_initial_state()
    
    # Deal cards
    while state.cur_player() == pyhanabi.CHANCE_PLAYER_ID:
        state.deal_random_card()
    
    return HLEGameState(game=game, state=state)


def advance_game_to_stage(state: HLEGameState, target_moves: int) -> HLEGameState:
    """
    Advance game state by playing convention-based moves.
    
    Args:
        state: Initial game state
        target_moves: Number of moves to make
        
    Returns:
        Advanced game state
    """
    current_state = state.copy()
    moves_made = 0
    
    policy = ConventionRolloutPolicy()
    
    while moves_made < target_moves and not current_state.is_terminal():
        move = policy.select_move(current_state)
        current_state.apply_move(move)
        moves_made += 1
    
    return current_state


def run_sequential_rollouts(state: HLEGameState, num_rollouts: int, 
                            max_depth: int = 50) -> Tuple[float, float]:
    """
    Run rollouts sequentially.
    
    Returns:
        Tuple of (total_time_seconds, avg_score)
    """
    policy = ConventionRolloutPolicy()
    scores = []
    
    start = time.perf_counter()
    for _ in range(num_rollouts):
        score, _ = policy.rollout(state, max_depth)
        scores.append(score)
    end = time.perf_counter()
    
    return (end - start), float(np.mean(scores))


def _thread_worker_rollout(args) -> Tuple[float, np.ndarray]:
    """Worker function for thread-parallel rollouts."""
    state_copy, max_depth, play_weight, clue_weight, save_weight, discard_weight = args
    
    policy = ConventionRolloutPolicy(
        play_weight=play_weight,
        clue_weight=clue_weight, 
        save_weight=save_weight,
        discard_weight=discard_weight
    )
    
    return policy.rollout(state_copy, max_depth)


def run_thread_parallel_rollouts(state: HLEGameState, num_rollouts: int,
                                  num_workers: int, max_depth: int = 50) -> Tuple[float, float]:
    """
    Run rollouts using ThreadPoolExecutor.
    
    Returns:
        Tuple of (total_time_seconds, avg_score)
    """
    args_list = [
        (state.copy(), max_depth, 10.0, 5.0, 8.0, 1.0)
        for _ in range(num_rollouts)
    ]
    
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_thread_worker_rollout, args_list))
    end = time.perf_counter()
    
    scores = [r[0] for r in results]
    return (end - start), float(np.mean(scores))


def _process_worker_rollout(args) -> float:
    """
    Worker function for process-parallel rollouts.
    Creates a fresh game state and advances it, since HLE states can't be pickled.
    """
    game_params, target_moves, max_depth, play_weight, clue_weight, save_weight, discard_weight = args
    
    # Create fresh state in this process
    state = create_game_state(seed=-1)  # Random seed
    
    # Advance to target game stage
    if target_moves > 0:
        state = advance_game_to_stage(state, target_moves)
    
    if state.is_terminal():
        return state.score() / state.max_score()
    
    policy = ConventionRolloutPolicy(
        play_weight=play_weight,
        clue_weight=clue_weight, 
        save_weight=save_weight,
        discard_weight=discard_weight
    )
    
    score, _ = policy.rollout(state, max_depth)
    return score


def run_process_parallel_rollouts(target_moves: int,
                                   num_rollouts: int,
                                   num_workers: int, 
                                   max_depth: int = 50) -> Tuple[float, float]:
    """
    Run rollouts using ProcessPoolExecutor.
    Each worker creates its own fresh game state since HLE states can't be pickled.
    
    Returns:
        Tuple of (total_time_seconds, avg_score)
    """
    args_list = [
        (DEFAULT_GAME_PARAMS, target_moves, max_depth, 10.0, 5.0, 8.0, 1.0)
        for _ in range(num_rollouts)
    ]
    
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_process_worker_rollout, args_list))
    end = time.perf_counter()
    
    return (end - start), float(np.mean(results))


def benchmark_all_strategies(state: HLEGameState, 
                              target_moves: int,
                              game_stage: str,
                              num_rollouts: int = 100,
                              max_depth: int = 50) -> List[BenchmarkResult]:
    """
    Benchmark all parallelization strategies for a given game state.
    
    Args:
        state: The game state to benchmark (for sequential and thread-parallel)
        target_moves: Number of moves to reach this state (for process-parallel)
        game_stage: Name of the game stage (for reporting)
        num_rollouts: Number of rollouts to perform
        max_depth: Maximum depth per rollout
        
    Returns:
        List of BenchmarkResult objects
    """
    results = []
    num_cpus = multiprocessing.cpu_count()
    
    print(f"\n{'='*60}")
    print(f"Benchmarking {game_stage} (Score: {state.score()}, "
          f"Deck: {state.deck_size()}, Terminal: {state.is_terminal()})")
    print(f"{'='*60}")
    
    # Sequential
    print(f"\n  Running {num_rollouts} sequential rollouts...")
    seq_time, seq_score = run_sequential_rollouts(state, num_rollouts, max_depth)
    results.append(BenchmarkResult(
        strategy="Sequential",
        game_stage=game_stage,
        num_rollouts=num_rollouts,
        total_time_ms=seq_time * 1000,
        avg_time_per_rollout_ms=(seq_time * 1000) / num_rollouts,
        avg_score=seq_score,
        num_workers=1
    ))
    print(f"    Total: {seq_time*1000:.2f}ms, Per rollout: {(seq_time*1000)/num_rollouts:.2f}ms, Avg score: {seq_score:.3f}")
    
    # Thread parallel with different worker counts
    for num_workers in [2, 4, num_cpus]:
        if num_workers > num_cpus:
            continue
        print(f"\n  Running {num_rollouts} thread-parallel rollouts ({num_workers} workers)...")
        thread_time, thread_score = run_thread_parallel_rollouts(
            state, num_rollouts, num_workers, max_depth
        )
        results.append(BenchmarkResult(
            strategy=f"Thread-{num_workers}",
            game_stage=game_stage,
            num_rollouts=num_rollouts,
            total_time_ms=thread_time * 1000,
            avg_time_per_rollout_ms=(thread_time * 1000) / num_rollouts,
            avg_score=thread_score,
            num_workers=num_workers
        ))
        speedup = seq_time / thread_time if thread_time > 0 else 0
        print(f"    Total: {thread_time*1000:.2f}ms, Per rollout: {(thread_time*1000)/num_rollouts:.2f}ms, "
              f"Speedup: {speedup:.2f}x, Avg score: {thread_score:.3f}")
    
    # Process parallel with different worker counts
    for num_workers in [2, 4, num_cpus]:
        if num_workers > num_cpus:
            continue
        print(f"\n  Running {num_rollouts} process-parallel rollouts ({num_workers} workers)...")
        proc_time, proc_score = run_process_parallel_rollouts(
            target_moves, num_rollouts, num_workers, max_depth
        )
        results.append(BenchmarkResult(
            strategy=f"Process-{num_workers}",
            game_stage=game_stage,
            num_rollouts=num_rollouts,
            total_time_ms=proc_time * 1000,
            avg_time_per_rollout_ms=(proc_time * 1000) / num_rollouts,
            avg_score=proc_score,
            num_workers=num_workers
        ))
        speedup = seq_time / proc_time if proc_time > 0 else 0
        print(f"    Total: {proc_time*1000:.2f}ms, Per rollout: {(proc_time*1000)/num_rollouts:.2f}ms, "
              f"Speedup: {speedup:.2f}x, Avg score: {proc_score:.3f}")
    
    return results


def print_summary_table(all_results: List[BenchmarkResult]):
    """Print a summary table of all results."""
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Stage':<15} {'Strategy':<15} {'Rollouts':<10} {'Total(ms)':<12} {'Per Roll(ms)':<14} {'Speedup':<10} {'Score':<8}")
    print("-"*100)
    
    # Group by game stage
    stages = sorted(set(r.game_stage for r in all_results))
    
    for stage in stages:
        stage_results = [r for r in all_results if r.game_stage == stage]
        seq_time = next((r.total_time_ms for r in stage_results if r.strategy == "Sequential"), 0)
        
        for r in stage_results:
            speedup = seq_time / r.total_time_ms if r.total_time_ms > 0 else 0
            speedup_str = f"{speedup:.2f}x" if r.strategy != "Sequential" else "-"
            print(f"{r.game_stage:<15} {r.strategy:<15} {r.num_rollouts:<10} "
                  f"{r.total_time_ms:<12.2f} {r.avg_time_per_rollout_ms:<14.2f} "
                  f"{speedup_str:<10} {r.avg_score:<8.3f}")
        print("-"*100)


def run_scaling_test(state: HLEGameState, 
                     target_moves: int,
                     game_stage: str,
                     rollout_counts: List[int] = [10, 50, 100, 200, 500]) -> List[BenchmarkResult]:
    """
    Test how performance scales with number of rollouts.
    """
    results = []
    num_cpus = multiprocessing.cpu_count()
    
    print(f"\n{'='*60}")
    print(f"Scaling Test - {game_stage}")
    print(f"{'='*60}")
    
    for num_rollouts in rollout_counts:
        print(f"\n  Testing {num_rollouts} rollouts...")
        
        # Sequential
        seq_time, seq_score = run_sequential_rollouts(state, num_rollouts)
        results.append(BenchmarkResult(
            strategy="Sequential",
            game_stage=f"{game_stage}-{num_rollouts}",
            num_rollouts=num_rollouts,
            total_time_ms=seq_time * 1000,
            avg_time_per_rollout_ms=(seq_time * 1000) / num_rollouts,
            avg_score=seq_score,
            num_workers=1
        ))
        
        # Process parallel (typically best for CPU-bound)
        proc_time, proc_score = run_process_parallel_rollouts(
            target_moves, num_rollouts, num_cpus
        )
        results.append(BenchmarkResult(
            strategy=f"Process-{num_cpus}",
            game_stage=f"{game_stage}-{num_rollouts}",
            num_rollouts=num_rollouts,
            total_time_ms=proc_time * 1000,
            avg_time_per_rollout_ms=(proc_time * 1000) / num_rollouts,
            avg_score=proc_score,
            num_workers=num_cpus
        ))
        
        speedup = seq_time / proc_time if proc_time > 0 else 0
        print(f"    Sequential: {seq_time*1000:.2f}ms, Process: {proc_time*1000:.2f}ms, Speedup: {speedup:.2f}x")
    
    return results


def main():
    """Run the complete benchmark suite."""
    print("="*60)
    print("HANABI ROLLOUT PARALLELIZATION BENCHMARK")
    print(f"CPU Count: {multiprocessing.cpu_count()}")
    print("="*60)
    
    all_results = []
    
    # Create initial state
    base_state = create_game_state(seed=-1)
    
    # Define game stages
    game_stages = [
        ("Early Game (0 moves)", 0),
        ("Mid-Early Game (10 moves)", 10),
        ("Mid Game (25 moves)", 25),
        ("Mid-Late Game (40 moves)", 40),
    ]
    
    # Benchmark each game stage
    for stage_name, num_moves in game_stages:
        state = advance_game_to_stage(base_state, num_moves)
        
        if state.is_terminal():
            print(f"\n{stage_name}: Game ended early (terminal state)")
            continue
        
        results = benchmark_all_strategies(
            state, num_moves, stage_name, num_rollouts=100
        )
        all_results.extend(results)
    
    # Print summary
    print_summary_table(all_results)
    
    # Scaling test on early game
    print("\n" + "="*60)
    print("SCALING TEST (Early Game)")
    print("="*60)
    scaling_results = run_scaling_test(
        base_state, 0,  # 0 moves for early game
        "Early", 
        rollout_counts=[10, 25, 50, 100, 200]
    )
    
    print("\n" + "="*60)
    print("SCALING RESULTS")
    print("="*60)
    print(f"{'Rollouts':<12} {'Seq Time(ms)':<15} {'Proc Time(ms)':<15} {'Speedup':<10}")
    print("-"*60)
    
    rollout_counts = [10, 25, 50, 100, 200]
    for count in rollout_counts:
        seq_r = next((r for r in scaling_results if r.num_rollouts == count and "Sequential" in r.strategy), None)
        proc_r = next((r for r in scaling_results if r.num_rollouts == count and "Process" in r.strategy), None)
        
        if seq_r and proc_r:
            speedup = seq_r.total_time_ms / proc_r.total_time_ms if proc_r.total_time_ms > 0 else 0
            print(f"{count:<12} {seq_r.total_time_ms:<15.2f} {proc_r.total_time_ms:<15.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    main()
