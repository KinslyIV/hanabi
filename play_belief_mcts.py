#!/usr/bin/env python3
"""
Play a Hanabi game using Belief-Integrated MCTS.

This script demonstrates the belief-integrated MCTS that uses:
1. Belief states to track card probabilities
2. Neural network policy head for move priors
3. Neural network value head for state evaluation
"""

import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from hanabi_learning_environment import pyhanabi

from rl_hanabi.model.belief_model import ActionDecoder
from rl_hanabi.game.hle_state import HLEGameState
from rl_hanabi.belief.belief_state import BeliefState
from rl_hanabi.mcts.belief_mcts import BeliefMCTS


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    
    CARD_COLORS = [RED, YELLOW, GREEN, BLUE, WHITE]


def color_name(color_idx: int) -> str:
    """Get the name of a color from its index."""
    names = ["Red", "Yellow", "Green", "Blue", "White"]
    return names[color_idx] if color_idx < len(names) else f"Color{color_idx}"


def colorize_card(card: pyhanabi.HanabiCard, show_color: bool = True) -> str:
    """Format a card with colors."""
    color_idx = card.color()
    rank = card.rank() + 1
    
    if show_color and color_idx < len(Colors.CARD_COLORS):
        return f"{Colors.CARD_COLORS[color_idx]}{color_name(color_idx)[0]}{rank}{Colors.RESET}"
    return f"{color_name(color_idx)[0]}{rank}"


def format_hand(hand: List[pyhanabi.HanabiCard], hidden: bool = False) -> str:
    """Format a hand of cards."""
    if hidden:
        return " ".join(["[??]" for _ in hand])
    return " ".join([f"[{colorize_card(card)}]" for card in hand])


def format_fireworks(fireworks: List[int]) -> str:
    """Format the firework stacks."""
    parts = []
    for i, height in enumerate(fireworks):
        if i < len(Colors.CARD_COLORS):
            color = Colors.CARD_COLORS[i]
        else:
            color = ""
        parts.append(f"{color}{color_name(i)[0]}:{height}{Colors.RESET}")
    return " | ".join(parts)


def format_move(move: pyhanabi.HanabiMove, current_player: int, num_players: int) -> str:
    """Format a move description."""
    move_type = move.type()
    
    if move_type == pyhanabi.HanabiMoveType.PLAY:
        return f"{Colors.GREEN}PLAY{Colors.RESET} card at slot {move.card_index()}"
    
    elif move_type == pyhanabi.HanabiMoveType.DISCARD:
        return f"{Colors.YELLOW}DISCARD{Colors.RESET} card at slot {move.card_index()}"
    
    elif move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
        target = (current_player + move.target_offset()) % num_players
        color = color_name(move.color())
        return f"{Colors.CYAN}CLUE{Colors.RESET} Player {target}: {Colors.CARD_COLORS[move.color()]}{color}{Colors.RESET}"
    
    elif move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
        target = (current_player + move.target_offset()) % num_players
        rank = move.rank() + 1
        return f"{Colors.CYAN}CLUE{Colors.RESET} Player {target}: Rank {rank}"
    
    return str(move)


def load_model(checkpoint_path: Path, device: torch.device) -> ActionDecoder:
    """Load the model from a checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model_config = checkpoint.get("model_config", {})
    
    max_num_colors = model_config.get("max_num_colors", 5)
    max_num_ranks = model_config.get("max_num_ranks", 5)
    max_hand_size = model_config.get("max_hand_size", 5)
    max_num_players = model_config.get("max_num_players", 5)
    num_heads = model_config.get("num_heads", 4)
    num_layers = model_config.get("num_layers", 4)
    d_model = model_config.get("d_model", 128)
    
    model = ActionDecoder(
        max_num_colors=max_num_colors,
        max_num_ranks=max_num_ranks,
        max_hand_size=max_hand_size,
        max_num_players=max_num_players,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
        action_dim=4,
    )
    
    # Try to load state dict, but handle missing value head gracefully
    state_dict = checkpoint["model_state_dict"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Value head might be missing from old checkpoints
        print(f"{Colors.YELLOW}Warning: Some weights not found in checkpoint (likely value head). Initializing randomly.{Colors.RESET}")
        model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    return model


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the directory."""
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
    if checkpoints:
        def get_iter_num(p: Path) -> int:
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return -1
        checkpoints.sort(key=get_iter_num, reverse=True)
        return checkpoints[0]
    
    final = checkpoint_dir / "checkpoint_final.pt"
    if final.exists():
        return final
    
    return None


def print_game_state(
    state: HLEGameState,
    turn: int,
    current_player: int,
    num_players: int,
    show_all_hands: bool = True,
):
    """Print the current game state."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}Turn {turn} - Player {current_player}'s turn{Colors.RESET}")
    print(f"{'='*70}")
    
    print(f"\n{Colors.CYAN}Game Status:{Colors.RESET}")
    print(f"  Score: {Colors.GREEN}{state.score()}{Colors.RESET}/{state.max_score()}")
    print(f"  Clue tokens: {Colors.BLUE}{state.clue_tokens}{Colors.RESET}/8")
    print(f"  Lives: {Colors.RED}{state.life_tokens()}{Colors.RESET}/3")
    print(f"  Deck: {state.deck_size()} cards remaining")
    
    print(f"\n{Colors.CYAN}Fireworks:{Colors.RESET}")
    print(f"  {format_fireworks(state.fireworks())}")
    
    hands = state.get_hands()
    print(f"\n{Colors.CYAN}Hands:{Colors.RESET}")
    for p in range(num_players):
        if p == current_player:
            marker = f"{Colors.YELLOW}→{Colors.RESET}"
            if show_all_hands:
                hand_str = format_hand(hands[p], hidden=False)
                hand_str += f" {Colors.WHITE}(hidden from player){Colors.RESET}"
            else:
                hand_str = format_hand(hands[p], hidden=True)
        else:
            marker = " "
            hand_str = format_hand(hands[p], hidden=False)
        print(f"  {marker} Player {p}: {hand_str}")


def print_mcts_info(
    mcts: BeliefMCTS,
    action_idx: int,
    policy: np.ndarray,
    value: float,
    state: HLEGameState,
    current_player: int,
    num_players: int,
):
    """Print information about the MCTS decision."""
    move = state.index_to_move(action_idx)
    
    print(f"\n{Colors.CYAN}Belief MCTS Decision:{Colors.RESET}")
    print(f"  Action: {format_move(move, current_player, num_players)}")
    print(f"  Root Value: {value:.3f}")
    print(f"  Action Probability: {policy[action_idx]*100:.1f}%")
    
    if mcts.root is not None:
        print(f"  Root Visits: {mcts.root.n_visits}")
        
        # Show top 5 moves by visit count
        visit_counts = []
        for act_idx, child in mcts.root.children.items():
            visit_counts.append((act_idx, child.n_visits, child.q_value, child.prior))
        visit_counts.sort(key=lambda x: x[1], reverse=True)
        
        if visit_counts:
            print(f"\n  {Colors.WHITE}Top moves by visit count:{Colors.RESET}")
            for i, (act_idx, visits, q_val, prior) in enumerate(visit_counts[:5]):
                alt_move = state.index_to_move(act_idx)
                marker = "→" if act_idx == action_idx else " "
                print(f"    {marker} {format_move(alt_move, current_player, num_players)}: "
                      f"visits={visits}, Q={q_val:.3f}, prior={prior:.3f}")


def print_move_result(
    state: HLEGameState,
    move: pyhanabi.HanabiMove,
    prev_score: int,
    prev_lives: int,
):
    """Print the result of a move."""
    new_score = state.score()
    new_lives = state.life_tokens()
    
    if new_score > prev_score:
        print(f"\n  {Colors.GREEN}✓ Successful play! Score: {prev_score} → {new_score}{Colors.RESET}")
    elif new_lives < prev_lives:
        print(f"\n  {Colors.RED}✗ Failed play! Lives: {prev_lives} → {new_lives}{Colors.RESET}")


def print_belief_summary(belief_state: BeliefState, player: int, hand_size: int):
    """Print a summary of the belief state for a player."""
    print(f"\n{Colors.CYAN}Belief State Summary (Player {player}'s view of own hand):{Colors.RESET}")
    
    for slot in range(hand_size):
        color_probs = belief_state.color_belief[player, slot]
        rank_probs = belief_state.rank_belief[player, slot]
        
        # Get top color and rank
        top_color = np.argmax(color_probs)
        top_rank = np.argmax(rank_probs)
        
        color_conf = color_probs[top_color] * 100
        rank_conf = rank_probs[top_rank] * 100
        
        color_str = f"{Colors.CARD_COLORS[top_color]}{color_name(top_color)}{Colors.RESET}"
        
        print(f"  Slot {slot}: {color_str}({color_conf:.0f}%) Rank {top_rank+1}({rank_conf:.0f}%)")


def play_game_with_belief_mcts(
    model: ActionDecoder,
    num_players: int = 2,
    num_colors: int = 5,
    num_ranks: int = 5,
    hand_size: int = 5,
    seed: int = -1,
    time_ms: int = 1000,
    num_simulations: Optional[int] = None,
    c_puct: float = 1.4,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
    show_all_hands: bool = True,
    show_beliefs: bool = False,
    step_by_step: bool = False,
    verbose: bool = True,
):
    """
    Play a complete game using Belief MCTS.
    
    Args:
        model: The ActionDecoder model
        num_players: Number of players
        num_colors: Number of colors/suits
        num_ranks: Number of ranks
        hand_size: Cards per hand
        seed: Random seed (-1 for random)
        time_ms: MCTS time budget per move in milliseconds
        num_simulations: Fixed number of simulations (overrides time_ms)
        c_puct: PUCT exploration constant
        temperature: Temperature for action selection
        device: Device for inference
        show_all_hands: Show all hands (including current player's)
        show_beliefs: Show belief state summary
        step_by_step: Wait for input between turns
        verbose: Print detailed output
    
    Returns:
        Final score
    """
    # Game configuration
    options = {
        "numSuits": num_colors,
        "numRanks": num_ranks,
        "cardsPerHand": hand_size,
        "clueTokens": 8,
        "strikeTokens": 3,
        "seed": seed,
        "startingPlayer": 0,
    }
    
    # Initialize game
    state = HLEGameState.from_table_options(options, num_players)
    
    # Create belief states for all players
    belief_states = [
        BeliefState(state, player=p)
        for p in range(num_players)
    ]
    
    # Create MCTS
    mcts = BeliefMCTS(
        model=model,
        device=device,
        time_ms=time_ms,
        c_puct=c_puct,
        num_simulations=num_simulations,
        temperature=temperature,
    )
    
    if verbose:
        print(f"\n{Colors.BOLD}{'#'*70}{Colors.RESET}")
        print(f"{Colors.BOLD}#  HANABI GAME - Belief MCTS{Colors.RESET}")
        print(f"{Colors.BOLD}#  {num_players} Players, {num_colors} Colors, {num_ranks} Ranks{Colors.RESET}")
        print(f"{Colors.BOLD}#  Time: {time_ms}ms, Simulations: {num_simulations or 'unlimited'}{Colors.RESET}")
        print(f"{Colors.BOLD}{'#'*70}{Colors.RESET}")
    
    turn = 0
    
    while not state.is_terminal():
        current_player = state.current_player_index
        
        if verbose:
            print_game_state(state, turn, current_player, num_players, show_all_hands)
            
            if show_beliefs:
                print_belief_summary(belief_states[current_player], current_player, hand_size)
        
        # Initialize MCTS with current state and beliefs
        mcts.init_root(state, belief_states)
        
        # Run MCTS
        policy, value = mcts.run()
        
        # Select action
        action_idx, move = mcts.select_action()
        
        if verbose:
            print_mcts_info(mcts, action_idx, policy, value, state, current_player, num_players)
        
        # Store pre-move info
        prev_score = state.score()
        prev_lives = state.life_tokens()
        
        # Apply move
        state.apply_move(move)
        
        # Update all belief states
        for belief in belief_states:
            belief.reinit_belief_state()
            belief.state = state
            belief.update_from_move(model=model)
        
        if verbose:
            print_move_result(state, move, prev_score, prev_lives)
        
        if step_by_step and not state.is_terminal():
            input(f"\n{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
        
        turn += 1
    
    # Game over
    final_score = state.score()
    max_score = state.max_score()
    
    if verbose:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}GAME OVER{Colors.RESET}")
        print(f"{'='*70}")
        print(f"\nFinal Score: {Colors.GREEN}{final_score}{Colors.RESET}/{max_score}")
        print(f"Turns played: {turn}")
        
        if final_score == max_score:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎆 PERFECT GAME! 🎆{Colors.RESET}")
        elif final_score >= 20:
            print(f"\n{Colors.CYAN}Great game!{Colors.RESET}")
        elif final_score >= 15:
            print(f"\n{Colors.YELLOW}Good game!{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}Better luck next time!{Colors.RESET}")
    
    return final_score


def main():
    parser = argparse.ArgumentParser(description="Play Hanabi with Belief-Integrated MCTS")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory containing checkpoints")
    
    # Game arguments
    parser.add_argument("--players", type=int, default=2,
                        help="Number of players (2-5)")
    parser.add_argument("--colors", type=int, default=5,
                        help="Number of colors/suits")
    parser.add_argument("--ranks", type=int, default=5,
                        help="Number of ranks")
    parser.add_argument("--hand-size", type=int, default=5,
                        help="Cards per hand")
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    
    # MCTS arguments
    parser.add_argument("--time-ms", type=int, default=1000,
                        help="MCTS time budget per move in milliseconds")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Fixed number of MCTS simulations (overrides time-ms)")
    parser.add_argument("--c-puct", type=float, default=1.4,
                        help="PUCT exploration constant")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for action selection (0 = greedy)")
    
    # Display arguments
    parser.add_argument("--hide-hands", action="store_true",
                        help="Hide current player's hand")
    parser.add_argument("--show-beliefs", action="store_true",
                        help="Show belief state summary each turn")
    parser.add_argument("--step", action="store_true",
                        help="Step through game turn by turn")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cpu or cuda)")
    
    # Batch mode
    parser.add_argument("--num-games", type=int, default=1,
                        help="Number of games to play")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    
    if checkpoint_path is None or not checkpoint_path.exists():
        print(f"{Colors.RED}Error: No checkpoint found!{Colors.RESET}")
        print(f"Searched in: {args.checkpoint_dir}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    print("Model loaded successfully!")
    
    # Play games
    scores = []
    for game_num in range(args.num_games):
        if args.num_games > 1:
            print(f"\n{Colors.MAGENTA}{'='*70}{Colors.RESET}")
            print(f"{Colors.MAGENTA}Game {game_num + 1}/{args.num_games}{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'='*70}{Colors.RESET}")
        
        # Use different seed for each game if seed not fixed
        game_seed = args.seed if args.seed != -1 else random.randint(0, 2**31 - 1)
        
        score = play_game_with_belief_mcts(
            model=model,
            num_players=args.players,
            num_colors=args.colors,
            num_ranks=args.ranks,
            hand_size=args.hand_size,
            seed=game_seed,
            time_ms=args.time_ms,
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            temperature=args.temperature,
            device=device,
            show_all_hands=not args.hide_hands,
            show_beliefs=args.show_beliefs,
            step_by_step=args.step,
            verbose=not args.quiet,
        )
        scores.append(score)
    
    # Summary for multiple games
    if args.num_games > 1:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY ({args.num_games} games){Colors.RESET}")
        print(f"{'='*70}")
        print(f"Average Score: {Colors.GREEN}{np.mean(scores):.2f}{Colors.RESET}")
        print(f"Max Score: {Colors.GREEN}{max(scores)}{Colors.RESET}")
        print(f"Min Score: {Colors.YELLOW}{min(scores)}{Colors.RESET}")
        print(f"Std Dev: {np.std(scores):.2f}")
        perfect_games = sum(1 for s in scores if s == args.colors * args.ranks)
        print(f"Perfect Games: {perfect_games}/{args.num_games}")


if __name__ == "__main__":
    main()
