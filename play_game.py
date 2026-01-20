#!/usr/bin/env python3
"""
Play a Hanabi game using the latest trained model and display the game progress.

This script loads the latest checkpoint, runs a complete game with the trained
model making all decisions, and displays detailed information about each move.
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
    
    # Card colors
    CARD_COLORS = [RED, YELLOW, GREEN, BLUE, WHITE]


def color_name(color_idx: int) -> str:
    """Get the name of a color from its index."""
    names = ["Red", "Yellow", "Green", "Blue", "White"]
    return names[color_idx] if color_idx < len(names) else f"Color{color_idx}"


def colorize_card(card: pyhanabi.HanabiCard, show_color: bool = True) -> str:
    """Format a card with colors."""
    color_idx = card.color()
    rank = card.rank() + 1  # Convert 0-indexed to 1-indexed
    
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
    
    # Try to get model config from checkpoint, or use defaults
    model_config = checkpoint.get("model_config", {})
    
    # Use defaults if not in checkpoint
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
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the latest checkpoint in the directory."""
    # First try checkpoint_latest.pt
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    
    # Find the highest numbered checkpoint
    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
    if checkpoints:
        # Sort by iteration number
        def get_iter_num(p: Path) -> int:
            try:
                return int(p.stem.split("_")[-1])
            except ValueError:
                return -1
        checkpoints.sort(key=get_iter_num, reverse=True)
        return checkpoints[0]
    
    # Try checkpoint_final.pt
    final = checkpoint_dir / "checkpoint_final.pt"
    if final.exists():
        return final
    
    return None


def pad_observation(
    all_hands: np.ndarray,
    fireworks: np.ndarray,
    discard_pile: np.ndarray,
    affected_mask: np.ndarray,
    num_players: int,
    num_colors: int,
    num_ranks: int,
    hand_size: int,
    max_num_colors: int,
    max_num_ranks: int,
    max_hand_size: int,
    max_num_players: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad observations to match model's expected max dimensions."""
    
    # Pad slot_beliefs: [P, H, C+R] -> [max_P, max_H, max_C + max_R]
    padded_beliefs = np.zeros(
        (max_num_players, max_hand_size, max_num_colors + max_num_ranks),
        dtype=np.float32
    )
    # Copy color beliefs
    padded_beliefs[:num_players, :hand_size, :num_colors] = all_hands[:, :, :num_colors]
    # Copy rank beliefs (shifted to after max_colors)
    padded_beliefs[:num_players, :hand_size, max_num_colors:max_num_colors + num_ranks] = \
        all_hands[:, :, num_colors:num_colors + num_ranks]
    
    # Pad fireworks: [C] -> [max_C]
    padded_fireworks = np.zeros(max_num_colors, dtype=np.float32)
    padded_fireworks[:num_colors] = fireworks
    
    # Pad discard_pile: [C*R] -> [max_C * max_R]
    padded_discard = np.zeros(max_num_colors * max_num_ranks, dtype=np.float32)
    # Remap indices from original layout to padded layout
    for orig_idx in range(len(discard_pile)):
        if discard_pile[orig_idx] > 0:
            orig_color = orig_idx // num_ranks
            orig_rank = orig_idx % num_ranks
            new_idx = orig_color * max_num_ranks + orig_rank
            padded_discard[new_idx] = discard_pile[orig_idx]
    
    # Pad affected_mask: [P, H] -> [max_P, max_H]
    padded_mask = np.zeros((max_num_players, max_hand_size), dtype=np.float32)
    padded_mask[:num_players, :hand_size] = affected_mask[:num_players, :hand_size]
    
    return padded_beliefs, padded_fireworks, padded_discard, padded_mask


def select_action(
    model: ActionDecoder,
    belief_state: BeliefState,
    legal_moves_mask: np.ndarray,
    num_players: int,
    num_colors: int,
    num_ranks: int,
    hand_size: int,
    device: torch.device,
    temperature: float = 0.5,
) -> Tuple[int, np.ndarray, np.ndarray]:
    """Select an action using the model."""
    
    player = belief_state.player
    all_hands, fireworks, discard_pile_one_hot, tokens = belief_state.prepare_belief_obs(player)
    
    # Create action encoding from last action
    action_encoding, affected_mask = belief_state.encode_last_action()
    
    # Pad observations
    padded_beliefs, padded_fireworks, padded_discard, padded_mask = pad_observation(
        all_hands, fireworks, discard_pile_one_hot, affected_mask,
        num_players, num_colors, num_ranks, hand_size,
        model.max_num_colors, model.max_num_ranks, model.max_hand_size, model.max_num_players,
    )
    
    player_idx, target_idx = belief_state.get_last_player_and_target_index()
    player_offset = (player_idx - player) % num_players
    target_offset = (target_idx - player) % num_players
    
    # Convert to tensors
    slot_beliefs_tensor = torch.from_numpy(padded_beliefs).float().unsqueeze(0).to(device)
    affected_mask_tensor = torch.from_numpy(padded_mask).float().unsqueeze(0).to(device)
    action_tensor = torch.from_numpy(action_encoding).float().unsqueeze(0).to(device)
    fireworks_tensor = torch.from_numpy(padded_fireworks).float().unsqueeze(0).to(device)
    discard_pile_tensor = torch.from_numpy(padded_discard).float().unsqueeze(0).to(device)
    target_player_tensor = torch.tensor([target_offset], dtype=torch.long, device=device)
    acting_player_tensor = torch.tensor([player_offset], dtype=torch.long, device=device)
    
    with torch.no_grad():
        _, _, action_logits = model(
            slot_beliefs=slot_beliefs_tensor,
            affected_mask=affected_mask_tensor,
            move_target_player=target_player_tensor,
            acting_player=acting_player_tensor,
            action=action_tensor,
            fireworks=fireworks_tensor,
            discard_pile=discard_pile_tensor,
        )
    
    action_logits = action_logits.squeeze(0).cpu().numpy()
    
    # Mask illegal actions
    action_logits = action_logits[:len(legal_moves_mask)]
    action_logits[~legal_moves_mask] = -float('inf')
    
    # Apply temperature
    if temperature > 0:
        action_logits = action_logits / temperature
    
    # Softmax to get probabilities
    exp_logits = np.exp(action_logits - np.max(action_logits))
    probs = exp_logits / (exp_logits.sum() + 1e-10)
    
    # Select action (greedy for display)
    if temperature == 0:
        action_idx = np.argmax(probs)
    else:
        action_idx = np.random.choice(len(probs), p=probs)
    
    return action_idx, probs, action_logits


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
    
    # Game status
    print(f"\n{Colors.CYAN}Game Status:{Colors.RESET}")
    print(f"  Score: {Colors.GREEN}{state.score()}{Colors.RESET}/{state.max_score()}")
    print(f"  Clue tokens: {Colors.BLUE}{state.clue_tokens}{Colors.RESET}/8")
    print(f"  Lives: {Colors.RED}{state.life_tokens()}{Colors.RESET}/3")
    print(f"  Deck: {state.deck_size()} cards remaining")
    
    # Fireworks
    print(f"\n{Colors.CYAN}Fireworks:{Colors.RESET}")
    print(f"  {format_fireworks(state.fireworks())}")
    
    # Hands
    hands = state.get_hands()
    print(f"\n{Colors.CYAN}Hands:{Colors.RESET}")
    for p in range(num_players):
        if p == current_player:
            marker = f"{Colors.YELLOW}→{Colors.RESET}"
            # Current player can't see their own hand
            if show_all_hands:
                hand_str = format_hand(hands[p], hidden=False)
                hand_str += f" {Colors.WHITE}(hidden from player){Colors.RESET}"
            else:
                hand_str = format_hand(hands[p], hidden=True)
        else:
            marker = " "
            hand_str = format_hand(hands[p], hidden=False)
        print(f"  {marker} Player {p}: {hand_str}")


def print_move_selection(
    state: HLEGameState,
    action_idx: int,
    probs: np.ndarray,
    current_player: int,
    num_players: int,
):
    """Print information about the selected move."""
    move = state.index_to_move(action_idx)
    
    print(f"\n{Colors.CYAN}Model's Decision:{Colors.RESET}")
    print(f"  Action: {format_move(move, current_player, num_players)}")
    print(f"  Confidence: {probs[action_idx]*100:.1f}%")
    
    # Show top 3 alternative moves
    top_indices = np.argsort(probs)[::-1][:5]
    if len(top_indices) > 1:
        print(f"\n  {Colors.WHITE}Top alternatives:{Colors.RESET}")
        for i, idx in enumerate(top_indices):
            if probs[idx] > 0.001:  # Only show moves with >0.1% probability
                alt_move = state.index_to_move(idx)
                marker = "→" if idx == action_idx else " "
                print(f"    {marker} {format_move(alt_move, current_player, num_players)}: {probs[idx]*100:.1f}%")


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


def play_game(
    model: ActionDecoder,
    num_players: int = 3,
    num_colors: int = 5,
    num_ranks: int = 5,
    hand_size: int = 5,
    seed: int = -1,
    temperature: float = 0.5,
    device: torch.device = torch.device("cpu"),
    show_all_hands: bool = True,
    step_by_step: bool = False,
):
    """Play a complete game and display it."""
    
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
    
    print(f"\n{Colors.BOLD}{'#'*70}{Colors.RESET}")
    print(f"{Colors.BOLD}#  HANABI GAME - {num_players} Players, {num_colors} Colors, {num_ranks} Ranks{Colors.RESET}")
    print(f"{Colors.BOLD}{'#'*70}{Colors.RESET}")
    
    legal_moves_mask = state.legal_moves_mask()
    turn = 0
    
    while not state.is_terminal():
        current_player = state.current_player_index
        
        # Print game state
        print_game_state(state, turn + 1, current_player, num_players, show_all_hands)
        
        # Get legal moves
        legal_moves = state.legal_moves()
        legal_moves_mask = state.legal_moves_mask()
        
        if not legal_moves:
            print(f"\n{Colors.RED}No legal moves available!{Colors.RESET}")
            break
        
        # Select action
        action_idx, probs, _ = select_action(
            model,
            belief_states[current_player],
            legal_moves_mask,
            num_players,
            num_colors,
            num_ranks,
            hand_size,
            device,
            temperature,
        )
        
        # Print move selection
        print_move_selection(state, action_idx, probs, current_player, num_players)
        
        # Get move and apply it
        move = state.index_to_move(action_idx)
        prev_score = state.score()
        prev_lives = state.life_tokens()
        
        state.apply_move(move)
        
        # Print move result
        print_move_result(state, move, prev_score, prev_lives)
        
        # Update belief states
        for bs in belief_states:
            bs.state = state
            bs.update_from_move()
        
        turn += 1
        
        if step_by_step:
            input(f"\n{Colors.WHITE}Press Enter to continue...{Colors.RESET}")
    
    # Final results
    print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}GAME OVER{Colors.RESET}")
    print(f"{'='*70}")
    
    final_score = state.score()
    max_score = state.max_score()
    
    if final_score == max_score:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 PERFECT SCORE! {final_score}/{max_score}{Colors.RESET}")
    elif final_score >= max_score * 0.8:
        print(f"\n{Colors.GREEN}Great game! Score: {final_score}/{max_score} ({final_score/max_score*100:.0f}%){Colors.RESET}")
    elif final_score >= max_score * 0.5:
        print(f"\n{Colors.YELLOW}Good effort! Score: {final_score}/{max_score} ({final_score/max_score*100:.0f}%){Colors.RESET}")
    else:
        print(f"\n{Colors.RED}Score: {final_score}/{max_score} ({final_score/max_score*100:.0f}%){Colors.RESET}")
    
    print(f"Total turns: {turn}")
    print(f"Lives remaining: {state.life_tokens()}")
    
    return final_score, max_score, turn


def main():
    parser = argparse.ArgumentParser(description="Play Hanabi with a trained model")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Specific checkpoint file to load (overrides --checkpoint-dir)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=3,
        help="Number of players (default: 3)",
    )
    parser.add_argument(
        "--num-colors",
        type=int,
        default=5,
        help="Number of colors/suits (default: 5)",
    )
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=5,
        help="Number of ranks (default: 5)",
    )
    parser.add_argument(
        "--hand-size",
        type=int,
        default=5,
        help="Hand size (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed for game (-1 for random)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="Temperature for action selection (0=greedy, higher=more random)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on (cpu, cuda, auto)",
    )
    parser.add_argument(
        "--hide-hands",
        action="store_true",
        help="Hide cards from current player's hand (more realistic view)",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="Step through game turn by turn (press Enter to continue)",
    )
    parser.add_argument(
        "--num-games",
        type=int,
        default=1,
        help="Number of games to play (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
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
        print(f"Looked in: {args.checkpoint_dir}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    print(f"Model loaded successfully!")
    
    # Play games
    scores = []
    for game_num in range(args.num_games):
        if args.num_games > 1:
            print(f"\n{Colors.BOLD}Game {game_num + 1}/{args.num_games}{Colors.RESET}")
        
        # Use different seed for each game if seed is -1
        game_seed = args.seed if args.seed >= 0 else random.randint(0, 1000000)
        
        score, max_score, turns = play_game(
            model=model,
            num_players=args.num_players,
            num_colors=args.num_colors,
            num_ranks=args.num_ranks,
            hand_size=args.hand_size,
            seed=game_seed,
            temperature=args.temperature,
            device=device,
            show_all_hands=not args.hide_hands,
            step_by_step=args.step,
        )
        scores.append(score)
    
    # Summary for multiple games
    if args.num_games > 1:
        print(f"\n{Colors.BOLD}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}SUMMARY - {args.num_games} Games{Colors.RESET}")
        print(f"{'='*70}")
        print(f"Average score: {np.mean(scores):.1f}/{max_score}")
        print(f"Min score: {np.min(scores)}/{max_score}")
        print(f"Max score: {np.max(scores)}/{max_score}")
        print(f"Perfect games: {sum(1 for s in scores if s == max_score)}/{args.num_games}")


if __name__ == "__main__":
    main()
