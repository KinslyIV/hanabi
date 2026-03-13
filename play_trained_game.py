#!/usr/bin/env python3
"""
Simulate a Hanabi game with the trained model and print states and action probs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from hanabi_learning_environment import pyhanabi

from rl_hanabi.game import HLEGameState
from rl_hanabi.model.action_decoder import ActionDecoder, ActionDecoderConfig
from rl_hanabi.model.tokenizer import HLETokenizer, TokenizationConfig
from rl_hanabi.training.utils import build_game_config, load_config


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest

    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
    if checkpoints:
        def get_iter_num(path: Path) -> int:
            try:
                return int(path.stem.split("_")[-1])
            except ValueError:
                return -1

        checkpoints.sort(key=get_iter_num, reverse=True)
        return checkpoints[0]

    final = checkpoint_dir / "checkpoint_final.pt"
    if final.exists():
        return final

    return None


def format_move(move: pyhanabi.HanabiMove, current_player: int, num_players: int) -> str:
    move_type = move.type()

    if move_type == pyhanabi.HanabiMoveType.PLAY:
        return f"PLAY slot {move.card_index()}"
    if move_type == pyhanabi.HanabiMoveType.DISCARD:
        return f"DISCARD slot {move.card_index()}"

    if move_type == pyhanabi.HanabiMoveType.REVEAL_COLOR:
        target = (current_player + move.target_offset()) % num_players
        return f"CLUE color {move.color()} -> player {target}"
    if move_type == pyhanabi.HanabiMoveType.REVEAL_RANK:
        target = (current_player + move.target_offset()) % num_players
        return f"CLUE rank {move.rank() + 1} -> player {target}"

    return str(move)


def print_state(state: HLEGameState, show_hands: bool, turn: int) -> None:
    print("\n" + "=" * 72)
    print(f"Turn {turn}, current_player={state.current_player_index}")
    print(f"Score {state.score()}/{state.max_score()}")
    print(state.state)



def main() -> None:
    parser = argparse.ArgumentParser(description="Play a Hanabi game with the trained model")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("rl_hanabi").joinpath("training", "config.toml")))
    parser.add_argument("--preset", type=str, default=None)

    args = parser.parse_args()

    config = load_config(Path(args.config), args.preset)
    play_cfg = config.get("play", {})

    device_name = play_cfg.get("device", config["default"].get("device", "auto"))
    checkpoint_dir = play_cfg.get("checkpoint_dir", config["logging"].get("checkpoint_dir", "checkpoints"))
    checkpoint_path_arg = play_cfg.get("checkpoint")
    temperature = float(play_cfg.get("temperature", 0.5))
    topk = int(play_cfg.get("topk", 5))
    show_hands = bool(play_cfg.get("show_hands", False))
    seed = int(play_cfg.get("seed", -1))
    max_turns = int(play_cfg.get("max_turns", 0))

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    game_config = build_game_config(config, seed)


    model_cfg = config["model"]
    token_config = TokenizationConfig(
        num_colors=model_cfg["max_colors"],
        num_ranks=model_cfg["max_ranks"],
        hand_size=model_cfg["max_hand_size"],
        num_players=model_cfg["max_players"],
        max_info_tokens=model_cfg["max_info_tokens"],
        max_life_tokens=model_cfg["max_life_tokens"],
    )
    decoder_config = ActionDecoderConfig(
        num_colors=model_cfg["max_colors"],
        num_ranks=model_cfg["max_ranks"],
        max_cards=token_config.total_cards,
        hand_size=model_cfg["max_hand_size"],
        num_players=model_cfg["max_players"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        d_model=model_cfg["d_model"],
    )

    if checkpoint_path_arg:
        checkpoint_path = Path(checkpoint_path_arg)
    else:
        checkpoint_path = find_latest_checkpoint(Path(checkpoint_dir))

    if checkpoint_path is None or not checkpoint_path.exists():
        raise FileNotFoundError("No checkpoint found")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = ActionDecoder(config=decoder_config, token_config=token_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = HLETokenizer(token_config)

    state = HLEGameState.from_table_options(game_config)
    previous_action_idx = -1
    turn = 0

    while not state.is_terminal():
        if max_turns > 0 and turn >= max_turns:
            break

        print_state(state, show_hands, turn)

        legal_moves_mask = state.legal_moves_mask()
        tokens = tokenizer.tokenize_state_and_action(
            state,
            previous_action_idx,
            state.current_player_index,
        )
        tokens = tokenizer.mask_player_hand_in_tokens(tokens, state.current_player_index)
        token_tensor = torch.tensor(
            tokenizer.pad_tokens(tokens),
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)

        with torch.no_grad():
            card_action_logits, value = model(token_tensor)
            current_player_tensor = torch.tensor([state.current_player_index], device=device)
            action_logits = tokenizer.action_logits_from_model(
                card_action_logits,
                token_tensor,
                current_player_tensor,
            )

        legal_mask_tensor = torch.as_tensor(legal_moves_mask, device=device)
        masked_logits = action_logits.masked_fill(~legal_mask_tensor, -1e9)

        if temperature > 0:
            masked_logits = masked_logits / temperature
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
            action_idx = int(torch.multinomial(probs, 1).item())
        else:
            probs = torch.softmax(masked_logits, dim=-1).squeeze(0)
            action_idx = int(masked_logits.argmax(dim=-1).item())

        topk_value = min(topk, probs.numel())
        top_probs, top_indices = torch.topk(probs, k=topk_value)

        print(f"Value estimate: {value.item():.4f}")
        print("Top actions:")
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            move = state.index_to_move(idx)
            print(f"  p={prob:.4f} idx={idx:3d} {format_move(move, state.current_player_index, state.num_players)}")

        chosen_move = state.index_to_move(action_idx)
        print(f"Chosen action: idx={action_idx} {format_move(chosen_move, state.current_player_index, state.num_players)}")

        state.apply_move_by_index(action_idx)
        previous_action_idx = action_idx
        turn += 1

    print("\n" + "=" * 72)
    print("Game finished")
    print(f"Final score: {state.score()}/{state.max_score()}")
    print(f"Turns: {turn}")


if __name__ == "__main__":
    main()
