"""Training utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import tomllib

from rl_hanabi.game import GameConfig


def apply_overrides(config: Dict[str, Dict[str, Any]], overrides: Dict[str, Any]) -> None:
    sections = [
        "default",
        "model",
        "training",
        "selfplay",
        "game_config",
        "exploration",
        "logging",
        "wandb",
    ]
    for key, value in overrides.items():
        if key == "wandb_enabled":
            config["wandb"]["enabled"] = value
            continue
        applied = False
        for section in sections:
            if key in config[section]:
                config[section][key] = value
                applied = True
                break
        if not applied:
            raise ValueError(f"Unknown preset override key: {key}")


def load_config(config_path: Path, preset: Optional[str]) -> Dict[str, Dict[str, Any]]:
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    config: Dict[str, Dict[str, Any]] = {
        "default": raw.get("default", {}),
        "model": raw.get("model", {}),
        "training": raw.get("training", {}),
        "selfplay": raw.get("selfplay", {}),
        "game_config": raw.get("game_config", {}),
        "exploration": raw.get("exploration", {}),
        "logging": raw.get("logging", {}),
        "wandb": raw.get("wandb", {}),
    }

    if preset:
        presets = raw.get("presets", {})
        overrides = presets.get(preset)
        if overrides is None:
            raise ValueError(f"Unknown preset: {preset}")
        apply_overrides(config, overrides)

    if "wandb_enabled" in config["default"]:
        config["wandb"]["enabled"] = config["default"].pop("wandb_enabled")

    return config


def build_game_config(config: Dict[str, Dict[str, Any]]) -> GameConfig:
    game_cfg = config["game_config"]
    model_cfg = config["model"]

    num_players = game_cfg.get("num_players")
    num_colors = game_cfg.get("num_colors")
    num_ranks = game_cfg.get("num_ranks")

    if num_players is None or num_colors is None or num_ranks is None:
        raise ValueError("game_config must define num_players, num_colors, num_ranks")

    if num_players != model_cfg["max_players"]:
        raise ValueError("game_config num_players must match model max_players")
    if num_colors != model_cfg["max_colors"]:
        raise ValueError("game_config num_colors must match model max_colors")
    if num_ranks != model_cfg["max_ranks"]:
        raise ValueError("game_config num_ranks must match model max_ranks")

    return GameConfig(
        num_players=num_players,
        num_colors=num_colors,
        num_ranks=num_ranks,
        hand_size=model_cfg["max_hand_size"],
        max_information_tokens=model_cfg["max_info_tokens"],
        max_life_tokens=model_cfg["max_life_tokens"],
        seed=-1,
    )
