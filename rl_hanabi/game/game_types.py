from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


class ACTION:
    PLAY = 0
    DISCARD = 1
    COLOUR = 2
    RANK = 3
    END_GAME = 4


class CLUE:
    COLOUR = 0
    RANK = 1


# Incoming actions from server (subset used for random bot)
@dataclass
class StatusAction:
    type: Literal["status"]
    clues: int
    score: int
    maxScore: int


@dataclass
class TurnAction:
    type: Literal["turn"]
    num: int
    currentPlayerIndex: int


@dataclass
class BaseClue:
    type: Literal[CLUE.COLOUR, CLUE.RANK] # type: ignore
    value: int


@dataclass
class ClueAction:
    type: Literal["clue"]
    giver: int
    target: int
    list: list[int]
    clue: BaseClue
    mistake: Optional[bool] = None


@dataclass
class CardAction:
    order: int
    playerIndex: int
    suitIndex: int
    rank: int


@dataclass
class DrawAction(CardAction):
    type: Literal["draw"]


@dataclass
class PlayAction(CardAction):
    type: Literal["play"]


@dataclass
class DiscardAction(CardAction):
    type: Literal["discard"]
    failed: bool


# Outgoing perform action (what we send on our turn)
@dataclass
class PerformAction:
    type: int  # one of ACTION.*
    target: int  # for PLAY/DISCARD: card order; for CLUE: player index
    value: Optional[int] = None  # for CLUE: colour index or rank value

