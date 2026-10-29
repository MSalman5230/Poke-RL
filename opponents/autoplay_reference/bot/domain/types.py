"""Typed boundary records; observations and hypotheses own separate snapshots."""
from dataclasses import dataclass, field
from typing import Any
from opponents.autoplay_reference.bot.domain.actions import Action, Decision as DecisionResult


@dataclass(frozen=True)
class ObservedBattleState:
    prompt: str
    state: dict[str, Any]
    turn: int


@dataclass(frozen=True)
class BuildPrediction:
    species: str
    member: dict[str, Any] | None
    battle_form: str | None
    fields: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def complete(self):
        return self.member is not None and not self.errors


@dataclass(frozen=True)
class TeamSelection:
    picks: tuple[str, ...]
    matrix: dict[str, dict[str, float]]
    candidates: tuple[dict[str, Any], ...]
    evidence: tuple[dict[str, Any], ...]
    lead_scores: dict[str, float]
    mega_owner: str | None = None
    forms: dict[str, str] = field(default_factory=dict)
    allowed_leads: tuple[str, ...] = ()
    policy: str = 'statistical-coverage-v1'
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionInput:
    observed: ObservedBattleState
    roster: dict[str, Any]
    predictions: tuple[BuildPrediction, ...]
    available: tuple[Action, ...]
    seeds: tuple[tuple[int, ...], ...]
    deadline: float
