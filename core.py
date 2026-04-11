"""Monte Carlo Voter — core types and PersonaDecider."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Persona:
    id: str
    name: str
    description: str
    cohort: str             # e.g. "College students 18-24, casual gamers"
    motivations: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)


@dataclass
class DecisionResult:
    value: Any                          # str for classify, float for score, bool for validate
    confidence: float                   # 0.0–1.0, fraction of personas that agreed
    distribution: dict[str, float]      # option → fraction of votes
    mode: str                           # "fast" | "validated"
    tokens_used: int = 0
    raw_votes: list[Any] = field(default_factory=list)   # per-persona values


class PersonaDecider:
    def __init__(self, personas: list[Persona], api_key: str, mode: str = "fast"):
        self.personas = personas
        self.api_key = api_key
        self.mode = mode

    def classify(self, question: str, options: list[str], context: str, batch: list[dict] | None = None) -> "DecisionResult | list[DecisionResult]":
        raise NotImplementedError

    def score(self, question: str, lo: float, hi: float, context: str, batch: list[dict] | None = None) -> "DecisionResult | list[DecisionResult]":
        raise NotImplementedError

    def validate(self, assertion: str, context: str) -> "DecisionResult":
        raise NotImplementedError
