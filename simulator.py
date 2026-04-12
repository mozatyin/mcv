"""Progressive Monte Carlo behavioral simulator."""
from __future__ import annotations

from dataclasses import dataclass, field

from mcv.scenarios import ScenarioContext


@dataclass
class SimulationRun:
    """One simulated user session — a single roll of the die."""
    persona_id: str
    context: ScenarioContext
    features_used: list[str] = field(default_factory=list)
    features_skipped: list[str] = field(default_factory=list)


@dataclass
class FeatureSignal:
    """Empirical signal for one feature, aggregated from N simulation runs."""
    feature_id: str
    feature_name: str
    n_simulations: int
    usage_rate: float
    exposure_rate: float
    skip_rate: float
    context_map: dict[str, float]
    day_curve: dict[int, float]
    implied_kano: str
    implied_aarrr_score: float


def _derive_kano(usage_rate: float) -> str:
    """Derive Kano category from empirical usage frequency.

    Boundaries (inclusive lower, exclusive upper):
      Must-Have  : usage_rate >  0.80
      Performance: usage_rate >= 0.50
      Delighter  : usage_rate >= 0.20
      Indifferent: usage_rate <  0.20
    """
    if usage_rate > 0.80:
        return "Must-Have"
    if usage_rate >= 0.50:
        return "Performance"
    if usage_rate >= 0.20:
        return "Delighter"
    return "Indifferent"


def _derive_aarrr(day_curve: dict[int, float]) -> float:
    """Derive AARRR score: activation (day1) + retention (day7) + revenue proxy (day30)."""
    day1 = day_curve.get(1, 0.0)
    day7 = day_curve.get(7, 0.0)
    day30 = day_curve.get(30, 0.0)
    return round(0.30 * day1 + 0.30 * day7 + 0.40 * day30, 4)


class PersonaSimulator:
    """Stub — implemented in Tasks 3 and 4."""
    def __init__(self, personas: list[dict], api_key: str):
        self.personas = personas
        self.api_key = api_key

    def simulate(self, features: list[dict], n_runs: int = 5) -> list[FeatureSignal]:
        raise NotImplementedError

    def _simulate_one(self, persona: dict, features: list[dict], context: ScenarioContext) -> SimulationRun:
        raise NotImplementedError
