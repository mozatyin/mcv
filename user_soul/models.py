"""User-Soul data models — all dataclasses in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Persona types (from population.py)
# ---------------------------------------------------------------------------

@dataclass
class TraitDimension:
    name: str
    description: str
    low_label: str
    high_label: str
    distribution: str  # "normal" | "uniform" | "bimodal" | "right_skewed" | "left_skewed"
    mean: float
    std: float
    source: str  # "space1" | "space2" | "space3"


@dataclass
class Archetype:
    name: str
    frequency: float
    description: str
    trait_constraints: dict[str, tuple[float, float]]
    background_story: str = ""


@dataclass
class PersonaStructure:
    population_label: str
    product_context: str
    trait_dimensions: list[TraitDimension]
    archetypes: list[Archetype]
    research_notes: str = ""


@dataclass
class AgentProfile:
    agent_id: str
    archetype_name: str
    trait_vector: dict[str, float]
    dims: list = field(default_factory=list, repr=False, compare=False)
    background_story: str = ""

    def to_human_story(self) -> str:
        if self.background_story:
            return self.background_story
        return f"一个{self.archetype_name}类型的普通用户。"


# ---------------------------------------------------------------------------
# Evaluation types (from schema_extractor.py)
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetric:
    name: str
    type: str      # "bool" | "scale_1_5" | "text"
    question: str


# ---------------------------------------------------------------------------
# Simulation types (from user_simulator.py + report.py)
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    scenario: Any
    narrative: str
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class MetricResult:
    name: str
    type: str
    true_rate: float | None = None
    mean: float | None = None
    distribution: dict[int, float] | None = None
    themes: list[str] | None = None
    samples: list[str] | None = None
    stdev: float | None = None
    ci_95_low: float | None = None
    ci_95_high: float | None = None
    n_samples: int = 0


@dataclass
class SimulationReport:
    n_simulations: int
    user_type: str
    product_summary: str
    metrics: dict[str, MetricResult]
    key_findings: str = ""
    adversarial_frictions: list[str] = field(default_factory=list)
    _metrics_list: list = field(default_factory=list, repr=False, compare=False)

    @property
    def day1_return_rate(self) -> float | None:
        for mr in self.metrics.values():
            if mr.type == "bool" and mr.true_rate is not None:
                return mr.true_rate
        return None

    @property
    def friction_themes(self) -> list[str]:
        for mr in self.metrics.values():
            if mr.type == "text" and mr.themes:
                return mr.themes
        return []

    @property
    def locked_schema(self) -> list[dict]:
        return [
            {"name": mr.name, "type": mr.type,
             "question": self._metrics_list[i].question if i < len(self._metrics_list) else ""}
            for i, mr in enumerate(self.metrics.values())
        ]

    @property
    def day1_return_rate_adjusted(self) -> float | None:
        rate = self.day1_return_rate
        if rate is None:
            return None
        from user_soul.calibration import SYCOPHANCY_DEFLATOR
        return round(rate * SYCOPHANCY_DEFLATOR, 4)

    @property
    def hook_completion_rate(self) -> float | None:
        mr = self.metrics.get("hook_completed")
        if mr is not None and mr.true_rate is not None:
            return mr.true_rate
        return None

    @property
    def benchmark_context(self) -> str:
        rate = self.day1_return_rate_adjusted
        if rate is None:
            return ""
        if rate >= 0.35:
            return f"Excellent — top quartile (adjusted {rate:.0%} vs benchmark 35%+)"
        if rate >= 0.28:
            return f"Good — above industry average (adjusted {rate:.0%} vs benchmark 26-28%)"
        if rate >= 0.20:
            return f"Near industry average (adjusted {rate:.0%} vs benchmark 26-28%)"
        if rate >= 0.10:
            return f"Poor — below industry average (adjusted {rate:.0%} vs benchmark 26-28%)"
        return f"Below survival threshold (adjusted {rate:.0%} vs industry avg 26-28%)"


@dataclass
class CompareReport:
    n_runs_per_variant: int
    variant_a_label: str
    variant_b_label: str
    variant_a: SimulationReport
    variant_b: SimulationReport
    deltas: dict[str, float]
    improvements: list[str]
    regressions: list[str]
    key_diff: str


@dataclass
class FeatureAAR:
    feature_id: str
    acquisition: float
    activation: float
    retention: float
    revenue: float
    referral: float
    confidence: float
    archetype_votes: dict


@dataclass
class CoherenceReport:
    selected_feature_ids: list
    missing_dependencies: list
    blocked_journeys: list
    reinstate_recommendations: list
    is_coherent: bool


# ---------------------------------------------------------------------------
# Journey types (from journey.py)
# ---------------------------------------------------------------------------

@dataclass
class JourneyReport:
    target_flow: list[str]
    completion_rate: float
    drop_off_by_screen: dict
    fogg_violations: list[str]
    blocked_journeys: list[str]
    personas_completed: int
    personas_total: int

    @property
    def passes_gate(self) -> bool:
        return self.completion_rate >= 0.70

    @property
    def benchmark_context(self) -> str:
        r = self.completion_rate
        if r >= 0.85:
            return f"Strong ({r:.0%}) — above 85% excellence threshold"
        if r >= 0.70:
            return f"Acceptable ({r:.0%}) — above 70% gate threshold"
        if r >= 0.50:
            return f"Weak ({r:.0%}) — below 70% gate, flow needs redesign"
        return f"Critical ({r:.0%}) — majority cannot complete flow"


# ---------------------------------------------------------------------------
# Decision types (from core.py)
# ---------------------------------------------------------------------------

@dataclass
class DecisionResult:
    value: Any
    confidence: float
    distribution: dict[str, float]
    mode: str
    tokens_used: int = 0
    raw_votes: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vision types (NEW)
# ---------------------------------------------------------------------------

@dataclass
class PairwiseResult:
    winner: str  # "ours" | "theirs" | "tie"
    dimension_results: dict
    overall_reason: str
    confidence: float


@dataclass
class ReviewResult:
    issues: list
    overall_score: str  # "professional" | "acceptable" | "amateur"
    suggestions: list[str]


# ---------------------------------------------------------------------------
# Stage report types (NEW)
# ---------------------------------------------------------------------------

@dataclass
class ResearchReport:
    persona_structure: PersonaStructure | None
    feature_priorities: list[FeatureAAR]
    visual_preferences: list[PairwiseResult]
    latent_needs: list[str]


@dataclass
class DesignReviewReport:
    journey: JourneyReport | None
    layout_reviews: list[ReviewResult]
    competitor_gaps: list[PairwiseResult]
    passes_gate: bool


@dataclass
class ModuleUATReport:
    behavior: SimulationReport | None
    visual_issues: list[dict]
    friction_manifest: list[dict]
    passes_gate: bool


@dataclass
class LaunchReport:
    taste_results: list[PairwiseResult]
    taste_win_rate: float
    behavior: SimulationReport | None
    day1_return_adjusted: float | None
    benchmark_context: str
    recommendation: str  # "SHIP" | "IMPROVE" | "ABANDON"
    improvement_areas: list[str]
