"""S5 LaunchGate — pre-launch taste validation."""
from __future__ import annotations

from user_soul.backend import LLMBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.behavior import BehaviorEngine
from user_soul.engines.vision import VisionEngine
from user_soul.models import (
    LaunchReport, AgentProfile, EvaluationMetric, PairwiseResult,
)


class LaunchGate:

    def __init__(self, backend: LLMBackend):
        self._persona = PersonaEngine(backend)
        self._behavior = BehaviorEngine(backend)
        self._vision = VisionEngine(backend)

    def run(self, product_description: str,
            product_screenshots: list[bytes],
            competitor_screenshots: list[tuple[str, bytes]], *,
            personas: list[AgentProfile] | None = None,
            metrics: list[EvaluationMetric] | None = None,
            goal: str | None = None) -> LaunchReport:
        pool = personas or self._persona.get_or_create(product_description, n=12)
        eval_metrics = metrics or self._behavior.extract_metrics(
            goal or product_description)

        # 1. Taste comparison vs each competitor
        taste_results: list[PairwiseResult] = []
        if product_screenshots and competitor_screenshots:
            taste_results = self._vision.batch_compare(
                product_screenshots[0], competitor_screenshots)

        wins = sum(1 for r in taste_results if r.winner == "ours")
        taste_win_rate = wins / max(len(taste_results), 1)

        # 2. Full behavioral simulation
        behavior = self._behavior.simulate(
            product_description, pool, eval_metrics,
            n_runs=30, adversarial=True)
        day1_return_adjusted = behavior.day1_return_rate_adjusted
        benchmark_context = behavior.benchmark_context

        # 3. Judge
        improvement_areas: list[str] = []
        taste_fail = taste_win_rate < 0.3 and len(taste_results) > 0
        behavior_fail = (day1_return_adjusted is not None
                         and day1_return_adjusted < 0.10)

        if taste_fail:
            improvement_areas.append("visual taste below competitors")
        if behavior_fail:
            improvement_areas.append("below survival threshold")

        if taste_fail and behavior_fail:
            recommendation = "ABANDON"
        elif taste_fail or behavior_fail:
            recommendation = "IMPROVE"
        else:
            recommendation = "SHIP"

        return LaunchReport(
            taste_results=taste_results,
            taste_win_rate=round(taste_win_rate, 4),
            behavior=behavior,
            day1_return_adjusted=day1_return_adjusted,
            benchmark_context=benchmark_context,
            recommendation=recommendation,
            improvement_areas=improvement_areas,
        )
