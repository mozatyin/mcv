"""S4 ModuleUAT — functional acceptance testing."""
from __future__ import annotations

from user_soul.backend import LLMBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.behavior import BehaviorEngine
from user_soul.engines.vision import VisionEngine
from user_soul.models import ModuleUATReport, AgentProfile, EvaluationMetric


class ModuleUAT:

    def __init__(self, backend: LLMBackend):
        self._persona = PersonaEngine(backend)
        self._behavior = BehaviorEngine(backend)
        self._vision = VisionEngine(backend)

    def run(self, product_description: str, *,
            personas: list[AgentProfile] | None = None,
            metrics: list[EvaluationMetric] | None = None,
            html_screenshots: list[bytes] | None = None,
            goal: str | None = None) -> ModuleUATReport:
        pool = personas or self._persona.get_or_create(product_description, n=12)
        eval_metrics = metrics or self._behavior.extract_metrics(
            goal or product_description)

        behavior = self._behavior.simulate(
            product_description, pool, eval_metrics,
            n_runs=30, adversarial=True)

        visual_issues: list[dict] = []
        if html_screenshots:
            for screenshot in html_screenshots:
                review = self._vision.screenshot_review(
                    screenshot, context=product_description)
                visual_issues.extend(review.issues)

        friction_manifest: list[dict] = []
        for friction in behavior.adversarial_frictions:
            friction_manifest.append({
                "source": "adversarial", "description": friction})
        for issue in visual_issues:
            if issue.get("severity") in ("P0", "P1"):
                friction_manifest.append({
                    "source": "visual",
                    "description": issue.get("description", "")})

        has_p0 = any(
            issue.get("severity") == "P0" for issue in visual_issues)

        return ModuleUATReport(
            behavior=behavior,
            visual_issues=visual_issues,
            friction_manifest=friction_manifest,
            passes_gate=not has_p0,
        )
