"""S2 DesignReview — wireframe usability test."""
from __future__ import annotations

from user_soul.backend import LLMBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.behavior import BehaviorEngine
from user_soul.engines.vision import VisionEngine
from user_soul.models import (
    DesignReviewReport, AgentProfile, ReviewResult, PairwiseResult,
)


class DesignReview:

    def __init__(self, backend: LLMBackend):
        self._persona = PersonaEngine(backend)
        self._behavior = BehaviorEngine(backend)
        self._vision = VisionEngine(backend)

    def run(self, product_description: str, screens: list[dict],
            target_flow: list[str], *,
            personas: list[AgentProfile] | None = None,
            wireframe_screenshots: list[bytes] | None = None,
            competitor_screenshots: list[tuple[str, bytes]] | None = None,
            ) -> DesignReviewReport:
        pool = personas or self._persona.get_or_create(product_description, n=12)

        journey = self._behavior.simulate_journey(screens, target_flow, pool)

        layout_reviews: list[ReviewResult] = []
        if wireframe_screenshots:
            for screenshot in wireframe_screenshots:
                review = self._vision.screenshot_review(
                    screenshot, context=product_description)
                layout_reviews.append(review)

        competitor_gaps: list[PairwiseResult] = []
        if wireframe_screenshots and competitor_screenshots:
            competitor_gaps = self._vision.batch_compare(
                wireframe_screenshots[0], competitor_screenshots)

        return DesignReviewReport(
            journey=journey,
            layout_reviews=layout_reviews,
            competitor_gaps=competitor_gaps,
            passes_gate=journey.passes_gate,
        )
