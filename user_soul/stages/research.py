"""S1 ResearchPanel — ELTM invites a focus group."""
from __future__ import annotations

from user_soul.backend import LLMBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.vision import VisionEngine
from user_soul.engines.vote import VoteEngine
from user_soul.models import ResearchReport, PairwiseResult


class ResearchPanel:

    def __init__(self, backend: LLMBackend):
        self._persona = PersonaEngine(backend)
        self._vision = VisionEngine(backend)
        self._vote = VoteEngine(backend)

    def run(self, product_description: str, features: list[dict], *,
            competitor_screenshots: list[tuple[str, bytes]] | None = None,
            our_screenshot: bytes | None = None) -> ResearchReport:
        structure = self._persona.research(product_description)

        feature_priorities = self._vote.aarrr(
            product_description, features, structure.archetypes
        ) if features else []

        visual_preferences: list[PairwiseResult] = []
        if our_screenshot and competitor_screenshots:
            visual_preferences = self._vision.batch_compare(
                our_screenshot, competitor_screenshots
            )

        latent_needs = [structure.research_notes] if structure.research_notes else []

        return ResearchReport(
            persona_structure=structure,
            feature_priorities=feature_priorities,
            visual_preferences=visual_preferences,
            latent_needs=latent_needs,
        )
