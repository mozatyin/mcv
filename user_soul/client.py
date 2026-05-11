"""UserSoulClient — unified entry point for all User-Soul capabilities."""
from __future__ import annotations

from user_soul.backend import LLMBackend
from user_soul.engines.persona import PersonaEngine
from user_soul.engines.behavior import BehaviorEngine
from user_soul.engines.vision import VisionEngine
from user_soul.engines.vote import VoteEngine
from user_soul.stages.research import ResearchPanel
from user_soul.stages.design_review import DesignReview
from user_soul.stages.module_uat import ModuleUAT
from user_soul.stages.launch import LaunchGate
from user_soul.models import (
    AgentProfile, EvaluationMetric,
    ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport,
)


class UserSoulClient:

    def __init__(self, backend: LLMBackend):
        self._backend = backend
        self._persona = PersonaEngine(backend)
        self._behavior = BehaviorEngine(backend)
        self._vision = VisionEngine(backend)
        self._vote = VoteEngine(backend)

    def create_persona_pool(self, product_description: str,
                            n: int = 12) -> list[AgentProfile]:
        return self._persona.get_or_create(product_description, n)

    def research(self, product_description: str, features: list[dict], *,
                 competitor_screenshots: list[tuple[str, bytes]] | None = None,
                 our_screenshot: bytes | None = None) -> ResearchReport:
        panel = ResearchPanel(self._backend)
        return panel.run(product_description, features,
                         competitor_screenshots=competitor_screenshots,
                         our_screenshot=our_screenshot)

    def review(self, product_description: str, screens: list[dict],
               target_flow: list[str], *,
               personas: list[AgentProfile] | None = None,
               wireframe_screenshots: list[bytes] | None = None,
               competitor_screenshots: list[tuple[str, bytes]] | None = None) -> DesignReviewReport:
        stage = DesignReview(self._backend)
        return stage.run(product_description, screens, target_flow,
                         personas=personas,
                         wireframe_screenshots=wireframe_screenshots,
                         competitor_screenshots=competitor_screenshots)

    def verify(self, product_description: str, *,
               personas: list[AgentProfile] | None = None,
               metrics: list[EvaluationMetric] | None = None,
               html_screenshots: list[bytes] | None = None,
               goal: str | None = None) -> ModuleUATReport:
        stage = ModuleUAT(self._backend)
        return stage.run(product_description,
                         personas=personas, metrics=metrics,
                         html_screenshots=html_screenshots, goal=goal)

    def launch(self, product_description: str,
               product_screenshots: list[bytes],
               competitor_screenshots: list[tuple[str, bytes]], *,
               personas: list[AgentProfile] | None = None,
               metrics: list[EvaluationMetric] | None = None,
               goal: str | None = None) -> LaunchReport:
        stage = LaunchGate(self._backend)
        return stage.run(product_description, product_screenshots,
                         competitor_screenshots,
                         personas=personas, metrics=metrics, goal=goal)
