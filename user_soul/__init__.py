"""User-Soul — user advocate across the entire product lifecycle."""
from user_soul.client import UserSoulClient
from user_soul.backend import LLMBackend
from user_soul.models import (
    AgentProfile, EvaluationMetric, PersonaStructure,
    SimulationReport, CompareReport, JourneyReport,
    ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport,
    PairwiseResult, ReviewResult, DecisionResult, FeatureAAR,
    PlaytestFeedback, PlaytestIssue,
    GradedPlaytestFeedback, ActionSpec,
)
from user_soul.playtest_bridge import extract_game_rules, run_graded_playtest
from user_soul.game_knowledge import (
    GameKnowledge, KnowledgeTier, brief_for_tier,
    DifferentialDiagnosis, TierResult, DiagnosisItem, DiagnosisCategory,
)
from user_soul.action_router import (
    route_diagnosis, route_flat_feedback,
    group_by_owner, format_action_summary,
)

__all__ = [
    "UserSoulClient", "LLMBackend",
    "AgentProfile", "EvaluationMetric", "PersonaStructure",
    "SimulationReport", "CompareReport", "JourneyReport",
    "ResearchReport", "DesignReviewReport", "ModuleUATReport", "LaunchReport",
    "PairwiseResult", "ReviewResult", "DecisionResult", "FeatureAAR",
    "PlaytestFeedback", "PlaytestIssue",
    "GradedPlaytestFeedback", "ActionSpec",
    "extract_game_rules", "run_graded_playtest",
    "GameKnowledge", "KnowledgeTier", "brief_for_tier",
    "DifferentialDiagnosis", "TierResult", "DiagnosisItem", "DiagnosisCategory",
    "route_diagnosis", "route_flat_feedback",
    "group_by_owner", "format_action_summary",
]
