from user_soul.models import (
    PersonaStructure, Archetype, TraitDimension, AgentProfile,
    EvaluationMetric, SessionResult, MetricResult, SimulationReport,
    CompareReport, FeatureAAR, CoherenceReport, JourneyReport,
    DecisionResult, PairwiseResult, ReviewResult,
    ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport,
)


def test_pairwise_result_fields():
    r = PairwiseResult(winner="ours", dimension_results={}, overall_reason="better", confidence=0.9)
    assert r.winner == "ours"


def test_review_result_fields():
    r = ReviewResult(issues=[], overall_score="professional", suggestions=[])
    assert r.overall_score == "professional"


def test_launch_report_fields():
    r = LaunchReport(
        taste_results=[], taste_win_rate=0.8,
        behavior=None, day1_return_adjusted=0.3,
        benchmark_context="Good", recommendation="SHIP", improvement_areas=[],
    )
    assert r.recommendation == "SHIP"


def test_journey_report_passes_gate():
    r = JourneyReport(
        target_flow=["a", "b"], completion_rate=0.75,
        drop_off_by_screen={}, fogg_violations=[], blocked_journeys=[],
        personas_completed=9, personas_total=12,
    )
    assert r.passes_gate is True


def test_journey_report_fails_gate():
    r = JourneyReport(
        target_flow=["a", "b"], completion_rate=0.60,
        drop_off_by_screen={}, fogg_violations=[], blocked_journeys=[],
        personas_completed=7, personas_total=12,
    )
    assert r.passes_gate is False


def test_agent_profile_to_human_story():
    a = AgentProfile(agent_id="x", archetype_name="Casual",
                     trait_vector={}, background_story="小明25岁学生")
    assert a.to_human_story() == "小明25岁学生"


def test_agent_profile_fallback_story():
    a = AgentProfile(agent_id="x", archetype_name="Casual", trait_vector={})
    assert "Casual" in a.to_human_story()


def test_simulation_report_day1_return():
    mr = MetricResult(name="day_1_return_intent", type="bool", true_rate=0.80, n_samples=60)
    r = SimulationReport(n_simulations=60, user_type="test", product_summary="x",
                         metrics={"day_1_return_intent": mr})
    assert r.day1_return_rate == 0.80
    assert r.day1_return_rate_adjusted is not None
    assert r.day1_return_rate_adjusted < 0.80
