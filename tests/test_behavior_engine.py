from user_soul.engines.behavior import BehaviorEngine, DomainConfig
from user_soul.models import (
    AgentProfile, EvaluationMetric, SimulationReport,
    CompareReport, JourneyReport,
)


class _FakeBackend:
    def text(self, prompt, **kw):
        return (
            "他打开了app，点击了开始按钮，选择了对手。\n"
            "他下了第一步棋，等待对方回应。\n"
            "day_1_return_intent: yes\n"
            "friction_points: 等待时间长\n"
        )
    def vision(self, prompt, images, **kw):
        return ""


def _make_pool(n=3):
    return [
        AgentProfile(agent_id=f"a{i}", archetype_name="Casual",
                     trait_vector={"skill": 3.0}, background_story=f"用户{i}")
        for i in range(n)
    ]


def _make_metrics():
    return [
        EvaluationMetric("day_1_return_intent", "bool", "Will user return tomorrow?"),
        EvaluationMetric("friction_points", "text", "Key friction points?"),
    ]


def test_simulate_returns_report():
    engine = BehaviorEngine(_FakeBackend())
    report = engine.simulate("chess game", _make_pool(), _make_metrics(), n_runs=3, adversarial=False)
    assert isinstance(report, SimulationReport)
    assert report.n_simulations == 3


def test_simulate_with_adversarial():
    engine = BehaviorEngine(_FakeBackend())
    report = engine.simulate("chess game", _make_pool(), _make_metrics(), n_runs=5, adversarial=True)
    assert isinstance(report, SimulationReport)


def test_compare_returns_compare_report():
    engine = BehaviorEngine(_FakeBackend())
    report = engine.compare("chess v1", "chess v2", _make_pool(), _make_metrics(), n_runs=3)
    assert isinstance(report, CompareReport)


def test_simulate_journey_returns_journey_report():
    class _JourneyBackend:
        def text(self, prompt, **kw):
            return "proceed: yes\nreason: clear navigation\nfogg_issue: none"
        def vision(self, prompt, images, **kw):
            return ""

    engine = BehaviorEngine(_JourneyBackend())
    screens = [
        {"screen_id": "home", "navigates_to": ["game"]},
        {"screen_id": "game", "navigates_to": ["result"]},
        {"screen_id": "result", "navigates_to": []},
    ]
    report = engine.simulate_journey(screens, ["home", "game", "result"], _make_pool())
    assert isinstance(report, JourneyReport)
    assert report.completion_rate > 0


def test_extract_metrics():
    class _MetricBackend:
        def text(self, prompt, **kw):
            return '[{"name":"retention","type":"bool","question":"Return?"},{"name":"satisfaction","type":"scale_1_5","question":"Rate?"}]'
        def vision(self, prompt, images, **kw):
            return ""
    engine = BehaviorEngine(_MetricBackend())
    metrics = engine.extract_metrics("chess game retention")
    assert len(metrics) == 2
    assert all(isinstance(m, EvaluationMetric) for m in metrics)
