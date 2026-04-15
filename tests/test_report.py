
from unittest.mock import patch as _patch

from mcv.report import (
    _aggregate_bool, _aggregate_scale, _aggregate_text,
    aggregate, SimulationReport, MetricResult,
)
from mcv.schema_extractor import EvaluationMetric
from mcv.user_simulator import SessionResult
from mcv.scenarios import ScenarioContext


CTX = ScenarioContext("evening", "calm", 1, "curiosity")


def test_aggregate_bool_true_rate():
    r = _aggregate_bool(["yes", "yes", "no", "yes"])
    assert abs(r.true_rate - 0.75) < 0.01


def test_aggregate_bool_handles_chinese_yes():
    r = _aggregate_bool(["是", "否", "是"])
    assert abs(r.true_rate - 0.667) < 0.01


def test_aggregate_bool_empty():
    r = _aggregate_bool([])
    assert r.true_rate == 0.0


def test_aggregate_scale_mean():
    r = _aggregate_scale(["4", "3", "5", "4"])
    assert abs(r.mean - 4.0) < 0.01


def test_aggregate_scale_distribution_keys():
    r = _aggregate_scale(["4", "4", "3"])
    assert 4 in r.distribution
    assert 3 in r.distribution
    assert 1 not in r.distribution


def test_aggregate_scale_empty():
    r = _aggregate_scale([])
    assert r.mean == 0.0


def test_aggregate_text_returns_samples():
    r = _aggregate_text(["教程太长", "UI 复杂", "第一局输了"])
    assert "教程太长" in r.samples
    assert len(r.samples) <= 10


def test_aggregate_full_report():
    metrics = [
        EvaluationMetric("ret", "bool", "回来吗？"),
        EvaluationMetric("eng", "scale_1_5", "投入度？"),
    ]
    sessions = [
        SessionResult(CTX, "叙述...", {"ret": "yes", "eng": "4"}),
        SessionResult(CTX, "叙述...", {"ret": "no",  "eng": "3"}),
        SessionResult(CTX, "叙述...", {"ret": "yes", "eng": "5"}),
    ]
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert isinstance(report, SimulationReport)
    assert report.n_simulations == 3
    assert report.user_type == "玩家"
    assert abs(report.metrics["ret"].true_rate - 0.667) < 0.01
    assert abs(report.metrics["eng"].mean - 4.0) < 0.01


def test_aggregate_missing_metric_values():
    """Sessions with no value for a metric → zero/empty result, no crash."""
    metrics = [EvaluationMetric("ret", "bool", "?")]
    sessions = [SessionResult(CTX, "叙述...", {})]  # no values parsed
    report = aggregate(sessions, metrics, "玩家", "游戏")
    assert report.metrics["ret"].true_rate == 0.0


def test_aggregate_text_skips_llm_with_fewer_than_3_values():
    """Text aggregation with api_key but < 3 values → no LLM call, empty themes."""
    r = _aggregate_text(["只有一条"], api_key="test")
    assert r.themes == []
    assert r.samples == ["只有一条"]


def test_aggregate_bool_populates_ci():
    r = _aggregate_bool(["yes"] * 18 + ["no"] * 12)  # 30 samples, true_rate=0.6
    assert r.n_samples == 30
    assert r.stdev is not None
    assert r.ci_95_low is not None and r.ci_95_high is not None
    assert r.ci_95_low < r.true_rate < r.ci_95_high


def test_aggregate_bool_ci_empty():
    r = _aggregate_bool([])
    assert r.n_samples == 0
    assert r.ci_95_low == 0.0
    assert r.ci_95_high == 0.0


def test_aggregate_scale_populates_ci():
    r = _aggregate_scale(["4", "3", "5", "4", "4", "3"])
    assert r.n_samples == 6
    assert r.stdev is not None and r.stdev > 0
    assert r.ci_95_low is not None and r.ci_95_high is not None
    assert r.ci_95_low < r.mean < r.ci_95_high


def test_aggregate_scale_single_value_ci():
    r = _aggregate_scale(["3"])
    assert r.n_samples == 1
    assert r.stdev == 0.0
    assert r.ci_95_low == 3.0
    assert r.ci_95_high == 3.0


def test_aggregate_auto_key_findings():
    metrics = [
        EvaluationMetric("ret", "bool", "回来吗？"),
        EvaluationMetric("eng", "scale_1_5", "投入度？"),
    ]
    sessions = [
        SessionResult(CTX, "叙述", {"ret": "yes", "eng": "4"}),
        SessionResult(CTX, "叙述", {"ret": "no",  "eng": "3"}),
        SessionResult(CTX, "叙述", {"ret": "yes", "eng": "5"}),
    ]
    with _patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("Day-1 留存率为 67%，参与度均分 4.0。", 80)
        report = aggregate(sessions, metrics, "玩家", "游戏", api_key="test")
    assert len(report.key_findings) > 10


def test_aggregate_no_key_findings_without_api_key():
    metrics = [EvaluationMetric("ret", "bool", "?"), EvaluationMetric("eng", "scale_1_5", "?")]
    sessions = [SessionResult(CTX, "叙述", {"ret": "yes", "eng": "4"})]
    report = aggregate(sessions, metrics, "玩家", "游戏", api_key=None)
    assert report.key_findings == ""


def test_aggregate_no_key_findings_with_single_metric():
    metrics = [EvaluationMetric("ret", "bool", "?")]
    sessions = [SessionResult(CTX, "叙述", {"ret": "yes"})]
    with _patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("some findings", 50)
        report = aggregate(sessions, metrics, "玩家", "游戏", api_key="test")
    # Single metric → no key_findings LLM call
    assert report.key_findings == ""
