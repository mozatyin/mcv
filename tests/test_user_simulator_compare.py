"""Tests for UserSimulator.compare()."""
from unittest.mock import patch, call
from mcv.user_simulator import UserSimulator
from mcv.domain_configs import GameDomainConfig
from mcv.schema_extractor import EvaluationMetric
from mcv.report import CompareReport


METRICS = [EvaluationMetric("ret", "bool", "回来吗？")]


def test_compare_returns_compare_report():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nret: yes", 100)
        with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_schema:
            mock_schema.return_value = METRICS
            result = sim.compare("prd_v1", "prd_v2", n_runs=4)
    assert isinstance(result, CompareReport)
    assert result.n_runs_per_variant == 4


def test_compare_runs_both_variants_same_n():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nret: yes", 100)
        with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_schema:
            mock_schema.return_value = METRICS
            sim.compare("prd_v1", "prd_v2", n_runs=5)
    # 10 LLM calls for sessions (5×2) plus possibly 1 for key_diff summary
    assert mock_llm.call_count >= 10


def test_compare_uses_shared_scenario_seeds():
    """Both variants receive prompts — v1 prompts contain prd_v1, v2 contain prd_v2."""
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    seen_prompts = []

    def capture_call(prompt, api_key, **kwargs):
        seen_prompts.append(prompt)
        return ("叙述...\nret: yes", 100)

    with patch("mcv.core._llm_call", side_effect=capture_call):
        with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_schema:
            mock_schema.return_value = METRICS
            sim.compare("prd_v1", "prd_v2", n_runs=4)

    session_prompts = [p for p in seen_prompts if "叙述" not in p]  # exclude any LLM summaries
    v1_prompts = [p for p in seen_prompts if "prd_v1" in p]
    v2_prompts = [p for p in seen_prompts if "prd_v2" in p]
    assert len(v1_prompts) == 4
    assert len(v2_prompts) == 4


def test_compare_labels_preserved():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nret: yes", 100)
        with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_schema:
            mock_schema.return_value = METRICS
            result = sim.compare("prd_v1", "prd_v2",
                                 label_a="Round0", label_b="Round1", n_runs=2)
    assert result.variant_a_label == "Round0"
    assert result.variant_b_label == "Round1"


def test_compare_respects_locked_metrics():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    locked = [EvaluationMetric("eng", "scale_1_5", "投入度？")]
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\neng: 4", 100)
        with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_schema:
            result = sim.compare("v1", "v2", n_runs=2, locked_metrics=locked)
    mock_schema.assert_not_called()
    assert "eng" in result.variant_a.metrics
