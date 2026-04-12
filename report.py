"""SimulationReport — aggregate N SessionResults into empirical distributions."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass

from mcv.schema_extractor import EvaluationMetric
from mcv.user_simulator import SessionResult


@dataclass
class MetricResult:
    name: str
    type: str
    # bool
    true_rate: float | None = None
    # scale_1_5
    mean: float | None = None
    distribution: dict[int, float] | None = None
    # text
    themes: list[str] | None = None
    samples: list[str] | None = None


@dataclass
class SimulationReport:
    n_simulations: int
    user_type: str
    product_summary: str
    metrics: dict[str, MetricResult]
    key_findings: str = ""  # populated externally if needed


def _aggregate_bool(values: list[str]) -> MetricResult:
    # Unrecognized values (e.g. "maybe", "有可能") count as false.
    # At temperature=1.0, hedged answers depress true_rate slightly — this is intentional.
    if not values:
        return MetricResult(name="", type="bool", true_rate=0.0)
    true_count = sum(
        1 for v in values
        if v.lower().strip() in ("yes", "true", "1", "是", "会", "会的", "y")
    )
    return MetricResult(name="", type="bool", true_rate=round(true_count / len(values), 4))


def _aggregate_scale(values: list[str]) -> MetricResult:
    nums = []
    for v in values:
        m = re.search(r'(?<!\d)([1-5])(?!\d)', v)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return MetricResult(name="", type="scale_1_5", mean=0.0, distribution={})
    mean = round(sum(nums) / len(nums), 4)
    dist = {i: round(nums.count(i) / len(nums), 4) for i in range(1, 6) if nums.count(i) > 0}
    return MetricResult(name="", type="scale_1_5", mean=mean, distribution=dist)


def _aggregate_text(values: list[str], api_key: str | None = None) -> MetricResult:
    samples = values[:10]
    themes: list[str] = []
    if api_key and len(values) >= 3:
        import mcv.core as _core
        joined = "\n".join(f"- {v}" for v in values[:30])
        prompt = (
            f"以下是用户反馈列表：\n{joined}\n\n"
            "提取 3-5 个主要主题，用简短短语表示。\n"
            '只输出 JSON 数组：["主题1", "主题2", ...]'
        )
        raw, _ = _core._llm_call(prompt, api_key, max_tokens=256)
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                themes = [str(t) for t in parsed if isinstance(t, str)][:5]
            except (json.JSONDecodeError, ValueError):
                pass
    return MetricResult(name="", type="text", themes=themes, samples=samples)


def aggregate(
    session_results: list[SessionResult],
    metrics: list[EvaluationMetric],
    user_type: str,
    product_summary: str,
    api_key: str | None = None,
) -> SimulationReport:
    """Aggregate N SessionResults → one SimulationReport per metric."""
    metric_values: dict[str, list[str]] = defaultdict(list)
    for sr in session_results:
        for name, value in sr.values.items():
            metric_values[name].append(value)

    results: dict[str, MetricResult] = {}
    for metric in metrics:
        vals = metric_values.get(metric.name, [])
        if metric.type == "bool":
            r = _aggregate_bool(vals)
        elif metric.type == "scale_1_5":
            r = _aggregate_scale(vals)
        else:
            r = _aggregate_text(vals, api_key)
        r.name = metric.name
        results[metric.name] = r

    return SimulationReport(
        n_simulations=len(session_results),
        user_type=user_type,
        product_summary=product_summary,
        metrics=results,
    )
