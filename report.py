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
    # statistical
    stdev: float | None = None
    ci_95_low: float | None = None
    ci_95_high: float | None = None
    n_samples: int = 0


@dataclass
class SimulationReport:
    n_simulations: int
    user_type: str
    product_summary: str
    metrics: dict[str, MetricResult]
    key_findings: str = ""  # populated externally if needed


def _wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion p with n samples."""
    if n == 0:
        return 0.0, 0.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4)


def _aggregate_bool(values: list[str]) -> MetricResult:
    if not values:
        return MetricResult(name="", type="bool", true_rate=0.0,
                            stdev=0.0, ci_95_low=0.0, ci_95_high=0.0, n_samples=0)
    true_count = sum(
        1 for v in values
        if v.lower().strip() in ("yes", "true", "1", "是", "会", "会的", "y")
    )
    n = len(values)
    p = round(true_count / n, 4)
    lo, hi = _wilson_ci(p, n)
    import math
    stdev = round(math.sqrt(p * (1 - p)), 4)
    return MetricResult(name="", type="bool", true_rate=p,
                        stdev=stdev, ci_95_low=lo, ci_95_high=hi, n_samples=n)


def _aggregate_scale(values: list[str]) -> MetricResult:
    import statistics, math
    nums = []
    for v in values:
        m = re.search(r'(?<!\d)([1-5])(?!\d)', v)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return MetricResult(name="", type="scale_1_5", mean=0.0, distribution={},
                            stdev=0.0, ci_95_low=0.0, ci_95_high=0.0, n_samples=0)
    n = len(nums)
    mean = round(sum(nums) / n, 4)
    dist = {i: round(nums.count(i) / n, 4) for i in range(1, 6) if nums.count(i) > 0}
    stdev = round(statistics.stdev(nums), 4) if n > 1 else 0.0
    margin = round(1.96 * stdev / math.sqrt(n), 4) if n > 1 else 0.0
    return MetricResult(name="", type="scale_1_5", mean=mean, distribution=dist,
                        stdev=stdev,
                        ci_95_low=round(max(1.0, mean - margin), 4),
                        ci_95_high=round(min(5.0, mean + margin), 4),
                        n_samples=n)


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
