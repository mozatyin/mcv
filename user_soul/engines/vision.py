"""VisionEngine — VLM-driven visual taste evaluation.

All evaluation uses pairwise comparison (93% accuracy) rather than
absolute scoring (35% accuracy). Research: MLLM-as-UI-Judge (2025).
"""
from __future__ import annotations

import json
import random
import re

from user_soul.backend import LLMBackend
from user_soul.models import PairwiseResult, ReviewResult

_DEFAULT_DIMENSIONS = ["视觉精致度", "色彩和谐", "信息层级", "专业感"]

_PAIRWISE_PROMPT = """你是一位资深 UI 设计评审专家。

下面有两张产品界面截图：
- 图1（左）= 候选 A
- 图2（右）= 候选 B

请从以下维度逐一对比：
{dimensions}

对每个维度，判断 A 更好、B 更好、还是持平，并给出一句话理由。
最后给出综合判决。

只输出 JSON（不要 markdown）：
{{
  "dimensions": {{
    "维度名": {{"winner": "A"|"B"|"tie", "reason": "..."}},
    ...
  }},
  "overall_winner": "A"|"B"|"tie",
  "overall_reason": "一句话总结"
}}"""

_REVIEW_PROMPT = """你是一位资深 UI 设计评审专家。

审查这张产品界面截图。{context}

检查项：
{checklist}

找出所有视觉问题，评估整体质量。

只输出 JSON（不要 markdown）：
{{
  "issues": [
    {{"severity": "P0"|"P1"|"P2", "dimension": "...", "description": "..."}}
  ],
  "overall_score": "professional"|"acceptable"|"amateur",
  "suggestions": ["具体改进建议", ...]
}}"""


def _safe_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


class VisionEngine:

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def pairwise_compare(self, ours: bytes, theirs: bytes, *,
                         dimensions: list[str] | None = None) -> PairwiseResult:
        dims = dimensions or _DEFAULT_DIMENSIONS
        dim_str = "\n".join(f"- {d}" for d in dims)
        prompt = _PAIRWISE_PROMPT.format(dimensions=dim_str)

        # Randomize order to mitigate position bias (~5%)
        if random.random() < 0.5:
            images = [ours, theirs]
            ours_label, theirs_label = "A", "B"
        else:
            images = [theirs, ours]
            ours_label, theirs_label = "B", "A"

        raw = self._backend.vision(prompt, images, max_tokens=800, model_tier="smart")
        return self._parse_pairwise(raw, ours_label, theirs_label)

    def batch_compare(self, ours: bytes,
                      competitors: list[tuple[str, bytes]], *,
                      dimensions: list[str] | None = None) -> list[PairwiseResult]:
        return [
            self.pairwise_compare(ours, comp_img, dimensions=dimensions)
            for _, comp_img in competitors
        ]

    def screenshot_review(self, screenshot: bytes, *,
                          context: str = "",
                          checklist: list[str] | None = None) -> ReviewResult:
        checks = checklist or ["布局合理", "文字可读", "色彩和谐", "无渲染错误"]
        check_str = "\n".join(f"- {c}" for c in checks)
        ctx = f"产品背景：{context}" if context else ""
        prompt = _REVIEW_PROMPT.format(context=ctx, checklist=check_str)
        raw = self._backend.vision(prompt, [screenshot], max_tokens=600, model_tier="smart")
        return self._parse_review(raw)

    def _parse_pairwise(self, raw: str, ours_label: str,
                        theirs_label: str) -> PairwiseResult:
        data = _safe_json(raw)
        dim_results = {}
        for dim_name, dim_data in data.get("dimensions", {}).items():
            if not isinstance(dim_data, dict):
                continue
            raw_winner = dim_data.get("winner", "tie")
            if raw_winner == ours_label:
                winner = "ours"
            elif raw_winner == theirs_label:
                winner = "theirs"
            else:
                winner = "tie"
            dim_results[dim_name] = {
                "winner": winner,
                "reason": dim_data.get("reason", ""),
            }
        raw_overall = data.get("overall_winner", "tie")
        if raw_overall == ours_label:
            overall = "ours"
        elif raw_overall == theirs_label:
            overall = "theirs"
        else:
            overall = "tie"

        ours_wins = sum(1 for d in dim_results.values() if d["winner"] == "ours")
        total = max(len(dim_results), 1)
        confidence = round(max(ours_wins, total - ours_wins) / total, 4)

        return PairwiseResult(
            winner=overall,
            dimension_results=dim_results,
            overall_reason=data.get("overall_reason", ""),
            confidence=confidence,
        )

    def _parse_review(self, raw: str) -> ReviewResult:
        data = _safe_json(raw)
        issues = []
        for item in data.get("issues", []):
            if isinstance(item, dict):
                issues.append({
                    "severity": item.get("severity", "P1"),
                    "dimension": item.get("dimension", ""),
                    "description": item.get("description", ""),
                })
        return ReviewResult(
            issues=issues,
            overall_score=data.get("overall_score", "acceptable"),
            suggestions=data.get("suggestions", []),
        )
