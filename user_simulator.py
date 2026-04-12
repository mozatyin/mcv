"""UserSimulator — domain-agnostic behavioral simulation engine."""
from __future__ import annotations

import re
import random
from dataclasses import dataclass, field

from mcv.scenarios import ScenarioContext
from mcv.schema_extractor import EvaluationMetric
from mcv.domain_configs import DomainConfig


@dataclass
class SessionResult:
    """One simulated user session — narrative + extracted metric values."""
    scenario: ScenarioContext
    narrative: str
    values: dict[str, str] = field(default_factory=dict)   # {metric_name: raw_string}


def _random_context_for_domain(role: str | None, domain_config: DomainConfig) -> ScenarioContext:
    """Randomize scenario context using DomainConfig options."""
    time_of_day = random.choice(domain_config.time_options)
    emotional_state = random.choice(domain_config.emotional_states)

    if role and role in domain_config.user_roles:
        usage_day = random.choice(domain_config.user_roles[role])
    else:
        all_days = [d for days in domain_config.user_roles.values() for d in days]
        usage_day = random.choice(all_days) if all_days else 1

    trigger = random.choice(domain_config.triggers)
    return ScenarioContext(
        time_of_day=time_of_day,
        emotional_state=emotional_state,
        usage_day=usage_day,
        trigger=trigger,
    )


def _build_session_prompt(
    user_type: str,
    context: ScenarioContext,
    product: str,
    metrics: list[EvaluationMetric],
    domain_config: DomainConfig,
    screen_id: str | None = None,
) -> str:
    """Build the per-session LLM prompt from user type, context, product, and metrics.

    The prompt instructs third-person behavior narration followed by one answer
    per metric. screen_id optionally focuses the session on a specific wireframe screen.
    """
    product_section = product
    if screen_id:
        product_section = f"[只关注 screen_id='{screen_id}' 的部分]\n{product}"

    metric_lines = "\n".join(
        f"{m.name}: {m.question}"
        + (" (回答 yes 或 no)" if m.type == "bool"
           else " (回答 1-5 的数字)" if m.type == "scale_1_5"
           else " (简短文字)")
        for m in metrics
    )

    return (
        f"你是：{user_type}\n\n"
        f"现在的情况：\n"
        f"  时间：{context.time_of_day.replace('_', ' ')}\n"
        f"  状态：{context.emotional_state}\n"
        f"  触发：{context.trigger.replace('_', ' ')}\n"
        f"  使用天数：{context.usage_day}\n\n"
        f"{domain_config.session_framing}：\n{product_section}\n\n"
        f"叙述你接下来的 6-8 个操作。只写你做了什么，不写感想。\n"
        f"用第三人称叙述行为，比如：\"他点击了...\"，\"他滑过了...\"\n\n"
        f"叙述完毕后，每行回答一个问题：\n{metric_lines}\n"
    )


def _parse_session_output(raw: str, metrics: list[EvaluationMetric]) -> dict[str, str]:
    """Extract metric values from raw session output."""
    values: dict[str, str] = {}
    for metric in metrics:
        pattern = rf"^{re.escape(metric.name)}\s*[:：]\s*(.+)$"
        for line in raw.splitlines():
            m = re.match(pattern, line.strip(), re.IGNORECASE)
            if m:
                values[metric.name] = m.group(1).strip()
                break
    return values
