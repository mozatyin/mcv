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


class UserSimulator:
    """Domain-agnostic behavioral simulation engine.

    Usage:
        sim = UserSimulator("18岁手游玩家", GameDomainConfig, api_key=key)
        sim.prepare(product=prd_text, goal="玩家会在Day-1后回来吗？")
        report = sim.simulate(n_runs=60).report()
    """

    def __init__(self, user_type: str, domain_config: DomainConfig, api_key: str):
        self.user_type = user_type
        self.domain_config = domain_config
        self.api_key = api_key
        self._metrics: list[EvaluationMetric] = []
        self._product: str = ""
        self._screen_id: str | None = None
        self._session_results: list[SessionResult] = []

    def prepare(self, product: str, goal: str, screen_id: str | None = None) -> "UserSimulator":
        """Extract EvaluationSchema from goal. Call before simulate()."""
        from mcv.schema_extractor import extract_evaluation_schema
        self._product = product
        self._screen_id = screen_id
        self._metrics = extract_evaluation_schema(goal, self.api_key)
        return self

    def simulate(self, n_runs: int = 60) -> "UserSimulator":
        """Run N independent sessions at temperature=1.0. Returns self for chaining."""
        import mcv.core as _core
        self._session_results = []
        roles = list(self.domain_config.user_roles.keys())
        for i in range(n_runs):
            role = roles[i % len(roles)] if roles else None
            ctx = _random_context_for_domain(role, self.domain_config)
            prompt = _build_session_prompt(
                user_type=self.user_type,
                context=ctx,
                product=self._product,
                metrics=self._metrics,
                domain_config=self.domain_config,
                screen_id=self._screen_id,
            )
            raw, _ = _core._llm_call(
                prompt,
                self.api_key,
                max_tokens=800,
                temperature=1.0,
                model=_core._haiku_model(self.api_key),
            )
            values = _parse_session_output(raw, self._metrics)
            self._session_results.append(SessionResult(
                scenario=ctx,
                narrative=raw,
                values=values,
            ))
        return self

    def report(self) -> "SimulationReport":
        """Aggregate session results into SimulationReport."""
        from mcv.report import aggregate
        return aggregate(
            self._session_results,
            self._metrics,
            self.user_type,
            self._product[:100],
            api_key=self.api_key,
        )
