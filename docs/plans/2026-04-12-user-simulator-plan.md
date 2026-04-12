# UserSimulator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a domain-agnostic behavioral simulation library — initialize with a user type + domain config, simulate how that user responds to any product, aggregate N runs into an empirical SimulationReport.

**Architecture:** Six new files added to the existing `mcv` package. Existing `PersonaSimulator` and all current code are preserved unchanged. `UserSimulator` is the new primary class: `prepare(product, goal)` extracts `EvaluationSchema` from goal via one LLM call, `simulate(n_runs)` runs N behavioral sessions at temperature=1.0 using Haiku, `report()` aggregates into `SimulationReport`. `DomainConfig` (injected at init) controls the session "world" — game vs app vs web vs custom.

**Tech Stack:** Python 3.11+, `anthropic` SDK, `dataclasses`, `re`, `json`, `collections.defaultdict`. Use `python3 -m pytest` (no venv in mcv). All LLM calls via existing `mcv.core._llm_call`.

---

## Key invariants

```
temperature=1.0      ← hardcoded in simulate(), never 0
model=Haiku          ← for simulation sessions (cheap, fast)
model=Sonnet         ← for schema extraction (one call, reasoning task)
All existing code    ← PersonaSimulator / cache / __main__ unchanged
```

---

## Task 1: DomainConfig dataclass + pre-built configs + random_context_for_domain

**Files:**
- Create: `/Users/michael/mcv/domain_configs.py`
- Modify: `/Users/michael/mcv/scenarios.py` (add `random_context_for_domain`)
- Create: `/Users/michael/mcv/tests/test_domain_configs.py`

**Step 1: Write the failing test**

Create `/Users/michael/mcv/tests/test_domain_configs.py`:

```python
import sys
sys.path.insert(0, '/Users/michael/mcv')

from mcv.domain_configs import DomainConfig, GameDomainConfig, AppDomainConfig, WebDomainConfig
from mcv.scenarios import random_context_for_domain, ScenarioContext


def test_game_domain_config_fields():
    assert GameDomainConfig.session_framing == "你开始了一局游戏"
    assert "competitive" in GameDomainConfig.emotional_states
    assert "Newcomer" in GameDomainConfig.user_roles
    assert "want_to_rank_up" in GameDomainConfig.triggers


def test_app_domain_config_fields():
    assert "stressed" in AppDomainConfig.emotional_states
    assert "Explorer" in AppDomainConfig.user_roles


def test_web_domain_config_fields():
    assert WebDomainConfig.session_framing == "你在刷新闻"
    assert "curious" in WebDomainConfig.emotional_states


def test_custom_domain_config():
    custom = DomainConfig(
        session_framing="你在体验VR",
        emotional_states=["immersed", "curious"],
        triggers=["boredom", "curiosity"],
        time_options=["evening", "weekend"],
        user_roles={"Newbie": [1, 7]},
    )
    assert custom.session_framing == "你在体验VR"
    assert "Newbie" in custom.user_roles


def test_random_context_for_domain_uses_config_options():
    ctx = random_context_for_domain(role="Newcomer", domain_config=GameDomainConfig)
    assert isinstance(ctx, ScenarioContext)
    assert ctx.emotional_state in GameDomainConfig.emotional_states
    assert ctx.trigger in GameDomainConfig.triggers
    assert ctx.usage_day in GameDomainConfig.user_roles["Newcomer"]


def test_random_context_for_domain_produces_variance():
    contexts = [random_context_for_domain(domain_config=GameDomainConfig) for _ in range(20)]
    unique = {(c.time_of_day, c.emotional_state, c.usage_day, c.trigger) for c in contexts}
    assert len(unique) >= 3
```

**Step 2: Run to verify it fails**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_domain_configs.py -v
```
Expected: `ModuleNotFoundError: No module named 'mcv.domain_configs'`

**Step 3: Create `/Users/michael/mcv/domain_configs.py`**

```python
"""Domain configurations for UserSimulator — controls session 'world'."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DomainConfig:
    session_framing: str          # "你开始了一局游戏" / "你打开了 app" / "你在刷新闻"
    emotional_states: list[str]
    triggers: list[str]
    time_options: list[str]
    user_roles: dict[str, list[int]]   # role_name → [usage_day values]


GameDomainConfig = DomainConfig(
    session_framing="你开始了一局游戏",
    emotional_states=["competitive", "casual", "tilted", "bored", "hyped"],
    triggers=["want_to_rank_up", "friend_challenged_me", "kill_time", "daily_login", "revenge_match"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={
        "Newcomer": [1, 3],
        "Casual":   [3, 14],
        "Grinder":  [14, 30],
        "Veteran":  [30],
    },
)

AppDomainConfig = DomainConfig(
    session_framing="你打开了这个 app",
    emotional_states=["stressed", "calm", "bored", "excited", "sad", "anxious"],
    triggers=["habit", "work_stress", "relationship_tension", "boredom", "notification", "curiosity"],
    time_options=["morning_commute", "lunch_break", "evening_wind_down", "night"],
    user_roles={
        "Explorer":  [1, 3],
        "Skeptic":   [3, 7],
        "Habituer":  [14, 30],
        "Advocate":  [30],
    },
)

WebDomainConfig = DomainConfig(
    session_framing="你在刷新闻",
    emotional_states=["curious", "bored", "anxious", "relaxed", "rushed"],
    triggers=["morning_routine", "notification", "kill_time", "topic_interest", "breaking_news"],
    time_options=["morning_commute", "lunch_break", "evening", "late_night"],
    user_roles={
        "Casual":      [1, 7],
        "Regular":     [7, 30],
        "PowerReader": [30],
    },
)
```

**Step 4: Add `random_context_for_domain` to `/Users/michael/mcv/scenarios.py`**

Append to the end of the file (do NOT modify existing `random_context`):

```python
def random_context_for_domain(
    role: str | None = None,
    domain_config=None,
) -> ScenarioContext:
    """Generate scenario context using DomainConfig options.

    Falls back to random_context() if domain_config is None.
    """
    if domain_config is None:
        return random_context(role=role)

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
```

**Step 5: Run tests**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_domain_configs.py -v
```
Expected: 6 PASS

Full suite:
```bash
cd /Users/michael/mcv && python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all 58 + 6 = 64 tests pass

**Step 6: Commit**

```bash
cd /Users/michael/mcv && git add domain_configs.py scenarios.py tests/test_domain_configs.py
git commit -m "feat(mcv): DomainConfig + GameDomainConfig/AppDomainConfig/WebDomainConfig + random_context_for_domain"
```

---

## Task 2: EvaluationMetric + schema_extractor

**Files:**
- Create: `/Users/michael/mcv/schema_extractor.py`
- Create: `/Users/michael/mcv/tests/test_schema_extractor.py`

**Step 1: Write the failing test**

Create `/Users/michael/mcv/tests/test_schema_extractor.py`:

```python
import sys
sys.path.insert(0, '/Users/michael/mcv')

from unittest.mock import patch
from mcv.schema_extractor import EvaluationMetric, extract_evaluation_schema


def test_extract_returns_evaluation_metrics():
    mock_resp = '[{"name": "retention", "type": "bool", "question": "会回来吗？"}, {"name": "engagement", "type": "scale_1_5", "question": "投入程度？"}]'
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = (mock_resp, 200)
        metrics = extract_evaluation_schema("用户会留存吗？", api_key="test")
    assert len(metrics) == 2
    assert isinstance(metrics[0], EvaluationMetric)
    assert metrics[0].name == "retention"
    assert metrics[0].type == "bool"
    assert metrics[1].type == "scale_1_5"


def test_extract_filters_invalid_types():
    mock_resp = '[{"name": "x", "type": "invalid_type", "question": "?"}, {"name": "y", "type": "bool", "question": "yes?"}]'
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = (mock_resp, 200)
        metrics = extract_evaluation_schema("test", api_key="test")
    assert len(metrics) == 1
    assert metrics[0].name == "y"


def test_extract_handles_malformed_json():
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("not json at all", 100)
        metrics = extract_evaluation_schema("test", api_key="test")
    assert metrics == []


def test_extract_uses_sonnet_model():
    mock_resp = '[{"name": "x", "type": "bool", "question": "?"}]'
    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = (mock_resp, 200)
        extract_evaluation_schema("test", api_key="test")
    # schema extraction uses Sonnet (default model), not Haiku
    call_kwargs = mock_llm.call_args[1]
    model = call_kwargs.get("model")
    assert model is None or "haiku" not in str(model).lower()


def test_evaluation_metric_fields():
    m = EvaluationMetric(name="ret", type="bool", question="回来吗？")
    assert m.name == "ret"
    assert m.type == "bool"
    assert m.question == "回来吗？"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_schema_extractor.py -v
```
Expected: `ModuleNotFoundError: No module named 'mcv.schema_extractor'`

**Step 3: Create `/Users/michael/mcv/schema_extractor.py`**

```python
"""EvaluationMetric + schema extraction from goal text."""
from __future__ import annotations

from dataclasses import dataclass

from mcv.core import _llm_call, _safe_json_arr


@dataclass
class EvaluationMetric:
    name: str
    type: str      # "bool" | "scale_1_5" | "text"
    question: str  # asked at end of each simulated session


def extract_evaluation_schema(goal: str, api_key: str) -> list[EvaluationMetric]:
    """Extract 3-6 evaluation metrics from a goal description or PRD text.

    Uses Sonnet (one call) — this is a reasoning task, not simulation.
    """
    prompt = (
        "You are a product analyst. Read the following product goal and extract 3-6 "
        "evaluation metrics that would tell us if the product is succeeding.\n\n"
        f"Goal:\n{goal[:2000]}\n\n"
        "For each metric, decide:\n"
        '- type: "bool" (yes/no), "scale_1_5" (intensity 1-5), or "text" (qualitative)\n'
        "- question: a specific question to ask a simulated user at the end of their session\n\n"
        "Reply with JSON array only:\n"
        '[{"name": "snake_case_name", "type": "bool|scale_1_5|text", "question": "..."}]'
    )
    raw, _ = _llm_call(prompt, api_key, max_tokens=512)
    items = _safe_json_arr(raw)
    metrics = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "")
        typ = item.get("type", "")
        question = item.get("question", "")
        if name and typ in ("bool", "scale_1_5", "text") and question:
            metrics.append(EvaluationMetric(name=name, type=typ, question=question))
    return metrics
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_schema_extractor.py -v
```
Expected: 5 PASS

Full suite:
```bash
cd /Users/michael/mcv && python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all pass

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add schema_extractor.py tests/test_schema_extractor.py
git commit -m "feat(mcv): EvaluationMetric + extract_evaluation_schema — PRD/question → typed metric list"
```

---

## Task 3: SessionResult + prompt builder + parser

**Files:**
- Create: `/Users/michael/mcv/user_simulator.py`
- Create: `/Users/michael/mcv/tests/test_user_simulator_prompt.py`

**Step 1: Write the failing test**

Create `/Users/michael/mcv/tests/test_user_simulator_prompt.py`:

```python
import sys
sys.path.insert(0, '/Users/michael/mcv')

from mcv.user_simulator import SessionResult, _build_session_prompt, _parse_session_output
from mcv.schema_extractor import EvaluationMetric
from mcv.domain_configs import GameDomainConfig
from mcv.scenarios import ScenarioContext

METRICS = [
    EvaluationMetric(name="day1_return", type="bool", question="你想回来吗？"),
    EvaluationMetric(name="engagement", type="scale_1_5", question="投入程度？"),
    EvaluationMetric(name="drop_moment", type="text", question="哪里想退出？"),
]
CTX = ScenarioContext("evening", "competitive", 1, "want_to_rank_up")


def test_build_prompt_contains_user_type():
    prompt = _build_session_prompt("18岁手游玩家", CTX, "一款棋牌游戏", METRICS, GameDomainConfig)
    assert "18岁手游玩家" in prompt


def test_build_prompt_contains_all_metric_names():
    prompt = _build_session_prompt("玩家", CTX, "游戏", METRICS, GameDomainConfig)
    assert "day1_return" in prompt
    assert "engagement" in prompt
    assert "drop_moment" in prompt


def test_build_prompt_contains_session_framing():
    prompt = _build_session_prompt("玩家", CTX, "游戏", METRICS, GameDomainConfig)
    assert "你开始了一局游戏" in prompt


def test_build_prompt_with_screen_id():
    prompt = _build_session_prompt("玩家", CTX, "wireframe", METRICS, GameDomainConfig, screen_id="home_screen")
    assert "home_screen" in prompt


def test_build_prompt_no_opinion_words():
    prompt = _build_session_prompt("玩家", CTX, "游戏", METRICS, GameDomainConfig)
    # prompt instructs behavior narration, must not ask for ratings/scores outside metric lines
    lower = prompt.lower()
    assert "rate this" not in lower
    assert "score the" not in lower


def test_parse_session_output_extracts_values():
    raw = "用户进入了游戏大厅。他点击了快速匹配...\nday1_return: yes\nengagement: 4\ndrop_moment: 教程太长"
    values = _parse_session_output(raw, METRICS)
    assert values["day1_return"] == "yes"
    assert values["engagement"] == "4"
    assert values["drop_moment"] == "教程太长"


def test_parse_session_output_handles_missing():
    raw = "用户进入了游戏大厅，然后退出了。"
    values = _parse_session_output(raw, METRICS)
    assert "day1_return" not in values
    assert "engagement" not in values


def test_session_result_fields():
    sr = SessionResult(
        scenario=CTX,
        narrative="叙述...",
        values={"day1_return": "yes"},
    )
    assert sr.scenario is CTX
    assert sr.values["day1_return"] == "yes"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_user_simulator_prompt.py -v
```
Expected: `ModuleNotFoundError: No module named 'mcv.user_simulator'`

**Step 3: Create `/Users/michael/mcv/user_simulator.py`** (data classes + helpers only — no UserSimulator class yet)

```python
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
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_user_simulator_prompt.py -v
```
Expected: 8 PASS

Full suite:
```bash
cd /Users/michael/mcv && python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all pass

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add user_simulator.py tests/test_user_simulator_prompt.py
git commit -m "feat(mcv): SessionResult + _build_session_prompt + _parse_session_output"
```

---

## Task 4: UserSimulator class — prepare + simulate

**Files:**
- Modify: `/Users/michael/mcv/user_simulator.py` (add UserSimulator class)
- Create: `/Users/michael/mcv/tests/test_user_simulator_core.py`

**Step 1: Write the failing test**

Create `/Users/michael/mcv/tests/test_user_simulator_core.py`:

```python
import sys
sys.path.insert(0, '/Users/michael/mcv')

from unittest.mock import patch, MagicMock
from mcv.user_simulator import UserSimulator
from mcv.domain_configs import GameDomainConfig
from mcv.schema_extractor import EvaluationMetric


def test_prepare_extracts_schema():
    with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_extract:
        mock_extract.return_value = [EvaluationMetric("x", "bool", "?")]
        sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
        sim.prepare(product="游戏描述", goal="用户会留存吗？")
        mock_extract.assert_called_once()
        assert len(sim._metrics) == 1


def test_prepare_returns_self_for_chaining():
    with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_extract:
        mock_extract.return_value = []
        sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
        result = sim.prepare(product="游戏", goal="?")
        assert result is sim


def test_simulate_calls_llm_n_times():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        sim.simulate(n_runs=5)

    assert mock_llm.call_count == 5
    assert len(sim._session_results) == 5


def test_simulate_uses_temperature_1():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        sim.simulate(n_runs=1)

    assert mock_llm.call_args[1]["temperature"] == 1.0


def test_simulate_uses_haiku():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        sim.simulate(n_runs=1)

    model = mock_llm.call_args[1].get("model", "")
    assert "haiku" in model.lower()


def test_simulate_returns_self_for_chaining():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("x", "bool", "?")]
    sim._product = "游戏"

    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("叙述...\nx: yes", 200)
        result = sim.simulate(n_runs=2)

    assert result is sim


def test_simulate_stores_session_results():
    sim = UserSimulator("玩家", GameDomainConfig, api_key="test")
    sim._metrics = [EvaluationMetric("ret", "bool", "回来吗？")]
    sim._product = "游戏"

    with patch("mcv.core._llm_call") as mock_llm:
        mock_llm.return_value = ("他进入了游戏...\nret: yes", 200)
        sim.simulate(n_runs=3)

    from mcv.user_simulator import SessionResult
    assert all(isinstance(r, SessionResult) for r in sim._session_results)
    assert sim._session_results[0].values.get("ret") == "yes"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_user_simulator_core.py -v
```
Expected: `AttributeError` or `ImportError` (UserSimulator not defined yet)

**Step 3: Add UserSimulator class to `/Users/michael/mcv/user_simulator.py`**

Append to the end of the file:

```python
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
        from mcv.core import _llm_call, _haiku_model
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
            raw, _ = _llm_call(
                prompt,
                self.api_key,
                max_tokens=800,
                temperature=1.0,
                model=_haiku_model(self.api_key),
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
```

Note: `SimulationReport` is imported lazily to avoid circular imports (report.py imports from user_simulator.py).

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_user_simulator_core.py -v
```
Expected: 7 PASS

Full suite:
```bash
cd /Users/michael/mcv && python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all pass

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add user_simulator.py tests/test_user_simulator_core.py
git commit -m "feat(mcv): UserSimulator class — prepare() + simulate() with temperature=1.0"
```

---

## Task 5: SimulationReport + aggregation

**Files:**
- Create: `/Users/michael/mcv/report.py`
- Create: `/Users/michael/mcv/tests/test_report.py`

**Step 1: Write the failing test**

Create `/Users/michael/mcv/tests/test_report.py`:

```python
import sys
sys.path.insert(0, '/Users/michael/mcv')

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
```

**Step 2: Run to verify it fails**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_report.py -v
```
Expected: `ModuleNotFoundError: No module named 'mcv.report'`

**Step 3: Create `/Users/michael/mcv/report.py`**

```python
"""SimulationReport — aggregate N SessionResults into empirical distributions."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

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
    key_findings: str = ""


def _aggregate_bool(values: list[str]) -> MetricResult:
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
        m = re.search(r"[1-5]", v)
        if m:
            nums.append(int(m.group()))
    if not nums:
        return MetricResult(name="", type="scale_1_5", mean=0.0, distribution={})
    mean = round(sum(nums) / len(nums), 4)
    dist = {i: round(nums.count(i) / len(nums), 4) for i in range(1, 6) if nums.count(i) > 0}
    return MetricResult(name="", type="scale_1_5", mean=mean, distribution=dist)


def _aggregate_text(values: list[str], api_key: str | None = None) -> MetricResult:
    samples = values[:10]
    themes: list[str] = []
    if api_key and len(values) >= 3:
        from mcv.core import _llm_call
        joined = "\n".join(f"- {v}" for v in values[:30])
        prompt = (
            f"以下是用户反馈列表：\n{joined}\n\n"
            "提取 3-5 个主要主题，用简短短语表示。\n"
            '只输出 JSON 数组：["主题1", "主题2", ...]'
        )
        raw, _ = _llm_call(prompt, api_key, max_tokens=256)
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
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_report.py -v
```
Expected: 10 PASS

Full suite:
```bash
cd /Users/michael/mcv && python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all pass

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add report.py tests/test_report.py
git commit -m "feat(mcv): SimulationReport + aggregate() — bool/scale/text metric aggregation"
```

---

## Task 6: Export new types + smoke test

**Files:**
- Modify: `/Users/michael/mcv/__init__.py`
- Create: `/Users/michael/mcv/tests/test_user_simulator_exports.py`

**Step 1: Write the failing test**

Create `/Users/michael/mcv/tests/test_user_simulator_exports.py`:

```python
import sys
sys.path.insert(0, '/Users/michael/mcv')


def test_user_simulator_importable_from_mcv():
    from mcv import UserSimulator
    assert UserSimulator is not None


def test_domain_configs_importable_from_mcv():
    from mcv import GameDomainConfig, AppDomainConfig, WebDomainConfig, DomainConfig
    assert GameDomainConfig.session_framing == "你开始了一局游戏"


def test_simulation_report_importable_from_mcv():
    from mcv import SimulationReport, MetricResult
    assert SimulationReport is not None


def test_evaluation_metric_importable_from_mcv():
    from mcv import EvaluationMetric
    assert EvaluationMetric is not None


def test_full_pipeline_smoke_test_no_llm():
    """Verify the pipeline wires together without calling LLM."""
    from unittest.mock import patch
    from mcv import UserSimulator, GameDomainConfig
    from mcv.schema_extractor import EvaluationMetric

    with patch("mcv.schema_extractor.extract_evaluation_schema") as mock_schema, \
         patch("mcv.core._llm_call") as mock_llm, \
         patch("mcv.report._aggregate_text") as mock_text:

        mock_schema.return_value = [
            EvaluationMetric("ret", "bool", "回来吗？"),
            EvaluationMetric("eng", "scale_1_5", "投入度？"),
        ]
        mock_llm.return_value = ("他进入游戏...\nret: yes\neng: 4", 200)
        from mcv.report import MetricResult
        mock_text.return_value = MetricResult(name="", type="text", themes=[], samples=[])

        sim = UserSimulator("游戏玩家", GameDomainConfig, api_key="test")
        sim.prepare(product="一款棋牌游戏", goal="用户会留存吗？")
        report = sim.simulate(n_runs=3).report()

    assert report.n_simulations == 3
    assert "ret" in report.metrics
    assert report.metrics["ret"].true_rate == 1.0   # all 3 returned "yes"
    assert abs(report.metrics["eng"].mean - 4.0) < 0.01
```

**Step 2: Run to verify it fails**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_user_simulator_exports.py -v
```
Expected: `ImportError: cannot import name 'UserSimulator' from 'mcv'`

**Step 3: Update `/Users/michael/mcv/__init__.py`**

Read the current file first, then append the new exports:

```python
from mcv.user_simulator import UserSimulator, SessionResult
from mcv.domain_configs import DomainConfig, GameDomainConfig, AppDomainConfig, WebDomainConfig
from mcv.schema_extractor import EvaluationMetric, extract_evaluation_schema
from mcv.report import SimulationReport, MetricResult
```

Also add the new names to `__all__`.

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python3 -m pytest tests/test_user_simulator_exports.py -v
```
Expected: 5 PASS

Full suite:
```bash
cd /Users/michael/mcv && python3 -m pytest tests/ --tb=short 2>&1 | tail -5
```
Expected: all tests pass (58 prior + 6 + 5 + 8 + 7 + 10 + 5 = ~99 total)

**Step 5: Secret scan + commit + push**

```bash
grep -rn "sk-or-v1\|sk-ant\|ghp_\|password=" /Users/michael/mcv/ 2>/dev/null | grep -v ".git/" | grep -v "test_" | grep -v "#" || echo "CLEAN"
```

```bash
cd /Users/michael/mcv && git add __init__.py tests/test_user_simulator_exports.py
git commit -m "feat(mcv): export UserSimulator + DomainConfig + SimulationReport from package root"
```

```bash
cd /Users/michael/mcv && git push https://<GH_TOKEN>@github.com/mozatyin/mcv.git main
```
