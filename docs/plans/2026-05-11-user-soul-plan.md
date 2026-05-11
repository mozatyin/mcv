# User-Soul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure MCV into User-Soul — a 4-stage user simulation system with model-agnostic LLM backend and new VisionEngine capability.

**Architecture:** Layered: LLMBackend Protocol → 4 Engines (Persona, Behavior, Vision, Vote) → 4 Stage Orchestrators (Research, DesignReview, ModuleUAT, LaunchGate) → UserSoulClient. All code under `user_soul/` package, old `mcv/` deleted after migration.

**Tech Stack:** Python 3.11+, anthropic SDK (in backend only), pytest. No new dependencies.

**Design doc:** `docs/plans/2026-05-11-user-soul-design.md`

**Baseline:** 210 tests passing, 6 skipped. MCV has no pyproject.toml — imports via sys.path.

---

## Phase 1: Foundation

### Task 1: Create directory structure + LLMBackend Protocol

**Files:**
- Create: `user_soul/__init__.py`
- Create: `user_soul/backend.py`
- Create: `user_soul/backends/__init__.py`
- Create: `user_soul/engines/__init__.py`
- Create: `user_soul/stages/__init__.py`
- Test: `tests/test_backend.py`

**Step 1: Create directory structure**

```bash
cd ~/mcv
mkdir -p user_soul/backends user_soul/engines user_soul/stages
touch user_soul/__init__.py user_soul/backends/__init__.py
touch user_soul/engines/__init__.py user_soul/stages/__init__.py
```

**Step 2: Write failing test for LLMBackend Protocol**

```python
# tests/test_backend.py
from user_soul.backend import LLMBackend


def test_protocol_has_text_method():
    """LLMBackend Protocol defines text()."""
    assert hasattr(LLMBackend, 'text')


def test_protocol_has_vision_method():
    """LLMBackend Protocol defines vision()."""
    assert hasattr(LLMBackend, 'vision')


class _StubBackend:
    def text(self, prompt, *, max_tokens=512, temperature=0.0,
             model_tier="fast"):
        return "stub"

    def vision(self, prompt, images, *, max_tokens=512,
               temperature=0.0, model_tier="smart"):
        return "stub"


def test_stub_satisfies_protocol():
    """A class with text() and vision() satisfies LLMBackend."""
    from typing import runtime_checkable
    backend: LLMBackend = _StubBackend()
    assert isinstance(backend, LLMBackend)
```

**Step 3: Run test to verify it fails**

Run: `cd ~/mcv && python3 -m pytest tests/test_backend.py -v`
Expected: FAIL (ModuleNotFoundError)

**Step 4: Write LLMBackend Protocol**

```python
# user_soul/backend.py
"""LLMBackend — model-agnostic LLM interface.

User-Soul never imports a specific SDK. All LLM access goes through this protocol.
Callers provide a concrete implementation (e.g. AnthropicBackend).
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):

    def text(self, prompt: str, *,
             max_tokens: int = 512,
             temperature: float = 0.0,
             model_tier: str = "fast") -> str:
        """Generate text. model_tier: 'fast' (Haiku-class) or 'smart' (Sonnet-class)."""
        ...

    def vision(self, prompt: str, images: list[bytes], *,
               max_tokens: int = 512,
               temperature: float = 0.0,
               model_tier: str = "smart") -> str:
        """Generate text from prompt + images. images: list of raw PNG/JPEG bytes."""
        ...
```

**Step 5: Run test to verify it passes**

Run: `cd ~/mcv && python3 -m pytest tests/test_backend.py -v`
Expected: 3 PASS

**Step 6: Commit**

```bash
cd ~/mcv && git add user_soul/ tests/test_backend.py
git commit -m "feat(user-soul): LLMBackend Protocol — model-agnostic foundation"
```

---

### Task 2: AnthropicBackend implementation

**Files:**
- Create: `user_soul/backends/anthropic.py`
- Source: `core.py:90-121` (`_llm_call` + `_model_name` + `_haiku_model` + `_resolve_local_address`)
- Test: `tests/test_anthropic_backend.py`

**Step 1: Write failing test**

```python
# tests/test_anthropic_backend.py
from unittest.mock import patch, MagicMock
from user_soul.backends.anthropic import AnthropicBackend
from user_soul.backend import LLMBackend


def test_satisfies_protocol():
    backend = AnthropicBackend(api_key="sk-test")
    assert isinstance(backend, LLMBackend)


def test_text_calls_anthropic(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="hello")],
        usage=MagicMock(input_tokens=10, output_tokens=5),
    )
    monkeypatch.setattr("user_soul.backends.anthropic.anthropic.Anthropic",
                        lambda **kw: mock_client)
    backend = AnthropicBackend(api_key="sk-test")
    result = backend.text("prompt", model_tier="fast")
    assert result == "hello"
    mock_client.messages.create.assert_called_once()


def test_model_tier_fast_uses_haiku():
    backend = AnthropicBackend(api_key="sk-test")
    assert "haiku" in backend._resolve_model("fast")


def test_model_tier_smart_uses_sonnet():
    backend = AnthropicBackend(api_key="sk-test")
    assert "sonnet" in backend._resolve_model("smart")


def test_vision_builds_image_content(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.create.return_value = MagicMock(
        content=[MagicMock(text="looks good")],
        usage=MagicMock(input_tokens=100, output_tokens=10),
    )
    monkeypatch.setattr("user_soul.backends.anthropic.anthropic.Anthropic",
                        lambda **kw: mock_client)
    backend = AnthropicBackend(api_key="sk-test")
    result = backend.vision("describe", [b"\x89PNG fake"])
    assert result == "looks good"
    call_kwargs = mock_client.messages.create.call_args[1]
    messages = call_kwargs["messages"]
    content = messages[0]["content"]
    assert any(c["type"] == "image" for c in content)
```

**Step 2: Run test to verify it fails**

Run: `cd ~/mcv && python3 -m pytest tests/test_anthropic_backend.py -v`
Expected: FAIL

**Step 3: Write AnthropicBackend**

Extract from `core.py:31-121`. Key changes:
- `_llm_call()` → `text()` method
- Add `vision()` method with base64 image content blocks
- `_model_name()` / `_haiku_model()` → `_resolve_model(tier)`
- Keep `_resolve_local_address()` for OpenRouter VPN fix

```python
# user_soul/backends/anthropic.py
"""AnthropicBackend — Anthropic SDK implementation of LLMBackend."""
from __future__ import annotations
import base64

import anthropic


_LOCAL_ADDRESS_CACHE: str | None = None


def _resolve_local_address(target_host: str, port: int = 443) -> str | None:
    """Probe for valid local address (VPN workaround). Cached per process."""
    global _LOCAL_ADDRESS_CACHE
    if _LOCAL_ADDRESS_CACHE is not None:
        return _LOCAL_ADDRESS_CACHE or None
    import socket as _socket
    try:
        candidates = [
            info[4][0]
            for info in _socket.getaddrinfo(
                _socket.gethostname(), None, _socket.AF_INET, _socket.SOCK_STREAM
            )
            if not info[4][0].startswith("127.")
        ]
    except Exception:
        candidates = []
    for addr in candidates:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.bind((addr, 0))
            s.connect((target_host, port))
            s.close()
            _LOCAL_ADDRESS_CACHE = addr
            return addr
        except Exception:
            try:
                s.close()
            except Exception:
                pass
    _LOCAL_ADDRESS_CACHE = ""
    return None


class AnthropicBackend:
    """LLMBackend implementation using Anthropic SDK.

    Supports both direct Anthropic API and OpenRouter (sk-or- prefix).
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _resolve_model(self, tier: str) -> str:
        if tier == "fast":
            return "claude-haiku-4-5-20251001"
        return "claude-sonnet-4-20250514"

    def _make_client(self) -> anthropic.Anthropic:
        if self._api_key.startswith("sk-or-"):
            import httpx
            _local = _resolve_local_address("104.18.3.115")
            _transport = httpx.HTTPTransport(local_address=_local) if _local else None
            return anthropic.Anthropic(
                api_key=self._api_key,
                base_url="https://openrouter.ai/api",
                http_client=httpx.Client(transport=_transport) if _transport else None,
            )
        return anthropic.Anthropic(api_key=self._api_key)

    def text(self, prompt: str, *, max_tokens: int = 512,
             temperature: float = 0.0, model_tier: str = "fast") -> str:
        client = self._make_client()
        kwargs: dict = dict(
            model=self._resolve_model(model_tier),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if temperature > 0.0:
            kwargs["temperature"] = temperature
        resp = client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""

    def vision(self, prompt: str, images: list[bytes], *,
               max_tokens: int = 512, temperature: float = 0.0,
               model_tier: str = "smart") -> str:
        client = self._make_client()
        content: list[dict] = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode(),
                },
            })
        content.append({"type": "text", "text": prompt})
        kwargs: dict = dict(
            model=self._resolve_model(model_tier),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if temperature > 0.0:
            kwargs["temperature"] = temperature
        resp = client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""
```

**Step 4: Run test to verify it passes**

Run: `cd ~/mcv && python3 -m pytest tests/test_anthropic_backend.py -v`
Expected: 5 PASS

**Step 5: Commit**

```bash
cd ~/mcv && git add user_soul/backends/anthropic.py tests/test_anthropic_backend.py
git commit -m "feat(user-soul): AnthropicBackend — text() + vision() with OpenRouter support"
```

---

### Task 3: Models + calibration + framework

**Files:**
- Create: `user_soul/models.py`
- Create: `user_soul/calibration.py`
- Create: `user_soul/framework.py`
- Source: `population.py:1-67` (dataclasses), `report.py:12-155` (dataclasses), `journey.py:24-49`, `core.py:1-28`, `behavioral_framework.py` (all), `schema_extractor.py:1-20`
- Test: `tests/test_models.py`

**Step 1: Write failing test**

```python
# tests/test_models.py
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
```

**Step 2: Run to verify fail, then implement**

Collect ALL dataclasses from existing code into `user_soul/models.py`. Add the new types (PairwiseResult, ReviewResult, ResearchReport, DesignReviewReport, ModuleUATReport, LaunchReport).

Copy `behavioral_framework.py` → `user_soul/framework.py` (no changes).

Create `user_soul/calibration.py`:

```python
# user_soul/calibration.py
"""Calibration constants and helpers for sycophancy correction."""

SYCOPHANCY_DEFLATOR: float = 0.70

def deflate(rate: float) -> float:
    return round(rate * SYCOPHANCY_DEFLATOR, 4)
```

**Step 3: Run tests, commit**

Run: `cd ~/mcv && python3 -m pytest tests/test_models.py -v`
Expected: 5 PASS

```bash
cd ~/mcv && git add user_soul/models.py user_soul/calibration.py user_soul/framework.py tests/test_models.py
git commit -m "feat(user-soul): models + calibration + behavioral framework"
```

---

## Phase 2: Core Engines

### Task 4: PersonaEngine

**Files:**
- Create: `user_soul/engines/persona.py`
- Source: `population.py` (PopulationResearcher + PersonaPool)
- Test: `tests/test_persona_engine.py`

**Step 1: Write failing test**

```python
# tests/test_persona_engine.py
from user_soul.engines.persona import PersonaEngine
from user_soul.models import PersonaStructure, AgentProfile


class _FakeBackend:
    def text(self, prompt, **kw):
        return '{"population_label":"gamers","product_context":"chess","trait_dimensions":[{"name":"skill","description":"d","low_label":"l","high_label":"h","distribution":"normal","mean":5,"std":2,"source":"space1"}],"archetypes":[{"name":"Casual","frequency":0.6,"description":"casual player","background_story":"小明，25岁","trait_constraints":{"skill":[1,5]}},{"name":"Hardcore","frequency":0.4,"description":"serious player","background_story":"老王，40岁","trait_constraints":{"skill":[6,10]}}],"research_notes":"test"}'

    def vision(self, prompt, images, **kw):
        return ""


def test_research_returns_persona_structure():
    engine = PersonaEngine(_FakeBackend())
    result = engine.research("chess game")
    assert isinstance(result, PersonaStructure)
    assert len(result.archetypes) == 2


def test_generate_pool_returns_agent_profiles():
    engine = PersonaEngine(_FakeBackend())
    structure = engine.research("chess game")
    pool = engine.generate_pool(structure, n=5)
    assert len(pool) == 5
    assert all(isinstance(a, AgentProfile) for a in pool)


def test_get_or_create_convenience():
    engine = PersonaEngine(_FakeBackend())
    pool = engine.get_or_create("chess game", n=3)
    assert len(pool) == 3


def test_pool_respects_archetype_constraints():
    engine = PersonaEngine(_FakeBackend())
    structure = engine.research("chess game")
    pool = engine.generate_pool(structure, n=20)
    for agent in pool:
        if agent.archetype_name == "Casual":
            assert agent.trait_vector["skill"] <= 5.0
        elif agent.archetype_name == "Hardcore":
            assert agent.trait_vector["skill"] >= 6.0
```

**Step 2: Implement PersonaEngine**

Migrate from `population.py`. Key change: replace `_core._llm_call(prompt, self._api_key, ...)` with `self._backend.text(prompt, ...)`.

```python
# user_soul/engines/persona.py
"""PersonaEngine — population research + persona pool generation."""
from __future__ import annotations
import random
import json
import re
from user_soul.backend import LLMBackend
from user_soul.models import (
    TraitDimension, Archetype, PersonaStructure, AgentProfile,
)

_RESEARCH_PROMPT = """..."""  # ← copy from population.py:69-114

class PersonaEngine:
    def __init__(self, backend: LLMBackend):
        self._backend = backend
        self._counter = 0

    def research(self, product_description: str) -> PersonaStructure:
        prompt = _RESEARCH_PROMPT.format(product_description=product_description[:2000])
        raw = self._backend.text(prompt, max_tokens=2048, model_tier="smart")
        return self._parse(raw, product_description)

    def generate_pool(self, structure: PersonaStructure, n: int) -> list[AgentProfile]:
        # ← copy from PersonaPool.generate() in population.py:229-265
        ...

    def get_or_create(self, product_description: str, n: int = 12) -> list[AgentProfile]:
        structure = self.research(product_description)
        return self.generate_pool(structure, n)

    def _parse(self, raw: str, product_description: str) -> PersonaStructure:
        # ← copy from PopulationResearcher._parse() in population.py:135-192
        ...

    def _fallback(self, product_description: str) -> PersonaStructure:
        # ← copy from PopulationResearcher._fallback() in population.py:194-214
        ...
```

**Step 3: Run test, commit**

Run: `cd ~/mcv && python3 -m pytest tests/test_persona_engine.py -v`

```bash
cd ~/mcv && git add user_soul/engines/persona.py tests/test_persona_engine.py
git commit -m "feat(user-soul): PersonaEngine — population research + pool generation"
```

---

### Task 5: VoteEngine

**Files:**
- Create: `user_soul/engines/vote.py`
- Source: `core.py:162-553` (PersonaDecider), `client.py:195-300` (research_aarrr)
- Test: `tests/test_vote_engine.py`

**Step 1: Write failing test**

```python
# tests/test_vote_engine.py
from user_soul.engines.vote import VoteEngine
from user_soul.models import AgentProfile, DecisionResult, FeatureAAR


class _FakeBackend:
    def text(self, prompt, **kw):
        return '{"choice": "Must-Have", "reasoning": "core feature"}'
    def vision(self, prompt, images, **kw):
        return ""


def _make_persona():
    return AgentProfile(
        agent_id="a1", archetype_name="Casual",
        trait_vector={"skill": 3.0}, background_story="小明，25岁学生")


def test_classify_returns_decision():
    engine = VoteEngine(_FakeBackend())
    result = engine.classify(
        question="Kano category?",
        options=["Must-Have", "Delighter"],
        context="chess game", personas=[_make_persona()])
    assert isinstance(result, DecisionResult)
    assert result.value == "Must-Have"


def test_score_returns_number():
    class _ScoreBackend:
        def text(self, prompt, **kw):
            return '{"score": 7.5, "reasoning": "good"}'
        def vision(self, prompt, images, **kw):
            return ""
    engine = VoteEngine(_ScoreBackend())
    result = engine.score("How important?", 0, 10, "chess", [_make_persona()])
    assert isinstance(result.value, float)
    assert 0 <= result.value <= 10
```

**Step 2: Implement VoteEngine**

Migrate PersonaDecider from `core.py`. Replace `_llm_call(prompt, self.api_key, ...)` with `self._backend.text(prompt, ...)`. Keep fast/validated mode logic.

Also migrate `research_aarrr()` from `client.py:195-300`.

**Step 3: Run test, commit**

```bash
cd ~/mcv && git add user_soul/engines/vote.py tests/test_vote_engine.py
git commit -m "feat(user-soul): VoteEngine — classify + score + aarrr"
```

---

### Task 6: BehaviorEngine

**Files:**
- Create: `user_soul/engines/behavior.py`
- Source: `user_simulator.py`, `journey.py`, `report.py`, `schema_extractor.py`, `scenarios.py`, `domain_configs.py`
- Test: `tests/test_behavior_engine.py`

This is the largest engine. It bundles: UserSimulator + Journey + Report + SchemaExtractor + Scenarios + DomainConfigs.

**Step 1: Write failing test**

```python
# tests/test_behavior_engine.py
from user_soul.engines.behavior import BehaviorEngine
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
        EvaluationMetric("day_1_return_intent", "bool",
                         "Will user return tomorrow?"),
        EvaluationMetric("friction_points", "text",
                         "Key friction points?"),
    ]


def test_simulate_returns_report():
    engine = BehaviorEngine(_FakeBackend())
    report = engine.simulate("chess game", _make_pool(), _make_metrics(), n_runs=3)
    assert isinstance(report, SimulationReport)
    assert report.n_simulations == 3


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
```

**Step 2: Implement BehaviorEngine**

This is a larger migration. Internal modules (scenarios, domain_configs, schema_extractor, report aggregation) become private helpers inside `engines/behavior.py` or imported from sub-modules.

Key change throughout: `_core._llm_call(prompt, api_key, ...)` → `self._backend.text(prompt, ...)`.

The `_build_session_prompt`, `_parse_session_output`, aggregate functions, Wilson CI — all move as-is.

**Step 3: Run test, commit**

```bash
cd ~/mcv && git add user_soul/engines/behavior.py tests/test_behavior_engine.py
git commit -m "feat(user-soul): BehaviorEngine — simulate + compare + journey"
```

---

### Task 7: VisionEngine (NEW CODE)

**Files:**
- Create: `user_soul/engines/vision.py`
- Test: `tests/test_vision_engine.py`

**Step 1: Write failing test**

```python
# tests/test_vision_engine.py
import json
from user_soul.engines.vision import VisionEngine
from user_soul.models import PairwiseResult, ReviewResult


class _FakeVisionBackend:
    def text(self, prompt, **kw):
        return ""
    def vision(self, prompt, images, **kw):
        return json.dumps({
            "dimensions": {
                "视觉精致度": {"winner": "ours", "reason": "cleaner layout"},
                "色彩和谐": {"winner": "theirs", "reason": "better contrast"},
                "信息层级": {"winner": "tie", "reason": "similar"},
                "专业感": {"winner": "ours", "reason": "more polished"},
            },
            "overall_winner": "ours",
            "overall_reason": "Better polish despite weaker colors",
        })


def test_pairwise_compare_returns_result():
    engine = VisionEngine(_FakeVisionBackend())
    result = engine.pairwise_compare(b"png_ours", b"png_theirs")
    assert isinstance(result, PairwiseResult)
    assert result.winner in ("ours", "theirs", "tie")


def test_pairwise_dimension_results():
    engine = VisionEngine(_FakeVisionBackend())
    result = engine.pairwise_compare(b"png_ours", b"png_theirs")
    assert "视觉精致度" in result.dimension_results
    assert result.dimension_results["视觉精致度"]["winner"] == "ours"


def test_pairwise_confidence_from_dimension_agreement():
    engine = VisionEngine(_FakeVisionBackend())
    result = engine.pairwise_compare(b"png_ours", b"png_theirs")
    assert 0.0 <= result.confidence <= 1.0


def test_batch_compare():
    engine = VisionEngine(_FakeVisionBackend())
    results = engine.batch_compare(
        b"png_ours",
        [("comp_a", b"png_a"), ("comp_b", b"png_b")])
    assert len(results) == 2
    assert all(isinstance(r, PairwiseResult) for r in results)


class _FakeReviewBackend:
    def text(self, prompt, **kw):
        return ""
    def vision(self, prompt, images, **kw):
        return json.dumps({
            "issues": [
                {"severity": "P1", "dimension": "布局", "description": "text overflow"}
            ],
            "overall_score": "acceptable",
            "suggestions": ["Fix text overflow in header"],
        })


def test_screenshot_review():
    engine = VisionEngine(_FakeReviewBackend())
    result = engine.screenshot_review(b"png_data", context="chess game")
    assert isinstance(result, ReviewResult)
    assert result.overall_score == "acceptable"
    assert len(result.issues) == 1
```

**Step 2: Implement VisionEngine**

```python
# user_soul/engines/vision.py
"""VisionEngine — VLM-driven visual taste evaluation.

Core new capability of User-Soul. All evaluation uses pairwise comparison
(93% accuracy) rather than absolute scoring (35% accuracy).

Research: MLLM-as-UI-Judge (2025), AesEval-Bench (Microsoft 2026).
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
        data = self._safe_json(raw)
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
        data = self._safe_json(raw)
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

    @staticmethod
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
```

**Step 3: Run test, commit**

Run: `cd ~/mcv && python3 -m pytest tests/test_vision_engine.py -v`
Expected: 6 PASS

```bash
cd ~/mcv && git add user_soul/engines/vision.py tests/test_vision_engine.py
git commit -m "feat(user-soul): VisionEngine — pairwise compare + screenshot review"
```

---

## Phase 3: Stage Orchestrators

### Task 8: ResearchPanel (S1)

**Files:**
- Create: `user_soul/stages/research.py`
- Test: `tests/test_research_stage.py`

Compose PersonaEngine + VisionEngine + VoteEngine.
Implement `run()` as described in design Section 3.
Test with fake backends that return canned JSON.

```bash
git commit -m "feat(user-soul): S1 ResearchPanel — focus group orchestrator"
```

### Task 9: DesignReview (S2)

**Files:**
- Create: `user_soul/stages/design_review.py`
- Test: `tests/test_design_review_stage.py`

Compose PersonaEngine + BehaviorEngine + VisionEngine.
Gate: `passes_gate = journey.completion_rate >= 0.70`.

```bash
git commit -m "feat(user-soul): S2 DesignReview — wireframe usability test"
```

### Task 10: ModuleUAT (S4)

**Files:**
- Create: `user_soul/stages/module_uat.py`
- Test: `tests/test_module_uat_stage.py`

Compose PersonaEngine + BehaviorEngine + VisionEngine.
Gate: no P0 frictions.

```bash
git commit -m "feat(user-soul): S4 ModuleUAT — functional acceptance testing"
```

### Task 11: LaunchGate (S5)

**Files:**
- Create: `user_soul/stages/launch.py`
- Test: `tests/test_launch_stage.py`

Compose PersonaEngine + BehaviorEngine + VisionEngine.
`_judge()` logic: taste_win_rate + day1_return_adjusted → SHIP/IMPROVE/ABANDON.
Human confirms — recommendation only.

```bash
git commit -m "feat(user-soul): S5 LaunchGate — pre-launch taste validation"
```

---

## Phase 4: Integration

### Task 12: UserSoulClient + package __init__

**Files:**
- Create: `user_soul/client.py`
- Modify: `user_soul/__init__.py`
- Test: `tests/test_user_soul_client.py`

```python
# user_soul/client.py
class UserSoulClient:
    def __init__(self, backend: LLMBackend):
        self._persona = PersonaEngine(backend)
        self._behavior = BehaviorEngine(backend)
        self._vision = VisionEngine(backend)
        self._vote = VoteEngine(backend)

    def research(self, **kwargs) -> ResearchReport: ...
    def review(self, **kwargs) -> DesignReviewReport: ...
    def verify(self, **kwargs) -> ModuleUATReport: ...
    def launch(self, **kwargs) -> LaunchReport: ...
    def create_persona_pool(self, product_description, n=12) -> list[AgentProfile]: ...
```

Test: call each method with fake backend, verify correct Report type returned.

```bash
git commit -m "feat(user-soul): UserSoulClient — unified entry point"
```

### Task 13: PM-Soul adaptation

**Files:**
- Modify: `~/pm-soul/pm_soul/mcv_bridge.py`

Replace:
```python
sys.path.insert(0, "/Users/michael/mcv")
from mcv import MCVClient
from mcv.schema_extractor import EvaluationMetric
```
With:
```python
from user_soul import UserSoulClient
from user_soul.backends.anthropic import AnthropicBackend
from user_soul.models import EvaluationMetric
```

Update `_mcv_validate_prd()` and `_mcv_compare_prd_versions()` to use `UserSoulClient.verify()`.

Run PM-Soul tests: `cd ~/pm-soul && python3 -m pytest tests/test_mcv_bridge.py -v`

```bash
cd ~/pm-soul && git commit -m "refactor: migrate mcv_bridge to user-soul"
```

### Task 14: ELTM adaptation

**Files:**
- Modify: `~/eltm/eltm/__init__.py`
- Modify: `~/eltm/eltm/stages/battle.py`
- Modify: `~/eltm/eltm/synthesize/feature_intelligence.py`
- Modify: `~/eltm/tests/test_battle_honest.py`
- Modify: `~/eltm/tests/test_simulation_integration.py`

Replace all `from mcv import ...` with `from user_soul import ...` / `from user_soul.models import ...`.

Run ELTM tests: `cd ~/eltm && python3 -m pytest tests/ -v`

```bash
cd ~/eltm && git commit -m "refactor: migrate mcv imports to user-soul"
```

---

## Phase 5: Cleanup

### Task 15: Delete old MCV files + verify

**Files to delete:**
- `__init__.py` (old mcv package root)
- `__main__.py`
- `cache.py`
- `client.py` (old MCVClient)
- `core.py`
- `domain_configs.py`
- `gate_ledger.py`
- `journey.py`
- `personas.py`
- `population.py`
- `report.py`
- `scenarios.py`
- `schema_extractor.py`
- `simulator.py`
- `user_simulator.py`
- `behavioral_framework.py`

**Step 1: Delete old files**

```bash
cd ~/mcv
rm -f __init__.py __main__.py cache.py client.py core.py domain_configs.py
rm -f gate_ledger.py journey.py personas.py population.py report.py
rm -f scenarios.py schema_extractor.py simulator.py user_simulator.py
rm -f behavioral_framework.py
```

**Step 2: Delete old tests**

```bash
cd ~/mcv
rm -f tests/test_backend.py  # keep — it's for user_soul
# Delete all tests that test old mcv code:
rm -f tests/test_cache.py tests/test_classify_fast.py tests/test_client.py
rm -f tests/test_compare_report.py tests/test_domain_configs.py
rm -f tests/test_env_mode.py tests/test_import.py tests/test_main_cli.py
rm -f tests/test_personas.py tests/test_population.py tests/test_score_fast.py
rm -f tests/test_simulate_aggregate.py tests/test_simulate_one.py
rm -f tests/test_simulator_types.py tests/test_user_simulator_compare.py
rm -f tests/test_user_simulator_core.py tests/test_user_simulator_exports.py
rm -f tests/test_user_simulator_prompt.py tests/test_validate_fast.py
rm -f tests/test_validated_mode.py tests/test_scenarios.py
rm -f tests/test_schema_extractor.py tests/test_attribute_frictions.py
rm -f tests/test_behavioral_framework.py tests/test_bridge_smoke.py
rm -f tests/test_journey.py tests/test_report.py
rm -f tests/test_research_aarrr.py tests/test_validate_coherence.py
```

**Step 3: Run all user_soul tests**

```bash
cd ~/mcv && python3 -m pytest tests/ -v
```

Expected: all user_soul tests PASS, zero old mcv tests remain.

**Step 4: Commit**

```bash
cd ~/mcv && git add -A
git commit -m "chore: remove old mcv code — user_soul is the sole package"
```

### Task 16: Rename repo directory

```bash
cd ~ && mv mcv user-soul
```

Update `sys.path` references in PM-Soul and ELTM if they use path-based imports.
Update git remote if desired: `gh repo rename user-soul` (ask user first).

```bash
cd ~/user-soul && git add -A
git commit -m "chore: rename repository mcv → user-soul"
```

---

## Summary

| Phase | Tasks | New files | Tests |
|-------|-------|-----------|-------|
| 1: Foundation | 1-3 | backend.py, anthropic.py, models.py, calibration.py, framework.py | ~15 |
| 2: Engines | 4-7 | persona.py, vote.py, behavior.py, vision.py | ~25 |
| 3: Stages | 8-11 | research.py, design_review.py, module_uat.py, launch.py | ~20 |
| 4: Integration | 12-14 | client.py, pm-soul + eltm patches | ~10 |
| 5: Cleanup | 15-16 | delete old, rename | 0 |
| **Total** | **16 tasks** | | **~70 new tests** |
