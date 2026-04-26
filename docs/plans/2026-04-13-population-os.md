# PopulationOS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a population generator that takes a product description, researches the full behavioral spectrum of that product's potential users, and produces a confirmed PersonaStructure that generates N heterogeneous AgentProfiles — each with a frozen trait vector that constrains their behavior during simulation.

**Architecture:** Three research spaces (Product Intent, Behavioral Research, Market Context) each contribute candidate TraitDimensions; a merger produces a PersonaStructure for human confirmation; PersonaPool samples N AgentProfiles from it; UserSimulator gains a `prepare_with_pool()` path that uses one AgentProfile per session instead of round-robin roles. All new code lives in `/Users/michael/mcv/population.py`. UserSimulator integration is additive — existing interface unchanged.

**Tech Stack:** Python 3.9, dataclasses, mcv.core._llm_call (Sonnet for research, Haiku for session simulation), random/numpy-free sampling (pure stdlib), pytest

---

## Context: What Exists

- `mcv/user_simulator.py` — `UserSimulator(user_type: str, domain_config, api_key)` runs N sessions. Each session uses a round-robin role from `domain_config.user_roles` (25/25/25/25 equal split). The `user_type` string goes verbatim into the prompt.
- `mcv/domain_configs.py` — `DomainConfig` is a static config (emotional_states, triggers, time_options, user_roles). Nothing is sampled from a distribution.
- `mcv/schema_extractor.py` — `EvaluationMetric` dataclass, one LLM call to extract metrics from a goal.
- `mcv/core.py` — `_llm_call(prompt, api_key, max_tokens, temperature, model)` is the single LLM primitive. `_haiku_model()` returns haiku model ID. `_model_name()` returns sonnet model ID.

The problem: all 30/60 sessions share the same `user_type` string. Temperature=1.0 adds token noise but not behavioral heterogeneity. We need each session to be a different person with a frozen behavioral profile.

---

## Task 1: Core Data Structures

**Files:**
- Create: `mcv/population.py`
- Test: `mcv/tests/test_population.py`

**Step 1: Write the failing tests**

```python
# mcv/tests/test_population.py
from mcv.population import TraitDimension, Archetype, PersonaStructure, AgentProfile

def test_trait_dimension_fields():
    td = TraitDimension(
        name="ludo_familiarity",
        description="Real-world Ludo experience",
        low_label="never played Ludo before",
        high_label="expert Ludo player who teaches others",
        distribution="bimodal",
        mean=5.0,
        std=2.5,
        source="space2",
    )
    assert td.name == "ludo_familiarity"
    assert td.distribution == "bimodal"
    assert td.source == "space2"

def test_archetype_fields():
    arch = Archetype(
        name="Family Socializer",
        frequency=0.45,
        description="Plays Ludo with family, social motivation dominates",
        trait_constraints={"social_motivation": (6.0, 10.0), "ludo_familiarity": (3.0, 9.0)},
    )
    assert arch.frequency == 0.45
    assert arch.trait_constraints["social_motivation"] == (6.0, 10.0)

def test_persona_structure_fields():
    dims = [
        TraitDimension("social_motivation", "Social drive", "solo player", "social-first", "normal", 6.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Family Socializer", 0.60, "social player", {"social_motivation": (6.0, 10.0)}),
        Archetype("Solo Grinder",       0.40, "solo player",  {"social_motivation": (0.0, 4.0)}),
    ]
    ps = PersonaStructure(
        population_label="Arabic Ludo Players",
        product_context="Ludo mobile app for Arabic families",
        trait_dimensions=dims,
        archetypes=archs,
    )
    assert ps.population_label == "Arabic Ludo Players"
    assert len(ps.archetypes) == 2

def test_agent_profile_fields():
    ap = AgentProfile(
        agent_id="agent_001",
        archetype_name="Family Socializer",
        trait_vector={"social_motivation": 8.3, "ludo_familiarity": 5.1, "patience": 6.2},
    )
    assert ap.agent_id == "agent_001"
    assert ap.trait_vector["social_motivation"] == 8.3
```

**Step 2: Run to verify they fail**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py::test_trait_dimension_fields -v
```
Expected: `ImportError: cannot import name 'TraitDimension' from 'mcv.population'`

**Step 3: Implement the data structures**

```python
# mcv/population.py
"""PopulationOS — research-driven heterogeneous user population generator.

Three spaces merged into PersonaStructure → PersonaPool → AgentProfile per session.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TraitDimension:
    """One behavioral axis that distinguishes users of this product."""
    name: str           # snake_case, e.g. "ludo_familiarity"
    description: str    # human-readable, e.g. "Real-world Ludo experience"
    low_label: str      # behavior at value=0, e.g. "never played Ludo before"
    high_label: str     # behavior at value=10, e.g. "expert Ludo player who teaches others"
    distribution: str   # "normal" | "uniform" | "bimodal" | "right_skewed" | "left_skewed"
    mean: float         # 0-10 scale center
    std: float          # standard deviation on 0-10 scale
    source: str         # "space1" | "space2" | "space3"


@dataclass
class Archetype:
    """A named user cluster with frequency and trait constraints."""
    name: str
    frequency: float                              # 0-1, must sum to 1.0 across all archetypes
    description: str
    trait_constraints: dict[str, tuple[float, float]]  # trait_name → (min_val, max_val)


@dataclass
class PersonaStructure:
    """Complete confirmed population model — output of research, input to PersonaPool."""
    population_label: str
    product_context: str
    trait_dimensions: list[TraitDimension]
    archetypes: list[Archetype]
    research_notes: str = ""


@dataclass
class AgentProfile:
    """Frozen per-session persona sampled from PersonaStructure."""
    agent_id: str
    archetype_name: str
    trait_vector: dict[str, float]    # {trait_name: value (0-10)}
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py -v
```
Expected: 4 PASS

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add mcv/population.py mcv/tests/test_population.py
git commit -m "feat: add TraitDimension, Archetype, PersonaStructure, AgentProfile dataclasses"
```

---

## Task 2: PersonaPool — sampling agents from PersonaStructure

**Files:**
- Modify: `mcv/population.py`
- Modify: `mcv/tests/test_population.py`

**Step 1: Write failing tests**

```python
# Add to mcv/tests/test_population.py
from mcv.population import PersonaPool

def _make_structure() -> PersonaStructure:
    dims = [
        TraitDimension("social_motivation", "Social drive", "solo", "social", "normal", 6.0, 2.0, "space2"),
        TraitDimension("patience",           "Patience",    "quits fast", "very patient", "normal", 5.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Socializer", 0.60, "social", {"social_motivation": (6.0, 10.0)}),
        Archetype("Grinder",    0.40, "solo",   {"social_motivation": (0.0, 5.0)}),
    ]
    return PersonaStructure("Test Pop", "test product", dims, archs)

def test_pool_generates_n_agents():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(30)
    assert len(agents) == 30

def test_pool_agents_have_unique_ids():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(10)
    ids = [a.agent_id for a in agents]
    assert len(set(ids)) == 10

def test_pool_archetype_frequency_respected():
    """60% Socializer, 40% Grinder → at N=100, within ±15%."""
    pool = PersonaPool(_make_structure())
    agents = pool.generate(100)
    socializers = sum(1 for a in agents if a.archetype_name == "Socializer")
    assert 45 <= socializers <= 75  # 60% ± 15pp

def test_pool_trait_constraints_respected():
    """Socializer agents must have social_motivation in [6.0, 10.0]."""
    pool = PersonaPool(_make_structure())
    agents = pool.generate(60)
    for a in agents:
        if a.archetype_name == "Socializer":
            assert 6.0 <= a.trait_vector["social_motivation"] <= 10.0
        elif a.archetype_name == "Grinder":
            assert 0.0 <= a.trait_vector["social_motivation"] <= 5.0

def test_pool_all_traits_in_0_10():
    pool = PersonaPool(_make_structure())
    agents = pool.generate(30)
    for a in agents:
        for val in a.trait_vector.values():
            assert 0.0 <= val <= 10.0
```

**Step 2: Run to verify failures**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py::test_pool_generates_n_agents -v
```
Expected: `ImportError: cannot import name 'PersonaPool'`

**Step 3: Implement PersonaPool**

Add to `mcv/population.py` after the dataclasses:

```python
import random


class PersonaPool:
    """Samples N AgentProfiles from a confirmed PersonaStructure.

    Archetype assignment respects frequency weights.
    Trait values are sampled within archetype constraints using normal distribution,
    clipped to [0, 10] and to the archetype's [min, max] window.
    """

    def __init__(self, structure: PersonaStructure):
        self._structure = structure

    def generate(self, n: int) -> list[AgentProfile]:
        """Generate N agents sampled from structure distributions."""
        agents = []
        archetypes = self._structure.archetypes
        dims = self._structure.trait_dimensions

        # Normalize frequencies in case they don't sum to exactly 1.0
        total = sum(a.frequency for a in archetypes)
        weights = [a.frequency / total for a in archetypes]

        for i in range(n):
            # Pick archetype by frequency weight
            arch = random.choices(archetypes, weights=weights, k=1)[0]

            # Sample trait vector
            trait_vector: dict[str, float] = {}
            for dim in dims:
                lo, hi = arch.trait_constraints.get(dim.name, (0.0, 10.0))
                # Center normal distribution within [lo, hi]
                center = (lo + hi) / 2
                span = (hi - lo) / 4  # ±2σ covers the range
                raw = random.gauss(center, max(span, 0.5))
                val = max(lo, min(hi, raw))   # clip to archetype window
                val = max(0.0, min(10.0, val))  # clip to absolute scale
                trait_vector[dim.name] = round(val, 2)

            agents.append(AgentProfile(
                agent_id=f"agent_{i+1:03d}",
                archetype_name=arch.name,
                trait_vector=trait_vector,
            ))

        return agents
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py -v
```
Expected: 9 PASS

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add mcv/population.py mcv/tests/test_population.py
git commit -m "feat: add PersonaPool with frequency-weighted archetype sampling"
```

---

## Task 3: AgentProfile → behavioral constraint prompt string

**Files:**
- Modify: `mcv/population.py`
- Modify: `mcv/tests/test_population.py`

The trait vector must become actionable text that constrains LLM behavior in each session prompt. This is the anti-rationalization layer.

**Step 1: Write failing tests**

```python
# Add to mcv/tests/test_population.py

def test_behavioral_constraints_low_trait():
    dims = [TraitDimension("patience", "Patience", "quits immediately if confused", "waits indefinitely", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Grinder", {"patience": 1.5})
    text = agent.to_behavioral_constraints(dims)
    assert "quits immediately if confused" in text
    assert "1.5" in text

def test_behavioral_constraints_high_trait():
    dims = [TraitDimension("patience", "Patience", "quits immediately if confused", "waits indefinitely", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Socializer", {"patience": 9.2})
    text = agent.to_behavioral_constraints(dims)
    assert "waits indefinitely" in text

def test_behavioral_constraints_includes_anti_rationalization():
    dims = [TraitDimension("ludo_familiarity", "Ludo XP", "never played Ludo", "expert", "normal", 5.0, 2.0, "space2")]
    agent = AgentProfile("a1", "Arch", {"ludo_familiarity": 1.0})
    text = agent.to_behavioral_constraints(dims)
    # Must include the three anti-rationalization rules
    assert "放弃" in text or "abandon" in text.lower()
    assert "礼貌" in text or "polite" in text.lower()

def test_behavioral_constraints_covers_all_traits():
    dims = [
        TraitDimension("patience", "Patience", "low patience", "high patience", "normal", 5.0, 2.0, "space2"),
        TraitDimension("social_motivation", "Social", "solo player", "social-first", "normal", 6.0, 2.0, "space2"),
    ]
    agent = AgentProfile("a1", "Arch", {"patience": 7.0, "social_motivation": 3.0})
    text = agent.to_behavioral_constraints(dims)
    assert "patience" in text.lower() or "high patience" in text
    assert "social" in text.lower()
```

**Step 2: Run to verify failures**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py::test_behavioral_constraints_low_trait -v
```
Expected: `AttributeError: 'AgentProfile' object has no attribute 'to_behavioral_constraints'`

**Step 3: Implement `to_behavioral_constraints()`**

Add method to `AgentProfile` dataclass. Replace the `@dataclass` definition with a class that adds the method:

```python
@dataclass
class AgentProfile:
    """Frozen per-session persona sampled from PersonaStructure."""
    agent_id: str
    archetype_name: str
    trait_vector: dict[str, float]    # {trait_name: value (0-10)}

    def to_behavioral_constraints(self, dimensions: list["TraitDimension"]) -> str:
        """Convert trait vector to behavioral constraint string for session prompt.

        Includes three mandatory anti-rationalization rules that suppress LLM's
        natural tendency toward rational, helpful, task-completing behavior.
        """
        lines = []
        for dim in dimensions:
            val = self.trait_vector.get(dim.name, 5.0)
            if val <= 3.0:
                label = dim.low_label
                intensity = "强烈地"
            elif val <= 4.5:
                label = dim.low_label
                intensity = "有些"
            elif val >= 7.0:
                label = dim.high_label
                intensity = "强烈地"
            elif val >= 5.5:
                label = dim.high_label
                intensity = "有些"
            else:
                label = f"在{dim.name}上处于中间状态"
                intensity = ""
            lines.append(f"- {intensity}{label}（{dim.name}={val:.1f}/10）")

        # Anti-rationalization rules — always included
        anti_rules = (
            "\n【行为约束 — 必须遵守】\n"
            "- 你只记得这个 app 里 2-3 件事，其余都忘了或从未注意到\n"
            "- 随时可以放弃操作——放弃是最正常的结果，不需要解释原因\n"
            "- 不要对产品问题保持礼貌；真实用户会沮丧、困惑、直接离开\n"
            "- 你不是一个乐于助人的 AI，你是一个有自己目标和耐心极限的真实用户"
        )

        trait_block = "\n".join(lines) if lines else "（无特质约束）"
        return f"【你的特质】\n{trait_block}{anti_rules}"
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py -v
```
Expected: 13 PASS

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add mcv/population.py mcv/tests/test_population.py
git commit -m "feat: add AgentProfile.to_behavioral_constraints() with anti-rationalization rules"
```

---

## Task 4: PopulationResearcher — 3-space LLM research → PersonaStructure

**Files:**
- Modify: `mcv/population.py`
- Modify: `mcv/tests/test_population.py`

This is the core research step. One LLM call (Sonnet) takes the product description and returns structured PersonaStructure data covering all three spaces.

**Step 1: Write failing tests**

```python
# Add to mcv/tests/test_population.py
import json
from unittest.mock import patch

def _mock_researcher_response() -> str:
    return json.dumps({
        "population_label": "Arabic Ludo App Users",
        "product_context": "Ludo mobile app for Arabic families",
        "trait_dimensions": [
            {
                "name": "ludo_familiarity",
                "description": "Prior Ludo experience",
                "low_label": "never played Ludo, confused by rules",
                "high_label": "expert player who knows all strategies",
                "distribution": "bimodal",
                "mean": 5.0,
                "std": 2.5,
                "source": "space2",
            },
            {
                "name": "social_motivation",
                "description": "Motivation to play with others vs solo",
                "low_label": "prefers solo play, ignores social features",
                "high_label": "only plays with friends and family",
                "distribution": "right_skewed",
                "mean": 7.0,
                "std": 2.0,
                "source": "space3",
            },
        ],
        "archetypes": [
            {
                "name": "Family Socializer",
                "frequency": 0.50,
                "description": "Plays with family, social-first motivation",
                "trait_constraints": {"social_motivation": [6.0, 10.0], "ludo_familiarity": [2.0, 9.0]},
            },
            {
                "name": "Competitive Grinder",
                "frequency": 0.30,
                "description": "Plays to win, high familiarity",
                "trait_constraints": {"social_motivation": [2.0, 6.0], "ludo_familiarity": [6.0, 10.0]},
            },
            {
                "name": "Casual Dabbler",
                "frequency": 0.20,
                "description": "First-time user, low commitment",
                "trait_constraints": {"social_motivation": [3.0, 7.0], "ludo_familiarity": [0.0, 4.0]},
            },
        ],
        "research_notes": "Ludo is dominant in Arabic family gaming. Social motivation is primary driver.",
    })

def test_researcher_returns_persona_structure():
    from mcv.population import PopulationResearcher
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo mobile app for Arabic families")
    assert isinstance(result, PersonaStructure)
    assert result.population_label == "Arabic Ludo App Users"
    assert len(result.trait_dimensions) == 2
    assert len(result.archetypes) == 3

def test_researcher_trait_dimensions_parsed():
    from mcv.population import PopulationResearcher
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    dim = result.trait_dimensions[0]
    assert dim.name == "ludo_familiarity"
    assert dim.distribution == "bimodal"
    assert dim.source == "space2"

def test_researcher_archetypes_parsed():
    from mcv.population import PopulationResearcher
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    arch = result.archetypes[0]
    assert arch.name == "Family Socializer"
    assert arch.frequency == 0.50
    assert arch.trait_constraints["social_motivation"] == (6.0, 10.0)

def test_researcher_frequencies_sum_to_1():
    from mcv.population import PopulationResearcher
    with patch("mcv.core._llm_call", return_value=(_mock_researcher_response(), {})):
        researcher = PopulationResearcher(api_key="test")
        result = researcher.research("A Ludo app")
    total = sum(a.frequency for a in result.archetypes)
    assert abs(total - 1.0) < 0.01
```

**Step 2: Run to verify failures**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py::test_researcher_returns_persona_structure -v
```
Expected: `ImportError: cannot import name 'PopulationResearcher'`

**Step 3: Implement PopulationResearcher**

Add to `mcv/population.py`:

```python
import json
import re

_RESEARCH_PROMPT = """You are a behavioral researcher and product designer.

A product manager wants to simulate realistic users for their product:
"{product_description}"

Research and define the complete behavioral population for this product by combining:
- Space 1 (Product Intent): what the product does and who it targets
- Space 2 (Behavioral Research): what psychology and UX research says about this user type
- Space 3 (Market Context): cultural, competitive, and demographic context

Output ONLY valid JSON (no markdown):
{{
  "population_label": "short label for this user population",
  "product_context": "one sentence describing the product",
  "trait_dimensions": [
    {{
      "name": "snake_case_name",
      "description": "what this dimension measures",
      "low_label": "specific behavior when this trait is very low (0-3)",
      "high_label": "specific behavior when this trait is very high (7-10)",
      "distribution": "normal|uniform|bimodal|right_skewed|left_skewed",
      "mean": 5.0,
      "std": 2.0,
      "source": "space1|space2|space3"
    }}
  ],
  "archetypes": [
    {{
      "name": "Archetype Name",
      "frequency": 0.40,
      "description": "who they are and why they use this product",
      "trait_constraints": {{"trait_name": [min_val, max_val]}}
    }}
  ],
  "research_notes": "key insight about this user population"
}}

Rules:
- 4-8 trait_dimensions that actually predict different behaviors in this product
- 3-5 archetypes covering the full spectrum from most enthusiastic to least
- archetype frequencies must sum to 1.0
- trait_constraints only include dimensions that differentiate this archetype
- low_label and high_label must be specific behaviors, not just adjectives
- cover cultural/regional context in the dimensions if relevant"""


class PopulationResearcher:
    """Runs 3-space research to produce a PersonaStructure for human confirmation.

    One Sonnet call synthesizes product intent, behavioral research, and market context.
    """

    def __init__(self, api_key: str):
        self._api_key = api_key

    def research(self, product_description: str) -> PersonaStructure:
        """Research user population for a product description.

        Returns a PersonaStructure ready for human confirmation.
        """
        import mcv.core as _core
        prompt = _RESEARCH_PROMPT.format(product_description=product_description[:2000])
        raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=2048)
        return self._parse(raw, product_description)

    def _parse(self, raw: str, product_description: str) -> PersonaStructure:
        """Parse LLM response into PersonaStructure. Returns minimal fallback on failure."""
        # Strip markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', raw.strip())
        text = re.sub(r'\s*```$', '', text.strip())
        # Find first {...} block
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if not m:
            return self._fallback(product_description)
        try:
            data = json.loads(m.group())
        except (json.JSONDecodeError, ValueError):
            return self._fallback(product_description)

        dims = []
        for d in data.get("trait_dimensions", []):
            if not isinstance(d, dict) or not d.get("name"):
                continue
            dims.append(TraitDimension(
                name=d["name"],
                description=d.get("description", ""),
                low_label=d.get("low_label", "low"),
                high_label=d.get("high_label", "high"),
                distribution=d.get("distribution", "normal"),
                mean=float(d.get("mean", 5.0)),
                std=float(d.get("std", 2.0)),
                source=d.get("source", "space2"),
            ))

        archs = []
        for a in data.get("archetypes", []):
            if not isinstance(a, dict) or not a.get("name"):
                continue
            raw_constraints = a.get("trait_constraints", {})
            # Normalise [min, max] lists to tuples
            constraints = {
                k: (float(v[0]), float(v[1]))
                for k, v in raw_constraints.items()
                if isinstance(v, (list, tuple)) and len(v) == 2
            }
            archs.append(Archetype(
                name=a["name"],
                frequency=float(a.get("frequency", 1.0 / max(len(data.get("archetypes", [1])), 1))),
                description=a.get("description", ""),
                trait_constraints=constraints,
            ))

        # Normalise frequencies
        if archs:
            total = sum(a.frequency for a in archs)
            if total > 0:
                archs = [Archetype(a.name, a.frequency / total, a.description, a.trait_constraints) for a in archs]

        if not dims or not archs:
            return self._fallback(product_description)

        return PersonaStructure(
            population_label=data.get("population_label", "App Users"),
            product_context=data.get("product_context", product_description[:100]),
            trait_dimensions=dims,
            archetypes=archs,
            research_notes=data.get("research_notes", ""),
        )

    def _fallback(self, product_description: str) -> PersonaStructure:
        """Minimal 2-archetype structure when LLM parsing fails."""
        dims = [
            TraitDimension("engagement_drive", "Motivation to engage", "passive, easily disengaged",
                           "highly motivated, explores all features", "normal", 5.0, 2.5, "space1"),
        ]
        archs = [
            Archetype("Engaged User",  0.60, "actively uses the product", {"engagement_drive": (5.0, 10.0)}),
            Archetype("Passive User",  0.40, "minimal engagement",         {"engagement_drive": (0.0, 5.0)}),
        ]
        return PersonaStructure(
            population_label="App Users",
            product_context=product_description[:100],
            trait_dimensions=dims,
            archetypes=archs,
            research_notes="fallback — LLM parsing failed",
        )
```

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py -v
```
Expected: 17 PASS

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add mcv/population.py mcv/tests/test_population.py
git commit -m "feat: add PopulationResearcher — 3-space LLM research produces PersonaStructure"
```

---

## Task 5: UserSimulator integration — prepare_with_pool()

**Files:**
- Modify: `mcv/user_simulator.py`
- Modify: `mcv/tests/test_population.py`

Add `prepare_with_pool()` to `UserSimulator`. When a pool is used, each of the N sessions uses a different `AgentProfile`. The session prompt replaces the plain `user_type` string with the agent's `to_behavioral_constraints()` output.

**Step 1: Write failing tests**

```python
# Add to mcv/tests/test_population.py
from unittest.mock import MagicMock, patch
from mcv.user_simulator import UserSimulator
from mcv.domain_configs import AppDomainConfig
from mcv.schema_extractor import EvaluationMetric

def _make_pool_agents() -> list[AgentProfile]:
    dims = [
        TraitDimension("patience", "Patience", "quits immediately", "very patient", "normal", 5.0, 2.0, "space2"),
    ]
    archs = [
        Archetype("Engaged", 0.70, "engaged user", {"patience": (5.0, 10.0)}),
        Archetype("Passive",  0.30, "passive user", {"patience": (0.0, 5.0)}),
    ]
    structure = PersonaStructure("Test Pop", "test app", dims, archs)
    return PersonaPool(structure).generate(6)

def test_prepare_with_pool_sets_pool():
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = _make_pool_agents()
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)
    assert sim._agent_pool == agents
    assert sim._metrics == metrics

def test_simulate_with_pool_uses_one_agent_per_session():
    """Each session should use a different agent from the pool."""
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = _make_pool_agents()
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)

    captured_prompts = []
    def fake_llm(prompt, api_key, **kwargs):
        captured_prompts.append(prompt)
        return "day1_return: yes", {}

    with patch("mcv.core._llm_call", side_effect=fake_llm):
        sim.simulate(n_runs=6)

    # Each prompt should contain a different agent's constraints
    # (different patience values → different behavioral constraint text)
    assert len(captured_prompts) == 6
    # All prompts should contain the anti-rationalization rules
    for p in captured_prompts:
        assert "放弃" in p or "abandon" in p.lower()

def test_simulate_with_pool_cycles_when_n_exceeds_pool():
    """If n_runs > len(pool), cycle through pool."""
    sim = UserSimulator("test user", AppDomainConfig, api_key="test")
    agents = _make_pool_agents()  # 6 agents
    metrics = [EvaluationMetric("day1_return", "bool", "会回来吗？")]
    sim.prepare_with_pool(product="test product", pool=agents, locked_metrics=metrics)

    with patch("mcv.core._llm_call", return_value=("day1_return: yes", {})):
        sim.simulate(n_runs=10)  # more than 6 agents

    assert len(sim._session_results) == 10
```

**Step 2: Run to verify failures**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py::test_prepare_with_pool_sets_pool -v
```
Expected: `AttributeError: 'UserSimulator' object has no attribute 'prepare_with_pool'`

**Step 3: Implement `prepare_with_pool()` in UserSimulator**

In `mcv/user_simulator.py`, add `_agent_pool` attribute and `prepare_with_pool()` method, and update `simulate()` to use it.

Add to `UserSimulator.__init__`:
```python
self._agent_pool: list | None = None   # populated by prepare_with_pool()
```

Add method after `prepare()`:
```python
def prepare_with_pool(
    self,
    product: str,
    pool: list,                                          # list[AgentProfile]
    goal: str | None = None,
    locked_metrics: list[EvaluationMetric] | None = None,
) -> "UserSimulator":
    """Like prepare() but uses a pool of AgentProfiles for heterogeneous simulation.

    Each session in simulate() will use a different AgentProfile from the pool.
    When n_runs > len(pool), cycles through pool in order.
    """
    self._product = product
    self._screen_id = None
    self._agent_pool = pool
    if locked_metrics is not None:
        self._metrics = locked_metrics
    else:
        from mcv.schema_extractor import extract_evaluation_schema
        self._metrics = extract_evaluation_schema(goal or "", self.api_key)
    return self
```

Modify `simulate()` — add agent pool branch at the top of the for loop:

```python
def simulate(self, n_runs: int = 60) -> "UserSimulator":
    """Run N independent sessions. Uses AgentPool if prepared with prepare_with_pool()."""
    if not self._product:
        raise RuntimeError("call prepare() or prepare_with_pool() before simulate()")
    import mcv.core as _core
    self._session_results = []
    roles = list(self.domain_config.user_roles.keys())

    for i in range(n_runs):
        # Pool path: one distinct AgentProfile per session
        if self._agent_pool:
            agent = self._agent_pool[i % len(self._agent_pool)]
            # Import here to avoid circular at module level
            from mcv.population import TraitDimension  # only used to pass dims to constraints
            # Get dimensions from pool's structure if available, else skip dim labels
            dims = getattr(getattr(agent, "_structure", None), "trait_dimensions", [])
            user_type_text = agent.to_behavioral_constraints(dims)
        else:
            user_type_text = self.user_type

        role = None if self._agent_pool else (roles[i % len(roles)] if roles else None)
        ctx = _random_context_for_domain(role, self.domain_config)
        prompt = _build_session_prompt(
            user_type=user_type_text,
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
```

**Note on `agent._structure`:** The AgentProfile doesn't hold a reference to the PersonaStructure. We need to pass dimensions separately. Simplest fix: `PersonaPool.generate()` stores dims on each agent.

Update `PersonaPool.generate()` — add after creating `AgentProfile`:
```python
# Store dims reference for constraint generation
import types
agent = AgentProfile(...)
agent._dims = dims          # attach dims to agent instance (not in dataclass to keep it clean)
```

Then in `simulate()`, get dims via `getattr(agent, "_dims", [])`.

**Step 4: Run tests**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py -v
```
Expected: 21 PASS

**Step 5: Commit**

```bash
cd /Users/michael/mcv && git add mcv/user_simulator.py mcv/tests/test_population.py
git commit -m "feat: UserSimulator.prepare_with_pool() — one AgentProfile per session"
```

---

## Task 6: Export new types + smoke test + push

**Files:**
- Modify: `mcv/__init__.py`
- Modify: `mcv/tests/test_population.py`

**Step 1: Write export test**

```python
# Add to mcv/tests/test_population.py

def test_population_exports_available_at_mcv_root():
    import mcv
    assert hasattr(mcv, "TraitDimension")
    assert hasattr(mcv, "Archetype")
    assert hasattr(mcv, "PersonaStructure")
    assert hasattr(mcv, "AgentProfile")
    assert hasattr(mcv, "PersonaPool")
    assert hasattr(mcv, "PopulationResearcher")
```

**Step 2: Run to verify failure**

```bash
cd /Users/michael/mcv && python -m pytest tests/test_population.py::test_population_exports_available_at_mcv_root -v
```
Expected: `AssertionError` on `hasattr(mcv, "TraitDimension")`

**Step 3: Add exports to `mcv/__init__.py`**

Add to the existing `__init__.py` after the existing imports:

```python
from mcv.population import (
    TraitDimension, Archetype, PersonaStructure, AgentProfile,
    PersonaPool, PopulationResearcher,
)
```

Add to `__all__`:
```python
"TraitDimension", "Archetype", "PersonaStructure", "AgentProfile",
"PersonaPool", "PopulationResearcher",
```

**Step 4: Run full test suite**

```bash
cd /Users/michael/mcv && python -m pytest tests/ -v --tb=short
```
Expected: all existing tests PASS + new population tests PASS

**Step 5: Smoke test with real API**

```python
# Run interactively to verify end-to-end:
import sys; sys.path.insert(0, '/Users/michael')
import os; os.environ['ANTHROPIC_API_KEY'] = open('/Users/michael/eltm/.env').read().split('=',1)[1].strip().split('\n')[0]

from mcv import PopulationResearcher, PersonaPool, UserSimulator, AppDomainConfig
from mcv.schema_extractor import EvaluationMetric

researcher = PopulationResearcher(api_key=os.environ['ANTHROPIC_API_KEY'])
structure = researcher.research("A Ludo mobile app for Arabic families")
print(f"Population: {structure.population_label}")
print(f"Dimensions: {[d.name for d in structure.trait_dimensions]}")
print(f"Archetypes: {[(a.name, a.frequency) for a in structure.archetypes]}")

pool = PersonaPool(structure)
agents = pool.generate(6)
print(f"\nGenerated {len(agents)} agents:")
for a in agents[:3]:
    print(f"  {a.agent_id} [{a.archetype_name}] {a.trait_vector}")
```

**Step 6: Commit + push**

```bash
cd /Users/michael/mcv
git add mcv/__init__.py
git commit -m "feat: export PopulationOS types from mcv root"
export PATH="$HOME/bin:$PATH"
GH_TOKEN="$GH_TOKEN" ~/bin/gh repo view mozatyin/mcv --json name  # verify auth
git push origin main
```
