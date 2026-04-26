# MCV × ELTM Research Bridge — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add three MCV capabilities that close the feedback loop between behavioral simulation and ELTM's M1 feature research + M4/M5 redesign.

**Architecture:** Three new methods on `MCVClient` + two new dataclasses in `report.py`. Each method is independently callable and composes into a PDCA research loop: `research_aarrr()` grounds M1 scoring in population data, `validate_coherence()` catches dropped dependencies before M2, and `attribute_frictions()` converts simulation friction themes into `reforge()` defect manifests.

**Tech Stack:** Python 3.11, anthropic SDK (via `mcv.core._llm_call`), existing `PopulationResearcher` + `PersonaPool` from `mcv/population.py`

---

## Composition Loop (what we're building toward)

```
PRD
 → MCVClient.research_aarrr()           [Position 1] grounds M1 AARRR with population votes
 → ELTM M1 feature selection (Top-K)
 → MCVClient.validate_coherence()       [Position 2] detects broken dependencies in selected set
 → reinstated features if needed
 → ELTM M2-M5 → wireframe + contract
 → MCVClient.simulate()                 [existing]   behavioral frictions from real user journeys
 → MCVClient.attribute_frictions()      [Position 3] maps friction themes → reforge() manifest
 → eltm.reforge(fix_requirement=...)    [existing]   targeted M4+M5 redesign
```

---

## New Types

### `FeatureAAR` (in `mcv/report.py`)
```python
@dataclass
class FeatureAAR:
    feature_id: str
    acquisition: float          # 0.0-1.0 population-weighted mean
    activation: float
    retention: float
    revenue: float
    referral: float
    confidence: float           # 1 - mean_stdev across archetypes, 0-1
    archetype_votes: dict[str, dict[str, float]]  # {archetype_name: {dim: score}}
```

### `CoherenceReport` (in `mcv/report.py`)
```python
@dataclass
class CoherenceReport:
    selected_feature_ids: list[str]
    missing_dependencies: list[dict]        # [{"feature_id": str, "required_by": [str], "reason": str}]
    blocked_journeys: list[str]             # narrative strings
    reinstate_recommendations: list[str]   # feature_ids to add back
    is_coherent: bool                      # True when missing_dependencies is empty
```

---

## Task 1: Add `FeatureAAR` and `CoherenceReport` dataclasses to `report.py`

**Files:**
- Modify: `mcv/report.py` (append two dataclasses at top, after imports)
- Test: `mcv/tests/test_report.py`

**Step 1: Write the failing tests**

```python
# mcv/tests/test_report.py — append to existing file

from mcv.report import FeatureAAR, CoherenceReport

def test_feature_aar_fields():
    f = FeatureAAR(
        feature_id="invite_friends",
        acquisition=0.3, activation=0.5, retention=0.4,
        revenue=0.1, referral=0.9,
        confidence=0.8,
        archetype_votes={"Gamer": {"acquisition": 0.3, "referral": 0.9}},
    )
    assert f.feature_id == "invite_friends"
    assert f.referral == 0.9
    assert "Gamer" in f.archetype_votes

def test_coherence_report_is_coherent_true():
    r = CoherenceReport(
        selected_feature_ids=["ludo", "invite_friends"],
        missing_dependencies=[],
        blocked_journeys=[],
        reinstate_recommendations=[],
        is_coherent=True,
    )
    assert r.is_coherent

def test_coherence_report_is_coherent_false():
    r = CoherenceReport(
        selected_feature_ids=["ludo"],
        missing_dependencies=[{"feature_id": "invite_friends", "required_by": ["ludo"],
                               "reason": "ludo needs multiple players"}],
        blocked_journeys=["User tried to play Ludo but had no friends"],
        reinstate_recommendations=["invite_friends"],
        is_coherent=False,
    )
    assert not r.is_coherent
    assert "invite_friends" in r.reinstate_recommendations
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_report.py::test_feature_aar_fields -v
```
Expected: FAIL with `ImportError: cannot import name 'FeatureAAR'`

**Step 3: Add dataclasses to `report.py`**

Insert after the existing `CompareReport` dataclass (before `_compute_compare`):

```python
@dataclass
class FeatureAAR:
    """Population-grounded AARRR scores for a single product feature."""
    feature_id: str
    acquisition: float          # 0.0–1.0 population-weighted mean
    activation: float
    retention: float
    revenue: float
    referral: float
    confidence: float           # 1 − mean stdev across archetypes (0–1)
    archetype_votes: dict       # {archetype_name: {dimension: score}}


@dataclass
class CoherenceReport:
    """Dependency validation result for a selected feature set."""
    selected_feature_ids: list
    missing_dependencies: list  # [{"feature_id", "required_by", "reason"}]
    blocked_journeys: list      # human-readable narrative strings
    reinstate_recommendations: list  # feature_ids to add back
    is_coherent: bool
```

**Step 4: Run tests to verify they pass**

```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_report.py::test_feature_aar_fields tests/test_report.py::test_coherence_report_is_coherent_true tests/test_report.py::test_coherence_report_is_coherent_false -v
```
Expected: 3 PASS

**Step 5: Commit**
```bash
cd /Users/michael/mcv && git add mcv/report.py mcv/tests/test_report.py && git commit -m "feat: add FeatureAAR and CoherenceReport dataclasses"
```

---

## Task 2: `MCVClient.research_aarrr()` — population-grounded AARRR scoring

**Files:**
- Modify: `mcv/client.py` (add method)
- Create: `mcv/tests/test_research_aarrr.py`

**What it does:**
1. `PopulationResearcher.research(product_description)` → `PersonaStructure` (1 Sonnet call)
2. Build one batch prompt with all archetypes + all features → single Sonnet call returns `[{feature_id, acquisition, activation, retention, revenue, referral, archetype_votes}]`
3. Compute `confidence` = `1 - mean(stdev of {dim scores across archetypes})`, clipped to [0, 1]
4. Return `list[FeatureAAR]`

**Step 1: Write the failing tests**

```python
# mcv/tests/test_research_aarrr.py
import os
import pytest

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def test_research_aarrr_returns_correct_count():
    """Each input feature gets exactly one FeatureAAR."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    features = [
        {"id": "invite_friends", "name": "Invite Friends", "description": "Invite friends to join rooms"},
        {"id": "ludo",           "name": "Ludo Game",      "description": "Multiplayer board game"},
        {"id": "daily_tasks",    "name": "Daily Tasks",    "description": "Daily quest checklist"},
    ]
    result = client.research_aarrr(
        product_description="Arabic social gaming platform with voice rooms and board games",
        features=features,
    )
    assert len(result) == 3
    ids = {f.feature_id for f in result}
    assert ids == {"invite_friends", "ludo", "daily_tasks"}

def test_research_aarrr_scores_in_range():
    """All AARRR scores are 0.0–1.0, confidence is 0.0–1.0."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    features = [{"id": "coins", "name": "Coins", "description": "Virtual currency for gifts"}]
    result = client.research_aarrr(
        product_description="Social gifting app",
        features=features,
    )
    f = result[0]
    for dim in (f.acquisition, f.activation, f.retention, f.revenue, f.referral, f.confidence):
        assert 0.0 <= dim <= 1.0, f"dimension out of range: {dim}"

def test_research_aarrr_fallback_on_empty_features():
    """Empty feature list → empty result, no crash."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    result = client.research_aarrr(product_description="any app", features=[])
    assert result == []

def test_research_aarrr_invite_friends_referral_high():
    """Invite Friends must score high on referral relative to other dimensions."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    features = [
        {"id": "invite_friends", "name": "Invite Friends",
         "description": "Send invitation links to friends to join the platform"},
    ]
    result = client.research_aarrr(
        product_description="Arabic social gaming platform with multiplayer games",
        features=features,
    )
    f = result[0]
    assert f.referral >= 0.5, f"invite_friends referral should be ≥0.5, got {f.referral}"
    assert f.referral >= f.revenue, "referral should dominate revenue for invite feature"
```

**Step 2: Run to verify fail**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_research_aarrr.py::test_research_aarrr_fallback_on_empty_features -v
```
Expected: FAIL with `AttributeError: 'MCVClient' object has no attribute 'research_aarrr'`

**Step 3: Implement `research_aarrr()` in `client.py`**

Add these imports at top of `client.py`:
```python
from mcv.population import PopulationResearcher, PersonaPool
from mcv.report import FeatureAAR, CoherenceReport
```

Add private helper and method to `MCVClient`:

```python
_AARRR_VOTE_PROMPT = """You are scoring product features for a mobile app from the perspective of different user archetypes.

Product: {product_description}

User archetypes (each with their background story):
{archetypes_block}

Features to score:
{features_block}

For each feature, score its impact on each AARRR dimension (0.0 = no impact, 1.0 = primary driver)
from each archetype's perspective. Then compute the mean across archetypes.

Return ONLY valid JSON (no markdown):
[
  {{
    "feature_id": "...",
    "archetype_votes": {{
      "ArchetypeName": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}},
      ...
    }},
    "mean": {{"acquisition": 0.0, "activation": 0.0, "retention": 0.0, "revenue": 0.0, "referral": 0.0}}
  }},
  ...
]"""

def research_aarrr(
    self,
    product_description: str,
    features: list[dict],
    objectives: dict | None = None,
) -> list[FeatureAAR]:
    """Score features on AARRR dimensions using population-grounded archetype voting.

    Two LLM calls: (1) PopulationResearcher to build persona archetypes,
    (2) batch AARRR vote for all features across all archetypes.

    Args:
        product_description: PRD or one-paragraph product description.
        features: [{"id": str, "name": str, "description": str}, ...]
        objectives: ignored (reserved for future weighting) — scores are raw 0-1

    Returns:
        list[FeatureAAR] in same order as input features.
        Empty list if features is empty.
    """
    import statistics as _stats
    import json as _json
    import re as _re

    if not features:
        return []

    # Step 1: Research population (1 Sonnet call)
    researcher = PopulationResearcher(self._api_key)
    try:
        structure = researcher.research(product_description)
    except Exception:
        structure = researcher._fallback(product_description)

    archetypes = structure.archetypes

    # Step 2: Build batch prompt
    archetypes_block = "\n".join(
        f"- {a.name}: {a.background_story or a.description}"
        for a in archetypes
    )
    features_block = "\n".join(
        f"- id={f['id']}: {f['name']} — {f.get('description', '')}"
        for f in features
    )
    prompt = _AARRR_VOTE_PROMPT.format(
        product_description=product_description[:800],
        archetypes_block=archetypes_block,
        features_block=features_block,
    )

    # Step 3: Single LLM vote call (1 Sonnet call)
    from mcv import core as _core
    raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=3000)

    # Parse
    scored: list[FeatureAAR] = []
    m = _re.search(r'\[.*\]', raw, _re.DOTALL)
    items = []
    if m:
        try:
            items = _json.loads(m.group())
        except (ValueError, _json.JSONDecodeError):
            pass

    items_by_id = {item["feature_id"]: item for item in items if isinstance(item, dict) and "feature_id" in item}

    _DIMS = ("acquisition", "activation", "retention", "revenue", "referral")

    for f in features:
        fid = f["id"]
        item = items_by_id.get(fid)
        if item:
            mean = item.get("mean", {})
            arch_votes = item.get("archetype_votes", {})
            # Compute confidence = 1 - mean stdev across archetypes per dimension
            stdevs = []
            for dim in _DIMS:
                vals = [v.get(dim, 0.5) for v in arch_votes.values() if isinstance(v, dict)]
                if len(vals) >= 2:
                    stdevs.append(_stats.stdev(vals))
            confidence = round(max(0.0, 1.0 - (sum(stdevs) / len(stdevs) if stdevs else 0.0)), 4)
            scored.append(FeatureAAR(
                feature_id=fid,
                acquisition=round(float(mean.get("acquisition", 0.5)), 4),
                activation=round(float(mean.get("activation", 0.5)), 4),
                retention=round(float(mean.get("retention", 0.5)), 4),
                revenue=round(float(mean.get("revenue", 0.2)), 4),
                referral=round(float(mean.get("referral", 0.2)), 4),
                confidence=confidence,
                archetype_votes=arch_votes,
            ))
        else:
            # Fallback: neutral scores
            scored.append(FeatureAAR(
                feature_id=fid,
                acquisition=0.5, activation=0.5, retention=0.5,
                revenue=0.2, referral=0.2,
                confidence=0.0,
                archetype_votes={},
            ))
    return scored
```

**Step 4: Run tests**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_research_aarrr.py -v
```
Expected: 4 PASS (skipped if no API key)

**Step 5: Commit**
```bash
cd /Users/michael/mcv && git add mcv/client.py mcv/tests/test_research_aarrr.py && git commit -m "feat: add MCVClient.research_aarrr() — population-grounded AARRR scoring"
```

---

## Task 3: `MCVClient.validate_coherence()` — dependency gap detection

**Files:**
- Modify: `mcv/client.py` (add method)
- Create: `mcv/tests/test_validate_coherence.py`

**What it does:**
1. **Rule-based pass (free):** scan for social enabler/dependent mismatch using the same keywords as ELTM's `_SOCIAL_ENABLER_KEYWORDS` / `_SOCIAL_DEPENDENT_KEYWORDS`. If a "dependent" feature is selected (ludo/game/room) but no "enabler" feature (invite/friends/refer), flag it.
2. **LLM gap analysis (1 Sonnet call):** given the constrained feature set, ask "what Day-1 journeys are blocked or degraded? Which dropped features are most needed?" — only runs when `dropped_features` is provided and rule-based check finds issues.
3. Returns `CoherenceReport` with structured findings.

**Design decision:** No simulation here. Simulation is expensive (~20s). This method must be fast (<5s) to fit in M1 loop between feature selection and M2. Rule-based check is instant; LLM gap analysis is 1 call.

**Step 1: Write the failing tests**

```python
# mcv/tests/test_validate_coherence.py
import os
import pytest

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def test_validate_coherence_detects_missing_enabler():
    """Ludo selected without Invite Friends → reinstate invite_friends recommended."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    selected = [{"id": "ludo", "name": "Ludo", "description": "multiplayer board game"}]
    dropped  = [{"id": "invite_friends", "name": "Invite Friends",
                 "description": "invite friends to join rooms"}]
    report = client.validate_coherence(
        product_description="Arabic social gaming platform",
        selected_features=selected,
        dropped_features=dropped,
    )
    assert not report.is_coherent
    assert "invite_friends" in report.reinstate_recommendations

def test_validate_coherence_coherent_when_enabler_present():
    """Both ludo and invite_friends selected → no gap."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    selected = [
        {"id": "ludo",           "name": "Ludo",           "description": "multiplayer board game"},
        {"id": "invite_friends", "name": "Invite Friends", "description": "invite friends to join"},
    ]
    report = client.validate_coherence(
        product_description="Arabic social gaming platform",
        selected_features=selected,
    )
    assert report.is_coherent
    assert report.reinstate_recommendations == []

def test_validate_coherence_empty_selected():
    """Empty feature set → not coherent, no crash."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    report = client.validate_coherence(
        product_description="any app", selected_features=[]
    )
    assert isinstance(report.is_coherent, bool)

def test_validate_coherence_no_social_features():
    """Non-social features → coherent (no dependency violations)."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    selected = [
        {"id": "daily_tasks", "name": "Daily Tasks",   "description": "daily quest checklist"},
        {"id": "coins",       "name": "Coins",         "description": "virtual currency"},
    ]
    report = client.validate_coherence(
        product_description="casual mobile game", selected_features=selected
    )
    assert report.is_coherent
```

**Step 2: Run to verify fail**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_validate_coherence.py::test_validate_coherence_empty_selected -v
```
Expected: FAIL with `AttributeError`

**Step 3: Implement `validate_coherence()` in `client.py`**

Private constants (add near top of file, after imports):
```python
_SOCIAL_ENABLER_KW = frozenset({"invite", "friend", "friends", "refer", "referral", "connect", "network", "share"})
_SOCIAL_DEPENDENT_KW = frozenset({"game", "ludo", "play", "battle", "versus", "match",
                                   "multiplayer", "players", "opponent", "opponents", "party", "room"})
```

Method:
```python
_COHERENCE_PROMPT = """You are a product designer reviewing a mobile app feature set.

Product: {product_description}

Selected features (only these will be in the app):
{selected_block}

Dropped features (not in the app):
{dropped_block}

Question: What Day-1 user journeys are blocked or significantly degraded because of missing features?
Which dropped features are most critical to reinstate?

Return ONLY valid JSON (no markdown):
{{
  "blocked_journeys": ["narrative description of blocked flow", ...],
  "critical_to_reinstate": ["feature_id", ...]
}}"""

def validate_coherence(
    self,
    product_description: str,
    selected_features: list[dict],
    dropped_features: list[dict] | None = None,
    user_type: str = "普通用户",
) -> CoherenceReport:
    """Detect dependency gaps in a selected feature set.

    Pass 1 (rule-based, free): flag social-dependent features (ludo/game/room)
    selected without any social-enabler (invite/friends/refer).

    Pass 2 (1 Sonnet call, only when gaps found and dropped_features provided):
    identify blocked Day-1 journeys and critical features to reinstate.

    Args:
        product_description: PRD or product summary.
        selected_features: features kept after M1 Top-K selection.
        dropped_features: features that were trimmed (needed for reinstate suggestions).
        user_type: ignored currently, reserved for future persona filtering.

    Returns:
        CoherenceReport. is_coherent=True when no dependency violations found.
    """
    import json as _json
    import re as _re
    from mcv import core as _core

    selected_ids = [f["id"] for f in selected_features]
    missing_deps: list[dict] = []
    blocked_journeys: list[str] = []
    reinstate: list[str] = []

    # Pass 1: rule-based dependency check
    has_enabler = any(
        bool(set((f.get("description", "") or f["name"]).lower().split()) & _SOCIAL_ENABLER_KW)
        for f in selected_features
    )
    dependent_features = [
        f for f in selected_features
        if bool(set((f.get("description", "") or f["name"]).lower().split()) & _SOCIAL_DEPENDENT_KW)
    ]

    if dependent_features and not has_enabler:
        dep_ids = [f["id"] for f in dependent_features]
        missing_deps.append({
            "feature_id": "social_enabler",
            "required_by": dep_ids,
            "reason": "multiplayer features need a way to connect with friends",
        })
        blocked_journeys.append(
            f"User cannot play {', '.join(dep_ids)} without friends — no invite/friend feature selected"
        )
        # Find best candidate in dropped_features
        if dropped_features:
            for df in dropped_features:
                df_words = set((df.get("description", "") or df["name"]).lower().split())
                if df_words & _SOCIAL_ENABLER_KW:
                    reinstate.append(df["id"])

    # Pass 2: LLM gap analysis (only when gaps exist and dropped features provided)
    if missing_deps and dropped_features:
        selected_block = "\n".join(f"- {f['id']}: {f['name']} — {f.get('description', '')}" for f in selected_features)
        dropped_block  = "\n".join(f"- {f['id']}: {f['name']} — {f.get('description', '')}" for f in dropped_features)
        prompt = _COHERENCE_PROMPT.format(
            product_description=product_description[:600],
            selected_block=selected_block,
            dropped_block=dropped_block,
        )
        try:
            raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=800)
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if m:
                data = _json.loads(m.group())
                blocked_journeys.extend(data.get("blocked_journeys", []))
                for fid in data.get("critical_to_reinstate", []):
                    if fid not in reinstate:
                        reinstate.append(fid)
        except Exception:
            pass  # LLM gap analysis is best-effort; rule-based result stands

    return CoherenceReport(
        selected_feature_ids=selected_ids,
        missing_dependencies=missing_deps,
        blocked_journeys=list(dict.fromkeys(blocked_journeys)),  # deduplicate
        reinstate_recommendations=list(dict.fromkeys(reinstate)),
        is_coherent=len(missing_deps) == 0,
    )
```

**Step 4: Run tests**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_validate_coherence.py -v
```
Expected: 4 PASS

**Step 5: Commit**
```bash
cd /Users/michael/mcv && git add mcv/client.py mcv/tests/test_validate_coherence.py && git commit -m "feat: add MCVClient.validate_coherence() — feature dependency gap detection"
```

---

## Task 4: `MCVClient.attribute_frictions()` — friction → reforge() manifest

**Files:**
- Modify: `mcv/client.py` (add method)
- Create: `mcv/tests/test_attribute_frictions.py`

**What it does:**
1. One Sonnet call: maps friction themes to features → structured defect list
2. Output shape: exactly what `eltm.reforge(fix_requirement=...)` expects
3. Each defect has: `type` (design/ux/rules), `severity` (P0/P1/P2), `description`, `affected_screens`, `suggested_fix`

**Step 1: Write the failing tests**

```python
# mcv/tests/test_attribute_frictions.py
import os
import pytest

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

def test_attribute_frictions_returns_defects_key():
    """Output must have 'defects' key — direct input to reforge()."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    manifest = client.attribute_frictions(
        product="Arabic social gaming platform with Ludo and voice rooms",
        frictions=["Ludo需要好友门槛高", "新手引导不足"],
        features=[
            {"id": "ludo",           "name": "Ludo",           "description": "multiplayer board game"},
            {"id": "invite_friends", "name": "Invite Friends", "description": "invite friends"},
            {"id": "tutorial",       "name": "Tutorial",       "description": "onboarding tutorial"},
        ],
    )
    assert "defects" in manifest
    assert isinstance(manifest["defects"], list)

def test_attribute_frictions_defect_shape():
    """Each defect must have type/severity/description/affected_screens/suggested_fix."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    manifest = client.attribute_frictions(
        product="social gaming app",
        frictions=["onboarding is confusing"],
        features=[{"id": "tutorial", "name": "Tutorial", "description": "first-run tutorial"}],
    )
    assert len(manifest["defects"]) >= 1
    d = manifest["defects"][0]
    assert "type" in d
    assert "severity" in d
    assert "description" in d
    assert "affected_screens" in d
    assert "suggested_fix" in d
    assert d["type"] in ("design", "ux", "rules")
    assert d["severity"] in ("P0", "P1", "P2")

def test_attribute_frictions_empty_frictions():
    """No frictions → empty defects list, no crash."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    manifest = client.attribute_frictions(
        product="any app", frictions=[], features=[]
    )
    assert manifest["defects"] == []

def test_attribute_frictions_game_name_propagated():
    """game_name and original_slug are passed through to manifest."""
    if not API_KEY:
        pytest.skip("no ANTHROPIC_API_KEY")
    from mcv.client import MCVClient
    client = MCVClient(api_key=API_KEY)
    manifest = client.attribute_frictions(
        product="Arabic gaming app",
        frictions=["gift center is confusing"],
        features=[{"id": "gift_center", "name": "Gift Center", "description": "virtual gifts"}],
        game_name="GAMZEE",
        original_slug="gamzee_v8",
    )
    assert manifest.get("game_name") == "GAMZEE"
    assert manifest.get("original_slug") == "gamzee_v8"
```

**Step 2: Run to verify fail**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_attribute_frictions.py::test_attribute_frictions_empty_frictions -v
```
Expected: FAIL with `AttributeError`

**Step 3: Implement `attribute_frictions()` in `client.py`**

```python
_ATTRIBUTE_PROMPT = """You are a product analyst mapping user friction reports to specific app features.

Product: {product_description}

Features in this product:
{features_block}

Observed user frictions (from behavioral simulation):
{frictions_block}

For each friction, identify the responsible feature(s) and generate a specific fix.
Map to the SMALLEST set of defects that covers all frictions.

Return ONLY valid JSON (no markdown):
{{
  "defects": [
    {{
      "type": "ux",
      "severity": "P1",
      "description": "clear description of what is broken",
      "affected_screens": ["screen_id_from_feature_id"],
      "suggested_fix": "concrete actionable fix instruction"
    }}
  ]
}}

Rules:
- type must be one of: "design", "ux", "rules"
- severity must be one of: "P0" (blocking), "P1" (major), "P2" (minor)
- affected_screens: use feature_id values as screen references
- suggested_fix: write what the designer should change, not what the problem is"""

def attribute_frictions(
    self,
    product: str,
    frictions: list[str],
    features: list[dict],
    game_name: str = "",
    original_slug: str = "",
) -> dict:
    """Map simulation friction themes to a reforge()-compatible defect manifest.

    One Sonnet call: friction themes + feature list → structured defect list.

    Args:
        product: PRD text or product description.
        frictions: friction theme strings from SimulationReport.friction_themes.
        features: [{"id": str, "name": str, "description": str}, ...]
        game_name: passed through to manifest (used by reforge()).
        original_slug: passed through to manifest (used by reforge()).

    Returns:
        dict with shape: {"defects": [...], "game_name": str, "original_slug": str}
        Directly usable as fix_requirement in eltm.reforge().
    """
    import json as _json
    import re as _re
    from mcv import core as _core

    if not frictions:
        return {"defects": [], "game_name": game_name, "original_slug": original_slug}

    features_block = "\n".join(f"- {f['id']}: {f['name']} — {f.get('description', '')}" for f in features) or "(none)"
    frictions_block = "\n".join(f"- {fr}" for fr in frictions)

    prompt = _ATTRIBUTE_PROMPT.format(
        product_description=product[:800],
        features_block=features_block,
        frictions_block=frictions_block,
    )

    raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=1500)

    defects: list[dict] = []
    m = _re.search(r'\{.*\}', raw, _re.DOTALL)
    if m:
        try:
            data = _json.loads(m.group())
            raw_defects = data.get("defects", [])
            for d in raw_defects:
                if not isinstance(d, dict):
                    continue
                # Enforce valid type/severity
                dtype = d.get("type", "design")
                if dtype not in ("design", "ux", "rules"):
                    dtype = "design"
                severity = d.get("severity", "P1")
                if severity not in ("P0", "P1", "P2"):
                    severity = "P1"
                defects.append({
                    "type": dtype,
                    "severity": severity,
                    "description": str(d.get("description", "")),
                    "affected_screens": list(d.get("affected_screens", [])),
                    "suggested_fix": str(d.get("suggested_fix", "")),
                })
        except (ValueError, _json.JSONDecodeError):
            pass

    return {
        "defects": defects,
        "game_name": game_name,
        "original_slug": original_slug,
    }
```

**Step 4: Run tests**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_attribute_frictions.py -v
```
Expected: 4 PASS

**Step 5: Commit**
```bash
cd /Users/michael/mcv && git add mcv/client.py mcv/tests/test_attribute_frictions.py && git commit -m "feat: add MCVClient.attribute_frictions() — friction → reforge() manifest"
```

---

## Task 5: Export new types from `mcv/__init__.py` + smoke test + push

**Files:**
- Modify: `mcv/__init__.py` (add exports)
- Create: `mcv/tests/test_bridge_smoke.py`

**Step 1: Write failing smoke test**

```python
# mcv/tests/test_bridge_smoke.py
def test_eltm_bridge_types_importable():
    """All three new capabilities and types are importable from mcv root."""
    from mcv import MCVClient, FeatureAAR, CoherenceReport
    assert MCVClient is not None
    assert FeatureAAR is not None
    assert CoherenceReport is not None

def test_eltm_bridge_methods_exist():
    """MCVClient exposes all three bridge methods."""
    from mcv import MCVClient
    assert hasattr(MCVClient, "research_aarrr")
    assert hasattr(MCVClient, "validate_coherence")
    assert hasattr(MCVClient, "attribute_frictions")
```

**Step 2: Run to verify fail**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/test_bridge_smoke.py::test_eltm_bridge_types_importable -v
```
Expected: FAIL with `ImportError`

**Step 3: Add exports to `mcv/__init__.py`**

Read current `__init__.py` first, then add:
```python
from mcv.report import FeatureAAR, CoherenceReport
```
alongside the existing `SimulationReport`, `CompareReport` exports.

**Step 4: Run all tests**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/ -v
```
Expected: all passing (API-gated tests skip without key)

**Step 5: Run full suite count check**
```bash
cd /Users/michael/mcv && .venv/bin/pytest tests/ --tb=short -q
```
Expected: no failures

**Step 6: Secret scan + commit + push**
```bash
cd /Users/michael/mcv
grep -rn "sk-\|ghp_\|AKIA\|Bearer \|password=\|secret=" mcv/ --include="*.py" | grep -v "test\|#" || echo "CLEAN"
git add mcv/__init__.py mcv/tests/test_bridge_smoke.py
git commit -m "feat: export FeatureAAR, CoherenceReport from mcv root"
git push origin HEAD
```

---

## Integration Reference

After all tasks complete, the full research loop looks like:

```python
import os
import eltm
from mcv import MCVClient

api_key = os.environ["ANTHROPIC_API_KEY"]
client = MCVClient(api_key=api_key)

# --- Position 1: ground AARRR scoring in population research ---
# (called from ELTM M1 or as pre-processing)
feature_aars = client.research_aarrr(
    product_description=prd_text,
    features=[{"id": f.id, "name": f.name, "description": f.description}
              for f in extracted_features],
)
# Map back: feature_aars[i].referral replaces the generic LLM referral score in M1

# --- Position 2: validate after Top-K selection ---
coherence = client.validate_coherence(
    product_description=prd_text,
    selected_features=[{"id": f.id, "name": f.name, "description": f.description}
                       for f in selected],
    dropped_features=[{"id": f.id, "name": f.name, "description": f.description}
                      for f in dropped],
)
if not coherence.is_coherent:
    print("Reinstating:", coherence.reinstate_recommendations)
    # Add back recommended features before proceeding to M2

# --- ELTM M2-M5 builds the wireframe ---
result = eltm.improve(prd_text, api_key=api_key, output_dir="/tmp/out")

# --- Position 3: friction → reforge() ---
sim = client.simulate(product=prd_text, user_type="Arabic mobile gamer", goal="Day-1 return?")
manifest = client.attribute_frictions(
    product=prd_text,
    frictions=sim.friction_themes,
    features=[{"id": s["id"], "name": s["name"], "description": ""}
              for s in result.get("screens", [])],
    game_name="GAMZEE",
)
reforged = eltm.reforge(fix_requirement=manifest, original_contract=result, api_key=api_key)
```
