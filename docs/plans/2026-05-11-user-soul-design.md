# User-Soul Design Document

> Date: 2026-05-11
> Status: Approved
> Replaces: MCV (Monte Carlo Voter)

## Overview

User-Soul is the user advocate across the entire product lifecycle. It simulates
target users at four stages of development, providing feedback that ranges from
market research (S1) to launch taste validation (S5).

**No existing system — commercial or academic — does multi-stage user simulation.**
Blok does design-only, Synthetic Users does research-only, PlaytestCloud does
testing-only. User-Soul is the first to unify S1→S2→S4→S5 with a shared persona
pool.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| LLM coupling | Model-agnostic (LLMBackend Protocol) | Value is in methodology, not model binding |
| Rename strategy | In-place mcv → user-soul | Few external callers, one-time fix |
| Sleeping modules | Activate then iterate | Working code with tests; optimize in PDCA |
| S5 verdict | Recommend + human confirm | AI cannot be final judge (Constitution §4) |
| Architecture | Layered (engines → stages → client) | Engines shared across stages (§9 modularity) |
| Visual evaluation | Pairwise comparison > absolute scoring | 93% accuracy vs 35% (research consensus) |

## Architecture

```
┌─────────────────────────────────────┐
│         UserSoulClient              │  Unified entry point
│  .research() .review() .verify()    │  ELTM/PM-Soul call this
│  .launch()                          │
├─────────────────────────────────────┤
│         Stage Orchestrators         │  Combine engines per stage
│  S1: ResearchPanel                  │
│  S2: DesignReview                   │
│  S4: ModuleUAT                      │
│  S5: LaunchGate                     │
├─────────────────────────────────────┤
│          Core Engines               │  Reusable atomic capabilities
│  PersonaEngine (population+pool)    │
│  BehaviorEngine (simulator+report)  │
│  VisionEngine (screenshot compare)  │
│  VoteEngine (decider+aarrr)         │
├─────────────────────────────────────┤
│         LLMBackend Protocol         │  Model-agnostic abstraction
│  .text()  .vision()                 │
└─────────────────────────────────────┘
```

## Layer 1: LLMBackend Protocol

```python
class LLMBackend(Protocol):
    def text(self, prompt: str, *,
             max_tokens: int = 512,
             temperature: float = 0.0,
             model_tier: str = "fast") -> str: ...

    def vision(self, prompt: str, images: list[bytes], *,
               max_tokens: int = 512,
               temperature: float = 0.0,
               model_tier: str = "smart") -> str: ...
```

- `model_tier`: "fast" (Haiku-class) or "smart" (Sonnet-class). User-Soul never
  names a specific model.
- `images`: raw bytes. Backend converts to provider format.
- Returns plain `str`. Token accounting is Backend's concern.
- First implementation: `AnthropicBackend` (from existing `core.py._llm_call`).

## Layer 2: Core Engines

### PersonaEngine

Source: `population.py` (PopulationResearcher + PersonaPool).
Change: `_llm_call` → `backend.text()`.

```python
class PersonaEngine:
    def research(self, product_description: str) -> PersonaStructure
    def generate_pool(self, structure: PersonaStructure, n: int) -> list[AgentProfile]
    def get_or_create(self, product_description: str, n: int = 12) -> list[AgentProfile]
```

Shared across all four stages. `get_or_create()` generates once, passed to all
stages so the same "users" participate from S1 to S5.

### BehaviorEngine

Source: `user_simulator.py` + `journey.py` + `report.py` + `behavioral_framework.py`.
Change: `_llm_call` → `backend.text()`.

```python
class BehaviorEngine:
    def simulate(self, product, personas, metrics, *, n_runs=30, adversarial=True) -> SimulationReport
    def compare(self, product_a, product_b, personas, metrics, *, n_runs=30) -> CompareReport
    def simulate_journey(self, screens, target_flow, personas, *, n_personas=12) -> JourneyReport
```

Includes sycophancy deflator (×0.70), adversarial persona pass, Fogg/Hook/Peak-End
framework. All proven in production.

### VisionEngine (NEW)

The core new capability. Zero existing code — built from scratch.

```python
class VisionEngine:
    def pairwise_compare(self, ours: bytes, theirs: bytes, *,
                         dimensions: list[str] | None = None) -> PairwiseResult
    def batch_compare(self, ours: bytes,
                      competitors: list[tuple[str, bytes]], *,
                      dimensions: list[str] | None = None) -> list[PairwiseResult]
    def screenshot_review(self, screenshot: bytes, *,
                          context: str = "",
                          checklist: list[str] | None = None) -> ReviewResult
```

Research basis:
- Pairwise comparison: 93% accuracy for high-difference pairs (MLLM-as-Judge 2025)
- Absolute scoring: only 35% exact match — never use for taste judgment
- Default dimensions: visual polish, color harmony, information hierarchy, professionalism
- Position bias ~5% — mitigate by randomizing left/right order

### VoteEngine

Source: `core.py` (PersonaDecider) + `client.py` (research_aarrr).
Change: `_llm_call` → `backend.text()`.

```python
class VoteEngine:
    def classify(self, question, options, context, personas) -> DecisionResult
    def score(self, question, lo, hi, context, personas) -> DecisionResult
    def aarrr(self, product, features, archetypes) -> list[FeatureAAR]
```

## Layer 3: Stage Orchestrators

### S1: ResearchPanel — "ELTM invites a focus group"

```
Input:  product_description + features + competitor_screenshots (optional)
Output: ResearchReport (persona_structure, feature_priorities, visual_preferences, latent_needs)
Gate:   None (advisory to ELTM)
Uses:   PersonaEngine + VisionEngine + VoteEngine
```

1. PopulationResearcher generates persona structure
2. AARRR vote on features
3. Pairwise preference on competitor screenshots (if provided)
4. Coherence check on feature dependencies

### S2: DesignReview — "Usability test on wireframes"

```
Input:  screens + target_flow + wireframe_screenshots (optional) + competitor_screenshots (optional)
Output: DesignReviewReport (journey, layout_reviews, competitor_gaps, passes_gate)
Gate:   completion_rate >= 0.70 (from simulate_journey)
Uses:   PersonaEngine + BehaviorEngine + VisionEngine
```

1. Journey simulation — personas walk target flow
2. VLM layout review on wireframe screenshots
3. Pairwise compare wireframe vs competitor layouts

### S4: ModuleUAT — "Functional acceptance testing"

```
Input:  product_description + html_screenshots (optional) + features (optional)
Output: ModuleUATReport (behavior, visual_issues, friction_manifest, passes_gate)
Gate:   No P0 frictions
Uses:   PersonaEngine + BehaviorEngine + VisionEngine
```

1. Behavioral simulation with adversarial pass
2. VLM screenshot quality check
3. Friction attribution → defect manifest for Code-Soul

### S5: LaunchGate — "Pre-launch taste validation"

```
Input:  product_screenshots + competitor_screenshots + product_description
Output: LaunchReport (taste_results, taste_win_rate, behavior, recommendation, improvement_areas)
Gate:   Recommendation = SHIP / IMPROVE / ABANDON — HUMAN CONFIRMS
Uses:   PersonaEngine + BehaviorEngine + VisionEngine
```

1. VLM pairwise taste comparison vs each competitor
2. Full behavioral simulation with day1_return
3. Composite judgment → recommendation
4. Human makes final call

Judgment logic:
- taste_win_rate < 0.3 → "visual taste below competitors"
- day1_return_adjusted < 0.10 → "below survival threshold"
- No issues → SHIP
- Both fail → ABANDON
- One fails → IMPROVE

## File Structure

```
~/user-soul/
├── user_soul/
│   ├── __init__.py
│   ├── client.py                # UserSoulClient
│   ├── backend.py               # LLMBackend Protocol
│   ├── models.py                # All dataclasses
│   ├── calibration.py           # Sycophancy deflator + calibration
│   ├── framework.py             # Fogg/Hook/Peak-End constants
│   ├── backends/
│   │   ├── __init__.py
│   │   └── anthropic.py         # ← core.py _llm_call
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── persona.py           # ← population.py
│   │   ├── behavior.py          # ← user_simulator + journey + report
│   │   ├── vision.py            # NEW
│   │   └── vote.py              # ← core.py PersonaDecider
│   └── stages/
│       ├── __init__.py
│       ├── research.py          # S1
│       ├── design_review.py     # S2
│       ├── module_uat.py        # S4
│       └── launch.py            # S5
├── tests/
│   ├── test_backend.py
│   ├── test_persona_engine.py
│   ├── test_behavior_engine.py
│   ├── test_vision_engine.py
│   ├── test_vote_engine.py
│   ├── test_research.py
│   ├── test_design_review.py
│   ├── test_module_uat.py
│   └── test_launch.py
└── docs/
    └── plans/
```

## Migration Map

| MCV file | Destination | Change |
|----------|-------------|--------|
| core.py → PersonaDecider | engines/vote.py | _llm_call → backend.text() |
| core.py → _llm_call | backends/anthropic.py | Extract to class |
| population.py | engines/persona.py | _llm_call → backend.text() |
| user_simulator.py | engines/behavior.py | _llm_call → backend.text() |
| journey.py | engines/behavior.py | Merge |
| report.py | engines/behavior.py (internal) | No change |
| behavioral_framework.py | framework.py | No change |
| client.py → MCVClient | client.py → UserSoulClient | Rewrite (compose engines) |
| schema_extractor.py | engines/behavior.py (internal) | No change |
| domain_configs.py | engines/behavior.py (internal) | No change |
| scenarios.py | engines/behavior.py (internal) | No change |
| personas.py | DELETE (replaced by PersonaEngine) | — |
| simulator.py | DELETE (replaced by BehaviorEngine) | — |
| cache.py | DELETE (rebuild if needed) | — |
| gate_ledger.py | DELETE (replaced by stage Reports) | — |
| __main__.py | DELETE (rebuild CLI if needed) | — |

## External Caller Adaptation

PM-Soul `mcv_bridge.py`:
```python
# Before:
sys.path.insert(0, "/Users/michael/mcv")
from mcv import MCVClient

# After:
from user_soul import UserSoulClient
from user_soul.backends.anthropic import AnthropicBackend
backend = AnthropicBackend(api_key=self.api_key)
client = UserSoulClient(backend)
```

ELTM battle/intelligence: same pattern.

## Research Basis

Key papers and findings that informed this design:

- **Elicitron (Stanford/Autodesk 2024)**: LLM agents find +67% more latent needs
  than human interviews. Basis for S1 ResearchPanel.
- **Stanford/DeepMind Digital Twins (2024)**: Rich background stories yield 85%
  accuracy vs 71% for demographic labels. Basis for PersonaEngine design.
- **AgentA/B (2025)**: Direction correct but magnitude off by 30-40%.
  Basis for sycophancy deflator.
- **MLLM as UI Judge (2025)**: Pairwise 93% accuracy (high diff), absolute only 35%.
  Basis for VisionEngine using pairwise comparison exclusively.
- **AesEval-Bench (Microsoft 2026)**: GPT-5 at 72.5% aesthetic judgment.
  Reasoning models NOT better than standard models for aesthetics.
- **Population-Aligned Personas (Microsoft 2025)**: Importance sampling reduces
  population-level bias. Future upgrade for PersonaEngine.
- **NN/G (2024-2025)**: Synthetic users good for hypothesis generation, bad for
  final decisions. Basis for S5 human-confirm requirement.

## Constitution Alignment

| Article | How User-Soul satisfies it |
|---------|---------------------------|
| §1 Build factory | User-Soul is a generic tool, zero game-specific logic |
| §4 AI cannot self-evaluate | User-Soul is independent from Code-Soul; S5 requires human confirm |
| §7 Anti-pollution | No domain keywords; product context comes from caller |
| §8 PDCA | S5 IMPROVE feeds back to S2/S3 forming outer PDCA loop |
| §9 Linux modularity | 4 engines × 4 stages, each independently testable |
| §11 Jidoka | Launch Gate = outermost ring; defective products don't ship |
