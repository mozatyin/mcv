# MCV v2 Design — 5 Gap Resolution

**Date:** 2026-04-15  
**Status:** Approved  
**Primary caller:** ELTM `BattleStage` / `ForgeBattleLoop`

---

## Background

MCV (Monte Carlo Voter) is the shared persona-cohort decision engine for ELTM, Code-Soul,
and SoulGraph. Five design gaps were identified through usage in ELTM's Improve module:

1. No A/B comparison primitive
2. `SimulationReport.key_findings` never populated
3. `DomainConfig` only covers Game / App / Web — no extensibility
4. `PersonaDecider` and `UserSimulator` are parallel systems with no unified entry
5. `MetricResult` lacks stdev / confidence intervals

---

## Approach: Option A — In-place Enhancement

All changes are additive to the existing MCV library. No breaking changes to existing APIs.
ELTM's `battle.py` changes are < 10 lines.

---

## Section 1: Data Structure Changes

### `MetricResult` — add statistical fields

```python
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
    # NEW (Gap 5)
    stdev: float | None = None        # scale and bool both populated
    ci_95_low: float | None = None    # 95% confidence interval lower bound
    ci_95_high: float | None = None   # 95% confidence interval upper bound
    n_samples: int = 0                # number of parsed values
```

- **bool**: Wilson interval (accurate for small samples)
- **scale**: `mean ± 1.96 · stdev / √n`

### `CompareReport` — new dataclass in `report.py`

```python
@dataclass
class CompareReport:
    n_runs_per_variant: int
    variant_a_label: str
    variant_b_label: str
    variant_a: SimulationReport
    variant_b: SimulationReport
    deltas: dict[str, float]       # metric → B - A (bool: Δtrue_rate, scale: Δmean)
    improvements: list[str]        # metric names where delta exceeds variant_a CI
    regressions: list[str]         # metric names where delta is worse than variant_a CI
    key_diff: str                  # Haiku-generated one-line comparison summary
```

"Significant" = delta exceeds the 95% CI width of variant_a's metric.

### `SimulationReport` — two new computed properties

```python
@property
def day1_return_rate(self) -> float | None:
    """First bool metric's true_rate. Convenience for ELTM BattleStage."""

@property  
def friction_themes(self) -> list[str]:
    """First text metric's themes. Convenience for ELTM BattleStage."""

@property
def locked_schema(self) -> list[dict]:
    """Serialized metric schema for cross-round reuse."""
```

---

## Section 2: New Functions and Methods

### `UserSimulator.compare()` (Gap 1)

```python
def compare(
    self,
    product_a: str,
    product_b: str,
    label_a: str = "v_a",
    label_b: str = "v_b",
    n_runs: int = 30,
    locked_metrics: list[EvaluationMetric] | None = None,
) -> CompareReport:
```

**Key design decision:** Both variants share the same `ScenarioContext` seed sequence.
Generate N contexts first, run both products through identical scenarios.
This removes context noise — deltas reflect product differences only.

### `build_domain_config()` (Gap 3)

```python
# domain_configs.py
def build_domain_config(
    product_description: str,
    api_key: str,
    cache_path: Path | None = None,
) -> DomainConfig:
```

One Sonnet call generates `emotional_states`, `triggers`, `time_options`, `user_roles`.
Result is optionally cached to JSON to avoid repeated LLM calls.
Replaces `_pick_domain_from_prd()` keyword heuristic in ELTM's `battle.py`.

### `_generate_key_findings()` (Gap 2)

```python
# report.py
def _generate_key_findings(
    metrics: dict[str, MetricResult],
    user_type: str,
    api_key: str,
) -> str:
```

One Haiku call at the end of `aggregate()`.  
Only fires when `api_key` is not None and at least 2 metrics are present.
Produces 2–3 sentence product insight. No change to callers that pass `api_key=None`.

### `MCVClient` facade — new file `mcv/client.py` (Gap 4)

```python
class MCVClient:
    def __init__(self, api_key: str, mode: str = "fast"): ...

    def simulate(
        self,
        product: str,
        user_type: str,
        goal: str,
        domain_config: DomainConfig | None = None,  # None → auto build_domain_config
        n_runs: int = 60,
        locked_metrics: list | None = None,
    ) -> SimulationReport: ...

    def compare(
        self,
        product_a: str,
        product_b: str,
        user_type: str,
        goal: str,
        n_runs: int = 30,
        locked_metrics: list | None = None,
    ) -> CompareReport: ...

    def decide(
        self,
        question: str,
        options: list[str],
        context: str,
        personas: list | None = None,    # None → auto-generate from product context
    ) -> DecisionResult: ...
```

- `decide()` routes to `PersonaDecider`
- `simulate()` / `compare()` route to `UserSimulator`
- `domain_config=None` auto-calls `build_domain_config(product, api_key)`

---

## Section 3: ELTM Integration Changes

### `battle.py` — before (~40 lines of manual extraction)

```python
from mcv import UserSimulator
from mcv.schema_extractor import EvaluationMetric
sim = UserSimulator(user_type=..., domain_config=_pick_domain_from_prd(prd), api_key=...)
sim.prepare(product=prd_summary, goal=sim_goal, locked_metrics=locked_metrics)
sim_report = sim.simulate(n_runs=30).report()
for mr in metrics.values():
    if mr.type == "bool": data["day1_return_rate"] = mr.true_rate
for mr in metrics.values():
    if mr.type == "text": data["weaknesses"] = mr.themes
```

### `battle.py` — after (~10 lines)

```python
from mcv import MCVClient
client = MCVClient(self._api_key)
sim_report = client.simulate(
    product=prd_summary,
    user_type=_extract_user_type_from_prd(prd),
    goal=sim_goal,
    n_runs=30,
    locked_metrics=locked_metrics,
)
data["day1_return_rate"] = sim_report.day1_return_rate
data["weaknesses"]       = sim_report.friction_themes
data["key_findings"]     = sim_report.key_findings
data["_sim_schema"]      = sim_report.locked_schema
```

### `run_gamzee_pdca.py` — new compare usage in Round 1+

```python
compare_report = client.compare(
    product_a=prev_prd,
    product_b=new_prd,
    user_type=user_type,
    goal=sim_goal,
    n_runs=30,
    locked_metrics=locked_metrics,
)
print(f"  improvements: {compare_report.improvements}")
print(f"  regressions:  {compare_report.regressions}")
print(f"  {compare_report.key_diff}")
```

---

## Data Flow

```
MCVClient.simulate()
    └─ build_domain_config()         ← Gap 3 (auto-infer)
    └─ UserSimulator.prepare()
    └─ UserSimulator.simulate()      ← Haiku × N runs, temperature=1.0
    └─ aggregate()
        ├─ _aggregate_bool()         ← + Wilson CI        (Gap 5)
        ├─ _aggregate_scale()        ← + stdev / CI       (Gap 5)
        ├─ _aggregate_text()
        └─ _generate_key_findings()  ← Haiku 1 call       (Gap 2)
    └─ SimulationReport              ← .day1_return_rate / .friction_themes

MCVClient.compare()
    └─ generate shared ScenarioContext seeds
    └─ simulate(A) + simulate(B)     ← same seeds
    └─ CompareReport                 ← deltas + Wilson significance (Gap 1 + 5)

MCVClient.decide()
    └─ PersonaDecider                ← fast classify/score/validate (Gap 4)
```

---

## Files Changed

| File | Change type |
|------|-------------|
| `mcv/report.py` | Add CI fields to `MetricResult`, add `CompareReport`, add `_generate_key_findings()` |
| `mcv/domain_configs.py` | Add `build_domain_config()` |
| `mcv/user_simulator.py` | Add `UserSimulator.compare()` |
| `mcv/client.py` | New file — `MCVClient` facade |
| `mcv/__init__.py` | Export `MCVClient`, `CompareReport` |
| `eltm/stages/battle.py` | Replace manual MCV calls with `MCVClient` |

---

## Out of Scope

- Changes to Code-Soul or SoulGraph (benefit from MCVClient automatically once released)
- ELTM `run_gamzee_pdca.py` compare integration (optional follow-up)
- Visualization / HTML report output
