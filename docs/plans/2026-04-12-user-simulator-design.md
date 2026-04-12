# UserSimulator Design

**Goal:** A domain-agnostic behavioral simulation library. Initialize with a user type; simulate how that user responds to any product. Callable by ELTM, news reader, and any other project that needs to simulate user behavior.

**Core principle:** Roll the die N times at temperature=1.0 and observe what happens — don't ask for opinions. Empirical frequency distribution replaces LLM judgment.

---

## What question does it answer?

Any of:
- **A** — Which features/content does this user type actually use?
- **B** — Does this user come back after Day-1? What caused them to leave?
- **C** — Between design A and B, which does this user prefer?

The question is defined by the caller via `goal`. The simulator doesn't hardcode what to measure.

---

## API

```python
# 1. Initialize — define WHO to simulate
sim = UserSimulator(
    user_type="18岁阿拉伯竞技手游玩家，追求排名",
    domain_config=GameDomainConfig(),
    api_key=api_key,
)

# 2. Prepare — define WHAT to simulate + WHAT to evaluate
sim.prepare(
    product="一款阿拉伯社交棋牌游戏，核心是 Baloot",  # text / PRD / wireframe HTML
    goal="玩家第一天之后会不会回来？哪个环节让他们离开？",
    screen_id=None,  # optional: focus on one screen of a wireframe
)

# 3. Run + Report
report = sim.simulate(n_runs=60).report()
```

---

## Data Flow

```
prepare(product, goal)
  → LLM reads goal → generates EvaluationSchema (3-6 metrics)
      [
        { name: "day1_return",  type: bool,      question: "玩完第一局，你还想再来吗？" },
        { name: "drop_moment",  type: text,      question: "你在哪个时刻想关掉游戏？" },
        { name: "engagement",   type: scale_1_5, question: "这局游戏的投入程度？" },
      ]
  → stores schema + product internally

simulate(n_runs=60)
  → for each run:
      1. randomize ScenarioContext from DomainConfig
      2. build prompt: user_type + scenario + product + evaluation questions
      3. LLM narrates session at temperature=1.0 (Haiku model)
      4. parse structured output per EvaluationSchema → SessionResult
  → stores list[SessionResult]

report()
  → aggregate N SessionResults per metric
  → SimulationReport with distributions + LLM key_findings synthesis
```

---

## DomainConfig

Controls the "world" of each simulated session. Pre-built configs ship with the package; callers can also pass custom configs.

```python
@dataclass
class DomainConfig:
    session_framing: str          # "你开始了一局游戏" / "你打开了 app" / "你在刷新闻"
    emotional_states: list[str]
    triggers: list[str]
    time_options: list[str]
    user_roles: dict[str, list[int]]  # role → usage_day ranges

# Pre-built
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

---

## Prompt Structure (per session)

```
你是：{user_type}

你的背景：
{persona context — motivations, pain_points, cohort}

现在的情况：
  时间：{time_option}
  状态：{emotional_state}
  触发：{trigger}
  使用天数：{usage_day}

{session_framing}：
{product_description or wireframe HTML}

叙述你接下来的 6-8 个操作。只写你做了什么，不写感想。
用第三人称："{name} 点击了..."，"{name} 滑过了..."

叙述完毕后，每行回答一个问题：
{metric_1_name}: {metric_1_question}
{metric_2_name}: {metric_2_question}
{metric_3_name}: {metric_3_question}
```

Key constraints:
- `temperature=1.0` — hardcoded, never 0. This is what makes it Monte Carlo.
- Model: `claude-haiku-4-5-20251001` — cheap enough for 60 calls (~$0.04/full run)
- Narrative: third person, behavior only, no opinions

---

## EvaluationSchema

Generated from `goal` by one LLM call before simulations begin.

```python
@dataclass
class EvaluationMetric:
    name: str
    type: str              # "bool" | "scale_1_5" | "text"
    question: str          # what to ask the simulated user at end of session
```

Metric types:
- **bool** → `yes/no` answer → aggregates to `true_rate` (0.0–1.0)
- **scale_1_5** → numeric rating → aggregates to `mean` + `distribution`
- **text** → free text → aggregates to `themes` (LLM cluster) + `samples`

---

## Output: SimulationReport

```python
@dataclass
class MetricResult:
    name: str
    type: str
    # bool
    true_rate: float | None
    # scale
    mean: float | None
    distribution: dict[int, float] | None   # {1: 0.05, 2: 0.15, 3: 0.40, ...}
    # text
    themes: list[str] | None                # ["教程太长", "第一局必须赢"]
    samples: list[str] | None               # up to 10 raw text samples

@dataclass
class SimulationReport:
    n_simulations: int
    user_type: str
    product_summary: str                    # first 100 chars of product description
    metrics: dict[str, MetricResult]
    key_findings: str                       # LLM-synthesized paragraph
```

Example output for Day-1 retention question:
```
n_simulations: 60
metrics:
  day1_return:  { true_rate: 0.62 }
  drop_moment:  { themes: ["教程太长", "第一局输了无引导", "看不懂界面"] }
  engagement:   { mean: 3.1, distribution: {1:5%, 2:15%, 3:40%, 4:30%, 5:10%} }
key_findings: "62% 的模拟玩家有回访意愿。主要流失点在第一局失败后缺乏正向反馈，
              以及教程时长超过玩家耐心阈值。高参与度（4-5分）占40%，
              说明核心机制有吸引力，但首次体验摩擦过大。"
```

---

## Package Structure

Upgrade to existing `mcv` package. Old `PersonaSimulator` preserved — no breaking changes.

```
mcv/
  user_simulator.py    ← NEW: UserSimulator main class
  domain_configs.py    ← NEW: GameDomainConfig / AppDomainConfig / WebDomainConfig
  schema_extractor.py  ← NEW: goal → EvaluationSchema (one LLM call)
  report.py            ← NEW: list[SessionResult] → SimulationReport aggregation
  simulator.py         ← KEEP: PersonaSimulator (legacy, unchanged)
  scenarios.py         ← KEEP: ScenarioContext (reused by new code)
  cache.py             ← KEEP: three-tier cache (reused by new code)
  core.py              ← KEEP: _llm_call (shared)
```

---

## Integration Examples

**ELTM (game analysis):**
```python
sim = UserSimulator(user_type=prd_target_audience, domain_config=GameDomainConfig(), api_key=api_key)
sim.prepare(product=prd_text, goal="哪些功能会被使用？用户留存信号如何？")
report = sim.simulate(n_runs=60).report()
```

**News reader (brain2):**
```python
sim = UserSimulator(user_type="25岁都市白领，早晨通勤", domain_config=WebDomainConfig(), api_key=api_key)
sim.prepare(product=story_list_description, goal="用户会点哪类新闻？哪些会被跳过？")
report = sim.simulate(n_runs=30).report()
```

**Wireframe validation:**
```python
sim = UserSimulator(user_type="阿拉伯家庭主妇，首次使用理财app", domain_config=AppDomainConfig(), api_key=api_key)
sim.prepare(product=wireframe_html, screen_id="onboarding_screen", goal="用户能完成注册流程吗？在哪里会卡住？")
report = sim.simulate(n_runs=40).report()
```
