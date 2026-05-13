"""Knowledge-Graded Virtual Testing — tiered game knowledge for differential UX diagnosis.

Three knowledge tiers (NOVICE / CASUAL / INFORMED) extract structured subsets
of a GDD's formal_rules and produce tier-appropriate LLM briefings.  Running
all three tiers on the same game yields a DifferentialDiagnosis that routes
issues to the correct owner (Code-Soul, ELTM, or PM-Soul).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class KnowledgeTier(str, Enum):
    NOVICE = "novice"
    CASUAL = "casual"
    INFORMED = "informed"

    @classmethod
    def all_tiers(cls) -> list["KnowledgeTier"]:
        return [cls.NOVICE, cls.CASUAL, cls.INFORMED]


@dataclass
class GameKnowledge:
    game_name: str = ""
    game_category: str = ""

    # CASUAL tier and above
    objective: str = ""
    turn_structure: str = ""
    win_conditions: list[str] = field(default_factory=list)
    game_loop_type: str = ""

    # INFORMED tier only
    interaction_rules: list[str] = field(default_factory=list)
    setup_info: str = ""
    scoring_info: str = ""
    special_constraints: list[str] = field(default_factory=list)

    # NEVER exposed
    strategy_hints: list[str] = field(default_factory=list)

    @classmethod
    def from_gdd(cls, gdd: dict) -> "GameKnowledge":
        rules = gdd.get("formal_rules") or gdd.get("game_rules") or {}
        if not rules:
            return cls(game_name=gdd.get("game_name", ""))

        game_name = gdd.get("game_name", "")
        game_loop = rules.get("game_loop_type", "")
        game_category = _infer_category(gdd, game_loop)

        raw_wins = rules.get("win_conditions", [])
        if isinstance(raw_wins, str):
            raw_wins = [raw_wins]
        win_conditions = [w for w in raw_wins if isinstance(w, str)]

        specials = rules.get("special_rules", [])
        interaction_rules = []
        special_constraints = []
        strategy_hints = []

        for rule in specials:
            if not isinstance(rule, str):
                continue
            if _is_strategy(rule):
                strategy_hints.append(rule)
            elif _is_interaction_constraint(rule):
                interaction_rules.append(rule)
            else:
                special_constraints.append(rule)

        setup = rules.get("setup", {})
        setup_info = ""
        if isinstance(setup, dict):
            parts = []
            if setup.get("players"):
                parts.append(f"Players: {setup['players']}")
            if setup.get("initial_state"):
                parts.append(f"Initial state: {setup['initial_state']}")
            setup_info = ". ".join(parts)
        elif isinstance(setup, str):
            setup_info = setup

        scoring = rules.get("scoring", {})
        scoring_info = ""
        if isinstance(scoring, dict):
            method = scoring.get("method", "")
            details = scoring.get("details", "")
            scoring_info = f"{method}: {details}" if method else details
        elif isinstance(scoring, str):
            scoring_info = scoring

        return cls(
            game_name=game_name,
            game_category=game_category,
            objective=rules.get("objective", ""),
            turn_structure=rules.get("turn_structure", ""),
            win_conditions=win_conditions,
            game_loop_type=game_loop,
            interaction_rules=interaction_rules,
            setup_info=setup_info,
            scoring_info=scoring_info,
            special_constraints=special_constraints,
            strategy_hints=strategy_hints,
        )


_NOVICE_COGNITIVE_PREAMBLE = (
    "You have NEVER played this game before. You do not know the rules.\n"
    "You know only one thing: this is a {category}.\n\n"
    "CRITICAL INSTRUCTIONS — you must follow these exactly:\n"
    "1. LOOK at the screenshot FIRST. Describe what you see before deciding.\n"
    "2. Do NOT assume any rules. If you are unsure what clicking something "
    "does, that confusion IS valuable feedback.\n"
    "3. Base every decision ONLY on what is visually apparent: labels, "
    "colors, layout, highlighted elements, text on screen.\n"
    "4. If nothing on screen tells you what to do next, report that as "
    "your reason — this means the game lacks guidance.\n"
    "5. Do NOT use any prior knowledge of {game_name} or similar games. "
    "Pretend this is a completely alien interface.\n"
    "6. When you give_up, describe exactly what was confusing or missing "
    "from the UI that prevented you from continuing.\n"
)

_CASUAL_PREAMBLE = (
    "You have a basic understanding of this {category}.\n\n"
    "What you know:\n"
    "- Objective: {objective}\n"
    "- How turns work: {turn_structure}\n"
    "- How to win: {win_conditions}\n\n"
    "What you do NOT know:\n"
    "- Specific interaction details (how exactly to make moves in this UI)\n"
    "- Any constraints beyond the basics above\n"
    "- Any strategy or tactics\n\n"
    "Try to figure out the interaction details from the UI itself. "
    "If something is unclear, that confusion is valuable feedback — "
    "describe what you expected vs what happened.\n"
)

_INFORMED_PREAMBLE = (
    "You know the complete rules of {game_name}.\n\n"
    "Rules:\n"
    "- Objective: {objective}\n"
    "- Turn structure: {turn_structure}\n"
    "- Win conditions: {win_conditions}\n"
    "- Interaction rules: {interaction_rules}\n"
    "- Setup: {setup_info}\n\n"
    "You know HOW to play — you do NOT know strategy or tactics.\n"
    "Do not try to play optimally. Just play valid moves.\n\n"
    "Focus on: Does the UI let you execute the rules you know? "
    "If a rule says you can do X but the UI makes it hard or impossible, "
    "that is friction — report it.\n"
)


def brief_for_tier(knowledge: GameKnowledge, tier: KnowledgeTier) -> str:
    if tier == KnowledgeTier.NOVICE:
        return _brief_novice(knowledge)
    elif tier == KnowledgeTier.CASUAL:
        return _brief_casual(knowledge)
    elif tier == KnowledgeTier.INFORMED:
        return _brief_informed(knowledge)
    raise ValueError(f"Unknown tier: {tier}")


def _brief_novice(k: GameKnowledge) -> str:
    return _NOVICE_COGNITIVE_PREAMBLE.format(
        category=k.game_category or "game",
        game_name=k.game_name or "this game",
    )


def _brief_casual(k: GameKnowledge) -> str:
    win_str = "; ".join(k.win_conditions[:4]) if k.win_conditions else "(unknown)"
    return _CASUAL_PREAMBLE.format(
        category=k.game_category or "game",
        objective=k.objective or "(unknown)",
        turn_structure=k.turn_structure or "(unknown)",
        win_conditions=win_str,
    )


def _brief_informed(k: GameKnowledge) -> str:
    win_str = "; ".join(k.win_conditions[:4]) if k.win_conditions else "(none specified)"
    interaction_str = "; ".join(k.interaction_rules[:8]) if k.interaction_rules else "(none specified)"
    return _INFORMED_PREAMBLE.format(
        game_name=k.game_name or "this game",
        objective=k.objective or "(unknown)",
        turn_structure=k.turn_structure or "(unknown)",
        win_conditions=win_str,
        interaction_rules=interaction_str,
        setup_info=k.setup_info or "(standard setup)",
    )


# --- Differential Diagnosis ---

class DiagnosisCategory(str, Enum):
    CODE_BUG = "code_bug"
    TUTORIAL_NEEDED = "tutorial_needed"
    DISCOVERABILITY = "discoverability"
    HELP_SKIPPABLE = "help_skippable"
    FLOW_FRICTION = "flow_friction"
    INCONCLUSIVE = "inconclusive"


@dataclass
class TierResult:
    tier: KnowledgeTier
    score: float
    completed_rate: float
    gave_up_rate: float
    friction_count: int
    issue_kinds: dict[str, int] = field(default_factory=dict)
    failure_turns: list[int] = field(default_factory=list)
    raw_issues: list[Any] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.gave_up_rate > 0.5 or self.score < 50

    @property
    def struggled(self) -> bool:
        return not self.failed and (self.gave_up_rate > 0.2 or self.score < 70)

    @property
    def smooth(self) -> bool:
        return self.score >= 70 and self.gave_up_rate <= 0.2


@dataclass
class DiagnosisItem:
    category: DiagnosisCategory
    description: str
    evidence: dict[str, str] = field(default_factory=dict)
    owner: str = ""
    severity: str = "P1"
    common_failure_turn: int | None = None


@dataclass
class DifferentialDiagnosis:
    game_name: str
    tier_results: dict[str, TierResult]
    diagnoses: list[DiagnosisItem] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_tier_results(
        cls,
        game_name: str,
        tier_results: dict[str, TierResult],
    ) -> "DifferentialDiagnosis":
        novice = tier_results.get(KnowledgeTier.NOVICE.value)
        casual = tier_results.get(KnowledgeTier.CASUAL.value)
        informed = tier_results.get(KnowledgeTier.INFORMED.value)

        diagnoses: list[DiagnosisItem] = []

        if not all([novice, casual, informed]):
            return cls(
                game_name=game_name,
                tier_results=tier_results,
                diagnoses=[],
                summary=f"Partial run ({sum(1 for v in tier_results.values() if v)}/3 tiers).",
            )

        # Pattern 1: ALL tiers fail -> CODE BUG
        if novice.failed and casual.failed and informed.failed:
            common_turns = _find_common_failure_turns(
                novice.failure_turns, casual.failure_turns, informed.failure_turns
            )
            diagnoses.append(DiagnosisItem(
                category=DiagnosisCategory.CODE_BUG,
                description=(
                    "All knowledge tiers fail — the game is broken regardless of "
                    "player knowledge. This is a code bug, not a UX issue."
                ),
                evidence={
                    "novice": f"score={novice.score:.0f}, gave_up={novice.gave_up_rate:.0%}",
                    "casual": f"score={casual.score:.0f}, gave_up={casual.gave_up_rate:.0%}",
                    "informed": f"score={informed.score:.0f}, gave_up={informed.gave_up_rate:.0%}",
                },
                owner="code-soul",
                severity="P0",
                common_failure_turn=common_turns[0] if common_turns else None,
            ))

        # Pattern 2: Only NOVICE fails -> TUTORIAL NEEDED
        elif novice.failed and not casual.failed and not informed.failed:
            diagnoses.append(DiagnosisItem(
                category=DiagnosisCategory.TUTORIAL_NEEDED,
                description=(
                    "Only zero-knowledge users fail — the game cannot teach itself. "
                    "A tutorial or onboarding flow is mandatory."
                ),
                evidence={
                    "novice": f"score={novice.score:.0f}, gave_up={novice.gave_up_rate:.0%}",
                    "casual": f"score={casual.score:.0f} (OK)",
                    "informed": f"score={informed.score:.0f} (OK)",
                },
                owner="eltm",
                severity="P1",
            ))

        # Pattern 3: NOVICE + CASUAL fail, INFORMED OK -> DISCOVERABILITY
        elif novice.failed and casual.failed and not informed.failed:
            diagnoses.append(DiagnosisItem(
                category=DiagnosisCategory.DISCOVERABILITY,
                description=(
                    "Users with partial knowledge cannot discover interaction patterns. "
                    "The UI needs hints, tooltips, or progressive disclosure."
                ),
                evidence={
                    "novice": f"score={novice.score:.0f}, gave_up={novice.gave_up_rate:.0%}",
                    "casual": f"score={casual.score:.0f}, gave_up={casual.gave_up_rate:.0%}",
                    "informed": f"score={informed.score:.0f} (smooth)",
                },
                owner="eltm",
                severity="P0",
            ))

        # Pattern 4: Only INFORMED smooth -> help works but traps experts
        elif novice.failed and casual.struggled and informed.smooth:
            diagnoses.append(DiagnosisItem(
                category=DiagnosisCategory.HELP_SKIPPABLE,
                description=(
                    "Help system is effective but may need to be skippable for "
                    "experienced users. Casual tier struggles suggest hints need enhancement."
                ),
                evidence={
                    "novice": f"score={novice.score:.0f}",
                    "casual": f"score={casual.score:.0f} (struggling)",
                    "informed": f"score={informed.score:.0f} (smooth)",
                },
                owner="pm-soul",
                severity="P2",
            ))

        # Pattern 5: INFORMED fails or struggles -> FLOW FRICTION
        if informed.failed or informed.struggled:
            already_code_bug = any(
                d.category == DiagnosisCategory.CODE_BUG for d in diagnoses
            )
            if not already_code_bug:
                diagnoses.append(DiagnosisItem(
                    category=DiagnosisCategory.FLOW_FRICTION,
                    description=(
                        "Users who know all the rules still experience friction. "
                        "The UI is blocking valid rule-following actions."
                    ),
                    evidence={
                        "informed": (
                            f"score={informed.score:.0f}, "
                            f"gave_up={informed.gave_up_rate:.0%}, "
                            f"friction_count={informed.friction_count}"
                        ),
                    },
                    owner="code-soul",
                    severity="P1" if informed.struggled else "P0",
                ))

        # Pattern 6: NOVICE struggles but doesn't fail -> partial self-teaching
        if novice.struggled and not novice.failed:
            diagnoses.append(DiagnosisItem(
                category=DiagnosisCategory.TUTORIAL_NEEDED,
                description=(
                    "Zero-knowledge users struggle but eventually muddle through. "
                    "Consider adding contextual hints at struggle points."
                ),
                evidence={
                    "novice": (
                        f"score={novice.score:.0f}, "
                        f"gave_up={novice.gave_up_rate:.0%}, "
                        f"friction_count={novice.friction_count}"
                    ),
                },
                owner="eltm",
                severity="P2",
            ))

        if not diagnoses:
            diagnoses.append(DiagnosisItem(
                category=DiagnosisCategory.INCONCLUSIVE,
                description="Cross-tier pattern does not match known diagnostic matrix.",
                evidence={
                    t: f"score={r.score:.0f}, gave_up={r.gave_up_rate:.0%}"
                    for t, r in tier_results.items() if r is not None
                },
                owner="pm-soul",
                severity="P2",
            ))

        summary = _build_diagnosis_summary(tier_results, diagnoses)

        return cls(
            game_name=game_name,
            tier_results=tier_results,
            diagnoses=diagnoses,
            summary=summary,
        )


def distribute_personas_across_tiers(
    personas: list[Any],
    tiers: list[KnowledgeTier] | None = None,
) -> dict[str, list[Any]]:
    tiers = tiers or KnowledgeTier.all_tiers()
    if not personas or not tiers:
        return {t.value: [] for t in tiers}

    distribution: dict[str, list[Any]] = {t.value: [] for t in tiers}
    n_tiers = len(tiers)
    per_tier = len(personas) // n_tiers
    remainder = len(personas) % n_tiers

    idx = 0
    for i, tier in enumerate(tiers):
        count = per_tier + (1 if i < remainder else 0)
        distribution[tier.value] = personas[idx : idx + count]
        idx += count

    return distribution


def minimum_personas_for_graded_test(n_tiers: int = 3) -> int:
    return n_tiers


# --- Internal helpers ---

_STRATEGY_KEYWORDS = [
    "strategically", "advantageous", "weaker", "best defense",
    "best strategic", "optimal", "optimally", "fork",
    "blocking", "priority over", "force at least", "defense",
    "should take", "typically last", "second-best",
    "strong strategy", "prevents opponent", "when ahead",
    "good position", "control the",
]

_INTERACTION_KEYWORDS = [
    "cannot", "must", "only", "may not", "not allowed",
    "first", "always goes", "ends immediately", "once per",
    "skip", "invalid", "select", "click", "move", "capture",
    "jump", "place", "mark",
]


def _is_strategy(rule: str) -> bool:
    lower = rule.lower()
    return any(kw in lower for kw in _STRATEGY_KEYWORDS)


def _is_interaction_constraint(rule: str) -> bool:
    lower = rule.lower()
    return any(kw in lower for kw in _INTERACTION_KEYWORDS)


def _infer_category(gdd: dict, game_loop: str) -> str:
    render = gdd.get("render_hints", {})
    board_type = render.get("board_type", "")
    render_mode = render.get("render_mode", "")

    if "grid" in board_type.lower() or "grid" in render_mode.lower():
        return "board game"
    if "card" in board_type.lower() or "card" in render_mode.lower():
        return "card game"
    if "canvas" in render_mode.lower():
        return "action game" if game_loop == "real-time" else "strategy game"
    if game_loop == "turn-based":
        return "turn-based game"
    if game_loop == "real-time":
        return "real-time game"
    return "game"


def _find_common_failure_turns(
    *turn_lists: list[int],
    tolerance: int = 2,
) -> list[int]:
    all_turns: list[int] = []
    for tl in turn_lists:
        all_turns.extend(tl)
    if not all_turns:
        return []

    all_turns.sort()
    clusters: list[list[int]] = []
    current_cluster = [all_turns[0]]

    for t in all_turns[1:]:
        if t - current_cluster[-1] <= tolerance:
            current_cluster.append(t)
        else:
            clusters.append(current_cluster)
            current_cluster = [t]
    clusters.append(current_cluster)

    common: list[int] = []
    for cluster in clusters:
        sources: set[int] = set()
        for tl_idx, tl in enumerate(turn_lists):
            if any(t in cluster for t in tl):
                sources.add(tl_idx)
        if len(sources) >= 2:
            median = sorted(cluster)[len(cluster) // 2]
            common.append(median)

    return common


def _build_diagnosis_summary(
    tier_results: dict[str, TierResult],
    diagnoses: list[DiagnosisItem],
) -> str:
    lines = ["=== Differential Diagnosis ===\n"]

    lines.append("Tier Scores:")
    for tier_name in [KnowledgeTier.NOVICE.value, KnowledgeTier.CASUAL.value,
                      KnowledgeTier.INFORMED.value]:
        result = tier_results.get(tier_name)
        if result is None:
            lines.append(f"  {tier_name.upper():10s}  (not run)")
            continue
        status = (
            "SMOOTH" if result.smooth
            else ("STRUGGLED" if result.struggled else "FAILED")
        )
        lines.append(
            f"  {tier_name.upper():10s}  score={result.score:.0f}  "
            f"completed={result.completed_rate:.0%}  "
            f"gave_up={result.gave_up_rate:.0%}  [{status}]"
        )

    lines.append("")

    if diagnoses:
        lines.append("Diagnoses:")
        for d in diagnoses:
            lines.append(f"  [{d.severity}] {d.category.value} -> owner: {d.owner}")
            lines.append(f"    {d.description}")
            if d.common_failure_turn is not None:
                lines.append(f"    Common failure at turn ~{d.common_failure_turn}")
    else:
        lines.append("No diagnostic patterns detected.")

    return "\n".join(lines)
