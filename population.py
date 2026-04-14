"""PopulationOS — research-driven heterogeneous user population generator.

Three spaces merged into PersonaStructure → PersonaPool → AgentProfile per session.
"""
from __future__ import annotations
import random
from dataclasses import dataclass


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
    frequency: float                                       # 0-1, must sum to 1.0 across all archetypes
    description: str
    trait_constraints: dict[str, tuple[float, float]]     # trait_name → (min_val, max_val)


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

    def to_behavioral_constraints(self, dimensions: list["TraitDimension"]) -> str:
        """Convert trait vector to behavioral constraint string for session prompt.

        Includes four mandatory anti-rationalization rules that suppress LLM's
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


class PersonaPool:
    """Samples N AgentProfiles from a confirmed PersonaStructure.

    Archetype assignment respects frequency weights.
    Trait values are sampled within archetype constraints using normal distribution,
    clipped to [0, 10] and to the archetype's [min, max] window.
    """

    def __init__(self, structure: PersonaStructure):
        self._structure = structure
        self._counter = 0

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
                val = max(lo, min(hi, raw))    # clip to archetype window
                val = max(0.0, min(10.0, val)) # clip to absolute scale
                trait_vector[dim.name] = round(val, 2)

            self._counter += 1
            agents.append(AgentProfile(
                agent_id=f"agent_{self._counter:03d}",
                archetype_name=arch.name,
                trait_vector=trait_vector,
            ))

        return agents
