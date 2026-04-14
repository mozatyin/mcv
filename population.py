"""PopulationOS — research-driven heterogeneous user population generator.

Three spaces merged into PersonaStructure → PersonaPool → AgentProfile per session.
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
import mcv.core as _core


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
    dims: list = field(default_factory=list, repr=False, compare=False)  # TraitDimension list from pool

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
                label = f"neutral on {dim.description.lower()}"
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
        prompt = _RESEARCH_PROMPT.format(product_description=product_description[:2000])
        raw, _ = _core._llm_call(prompt, self._api_key, max_tokens=2048)
        return self._parse(raw, product_description)

    def _parse(self, raw: str, product_description: str) -> PersonaStructure:
        """Parse LLM response into PersonaStructure. Returns minimal fallback on failure."""
        data = _core._safe_json(raw)
        if not data:
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

        # Normalise frequencies to sum to 1.0
        if archs:
            total = sum(a.frequency for a in archs)
            if total > 0:
                archs = [
                    Archetype(a.name, a.frequency / total, a.description, a.trait_constraints)
                    for a in archs
                ]

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
            TraitDimension(
                "engagement_drive", "Motivation to engage deeply",
                "passive, easily disengaged, skips most features",
                "highly motivated, explores all features eagerly",
                "normal", 5.0, 2.5, "space1",
            ),
        ]
        archs = [
            Archetype("Engaged User", 0.60, "actively uses the product", {"engagement_drive": (5.0, 10.0)}),
            Archetype("Passive User", 0.40, "minimal engagement",        {"engagement_drive": (0.0, 5.0)}),
        ]
        return PersonaStructure(
            population_label="App Users",
            product_context=product_description[:100],
            trait_dimensions=dims,
            archetypes=archs,
            research_notes="fallback — LLM parsing failed",
        )


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
            agent = AgentProfile(
                agent_id=f"agent_{self._counter:03d}",
                archetype_name=arch.name,
                trait_vector=trait_vector,
                dims=dims,
            )
            agents.append(agent)

        return agents
