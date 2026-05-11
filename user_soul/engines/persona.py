"""PersonaEngine — population research + persona pool generation."""
from __future__ import annotations

import json
import random
import re

from user_soul.backend import LLMBackend
from user_soul.models import (
    AgentProfile,
    Archetype,
    PersonaStructure,
    TraitDimension,
)


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
      "description": "one-line summary of who they are",
      "background_story": "具体人物：年龄、城市/国家、职业、手机上已有哪些同类产品、每天什么时候用手机、怎么发现这个产品、一个自然的生活张力（想要什么 vs 有什么限制）。≤60字。只写这个人是谁，不写行为预测。",
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
- cover cultural/regional context in the dimensions if relevant
- background_story: write a REAL PERSON (age, location, job, existing apps, daily routine, discovery channel). Do NOT write behavioral predictions or rules. The story should make someone's natural patience/engagement level obvious from context, not stated explicitly."""


class PersonaEngine:
    """Population research + persona pool generation.

    Merges PopulationResearcher and PersonaPool into a single engine.
    """

    def __init__(self, backend: LLMBackend):
        self._backend = backend
        self._counter = 0

    def research(self, product_description: str) -> PersonaStructure:
        prompt = _RESEARCH_PROMPT.format(product_description=product_description[:2000])
        raw = self._backend.text(prompt, max_tokens=2048, model_tier="smart")
        return self._parse(raw, product_description)

    def generate_pool(self, structure: PersonaStructure, n: int) -> list[AgentProfile]:
        agents: list[AgentProfile] = []
        archetypes = structure.archetypes
        dims = structure.trait_dimensions

        total = sum(a.frequency for a in archetypes)
        weights = [a.frequency / total for a in archetypes]

        for _ in range(n):
            arch = random.choices(archetypes, weights=weights, k=1)[0]

            trait_vector: dict[str, float] = {}
            for dim in dims:
                lo, hi = arch.trait_constraints.get(dim.name, (0.0, 10.0))
                center = (lo + hi) / 2
                span = (hi - lo) / 4
                raw = random.gauss(center, max(span, 0.5))
                val = max(lo, min(hi, raw))
                val = max(0.0, min(10.0, val))
                trait_vector[dim.name] = round(val, 2)

            self._counter += 1
            agents.append(AgentProfile(
                agent_id=f"agent_{self._counter:03d}",
                archetype_name=arch.name,
                trait_vector=trait_vector,
                dims=dims,
                background_story=arch.background_story,
            ))

        return agents

    def get_or_create(self, product_description: str, n: int = 12) -> list[AgentProfile]:
        structure = self.research(product_description)
        return self.generate_pool(structure, n)

    def _parse(self, raw: str, product_description: str) -> PersonaStructure:
        data = _safe_json(raw)
        if not data:
            return self._fallback(product_description)

        dims: list[TraitDimension] = []
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

        archs: list[Archetype] = []
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
                background_story=a.get("background_story", ""),
            ))

        if archs:
            total = sum(a.frequency for a in archs)
            if total > 0:
                archs = [
                    Archetype(a.name, a.frequency / total, a.description, a.trait_constraints, a.background_story)
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
            Archetype("Passive User", 0.40, "minimal engagement", {"engagement_drive": (0.0, 5.0)}),
        ]
        return PersonaStructure(
            "App Users", product_description[:100], dims, archs,
            "fallback — LLM parsing failed",
        )
