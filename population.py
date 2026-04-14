"""PopulationOS — research-driven heterogeneous user population generator.

Three spaces merged into PersonaStructure → PersonaPool → AgentProfile per session.
"""
from __future__ import annotations
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
