from dataclasses import dataclass, field
from typing import TypeAlias


@dataclass
class Stats:
    hp: int = 0
    attack: int = 0
    defense: int = 0
    sp_attack: int = 0
    sp_defense: int = 0
    speed: int = 0
    base_stat_total: int = 0

    @classmethod
    def from_dict(cls, stats):
        return cls(**{k: stats.get(k, 0) for k in cls.__dataclass_fields__})


@dataclass
class PokemonDoc:
    id: int
    name: str = ""
    types: list = field(default_factory=list)
    generation: str = ""
    stats: Stats = field(default_factory=Stats)
    height_m: float | None = None
    weight_kg: float | None = None
    abilities: list = field(default_factory=list)
    hidden_ability: str | None = None
    egg_groups: list = field(default_factory=list)
    color: str = ""
    shape: str = ""
    habitat: str | None = None
    growth_rate: str = ""
    capture_rate: int | None = None
    base_happiness: int | None = None
    base_experience: int | None = None
    genus: str = ""
    is_legendary: bool = False
    is_mythical: bool = False
    is_baby: bool = False
    evolution_chain_id: int | None = None
    flavor_text: str = ""
    sprite_url: str = ""
    evolves_from: str | None = None
    evolves_into: list = field(default_factory=list)
    type_effectiveness: dict = field(default_factory=dict)
    search_text: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(cls, doc):
        fields = {k: v for k, v in doc.items() if k in cls.__dataclass_fields__}
        if "stats" in fields:
            fields["stats"] = Stats.from_dict(fields["stats"] or {})
        return cls(**fields)


@dataclass
class TypeChartDoc:
    id: str
    type: str = ""
    search_text: str = ""
    score: float = 0.0

    @classmethod
    def from_dict(cls, doc):
        # Drops the kind marker and damage lists: the app only consumes id,
        # type and search_text (the damage info is embedded in search_text).
        return cls(
            id=doc["id"],
            type=doc.get("type", ""),
            search_text=doc.get("search_text", ""),
            score=doc.get("score", 0.0),
        )


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


SearchResult: TypeAlias = PokemonDoc | TypeChartDoc | WebResult


def parse_doc(doc):
    if doc.get("kind") == "type_chart":
        return TypeChartDoc.from_dict(doc)
    return PokemonDoc.from_dict(doc)
