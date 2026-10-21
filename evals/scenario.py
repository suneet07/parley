"""Scenario schema and loader for parley evaluations."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Expected(BaseModel):
    must_call: list[str] = Field(default_factory=list)
    must_not_call: list[str] = Field(default_factory=list)
    must_offer_one_of: list[str] = Field(default_factory=list)
    must_verify_identity: bool | None = None
    must_escalate: bool | None = None
    must_end_call: bool | None = None
    must_mention: list[str] = Field(default_factory=list)
    must_not_mention: list[str] = Field(default_factory=list)
    resolution: Literal["resolved", "escalated", "declined", "any"] = "any"


class Scenario(BaseModel):
    id: str
    category: Literal[
        "happy_path", "policy_edge", "difficult", "adversarial", "off_topic", "outbound"
    ]
    smoke: bool = False
    persona: str
    goal: str
    knows: dict[str, str] = Field(default_factory=dict)
    behavior: list[str] = Field(default_factory=list)
    opening: str | None = None
    direction: Literal["inbound", "outbound"] = "inbound"
    caller_number: str | None = None
    outbound_goal: str | None = None
    max_turns: int = 12
    expected: Expected = Field(default_factory=Expected)
    judge_notes: str = ""


def load_scenarios(root: Path = Path("evals/scenarios"), suite: str = "smoke") -> list[Scenario]:
    """Load and filter scenarios from directory.

    suite: "smoke" -> smoke=True only; "full" -> all; otherwise fnmatch glob on id.
    Duplicate ids raise ValueError. Returns scenarios sorted by id.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []

    files = sorted(list(root_path.rglob("*.yaml")) + list(root_path.rglob("*.yml")))
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()

    for file_path in files:
        data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        if not data:
            continue
        scn = Scenario.model_validate(data)
        if scn.id in seen_ids:
            raise ValueError(f"Duplicate scenario id: {scn.id}")
        seen_ids.add(scn.id)
        scenarios.append(scn)

    if suite == "smoke":
        filtered = [s for s in scenarios if s.smoke]
    elif suite == "full":
        filtered = scenarios
    else:
        filtered = [s for s in scenarios if fnmatch.fnmatch(s.id, suite)]

    filtered.sort(key=lambda s: s.id)
    return filtered
