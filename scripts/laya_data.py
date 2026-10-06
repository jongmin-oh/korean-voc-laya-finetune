"""Shared dataset schema for the Korean card-consultation Laya experiment."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path


QUESTION_ID = "consultation_group"
INSTRUCTION = "이 카드 상담의 주된 업무 그룹을 하나 선택하세요."


def load_schema(label_map_path: Path) -> tuple[list[str], OrderedDict[str, str]]:
    with label_map_path.open(encoding="utf-8") as f:
        mapping = json.load(f)

    groups = mapping["groups"]
    display = mapping.get("display", {})
    criteria: OrderedDict[str, str] = OrderedDict()
    for group in groups:
        labels = mapping["hierarchy"][group]
        rendered = [display.get(label, label) for label in labels]
        criteria[group] = ", ".join(rendered) + " 관련 상담"
    return groups, criteria


def internal_question(criteria: OrderedDict[str, str]) -> dict:
    return {"t": "choice", "ins": INSTRUCTION, "crit": criteria}


def public_questions(criteria: OrderedDict[str, str]) -> dict:
    return {
        QUESTION_ID: {
            "type": "choice",
            "instructions": INSTRUCTION,
            "criteria": criteria,
        }
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def make_item(row: dict[str, str], groups: list[str], criteria: OrderedDict[str, str], input_column: str) -> dict:
    group = row["group"].strip()
    if group not in groups:
        raise ValueError(f"unknown group {group!r} for source_id={row.get('source_id')}")
    state = row[input_column].strip()
    if not state:
        raise ValueError(f"empty {input_column!r} for source_id={row.get('source_id')}")
    return {
        "source_id": row["source_id"],
        "state": state,
        "q": internal_question(criteria),
        "gold_idx": groups.index(group),
        "gold": group,
        "lang": "ko",
        "family": "card_consultation_group",
    }
