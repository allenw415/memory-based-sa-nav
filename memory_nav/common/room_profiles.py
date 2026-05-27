from __future__ import annotations

from pathlib import Path
from typing import Any


VISUAL_PROFILE_GRAPH_NAME = "room_graph_with_visual_profiles.json"
DEFAULT_ROOM_GRAPH_NAME = "room_graph.json"


def preferred_room_graph_path(artifacts_dir: str | Path) -> Path:
    resolved_dir = Path(artifacts_dir)
    visual_profile_path = resolved_dir / VISUAL_PROFILE_GRAPH_NAME
    if visual_profile_path.exists():
        return visual_profile_path
    return resolved_dir / DEFAULT_ROOM_GRAPH_NAME


def compact_visual_profile(node: dict[str, Any], *, max_items: int = 4) -> dict[str, Any]:
    profile = node.get("visual_profile")
    if not isinstance(profile, dict):
        return {}

    fields: dict[str, Any] = {}
    short_description = profile.get("short_description")
    if isinstance(short_description, str) and short_description:
        fields["short_description"] = short_description

    list_fields = (("visual_cues", "visual_cues", max_items),)
    for source_key, target_key, limit in list_fields:
        values = [
            value
            for value in profile.get(source_key, [])
            if isinstance(value, str) and value
        ]
        if values and limit > 0:
            fields[target_key] = values[: max(0, limit)]
    return fields


def room_candidate_payload(
    *,
    room_id: str,
    node: dict[str, Any],
    entry: dict[str, Any] | None = None,
    transition_support: float | None = None,
) -> dict[str, Any]:
    entry = entry or {}
    aliases = []
    for value in list(node.get("aliases") or []) + list(entry.get("aliases") or []):
        if isinstance(value, str) and value and value not in aliases:
            aliases.append(value)

    payload: dict[str, Any] = {
        "room_id": room_id,
        "title": node.get("title"),
        "category": node.get("category"),
        "aliases": aliases,
    }
    if transition_support is not None:
        payload["transition_support"] = float(transition_support)
    visual_profile = compact_visual_profile(node)
    if visual_profile:
        payload.update(visual_profile)
    else:
        payload["anchor_entities"] = [
            value
            for value in entry.get("anchor_entities", [])
            if isinstance(value, str) and value
        ]
    return payload


def visual_profile_anchor_entities(node: dict[str, Any], *, max_items: int = 6) -> list[str]:
    profile = node.get("visual_profile")
    if not isinstance(profile, dict):
        return []

    anchors: list[str] = []
    for key in ("short_description",):
        value = profile.get(key)
        if isinstance(value, str) and value:
            anchors.append(value)
    for key in ("visual_cues",):
        for value in profile.get(key, []):
            if isinstance(value, str) and value and value not in anchors:
                anchors.append(value)
            if len(anchors) >= max_items:
                return anchors
    return anchors[:max_items]
