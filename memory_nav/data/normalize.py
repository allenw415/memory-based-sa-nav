from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from memory_nav.common.types import PanoNode, RoomNode

BRITISH_MUSEUM_EXPERIMENT_ROOM_IDS = {
    "Room 1",
    "Room 2",
    "Room 4",
    "Room 6",
    "Room 7",
    "Room 8",
    "Room 9",
    "Room 10",
    "Room 11",
    "Room 12",
    "Room 13",
    "Room 14",
    "Room 15",
    "Room 16",
    "Room 17",
    "Room 18",
    "Room 18a",
    "Room 18b",
    "Room 19",
    "Room 20",
    "Room 21",
    "Room 22",
    "Room 23",
    "Room 24",
    "Room 26",
    "Room 27",
    "Room 29a",
    "Room 29b",
}

BRITISH_MUSEUM_ROOM_CANONICAL_IDS = {
    "Room 6 bottom": "Room 6",
}

BRITISH_MUSEUM_DIRECTION_OVERRIDES = {
    ("Room 18a", "Room 18b"): ("north", 360.0),
    ("Room 18b", "Room 18a"): ("south", 180.0),
}

BRITISH_MUSEUM_TRANSITION_OVERRIDES = {
    ("Room 22", "Room 23"): "stairs",
    ("Room 23", "Room 22"): "stairs",
}

BRITISH_MUSEUM_EXCLUDED_EDGES = {
    ("Room 9", "Room 23"),
    ("Room 23", "Room 9"),
}


ROOM_DIRECTION_TO_CARDINAL = {
    "up": ("north", 360.0),
    "top": ("north", 360.0),
    "right": ("east", 90.0),
    "down": ("south", 180.0),
    "left": ("west", 270.0),
}

REVERSE_CARDINAL_DIRECTION = {
    "north": ("south", 180.0),
    "south": ("north", 360.0),
    "east": ("west", 270.0),
    "west": ("east", 90.0),
}


def _normalize_floor(value: object) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _infer_transition_type(name: str) -> str:
    lowered = name.lower()
    if "stairs" in lowered:
        return "stairs"
    if "lift" in lowered or "elevator" in lowered:
        return "lift"
    if "escalator" in lowered:
        return "escalator"
    return "passage"


def _room_number(room_id: str) -> int | None:
    prefix = room_id.strip()
    if not prefix.startswith("Room "):
        return None
    suffix = prefix[5:]
    digits = []
    for ch in suffix:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def _is_gallery_room_id(room_id: str, *, max_room_number: int) -> bool:
    number = _room_number(room_id)
    if number is None:
        return False
    return number <= max_room_number


def _alias_index(explicit_map: dict[str, dict]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    for room_id, record in explicit_map.items():
        aliases[room_id].append(room_id)
        display_name = str(record.get("name") or room_id)
        aliases[display_name].append(room_id)
        title = record.get("title")
        if isinstance(title, str) and title:
            aliases[title].append(room_id)
        elif isinstance(title, list):
            for value in title:
                if isinstance(value, str) and value:
                    aliases[value].append(room_id)
    return aliases


def _title_values(value: object) -> list[str]:
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item]
    return []


def _resolve_target_room_id(
    target_name: str,
    *,
    explicit_map: dict[str, dict],
    aliases: dict[str, list[str]],
) -> str:
    if target_name in explicit_map:
        return target_name
    candidate_ids = aliases.get(target_name, [])
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    return target_name


def normalize_room_graph(
    explicit_map: dict[str, dict],
    *,
    max_room_number: int = 33,
    allowed_room_ids: set[str] | None = None,
    canonical_room_ids: dict[str, str] | None = None,
    ensure_bidirectional: bool = False,
    direction_overrides: dict[tuple[str, str], tuple[str, float]] | None = None,
    transition_overrides: dict[tuple[str, str], str] | None = None,
    excluded_edges: set[tuple[str, str]] | None = None,
) -> dict[str, dict]:
    aliases = _alias_index(explicit_map)
    canonical_room_ids = canonical_room_ids or {}
    direction_overrides = direction_overrides or {}
    transition_overrides = transition_overrides or {}
    excluded_edges = excluded_edges or set()
    nodes: dict[str, RoomNode] = {}

    for room_id, record in explicit_map.items():
        canonical_room_id = canonical_room_ids.get(room_id, room_id)
        if not _is_gallery_room_id(canonical_room_id, max_room_number=max_room_number):
            continue
        if allowed_room_ids is not None and canonical_room_id not in allowed_room_ids:
            continue
        display_name = str(record.get("name") or room_id)
        alias_list = [room_id, canonical_room_id]
        if display_name != room_id:
            alias_list.append(display_name)
        title_values = _title_values(record.get("title"))
        alias_list.extend(title_values)
        title = "; ".join(title_values) if title_values else None

        if canonical_room_id in nodes:
            nodes[canonical_room_id].aliases = sorted(set(nodes[canonical_room_id].aliases) | set(alias_list))
            continue

        node = RoomNode(
            room_id=canonical_room_id,
            display_name=display_name,
            floor=_normalize_floor(record.get("Level")),
            category=record.get("category"),
            title=title,
            aliases=sorted(set(alias_list)),
        )
        nodes[canonical_room_id] = node

    for source_room_id, record in explicit_map.items():
        canonical_source_room_id = canonical_room_ids.get(source_room_id, source_room_id)
        if canonical_source_room_id not in nodes:
            continue
        for link in record.get("links", []):
            if not isinstance(link, dict):
                continue

            raw_targets = link.get("name")
            target_names = raw_targets if isinstance(raw_targets, list) else [raw_targets]
            raw_direction = str(link.get("direction") or "unknown")
            cardinal = ROOM_DIRECTION_TO_CARDINAL.get(raw_direction)

            for raw_target_name in target_names:
                if not isinstance(raw_target_name, str) or not raw_target_name:
                    continue
                target_room_id = _resolve_target_room_id(
                    raw_target_name,
                    explicit_map=explicit_map,
                    aliases=aliases,
                )
                target_room_id = canonical_room_ids.get(target_room_id, target_room_id)
                if not _is_gallery_room_id(target_room_id, max_room_number=max_room_number):
                    continue
                if allowed_room_ids is not None and target_room_id not in allowed_room_ids:
                    continue
                if target_room_id not in nodes:
                    continue
                if canonical_source_room_id == target_room_id:
                    continue
                if (canonical_source_room_id, target_room_id) in excluded_edges:
                    continue
                override = direction_overrides.get((canonical_source_room_id, target_room_id))
                allocentric_direction = override[0] if override else (cardinal[0] if cardinal else "unknown")
                allocentric_heading_deg = override[1] if override else (cardinal[1] if cardinal else None)
                transition_type = transition_overrides.get(
                    (canonical_source_room_id, target_room_id),
                    _infer_transition_type(raw_target_name),
                )
                edge = {
                    "target_room_id": target_room_id,
                    "target_display_name": nodes[target_room_id].display_name,
                    "allocentric_direction": allocentric_direction,
                    "allocentric_heading_deg": allocentric_heading_deg,
                    "transition_type": transition_type,
                }
                if edge in nodes[canonical_source_room_id].neighbors:
                    continue
                nodes[canonical_source_room_id].neighbors.append(edge)

    if ensure_bidirectional:
        for source_room_id, node in list(nodes.items()):
            for edge in list(node.neighbors):
                target_room_id = edge["target_room_id"]
                if target_room_id not in nodes:
                    continue
                reverse_exists = any(
                    reverse_edge["target_room_id"] == source_room_id
                    for reverse_edge in nodes[target_room_id].neighbors
                )
                if reverse_exists:
                    continue
                if (target_room_id, source_room_id) in excluded_edges:
                    continue
                reverse_direction, reverse_heading = REVERSE_CARDINAL_DIRECTION.get(
                    edge["allocentric_direction"],
                    ("unknown", None),
                )
                nodes[target_room_id].neighbors.append(
                    {
                        "target_room_id": source_room_id,
                        "target_display_name": nodes[source_room_id].display_name,
                        "allocentric_direction": reverse_direction,
                        "allocentric_heading_deg": reverse_heading,
                        "transition_type": edge["transition_type"],
                    }
                )

    return {room_id: node.to_dict() for room_id, node in sorted(nodes.items())}


def normalize_pano_graph(processed_panos: dict[str, dict]) -> dict[str, dict]:
    nodes: dict[str, PanoNode] = {}

    for pano_id, record in processed_panos.items():
        node = PanoNode(
            pano_id=pano_id,
            floor=_normalize_floor(record.get("floor")),
            lat=record.get("lat"),
            lng=record.get("lng"),
        )
        for link in record.get("links", []):
            if not isinstance(link, dict):
                continue
            target_pano_id = link.get("panoID")
            if not isinstance(target_pano_id, str) or not target_pano_id:
                continue
            node.neighbors.append(
                {
                    "target_pano_id": target_pano_id,
                    "geocentric_heading_deg": float(link["heading"]),
                    "description": link.get("description"),
                }
            )
        nodes[pano_id] = node

    return {pano_id: node.to_dict() for pano_id, node in sorted(nodes.items())}


def load_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return data
