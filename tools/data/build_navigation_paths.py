from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.data.memory_localization import load_json, write_json  # noqa: E402
from memory_nav.spatial.routing import (  # noqa: E402
    RoutePlanner,
    floor_pano_graph,
    floor_room_graph,
)


DEFAULT_ARTIFACTS_DIR = "dataset/sites/british_museum/normalized"
DEFAULT_OUTPUT_DIR = "outputs/navigation_paths/floor0"
DEFAULT_TARGET_GROUPS_JSON = None
DIFFICULTY_ORDER = ("easy", "medium", "hard")
DIFFICULTY_COLORS = {
    "easy": "#4C9F70",
    "medium": "#E3A72F",
    "hard": "#C84C4C",
}
DETOUR_HEATMAP_BIN_WIDTH = 0.2
SHORTEST_PATH_HEATMAP_BIN_WIDTH_M = 20.0
CSV_FIELDS = (
    "path_id",
    "floor",
    "start_pano_id",
    "start_room_id",
    "target_group_id",
    "target_group_theme",
    "acceptable_target_room_ids",
    "reference_end_room_id",
    "target_room_id",
    "end_pano_id",
    "end_room_id",
    "pano_step_count",
    "visited_room_count",
    "room_transition_count",
    "path_distance_m",
    "straight_line_distance_m",
    "detour_ratio",
    "difficulty",
    "path_constraint_mode",
    "topology_validation_status",
    "null_pano_count",
    "shortest_path_rooms",
    "shortest_path_panos",
    "shortest_path_pano_rooms",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build single-floor start-pano -> target-room shortest navigation paths, "
            "label difficulty by visited gallery count, and analyze raw detour ratios."
        )
    )
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-groups-json",
        default=DEFAULT_TARGET_GROUPS_JSON,
        help=(
            "Optional semantic-target configuration. Without it, each floor room "
            "is an independent target."
        ),
    )
    parser.add_argument(
        "--floor",
        default="0",
        help="Only use rooms and panorama nodes on this floor (default: 0).",
    )
    parser.add_argument(
        "--easy-max-visited-rooms",
        type=int,
        default=3,
        help="Largest visited_room_count labeled easy (default: 3).",
    )
    parser.add_argument(
        "--medium-max-visited-rooms",
        type=int,
        default=5,
        help="Largest visited_room_count labeled medium (default: 5).",
    )
    parser.add_argument("--limit-starts", type=int)
    parser.add_argument("--max-records", type=int)
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def normalized_room_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "null":
        return None
    return cleaned


def load_pano_room_mappings(path: Path) -> dict[str, str]:
    payload = load_json(path)
    mappings = payload.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(f"Expected 'mappings' object in {path}")
    return {
        str(pano_id): room_id
        for pano_id, raw_room_id in mappings.items()
        if (room_id := normalized_room_id(raw_room_id)) is not None
    }


def room_ids_for_floor(room_graph: dict[str, dict], *, floor: str) -> set[str]:
    room_ids: set[str] = set()
    for room_id, record in room_graph.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("floor")) != floor:
            continue
        if record.get("category") == "Circulation" or record.get("synthetic") is True:
            continue
        room_ids.add(str(record.get("room_id") or room_id))
    return room_ids


def pano_graph_for_floor(pano_graph: dict[str, dict], *, floor: str) -> dict[str, dict]:
    return {
        pano_id: record
        for pano_id, record in pano_graph.items()
        if isinstance(record, dict) and str(record.get("floor")) == floor
    }


def group_panos_by_room(
    pano_room_mappings: dict[str, str],
    *,
    valid_room_ids: set[str],
    pano_graph: dict[str, dict],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for pano_id, room_id in pano_room_mappings.items():
        if room_id not in valid_room_ids or pano_id not in pano_graph:
            continue
        grouped.setdefault(room_id, []).append(pano_id)
    return {room_id: sorted(pano_ids) for room_id, pano_ids in grouped.items()}


def build_target_groups(
    *,
    target_room_ids: set[str],
    room_themes: dict[str, str] | None = None,
    config: dict | None = None,
) -> list[dict]:
    """Build deterministic, non-overlapping target groups for the selected floor."""
    config = config or {}
    room_themes = room_themes or {}
    excluded = {
        str(room_id)
        for room_id in config.get("excluded_target_room_ids", [])
    }
    unknown_excluded = excluded - target_room_ids
    if unknown_excluded:
        raise ValueError(
            "Target-group configuration excludes unknown floor rooms: "
            + ", ".join(sorted(unknown_excluded))
        )

    grouped_rooms: set[str] = set()
    groups: list[dict] = []
    for raw_group in config.get("equivalent_groups", []):
        if not isinstance(raw_group, dict):
            raise ValueError("Each equivalent target group must be an object.")
        group_id = str(raw_group.get("target_group_id") or "").strip()
        theme = str(raw_group.get("target_group_theme") or "").strip()
        members = sorted(
            {
                str(room_id).strip()
                for room_id in raw_group.get("acceptable_target_room_ids", [])
                if str(room_id).strip()
            }
        )
        if not group_id or not theme or not members:
            raise ValueError(
                "Equivalent target groups require target_group_id, "
                "target_group_theme, and acceptable_target_room_ids."
            )
        unknown_members = set(members) - target_room_ids
        if unknown_members:
            raise ValueError(
                f"Target group {group_id!r} contains unknown floor rooms: "
                + ", ".join(sorted(unknown_members))
            )
        overlap = grouped_rooms.intersection(members)
        if overlap:
            raise ValueError(
                "Target room appears in more than one equivalent group: "
                + ", ".join(sorted(overlap))
            )
        if set(members).intersection(excluded):
            raise ValueError(
                f"Target group {group_id!r} contains an excluded target room."
            )
        grouped_rooms.update(members)
        group = {
            "target_group_id": group_id,
            "target_group_theme": theme,
            "acceptable_target_room_ids": members,
        }
        query_override = raw_group.get("query_override")
        if isinstance(query_override, str) and query_override.strip():
            group["query_override"] = query_override.strip()
        groups.append(group)

    for room_id in sorted(target_room_ids - excluded - grouped_rooms):
        groups.append(
            {
                "target_group_id": room_id,
                "target_group_theme": room_themes.get(room_id, room_id),
                "acceptable_target_room_ids": [room_id],
            }
        )

    group_ids = [group["target_group_id"] for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("target_group_id values must be unique.")
    return sorted(groups, key=lambda group: group["target_group_id"])


def load_target_group_config(path: str | Path | None, *, floor: str) -> dict | None:
    if path is None:
        return None
    config_path = resolve_project_path(path)
    config = load_json(config_path)
    configured_floor = config.get("floor")
    if configured_floor is not None and str(configured_floor) != floor:
        raise ValueError(
            f"Target-group configuration is for floor {configured_floor!r}, not {floor!r}."
        )
    return config


def lat_lng(record: dict | None) -> tuple[float, float] | None:
    if not isinstance(record, dict):
        return None
    lat = record.get("lat")
    lng = record.get("lng")
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return None
    return float(lat), float(lng)


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lng2 - lng1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * math.asin(math.sqrt(a))


def distance_between_panos_m(
    pano_graph: dict[str, dict],
    source_pano_id: str,
    target_pano_id: str,
) -> float | None:
    source = lat_lng(pano_graph.get(source_pano_id))
    target = lat_lng(pano_graph.get(target_pano_id))
    if source is None or target is None:
        return None
    return haversine_m(source[0], source[1], target[0], target[1])


def dijkstra(
    pano_graph: dict[str, dict],
    start_pano_id: str,
) -> tuple[dict[str, float], dict[str, str]]:
    distances = {start_pano_id: 0.0}
    parents: dict[str, str] = {}
    queue: list[tuple[float, str]] = [(0.0, start_pano_id)]
    while queue:
        current_distance, pano_id = heapq.heappop(queue)
        if current_distance != distances.get(pano_id):
            continue
        node = pano_graph.get(pano_id)
        if not isinstance(node, dict):
            continue
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_pano_id = neighbor.get("target_pano_id")
            if not isinstance(target_pano_id, str) or target_pano_id not in pano_graph:
                continue
            edge_distance = distance_between_panos_m(pano_graph, pano_id, target_pano_id)
            if edge_distance is None:
                continue
            candidate_distance = current_distance + edge_distance
            if candidate_distance < distances.get(target_pano_id, math.inf):
                distances[target_pano_id] = candidate_distance
                parents[target_pano_id] = pano_id
                heapq.heappush(queue, (candidate_distance, target_pano_id))
    return distances, parents


def reconstruct_path(
    parents: dict[str, str],
    *,
    start_pano_id: str,
    end_pano_id: str,
) -> list[str]:
    path = [end_pano_id]
    cursor = end_pano_id
    while cursor != start_pano_id:
        cursor = parents[cursor]
        path.append(cursor)
    path.reverse()
    return path


ConstrainedState = tuple[str, int]


def pano_edge_distances(
    pano_graph: dict[str, dict],
) -> dict[tuple[str, str], float]:
    """Precompute usable floor-local edge lengths once for repeated searches."""
    distances: dict[tuple[str, str], float] = {}
    for source_pano_id, node in pano_graph.items():
        if not isinstance(node, dict):
            continue
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_pano_id = neighbor.get("target_pano_id")
            if not isinstance(target_pano_id, str) or target_pano_id not in pano_graph:
                continue
            distance = distance_between_panos_m(
                pano_graph,
                source_pano_id,
                target_pano_id,
            )
            if distance is not None:
                distances[(source_pano_id, target_pano_id)] = distance
    return distances


def constrained_shortest_pano_path(
    pano_graph: dict[str, dict],
    pano_room_mappings: dict[str, str],
    *,
    start_pano_id: str,
    target_pano_ids: Iterable[str],
    topology_room_route: list[str],
    edge_distances: dict[tuple[str, str], float] | None = None,
) -> tuple[float, str, list[str]] | None:
    """Find the shortest pano path that follows a fixed room route in order.

    Unmapped/null panoramas retain the current route index. A mapped panorama may
    remain in the current room or advance exactly one room; every other mapped
    transition is rejected, so null panoramas cannot skip topology rooms.
    """
    if not topology_room_route or start_pano_id not in pano_graph:
        return None
    if pano_room_mappings.get(start_pano_id) != topology_room_route[0]:
        return None
    target_ids = {pano_id for pano_id in target_pano_ids if pano_id in pano_graph}
    if not target_ids:
        return None
    edge_distances = edge_distances or pano_edge_distances(pano_graph)

    start_state: ConstrainedState = (start_pano_id, 0)
    distances: dict[ConstrainedState, float] = {start_state: 0.0}
    parents: dict[ConstrainedState, ConstrainedState] = {}
    queue: list[tuple[float, str, int]] = [(0.0, start_pano_id, 0)]
    goal_state: ConstrainedState | None = None

    while queue:
        current_distance, pano_id, route_index = heapq.heappop(queue)
        state = (pano_id, route_index)
        if current_distance != distances.get(state):
            continue
        if route_index == len(topology_room_route) - 1 and pano_id in target_ids:
            goal_state = state
            break
        node = pano_graph.get(pano_id)
        if not isinstance(node, dict):
            continue
        for neighbor in node.get("neighbors", []):
            if not isinstance(neighbor, dict):
                continue
            target_pano_id = neighbor.get("target_pano_id")
            if not isinstance(target_pano_id, str) or target_pano_id not in pano_graph:
                continue
            target_room_id = pano_room_mappings.get(target_pano_id)
            next_route_index = route_index
            if target_room_id is not None:
                if target_room_id == topology_room_route[route_index]:
                    pass
                elif (
                    route_index + 1 < len(topology_room_route)
                    and target_room_id == topology_room_route[route_index + 1]
                ):
                    next_route_index += 1
                else:
                    continue
            edge_distance = edge_distances.get((pano_id, target_pano_id))
            if edge_distance is None:
                continue
            next_state = (target_pano_id, next_route_index)
            candidate_distance = current_distance + edge_distance
            if candidate_distance < distances.get(next_state, math.inf):
                distances[next_state] = candidate_distance
                parents[next_state] = state
                heapq.heappush(
                    queue,
                    (candidate_distance, target_pano_id, next_route_index),
                )

    if goal_state is None:
        return None
    path = [goal_state[0]]
    cursor = goal_state
    while cursor != start_state:
        cursor = parents[cursor]
        path.append(cursor[0])
    path.reverse()
    return distances[goal_state], goal_state[0], path


def validate_topology_constrained_path(
    path_panos: Iterable[str],
    topology_room_route: list[str],
    pano_room_mappings: dict[str, str],
) -> bool:
    """Validate ordered mapped-room progression; null panos never advance it."""
    if not topology_room_route:
        return False
    route_index = 0
    for pano_id in path_panos:
        room_id = pano_room_mappings.get(pano_id)
        if room_id is None:
            continue
        if room_id == topology_room_route[route_index]:
            continue
        if (
            route_index + 1 < len(topology_room_route)
            and room_id == topology_room_route[route_index + 1]
        ):
            route_index += 1
            continue
        return False
    return route_index == len(topology_room_route) - 1


def collapsed_room_sequence(
    path_panos: Iterable[str],
    pano_room_mappings: dict[str, str],
) -> list[str]:
    sequence: list[str] = []
    for pano_id in path_panos:
        room_id = pano_room_mappings.get(pano_id)
        if room_id is None:
            continue
        if not sequence or sequence[-1] != room_id:
            sequence.append(room_id)
    return sequence


def validate_difficulty_thresholds(
    easy_max_visited_rooms: int,
    medium_max_visited_rooms: int,
) -> None:
    if easy_max_visited_rooms < 2:
        raise ValueError("easy_max_visited_rooms must be at least 2.")
    if medium_max_visited_rooms <= easy_max_visited_rooms:
        raise ValueError(
            "medium_max_visited_rooms must be greater than easy_max_visited_rooms."
        )


def difficulty_for_visited_room_count(
    visited_room_count: int,
    *,
    easy_max_visited_rooms: int = 3,
    medium_max_visited_rooms: int = 5,
) -> str:
    validate_difficulty_thresholds(
        easy_max_visited_rooms,
        medium_max_visited_rooms,
    )
    if visited_room_count <= easy_max_visited_rooms:
        return "easy"
    if visited_room_count <= medium_max_visited_rooms:
        return "medium"
    return "hard"


def ratio_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    keys = (
        "count", "mean", "std", "min", "p25", "median",
        "p75", "p90", "p95", "p99", "max",
    )
    if array.size == 0:
        return {key: 0 if key == "count" else None for key in keys}
    percentiles = np.percentile(array, [25, 50, 75, 90, 95, 99])
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "p25": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
        "p95": float(percentiles[4]),
        "p99": float(percentiles[5]),
        "max": float(np.max(array)),
    }


def ratio_tertile_cutpoints(records: Iterable[dict]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for difficulty in DIFFICULTY_ORDER:
        values = np.asarray(
            [
                float(record["detour_ratio"])
                for record in records
                if record["difficulty"] == difficulty
            ],
            dtype=np.float64,
        )
        if values.size:
            low_middle, middle_high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
            result[difficulty] = {
                "count": int(values.size),
                "low_middle": float(low_middle),
                "middle_high": float(middle_high),
            }
        else:
            result[difficulty] = {
                "count": 0,
                "low_middle": None,
                "middle_high": None,
            }
    return result


def summarize_records(
    records: list[dict],
    *,
    floor: str,
    floor_room_ids: set[str],
    topology_room_graph: dict[str, dict],
    start_pano_count: int,
    target_groups: list[dict],
    skipped_same_target_group_count: int,
    skipped_topology_route_not_found_count: int,
    skipped_topology_constrained_unreachable_count: int,
    skipped_missing_distance_count: int,
    skipped_zero_straight_distance_count: int,
    easy_max_visited_rooms: int,
    medium_max_visited_rooms: int,
) -> dict:
    total = len(records)
    room_counts = Counter(int(record["visited_room_count"]) for record in records)
    difficulty_counts = Counter(str(record["difficulty"]) for record in records)
    target_group_counts = Counter(str(record["target_group_id"]) for record in records)
    room_distribution = []
    for visited_room_count in sorted(room_counts):
        count = room_counts[visited_room_count]
        room_distribution.append(
            {
                "visited_room_count": visited_room_count,
                "difficulty": difficulty_for_visited_room_count(
                    visited_room_count,
                    easy_max_visited_rooms=easy_max_visited_rooms,
                    medium_max_visited_rooms=medium_max_visited_rooms,
                ),
                "count": count,
                "percentage": 100.0 * count / total if total else 0.0,
            }
        )

    difficulty_distribution = {
        difficulty: {
            "count": difficulty_counts.get(difficulty, 0),
            "percentage": (
                100.0 * difficulty_counts.get(difficulty, 0) / total if total else 0.0
            ),
        }
        for difficulty in DIFFICULTY_ORDER
    }
    target_room_ids = sorted(
        {
            room_id
            for group in target_groups
            for room_id in group["acceptable_target_room_ids"]
        }
    )
    null_counts = [int(record["null_pano_count"]) for record in records]
    skipped_unreachable = (
        skipped_topology_route_not_found_count
        + skipped_topology_constrained_unreachable_count
    )
    return {
        "floor": floor,
        "floor_room_count": len(floor_room_ids),
        "floor_rooms": sorted(floor_room_ids),
        "topology_room_count": len(topology_room_graph),
        "topology_rooms": sorted(topology_room_graph),
        "path_constraint_mode": "room_topology_bfs_constrained_dijkstra",
        "topology_validation": {
            "valid_count": sum(
                record.get("topology_validation_status") == "valid"
                for record in records
            ),
            "invalid_count": sum(
                record.get("topology_validation_status") != "valid"
                for record in records
            ),
        },
        "null_pano_statistics": {
            "total_count_on_paths": sum(null_counts),
            "paths_with_null_panos": sum(count > 0 for count in null_counts),
            "maximum_on_one_path": max(null_counts, default=0),
            "definition": (
                "Null/unmapped panos count toward panorama steps and distance, "
                "but not visited_room_count and never advance the topology route."
            ),
        },
        "start_pano_count": start_pano_count,
        "target_group_count": len(target_groups),
        "target_groups": target_groups,
        "target_room_count": len(target_room_ids),
        "target_rooms": target_room_ids,
        "navigation_path_count": total,
        "skipped_same_target_group_count": skipped_same_target_group_count,
        "skipped_same_room_count": skipped_same_target_group_count,
        "skipped_unreachable_count": skipped_unreachable,
        "skipped_topology_route_not_found_count": skipped_topology_route_not_found_count,
        "skipped_topology_constrained_unreachable_count": (
            skipped_topology_constrained_unreachable_count
        ),
        "skipped_missing_distance_count": skipped_missing_distance_count,
        "skipped_zero_straight_distance_count": skipped_zero_straight_distance_count,
        "difficulty_definition": {
            "basis": "topology_route_room_count_including_start_target_and_circulation_rooms",
            "easy": f"2-{easy_max_visited_rooms}",
            "medium": f"{easy_max_visited_rooms + 1}-{medium_max_visited_rooms}",
            "hard": f">={medium_max_visited_rooms + 1}",
        },
        "detour_ratio_definition": (
            "topology_constrained_path_distance_m / straight_line_distance_m; "
            "raw and unclipped"
        ),
        "ratio_tertile_definition": (
            "Within each difficulty: low <= q1; q1 < middle <= q2; high > q2. "
            "Cutpoints use the complete raw candidate population."
        ),
        "visited_room_count_distribution": room_distribution,
        "difficulty_distribution": difficulty_distribution,
        "target_group_distribution": {
            group["target_group_id"]: {
                "target_group_theme": group["target_group_theme"],
                "count": target_group_counts.get(group["target_group_id"], 0),
                "percentage": (
                    100.0 * target_group_counts.get(group["target_group_id"], 0) / total
                    if total
                    else 0.0
                ),
            }
            for group in target_groups
        },
        "detour_ratio_overall": ratio_statistics(
            float(record["detour_ratio"]) for record in records
        ),
        "detour_ratio_by_difficulty": {
            difficulty: ratio_statistics(
                float(record["detour_ratio"])
                for record in records
                if record["difficulty"] == difficulty
            )
            for difficulty in DIFFICULTY_ORDER
        },
        "detour_ratio_tertiles_by_difficulty": ratio_tertile_cutpoints(records),
    }


def build_navigation_paths(args: argparse.Namespace) -> tuple[list[dict], dict]:
    floor = str(args.floor)
    easy_max = int(args.easy_max_visited_rooms)
    medium_max = int(args.medium_max_visited_rooms)
    validate_difficulty_thresholds(easy_max, medium_max)

    artifacts_dir = resolve_project_path(args.artifacts_dir)
    full_pano_graph = load_json(artifacts_dir / "pano_graph.json")
    room_graph = load_json(artifacts_dir / "room_graph.json")
    pano_room_mappings = load_pano_room_mappings(
        artifacts_dir / "pano_room_grounding.json"
    )

    floor_room_ids = room_ids_for_floor(room_graph, floor=floor)
    if not floor_room_ids:
        raise ValueError(f"No non-circulation rooms found for floor {floor!r}.")
    pano_graph = floor_pano_graph(full_pano_graph, floor=floor)
    topology_room_graph = floor_room_graph(
        room_graph,
        pano_graph=pano_graph,
        pano_room_mappings=pano_room_mappings,
    )
    route_planner = RoutePlanner(room_graph=topology_room_graph)
    edge_distances = pano_edge_distances(pano_graph)
    panos_by_room = group_panos_by_room(
        pano_room_mappings,
        valid_room_ids=floor_room_ids,
        pano_graph=pano_graph,
    )
    start_panos = sorted(
        pano_id for pano_ids in panos_by_room.values() for pano_id in pano_ids
    )
    if args.limit_starts is not None:
        start_panos = start_panos[: max(int(args.limit_starts), 0)]

    target_group_config = load_target_group_config(
        args.target_groups_json,
        floor=floor,
    )
    room_themes = {
        room_id: str(room_graph.get(room_id, {}).get("title") or room_id)
        for room_id in panos_by_room
    }
    target_groups = build_target_groups(
        target_room_ids=set(panos_by_room),
        room_themes=room_themes,
        config=target_group_config,
    )

    records: list[dict] = []
    skipped_topology_route_not_found = 0
    skipped_topology_constrained_unreachable = 0
    skipped_same_target_group = 0
    skipped_missing_distance = 0
    skipped_zero_straight_distance = 0
    route_cache: dict[tuple[str, str], list[str]] = {}

    stop = False
    for start_pano_id in start_panos:
        start_room_id = pano_room_mappings[start_pano_id]
        for target_group in target_groups:
            acceptable_room_ids = target_group["acceptable_target_room_ids"]
            if start_room_id in acceptable_room_ids:
                skipped_same_target_group += 1
                continue

            candidates: list[tuple[float, str, str, list[str], list[str]]] = []
            route_found = False
            for reference_room_id in acceptable_room_ids:
                route_key = (start_room_id, reference_room_id)
                if route_key not in route_cache:
                    route_cache[route_key] = route_planner.shortest_room_path(
                        start_room_id,
                        reference_room_id,
                    )
                topology_room_route = route_cache[route_key]
                if not topology_room_route:
                    continue
                route_found = True
                result = constrained_shortest_pano_path(
                    pano_graph,
                    pano_room_mappings,
                    start_pano_id=start_pano_id,
                    target_pano_ids=panos_by_room[reference_room_id],
                    topology_room_route=topology_room_route,
                    edge_distances=edge_distances,
                )
                if result is None:
                    continue
                path_distance_m, end_pano_id, shortest_path_panos = result
                candidates.append(
                    (
                        path_distance_m,
                        end_pano_id,
                        reference_room_id,
                        topology_room_route,
                        shortest_path_panos,
                    )
                )

            if not candidates:
                if route_found:
                    skipped_topology_constrained_unreachable += 1
                else:
                    skipped_topology_route_not_found += 1
                continue

            (
                path_distance_m,
                end_pano_id,
                reference_end_room_id,
                shortest_path_rooms,
                shortest_path_panos,
            ) = min(candidates, key=lambda item: (item[0], item[1], item[2]))
            if any(
                str(pano_graph[pano_id].get("floor")) != floor
                for pano_id in shortest_path_panos
            ):
                raise ValueError(
                    f"Path {start_pano_id} -> {end_pano_id} leaves floor {floor}."
                )
            if not validate_topology_constrained_path(
                shortest_path_panos,
                shortest_path_rooms,
                pano_room_mappings,
            ):
                raise ValueError(
                    f"Constrained path {start_pano_id} -> {end_pano_id} violates "
                    f"topology route {shortest_path_rooms}."
                )
            collapsed = collapsed_room_sequence(
                shortest_path_panos,
                pano_room_mappings,
            )
            if collapsed != shortest_path_rooms:
                raise ValueError(
                    f"Mapped pano sequence {collapsed} does not equal evaluator BFS "
                    f"route {shortest_path_rooms}."
                )

            straight_line_distance_m = distance_between_panos_m(
                pano_graph,
                start_pano_id,
                end_pano_id,
            )
            if straight_line_distance_m is None:
                skipped_missing_distance += 1
                continue
            if straight_line_distance_m <= 0.0:
                skipped_zero_straight_distance += 1
                continue

            shortest_path_pano_rooms = [
                pano_room_mappings.get(pano_id) for pano_id in shortest_path_panos
            ]
            null_pano_count = sum(
                room_id is None for room_id in shortest_path_pano_rooms
            )
            visited_room_count = len(shortest_path_rooms)
            record = {
                "path_id": f"NP{len(records) + 1:06d}",
                "floor": floor,
                "start_pano_id": start_pano_id,
                "start_room_id": start_room_id,
                "target_group_id": target_group["target_group_id"],
                "target_group_theme": target_group["target_group_theme"],
                "acceptable_target_room_ids": list(acceptable_room_ids),
                "reference_end_room_id": reference_end_room_id,
                "target_room_id": reference_end_room_id,
                "end_pano_id": end_pano_id,
                "end_room_id": reference_end_room_id,
                "shortest_path_panos": shortest_path_panos,
                "shortest_path_pano_rooms": shortest_path_pano_rooms,
                "shortest_path_rooms": list(shortest_path_rooms),
                "pano_step_count": max(len(shortest_path_panos) - 1, 0),
                "visited_room_count": visited_room_count,
                "room_transition_count": max(visited_room_count - 1, 0),
                "path_distance_m": path_distance_m,
                "straight_line_distance_m": straight_line_distance_m,
                "detour_ratio": path_distance_m / straight_line_distance_m,
                "difficulty": difficulty_for_visited_room_count(
                    visited_room_count,
                    easy_max_visited_rooms=easy_max,
                    medium_max_visited_rooms=medium_max,
                ),
                "path_constraint_mode": "room_topology_bfs_constrained_dijkstra",
                "topology_validation_status": "valid",
                "null_pano_count": null_pano_count,
            }
            records.append(record)
            if args.max_records is not None and len(records) >= args.max_records:
                stop = True
                break
        if stop:
            break

    summary = summarize_records(
        records,
        floor=floor,
        floor_room_ids=floor_room_ids,
        topology_room_graph=topology_room_graph,
        start_pano_count=len(start_panos),
        target_groups=target_groups,
        skipped_same_target_group_count=skipped_same_target_group,
        skipped_topology_route_not_found_count=skipped_topology_route_not_found,
        skipped_topology_constrained_unreachable_count=(
            skipped_topology_constrained_unreachable
        ),
        skipped_missing_distance_count=skipped_missing_distance,
        skipped_zero_straight_distance_count=skipped_zero_straight_distance,
        easy_max_visited_rooms=easy_max,
        medium_max_visited_rooms=medium_max,
    )
    summary["artifacts_dir"] = str(artifacts_dir)
    summary["target_groups_json"] = (
        str(resolve_project_path(args.target_groups_json))
        if args.target_groups_json is not None
        else None
    )
    summary["filtered_pano_graph_count"] = len(pano_graph)
    summary["mapped_floor_pano_count"] = sum(map(len, panos_by_room.values()))
    return records, summary


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in CSV_FIELDS}
            row["acceptable_target_room_ids"] = " | ".join(
                record["acceptable_target_room_ids"]
            )
            row["shortest_path_rooms"] = " -> ".join(record["shortest_path_rooms"])
            row["shortest_path_panos"] = " -> ".join(record["shortest_path_panos"])
            row["shortest_path_pano_rooms"] = " -> ".join(
                "null" if room_id is None else str(room_id)
                for room_id in record["shortest_path_pano_rooms"]
            )
            writer.writerow(row)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_visited_room_count_plot(output_dir: Path, summary: dict) -> Path:
    plt = _pyplot()
    distribution = summary["visited_room_count_distribution"]
    counts = [item["visited_room_count"] for item in distribution]
    frequencies = [item["count"] for item in distribution]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        counts,
        frequencies,
        color="#5470C6",
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, item in zip(bars, distribution, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{item['count']:,}\n{item['percentage']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title("Floor 0 candidate paths by visited room count")
    ax.set_xlabel("Visited room count (start, target, and circulation included)")
    ax.set_ylabel("Candidate path count")
    ax.set_xticks(counts)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "visited_room_count_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_detour_ratio_histograms(
    output_dir: Path,
    records: list[dict],
    summary: dict,
) -> dict[str, Path]:
    plt = _pyplot()
    overall = summary["detour_ratio_overall"]
    if not records or overall["p99"] is None or overall["min"] is None:
        raise ValueError("Cannot plot detour ratio histograms without records.")
    minimum = float(overall["min"])
    p99 = float(overall["p99"])
    if p99 <= minimum:
        p99 = minimum + 1e-6
    bins = np.linspace(minimum, p99, 51)
    widths = np.diff(bins)

    prepared = {}
    common_y_max = 0.0
    for difficulty in DIFFICULTY_ORDER:
        values = np.asarray(
            [
                float(record["detour_ratio"])
                for record in records
                if record["difficulty"] == difficulty
            ],
            dtype=np.float64,
        )
        visible = values[values <= p99]
        bin_counts, _ = np.histogram(visible, bins=bins)
        percentages = (
            100.0 * bin_counts / values.size
            if values.size
            else np.zeros(len(bins) - 1, dtype=np.float64)
        )
        common_y_max = max(common_y_max, float(np.max(percentages)))
        prepared[difficulty] = {
            "values": values,
            "percentages": percentages,
            "tail_count": int(values.size - visible.size),
        }
    common_y_max = max(common_y_max * 1.15, 1.0)

    output_paths = {}
    for difficulty in DIFFICULTY_ORDER:
        item = prepared[difficulty]
        values = item["values"]
        percentages = item["percentages"]
        tail_count = item["tail_count"]
        color = DIFFICULTY_COLORS[difficulty]
        stats = summary["detour_ratio_by_difficulty"][difficulty]

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(
            bins[:-1],
            percentages,
            width=widths,
            align="edge",
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
        )
        if values.size:
            p25 = float(stats["p25"])
            median = float(stats["median"])
            p75 = float(stats["p75"])
            ax.axvspan(
                p25,
                p75,
                color=color,
                alpha=0.12,
                label=f"IQR: {p25:.2f}-{p75:.2f}",
            )
            ax.axvline(
                median,
                color="#222222",
                linewidth=2.0,
                label=f"Median: {median:.2f}",
            )
            tail_percentage = 100.0 * tail_count / values.size
            annotation = (
                f"n = {values.size:,}\n"
                f"Above overall P99 ({p99:.2f}): "
                f"{tail_count:,} ({tail_percentage:.2f}%)"
            )
            ax.legend()
        else:
            annotation = "No records in this difficulty group"
            ax.text(
                0.5,
                0.5,
                annotation,
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        gallery_range = summary["difficulty_definition"][difficulty]
        ax.set_title(
            f"{difficulty.title()} paths ({gallery_range} visited rooms): "
            "raw detour ratio"
        )
        ax.set_xlabel(
            "Shortest-path distance / straight-line distance "
            "(zoomed to overall P99)"
        )
        ax.set_ylabel("Within-group percentage of paths per bin")
        ax.set_xlim(minimum, p99)
        ax.set_ylim(0.0, common_y_max)
        ax.grid(axis="y", alpha=0.25)
        if values.size:
            ax.text(
                0.98,
                0.96,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "alpha": 0.85,
                },
            )
        fig.tight_layout()
        output_path = output_dir / f"detour_ratio_histogram_{difficulty}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths[f"detour_ratio_histogram_{difficulty}"] = output_path
    return output_paths


def write_detour_ratio_ecdf(output_dir: Path, records: list[dict]) -> Path:
    plt = _pyplot()
    if not records:
        raise ValueError("Cannot plot detour ratio ECDF without records.")

    fig, ax = plt.subplots(figsize=(11, 6))
    for difficulty in DIFFICULTY_ORDER:
        values = np.sort(
            np.asarray(
                [
                    float(record["detour_ratio"])
                    for record in records
                    if record["difficulty"] == difficulty
                ],
                dtype=np.float64,
            )
        )
        if values.size:
            cumulative = np.arange(1, values.size + 1, dtype=np.float64) / values.size
            ax.step(
                values,
                cumulative,
                where="post",
                color=DIFFICULTY_COLORS[difficulty],
                label=difficulty.title(),
            )
    ax.set_xscale("log")
    ax.set_title("Full raw detour ratio ECDF by visited-room difficulty")
    ax.set_xlabel("Shortest-path distance / straight-line distance (log scale)")
    ax.set_ylabel("Cumulative proportion")
    ax.set_ylim(0.0, 1.01)
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "detour_ratio_ecdf.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_detour_ratio_by_visited_room_count_heatmap(
    output_dir: Path,
    records: list[dict],
    summary: dict,
    *,
    bin_width: float = DETOUR_HEATMAP_BIN_WIDTH,
) -> Path:
    """Plot column-normalized detour-ratio distributions by visited room count."""
    plt = _pyplot()
    if not records:
        raise ValueError("Cannot plot detour-ratio heatmap without records.")
    if not math.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("Heatmap bin width must be a positive finite number.")

    room_counts = sorted({int(record["visited_room_count"]) for record in records})
    overall = summary["detour_ratio_overall"]
    minimum = float(overall["min"])
    p99 = float(overall["p99"])
    display_minimum = math.floor(minimum / bin_width + 1e-12) * bin_width
    display_maximum = math.ceil(p99 / bin_width - 1e-12) * bin_width
    display_maximum = max(display_maximum, display_minimum + bin_width)
    edges = np.arange(
        display_minimum,
        display_maximum + bin_width * 0.5,
        bin_width,
        dtype=np.float64,
    )

    percentages = np.zeros((len(edges), len(room_counts)), dtype=np.float64)
    for column, room_count in enumerate(room_counts):
        values = np.asarray(
            [
                float(record["detour_ratio"])
                for record in records
                if int(record["visited_room_count"]) == room_count
            ],
            dtype=np.float64,
        )
        visible = values[values <= display_maximum]
        regular_counts, _ = np.histogram(visible, bins=edges)
        overflow_count = int(values.size - visible.size)
        all_counts = np.concatenate(
            [regular_counts, np.asarray([overflow_count], dtype=np.int64)]
        )
        percentages[:, column] = 100.0 * all_counts / values.size

    y_labels = [
        f"{lower:.1f}–{upper:.1f}"
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    ]
    y_labels.append(f">{display_maximum:.1f}")

    fig_width = max(12.0, 0.95 * len(room_counts) + 3.0)
    fig_height = max(8.0, 0.48 * len(y_labels) + 2.2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.038),
        wspace=0.025,
        left=0.10,
        right=0.90,
        bottom=0.08,
        top=0.92,
    )
    ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    x_positions = np.arange(len(room_counts))
    image = ax.imshow(
        percentages,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
    )

    for row in range(percentages.shape[0]):
        for column in range(percentages.shape[1]):
            value = float(percentages[row, column])
            if value <= 0.0:
                continue
            label = "<0.1" if 0.0 < value < 0.05 else f"{value:.1f}"
            color = "white" if image.norm(value) >= 0.55 else "#222222"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(room_counts)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(-0.5, len(room_counts), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle("Detour Ratio Distribution by Visited Room Count", fontsize=14)
    ax.set_xlabel("Visited Room Count")
    ax.set_ylabel("Detour Ratio")
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Path Percentage (%)")
    path = output_dir / "detour_ratio_by_visited_room_count_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_shortest_path_distance_by_visited_room_count_heatmap(
    output_dir: Path,
    records: list[dict],
    *,
    bin_width_m: float = SHORTEST_PATH_HEATMAP_BIN_WIDTH_M,
) -> Path:
    """Plot column-normalized shortest-path distance distributions."""
    plt = _pyplot()
    if not records:
        raise ValueError("Cannot plot shortest-path distance heatmap without records.")
    if not math.isfinite(bin_width_m) or bin_width_m <= 0.0:
        raise ValueError("Distance heatmap bin width must be positive and finite.")

    room_counts = sorted({int(record["visited_room_count"]) for record in records})
    all_distances = np.asarray(
        [float(record["path_distance_m"]) for record in records],
        dtype=np.float64,
    )
    display_minimum = math.floor(
        float(np.min(all_distances)) / bin_width_m + 1e-12
    ) * bin_width_m
    display_maximum = math.ceil(
        float(np.max(all_distances)) / bin_width_m - 1e-12
    ) * bin_width_m
    display_maximum = max(display_maximum, display_minimum + bin_width_m)
    edges = np.arange(
        display_minimum,
        display_maximum + bin_width_m * 0.5,
        bin_width_m,
        dtype=np.float64,
    )

    percentages = np.zeros((len(edges) - 1, len(room_counts)), dtype=np.float64)
    for column, room_count in enumerate(room_counts):
        values = np.asarray(
            [
                float(record["path_distance_m"])
                for record in records
                if int(record["visited_room_count"]) == room_count
            ],
            dtype=np.float64,
        )
        bin_counts, _ = np.histogram(values, bins=edges)
        percentages[:, column] = 100.0 * bin_counts / values.size

    y_labels = [
        f"{lower:g}–{upper:g}"
        for lower, upper in zip(edges[:-1], edges[1:], strict=True)
    ]
    fig_width = max(12.0, 0.95 * len(room_counts) + 3.0)
    fig_height = max(8.0, 0.42 * len(y_labels) + 2.2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    grid = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 0.038),
        wspace=0.025,
        left=0.10,
        right=0.90,
        bottom=0.08,
        top=0.92,
    )
    ax = fig.add_subplot(grid[0, 0])
    colorbar_ax = fig.add_subplot(grid[0, 1])
    x_positions = np.arange(len(room_counts))
    image = ax.imshow(
        percentages,
        origin="lower",
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
    )

    for row in range(percentages.shape[0]):
        for column in range(percentages.shape[1]):
            value = float(percentages[row, column])
            if value <= 0.0:
                continue
            label = "<0.1" if value < 0.05 else f"{value:.1f}"
            color = "white" if image.norm(value) >= 0.55 else "#222222"
            ax.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=6.5,
                color=color,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(room_counts)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(np.arange(-0.5, len(room_counts), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        "Shortest-Path Distance Distribution by Visited Room Count",
        fontsize=14,
    )
    ax.set_xlabel("Visited Room Count")
    ax.set_ylabel("Shortest-Path Distance (m)")
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_label("Path Percentage (%)")
    path = output_dir / "shortest_path_distance_by_visited_room_count_heatmap.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(output_dir: Path, records: list[dict], summary: dict) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": output_dir / "navigation_paths.jsonl",
        "csv": output_dir / "navigation_paths.csv",
        "summary": output_dir / "summary.json",
    }
    write_jsonl(paths["jsonl"], records)
    write_csv(paths["csv"], records)
    write_json(paths["summary"], summary)
    paths["visited_room_count_plot"] = write_visited_room_count_plot(output_dir, summary)
    paths.update(
        write_detour_ratio_histograms(
            output_dir,
            records,
            summary,
        )
    )
    paths["detour_ratio_ecdf"] = write_detour_ratio_ecdf(output_dir, records)
    paths["detour_ratio_heatmap"] = (
        write_detour_ratio_by_visited_room_count_heatmap(
            output_dir,
            records,
            summary,
        )
    )
    paths["shortest_path_distance_heatmap"] = (
        write_shortest_path_distance_by_visited_room_count_heatmap(
            output_dir,
            records,
        )
    )
    return paths


def main() -> None:
    args = build_parser().parse_args()
    records, summary = build_navigation_paths(args)
    output_dir = resolve_project_path(args.output_dir)
    paths = write_outputs(output_dir, records, summary)
    print(
        json.dumps(
            {
                "navigation_path_count": len(records),
                "floor": summary["floor"],
                "difficulty_distribution": summary["difficulty_distribution"],
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
