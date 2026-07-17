from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.data.memory_localization import write_json  # noqa: E402
from tools.data.build_navigation_paths import (  # noqa: E402
    DIFFICULTY_ORDER,
    resolve_project_path,
)

DEFAULT_INPUT = "outputs/navigation_paths/floor0_semantic_targets/navigation_paths.jsonl"
DEFAULT_TOPOLOGY_INPUT = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "navigation_paths.jsonl"
)
DEFAULT_PREVIOUS_FIXED_TEST_SET = (
    "outputs/navigation_paths/floor0_semantic_targets/"
    "pilot90_fixed_ratio_passage_controlled/test_set.jsonl"
)
DEFAULT_OUTPUT_DIR = "outputs/navigation_paths/floor0_semantic_targets/pilot90"
DEFAULT_FIXED_OUTPUT_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets/"
    "pilot90_fixed_ratio_passage_controlled"
)
DEFAULT_TOPOLOGY_FIXED_OUTPUT_DIR = (
    "outputs/navigation_paths/floor0_semantic_targets_topology_constrained/"
    "pilot90_fixed_ratio_passage_controlled"
)
DEFAULT_FAIL_EDGES_PATH = (
    "dataset/sites/british_museum/room_graph/manual_annotations/"
    "passage_selection_fail_edges_floor0.json"
)
RATIO_TERTILE_ORDER = ("low", "middle", "high")
FIXED_RATIO_STRATA = (
    {"label": "1.0-1.5", "lower_bound": 1.0, "upper_bound": 1.5, "lower_inclusive": True, "upper_inclusive": False},
    {"label": "1.5-2.0", "lower_bound": 1.5, "upper_bound": 2.0, "lower_inclusive": True, "upper_inclusive": False},
    {"label": "2.0-2.5", "lower_bound": 2.0, "upper_bound": 2.5, "lower_inclusive": True, "upper_inclusive": True},
)
FIXED_RATIO_STRATUM_ORDER = tuple(item["label"] for item in FIXED_RATIO_STRATA)
PASSAGE_PROFILE_ORDER = ("reliable", "risk")
PILOT_CSV_FIELDS = (
    "test_id",
    "path_id",
    "difficulty",
    "ratio_tertile",
    "ratio_tertile_low_exclusive",
    "ratio_tertile_high_inclusive",
    "ratio_stratum",
    "ratio_stratum_lower_bound",
    "ratio_stratum_upper_bound",
    "ratio_stratum_lower_inclusive",
    "ratio_stratum_upper_inclusive",
    "passage_profile",
    "known_failed_passage_edges_on_path",
    "query_id",
    "query",
    "target_group_id",
    "target_group_theme",
    "acceptable_target_room_ids",
    "reference_end_room_id",
    "start_pano_id",
    "start_room_id",
    "end_pano_id",
    "visited_room_count",
    "path_distance_m",
    "straight_line_distance_m",
    "detour_ratio",
    "path_constraint_mode",
    "topology_validation_status",
    "null_pano_count",
    "shortest_path_rooms",
    "shortest_path_panos",
    "shortest_path_pano_rooms",
)
ROOM_ID_PATTERN = re.compile(r"\bRooms?\s*\d+[A-Za-z]?(?:\s*[-–]\s*\d+[A-Za-z]?)?", re.I)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample a deterministic semantic-target navigation pilot using either "
            "the legacy within-difficulty tertiles or fixed ratio bins with known "
            "passage-failure exposure controlled by MILP."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--sampling-mode", choices=("tertile", "fixed-passage-controlled"), default="tertile")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--fail-edges", default=DEFAULT_FAIL_EDGES_PATH)
    parser.add_argument("--reliable-per-stratum", type=int, default=8)
    parser.add_argument("--risk-per-stratum", type=int, default=2)
    parser.add_argument(
        "--preserve-test-set",
        help=(
            "Existing fixed-ratio Pilot90 to preserve where its room route already "
            "matches the topology-constrained source. Only valid with "
            "fixed-passage-controlled mode."
        ),
    )
    parser.add_argument("--expected-preserved-count", type=int, default=43)
    return parser


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(record)
    return records


def ratio_tertile_cutpoints(records: Iterable[dict]) -> dict[str, dict[str, float]]:
    records = list(records)
    cutpoints: dict[str, dict[str, float]] = {}
    for difficulty in DIFFICULTY_ORDER:
        values = np.asarray(
            [
                float(record["detour_ratio"])
                for record in records
                if record["difficulty"] == difficulty
            ],
            dtype=np.float64,
        )
        if values.size == 0:
            raise ValueError(f"No candidate records for difficulty {difficulty!r}.")
        low_middle, middle_high = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
        cutpoints[difficulty] = {
            "low_middle": float(low_middle),
            "middle_high": float(middle_high),
        }
    return cutpoints


def ratio_tertile_for_value(value: float, *, low_middle: float, middle_high: float) -> str:
    if value <= low_middle:
        return "low"
    if value <= middle_high:
        return "middle"
    return "high"


def add_ratio_tertiles(
    records: Iterable[dict],
    cutpoints: dict[str, dict[str, float]],
) -> list[dict]:
    enriched: list[dict] = []
    for source in records:
        record = dict(source)
        difficulty = str(record["difficulty"])
        bounds = cutpoints[difficulty]
        tertile = ratio_tertile_for_value(
            float(record["detour_ratio"]),
            low_middle=bounds["low_middle"],
            middle_high=bounds["middle_high"],
        )
        record["ratio_tertile"] = tertile
        record["ratio_tertile_low_exclusive"] = (
            None if tertile == "low" else bounds["low_middle"]
            if tertile == "middle"
            else bounds["middle_high"]
        )
        record["ratio_tertile_high_inclusive"] = (
            bounds["low_middle"] if tertile == "low" else
            bounds["middle_high"] if tertile == "middle" else None
        )
        enriched.append(record)
    return enriched


def normalize_passage_room_id(room_id: str) -> str:
    normalized = str(room_id).strip()
    if normalized.casefold() == "north stairs":
        return "North stairs"
    return normalized


def load_directed_fail_edges(path: Path) -> tuple[tuple[str, str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("directed_fail_edges")
    if not isinstance(values, list):
        raise ValueError(f"{path} does not contain directed_fail_edges.")
    edges: list[tuple[str, str]] = []
    for index, item in enumerate(values):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid fail edge at index {index}: {item!r}")
        source = normalize_passage_room_id(item.get("from_room_id", ""))
        target = normalize_passage_room_id(item.get("to_room_id", ""))
        if not source or not target:
            raise ValueError(f"Fail edge {index} has an empty endpoint.")
        edges.append((source, target))
    if len(set(edges)) != len(edges):
        raise ValueError("directed_fail_edges contains duplicates.")
    return tuple(edges)


def fixed_ratio_stratum_for_value(value: float) -> dict | None:
    ratio = float(value)
    for item in FIXED_RATIO_STRATA:
        lower_ok = ratio >= item["lower_bound"]
        upper_ok = (
            ratio <= item["upper_bound"]
            if item["upper_inclusive"]
            else ratio < item["upper_bound"]
        )
        if lower_ok and upper_ok:
            return dict(item)
    return None


def add_fixed_ratio_passage_profiles(
    records: Iterable[dict],
    fail_edges: Iterable[tuple[str, str]],
) -> tuple[list[dict], dict]:
    fail_edge_set = {
        (normalize_passage_room_id(source), normalize_passage_room_id(target))
        for source, target in fail_edges
    }
    enriched: list[dict] = []
    excluded_by_difficulty: Counter[str] = Counter()
    for source in records:
        stratum = fixed_ratio_stratum_for_value(float(source["detour_ratio"]))
        difficulty = str(source["difficulty"])
        if stratum is None:
            excluded_by_difficulty[difficulty] += 1
            continue
        rooms = [
            normalize_passage_room_id(room_id)
            for room_id in source["shortest_path_rooms"]
        ]
        hits: list[tuple[str, str]] = []
        seen_hits: set[tuple[str, str]] = set()
        for edge in zip(rooms, rooms[1:]):
            if edge in fail_edge_set and edge not in seen_hits:
                hits.append(edge)
                seen_hits.add(edge)
        record = dict(source)
        record["ratio_stratum"] = stratum["label"]
        record["ratio_stratum_lower_bound"] = stratum["lower_bound"]
        record["ratio_stratum_upper_bound"] = stratum["upper_bound"]
        record["ratio_stratum_lower_inclusive"] = stratum["lower_inclusive"]
        record["ratio_stratum_upper_inclusive"] = stratum["upper_inclusive"]
        record["passage_profile"] = "risk" if hits else "reliable"
        record["known_failed_passage_edges_on_path"] = [
            {"from_room_id": edge[0], "to_room_id": edge[1]}
            for edge in hits
        ]
        enriched.append(record)

    candidate_counts: dict[str, dict[str, dict[str, int]]] = {}
    counts = Counter(
        (
            str(record["difficulty"]),
            str(record["ratio_stratum"]),
            str(record["passage_profile"]),
        )
        for record in enriched
    )
    for difficulty in DIFFICULTY_ORDER:
        candidate_counts[difficulty] = {}
        for stratum in FIXED_RATIO_STRATUM_ORDER:
            reliable = counts[(difficulty, stratum, "reliable")]
            risk = counts[(difficulty, stratum, "risk")]
            candidate_counts[difficulty][stratum] = {
                "total": reliable + risk,
                "reliable": reliable,
                "risk": risk,
            }
    diagnostics = {
        "eligible_candidate_count": len(enriched),
        "excluded_outside_ratio_range_count": sum(excluded_by_difficulty.values()),
        "excluded_outside_ratio_range_by_difficulty": {
            difficulty: excluded_by_difficulty[difficulty]
            for difficulty in DIFFICULTY_ORDER
        },
        "candidate_counts_by_stratum_and_passage_profile": candidate_counts,
    }
    return enriched, diagnostics


def _validate_fixed_sample(
    selected: list[dict],
    source_records: list[dict],
    *,
    reliable_per_stratum: int,
    risk_per_stratum: int,
) -> None:
    expected_per_stratum = reliable_per_stratum + risk_per_stratum
    expected_count = (
        len(DIFFICULTY_ORDER) * len(FIXED_RATIO_STRATUM_ORDER) * expected_per_stratum
    )
    if len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} selected cases, got {len(selected)}.")

    profile_counts = Counter(
        (
            str(record["difficulty"]),
            str(record["ratio_stratum"]),
            str(record["passage_profile"]),
        )
        for record in selected
    )
    for difficulty in DIFFICULTY_ORDER:
        for stratum in FIXED_RATIO_STRATUM_ORDER:
            expected = {
                "reliable": reliable_per_stratum,
                "risk": risk_per_stratum,
            }
            for profile, count in expected.items():
                observed = profile_counts[(difficulty, stratum, profile)]
                if observed != count:
                    raise ValueError(
                        f"{difficulty}/{stratum}/{profile}: expected {count}, got {observed}."
                    )

    start_panos = [str(record["start_pano_id"]) for record in selected]
    pairs = [
        (str(record["start_room_id"]), str(record["target_group_id"]))
        for record in selected
    ]
    sequences = [_ordered_room_sequence_key(record) for record in selected]
    if len(set(start_panos)) != len(selected):
        raise ValueError("Selected start panoramas are not unique.")
    if len(set(pairs)) != len(selected):
        raise ValueError("Selected start-room/target-group pairs are not unique.")
    if len(set(sequences)) != len(selected):
        raise ValueError("Selected ordered room sequences are not unique.")

    target_counts = Counter(str(record["target_group_id"]) for record in selected)
    all_targets = {str(record["target_group_id"]) for record in source_records}
    if set(target_counts) != all_targets or any(
        count < 3 or count > 4 for count in target_counts.values()
    ):
        raise ValueError("Each source target group must appear 3-4 times.")

    start_counts = Counter(str(record["start_room_id"]) for record in selected)
    all_start_rooms = {str(record["start_room_id"]) for record in source_records}
    if set(start_counts) != all_start_rooms or any(
        count > 4 for count in start_counts.values()
    ):
        raise ValueError("Every source start room must appear, with at most four cases.")

    required_endpoint_groups = {
        "assyria_nimrud": {"Room 7", "Room 8"},
        "india": {"Room 29a", "Room 29b"},
    }
    for group_id, expected_room_ids in required_endpoint_groups.items():
        observed = {
            str(record["reference_end_room_id"])
            for record in selected
            if str(record["target_group_id"]) == group_id
        }
        if observed != expected_room_ids:
            raise ValueError(
                f"Endpoint coverage for {group_id!r} is {sorted(observed)}, "
                f"expected {sorted(expected_room_ids)}."
            )


def sample_fixed_ratio_passage_controlled(
    records: Iterable[dict],
    fail_edges: Iterable[tuple[str, str]],
    *,
    seed: int = 42,
    reliable_per_stratum: int = 8,
    risk_per_stratum: int = 2,
) -> tuple[list[dict], dict]:
    if reliable_per_stratum <= 0 or risk_per_stratum <= 0:
        raise ValueError("Passage-profile quotas must both be positive.")
    source_records = list(records)
    eligible, diagnostics = add_fixed_ratio_passage_profiles(
        source_records,
        fail_edges,
    )
    required_by_profile = {
        "reliable": reliable_per_stratum,
        "risk": risk_per_stratum,
    }
    shortages: list[str] = []
    for difficulty in DIFFICULTY_ORDER:
        for stratum in FIXED_RATIO_STRATUM_ORDER:
            counts = diagnostics["candidate_counts_by_stratum_and_passage_profile"][
                difficulty
            ][stratum]
            for profile, required in required_by_profile.items():
                if counts[profile] < required:
                    shortages.append(
                        f"{difficulty}/{stratum}/{profile}: "
                        f"{counts[profile]} candidates for {required} required"
                    )
    if shortages:
        raise ValueError(
            "Fixed-ratio passage quotas are infeasible before diversity constraints. "
            + "; ".join(shortages)
        )

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except ImportError as exc:
        raise RuntimeError(
            "fixed-passage-controlled sampling requires SciPy."
        ) from exc

    candidates = sorted(eligible, key=lambda record: str(record["path_id"]))
    random.Random(seed).shuffle(candidates)
    variable_count = len(candidates)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(indices: Iterable[int], lower: float, upper: float) -> None:
        row = len(lower_bounds)
        for column in indices:
            row_indices.append(row)
            column_indices.append(column)
            values.append(1.0)
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))

    def grouped_indices(key_function) -> dict[object, list[int]]:
        groups: dict[object, list[int]] = defaultdict(list)
        for index, record in enumerate(candidates):
            groups[key_function(record)].append(index)
        return groups

    cell_profile_indices = grouped_indices(
        lambda record: (
            str(record["difficulty"]),
            str(record["ratio_stratum"]),
            str(record["passage_profile"]),
        )
    )
    for difficulty in DIFFICULTY_ORDER:
        for stratum in FIXED_RATIO_STRATUM_ORDER:
            for profile in PASSAGE_PROFILE_ORDER:
                required = required_by_profile[profile]
                add_constraint(
                    cell_profile_indices.get((difficulty, stratum, profile), []),
                    required,
                    required,
                )

    target_indices = grouped_indices(lambda record: str(record["target_group_id"]))
    for group_id in sorted(
        {str(record["target_group_id"]) for record in source_records}
    ):
        add_constraint(target_indices.get(group_id, []), 3, 4)

    start_room_indices = grouped_indices(lambda record: str(record["start_room_id"]))
    for room_id in sorted({str(record["start_room_id"]) for record in source_records}):
        add_constraint(start_room_indices.get(room_id, []), 1, 4)

    for indices in grouped_indices(
        lambda record: str(record["start_pano_id"])
    ).values():
        add_constraint(indices, 0, 1)
    for indices in grouped_indices(
        lambda record: (
            str(record["start_room_id"]),
            str(record["target_group_id"]),
        )
    ).values():
        add_constraint(indices, 0, 1)
    for indices in grouped_indices(_ordered_room_sequence_key).values():
        add_constraint(indices, 0, 1)

    required_endpoint_groups = {
        "assyria_nimrud": ("Room 7", "Room 8"),
        "india": ("Room 29a", "Room 29b"),
    }
    endpoint_indices = grouped_indices(
        lambda record: (
            str(record["target_group_id"]),
            str(record["reference_end_room_id"]),
        )
    )
    for group_id, room_ids in required_endpoint_groups.items():
        for room_id in room_ids:
            add_constraint(endpoint_indices.get((group_id, room_id), []), 1, np.inf)

    matrix = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(lower_bounds), variable_count),
    ).tocsr()
    result = milp(
        c=np.zeros(variable_count, dtype=np.float64),
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(
            np.zeros(variable_count, dtype=np.float64),
            np.ones(variable_count, dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        candidate_counts = json.dumps(
            diagnostics["candidate_counts_by_stratum_and_passage_profile"],
            ensure_ascii=False,
            sort_keys=True,
        )
        raise ValueError(
            "MILP could not satisfy the fixed-ratio passage-controlled constraints: "
            f"{result.message}. Candidate counts: {candidate_counts}"
        )

    selected = [
        record
        for record, selected_value in zip(candidates, result.x)
        if selected_value > 0.5
    ]
    _validate_fixed_sample(
        selected,
        source_records,
        reliable_per_stratum=reliable_per_stratum,
        risk_per_stratum=risk_per_stratum,
    )
    selected.sort(
        key=lambda record: (
            DIFFICULTY_ORDER.index(str(record["difficulty"])),
            FIXED_RATIO_STRATUM_ORDER.index(str(record["ratio_stratum"])),
            PASSAGE_PROFILE_ORDER.index(str(record["passage_profile"])),
            str(record["path_id"]),
        )
    )
    diagnostics["milp"] = {
        "solver": "scipy.optimize.milp",
        "status": int(result.status),
        "message": str(result.message),
        "constraint_count": len(lower_bounds),
        "variable_count": variable_count,
        "seed": seed,
    }
    return selected, diagnostics



def _source_case_key(record: dict) -> tuple[str, str]:
    return str(record["start_pano_id"]), str(record["target_group_id"])


def _start_target_pair_key(record: dict) -> tuple[str, str]:
    return str(record["start_room_id"]), str(record["target_group_id"])


def _fixed_slot_key(record: dict) -> tuple[str, str, str]:
    return (
        str(record["difficulty"]),
        str(record["ratio_stratum"]),
        str(record["passage_profile"]),
    )


def _assert_preserved_reference_unchanged(previous: dict, current: dict) -> None:
    exact_fields = (
        "path_id",
        "start_pano_id",
        "start_room_id",
        "target_group_id",
        "reference_end_room_id",
        "end_pano_id",
        "shortest_path_rooms",
        "shortest_path_panos",
        "pano_step_count",
        "visited_room_count",
        "room_transition_count",
        "difficulty",
        "ratio_stratum",
        "passage_profile",
        "known_failed_passage_edges_on_path",
    )
    for field in exact_fields:
        if previous.get(field) != current.get(field):
            raise ValueError(
                f"Topology-consistent case {previous.get('test_id')} changed {field}: "
                f"{previous.get(field)!r} != {current.get(field)!r}."
            )
    for field in (
        "path_distance_m",
        "straight_line_distance_m",
        "detour_ratio",
    ):
        if not np.isclose(
            float(previous[field]),
            float(current[field]),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(
                f"Topology-consistent case {previous.get('test_id')} changed {field}."
            )


def rebuild_fixed_ratio_passage_controlled(
    records: Iterable[dict],
    previous_test_records: Iterable[dict],
    fail_edges: Iterable[tuple[str, str]],
    queries_by_target: dict[str, dict],
    *,
    seed: int = 42,
    reliable_per_stratum: int = 8,
    risk_per_stratum: int = 2,
) -> tuple[list[dict], dict, dict, dict]:
    """Preserve topology-consistent tests and replace only inconsistent cases."""
    source_records = list(records)
    previous_records = sorted(
        (dict(record) for record in previous_test_records),
        key=lambda record: str(record["test_id"]),
    )
    if len(previous_records) != 90:
        raise ValueError(
            f"Preservation-aware rebuild expects 90 previous tests, got {len(previous_records)}."
        )
    if len({record["test_id"] for record in previous_records}) != 90:
        raise ValueError("Previous test IDs are not unique.")

    eligible, diagnostics = add_fixed_ratio_passage_profiles(
        source_records,
        fail_edges,
    )
    source_by_case = {_source_case_key(record): record for record in eligible}
    if len(source_by_case) != len(eligible):
        raise ValueError("Topology-constrained source contains duplicate start-pano/target cases.")

    preserved: list[tuple[dict, dict]] = []
    affected: list[tuple[dict, dict | None]] = []
    for previous in previous_records:
        current = source_by_case.get(_source_case_key(previous))
        if (
            current is not None
            and list(previous["shortest_path_rooms"])
            == list(current["shortest_path_rooms"])
        ):
            _assert_preserved_reference_unchanged(previous, current)
            preserved.append((previous, current))
        else:
            affected.append((previous, current))

    desired_by_profile = {
        "reliable": reliable_per_stratum,
        "risk": risk_per_stratum,
    }
    preserved_records = [current for _, current in preserved]
    preserved_slot_counts = Counter(_fixed_slot_key(record) for record in preserved_records)
    replacement_quota: dict[tuple[str, str, str], int] = {}
    for difficulty in DIFFICULTY_ORDER:
        for stratum in FIXED_RATIO_STRATUM_ORDER:
            for profile in PASSAGE_PROFILE_ORDER:
                slot = (difficulty, stratum, profile)
                quota = desired_by_profile[profile] - preserved_slot_counts[slot]
                if quota < 0:
                    raise ValueError(f"Preserved cases overfill slot {slot}: {-quota} extra.")
                replacement_quota[slot] = quota

    affected_previous_cases = {
        _source_case_key(previous) for previous, _ in affected
    }
    preserved_start_panos = {
        str(record["start_pano_id"]) for record in preserved_records
    }
    preserved_pairs = {
        _start_target_pair_key(record) for record in preserved_records
    }
    preserved_sequences = {
        _ordered_room_sequence_key(record) for record in preserved_records
    }
    candidates = [
        record
        for record in eligible
        if str(record["start_pano_id"]) not in preserved_start_panos
        and _start_target_pair_key(record) not in preserved_pairs
        and _source_case_key(record) not in affected_previous_cases
        and _ordered_room_sequence_key(record) not in preserved_sequences
    ]
    candidates.sort(key=lambda record: str(record["path_id"]))
    random.Random(seed).shuffle(candidates)

    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_array
    except ImportError as exc:
        raise RuntimeError(
            "preservation-aware fixed sampling requires SciPy."
        ) from exc

    variable_count = len(candidates)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []

    def add_constraint(indices: Iterable[int], lower: float, upper: float) -> None:
        row = len(lower_bounds)
        for column in indices:
            row_indices.append(row)
            column_indices.append(column)
            values.append(1.0)
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))

    def grouped_indices(key_function) -> dict[object, list[int]]:
        groups: dict[object, list[int]] = defaultdict(list)
        for index, record in enumerate(candidates):
            groups[key_function(record)].append(index)
        return groups

    slot_indices = grouped_indices(_fixed_slot_key)
    shortages = []
    for slot, quota in sorted(replacement_quota.items()):
        available = len(slot_indices.get(slot, []))
        if available < quota:
            shortages.append(f"{slot}: {available} available for {quota} required")
        add_constraint(slot_indices.get(slot, []), quota, quota)
    if shortages:
        raise ValueError(
            "Replacement quotas are infeasible before diversity constraints: "
            + "; ".join(shortages)
        )

    fixed_target_counts = Counter(
        str(record["target_group_id"]) for record in preserved_records
    )
    target_indices = grouped_indices(lambda record: str(record["target_group_id"]))
    for group_id in sorted({str(record["target_group_id"]) for record in source_records}):
        fixed = fixed_target_counts[group_id]
        add_constraint(target_indices.get(group_id, []), max(0, 3 - fixed), 4 - fixed)

    fixed_start_counts = Counter(
        str(record["start_room_id"]) for record in preserved_records
    )
    start_indices = grouped_indices(lambda record: str(record["start_room_id"]))
    for room_id in sorted({str(record["start_room_id"]) for record in source_records}):
        fixed = fixed_start_counts[room_id]
        add_constraint(start_indices.get(room_id, []), max(0, 1 - fixed), 4 - fixed)

    for indices in grouped_indices(lambda record: str(record["start_pano_id"])).values():
        add_constraint(indices, 0, 1)
    for indices in grouped_indices(_start_target_pair_key).values():
        add_constraint(indices, 0, 1)
    for indices in grouped_indices(_ordered_room_sequence_key).values():
        add_constraint(indices, 0, 1)

    fixed_endpoint_counts = Counter(
        (
            str(record["target_group_id"]),
            str(record["reference_end_room_id"]),
        )
        for record in preserved_records
    )
    endpoint_indices = grouped_indices(
        lambda record: (
            str(record["target_group_id"]),
            str(record["reference_end_room_id"]),
        )
    )
    for group_id, room_ids in {
        "assyria_nimrud": ("Room 7", "Room 8"),
        "india": ("Room 29a", "Room 29b"),
    }.items():
        for room_id in room_ids:
            endpoint = (group_id, room_id)
            add_constraint(
                endpoint_indices.get(endpoint, []),
                max(0, 1 - fixed_endpoint_counts[endpoint]),
                np.inf,
            )

    matrix = coo_array(
        (values, (row_indices, column_indices)),
        shape=(len(lower_bounds), variable_count),
    ).tocsr()
    result = milp(
        c=np.zeros(variable_count, dtype=np.float64),
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(
            np.zeros(variable_count, dtype=np.float64),
            np.ones(variable_count, dtype=np.float64),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lower_bounds, dtype=np.float64),
            np.asarray(upper_bounds, dtype=np.float64),
        ),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "MILP could not satisfy the preservation-aware replacement constraints: "
            f"{result.message}."
        )
    selected_replacements = [
        record
        for record, selected_value in zip(candidates, result.x)
        if selected_value > 0.5
    ]
    if len(selected_replacements) != len(affected):
        raise ValueError(
            f"Expected {len(affected)} replacements, got {len(selected_replacements)}."
        )

    affected_by_slot: dict[tuple[str, str, str], list[tuple[dict, dict | None]]] = defaultdict(list)
    replacement_by_slot: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for item in affected:
        affected_by_slot[_fixed_slot_key(item[0])].append(item)
    for replacement in selected_replacements:
        replacement_by_slot[_fixed_slot_key(replacement)].append(replacement)

    output_by_test_id: dict[str, dict] = {}
    manifest_records: list[dict] = []
    for previous, current in preserved:
        record = dict(current)
        record["test_id"] = previous["test_id"]
        record["query_id"] = previous["query_id"]
        record["query"] = previous["query"]
        output_by_test_id[str(record["test_id"])] = record
        manifest_records.append(
            {
                "test_id": record["test_id"],
                "status": "preserved",
                "reason": "previous_room_sequence_matches_evaluator_bfs_route",
                "old_path_id": previous["path_id"],
                "new_path_id": current["path_id"],
                "old_start_target_pair": list(_source_case_key(previous)),
                "new_start_target_pair": list(_source_case_key(current)),
                "old_shortest_path_rooms": previous["shortest_path_rooms"],
                "new_shortest_path_rooms": current["shortest_path_rooms"],
            }
        )

    for slot in sorted(replacement_quota):
        old_items = sorted(
            affected_by_slot.get(slot, []),
            key=lambda item: str(item[0]["test_id"]),
        )
        new_items = sorted(
            replacement_by_slot.get(slot, []),
            key=lambda record: str(record["path_id"]),
        )
        if len(old_items) != len(new_items):
            raise ValueError(
                f"Replacement assignment mismatch for {slot}: "
                f"{len(old_items)} test IDs, {len(new_items)} candidates."
            )
        for (previous, recalculated_previous), replacement in zip(old_items, new_items):
            query = queries_by_target[str(replacement["target_group_id"])]
            record = dict(replacement)
            record["test_id"] = previous["test_id"]
            record["query_id"] = query["query_id"]
            record["query"] = query["query"]
            output_by_test_id[str(record["test_id"])] = record
            manifest_records.append(
                {
                    "test_id": record["test_id"],
                    "status": "replaced",
                    "reason": "previous_room_sequence_did_not_match_evaluator_bfs_route",
                    "old_path_id": previous["path_id"],
                    "new_path_id": replacement["path_id"],
                    "old_start_target_pair": list(_source_case_key(previous)),
                    "new_start_target_pair": list(_source_case_key(replacement)),
                    "old_shortest_path_rooms": previous["shortest_path_rooms"],
                    "recalculated_old_pair_rooms": (
                        recalculated_previous["shortest_path_rooms"]
                        if recalculated_previous is not None
                        else None
                    ),
                    "new_shortest_path_rooms": replacement["shortest_path_rooms"],
                }
            )

    test_records = [output_by_test_id[str(record["test_id"])] for record in previous_records]
    _validate_fixed_sample(
        test_records,
        source_records,
        reliable_per_stratum=reliable_per_stratum,
        risk_per_stratum=risk_per_stratum,
    )
    reused_affected_cases = affected_previous_cases.intersection(
        {_source_case_key(record) for record in selected_replacements}
    )
    if reused_affected_cases:
        raise ValueError(
            "Replacement reused an affected previous start-pano/target pair: "
            f"{sorted(reused_affected_cases)}"
        )

    preserved_ids = sorted(previous["test_id"] for previous, _ in preserved)
    rerun_ids = sorted(previous["test_id"] for previous, _ in affected)
    diagnostics["topology_rebuild"] = {
        "preserved_count": len(preserved_ids),
        "replacement_count": len(rerun_ids),
        "preserved_test_ids": preserved_ids,
        "rerun_test_ids": rerun_ids,
        "excluded_affected_start_pano_target_pair_count": len(affected_previous_cases),
        "replacement_candidate_count_after_exclusions": len(candidates),
        "replacement_quota_by_slot": {
            "/".join(slot): count for slot, count in sorted(replacement_quota.items())
        },
        "milp": {
            "solver": "scipy.optimize.milp",
            "status": int(result.status),
            "message": str(result.message),
            "constraint_count": len(lower_bounds),
            "variable_count": variable_count,
            "seed": seed,
        },
    }
    manifest = {
        "schema_version": 1,
        "preserved_count": len(preserved_ids),
        "replacement_count": len(rerun_ids),
        "preserved_test_ids": preserved_ids,
        "replacement_test_ids": rerun_ids,
        "records": sorted(manifest_records, key=lambda item: str(item["test_id"])),
    }
    rerun = {
        "schema_version": 1,
        "count": len(rerun_ids),
        "test_ids": rerun_ids,
    }
    return test_records, diagnostics, manifest, rerun


def canonical_query_from_theme(target_group_theme: str) -> str:
    theme = " ".join(target_group_theme.strip().split())
    theme = re.sub(r";\s*The\s+.+?Gallery$", "", theme, flags=re.I)
    theme = theme.strip(" ,.;")
    if not theme:
        raise ValueError("Target group theme has no semantic content.")
    if theme.lower().startswith("the "):
        theme = theme[4:]
    query = f"Take me to the {theme} gallery."
    if ROOM_ID_PATTERN.search(query):
        raise ValueError(f"Canonical query contains a Room ID: {query}")
    return query


def build_queries_by_target(records: Iterable[dict]) -> dict[str, dict]:
    group_records: dict[str, dict] = {}
    for record in records:
        group_records.setdefault(str(record["target_group_id"]), record)
    queries: dict[str, dict] = {}
    for index, group_id in enumerate(sorted(group_records), start=1):
        record = group_records[group_id]
        acceptable_room_ids = list(record["acceptable_target_room_ids"])
        query = canonical_query_from_theme(str(record["target_group_theme"]))
        queries[group_id] = {
            "query_id": f"Q{index:02d}",
            "target_group_id": group_id,
            "target_group_theme": record["target_group_theme"],
            "acceptable_target_room_ids": acceptable_room_ids,
            "query": query,
        }
    return queries


def _ordered_room_sequence_key(record: dict) -> tuple[str, ...]:
    return tuple(str(room_id) for room_id in record["shortest_path_rooms"])


def sample_pilot(
    records: Iterable[dict],
    *,
    seed: int = 42,
    per_stratum: int = 10,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    source_records = list(records)
    if per_stratum <= 0:
        raise ValueError("per_stratum must be positive.")
    cutpoints = ratio_tertile_cutpoints(source_records)
    enriched = add_ratio_tertiles(source_records, cutpoints)
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in enriched:
        cells[(record["difficulty"], record["ratio_tertile"])].append(record)
    ordered_cells = tuple(
        (difficulty, tertile)
        for difficulty in DIFFICULTY_ORDER
        for tertile in RATIO_TERTILE_ORDER
    )
    expected_cells = set(ordered_cells)
    if set(cells) != expected_cells:
        raise ValueError("Candidate pool does not contain all nine sampling strata.")
    for candidates in cells.values():
        rng.shuffle(candidates)

    selected: list[dict] = []
    selected_per_cell: Counter[tuple[str, str]] = Counter()
    target_counts: Counter[str] = Counter()
    start_room_counts: Counter[str] = Counter()
    endpoint_counts: Counter[tuple[str, str]] = Counter()
    used_start_panos: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    used_room_sequences: set[tuple[str, ...]] = set()

    for _round_index in range(per_stratum):
        round_cells = list(ordered_cells)
        rng.shuffle(round_cells)
        for cell in round_cells:
            valid: list[dict] = []
            for record in cells[cell]:
                group_id = str(record["target_group_id"])
                start_room_id = str(record["start_room_id"])
                start_pano_id = str(record["start_pano_id"])
                pair = (start_room_id, group_id)
                room_sequence = _ordered_room_sequence_key(record)
                if start_pano_id in used_start_panos:
                    continue
                if pair in used_pairs or room_sequence in used_room_sequences:
                    continue
                if target_counts[group_id] >= 4 or start_room_counts[start_room_id] >= 4:
                    continue
                valid.append(record)
            if not valid:
                raise ValueError(
                    f"No candidate satisfies Pilot diversity constraints for stratum {cell}."
                )
            chosen = min(
                valid,
                key=lambda record: (
                    target_counts[str(record["target_group_id"])],
                    start_room_counts[str(record["start_room_id"])],
                    endpoint_counts[(
                        str(record["target_group_id"]),
                        str(record["reference_end_room_id"]),
                    )],
                ),
            )
            selected.append(chosen)
            group_id = str(chosen["target_group_id"])
            start_room_id = str(chosen["start_room_id"])
            start_pano_id = str(chosen["start_pano_id"])
            selected_per_cell[cell] += 1
            target_counts[group_id] += 1
            start_room_counts[start_room_id] += 1
            endpoint_counts[(group_id, str(chosen["reference_end_room_id"]))] += 1
            used_start_panos.add(start_pano_id)
            used_pairs.add((start_room_id, group_id))
            used_room_sequences.add(_ordered_room_sequence_key(chosen))

    expected_count = len(expected_cells) * per_stratum
    if len(selected) != expected_count:
        raise ValueError(f"Expected {expected_count} selected cases, got {len(selected)}.")
    if any(selected_per_cell[cell] != per_stratum for cell in expected_cells):
        raise ValueError("Pilot strata are not balanced.")
    if any(count < 3 or count > 4 for count in target_counts.values()):
        raise ValueError("Each target group must appear 3-4 times.")
    all_target_groups = {str(record["target_group_id"]) for record in source_records}
    if set(target_counts) != all_target_groups:
        raise ValueError("Pilot does not cover every target group.")
    all_start_rooms = {str(record["start_room_id"]) for record in source_records}
    if set(start_room_counts) != all_start_rooms:
        raise ValueError("Pilot does not cover every start room.")
    if any(count > 4 for count in start_room_counts.values()):
        raise ValueError("A start room appears more than four times.")
    required_endpoint_groups = {
        "assyria_nimrud": {"Room 7", "Room 8"},
        "india": {"Room 29a", "Room 29b"},
    }
    for group_id, expected_room_ids in required_endpoint_groups.items():
        observed = {
            end_room_id
            for (observed_group_id, end_room_id), count in endpoint_counts.items()
            if observed_group_id == group_id and count > 0
        }
        if observed != expected_room_ids:
            raise ValueError(
                f"Pilot endpoint coverage for {group_id!r} is {sorted(observed)}, "
                f"expected {sorted(expected_room_ids)}."
            )

    selected.sort(
        key=lambda record: (
            DIFFICULTY_ORDER.index(record["difficulty"]),
            RATIO_TERTILE_ORDER.index(record["ratio_tertile"]),
            record["path_id"],
        )
    )
    return selected, cutpoints


def attach_queries_and_test_ids(
    records: Iterable[dict],
    queries_by_target: dict[str, dict],
) -> list[dict]:
    test_records: list[dict] = []
    for index, source in enumerate(records, start=1):
        record = dict(source)
        query = queries_by_target[str(record["target_group_id"])]
        record["test_id"] = f"TEST{index:03d}"
        record["query_id"] = query["query_id"]
        record["query"] = query["query"]
        test_records.append(record)
    return test_records


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_csv(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PILOT_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in PILOT_CSV_FIELDS}
            row["acceptable_target_room_ids"] = " | ".join(
                record["acceptable_target_room_ids"]
            )
            row["shortest_path_rooms"] = " -> ".join(record["shortest_path_rooms"])
            row["shortest_path_panos"] = " -> ".join(record["shortest_path_panos"])
            row["shortest_path_pano_rooms"] = " -> ".join(
                "null" if room_id is None else str(room_id)
                for room_id in record.get("shortest_path_pano_rooms", [])
            )
            row["known_failed_passage_edges_on_path"] = " | ".join(
                f"{edge['from_room_id']} -> {edge['to_room_id']}"
                for edge in record.get("known_failed_passage_edges_on_path", [])
            )
            writer.writerow(row)


def build_sample_summary(
    records: list[dict],
    *,
    source_path_count: int,
    seed: int,
    per_stratum: int,
    cutpoints: dict[str, dict[str, float]],
) -> dict:
    strata = Counter(
        (str(record["difficulty"]), str(record["ratio_tertile"]))
        for record in records
    )
    target_counts = Counter(str(record["target_group_id"]) for record in records)
    start_room_counts = Counter(str(record["start_room_id"]) for record in records)
    endpoint_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        endpoint_counts[str(record["target_group_id"])][
            str(record["reference_end_room_id"])
        ] += 1
    return {
        "seed": seed,
        "source_navigation_path_count": source_path_count,
        "sample_count": len(records),
        "per_stratum": per_stratum,
        "ratio_tertile_definition": (
            "Within each difficulty: low <= q1; q1 < middle <= q2; high > q2."
        ),
        "ratio_tertile_cutpoints_by_difficulty": cutpoints,
        "stratum_counts": {
            difficulty: {
                tertile: strata[(difficulty, tertile)]
                for tertile in RATIO_TERTILE_ORDER
            }
            for difficulty in DIFFICULTY_ORDER
        },
        "uniqueness": {
            "start_panorama_count": len({record["start_pano_id"] for record in records}),
            "start_room_target_group_pair_count": len(
                {
                    (record["start_room_id"], record["target_group_id"])
                    for record in records
                }
            ),
            "ordered_room_sequence_count": len(
                {tuple(record["shortest_path_rooms"]) for record in records}
            ),
        },
        "target_group_count": len(target_counts),
        "target_group_sample_counts": dict(sorted(target_counts.items())),
        "reference_endpoint_counts_by_target_group": {
            group_id: dict(sorted(counts.items()))
            for group_id, counts in sorted(endpoint_counts.items())
        },
        "start_room_count": len(start_room_counts),
        "start_room_sample_counts": dict(sorted(start_room_counts.items())),
        "selected_path_ids": [record["path_id"] for record in records],
    }


def build_fixed_sample_summary(
    records: list[dict],
    *,
    source_path_count: int,
    seed: int,
    reliable_per_stratum: int,
    risk_per_stratum: int,
    fail_edges_path: Path,
    fail_edge_count: int,
    diagnostics: dict,
) -> dict:
    strata = Counter(
        (
            str(record["difficulty"]),
            str(record["ratio_stratum"]),
            str(record["passage_profile"]),
        )
        for record in records
    )
    profile_counts = Counter(str(record["passage_profile"]) for record in records)
    target_counts = Counter(str(record["target_group_id"]) for record in records)
    start_room_counts = Counter(str(record["start_room_id"]) for record in records)
    endpoint_counts: dict[str, Counter[str]] = defaultdict(Counter)
    fail_edge_counts: Counter[str] = Counter()
    for record in records:
        endpoint_counts[str(record["target_group_id"])][
            str(record["reference_end_room_id"])
        ] += 1
        for edge in record["known_failed_passage_edges_on_path"]:
            fail_edge_counts[
                f"{edge['from_room_id']} -> {edge['to_room_id']}"
            ] += 1

    return {
        "sampling_design": "passage_controlled_fixed_ratio",
        "selection_bias_note": (
            "This is a passage-controlled test set, not a random sample representing "
            "the original candidate-path distribution. Each cell deliberately contains "
            "8 paths that avoid the annotated fail edges and 2 paths that contain at "
            "least one annotated fail edge."
        ),
        "reporting_scope": (
            "Navigation capability after controlling exposure to known passage-selection "
            "failures; do not report it as an unbiased overall navigation success rate."
        ),
        "seed": seed,
        "source_navigation_path_count": source_path_count,
        "sample_count": len(records),
        "per_stratum": reliable_per_stratum + risk_per_stratum,
        "reliable_per_stratum": reliable_per_stratum,
        "risk_per_stratum": risk_per_stratum,
        "ratio_stratum_definition": [
            dict(item) for item in FIXED_RATIO_STRATA
        ],
        "eligible_ratio_range": "[1.0, 2.5]",
        "passage_profile_definition": {
            "reliable": (
                "The reference shortest room path avoids all directed edges in the "
                "current manual known-failure annotation. This is not a guarantee that "
                "passage selection will succeed."
            ),
            "risk": (
                "The reference shortest room path contains at least one directed edge "
                "in the current manual known-failure annotation."
            ),
        },
        "known_fail_edges_annotation_path": str(fail_edges_path),
        "known_directed_fail_edge_count": fail_edge_count,
        **diagnostics,
        "stratum_counts": {
            difficulty: {
                stratum: {
                    "total": sum(
                        strata[(difficulty, stratum, profile)]
                        for profile in PASSAGE_PROFILE_ORDER
                    ),
                    **{
                        profile: strata[(difficulty, stratum, profile)]
                        for profile in PASSAGE_PROFILE_ORDER
                    },
                }
                for stratum in FIXED_RATIO_STRATUM_ORDER
            }
            for difficulty in DIFFICULTY_ORDER
        },
        "passage_profile_counts": {
            profile: profile_counts[profile]
            for profile in PASSAGE_PROFILE_ORDER
        },
        "selected_fail_edge_exposure_counts": dict(sorted(fail_edge_counts.items())),
        "diversity_constraints": {
            "unique_start_panorama": True,
            "unique_start_room_target_group_pair": True,
            "unique_ordered_room_sequence": True,
            "target_group_min": 3,
            "target_group_max": 4,
            "all_start_rooms_covered": True,
            "start_room_max": 4,
            "nimrud_and_india_reference_endpoints_covered": True,
        },
        "uniqueness": {
            "start_panorama_count": len({record["start_pano_id"] for record in records}),
            "start_room_target_group_pair_count": len(
                {
                    (record["start_room_id"], record["target_group_id"])
                    for record in records
                }
            ),
            "ordered_room_sequence_count": len(
                {tuple(record["shortest_path_rooms"]) for record in records}
            ),
        },
        "target_group_count": len(target_counts),
        "target_group_sample_counts": dict(sorted(target_counts.items())),
        "reference_endpoint_counts_by_target_group": {
            group_id: dict(sorted(counts.items()))
            for group_id, counts in sorted(endpoint_counts.items())
        },
        "start_room_count": len(start_room_counts),
        "start_room_sample_counts": dict(sorted(start_room_counts.items())),
        "selected_path_ids": [record["path_id"] for record in records],
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.preserve_test_set and args.sampling_mode != "fixed-passage-controlled":
        raise ValueError("--preserve-test-set requires fixed-passage-controlled mode.")
    input_path = resolve_project_path(args.input)
    default_output_dir = (
        DEFAULT_TOPOLOGY_FIXED_OUTPUT_DIR
        if args.preserve_test_set
        else DEFAULT_FIXED_OUTPUT_DIR
        if args.sampling_mode == "fixed-passage-controlled"
        else DEFAULT_OUTPUT_DIR
    )
    output_dir = resolve_project_path(args.output_dir or default_output_dir)
    source_records = load_jsonl(input_path)
    queries_by_target = build_queries_by_target(source_records)
    replacement_manifest = None
    rerun_test_ids = None

    if args.sampling_mode == "fixed-passage-controlled":
        fail_edges_path = resolve_project_path(args.fail_edges)
        fail_edges = load_directed_fail_edges(fail_edges_path)
        if args.preserve_test_set:
            previous_path = resolve_project_path(args.preserve_test_set)
            previous_records = load_jsonl(previous_path)
            test_records, diagnostics, replacement_manifest, rerun_test_ids = (
                rebuild_fixed_ratio_passage_controlled(
                    source_records,
                    previous_records,
                    fail_edges,
                    queries_by_target,
                    seed=args.seed,
                    reliable_per_stratum=args.reliable_per_stratum,
                    risk_per_stratum=args.risk_per_stratum,
                )
            )
            if replacement_manifest["preserved_count"] != args.expected_preserved_count:
                raise ValueError(
                    f"Expected {args.expected_preserved_count} preserved cases, got "
                    f"{replacement_manifest['preserved_count']}."
                )
            diagnostics["topology_rebuild"]["previous_test_set"] = str(previous_path)
        else:
            selected, diagnostics = sample_fixed_ratio_passage_controlled(
                source_records,
                fail_edges,
                seed=args.seed,
                reliable_per_stratum=args.reliable_per_stratum,
                risk_per_stratum=args.risk_per_stratum,
            )
            test_records = attach_queries_and_test_ids(selected, queries_by_target)
        summary = build_fixed_sample_summary(
            test_records,
            source_path_count=len(source_records),
            seed=args.seed,
            reliable_per_stratum=args.reliable_per_stratum,
            risk_per_stratum=args.risk_per_stratum,
            fail_edges_path=fail_edges_path,
            fail_edge_count=len(fail_edges),
            diagnostics=diagnostics,
        )
    else:
        selected, cutpoints = sample_pilot(
            source_records,
            seed=args.seed,
            per_stratum=args.per_stratum,
        )
        test_records = attach_queries_and_test_ids(selected, queries_by_target)
        summary = build_sample_summary(
            test_records,
            source_path_count=len(source_records),
            seed=args.seed,
            per_stratum=args.per_stratum,
            cutpoints=cutpoints,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "test_set.jsonl", test_records)
    write_csv(output_dir / "test_set.csv", test_records)
    write_json(output_dir / "queries_by_target.json", queries_by_target)
    write_json(output_dir / "sample_summary.json", summary)
    if replacement_manifest is not None and rerun_test_ids is not None:
        write_json(output_dir / "replacement_manifest.json", replacement_manifest)
        write_json(output_dir / "rerun_test_ids.json", rerun_test_ids)
    print(
        json.dumps(
            {
                "sampling_mode": args.sampling_mode,
                "sample_count": len(test_records),
                "target_group_count": summary["target_group_count"],
                "start_room_count": summary["start_room_count"],
                "stratum_counts": summary["stratum_counts"],
                "passage_profile_counts": summary.get("passage_profile_counts"),
                "topology_rebuild": summary.get("topology_rebuild"),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
