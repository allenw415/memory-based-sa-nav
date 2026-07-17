from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.data.memory_localization import load_json, write_json  # noqa: E402


EARTH_RADIUS_M = 6_371_000.0
MBR_AREA_EPSILON_M2 = 1e-6
MBR_MIN_ASPECT_RATIO = 0.01
DEFAULT_METADATA = "artifacts/memory_localization/floor0_siglip2_images_fov90.metadata.json"
DEFAULT_PANO_GRAPH = "dataset/sites/british_museum/normalized/pano_graph.json"
DEFAULT_ROOM_GRAPH = "dataset/sites/british_museum/normalized/room_graph.json"
DEFAULT_OUTPUT_DIR = "outputs/dataset_statistics/floor0_room_image_density"
CSV_FIELDS = (
    "room_id",
    "room_title",
    "floor",
    "geometry_status",
    "image_count",
    "panorama_count",
    "images_per_panorama",
    "coordinate_point_count",
    "hull_point_count",
    "mbr_width_m",
    "mbr_height_m",
    "mbr_orientation_deg",
    "mbr_area_m2",
    "images_per_m2",
    "panoramas_per_m2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Count memory images per gallery and estimate image density from a "
            "minimum-area bounding rectangle around panorama positions."
        )
    )
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--pano-graph", default=DEFAULT_PANO_GRAPH)
    parser.add_argument("--room-graph", default=DEFAULT_ROOM_GRAPH)
    parser.add_argument("--floor", default="0")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def valid_rooms_for_floor(room_graph: dict, *, floor: str) -> dict[str, dict]:
    rooms: dict[str, dict] = {}
    for key, raw_record in room_graph.items():
        if not isinstance(raw_record, dict):
            continue
        if str(raw_record.get("floor")) != floor:
            continue
        if raw_record.get("category") == "Circulation":
            continue
        if raw_record.get("synthetic") is True:
            continue
        room_id = str(raw_record.get("room_id") or key)
        rooms[room_id] = raw_record
    return rooms


def lat_lng(record: object) -> tuple[float, float] | None:
    if not isinstance(record, dict):
        return None
    lat = record.get("lat")
    lng = record.get("lng")
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return None
    if not math.isfinite(float(lat)) or not math.isfinite(float(lng)):
        return None
    return float(lat), float(lng)


def project_lat_lng_to_local_m(
    lat: float,
    lng: float,
    *,
    reference_lat: float,
    reference_lng: float,
) -> tuple[float, float]:
    """Project a small local region to east/north metres."""

    delta_lat = math.radians(lat - reference_lat)
    delta_lng = math.radians(lng - reference_lng)
    east_m = EARTH_RADIUS_M * math.cos(math.radians(reference_lat)) * delta_lng
    north_m = EARTH_RADIUS_M * delta_lat
    return east_m, north_m


def _cross(
    origin: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return (
        (first[0] - origin[0]) * (second[1] - origin[1])
        - (first[1] - origin[1]) * (second[0] - origin[0])
    )


def convex_hull(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the counter-clockwise hull without repeating the first point."""

    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 1:
        return ordered

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def _empty_geometry(*, coordinate_point_count: int, hull_point_count: int) -> dict:
    return {
        "geometry_status": "insufficient_geometry",
        "coordinate_point_count": coordinate_point_count,
        "hull_point_count": hull_point_count,
        "mbr_width_m": None,
        "mbr_height_m": None,
        "mbr_orientation_deg": None,
        "mbr_area_m2": None,
    }


def _near_collinear_geometry(
    *,
    coordinate_point_count: int,
    hull_point_count: int,
) -> dict:
    geometry = _empty_geometry(
        coordinate_point_count=coordinate_point_count,
        hull_point_count=hull_point_count,
    )
    geometry["geometry_status"] = "near_collinear"
    return geometry


def minimum_area_rectangle(points: Sequence[tuple[float, float]]) -> dict:
    """Compute a rotating minimum-area rectangle around 2D points.

    Width is the longer rectangle side. Orientation is the counter-clockwise
    angle of that side from local east, normalized to [0, 180) degrees.
    """

    unique_points = sorted({(float(x), float(y)) for x, y in points})
    hull = convex_hull(unique_points)
    if len(hull) < 3:
        return _empty_geometry(
            coordinate_point_count=len(unique_points),
            hull_point_count=len(hull),
        )

    hull_array = np.asarray(hull, dtype=np.float64)
    best: tuple[float, float, float, float] | None = None
    for index, point in enumerate(hull):
        next_point = hull[(index + 1) % len(hull)]
        edge_x = next_point[0] - point[0]
        edge_y = next_point[1] - point[1]
        angle = math.atan2(edge_y, edge_x)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        rotated_x = hull_array[:, 0] * cosine + hull_array[:, 1] * sine
        rotated_y = -hull_array[:, 0] * sine + hull_array[:, 1] * cosine
        edge_width = float(np.max(rotated_x) - np.min(rotated_x))
        edge_height = float(np.max(rotated_y) - np.min(rotated_y))
        area = edge_width * edge_height

        width = edge_width
        height = edge_height
        long_side_angle = angle
        if height > width:
            width, height = height, width
            long_side_angle += math.pi / 2.0
        orientation_deg = math.degrees(long_side_angle) % 180.0
        candidate = (area, orientation_deg, width, height)
        if best is None or candidate < best:
            best = candidate

    if best is None or best[0] <= MBR_AREA_EPSILON_M2:
        return _empty_geometry(
            coordinate_point_count=len(unique_points),
            hull_point_count=len(hull),
        )

    area, orientation_deg, width, height = best
    if width <= 0.0 or height / width < MBR_MIN_ASPECT_RATIO:
        return _near_collinear_geometry(
            coordinate_point_count=len(unique_points),
            hull_point_count=len(hull),
        )
    return {
        "geometry_status": "valid",
        "coordinate_point_count": len(unique_points),
        "hull_point_count": len(hull),
        "mbr_width_m": width,
        "mbr_height_m": height,
        "mbr_orientation_deg": orientation_deg,
        "mbr_area_m2": area,
    }


def numeric_statistics(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def room_sort_key(room_id: str) -> tuple[int, str, str]:
    match = re.fullmatch(r"Room\s+(\d+)(.*)", room_id, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)), match.group(2).lower(), room_id.lower()
    return sys.maxsize, room_id.lower(), room_id.lower()


def build_room_image_density(args: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    metadata_path = resolve_project_path(args.metadata)
    pano_graph_path = resolve_project_path(args.pano_graph)
    room_graph_path = resolve_project_path(args.room_graph)
    floor = str(args.floor)

    metadata = load_json(metadata_path)
    pano_graph = load_json(pano_graph_path)
    room_graph = load_json(room_graph_path)
    rooms = valid_rooms_for_floor(room_graph, floor=floor)
    if not rooms:
        raise ValueError(f"No non-circulation rooms found for floor {floor!r}.")

    items = metadata.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected an items list in {metadata_path}.")

    room_items: dict[str, list[dict]] = defaultdict(list)
    excluded_item_count = 0
    for raw_item in items:
        if not isinstance(raw_item, dict):
            excluded_item_count += 1
            continue
        room_id = raw_item.get("room_id")
        pano_id = raw_item.get("pano_id")
        item_floor = raw_item.get("floor")
        if not isinstance(room_id, str) or room_id not in rooms:
            excluded_item_count += 1
            continue
        if item_floor is not None and str(item_floor) != floor:
            excluded_item_count += 1
            continue
        if not isinstance(pano_id, str) or not pano_id:
            raise ValueError(f"Metadata item for {room_id} has no valid pano_id.")
        pano_record = pano_graph.get(pano_id)
        coordinate = lat_lng(pano_record)
        if coordinate is None:
            raise ValueError(f"Panorama {pano_id!r} has no valid latitude/longitude.")
        if str(pano_record.get("floor")) != floor:
            raise ValueError(
                f"Panorama {pano_id!r} belongs to floor {pano_record.get('floor')!r}, "
                f"not requested floor {floor!r}."
            )
        room_items[room_id].append(raw_item)

    included_pano_ids = sorted(
        {
            str(item["pano_id"])
            for selected_items in room_items.values()
            for item in selected_items
        }
    )
    if not included_pano_ids:
        raise ValueError(f"No memory panorama items found for floor {floor!r}.")
    included_coordinates = [lat_lng(pano_graph[pano_id]) for pano_id in included_pano_ids]
    if any(coordinate is None for coordinate in included_coordinates):
        raise ValueError("Included panorama coordinates unexpectedly became invalid.")
    reference_lat = float(np.mean([coordinate[0] for coordinate in included_coordinates]))
    reference_lng = float(np.mean([coordinate[1] for coordinate in included_coordinates]))

    projected_coordinates = {
        pano_id: project_lat_lng_to_local_m(
            *lat_lng(pano_graph[pano_id]),
            reference_lat=reference_lat,
            reference_lng=reference_lng,
        )
        for pano_id in included_pano_ids
    }

    records: list[dict] = []
    for room_id in sorted(rooms, key=room_sort_key):
        selected_items = room_items.get(room_id, [])
        pano_ids = sorted({str(item["pano_id"]) for item in selected_items})
        geometry = minimum_area_rectangle(
            [projected_coordinates[pano_id] for pano_id in pano_ids]
        )
        image_count = len(selected_items)
        panorama_count = len(pano_ids)
        area = geometry["mbr_area_m2"]
        record = {
            "room_id": room_id,
            "room_title": str(rooms[room_id].get("title") or room_id),
            "floor": floor,
            "image_count": image_count,
            "panorama_count": panorama_count,
            "images_per_panorama": (
                image_count / panorama_count if panorama_count else None
            ),
            **geometry,
            "images_per_m2": image_count / area if area is not None else None,
            "panoramas_per_m2": (
                panorama_count / area if area is not None else None
            ),
        }
        records.append(record)

    valid_records = [record for record in records if record["mbr_area_m2"] is not None]
    summary = {
        "floor": floor,
        "room_count": len(records),
        "valid_mbr_room_count": len(valid_records),
        "insufficient_geometry_room_count": len(records) - len(valid_records),
        "insufficient_geometry_rooms": [
            record["room_id"]
            for record in records
            if record["geometry_status"] != "valid"
        ],
        "geometry_status_counts": {
            status: sum(record["geometry_status"] == status for record in records)
            for status in sorted({record["geometry_status"] for record in records})
        },
        "total_memory_image_count": sum(record["image_count"] for record in records),
        "total_unique_panorama_count": sum(
            record["panorama_count"] for record in records
        ),
        "excluded_metadata_item_count": excluded_item_count,
        "image_count_statistics": numeric_statistics(
            float(record["image_count"]) for record in records
        ),
        "panorama_count_statistics": numeric_statistics(
            float(record["panorama_count"]) for record in records
        ),
        "mbr_area_m2_statistics": numeric_statistics(
            float(record["mbr_area_m2"]) for record in valid_records
        ),
        "images_per_m2_statistics": numeric_statistics(
            float(record["images_per_m2"]) for record in valid_records
        ),
        "panoramas_per_m2_statistics": numeric_statistics(
            float(record["panoramas_per_m2"]) for record in valid_records
        ),
        "density_definition": {
            "images_per_m2": "image_count / minimum-area MBR area in square metres",
            "panoramas_per_m2": (
                "unique panorama_count / minimum-area MBR area in square metres"
            ),
        },
        "geometry_definition": (
            "Rotating minimum-area rectangle around the convex hull of unique "
            "memory panorama positions projected to local east/north metres."
        ),
    }
    configuration = {
        "metadata_path": str(metadata_path),
        "pano_graph_path": str(pano_graph_path),
        "room_graph_path": str(room_graph_path),
        "floor": floor,
        "projection": "local_equirectangular",
        "earth_radius_m": EARTH_RADIUS_M,
        "reference_lat": reference_lat,
        "reference_lng": reference_lng,
        "mbr_area_epsilon_m2": MBR_AREA_EPSILON_M2,
        "mbr_min_aspect_ratio": MBR_MIN_ASPECT_RATIO,
        "metadata_summary": metadata.get("summary"),
    }
    return records, summary, configuration


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_image_count_plot(output_dir: Path, records: list[dict]) -> Path:
    plt = _pyplot()
    room_ids = [record["room_id"] for record in records]
    counts = [record["image_count"] for record in records]
    figure_height = max(7.0, 0.36 * len(records))
    fig, ax = plt.subplots(figsize=(11, figure_height))
    bars = ax.barh(room_ids, counts, color="#4C78A8")
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3, fontsize=8)
    ax.set_title("Floor 0 memory image count by gallery")
    ax.set_xlabel("Memory image count")
    ax.set_ylabel("Gallery")
    ax.grid(axis="x", alpha=0.25)
    ax.margins(x=0.12)
    fig.tight_layout()
    path = output_dir / "image_count_by_room.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_image_density_plot(output_dir: Path, records: list[dict]) -> Path:
    plt = _pyplot()
    room_ids = [record["room_id"] for record in records]
    densities = [
        float(record["images_per_m2"])
        if record["images_per_m2"] is not None
        else 0.0
        for record in records
    ]
    colors = [
        "#59A14F" if record["images_per_m2"] is not None else "#BAB0AC"
        for record in records
    ]
    labels = [
        f"{record['images_per_m2']:.3f}"
        if record["images_per_m2"] is not None
        else "N/A"
        for record in records
    ]
    figure_height = max(7.0, 0.36 * len(records))
    fig, ax = plt.subplots(figsize=(11, figure_height))
    bars = ax.barh(room_ids, densities, color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_title("Floor 0 MBR-based memory image density by gallery")
    ax.set_xlabel("Memory images per square metre of panorama-position MBR")
    ax.set_ylabel("Gallery")
    ax.grid(axis="x", alpha=0.25)
    ax.margins(x=0.14)
    fig.tight_layout()
    path = output_dir / "image_density_by_room.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_outputs(
    output_dir: Path,
    records: list[dict],
    summary: dict,
    configuration: dict,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "room_image_density.json",
        "csv": output_dir / "room_image_density.csv",
        "summary": output_dir / "summary.json",
    }
    write_json(
        paths["json"],
        {
            "configuration": configuration,
            "summary": summary,
            "rooms": records,
        },
    )
    write_csv(paths["csv"], records)
    write_json(paths["summary"], summary)
    paths["image_count_plot"] = write_image_count_plot(output_dir, records)
    paths["image_density_plot"] = write_image_density_plot(output_dir, records)
    return paths


def main() -> None:
    args = build_parser().parse_args()
    records, summary, configuration = build_room_image_density(args)
    output_dir = resolve_project_path(args.output_dir)
    paths = write_outputs(output_dir, records, summary, configuration)
    print(
        json.dumps(
            {
                "room_count": summary["room_count"],
                "total_memory_image_count": summary["total_memory_image_count"],
                "total_unique_panorama_count": summary["total_unique_panorama_count"],
                "valid_mbr_room_count": summary["valid_mbr_room_count"],
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
