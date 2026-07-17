from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DENSITY_JSON = (
    "outputs/dataset_statistics/floor0_room_image_density/room_image_density.json"
)
DEFAULT_MAP_IMAGE = (
    "dataset/sites/british_museum/maps/level_0_allocentric.png"
)
DEFAULT_OUTPUT_DIR = "outputs/dataset_statistics/floor0_room_image_density"
EXPECTED_MAP_SIZE = (1244, 1260)

# Rectangles are (left, top, right, bottom) in the 1244 x 1260 allocentric map.
# They follow the gallery blocks printed on the museum map. Multiple rectangles
# are used for galleries whose footprint wraps around another room.
ROOM_MAP_REGIONS: dict[str, tuple[tuple[int, int, int, int], ...]] = {
    "Room 1": ((1088, 271, 1168, 872),),
    "Room 2": ((1088, 884, 1168, 1024), (1168, 908, 1202, 999)),
    "Room 4": ((497, 265, 578, 895),),
    "Room 6": ((427, 908, 578, 999), (497, 999, 578, 1057)),
    "Room 7": ((427, 641, 484, 895),),
    "Room 8": ((427, 526, 484, 629),),
    "Room 9": ((427, 271, 484, 513),),
    "Room 10": ((311, 641, 414, 895),),
    "Room 11": ((439, 1012, 484, 1057),),
    "Room 12": ((279, 1012, 439, 1057),),
    "Room 13": ((279, 908, 414, 999),),
    "Room 14": ((230, 861, 299, 895),),
    "Room 15": ((230, 734, 299, 850),),
    "Room 16": ((230, 641, 299, 721),),
    "Room 17": ((230, 433, 299, 629),),
    "Room 18": (
        (22, 328, 218, 421),
        (80, 421, 160, 734),
        (22, 734, 218, 850),
    ),
    "Room 18a": ((172, 559, 218, 721),),
    "Room 18b": ((172, 433, 218, 559),),
    "Room 19": ((230, 328, 299, 421),),
    "Room 20": ((230, 189, 299, 317),),
    "Room 21": ((311, 189, 414, 421),),
    "Room 22": ((311, 433, 414, 513),),
    "Room 23": ((311, 526, 414, 629),),
    "Room 24": ((751, 4, 912, 264),),
    "Room 26": ((923, 109, 1075, 179),),
    "Room 27": ((1086, 109, 1168, 179),),
    "Room 29a": ((579, 105, 662, 264),),
    "Room 29b": ((662, 105, 746, 264),),
}

# These rooms have complete-looking, room-spanning MBRs and clear rectangular
# footprints on the floor plan. The median scale is robust to one imperfect
# anchor while preserving Room 15 as the closest local reference for Room 14.
MAP_SCALE_ANCHORS = ("Room 1", "Room 4", "Room 15")
MAP_ESTIMATED_ROOMS = ("Room 14", "Room 18a")
LIGHT_BACKGROUND_ROOMS = frozenset({"Room 29a", "Room 29b"})

CSV_FIELDS = (
    "room_id",
    "room_title",
    "image_count",
    "panorama_count",
    "original_geometry_status",
    "original_mbr_area_m2",
    "map_region_pixel_area",
    "area_source",
    "effective_area_m2",
    "effective_images_per_m2",
    "estimated_area_low_m2",
    "estimated_area_high_m2",
    "estimated_images_per_m2_low",
    "estimated_images_per_m2_high",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate missing room areas from the Level 0 map and render a "
            "room-level memory-image-density choropleth."
        )
    )
    parser.add_argument("--density-json", default=DEFAULT_DENSITY_JSON)
    parser.add_argument("--map-image", default=DEFAULT_MAP_IMAGE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def rectangle_region_area(
    rectangles: Iterable[tuple[int, int, int, int]],
) -> int:
    area = 0
    for left, top, right, bottom in rectangles:
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid map rectangle: {(left, top, right, bottom)!r}")
        area += (right - left) * (bottom - top)
    return area


def room_region_pixel_area(room_id: str) -> int:
    try:
        rectangles = ROOM_MAP_REGIONS[room_id]
    except KeyError as exc:
        raise KeyError(f"No map region defined for {room_id}.") from exc
    return rectangle_region_area(rectangles)


def _records_by_room(payload: dict) -> dict[str, dict]:
    rooms = payload.get("rooms")
    if not isinstance(rooms, list):
        raise ValueError("Density JSON must contain a rooms list.")
    records: dict[str, dict] = {}
    for raw_record in rooms:
        if not isinstance(raw_record, dict):
            continue
        room_id = raw_record.get("room_id")
        if isinstance(room_id, str):
            records[room_id] = raw_record
    return records


def build_map_scale_calibration(records: dict[str, dict]) -> dict:
    anchor_records = []
    for room_id in MAP_SCALE_ANCHORS:
        record = records.get(room_id)
        if record is None:
            raise ValueError(f"Missing map-scale anchor record: {room_id}.")
        area = record.get("mbr_area_m2")
        if not isinstance(area, (int, float)) or not math.isfinite(float(area)):
            raise ValueError(f"Map-scale anchor {room_id} has no valid MBR area.")
        pixel_area = room_region_pixel_area(room_id)
        anchor_records.append(
            {
                "room_id": room_id,
                "mbr_area_m2": float(area),
                "map_region_pixel_area": pixel_area,
                "m2_per_pixel": float(area) / pixel_area,
            }
        )

    scales = [record["m2_per_pixel"] for record in anchor_records]
    selected_scale = float(statistics.median(scales))
    return {
        "method": (
            "Median MBR-area-to-map-footprint ratio from room-spanning anchor "
            "galleries Room 1, Room 4, and Room 15."
        ),
        "anchors": anchor_records,
        "selected_m2_per_pixel": selected_scale,
        "selected_m_per_pixel": math.sqrt(selected_scale),
        "anchor_m2_per_pixel_min": min(scales),
        "anchor_m2_per_pixel_max": max(scales),
    }


def build_effective_density_records(
    payload: dict,
) -> tuple[list[dict], dict]:
    records = _records_by_room(payload)
    calibration = build_map_scale_calibration(records)
    selected_scale = calibration["selected_m2_per_pixel"]
    low_scale = calibration["anchor_m2_per_pixel_min"]
    high_scale = calibration["anchor_m2_per_pixel_max"]

    output_records: list[dict] = []
    for raw_record in payload["rooms"]:
        room_id = raw_record["room_id"]
        image_count = int(raw_record["image_count"])
        original_area = raw_record.get("mbr_area_m2")
        pixel_area = (
            room_region_pixel_area(room_id)
            if room_id in ROOM_MAP_REGIONS
            else None
        )

        estimated_area_low = None
        estimated_area_high = None
        estimated_density_low = None
        estimated_density_high = None
        if isinstance(original_area, (int, float)):
            effective_area = float(original_area)
            effective_density = image_count / effective_area
            area_source = "panorama_position_mbr"
        elif room_id in MAP_ESTIMATED_ROOMS:
            if pixel_area is None:
                raise ValueError(f"No map footprint available for {room_id}.")
            effective_area = pixel_area * selected_scale
            effective_density = image_count / effective_area
            estimated_area_low = pixel_area * low_scale
            estimated_area_high = pixel_area * high_scale
            estimated_density_low = image_count / estimated_area_high
            estimated_density_high = image_count / estimated_area_low
            area_source = "map_relative_estimate"
        else:
            effective_area = None
            effective_density = None
            area_source = "unavailable"

        output_records.append(
            {
                "room_id": room_id,
                "room_title": raw_record.get("room_title"),
                "image_count": image_count,
                "panorama_count": int(raw_record["panorama_count"]),
                "original_geometry_status": raw_record.get("geometry_status"),
                "original_mbr_area_m2": original_area,
                "map_region_pixel_area": pixel_area,
                "area_source": area_source,
                "effective_area_m2": effective_area,
                "effective_images_per_m2": effective_density,
                "estimated_area_low_m2": estimated_area_low,
                "estimated_area_high_m2": estimated_area_high,
                "estimated_images_per_m2_low": estimated_density_low,
                "estimated_images_per_m2_high": estimated_density_high,
            }
        )
    return output_records, calibration


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _region_mask(
    size: tuple[int, int],
    rectangles: tuple[tuple[int, int, int, int], ...],
) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    for left, top, right, bottom in rectangles:
        mask[top:bottom, left:right] = True
    return mask


def _draw_dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    rectangle: tuple[int, int, int, int],
    *,
    color: tuple[int, int, int],
    width: int = 3,
    dash: int = 10,
    gap: int = 7,
) -> None:
    left, top, right, bottom = rectangle
    right -= 1
    bottom -= 1
    for start in range(left, right + 1, dash + gap):
        draw.line(
            [(start, top), (min(start + dash, right), top)],
            fill=color,
            width=width,
        )
        draw.line(
            [(start, bottom), (min(start + dash, right), bottom)],
            fill=color,
            width=width,
        )
    for start in range(top, bottom + 1, dash + gap):
        draw.line(
            [(left, start), (left, min(start + dash, bottom))],
            fill=color,
            width=width,
        )
        draw.line(
            [(right, start), (right, min(start + dash, bottom))],
            fill=color,
            width=width,
        )


def render_density_map(
    map_path: Path,
    output_path: Path,
    records: list[dict],
    calibration: dict,
) -> Path:
    plt = _pyplot()
    from matplotlib.colors import Normalize

    base_image = Image.open(map_path).convert("RGB")
    if base_image.size != EXPECTED_MAP_SIZE:
        raise ValueError(
            f"Expected map size {EXPECTED_MAP_SIZE}, got {base_image.size}."
        )
    base = np.asarray(base_image, dtype=np.float64)
    rendered = base.copy()
    grayscale = np.mean(base, axis=2)
    neutral = np.max(base, axis=2) - np.min(base, axis=2) <= 18

    valid_densities = [
        float(record["effective_images_per_m2"])
        for record in records
        if record["effective_images_per_m2"] is not None
    ]
    vmax = math.ceil(max(valid_densities) * 2.0) / 2.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    for record in records:
        room_id = record["room_id"]
        density = record["effective_images_per_m2"]
        rectangles = ROOM_MAP_REGIONS.get(room_id)
        if density is None or rectangles is None:
            continue
        region = _region_mask(base_image.size, rectangles)
        if room_id in LIGHT_BACKGROUND_ROOMS:
            background = neutral & (grayscale >= 145) & (grayscale <= 242)
        else:
            background = neutral & (grayscale >= 55) & (grayscale <= 190)
        mask = region & background
        color = np.asarray(cmap(norm(float(density)))[:3]) * 255.0
        rendered[mask] = 0.06 * base[mask] + 0.94 * color

    rendered_image = Image.fromarray(
        np.clip(rendered, 0, 255).astype(np.uint8),
        mode="RGB",
    )
    outline_draw = ImageDraw.Draw(rendered_image)
    for room_id in MAP_ESTIMATED_ROOMS:
        for rectangle in ROOM_MAP_REGIONS[room_id]:
            _draw_dashed_rectangle(
                outline_draw,
                rectangle,
                color=(185, 35, 100),
                width=3,
            )

    estimated = {
        record["room_id"]: record
        for record in records
        if record["area_source"] == "map_relative_estimate"
    }
    room_14 = estimated["Room 14"]
    room_18a = estimated["Room 18a"]

    fig = plt.figure(figsize=(14.5, 12.2))
    grid = fig.add_gridspec(1, 2, width_ratios=(4.6, 1.25), wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    info_ax = fig.add_subplot(grid[0, 1])
    ax.imshow(rendered_image)
    ax.set_axis_off()
    ax.set_title(
        "Floor 0 memory image density by gallery",
        fontsize=20,
        pad=12,
    )

    info_ax.set_axis_off()
    colorbar_ax = info_ax.inset_axes([0.18, 0.56, 0.20, 0.36])
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax)
    colorbar.set_label("Memory images per m²", fontsize=11)

    info_ax.text(
        0.05,
        0.96,
        "Darker colour =\nhigher density",
        transform=info_ax.transAxes,
        va="top",
        fontsize=13,
        fontweight="bold",
    )
    info_ax.text(
        0.05,
        0.49,
        "Map-relative estimates",
        transform=info_ax.transAxes,
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#A51458",
    )
    info_ax.text(
        0.05,
        0.445,
        (
            "Dashed magenta outline\n"
            f"Room 14: {room_14['image_count']} images / "
            f"{room_14['effective_area_m2']:.1f} m²\n"
            f"= {room_14['effective_images_per_m2']:.3f} images/m²\n\n"
            f"Room 18a: {room_18a['image_count']} images / "
            f"{room_18a['effective_area_m2']:.1f} m²\n"
            f"= {room_18a['effective_images_per_m2']:.3f} images/m²"
        ),
        transform=info_ax.transAxes,
        va="top",
        fontsize=11,
        linespacing=1.45,
    )
    info_ax.text(
        0.05,
        0.20,
        (
            "Area calibration\n"
            f"{calibration['selected_m_per_pixel']:.3f} m/pixel\n"
            "Median of Room 1, 4, and 15\n"
            "map-to-MBR area scales."
        ),
        transform=info_ax.transAxes,
        va="top",
        fontsize=10.5,
        linespacing=1.4,
    )
    info_ax.text(
        0.05,
        0.07,
        (
            "Caveat: existing rooms use panorama-position\n"
            "MBR density. Room 14 and Room 18a use\n"
            "approximate floor-plan-relative areas."
        ),
        transform=info_ax.transAxes,
        va="top",
        fontsize=9.5,
        color="#444444",
        linespacing=1.35,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def render_density_bar_plot(output_path: Path, records: list[dict]) -> Path:
    plt = _pyplot()
    room_ids = [record["room_id"] for record in records]
    densities = [
        float(record["effective_images_per_m2"])
        if record["effective_images_per_m2"] is not None
        else 0.0
        for record in records
    ]
    colors = [
        "#E15759"
        if record["area_source"] == "map_relative_estimate"
        else "#59A14F"
        if record["effective_images_per_m2"] is not None
        else "#BAB0AC"
        for record in records
    ]
    labels = [
        (
            f"{record['effective_images_per_m2']:.3f}*"
            if record["area_source"] == "map_relative_estimate"
            else f"{record['effective_images_per_m2']:.3f}"
        )
        if record["effective_images_per_m2"] is not None
        else "N/A"
        for record in records
    ]
    fig_height = max(7.0, 0.36 * len(records))
    fig, ax = plt.subplots(figsize=(11, fig_height))
    bars = ax.barh(room_ids, densities, color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
    ax.set_title("Floor 0 memory image density by gallery")
    ax.set_xlabel("Memory images per square metre")
    ax.set_ylabel("Gallery")
    ax.grid(axis="x", alpha=0.25)
    ax.margins(x=0.14)
    ax.text(
        0.0,
        -0.075,
        "* Map-relative area estimate; other rooms use panorama-position MBR area.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_csv(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})
    return path


def write_outputs(
    output_dir: Path,
    map_path: Path,
    records: list[dict],
    calibration: dict,
    *,
    source_density_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "room_image_density_with_map_estimates.json"
    csv_path = output_dir / "room_image_density_with_map_estimates.csv"
    map_output_path = output_dir / "image_density_choropleth_map.png"
    bar_output_path = output_dir / "image_density_by_room_with_map_estimates.png"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "configuration": {
                    "source_density_path": str(source_density_path),
                    "map_path": str(map_path),
                    "map_size_pixels": list(EXPECTED_MAP_SIZE),
                    "estimated_rooms": list(MAP_ESTIMATED_ROOMS),
                    "density_note": (
                        "Existing valid rooms retain panorama-position MBR density; "
                        "Room 14 and Room 18a use map-relative area estimates."
                    ),
                },
                "calibration": calibration,
                "rooms": records,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    write_csv(csv_path, records)
    render_density_map(map_path, map_output_path, records, calibration)
    render_density_bar_plot(bar_output_path, records)
    return {
        "json": json_path,
        "csv": csv_path,
        "choropleth_map": map_output_path,
        "bar_plot": bar_output_path,
    }


def main() -> None:
    args = build_parser().parse_args()
    density_path = resolve_project_path(args.density_json)
    map_path = resolve_project_path(args.map_image)
    output_dir = resolve_project_path(args.output_dir)

    with density_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records, calibration = build_effective_density_records(payload)
    paths = write_outputs(
        output_dir,
        map_path,
        records,
        calibration,
        source_density_path=density_path,
    )
    estimated = {
        record["room_id"]: {
            "image_count": record["image_count"],
            "estimated_area_m2": record["effective_area_m2"],
            "estimated_images_per_m2": record["effective_images_per_m2"],
            "estimated_area_range_m2": [
                record["estimated_area_low_m2"],
                record["estimated_area_high_m2"],
            ],
            "estimated_images_per_m2_range": [
                record["estimated_images_per_m2_low"],
                record["estimated_images_per_m2_high"],
            ],
        }
        for record in records
        if record["area_source"] == "map_relative_estimate"
    }
    print(
        json.dumps(
            {
                "calibration_m_per_pixel": calibration["selected_m_per_pixel"],
                "estimated_rooms": estimated,
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
