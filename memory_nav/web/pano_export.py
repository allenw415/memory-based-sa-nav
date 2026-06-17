from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memory_nav import get_env_value, load_dotenv
from memory_nav.cli._common import PROJECT_ROOT
from memory_nav.data.pano_visualization import (
    build_dot,
    build_floor_overview_svg,
    build_geojson,
    build_gexf,
    build_graphml,
    build_visualization_payload,
    load_json,
    shortest_pano_path,
)


DEFAULT_PANO_ARTIFACTS_DIR = PROJECT_ROOT / "dataset/sites/british_museum/normalized"
DEFAULT_PANO_OUTPUT_DIR = PROJECT_ROOT / "artifacts/pano_viewer/british_museum"
PANO_VIEWER_STATIC_ROOT = Path(__file__).resolve().parent / "static/pano_viewer"


@dataclass(frozen=True)
class PanoExportConfig:
    artifacts_dir: str = str(DEFAULT_PANO_ARTIFACTS_DIR)
    output_dir: str = str(DEFAULT_PANO_OUTPUT_DIR)
    pano_graph_path: str | None = None
    room_graph_path: str | None = None
    grounding_path: str | None = None
    dot_floor: str = "0"
    dot_room_id: list[str] = field(default_factory=list)
    route_source_pano_id: str | None = None
    route_target_pano_id: str | None = None
    copy_viewer: bool = True
    write_env_js: bool = True


def export_pano_viewer(config: PanoExportConfig) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    artifacts_dir = project_path(config.artifacts_dir)
    output_dir = project_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pano_graph_path = project_path(config.pano_graph_path) if config.pano_graph_path else artifacts_dir / "pano_graph.json"
    room_graph_path = project_path(config.room_graph_path) if config.room_graph_path else artifacts_dir / "room_graph.json"
    grounding_path = project_path(config.grounding_path) if config.grounding_path else artifacts_dir / "pano_room_grounding.json"

    pano_graph = load_json(pano_graph_path)
    room_graph = load_json(room_graph_path) if room_graph_path.exists() else {}
    grounding = load_json(grounding_path) if grounding_path.exists() else {}
    payload = build_visualization_payload(pano_graph, room_graph=room_graph, grounding_payload=grounding)

    route_pano_ids: list[str] = []
    if config.route_source_pano_id and config.route_target_pano_id:
        route_pano_ids = shortest_pano_path(payload, config.route_source_pano_id, config.route_target_pano_id)

    write_json(output_dir / "viewer_data.json", payload)
    write_json(output_dir / "pano_nodes.geojson", build_geojson(payload, feature_type="nodes"))
    write_json(output_dir / "pano_edges.geojson", build_geojson(payload, feature_type="edges"))
    write_text(output_dir / "pano_graph.gexf", build_gexf(payload))
    write_text(output_dir / "pano_graph.graphml", build_graphml(payload))
    write_text(
        output_dir / f"pano_graph_floor{safe_name(config.dot_floor)}.dot",
        build_dot(
            payload,
            floor=config.dot_floor,
            room_ids=set(config.dot_room_id),
            route_pano_ids=route_pano_ids,
        ),
    )

    svg_dir = output_dir / "publication"
    svg_dir.mkdir(exist_ok=True)
    for floor in payload["floors"]:
        write_text(
            svg_dir / f"floor_{safe_name(floor)}_overview.svg",
            build_floor_overview_svg(payload, floor=floor, route_pano_ids=route_pano_ids),
        )

    manifest = {
        "schema_version": 1,
        "source": {
            "pano_graph_path": str(pano_graph_path),
            "room_graph_path": str(room_graph_path),
            "grounding_path": str(grounding_path),
        },
        "summary": payload["summary"],
        "route": {
            "source_pano_id": config.route_source_pano_id,
            "target_pano_id": config.route_target_pano_id,
            "pano_ids": route_pano_ids,
        },
        "files": [
            "viewer_data.json",
            "pano_nodes.geojson",
            "pano_edges.geojson",
            "pano_graph.gexf",
            "pano_graph.graphml",
            f"pano_graph_floor{safe_name(config.dot_floor)}.dot",
            "publication/",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)

    if config.copy_viewer:
        copy_viewer(output_dir)
    if config.write_env_js:
        write_env_js(output_dir)

    return {
        "output_dir": str(output_dir),
        "summary": payload["summary"],
        "route": manifest["route"],
        "files": manifest["files"],
    }


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def copy_viewer(output_dir: Path) -> None:
    for name in ("index.html", "trajectory.js", "app.js", "styles.css"):
        shutil.copy2(PANO_VIEWER_STATIC_ROOT / name, output_dir / name)


def write_env_js(output_dir: Path) -> None:
    api_key = get_env_value("GMAPS_KEY", "NAV_GMAPS_KEY", "GMAPS_API_KEY")
    env_path = output_dir / ".env.js"
    if not api_key:
        env_path.unlink(missing_ok=True)
        return
    env_path.write_text(
        "window.GMAPS_API_KEY = " + json.dumps(api_key, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def safe_name(value: object) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
