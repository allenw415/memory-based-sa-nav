from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav import load_dotenv
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

DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "dataset/sites/british_museum/normalized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts/pano_viewer/british_museum"
VIEWER_SOURCE_DIR = PROJECT_ROOT / "tools/pano_viewer/web"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export panorama graph visualization artifacts.")
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--pano-graph-path")
    parser.add_argument("--room-graph-path")
    parser.add_argument("--grounding-path")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dot-floor", default="0")
    parser.add_argument("--dot-room-id", action="append", default=[])
    parser.add_argument("--route-source-pano-id")
    parser.add_argument("--route-target-pano-id")
    parser.add_argument("--copy-viewer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-env-js", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pano_graph_path = Path(args.pano_graph_path) if args.pano_graph_path else artifacts_dir / "pano_graph.json"
    room_graph_path = Path(args.room_graph_path) if args.room_graph_path else artifacts_dir / "room_graph.json"
    grounding_path = Path(args.grounding_path) if args.grounding_path else artifacts_dir / "pano_room_grounding.json"

    pano_graph = load_json(pano_graph_path)
    room_graph = load_json(room_graph_path) if room_graph_path.exists() else {}
    grounding = load_json(grounding_path) if grounding_path.exists() else {}
    payload = build_visualization_payload(pano_graph, room_graph=room_graph, grounding_payload=grounding)

    route_pano_ids: list[str] = []
    if args.route_source_pano_id and args.route_target_pano_id:
        route_pano_ids = shortest_pano_path(payload, args.route_source_pano_id, args.route_target_pano_id)

    write_json(output_dir / "viewer_data.json", payload)
    write_json(output_dir / "pano_nodes.geojson", build_geojson(payload, feature_type="nodes"))
    write_json(output_dir / "pano_edges.geojson", build_geojson(payload, feature_type="edges"))
    write_text(output_dir / "pano_graph.gexf", build_gexf(payload))
    write_text(output_dir / "pano_graph.graphml", build_graphml(payload))
    write_text(
        output_dir / f"pano_graph_floor{safe_name(args.dot_floor)}.dot",
        build_dot(
            payload,
            floor=args.dot_floor,
            room_ids=set(args.dot_room_id),
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
            "source_pano_id": args.route_source_pano_id,
            "target_pano_id": args.route_target_pano_id,
            "pano_ids": route_pano_ids,
        },
        "files": [
            "viewer_data.json",
            "pano_nodes.geojson",
            "pano_edges.geojson",
            "pano_graph.gexf",
            "pano_graph.graphml",
            f"pano_graph_floor{safe_name(args.dot_floor)}.dot",
            "publication/",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)

    if args.copy_viewer:
        copy_viewer(output_dir)
    if args.write_env_js:
        write_env_js(output_dir)

    print(
        "[pano-viz] "
        f"nodes={payload['summary']['node_count']} edges={payload['summary']['edge_count']} "
        f"floors={len(payload['floors'])} output={output_dir}"
    )


def copy_viewer(output_dir: Path) -> None:
    for name in ("index.html", "app.js", "styles.css"):
        shutil.copy2(VIEWER_SOURCE_DIR / name, output_dir / name)


def write_env_js(output_dir: Path) -> None:
    api_key = os.environ.get("GMAPS_API_KEY", "").strip()
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


if __name__ == "__main__":
    main()
