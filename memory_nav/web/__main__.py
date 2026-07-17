from __future__ import annotations

import argparse

from memory_nav.web import PanoExportConfig, WebConfig, create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the unified memory navigation FastAPI web tools.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--pano-export", choices=["auto", "missing", "never"], default="auto")
    parser.add_argument("--pano-output-dir", default="artifacts/pano_viewer/british_museum")
    parser.add_argument("--pano-artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--pano-graph-path")
    parser.add_argument("--pano-room-graph-path")
    parser.add_argument("--pano-grounding-path")
    parser.add_argument("--pano-dot-floor", default="0")
    parser.add_argument("--pano-dot-room-id", action="append", default=[])
    parser.add_argument("--pano-route-source-pano-id")
    parser.add_argument("--pano-route-target-pano-id")
    return parser


def config_from_args(args: argparse.Namespace) -> WebConfig:
    return WebConfig(
        host=args.host,
        port=args.port,
        pano_export=args.pano_export,
        pano_viewer=PanoExportConfig(
            artifacts_dir=args.pano_artifacts_dir,
            output_dir=args.pano_output_dir,
            pano_graph_path=args.pano_graph_path,
            room_graph_path=args.pano_room_graph_path,
            grounding_path=args.pano_grounding_path,
            dot_floor=args.pano_dot_floor,
            dot_room_id=args.pano_dot_room_id,
            route_source_pano_id=args.pano_route_source_pano_id,
            route_target_pano_id=args.pano_route_target_pano_id,
        ),
    )


def main() -> int:
    args = build_parser().parse_args()
    config = config_from_args(args)
    app = create_app(config)
    try:
        import uvicorn
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only before dependencies are installed.
        raise SystemExit("Missing uvicorn. Install requirements.txt or run `pip install uvicorn`.") from exc
    print(f"Serving memory navigation web tools at http://{config.host}:{config.port}")
    print(f"- Pano Viewer:     http://{config.host}:{config.port}/pano-viewer/")
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
