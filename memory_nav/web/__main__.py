from __future__ import annotations

import argparse

from memory_nav.web import MemoryGuidanceConfig, PanoExportConfig, WebConfig, create_app


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
    parser.add_argument("--memory-upload-dir", default="outputs/memory_guidance_web/uploads")
    parser.add_argument("--memory-artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--memory-index-path", default="artifacts/memory_localization/floor0_siglip2_images.npz")
    parser.add_argument("--memory-metadata-path", default="artifacts/memory_localization/floor0_siglip2_images.metadata.json")
    parser.add_argument("--memory-faiss-path", default="artifacts/memory_localization/floor0_siglip2_images.faiss")
    parser.add_argument("--memory-no-faiss", action="store_true")
    parser.add_argument("--embedding-model", default=MemoryGuidanceConfig.embedding_model)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-api-kind", default="responses")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    return parser


def config_from_args(args: argparse.Namespace) -> WebConfig:
    return WebConfig(
        host=args.host,
        port=args.port,
        pano_export=args.pano_export,
        memory_guidance=MemoryGuidanceConfig(
            upload_dir=args.memory_upload_dir,
            artifacts_dir=args.memory_artifacts_dir,
            index_path=args.memory_index_path,
            metadata_path=args.memory_metadata_path,
            faiss_path=args.memory_faiss_path,
            no_faiss=args.memory_no_faiss,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            retrieval_top_k=args.retrieval_top_k,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            llm_api_kind=args.llm_api_kind,
            llm_api_base=args.llm_api_base,
            llm_timeout=args.llm_timeout,
        ),
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
    print(f"- Memory Guidance: http://{config.host}:{config.port}/memory-guidance/")
    print(f"- Pano Viewer:     http://{config.host}:{config.port}/pano-viewer/")
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
