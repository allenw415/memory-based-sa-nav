from __future__ import annotations

import argparse
import os
from pathlib import Path

from ._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from memory_nav import (  # noqa: E402
    InteractiveMemoryNavigator,
    MemoryImageRetriever,
    MemoryRoomLocalizer,
    PassageAlignmentAdvisor,
    load_dotenv,
    resolve_model_environment,
)
from memory_nav.data.memory_localization import DEFAULT_SIGLIP2_MODEL  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
MODEL_ENV = resolve_model_environment(
    default_model="gpt-5-mini",
    default_api_base="https://api.openai.com/v1",
    default_api_kind="responses",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one interactive RAG-memory navigation guidance step.")
    parser.add_argument("--target-room-id", required=True)
    parser.add_argument("--waypoint-room-ids", default="")
    parser.add_argument("--localization-images", nargs="*")
    parser.add_argument(
        "--passage-images",
        nargs="*",
        help="Candidate passage images formatted as label=path, e.g. left=left.jpg front=front.jpg.",
    )
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--index-path", default="artifacts/memory_localization/floor0_siglip2_images.npz")
    parser.add_argument("--metadata-path", default="artifacts/memory_localization/floor0_siglip2_images.metadata.json")
    parser.add_argument("--faiss-path", default="artifacts/memory_localization/floor0_siglip2_images.faiss")
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--llm-model", default=MODEL_ENV.model_name)
    parser.add_argument("--llm-api-key", default=MODEL_ENV.api_key)
    parser.add_argument("--llm-api-kind", default=MODEL_ENV.api_kind)
    parser.add_argument("--llm-api-base", default=MODEL_ENV.api_base)
    parser.add_argument("--llm-timeout", type=float, default=MODEL_ENV.request_timeout or 60.0)
    parser.add_argument("--output-path")
    return parser


def parse_csv_rooms(value: str | None) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_passage_images(values: list[str] | None) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Expected passage image as label=path, got: {value}")
        label, path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Passage image label cannot be empty: {value}")
        parsed[label] = resolve_project_path(path.strip())
    return parsed


def resolve_images(values: list[str] | None) -> list[Path]:
    return [resolve_project_path(value) for value in values or []]


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(args.artifacts_dir, room_graph=True)
    room_graph = artifacts.room_graph or {}
    retriever = MemoryImageRetriever(
        index_path=resolve_project_path(args.index_path),
        metadata_path=resolve_project_path(args.metadata_path),
        faiss_path=resolve_project_path(args.faiss_path),
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        use_faiss=not args.no_faiss,
        project_root=PROJECT_ROOT,
    )
    localizer = MemoryRoomLocalizer(
        retriever,
        retrieval_top_k=args.retrieval_top_k,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    advisor = PassageAlignmentAdvisor(
        room_graph=room_graph,
        memory_retriever=retriever,
        model=args.llm_model,
        api_key=args.llm_api_key or os.environ.get("ST_NAV_API_KEY"),
        api_base=args.llm_api_base,
        api_kind=args.llm_api_kind,
        request_timeout=args.llm_timeout,
    )
    navigator = InteractiveMemoryNavigator(
        room_graph=room_graph,
        localizer=localizer,
        passage_advisor=advisor,
    )
    payload = navigator.guide(
        target_room_id=args.target_room_id,
        waypoint_room_ids=parse_csv_rooms(args.waypoint_room_ids),
        localization_images=resolve_images(args.localization_images),
        passage_images=parse_passage_images(args.passage_images),
    )
    output = render_json(payload)
    write_text_if_requested(output, args.output_path)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
