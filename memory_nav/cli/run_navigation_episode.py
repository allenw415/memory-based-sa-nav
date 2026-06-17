from __future__ import annotations

import argparse
import sys

from ._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from memory_nav.common.env import get_env_value, load_dotenv, resolve_model_environment  # noqa: E402
from memory_nav.common.model_client import (  # noqa: E402
    DEFAULT_OPENAI_API_BASE,
    ModelResponseClient,
    resolve_api_kind,
)
from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_SIGLIP2_MODEL,
    create_image_embedder,
)
from memory_nav.memory.retrieval import MemoryImageRetriever, MemoryRoomLocalizer  # noqa: E402
from memory_nav.navigation import (  # noqa: E402
    DEFAULT_PASSAGE_QUERY,
    DynamicPassageRetriever,
    EightViewVLMDirectionSelector,
    IndexedPanoramaViewStore,
    NavigationEpisodeRunner,
    PassageVLMSelector,
    RecordedDirectionSelector,
    RecordedPassageSelector,
)


DEFAULT_SIGLIP_INDEX = "artifacts/memory_localization/floor0_siglip2_images_fov90.npz"
DEFAULT_SIGLIP_METADATA = "artifacts/memory_localization/floor0_siglip2_images_fov90.metadata.json"
DEFAULT_SALAD_INDEX = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz"
DEFAULT_SALAD_METADATA = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
DEFAULT_MANIFEST_ROOT = "renders/room_grounding_fov90"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete retrieval-driven panorama navigation episode."
    )
    parser.add_argument("--start-pano-id", required=True)
    parser.add_argument("--target-room-id", required=True)
    parser.add_argument("--waypoint-room-id", action="append", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-total-steps", type=int, default=100)
    parser.add_argument("--max-local-steps", type=int, default=20)
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)

    parser.add_argument("--siglip-index-path", default=DEFAULT_SIGLIP_INDEX)
    parser.add_argument("--siglip-metadata-path", default=DEFAULT_SIGLIP_METADATA)
    parser.add_argument("--salad-index-path", default=DEFAULT_SALAD_INDEX)
    parser.add_argument("--salad-metadata-path", default=DEFAULT_SALAD_METADATA)
    parser.add_argument(
        "--direction-salad-alpha",
        type=float,
        default=1.0,
        help="Direction score weight: alpha * SALAD + (1-alpha) * SigLIP2.",
    )
    parser.add_argument(
        "--direction-policy",
        choices=["vlm", "image_similarity"],
        default="vlm",
    )
    parser.add_argument("--direction-burst-steps", type=int, default=3)
    parser.add_argument("--direction-max-turn-deg", type=float, default=45.0)
    parser.add_argument("--direction-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--recorded-direction-responses")
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--localization-top-k", type=int, default=10)
    parser.add_argument("--localization-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--localization-margin-threshold", type=float, default=0.15)
    parser.add_argument("--passage-query", default=DEFAULT_PASSAGE_QUERY)
    parser.add_argument("--passage-top-k", type=int, default=20)
    parser.add_argument("--passage-clusters", type=int, default=8)
    parser.add_argument("--passage-confidence-threshold", type=float, default=0.5)

    parser.add_argument("--recorded-vlm-responses")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key")
    parser.add_argument("--api-base", default=DEFAULT_OPENAI_API_BASE)
    parser.add_argument(
        "--api-kind",
        default="responses",
        choices=["responses", "chat_completions"],
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--output-path")
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(
        args.artifacts_dir,
        room_graph=True,
        pano_graph=True,
        pano_room_grounding=True,
    )
    grounding = artifacts.pano_room_grounding or {}
    mappings = grounding.get("mappings")
    if not isinstance(mappings, dict):
        raise RuntimeError("pano_room_grounding.json does not contain mappings.")

    manifest_root = resolve_project_path(args.manifest_root)
    siglip_index = resolve_project_path(args.siglip_index_path)
    siglip_metadata = resolve_project_path(args.siglip_metadata_path)
    salad_index = resolve_project_path(args.salad_index_path)
    salad_metadata = resolve_project_path(args.salad_metadata_path)

    shared_embedder = create_image_embedder(
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    image_retriever = MemoryImageRetriever(
        index_path=siglip_index,
        metadata_path=siglip_metadata,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        use_faiss=False,
        project_root=resolve_project_path("."),
        render_root=manifest_root,
        embedder=shared_embedder,
    )
    localizer = MemoryRoomLocalizer(
        image_retriever,
        retrieval_top_k=args.localization_top_k,
        confidence_threshold=args.localization_confidence_threshold,
        margin_threshold=args.localization_margin_threshold,
        dedup_by_pano=True,
    )
    passage_retriever = DynamicPassageRetriever(
        semantic_index_path=siglip_index,
        semantic_metadata_path=siglip_metadata,
        visual_index_path=salad_index,
        visual_metadata_path=salad_metadata,
        render_root=manifest_root,
        query=args.passage_query,
        retrieval_top_k=args.passage_top_k,
        target_clusters=args.passage_clusters,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        text_embedder=shared_embedder,
    )
    view_store = IndexedPanoramaViewStore(
        index_path=salad_index,
        metadata_path=salad_metadata,
        manifest_root=manifest_root,
        auxiliary_index_path=(
            siglip_index
            if args.direction_policy == "image_similarity"
            and args.direction_salad_alpha < 1.0
            else None
        ),
        auxiliary_metadata_path=(
            siglip_metadata
            if args.direction_policy == "image_similarity"
            and args.direction_salad_alpha < 1.0
            else None
        ),
    )

    needs_live_client = not args.recorded_vlm_responses or (
        args.direction_policy == "vlm" and not args.recorded_direction_responses
    )
    client = None
    model_name = args.model
    if needs_live_client:
        settings = resolve_model_environment(
            default_model=args.model,
            default_api_base=args.api_base,
            default_api_kind=args.api_kind,
        )
        client = ModelResponseClient(
            provider=settings.provider,
            api_key=args.api_key or settings.api_key or get_env_value("OPENAI_API_KEY"),
            api_base=(settings.api_base or args.api_base).rstrip("/"),
            api_kind=resolve_api_kind(settings.api_kind or args.api_kind),
            request_timeout=settings.request_timeout or args.timeout,
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
        )
        model_name = settings.model_name or args.model
        if not client.is_configured():
            raise RuntimeError(
                "Missing VLM configuration. Pass recorded responses or configure an API."
            )

    if args.recorded_vlm_responses:
        selector = RecordedPassageSelector.from_path(
            resolve_project_path(args.recorded_vlm_responses)
        )
    else:
        selector = PassageVLMSelector(
            model_client=client,  # type: ignore[arg-type]
            model=model_name,
            detail=args.detail,
        )

    direction_selector = None
    if args.direction_policy == "vlm":
        if args.recorded_direction_responses:
            direction_selector = RecordedDirectionSelector.from_path(
                resolve_project_path(args.recorded_direction_responses)
            )
        else:
            direction_selector = EightViewVLMDirectionSelector(
                model_client=client,  # type: ignore[arg-type]
                model=model_name,
                detail=args.detail,
            )

    runner = NavigationEpisodeRunner(
        room_graph=artifacts.room_graph or {},
        pano_graph=artifacts.pano_graph or {},
        pano_room_mappings=mappings,
        view_store=view_store,
        localizer=localizer,
        passage_retriever=passage_retriever,
        passage_selector=selector,
        passage_confidence_threshold=args.passage_confidence_threshold,
        direction_salad_alpha=args.direction_salad_alpha,
        direction_selector=direction_selector,
        direction_confidence_threshold=args.direction_confidence_threshold,
        direction_burst_steps=args.direction_burst_steps,
        direction_max_turn_deg=args.direction_max_turn_deg,
        progress_callback=lambda message: print(message, file=sys.stderr, flush=True),
        seed=args.seed,
    )
    result = runner.run(
        start_pano_id=args.start_pano_id,
        target_room_id=args.target_room_id,
        waypoint_room_ids=args.waypoint_room_id,
        max_total_steps=args.max_total_steps,
        max_local_steps=args.max_local_steps,
    )
    payload = {
        "method": "retrieval_localize_plan_select_eight_view",
        "configuration": {
            "seed": args.seed,
            "passage_query": args.passage_query,
            "passage_top_k": args.passage_top_k,
            "passage_clusters": args.passage_clusters,
            "passage_confidence_threshold": args.passage_confidence_threshold,
            "same_pano_localization_excluded": True,
            "siglip_index_path": str(siglip_index),
            "salad_index_path": str(salad_index),
            "direction_policy": args.direction_policy,
            "direction_burst_steps": args.direction_burst_steps,
            "direction_max_turn_deg": args.direction_max_turn_deg,
            "direction_confidence_threshold": args.direction_confidence_threshold,
            "direction_scoring": (
                "eight_view_vlm"
                if args.direction_policy == "vlm"
                else "alpha * SALAD + (1-alpha) * SigLIP2"
            ),
            "direction_salad_alpha": (
                args.direction_salad_alpha
                if args.direction_policy == "image_similarity"
                else None
            ),
            "direction_siglip_alpha": (
                1.0 - args.direction_salad_alpha
                if args.direction_policy == "image_similarity"
                else None
            ),
            "manifest_root": str(manifest_root),
        },
        **result.to_dict(),
    }
    output = render_json(payload)
    write_text_if_requested(output, args.output_path)
    print(output)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
