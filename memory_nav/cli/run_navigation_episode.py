from __future__ import annotations

import argparse
import shutil
import tempfile
import sys
from dataclasses import dataclass
from pathlib import Path

from ._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
)

ensure_project_root_on_path()

from memory_nav.common.env import (  # noqa: E402
    get_env_value,
    load_dotenv,
    resolve_model_environment,
    resolve_task_num_ctx,
)
from memory_nav.common.model_client import (  # noqa: E402
    DEFAULT_OPENAI_API_BASE,
    ModelResponseClient,
    resolve_api_kind,
)
from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_DINOV2_SALAD_MODEL,
    DEFAULT_SIGLIP2_MODEL,
    create_image_embedder,
)
from memory_nav.memory.retrieval import MemoryImageRetriever, MemoryRoomLocalizer  # noqa: E402
from memory_nav.navigation import (  # noqa: E402
    DEFAULT_PASSAGE_QUERY,
    DetectedContrastivePassageSelector,
    DynamicPassageRetriever,
    EightViewVLMDirectionSelector,
    IndexedPanoramaViewStore,
    MemoryTreeDirectionSelector,
    NavigationEpisodeRunner,
    NavigationQueryParser,
    ParsedNavigationQuery,
    PassageVLMSelector,
    RecordedDirectionSelector,
    RecordedPassageSelector,
    strict_detected_passage_configuration,
)
from memory_nav.navigation.image_goal import ImagePathSimilarityDirectionPolicy  # noqa: E402
from memory_nav.cli.run_similarity_passage_selection import (  # noqa: E402
    DreamSimImageEmbedder,
)

DEFAULT_SIGLIP_INDEX = "artifacts/memory_localization/floor0_1_siglip2_images_fov90.npz"
DEFAULT_SIGLIP_METADATA = "artifacts/memory_localization/floor0_1_siglip2_images_fov90.metadata.json"
DEFAULT_SALAD_INDEX = "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.npz"
DEFAULT_SALAD_METADATA = "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.metadata.json"
DEFAULT_MANIFEST_ROOT = "renders/room_grounding_fov90"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete retrieval-driven panorama navigation episode."
    )
    parser.add_argument("--start-pano-id", required=True)
    parser.add_argument(
        "--query",
        help="Natural-language navigation query to parse into target and waypoint rooms.",
    )
    parser.add_argument("--target-room-id")
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
        choices=["memory_tree", "vlm", "image_similarity"],
        default="memory_tree",
    )
    parser.add_argument(
        "--direction-similarity-backend",
        choices=["salad", "dreamsim"],
        default="salad",
        help="Similarity backend used when --direction-policy image_similarity.",
    )
    parser.add_argument(
        "--direction-dreamsim-type",
        default="ensemble",
        help="DreamSim checkpoint type used for image-similarity movement.",
    )
    parser.add_argument("--direction-burst-steps", type=int, default=3)
    parser.add_argument("--direction-max-turn-deg", type=float, default=45.0)
    parser.add_argument("--direction-confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--direction-commitment-mode",
        choices=["off", "visual_hysteresis"],
        default="off",
        help="Persist local direction intent across bursts without graph shortest-path planning.",
    )
    parser.add_argument("--direction-switch-margin", type=float, default=0.03)
    parser.add_argument("--direction-recovery-budget", type=int, default=1)
    parser.add_argument("--memory-tree-branching-factor", type=int, default=3)
    parser.add_argument("--memory-tree-max-depth", type=int, default=5)
    parser.add_argument(
        "--memory-tree-similarity-backend",
        choices=["dreamsim", "salad", "dinov2_patch_topk"],
        default="dinov2_patch_topk",
        help="Similarity backend used by memory-tree direction selection.",
    )
    parser.add_argument("--memory-tree-dreamsim-type", default="ensemble")
    parser.add_argument("--memory-tree-dinov2-patch-model", default="facebook/dinov2-base")
    parser.add_argument("--memory-tree-dinov2-patch-top-k", type=int, default=5)
    parser.add_argument("--memory-tree-dinov2-patch-max-patches", type=int, default=64)
    parser.add_argument(
        "--memory-tree-bridge-selection-mode",
        choices=["weighted", "bridge_then_continuity"],
        default="weighted",
    )
    parser.add_argument("--memory-tree-near-duplicate-threshold", type=float, default=0.82)
    parser.add_argument(
        "--memory-tree-bridge-similarity-tie-margin",
        type=float,
        default=0.01,
        help=(
            "When memory-tree bridge scores are within this margin, rank views by root-target "
            "similarity before continuity tie-breaks."
        ),
    )
    parser.add_argument(
        "--memory-tree-allow-same-bridge-item",
        action="store_true",
        help=(
            "Allow current and passage memory trees to bridge through the exact same memory item. "
            "Default excludes same-item bridges to avoid trivial 1.0 bridge scores."
        ),
    )
    parser.add_argument(
        "--memory-tree-patch-cache-dir",
        default="outputs/navigation_memory_tree_cache",
        help="Cache directory for DINOv2 patch features used by memory-tree direction selection.",
    )
    parser.add_argument("--recorded-direction-responses")
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--localization-top-k", type=int, default=10)
    parser.add_argument("--localization-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--localization-margin-threshold", type=float, default=0.15)
    parser.add_argument("--passage-query", default=DEFAULT_PASSAGE_QUERY)
    parser.add_argument("--passage-top-k", type=int, default=20)
    parser.add_argument("--passage-clusters", type=int, default=8)
    parser.add_argument("--passage-confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--passage-policy",
        choices=["detected_contrastive", "vlm", "recorded"],
        default="detected_contrastive",
    )

    parser.add_argument("--recorded-vlm-responses")
    parser.add_argument(
        "--provider",
        help="Shared live VLM provider, e.g. openai, gemini, or ollama.",
    )
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

    parser.add_argument(
        "--passage-profile",
        help="Environment profile for passage selection, e.g. openai or gemini.",
    )
    parser.add_argument("--passage-provider")
    parser.add_argument("--passage-model")
    parser.add_argument("--passage-api-key")
    parser.add_argument("--passage-api-base")
    parser.add_argument(
        "--passage-api-kind",
        choices=["responses", "chat_completions"],
    )
    parser.add_argument("--passage-timeout", type=float)
    parser.add_argument("--passage-detail", choices=["low", "high", "auto"])

    parser.add_argument(
        "--direction-profile",
        help="Environment profile for VLM direction selection, e.g. openai or gemini.",
    )
    parser.add_argument("--direction-provider")
    parser.add_argument("--direction-model")
    parser.add_argument("--direction-api-key")
    parser.add_argument("--direction-api-base")
    parser.add_argument(
        "--direction-api-kind",
        choices=["responses", "chat_completions"],
    )
    parser.add_argument("--direction-timeout", type=float)
    parser.add_argument("--direction-detail", choices=["low", "high", "auto"])
    parser.add_argument(
        "--output-path",
        help=(
            "Navigation output directory. If a .json path is given, a same-stem "
            "directory is created and episode.json is written inside it."
        ),
    )
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

    target_room_id, waypoint_room_ids, parsed_query = _resolve_navigation_goal(
        args,
        room_graph=artifacts.room_graph or {},
    )

    manifest_root = resolve_project_path(args.manifest_root)
    siglip_index = resolve_project_path(args.siglip_index_path)
    siglip_metadata = resolve_project_path(args.siglip_metadata_path)
    salad_index = resolve_project_path(args.salad_index_path)
    salad_metadata = resolve_project_path(args.salad_metadata_path)

    detected_passage_config = strict_detected_passage_configuration(seed=args.seed)
    effective_passage_policy = "recorded" if args.recorded_vlm_responses else args.passage_policy
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
        query=(
            detected_passage_config["combined_passage_query"]
            if effective_passage_policy == "detected_contrastive"
            else args.passage_query
        ),
        retrieval_top_k=(
            int(detected_passage_config["passage_candidate_limit"])
            if effective_passage_policy == "detected_contrastive"
            else args.passage_top_k
        ),
        target_clusters=args.passage_clusters,
        cluster_candidates=effective_passage_policy != "detected_contrastive",
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
            and args.direction_similarity_backend == "salad"
            and args.direction_salad_alpha < 1.0
            else None
        ),
        auxiliary_metadata_path=(
            siglip_metadata
            if args.direction_policy == "image_similarity"
            and args.direction_similarity_backend == "salad"
            and args.direction_salad_alpha < 1.0
            else None
        ),
    )

    passage_client = None
    if effective_passage_policy == "detected_contrastive":
        from memory_nav.cli.run_detected_passage_contrastive_selection import create_detector

        visual_retriever = MemoryImageRetriever(
            metadata_path=salad_metadata,
            project_root=PROJECT_ROOT,
            render_root=manifest_root,
            use_faiss=False,
        )
        detected_image_embedder = create_image_embedder(
            model_name=DEFAULT_DINOV2_SALAD_MODEL,
            device=args.device,
            batch_size=args.batch_size,
        )
        selector = DetectedContrastivePassageSelector(
            visual_retriever=visual_retriever,
            room_graph=artifacts.room_graph or {},
            detector=create_detector(
                backend=detected_passage_config["detector_backend"],
                model_name=detected_passage_config["detector_model"],
                device=args.device,
                box_threshold=float(detected_passage_config["box_threshold"]),
                text_threshold=float(detected_passage_config["text_threshold"]),
                cache_dir=resolve_project_path("models/huggingface"),
            ),
            image_embedder=detected_image_embedder,
            output_root=_passage_selection_output_root(args.output_path),
            pano_graph=artifacts.pano_graph or {},
            pano_room_mappings=mappings,
            seed=args.seed,
            configuration={
                **detected_passage_config,
                "siglip_index_path": str(siglip_index),
                "salad_index_path": str(salad_index),
                "salad_metadata_path": str(salad_metadata),
                "room_graph_path": str(resolve_project_path(args.artifacts_dir) / "room_graph.json"),
                "manifest_root": str(manifest_root),
            },
        )
    elif effective_passage_policy == "recorded":
        if not args.recorded_vlm_responses:
            raise RuntimeError("--passage-policy recorded requires --recorded-vlm-responses.")
        selector = RecordedPassageSelector.from_path(
            resolve_project_path(args.recorded_vlm_responses)
        )
    else:
        passage_client = _build_role_model_client(args, role="passage")
        selector = PassageVLMSelector(
            model_client=passage_client.client,
            model=passage_client.model_name,
            detail=args.passage_detail or args.detail,
        )

    image_goal_policy = None
    if (
        args.direction_policy == "image_similarity"
        and args.direction_similarity_backend == "dreamsim"
    ):
        image_goal_policy = ImagePathSimilarityDirectionPolicy(
            image_embedder=DreamSimImageEmbedder(
                dreamsim_type=args.direction_dreamsim_type,
                device=args.device,
                batch_size=args.batch_size,
            ),
            similarity_backend=f"dreamsim:{args.direction_dreamsim_type}",
        )

    direction_selector = None
    direction_client = None
    if args.direction_policy == "memory_tree":
        direction_selector = MemoryTreeDirectionSelector(
            metadata_items=view_store.metadata_items,
            image_embedder=_build_memory_tree_embedder(args),
            render_root=manifest_root,
            branching_factor=args.memory_tree_branching_factor,
            max_depth=args.memory_tree_max_depth,
            similarity_backend=args.memory_tree_similarity_backend,
            dreamsim_type=args.memory_tree_dreamsim_type,
            bridge_selection_mode=args.memory_tree_bridge_selection_mode,
            exclude_same_bridge_item=not args.memory_tree_allow_same_bridge_item,
            bridge_similarity_tie_margin=args.memory_tree_bridge_similarity_tie_margin,
            near_duplicate_threshold=args.memory_tree_near_duplicate_threshold,
            dinov2_patch_model=args.memory_tree_dinov2_patch_model,
            dinov2_patch_top_k=args.memory_tree_dinov2_patch_top_k,
            dinov2_patch_max_patches=args.memory_tree_dinov2_patch_max_patches,
            patch_cache_dir=resolve_project_path(args.memory_tree_patch_cache_dir),
            device=args.device,
            batch_size=args.batch_size,
        )
    elif args.direction_policy == "vlm":
        if args.recorded_direction_responses:
            direction_selector = RecordedDirectionSelector.from_path(
                resolve_project_path(args.recorded_direction_responses)
            )
        else:
            direction_client = _build_role_model_client(args, role="direction")
            direction_selector = EightViewVLMDirectionSelector(
                model_client=direction_client.client,
                model=direction_client.model_name,
                detail=args.direction_detail or args.detail,
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
        image_goal_policy=image_goal_policy,
        direction_selector=direction_selector,
        direction_confidence_threshold=args.direction_confidence_threshold,
        direction_burst_steps=args.direction_burst_steps,
        direction_max_turn_deg=args.direction_max_turn_deg,
        direction_commitment_mode=args.direction_commitment_mode,
        direction_switch_margin=args.direction_switch_margin,
        direction_recovery_budget=args.direction_recovery_budget,
        progress_callback=lambda message: print(message, file=sys.stderr, flush=True),
        seed=args.seed,
    )
    result = runner.run(
        start_pano_id=args.start_pano_id,
        target_room_id=target_room_id,
        waypoint_room_ids=waypoint_room_ids,
        max_total_steps=args.max_total_steps,
        max_local_steps=args.max_local_steps,
    )
    payload = {
        "method": _method_id(args, effective_passage_policy),
        "configuration": {
            "seed": args.seed,
            "navigation_query": parsed_query.to_dict() if parsed_query is not None else None,
            "target_room_id": target_room_id,
            "waypoint_room_ids": list(waypoint_room_ids),
            "passage_policy": effective_passage_policy,
            "passage_query": (
                detected_passage_config["passage_query"]
                if effective_passage_policy == "detected_contrastive"
                else args.passage_query
            ),
            "passage_queries": (
                detected_passage_config["passage_queries"]
                if effective_passage_policy == "detected_contrastive"
                else None
            ),
            "passage_query_fusion": (
                detected_passage_config["passage_query_fusion"]
                if effective_passage_policy == "detected_contrastive"
                else None
            ),
            "passage_candidate_limit": (
                detected_passage_config["passage_candidate_limit"]
                if effective_passage_policy == "detected_contrastive"
                else None
            ),
            "passage_top_k": (
                detected_passage_config["passage_candidate_limit"]
                if effective_passage_policy == "detected_contrastive"
                else args.passage_top_k
            ),
            "passage_clusters": (
                None if effective_passage_policy == "detected_contrastive" else args.passage_clusters
            ),
            "passage_clustering": effective_passage_policy != "detected_contrastive",
            "passage_confidence_threshold": args.passage_confidence_threshold,
            "same_pano_localization_excluded": True,
            "siglip_index_path": str(siglip_index),
            "salad_index_path": str(salad_index),
            "direction_policy": args.direction_policy,
            "direction_burst_steps": args.direction_burst_steps,
            "direction_max_turn_deg": args.direction_max_turn_deg,
            "direction_confidence_threshold": args.direction_confidence_threshold,
            "direction_commitment_mode": args.direction_commitment_mode,
            "direction_switch_margin": args.direction_switch_margin,
            "direction_recovery_budget": args.direction_recovery_budget,
            "memory_tree_branching_factor": (
                args.memory_tree_branching_factor
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_max_depth": (
                args.memory_tree_max_depth
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_similarity_backend": (
                args.memory_tree_similarity_backend
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_alignment_mode": (
                "bidirection" if args.direction_policy == "memory_tree" else None
            ),
            "memory_tree_bridge_score_mode": (
                "bidirection"
                if args.direction_policy == "memory_tree"
                and args.memory_tree_bridge_selection_mode == "weighted"
                else (args.memory_tree_bridge_selection_mode if args.direction_policy == "memory_tree" else None)
            ),
            "memory_tree_bridge_selection_mode": (
                args.memory_tree_bridge_selection_mode
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_near_duplicate_threshold": (
                args.memory_tree_near_duplicate_threshold
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_exclude_same_bridge_item": (
                not args.memory_tree_allow_same_bridge_item
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_bridge_similarity_tie_margin": (
                args.memory_tree_bridge_similarity_tie_margin
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_expansion_score_mode": (
                "path_continuity" if args.direction_policy == "memory_tree" else None
            ),
            "memory_tree_similarity_model": (
                _memory_tree_similarity_model_label(args)
                if args.direction_policy == "memory_tree"
                else None
            ),
            "memory_tree_dreamsim_type": (
                args.memory_tree_dreamsim_type
                if args.direction_policy == "memory_tree"
                and args.memory_tree_similarity_backend == "dreamsim"
                else None
            ),
            "memory_tree_dinov2_patch_top_k": (
                args.memory_tree_dinov2_patch_top_k
                if args.direction_policy == "memory_tree"
                and args.memory_tree_similarity_backend == "dinov2_patch_topk"
                else None
            ),
            "memory_tree_dinov2_patch_max_patches": (
                args.memory_tree_dinov2_patch_max_patches
                if args.direction_policy == "memory_tree"
                and args.memory_tree_similarity_backend == "dinov2_patch_topk"
                else None
            ),
            "memory_tree_patch_cache_dir": (
                str(resolve_project_path(args.memory_tree_patch_cache_dir))
                if args.direction_policy == "memory_tree"
                and args.memory_tree_similarity_backend == "dinov2_patch_topk"
                else None
            ),
            "detected_passage_selection": (
                _detected_passage_configuration_summary(detected_passage_config)
                if effective_passage_policy == "detected_contrastive"
                else None
            ),
            "direction_similarity_backend": (
                args.direction_similarity_backend
                if args.direction_policy == "image_similarity"
                else None
            ),
            "direction_dreamsim_type": (
                args.direction_dreamsim_type
                if args.direction_policy == "image_similarity"
                and args.direction_similarity_backend == "dreamsim"
                else None
            ),
            "direction_scoring": _direction_scoring_label(args),
            "direction_salad_alpha": (
                args.direction_salad_alpha
                if args.direction_policy == "image_similarity"
                and args.direction_similarity_backend == "salad"
                else None
            ),
            "direction_siglip_alpha": (
                1.0 - args.direction_salad_alpha
                if args.direction_policy == "image_similarity"
                and args.direction_similarity_backend == "salad"
                else None
            ),
            "passage_vlm": _model_client_configuration(
                passage_client,
                recorded=bool(args.recorded_vlm_responses),
                detail=args.passage_detail or args.detail,
            ),
            "direction_vlm": _model_client_configuration(
                direction_client,
                recorded=bool(args.recorded_direction_responses),
                detail=args.direction_detail or args.detail,
            )
            if args.direction_policy == "vlm"
            else None,
            "manifest_root": str(manifest_root),
        },
        **result.to_dict(),
    }
    output_bundle = _prepare_navigation_output_bundle(payload, args.output_path)
    if output_bundle is not None:
        payload["output_bundle"] = output_bundle
    output = render_json(payload)
    if output_bundle is not None:
        _write_navigation_output_bundle(output, output_bundle)
    print(output)
    return 0 if result.success else 1


def _resolve_navigation_goal(
    args,
    *,
    room_graph: dict,
) -> tuple[str, list[str], ParsedNavigationQuery | None]:
    query = str(args.query or "").strip()
    if query:
        if args.target_room_id or args.waypoint_room_id:
            raise RuntimeError(
                "--query cannot be combined with --target-room-id or --waypoint-room-id."
            )
        parser = _build_navigation_query_parser(args, room_graph=room_graph)
        parsed_query = parser.parse(query)
        return parsed_query.target_room_id, list(parsed_query.waypoint_room_ids), parsed_query

    target_room_id = str(args.target_room_id or "").strip()
    if not target_room_id:
        raise RuntimeError("--target-room-id is required unless --query is provided.")
    return target_room_id, list(args.waypoint_room_id or []), None


def _build_navigation_query_parser(args, *, room_graph: dict) -> NavigationQueryParser:
    if not room_graph:
        raise RuntimeError("Cannot parse navigation query without a room graph.")
    settings = resolve_model_environment(
        default_model=args.model,
        default_api_base=args.api_base,
        default_api_kind=args.api_kind,
        profile=args.provider,
    )
    provider = args.provider or settings.provider
    model_name = settings.model_name or args.model
    api_base = (settings.api_base or args.api_base).rstrip("/")
    api_kind = resolve_api_kind(settings.api_kind or args.api_kind)
    timeout = settings.request_timeout or args.timeout
    client = ModelResponseClient(
        provider=provider,
        api_key=args.api_key or settings.api_key or get_env_value("OPENAI_API_KEY"),
        api_base=api_base,
        api_kind=api_kind,
        request_timeout=timeout,
        num_ctx=resolve_task_num_ctx(
            "parse_instruction",
            fallback_num_ctx=settings.num_ctx,
        ),
        temperature=settings.temperature,
    )
    if not client.is_configured():
        raise RuntimeError(
            "Missing query parser model configuration. Pass --api-key or configure a model profile."
        )
    return NavigationQueryParser(
        model_client=client,
        model=model_name,
        room_graph=room_graph,
    )


def _method_id(args, passage_policy: str) -> str:
    if passage_policy == "detected_contrastive" and args.direction_policy == "memory_tree":
        return "retrieval_localize_plan_detected_contrastive_memory_tree_direction"
    return "retrieval_localize_plan_select_eight_view"


def _passage_selection_output_root(output_path: str | Path | None) -> Path:
    if output_path:
        return _navigation_output_dir(output_path) / "passage_selection"
    return Path(tempfile.gettempdir()) / "memory_nav_passage_selection"


def _detected_passage_configuration_summary(configuration: dict) -> dict:
    keys = [
        "passage_query",
        "passage_queries",
        "passage_query_fusion",
        "passage_candidate_limit",
        "passage_clustering",
        "detector_prompt",
        "detector_backend",
        "detector_model",
        "box_threshold",
        "text_threshold",
        "min_box_area_ratio",
        "min_box_width_ratio",
        "min_box_height_ratio",
        "min_box_aspect_ratio",
        "max_box_aspect_ratio",
        "min_box_bottom_ratio",
        "max_box_area_ratio",
        "max_crop_area_ratio",
        "crop_padding_ratio",
        "current_image_mode",
        "max_detections_per_image",
        "target_sample_count",
        "negative_sample_count",
        "similarity_top_m",
        "target_scoring",
        "contrastive_negative_weight",
        "similarity_backend",
        "visual_similarity_model",
    ]
    return {key: configuration[key] for key in keys if key in configuration}


@dataclass(frozen=True)
class _RoleModelClient:
    client: ModelResponseClient
    model_name: str
    provider: str | None
    api_base: str
    api_kind: str
    timeout: float
    profile: str | None


def _build_role_model_client(args, *, role: str) -> _RoleModelClient:
    role_profile = getattr(args, f"{role}_profile")
    role_provider = getattr(args, f"{role}_provider")
    role_model = getattr(args, f"{role}_model")
    role_api_key = getattr(args, f"{role}_api_key")
    role_api_base = getattr(args, f"{role}_api_base")
    role_api_kind = getattr(args, f"{role}_api_kind")
    role_timeout = getattr(args, f"{role}_timeout")

    profile = role_profile or role_provider or args.provider
    settings = resolve_model_environment(
        default_model=role_model or args.model,
        default_api_base=role_api_base or args.api_base,
        default_api_kind=role_api_kind or args.api_kind,
        profile=profile,
    )
    provider = role_provider or args.provider or settings.provider
    model_name = role_model or settings.model_name or args.model
    api_base = (role_api_base or settings.api_base or args.api_base).rstrip("/")
    api_kind = resolve_api_kind(role_api_kind or settings.api_kind or args.api_kind)
    timeout = role_timeout or settings.request_timeout or args.timeout
    client = ModelResponseClient(
        provider=provider,
        api_key=role_api_key
        or args.api_key
        or settings.api_key
        or get_env_value("OPENAI_API_KEY"),
        api_base=api_base,
        api_kind=api_kind,
        request_timeout=timeout,
        num_ctx=settings.num_ctx,
        temperature=settings.temperature,
    )
    if not client.is_configured():
        raise RuntimeError(
            f"Missing {role} VLM configuration. Pass recorded responses or configure an API."
        )
    return _RoleModelClient(
        client=client,
        model_name=model_name,
        provider=provider,
        api_base=api_base,
        api_kind=api_kind,
        timeout=float(timeout),
        profile=settings.active_profile,
    )


def _model_client_configuration(
    resolved: _RoleModelClient | None,
    *,
    recorded: bool,
    detail: str,
) -> dict:
    if recorded:
        return {"source": "recorded"}
    if resolved is None:
        return {"source": "disabled"}
    return {
        "source": "live",
        "profile": resolved.profile,
        "provider": resolved.provider,
        "model": resolved.model_name,
        "api_base": resolved.api_base,
        "api_kind": resolved.api_kind,
        "timeout": resolved.timeout,
        "detail": detail,
    }


def _prepare_navigation_output_bundle(
    payload: dict,
    output_path: str | Path | None,
) -> dict | None:
    if not output_path:
        return None
    output_dir = _navigation_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_passages_dir = output_dir / "selected_passages"
    selected_passages = _copy_selected_passage_images(
        payload,
        selected_passages_dir,
    )
    bundle = {
        "output_dir": str(output_dir),
        "episode_json_path": str(output_dir / "episode.json"),
        "manifest_json_path": str(output_dir / "manifest.json"),
        "selected_passages_dir": str(selected_passages_dir),
        "selected_passages": selected_passages,
    }
    return bundle


def _write_navigation_output_bundle(output: str, output_bundle: dict) -> None:
    episode_json_path = Path(output_bundle["episode_json_path"])
    manifest_json_path = Path(output_bundle["manifest_json_path"])
    episode_json_path.parent.mkdir(parents=True, exist_ok=True)
    episode_json_path.write_text(output, encoding="utf-8")
    manifest_json_path.write_text(render_json(output_bundle), encoding="utf-8")


def _navigation_output_dir(output_path: str | Path) -> Path:
    resolved = resolve_project_path(output_path)
    if resolved.suffix.lower() == ".json":
        return resolved.with_suffix("")
    return resolved


def _copy_selected_passage_images(payload: dict, output_dir: Path) -> list[dict]:
    rounds = payload.get("rounds") if isinstance(payload.get("rounds"), list) else []
    records = []
    for round_payload in rounds:
        if not isinstance(round_payload, dict):
            continue
        image_goal = round_payload.get("image_goal")
        if not isinstance(image_goal, dict):
            continue
        source_value = image_goal.get("image_path")
        if not isinstance(source_value, str) or not source_value:
            continue

        source_path = _resolve_output_source_path(source_value)
        round_index = int(round_payload.get("round_index") or len(records))
        label = str(
            image_goal.get("label")
            or round_payload.get("image_goal_label")
            or "passage"
        )
        current_room_id = _round_current_room_id(round_payload)
        subgoal_room_id = str(round_payload.get("subgoal_room_id") or "subgoal")
        destination_name = (
            f"round_{round_index:02d}_"
            f"{_safe_path_token(current_room_id)}_to_{_safe_path_token(subgoal_room_id)}_"
            f"{_safe_path_token(label)}{source_path.suffix or '.png'}"
        )
        destination_path = output_dir / destination_name
        record = {
            "round_index": round_index,
            "label": label,
            "current_room_id": current_room_id,
            "subgoal_room_id": subgoal_room_id,
            "source_image_path": str(source_path),
            "copied_image_path": str(destination_path),
            "copy_status": "copied" if source_path.exists() else "missing_source",
            "passage": _selected_passage_metadata(round_payload, label),
        }
        if source_path.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        records.append(record)
    return records


def _resolve_output_source_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return resolve_project_path(candidate)


def _round_current_room_id(round_payload: dict) -> str:
    localization = round_payload.get("localization")
    if isinstance(localization, dict) and localization.get("predicted_room_id"):
        return str(localization["predicted_room_id"])
    if round_payload.get("hidden_start_room_id"):
        return str(round_payload["hidden_start_room_id"])
    return "current"


def _selected_passage_metadata(round_payload: dict, label: str) -> dict:
    candidates = round_payload.get("current_room_passages")
    if not isinstance(candidates, list):
        return {}
    image_goal = round_payload.get("image_goal")
    source_label = image_goal.get("source_label") if isinstance(image_goal, dict) else None
    labels = [label, source_label]
    chosen = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and str(item.get("label")) in {str(value) for value in labels if value}
        ),
        None,
    )
    if not isinstance(chosen, dict):
        return {}
    keys = [
        "label",
        "room_id",
        "pano_id",
        "capture_index",
        "capture_label",
        "capture_heading",
        "memory_index",
        "semantic_score",
        "cluster_id",
        "cluster_size",
        "image_path",
        "source_image_path",
        "detection_status",
    ]
    return {key: chosen[key] for key in keys if key in chosen}


def _safe_path_token(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    ).strip("_") or "value"


def _build_memory_tree_embedder(args):
    if args.memory_tree_similarity_backend == "dinov2_patch_topk":
        return None
    if args.memory_tree_similarity_backend == "salad":
        return create_image_embedder(
            model_name=DEFAULT_DINOV2_SALAD_MODEL,
            device=args.device,
            batch_size=args.batch_size,
        )
    return DreamSimImageEmbedder(
        dreamsim_type=args.memory_tree_dreamsim_type,
        device=args.device,
        batch_size=args.batch_size,
    )


def _memory_tree_similarity_model_label(args) -> str:
    if args.memory_tree_similarity_backend == "salad":
        return DEFAULT_DINOV2_SALAD_MODEL
    if args.memory_tree_similarity_backend == "dinov2_patch_topk":
        return args.memory_tree_dinov2_patch_model
    return f"dreamsim:{args.memory_tree_dreamsim_type}"


def _direction_scoring_label(args) -> str:
    if args.direction_policy == "memory_tree":
        return f"{args.memory_tree_similarity_backend}_passage_memory_tree"
    if args.direction_policy == "vlm":
        return "eight_view_vlm"
    if args.direction_similarity_backend == "dreamsim":
        return f"dreamsim:{args.direction_dreamsim_type}"
    return "alpha * SALAD + (1-alpha) * SigLIP2"


if __name__ == "__main__":
    raise SystemExit(main())
