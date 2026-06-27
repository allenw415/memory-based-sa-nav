from __future__ import annotations

import argparse
import shutil
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
from memory_nav.navigation.image_goal import ImagePathSimilarityDirectionPolicy  # noqa: E402
from memory_nav.cli.run_similarity_passage_selection import (  # noqa: E402
    DreamSimImageEmbedder,
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
    if args.recorded_vlm_responses:
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
    if args.direction_policy == "vlm":
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
    chosen = next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and str(item.get("label")) == label
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
    ]
    return {key: chosen[key] for key in keys if key in chosen}


def _safe_path_token(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    ).strip("_") or "value"


def _direction_scoring_label(args) -> str:
    if args.direction_policy == "vlm":
        return "eight_view_vlm"
    if args.direction_similarity_backend == "dreamsim":
        return f"dreamsim:{args.direction_dreamsim_type}"
    return "alpha * SALAD + (1-alpha) * SigLIP2"


if __name__ == "__main__":
    raise SystemExit(main())
