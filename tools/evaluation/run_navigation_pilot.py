from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.cli import run_navigation_episode as episode_cli
from memory_nav.cli._common import load_normalized_artifacts, resolve_project_path
from memory_nav.common.env import load_dotenv
from memory_nav.data.memory_localization import (
    DEFAULT_DINOV2_SALAD_MODEL,
    create_image_embedder,
)
from memory_nav.memory.retrieval import MemoryImageRetriever, MemoryRoomLocalizer
from memory_nav.navigation import (
    DetectedContrastivePassageSelector,
    DynamicPassageRetriever,
    IndexedPanoramaViewStore,
    MemoryTreeDirectionSelector,
    NavigationEpisodeRunner,
    strict_detected_passage_configuration,
)
from memory_nav.spatial.routing import floor_pano_graph, floor_room_graph


DEFAULT_TEST_SET = (
    "outputs/navigation_paths/floor0_semantic_targets/pilot90/test_set.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/navigation_evaluation/floor0_semantic_targets/"
    "pilot90_gemma4_31b"
)
RESULT_FIELDS = [
    "test_id",
    "path_id",
    "difficulty",
    "ratio_stratum",
    "ratio_stratum_type",
    "ratio_tertile",
    "passage_profile",
    "known_failed_passage_edges_on_path",
    "query",
    "target_group_id",
    "acceptable_target_room_ids",
    "parsed_target_room_id",
    "query_grounding_correct",
    "query_parse_success",
    "episode_reported_success",
    "episode_reason",
    "success",
    "reason",
    "evaluation_outcome",
    "shortest_path_distance_m",
    "actual_path_distance_m",
    "actual_over_shortest_ratio",
    "spl",
    "shortest_pano_steps",
    "max_total_steps",
    "panorama_steps",
    "raw_room_transitions",
    "gallery_room_transitions",
    "cycle_incidence",
    "uncontrolled_cycle_incidence",
    "cycle_termination",
    "controlled_recovery_used",
    "controlled_recovery_event_count",
    "controlled_recovery_step_count",
    "terminal_room_id",
    "wrong_gallery_terminal",
    "query_parser_logical_calls",
    "query_parser_http_attempts",
    "query_parser_retries",
    "execution_time_s",
]


@dataclass
class PilotRuntime:
    runner: NavigationEpisodeRunner
    query_parser: Any
    room_graph: dict[str, dict]
    pano_graph: dict[str, dict]
    pano_room_mappings: dict[str, str | None]
    configuration: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the semantic-target Pilot90 through the end-to-end navigation pipeline."
    )
    parser.add_argument("--test-set", default=DEFAULT_TEST_SET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="gemma-4-31b-it")
    parser.add_argument("--floor", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-local-steps", type=int, default=60)
    parser.add_argument("--step-multiplier", type=float, default=3.0)
    parser.add_argument("--min-total-steps", type=int, default=50)
    parser.add_argument("--max-total-steps", type=int, default=300)
    parser.add_argument("--test-id", action="append", default=[])
    parser.add_argument("--one-per-stratum", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--quiet-navigation-progress",
        action="store_true",
        help="Suppress detailed per-step navigation messages.",
    )
    parser.add_argument(
        "--direction-commitment-mode",
        choices=["off", "visual_hysteresis"],
        default="off",
    )
    parser.add_argument("--direction-switch-margin", type=float, default=0.03)
    parser.add_argument("--direction-recovery-budget", type=int, default=1)
    return parser


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def ratio_stratum_for_case(case: dict) -> str:
    ratio_stratum = case.get("ratio_stratum")
    if ratio_stratum is not None and str(ratio_stratum).strip():
        return str(ratio_stratum)
    ratio_tertile = case.get("ratio_tertile")
    if ratio_tertile is not None and str(ratio_tertile).strip():
        return str(ratio_tertile)
    raise ValueError(
        f"Case {case.get('test_id', '<unknown>')} has neither ratio_stratum "
        "nor ratio_tertile."
    )


def ratio_stratum_type_for_case(case: dict) -> str:
    if case.get("ratio_stratum") is not None:
        return "fixed_detour_ratio"
    return "within_difficulty_tertile"




def _episode_args(args, output_dir: Path):
    return episode_cli.build_parser().parse_args(
        [
            "--start-pano-id",
            "__batch__",
            "--target-room-id",
            "Room 1",
            "--provider",
            args.provider,
            "--model",
            args.model,
            "--seed",
            str(args.seed),
            "--max-local-steps",
            str(args.max_local_steps),
            "--direction-commitment-mode",
            args.direction_commitment_mode,
            "--direction-switch-margin",
            str(args.direction_switch_margin),
            "--direction-recovery-budget",
            str(args.direction_recovery_budget),
            "--output-path",
            str(output_dir),
        ]
    )


def build_runtime(args, output_dir: Path) -> PilotRuntime:
    artifacts = load_normalized_artifacts(
        "dataset/sites/british_museum/normalized",
        room_graph=True,
        pano_graph=True,
        pano_room_grounding=True,
    )
    grounding = artifacts.pano_room_grounding or {}
    raw_mappings = grounding.get("mappings")
    if not isinstance(raw_mappings, dict):
        raise RuntimeError("pano_room_grounding.json does not contain mappings.")

    pano_graph = floor_pano_graph(artifacts.pano_graph or {}, floor=args.floor)
    mappings = {
        str(pano_id): (
            None
            if room_id is None or str(room_id).strip().lower() == "null"
            else str(room_id)
        )
        for pano_id, room_id in raw_mappings.items()
        if pano_id in pano_graph
    }
    room_graph = floor_room_graph(
        artifacts.room_graph or {},
        pano_graph=pano_graph,
        pano_room_mappings=mappings,
    )
    cli_args = _episode_args(args, output_dir)
    if cli_args.passage_policy != "detected_contrastive":
        raise ValueError("Pilot evaluator requires the default detected_contrastive passage policy.")
    if cli_args.direction_policy != "memory_tree":
        raise ValueError("Pilot evaluator requires the default memory_tree direction policy.")

    manifest_root = resolve_project_path(cli_args.manifest_root)
    siglip_index = resolve_project_path(cli_args.siglip_index_path)
    siglip_metadata = resolve_project_path(cli_args.siglip_metadata_path)
    salad_index = resolve_project_path(cli_args.salad_index_path)
    salad_metadata = resolve_project_path(cli_args.salad_metadata_path)
    detected_config = strict_detected_passage_configuration(seed=cli_args.seed)

    shared_embedder = create_image_embedder(
        model_name=cli_args.embedding_model,
        device=cli_args.device,
        batch_size=cli_args.batch_size,
    )
    image_retriever = MemoryImageRetriever(
        index_path=siglip_index,
        metadata_path=siglip_metadata,
        embedding_model=cli_args.embedding_model,
        device=cli_args.device,
        batch_size=cli_args.batch_size,
        use_faiss=False,
        project_root=resolve_project_path("."),
        render_root=manifest_root,
        embedder=shared_embedder,
    )
    localizer = MemoryRoomLocalizer(
        image_retriever,
        retrieval_top_k=cli_args.localization_top_k,
        confidence_threshold=cli_args.localization_confidence_threshold,
        margin_threshold=cli_args.localization_margin_threshold,
        dedup_by_pano=True,
    )
    passage_retriever = DynamicPassageRetriever(
        semantic_index_path=siglip_index,
        semantic_metadata_path=siglip_metadata,
        visual_index_path=salad_index,
        visual_metadata_path=salad_metadata,
        render_root=manifest_root,
        query=detected_config["combined_passage_query"],
        retrieval_top_k=int(detected_config["passage_candidate_limit"]),
        target_clusters=cli_args.passage_clusters,
        cluster_candidates=False,
        embedding_model=cli_args.embedding_model,
        device=cli_args.device,
        batch_size=cli_args.batch_size,
        text_embedder=shared_embedder,
    )
    view_store = IndexedPanoramaViewStore(
        index_path=salad_index,
        metadata_path=salad_metadata,
        manifest_root=manifest_root,
    )

    from memory_nav.cli.run_detected_passage_contrastive_selection import create_detector

    visual_retriever = MemoryImageRetriever(
        metadata_path=salad_metadata,
        project_root=PROJECT_ROOT,
        render_root=manifest_root,
        use_faiss=False,
    )
    detected_image_embedder = create_image_embedder(
        model_name=DEFAULT_DINOV2_SALAD_MODEL,
        device=cli_args.device,
        batch_size=cli_args.batch_size,
    )
    selector = DetectedContrastivePassageSelector(
        visual_retriever=visual_retriever,
        room_graph=room_graph,
        detector=create_detector(
            backend=detected_config["detector_backend"],
            model_name=detected_config["detector_model"],
            device=cli_args.device,
            box_threshold=float(detected_config["box_threshold"]),
            text_threshold=float(detected_config["text_threshold"]),
            cache_dir=resolve_project_path("models/huggingface"),
        ),
        image_embedder=detected_image_embedder,
        output_root=output_dir / "passage_selection",
        pano_graph=pano_graph,
        pano_room_mappings=mappings,
        seed=cli_args.seed,
        configuration={
            **detected_config,
            "siglip_index_path": str(siglip_index),
            "salad_index_path": str(salad_index),
            "salad_metadata_path": str(salad_metadata),
            "room_graph_path": str(
                resolve_project_path(cli_args.artifacts_dir) / "room_graph.json"
            ),
            "manifest_root": str(manifest_root),
        },
    )
    from tools.experiments.build_passage_memory_tree import (
        load_dinov2_patch_cache,
        safe_name,
    )

    patch_cache_path = (
        resolve_project_path(cli_args.memory_tree_patch_cache_dir)
        / "embedding_cache"
        / (
            "dinov2_patch_topk_"
            f"{safe_name(cli_args.memory_tree_dinov2_patch_model)}_"
            f"p{cli_args.memory_tree_dinov2_patch_max_patches}.npz"
        )
    )
    preloaded_patch_records = load_dinov2_patch_cache(
        patch_cache_path,
        model_name=cli_args.memory_tree_dinov2_patch_model,
        max_patches=cli_args.memory_tree_dinov2_patch_max_patches,
    )
    if not preloaded_patch_records:
        raise RuntimeError(
            f"DINOv2 patch cache is missing or invalid: {patch_cache_path}"
        )

    direction_selector = MemoryTreeDirectionSelector(
        metadata_items=view_store.metadata_items,
        image_embedder=episode_cli._build_memory_tree_embedder(cli_args),
        render_root=manifest_root,
        branching_factor=cli_args.memory_tree_branching_factor,
        max_depth=cli_args.memory_tree_max_depth,
        similarity_backend=cli_args.memory_tree_similarity_backend,
        dreamsim_type=cli_args.memory_tree_dreamsim_type,
        bridge_selection_mode=cli_args.memory_tree_bridge_selection_mode,
        exclude_same_bridge_item=not cli_args.memory_tree_allow_same_bridge_item,
        bridge_similarity_tie_margin=cli_args.memory_tree_bridge_similarity_tie_margin,
        near_duplicate_threshold=cli_args.memory_tree_near_duplicate_threshold,
        dinov2_patch_model=cli_args.memory_tree_dinov2_patch_model,
        dinov2_patch_top_k=cli_args.memory_tree_dinov2_patch_top_k,
        dinov2_patch_max_patches=cli_args.memory_tree_dinov2_patch_max_patches,
        patch_cache_dir=resolve_project_path(cli_args.memory_tree_patch_cache_dir),
        device=cli_args.device,
        batch_size=cli_args.batch_size,
    )

    progress = None
    if not args.quiet_navigation_progress:
        progress = lambda message: print(message, file=sys.stderr, flush=True)
    runner = NavigationEpisodeRunner(
        room_graph=room_graph,
        pano_graph=pano_graph,
        pano_room_mappings=mappings,
        view_store=view_store,
        localizer=localizer,
        passage_retriever=passage_retriever,
        passage_selector=selector,
        passage_confidence_threshold=cli_args.passage_confidence_threshold,
        direction_salad_alpha=cli_args.direction_salad_alpha,
        direction_selector=direction_selector,
        direction_confidence_threshold=cli_args.direction_confidence_threshold,
        direction_burst_steps=cli_args.direction_burst_steps,
        direction_max_turn_deg=cli_args.direction_max_turn_deg,
        direction_commitment_mode=cli_args.direction_commitment_mode,
        direction_switch_margin=cli_args.direction_switch_margin,
        direction_recovery_budget=cli_args.direction_recovery_budget,
        progress_callback=progress,
        seed=cli_args.seed,
    )
    query_parser = episode_cli._build_navigation_query_parser(
        cli_args,
        room_graph=room_graph,
    )
    query_client = query_parser.model_client
    configuration = {
        "evaluation_scope": "floor0_semantic_targets_pilot90",
        "floor": args.floor,
        "seed": args.seed,
        "provider": query_client.provider,
        "query_parser_model": query_parser.model,
        "query_parser_api_base": query_client.api_base,
        "query_parser_temperature": 0,
        "query_parser_thinking_level": "minimal",
        "query_parser_timeout_s": query_client.request_timeout,
        "query_parser_only_vlm_use": True,
        "passage_policy": cli_args.passage_policy,
        "direction_policy": cli_args.direction_policy,
        "localization_top_k": cli_args.localization_top_k,
        "localization_confidence_threshold": cli_args.localization_confidence_threshold,
        "localization_margin_threshold": cli_args.localization_margin_threshold,
        "passage_candidate_limit": detected_config["passage_candidate_limit"],
        "grounding_dino_model": detected_config["detector_model"],
        "grounding_dino_box_threshold": detected_config["box_threshold"],
        "grounding_dino_text_threshold": detected_config["text_threshold"],
        "grounding_dino_max_detections_per_image": detected_config[
            "max_detections_per_image"
        ],
        "passage_similarity_top_m": detected_config["similarity_top_m"],
        "passage_negative_weight": detected_config["contrastive_negative_weight"],
        "memory_tree_branching_factor": cli_args.memory_tree_branching_factor,
        "memory_tree_max_depth": cli_args.memory_tree_max_depth,
        "memory_tree_similarity_backend": cli_args.memory_tree_similarity_backend,
        "memory_tree_dinov2_patch_model": cli_args.memory_tree_dinov2_patch_model,
        "memory_tree_dinov2_patch_top_k": cli_args.memory_tree_dinov2_patch_top_k,
        "memory_tree_dinov2_patch_max_patches": cli_args.memory_tree_dinov2_patch_max_patches,
        "memory_tree_preloaded_patch_record_count": len(preloaded_patch_records),
        "direction_burst_steps": cli_args.direction_burst_steps,
        "direction_max_turn_deg": cli_args.direction_max_turn_deg,
        "direction_confidence_threshold": cli_args.direction_confidence_threshold,
        "direction_commitment_mode": cli_args.direction_commitment_mode,
        "direction_switch_margin": cli_args.direction_switch_margin,
        "direction_recovery_budget": cli_args.direction_recovery_budget,
        "max_local_steps": args.max_local_steps,
        "max_total_steps_formula": (
            f"min({args.max_total_steps}, max({args.min_total_steps}, "
            f"{args.step_multiplier:g} * shortest_pano_steps))"
        ),
        "step_multiplier": args.step_multiplier,
        "min_total_steps": args.min_total_steps,
        "max_total_steps": args.max_total_steps,
        "timing_scope": (
            "query parsing through navigation completion; shared model/component "
            "initialization excluded"
        ),
        "floor_pano_count": len(pano_graph),
        "floor_mapped_room_count": len(room_graph),
    }
    return PilotRuntime(
        runner=runner,
        query_parser=query_parser,
        room_graph=room_graph,
        pano_graph=pano_graph,
        pano_room_mappings=mappings,
        configuration=configuration,
    )


def max_total_steps_for_case(test_case: dict, args) -> int:
    shortest_steps = int(test_case["pano_step_count"])
    scaled = int(math.ceil(float(args.step_multiplier) * shortest_steps))
    return min(args.max_total_steps, max(args.min_total_steps, scaled))


def _counter_snapshot(client) -> tuple[int, int, int]:
    return (
        int(client.logical_request_count),
        int(client.http_attempt_count),
        int(client.retry_count),
    )


def _counter_delta(before: Sequence[int], after: Sequence[int]) -> tuple[int, int, int]:
    return tuple(int(end) - int(start) for start, end in zip(before, after))


def _collapsed_gallery_sequence(
    room_sequence: Iterable[str],
    room_graph: dict[str, dict],
) -> list[str]:
    result: list[str] = []
    for room_id in room_sequence:
        record = room_graph.get(room_id, {})
        if record.get("category") == "Circulation":
            continue
        if not result or result[-1] != room_id:
            result.append(room_id)
    return result


def _direction_recovery_metrics(episode_result) -> dict[str, Any]:
    if episode_result is None:
        return {
            "controlled_recovery_used": False,
            "controlled_recovery_event_count": 0,
            "controlled_recovery_step_count": 0,
            "uncontrolled_repeated_pano": False,
        }
    rounds = getattr(episode_result, "rounds", [])
    if not isinstance(rounds, list):
        rounds = []
    actions: list[dict] = []
    event_count = 0
    for round_payload in rounds:
        if not isinstance(round_payload, dict):
            continue
        round_actions = round_payload.get("movement_steps")
        if isinstance(round_actions, list):
            actions.extend(item for item in round_actions if isinstance(item, dict))
        commitment = round_payload.get("direction_commitment")
        history = commitment.get("recovery_history") if isinstance(commitment, dict) else None
        if isinstance(history, list):
            event_count += len(history)
    actions.sort(key=lambda item: int(item.get("step_index", 0)))
    recovery_actions = [
        item for item in actions if item.get("decision_source") == "recovery_backtrack"
    ]
    if event_count == 0 and recovery_actions:
        event_count = len(
            {
                (int(item.get("round_index", -1)), int(item.get("recovery_event_index", -1)))
                for item in recovery_actions
            }
        )

    pano_path = list(getattr(episode_result, "pano_path", []) or [])
    repeated_pano = len(set(pano_path)) < len(pano_path)
    if actions and pano_path:
        seen = {str(pano_path[0])}
        uncontrolled_repeated_pano = False
        for action in actions:
            next_pano_id = action.get("next_pano_id")
            if not isinstance(next_pano_id, str):
                continue
            if (
                next_pano_id in seen
                and action.get("decision_source") != "recovery_backtrack"
            ):
                uncontrolled_repeated_pano = True
                break
            seen.add(next_pano_id)
    else:
        uncontrolled_repeated_pano = repeated_pano
    return {
        "controlled_recovery_used": bool(event_count or recovery_actions),
        "controlled_recovery_event_count": int(event_count),
        "controlled_recovery_step_count": len(recovery_actions),
        "uncontrolled_repeated_pano": bool(uncontrolled_repeated_pano),
    }


def build_result_record(
    *,
    test_case: dict,
    parsed_query,
    parse_error: str | None,
    episode_result,
    episode_error: str | None,
    elapsed_s: float,
    query_counts: Sequence[int],
    max_total_steps: int,
    room_graph: dict[str, dict],
    pano_room_mappings: dict[str, str | None],
) -> dict:
    acceptable = [str(item) for item in test_case["acceptable_target_room_ids"]]
    parsed_target = parsed_query.target_room_id if parsed_query is not None else None
    query_parse_success = parsed_query is not None
    query_grounding_correct = bool(parsed_target in acceptable)
    if episode_result is None:
        episode_reported_success = False
        reason = "query_parse_failed" if parse_error else "episode_exception"
        final_pano_id = test_case["start_pano_id"]
        pano_path = [final_pano_id]
        metrics = {
            "pano_path_distance_m": 0.0,
            "pano_step_count": 0,
            "room_sequence": [],
            "room_transition_count": 0,
        }
    else:
        episode_reported_success = bool(episode_result.success)
        reason = str(episode_result.reason)
        final_pano_id = episode_result.final_pano_id
        pano_path = list(episode_result.pano_path)
        metrics = dict(episode_result.navigation_metrics)

    shortest_distance = float(test_case["path_distance_m"])
    actual_distance = float(metrics.get("pano_path_distance_m") or 0.0)
    actual_over_shortest = (
        actual_distance / shortest_distance if shortest_distance > 0.0 else None
    )
    room_sequence = [
        str(item) for item in metrics.get("room_sequence", []) if isinstance(item, str)
    ]
    gallery_sequence = _collapsed_gallery_sequence(room_sequence, room_graph)
    terminal_room_id = pano_room_mappings.get(final_pano_id)
    if terminal_room_id is not None and str(terminal_room_id).strip().lower() == "null":
        terminal_room_id = None
    terminal_record = room_graph.get(terminal_room_id, {}) if terminal_room_id else {}
    terminal_is_gallery = bool(
        terminal_room_id
        and terminal_room_id in room_graph
        and terminal_record.get("category") != "Circulation"
    )
    success = bool(terminal_room_id in acceptable)
    if success:
        evaluation_outcome = "ground_truth_target_reached"
    elif episode_reported_success:
        evaluation_outcome = "false_positive_target_relocalization"
    else:
        evaluation_outcome = reason
    spl = (
        shortest_distance / max(actual_distance, shortest_distance)
        if success and shortest_distance > 0.0
        else 0.0
    )
    repeated_pano = len(set(pano_path)) < len(pano_path)
    cycle_termination = reason == "cycle_detected"
    cycle_incidence = bool(repeated_pano or cycle_termination)
    recovery_metrics = _direction_recovery_metrics(episode_result)
    uncontrolled_cycle_incidence = bool(
        recovery_metrics["uncontrolled_repeated_pano"] or cycle_termination
    )
    wrong_gallery_terminal = bool(
        episode_result is not None
        and not success
        and terminal_is_gallery
        and terminal_room_id not in acceptable
    )
    logical_calls, http_attempts, retries = [int(value) for value in query_counts]
    error = parse_error or episode_error
    return {
        "test_id": test_case["test_id"],
        "path_id": test_case["path_id"],
        "difficulty": test_case["difficulty"],
        "ratio_stratum": ratio_stratum_for_case(test_case),
        "ratio_stratum_type": ratio_stratum_type_for_case(test_case),
        "ratio_tertile": test_case.get("ratio_tertile"),
        "passage_profile": test_case.get("passage_profile"),
        "known_failed_passage_edges_on_path": list(
            test_case.get("known_failed_passage_edges_on_path", [])
        ),
        "query": test_case["query"],
        "target_group_id": test_case["target_group_id"],
        "target_group_theme": test_case["target_group_theme"],
        "acceptable_target_room_ids": acceptable,
        "reference_end_room_id": test_case["reference_end_room_id"],
        "start_pano_id": test_case["start_pano_id"],
        "start_room_id": test_case["start_room_id"],
        "parsed_target_room_id": parsed_target,
        "query_grounding_correct": query_grounding_correct,
        "query_parse_success": query_parse_success,
        "episode_reported_success": episode_reported_success,
        "episode_reason": reason,
        "success": success,
        "reason": reason,
        "evaluation_outcome": evaluation_outcome,
        "error": error,
        "shortest_path_distance_m": shortest_distance,
        "actual_path_distance_m": actual_distance,
        "actual_over_shortest_ratio": actual_over_shortest,
        "spl": spl,
        "shortest_pano_steps": int(test_case["pano_step_count"]),
        "max_total_steps": int(max_total_steps),
        "panorama_steps": int(metrics.get("pano_step_count") or 0),
        "raw_room_sequence": room_sequence,
        "gallery_room_sequence": gallery_sequence,
        "raw_room_transitions": int(metrics.get("room_transition_count") or 0),
        "gallery_room_transitions": max(len(gallery_sequence) - 1, 0),
        "cycle_incidence": cycle_incidence,
        "uncontrolled_cycle_incidence": uncontrolled_cycle_incidence,
        "cycle_termination": cycle_termination,
        "controlled_recovery_used": recovery_metrics["controlled_recovery_used"],
        "controlled_recovery_event_count": recovery_metrics[
            "controlled_recovery_event_count"
        ],
        "controlled_recovery_step_count": recovery_metrics[
            "controlled_recovery_step_count"
        ],
        "terminal_pano_id": final_pano_id,
        "terminal_room_id": terminal_room_id,
        "wrong_gallery_terminal": wrong_gallery_terminal,
        "query_parser_logical_calls": logical_calls,
        "query_parser_http_attempts": http_attempts,
        "query_parser_retries": retries,
        "execution_time_s": float(elapsed_s),
    }


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return statistics.fmean(data) if data else None


def _median(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values]
    return statistics.median(data) if data else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    data = sorted(float(value) for value in values)
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    weight = position - lower
    return data[lower] * (1.0 - weight) + data[upper] * weight


def aggregate_records(records: Sequence[dict]) -> dict:
    count = len(records)
    success_records = [record for record in records if record["success"]]
    failure_records = [record for record in records if not record["success"]]
    success_count = len(success_records)
    cycles = sum(bool(record["cycle_incidence"]) for record in records)
    uncontrolled_cycles = sum(
        bool(record.get("uncontrolled_cycle_incidence", record["cycle_incidence"]))
        for record in records
    )
    cycle_terminations = sum(bool(record["cycle_termination"]) for record in records)
    controlled_recoveries = sum(
        bool(record.get("controlled_recovery_used", False)) for record in records
    )
    wrong = sum(bool(record["wrong_gallery_terminal"]) for record in records)
    unmapped = sum(record["terminal_room_id"] is None for record in records)
    reported_success = sum(
        bool(record.get("episode_reported_success", record["success"]))
        for record in records
    )
    false_positive_success = sum(
        bool(record.get("episode_reported_success", record["success"]))
        and not bool(record["success"])
        for record in records
    )
    false_negative_success = sum(
        not bool(record.get("episode_reported_success", record["success"]))
        and bool(record["success"])
        for record in records
    )
    parse_failures = sum(not bool(record["query_parse_success"]) for record in records)
    grounding_correct = sum(bool(record["query_grounding_correct"]) for record in records)

    def rate(value: int) -> float:
        return float(value) / count if count else 0.0

    return {
        "count": count,
        "success_count": success_count,
        "success_rate": rate(success_count),
        "spl_mean": _mean(record["spl"] for record in records),
        "actual_over_shortest_success_mean": _mean(
            record["actual_over_shortest_ratio"]
            for record in success_records
            if record["actual_over_shortest_ratio"] is not None
        ),
        "actual_over_shortest_success_median": _median(
            record["actual_over_shortest_ratio"]
            for record in success_records
            if record["actual_over_shortest_ratio"] is not None
        ),
        "actual_over_shortest_all_mean": _mean(
            record["actual_over_shortest_ratio"]
            for record in records
            if record["actual_over_shortest_ratio"] is not None
        ),
        "actual_over_shortest_failure_mean": _mean(
            record["actual_over_shortest_ratio"]
            for record in failure_records
            if record["actual_over_shortest_ratio"] is not None
        ),
        "panorama_steps_mean": _mean(record["panorama_steps"] for record in records),
        "panorama_steps_median": _median(record["panorama_steps"] for record in records),
        "raw_room_transitions_mean": _mean(
            record["raw_room_transitions"] for record in records
        ),
        "gallery_room_transitions_mean": _mean(
            record["gallery_room_transitions"] for record in records
        ),
        "cycle_incidence_count": cycles,
        "cycle_incidence_rate": rate(cycles),
        "uncontrolled_cycle_incidence_count": uncontrolled_cycles,
        "uncontrolled_cycle_incidence_rate": rate(uncontrolled_cycles),
        "cycle_termination_count": cycle_terminations,
        "cycle_termination_rate": rate(cycle_terminations),
        "controlled_recovery_count": controlled_recoveries,
        "controlled_recovery_rate": rate(controlled_recoveries),
        "controlled_recovery_events_mean": _mean(
            record.get("controlled_recovery_event_count", 0) for record in records
        ),
        "controlled_recovery_steps_mean": _mean(
            record.get("controlled_recovery_step_count", 0) for record in records
        ),
        "wrong_gallery_terminal_count": wrong,
        "wrong_gallery_terminal_rate": rate(wrong),
        "wrong_gallery_terminal_given_failure_rate": (
            float(wrong) / len(failure_records) if failure_records else 0.0
        ),
        "unmapped_terminal_count": unmapped,
        "unmapped_terminal_rate": rate(unmapped),
        "episode_reported_success_count": reported_success,
        "false_positive_success_count": false_positive_success,
        "false_negative_success_count": false_negative_success,
        "query_parse_failure_count": parse_failures,
        "query_parse_failure_rate": rate(parse_failures),
        "query_grounding_accuracy": rate(grounding_correct),
        "query_parser_logical_calls_mean": _mean(
            record["query_parser_logical_calls"] for record in records
        ),
        "query_parser_http_attempts_mean": _mean(
            record["query_parser_http_attempts"] for record in records
        ),
        "query_parser_retries_mean": _mean(
            record["query_parser_retries"] for record in records
        ),
        "execution_time_s_mean": _mean(
            record["execution_time_s"] for record in records
        ),
        "execution_time_s_median": _median(
            record["execution_time_s"] for record in records
        ),
        "execution_time_s_p90": _percentile(
            (record["execution_time_s"] for record in records), 0.90
        ),
        "failure_reasons": dict(
            sorted(
                Counter(
                    record.get("evaluation_outcome", record["reason"])
                    for record in failure_records
                ).items()
            )
        ),
        "terminal_room_counts": dict(
            sorted(
                Counter(
                    record["terminal_room_id"] or "unmapped"
                    for record in records
                ).items()
            )
        ),
    }


def summarize(records: Sequence[dict], configuration: dict) -> dict:
    by_difficulty: dict[str, dict] = {}
    by_ratio: dict[str, dict] = {}
    by_stratum: dict[str, dict] = {}
    by_passage_profile: dict[str, dict] = {}
    for difficulty in ("easy", "medium", "hard"):
        group = [record for record in records if record["difficulty"] == difficulty]
        by_difficulty[difficulty] = aggregate_records(group)

    observed_ratio_labels = {
        ratio_stratum_for_case(record) for record in records
    }
    preferred_labels = (
        "1.0-1.5",
        "1.5-2.0",
        "2.0-2.5",
        "low",
        "middle",
        "high",
    )
    ratio_labels = [
        label for label in preferred_labels if label in observed_ratio_labels
    ]
    ratio_labels.extend(sorted(observed_ratio_labels.difference(ratio_labels)))
    for ratio_label in ratio_labels:
        group = [
            record
            for record in records
            if ratio_stratum_for_case(record) == ratio_label
        ]
        by_ratio[ratio_label] = aggregate_records(group)
    for difficulty in ("easy", "medium", "hard"):
        for ratio_label in ratio_labels:
            key = f"{difficulty}/{ratio_label}"
            group = [
                record
                for record in records
                if record["difficulty"] == difficulty
                and ratio_stratum_for_case(record) == ratio_label
            ]
            by_stratum[key] = aggregate_records(group)
    for profile in ("reliable", "risk"):
        group = [
            record for record in records if record.get("passage_profile") == profile
        ]
        if group:
            by_passage_profile[profile] = aggregate_records(group)
    return {
        "configuration": configuration,
        "metric_definitions": {
            "success": (
                "Ground-truth room mapped from the final panorama is any "
                "acceptable_target_room_id."
            ),
            "episode_reported_success": (
                "Whether the navigation pipeline stopped after its localizer reported "
                "an acceptable target room."
            ),
            "spl": "success * shortest_distance / max(actual_distance, shortest_distance)",
            "actual_over_shortest_ratio": "Actual pano-path distance / reference shortest-path distance.",
            "cycle_incidence": (
                "Raw repeated pano in trajectory or cycle_detected termination; "
                "controlled recovery repeats remain included."
            ),
            "uncontrolled_cycle_incidence": (
                "Repeated pano outside an explicitly tagged recovery_backtrack action, "
                "or cycle_detected termination."
            ),
            "controlled_recovery": (
                "A bounded retrace to the nearest recorded branch with an untried safe edge."
            ),
            "wrong_gallery_terminal": (
                "Ground-truth failure whose final mapped non-circulation gallery is "
                "outside the acceptable target group; unmapped terminals are excluded."
            ),
            "execution_time": (
                "Per episode from query parser start through navigation completion; "
                "shared initialization excluded."
            ),
        },
        "overall": aggregate_records(records),
        "by_difficulty": by_difficulty,
        "by_ratio_stratum": by_ratio,
        "by_stratum": by_stratum,
        "by_passage_profile": by_passage_profile,
        **(
            {"by_ratio_tertile": by_ratio}
            if all(record.get("ratio_tertile") is not None for record in records)
            else {}
        ),
    }


def _json_value_for_csv(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_results_csv(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: _json_value_for_csv(record.get(field))
                    for field in RESULT_FIELDS
                }
            )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_aggregate_outputs(
    *,
    output_dir: Path,
    records: Sequence[dict],
    configuration: dict,
) -> None:
    ordered = sorted(records, key=lambda item: item["test_id"])
    write_results_csv(output_dir / "results.csv", ordered)
    atomic_write_json(output_dir / "summary.json", summarize(ordered, configuration))


def select_tests(test_cases: Sequence[dict], args) -> list[dict]:
    selected = list(test_cases)
    if args.test_id:
        wanted = set(args.test_id)
        selected = [case for case in selected if case["test_id"] in wanted]
        missing = wanted.difference(case["test_id"] for case in selected)
        if missing:
            raise ValueError(f"Unknown test ids: {', '.join(sorted(missing))}")
    if args.one_per_stratum:
        one_each: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for case in selected:
            key = (str(case["difficulty"]), ratio_stratum_for_case(case))
            if key in seen:
                continue
            seen.add(key)
            one_each.append(case)
        selected = one_each
    if args.limit is not None:
        selected = selected[: max(args.limit, 0)]
    return selected


def load_completed_records(path: Path) -> list[dict]:
    return load_jsonl(path) if path.exists() else []


def main() -> int:
    args = build_parser().parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    test_set_path = resolve_project_path(args.test_set)
    output_dir = resolve_project_path(args.output_dir)
    results_path = output_dir / "results.jsonl"

    test_cases = load_jsonl(test_set_path)
    selected_tests = select_tests(test_cases, args)
    if not selected_tests:
        raise RuntimeError("No Pilot90 cases were selected.")

    if results_path.exists() and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"{results_path} already exists; pass --resume or --overwrite."
        )
    if args.overwrite and results_path.exists():
        results_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    completed_records = load_completed_records(results_path) if args.resume else []
    completed_ids = {record["test_id"] for record in completed_records}
    remaining = [case for case in selected_tests if case["test_id"] not in completed_ids]
    print(
        f"[pilot] selected={len(selected_tests)} completed={len(completed_ids)} "
        f"remaining={len(remaining)}",
        file=sys.stderr,
        flush=True,
    )
    if not remaining:
        configuration_path = output_dir / "configuration.json"
        configuration = (
            json.loads(configuration_path.read_text(encoding="utf-8"))
            if configuration_path.exists()
            else {}
        )
        write_aggregate_outputs(
            output_dir=output_dir,
            records=completed_records,
            configuration=configuration,
        )
        return 0

    initialization_start = time.perf_counter()
    runtime = build_runtime(args, output_dir)
    initialization_s = time.perf_counter() - initialization_start
    runtime.configuration["shared_initialization_time_s"] = initialization_s
    runtime.configuration["test_set_path"] = str(test_set_path)
    stratum_types = sorted(
        {ratio_stratum_type_for_case(case) for case in selected_tests}
    )
    runtime.configuration["ratio_stratum_types"] = stratum_types
    passage_controlled = any(
        case.get("passage_profile") in {"reliable", "risk"}
        for case in selected_tests
    )
    runtime.configuration["passage_controlled_test_set"] = passage_controlled
    if passage_controlled:
        runtime.configuration["evaluation_scope"] = (
            "floor0_semantic_targets_pilot90_fixed_ratio_passage_controlled"
        )
    atomic_write_json(output_dir / "configuration.json", runtime.configuration)
    print(
        f"[pilot] runtime ready in {initialization_s:.1f}s; "
        f"model={runtime.query_parser.model}",
        file=sys.stderr,
        flush=True,
    )

    records = list(completed_records)
    episode_dir = output_dir / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)
    query_client = runtime.query_parser.model_client
    with results_path.open("a", encoding="utf-8") as results_handle:
        for index, test_case in enumerate(remaining, 1):
            test_id = str(test_case["test_id"])
            max_total_steps = max_total_steps_for_case(test_case, args)
            print(
                f"[pilot] {index}/{len(remaining)} {test_id} "
                f"{test_case['difficulty']}/{ratio_stratum_for_case(test_case)} "
                f"budget={max_total_steps}",
                file=sys.stderr,
                flush=True,
            )
            before = _counter_snapshot(query_client)
            parsed_query = None
            parse_error = None
            episode_result = None
            episode_error = None
            started = time.perf_counter()
            try:
                parsed_query = runtime.query_parser.parse(str(test_case["query"]))
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
            if parsed_query is not None:
                try:
                    episode_result = runtime.runner.run(
                        start_pano_id=str(test_case["start_pano_id"]),
                        target_room_id=parsed_query.target_room_id,
                        acceptable_target_room_ids=list(
                            test_case["acceptable_target_room_ids"]
                        ),
                        waypoint_room_ids=parsed_query.waypoint_room_ids,
                        max_total_steps=max_total_steps,
                        max_local_steps=args.max_local_steps,
                    )
                except Exception as exc:
                    episode_error = f"{type(exc).__name__}: {exc}"
            elapsed_s = time.perf_counter() - started
            after = _counter_snapshot(query_client)
            query_counts = _counter_delta(before, after)
            record = build_result_record(
                test_case=test_case,
                parsed_query=parsed_query,
                parse_error=parse_error,
                episode_result=episode_result,
                episode_error=episode_error,
                elapsed_s=elapsed_s,
                query_counts=query_counts,
                max_total_steps=max_total_steps,
                room_graph=runtime.room_graph,
                pano_room_mappings=runtime.pano_room_mappings,
            )
            raw_payload = {
                "test_case": test_case,
                "parsed_query": (
                    parsed_query.to_dict() if parsed_query is not None else None
                ),
                "parse_error": parse_error,
                "episode": (
                    episode_result.to_dict() if episode_result is not None else None
                ),
                "episode_error": episode_error,
                "evaluation": record,
            }
            atomic_write_json(episode_dir / f"{test_id}.json", raw_payload)
            results_handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            results_handle.flush()
            records.append(record)
            write_aggregate_outputs(
                output_dir=output_dir,
                records=records,
                configuration=runtime.configuration,
            )
            print(
                f"[pilot] {test_id} success={record['success']} "
                f"reason={record['reason']} steps={record['panorama_steps']} "
                f"time={elapsed_s:.1f}s",
                file=sys.stderr,
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
