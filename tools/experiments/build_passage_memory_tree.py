from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import shutil
import sys
from collections import deque
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.cli.run_similarity_passage_selection import DreamSimImageEmbedder  # noqa: E402
from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_DINOV2_SALAD_MODEL,
    create_image_embedder,
    load_json,
)


DEFAULT_METADATA_PATH = "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.metadata.json"
DEFAULT_RENDER_ROOT = "renders/room_grounding_fov90"
DEFAULT_ARTIFACTS_DIR = "dataset/sites/british_museum/normalized"
DEFAULT_BRANCHING_FACTOR = 5
DEFAULT_MAX_DEPTH = 4
DEFAULT_CURRENT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_CURRENT_RANK_GUARD = 10
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.82
DEFAULT_NEAR_DUPLICATE_PENALTY = 0.15
DEFAULT_PARENT_SIMILARITY_WEIGHT = 1.0
DEFAULT_CURRENT_SIMILARITY_WEIGHT = 0.0
DEFAULT_BRIDGE_DEPTH_PENALTY = 0.02
DEFAULT_BRIDGE_CONTINUITY_WEIGHT = 0.25
DEFAULT_BRIDGE_SIMILARITY_TIE_MARGIN = 0.01
TREE_EXPANSION_SCORE_MODES = ("parent", "path_continuity")
SIMILARITY_BACKENDS = ("dreamsim", "salad", "dinov2_patch_topk")
BRIDGE_SELECTION_MODES = ("weighted", "bridge_then_continuity")
DEFAULT_DINOV2_PATCH_MODEL = "facebook/dinov2-base"
DEFAULT_DINOV2_PATCH_TOP_K = 5
DEFAULT_DINOV2_PATCH_MAX_PATCHES = 24
_DINOV2_PATCH_RECORD_MEMORY_CACHE: dict[
    tuple[str, str, int],
    tuple[tuple[int, int], dict[str, dict]],
] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline visual-similarity Passage Memory Tree from a target passage "
            "toward a current observation."
        )
    )
    parser.add_argument("--room-id", required=True)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--target-passage-label")
    target_group.add_argument("--target-passage-image")
    current_group = parser.add_mutually_exclusive_group(required=True)
    current_group.add_argument("--current-image")
    current_group.add_argument("--current-pano-id")
    parser.add_argument(
        "--current-capture-index",
        type=int,
        help=(
            "When omitted with --current-pano-id, runs the 8-view "
            "alignment mode selected by --alignment-mode."
        ),
    )
    parser.add_argument("--representatives-path")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument("--similarity-backend", choices=SIMILARITY_BACKENDS, default="dreamsim")
    parser.add_argument("--dreamsim-type", default="ensemble")
    parser.add_argument("--dinov2-patch-model", default=DEFAULT_DINOV2_PATCH_MODEL)
    parser.add_argument("--dinov2-patch-top-k", type=int, default=DEFAULT_DINOV2_PATCH_TOP_K)
    parser.add_argument("--dinov2-patch-max-patches", type=int, default=DEFAULT_DINOV2_PATCH_MAX_PATCHES)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--branching-factor", type=int, default=DEFAULT_BRANCHING_FACTOR)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument(
        "--alignment-mode",
        choices=["bidirection"],
        default="bidirection",
        help="8-view bidirection alignment mode used by the formal direction pipeline.",
    )
    parser.add_argument(
        "--current-similarity-threshold",
        type=float,
        default=DEFAULT_CURRENT_SIMILARITY_THRESHOLD,
    )
    parser.add_argument("--current-rank-guard", type=int, default=DEFAULT_CURRENT_RANK_GUARD)
    parser.add_argument(
        "--near-duplicate-threshold",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    )
    parser.add_argument(
        "--near-duplicate-penalty",
        type=float,
        default=DEFAULT_NEAR_DUPLICATE_PENALTY,
    )
    parser.add_argument(
        "--parent-similarity-weight",
        type=float,
        default=DEFAULT_PARENT_SIMILARITY_WEIGHT,
    )
    parser.add_argument(
        "--current-similarity-weight",
        type=float,
        default=DEFAULT_CURRENT_SIMILARITY_WEIGHT,
        help=(
            "Optional debug/tuning weight. Default is 0 so tree expansion is "
            "driven by parent continuity; current similarity is used for stopping."
        ),
    )
    parser.add_argument(
        "--tree-expansion-score-mode",
        choices=TREE_EXPANSION_SCORE_MODES,
        default="path_continuity",
        help=(
            "Memory tree child expansion ranking. parent keeps the old one-step parent similarity; "
            "path_continuity ranks by the mean continuity of the full path ending at the candidate."
        ),
    )
    parser.add_argument("--bridge-depth-penalty", type=float, default=DEFAULT_BRIDGE_DEPTH_PENALTY)
    parser.add_argument(
        "--bridge-similarity-tie-margin",
        type=float,
        default=DEFAULT_BRIDGE_SIMILARITY_TIE_MARGIN,
        help=(
            "When ranking views in bridge_then_continuity mode, bridge scores within this margin "
            "are treated as a tie before root-target similarity and continuity tie-breaks."
        ),
    )
    parser.add_argument(
        "--bridge-continuity-weight",
        type=float,
        default=DEFAULT_BRIDGE_CONTINUITY_WEIGHT,
        help="Weight for each side's minimum chain continuity in bidirection bridge scoring.",
    )
    parser.add_argument(
        "--bridge-selection-mode",
        choices=BRIDGE_SELECTION_MODES,
        default="weighted",
        help=(
            "weighted keeps the original bridge + continuity formula; "
            "bridge_then_continuity ranks bridge similarity first and uses continuity as tie-break."
        ),
    )
    parser.add_argument(
        "--allow-same-bridge-item",
        action="store_true",
        help=(
            "Allow current and passage trees to bridge through the exact same memory item. "
            "By default same-item bridges are excluded because they give a trivial similarity of 1.0."
        ),
    )
    parser.add_argument(
        "--omit-embeddings",
        action="store_true",
        help="Do not write embedding vectors into tree.json.",
    )
    return parser


def create_memory_tree_embedder(*, similarity_backend: str, dreamsim_type: str, device: str, batch_size: int):
    if similarity_backend == "dreamsim":
        return DreamSimImageEmbedder(
            dreamsim_type=dreamsim_type,
            device=device,
            batch_size=batch_size,
        )
    if similarity_backend == "salad":
        return create_image_embedder(
            model_name=DEFAULT_DINOV2_SALAD_MODEL,
            device=device,
            batch_size=batch_size,
        )
    raise ValueError("similarity_backend must be dreamsim or salad for vector embedding encode.")


def similarity_method_prefix(similarity_backend: str) -> str:
    if similarity_backend == "salad":
        return "salad"
    if similarity_backend == "dinov2_patch_topk":
        return "dinov2_patch_topk"
    return "dreamsim"


def similarity_configuration(
    *,
    similarity_backend: str,
    dreamsim_type: str,
    dinov2_patch_model: str = DEFAULT_DINOV2_PATCH_MODEL,
    dinov2_patch_top_k: int = DEFAULT_DINOV2_PATCH_TOP_K,
    dinov2_patch_max_patches: int = DEFAULT_DINOV2_PATCH_MAX_PATCHES,
) -> dict:
    if similarity_backend == "dreamsim":
        similarity_model = f"dreamsim:{dreamsim_type}"
    elif similarity_backend == "salad":
        similarity_model = DEFAULT_DINOV2_SALAD_MODEL
    elif similarity_backend == "dinov2_patch_topk":
        similarity_model = str(dinov2_patch_model)
    else:
        raise ValueError(f"Unsupported similarity backend: {similarity_backend}")
    config = {
        "similarity_backend": similarity_backend,
        "similarity_model": similarity_model,
        "dreamsim_type": dreamsim_type if similarity_backend == "dreamsim" else None,
    }
    if similarity_backend == "dinov2_patch_topk":
        config.update(
            {
                "dinov2_patch_model": str(dinov2_patch_model),
                "dinov2_patch_top_k": int(dinov2_patch_top_k),
                "dinov2_patch_max_patches": int(dinov2_patch_max_patches),
                "similarity_kind": "symmetric_patch_topk",
            }
        )
    return config


def main() -> int:
    args = build_parser().parse_args()

    room_id = str(args.room_id)
    bidirectional_mode = bool(args.current_pano_id and args.current_capture_index is None)
    metadata_path = resolve_project_path(args.metadata_path)
    render_root = resolve_project_path(args.render_root)
    target_passage_image = (
        resolve_project_path(args.target_passage_image)
        if args.target_passage_image
        else None
    )
    if target_passage_image is not None and not target_passage_image.exists():
        raise SystemExit(f"Target passage image does not exist: {target_passage_image}")
    target_passage_label = (
        args.target_passage_label
        if args.target_passage_label
        else f"image_{safe_name(target_passage_image.stem)}"
    )
    representatives_path = None
    if target_passage_image is None:
        representatives_path = (
            resolve_project_path(args.representatives_path)
            if args.representatives_path
            else default_representatives_path(room_id)
        )
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            room_id=room_id,
            target_passage_label=target_passage_label,
            current_image=args.current_image,
            current_pano_id=args.current_pano_id,
            current_capture_index=args.current_capture_index,
        )
    )

    metadata_items = load_memory_metadata(metadata_path, render_root)
    room_items = [item for item in metadata_items if item.get("room_id") == room_id]
    if not room_items:
        raise SystemExit(f"No memory captures found for room: {room_id}")

    if target_passage_image is not None:
        target_index = find_matching_item_index(
            room_items,
            {"image_path": str(target_passage_image)},
        )
        if target_index is None:
            target_index = len(room_items)
            room_items.append(
                external_target_item(
                    target_passage_image,
                    room_id=room_id,
                    target_passage_label=target_passage_label,
                )
            )
    else:
        target_representative = load_target_representative(
            representatives_path=representatives_path,
            target_passage_label=target_passage_label,
        )
        target_index = find_matching_item_index(room_items, target_representative)
    if target_index is None:
        raise SystemExit(
            f"Target passage {target_passage_label!r} was not found in room metadata."
        )

    current_context: dict
    current_pool_index: int | None = None
    current_extra_path: Path | None = None
    current_view_indices: list[int] = []
    if args.current_image:
        current_extra_path = resolve_project_path(args.current_image)
        if not current_extra_path.exists():
            raise SystemExit(f"Current image does not exist: {current_extra_path}")
        current_context = {
            "mode": "image",
            "image_path": str(current_extra_path),
        }
    elif bidirectional_mode:
        current_view_indices = sorted(
            [
                index
                for index, item in enumerate(room_items)
                if item.get("pano_id") == args.current_pano_id
            ],
            key=lambda index: int(room_items[index].get("capture_index") or 0),
        )
        if not current_view_indices:
            raise SystemExit(f"No current views found for pano: {args.current_pano_id}")
        current_context = {
            "mode": "pano_8_views",
            "pano_id": args.current_pano_id,
            "view_count": len(current_view_indices),
            "views": [public_memory_item(room_items[index]) for index in current_view_indices],
        }
    else:
        current_pool_index = find_matching_item_index(
            room_items,
            {"pano_id": args.current_pano_id, "capture_index": args.current_capture_index},
        )
        if current_pool_index is None:
            raise SystemExit(
                "Current pano/capture was not found in room metadata: "
                f"{args.current_pano_id} #{args.current_capture_index}"
            )
        current_context = {
            "mode": "memory_capture",
            **public_memory_item(room_items[current_pool_index]),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    room_image_paths = [Path(str(item["image_path"])) for item in room_items]
    pairwise_similarities = None
    if args.similarity_backend == "dinov2_patch_topk":
        if not bidirectional_mode:
            raise SystemExit(
                "dinov2_patch_topk currently supports the 8-view bidirection mode; "
                "pass --current-pano-id without --current-capture-index."
            )
        patch_features = load_or_encode_dinov2_patch_features(
            image_paths=room_image_paths,
            memory_keys=[memory_key(item) for item in room_items],
            output_dir=output_dir,
            model_name=args.dinov2_patch_model,
            max_patches=args.dinov2_patch_max_patches,
            device=args.device,
            batch_size=args.batch_size,
        )
        pairwise_similarities = load_or_compute_dinov2_patch_similarity_matrix(
            patch_features=patch_features,
            image_paths=room_image_paths,
            memory_keys=[memory_key(item) for item in room_items],
            output_dir=output_dir,
            model_name=args.dinov2_patch_model,
            max_patches=args.dinov2_patch_max_patches,
            top_k=args.dinov2_patch_top_k,
        )
        encoded = np.zeros((len(room_items), 1), dtype=np.float32)
        room_embeddings = encoded
    else:
        embedder = create_memory_tree_embedder(
            similarity_backend=args.similarity_backend,
            dreamsim_type=args.dreamsim_type,
            device=args.device,
            batch_size=args.batch_size,
        )
        encode_paths = list(room_image_paths)
        if current_extra_path is not None:
            encode_paths.append(current_extra_path)
        encoded = normalize_rows(np.asarray(embedder.encode_image_paths(encode_paths), dtype=np.float32))
        room_embeddings = encoded[: len(room_items)]
    pano_graph = load_optional_pano_graph(resolve_project_path(args.artifacts_dir))

    if bidirectional_mode:
        if args.alignment_mode == "bidirection":
            result = build_bidirectional_alignment(
                room_items=room_items,
                room_embeddings=room_embeddings,
                target_index=target_index,
                target_passage_label=target_passage_label,
                current_view_indices=current_view_indices,
                current_context=current_context,
                branching_factor=args.branching_factor,
                max_depth=args.max_depth,
                near_duplicate_threshold=args.near_duplicate_threshold,
                near_duplicate_penalty_weight=args.near_duplicate_penalty,
                parent_similarity_weight=args.parent_similarity_weight,
                bridge_depth_penalty=args.bridge_depth_penalty,
                bridge_continuity_weight=args.bridge_continuity_weight,
                bridge_similarity_tie_margin=args.bridge_similarity_tie_margin,
                bridge_selection_mode=args.bridge_selection_mode,
                exclude_same_bridge_item=not args.allow_same_bridge_item,
                tree_expansion_score_mode=args.tree_expansion_score_mode,
                include_embeddings=(not args.omit_embeddings and args.similarity_backend != "dinov2_patch_topk"),
                similarity_backend=args.similarity_backend,
                dreamsim_type=args.dreamsim_type,
                pairwise_similarities=pairwise_similarities,
                dinov2_patch_model=args.dinov2_patch_model,
                dinov2_patch_top_k=args.dinov2_patch_top_k,
                dinov2_patch_max_patches=args.dinov2_patch_max_patches,
            )
            metrics = build_bidirectional_metrics(result, pano_graph=pano_graph)
            gallery_writer = write_bidirectional_gallery
        else:
            result = build_current_to_target_chain_alignment(
                room_items=room_items,
                room_embeddings=room_embeddings,
                target_index=target_index,
                target_passage_label=target_passage_label,
                current_view_indices=current_view_indices,
                current_context=current_context,
                branching_factor=args.branching_factor,
                max_depth=args.max_depth,
                near_duplicate_threshold=args.near_duplicate_threshold,
                near_duplicate_penalty_weight=args.near_duplicate_penalty,
                parent_similarity_weight=args.parent_similarity_weight,
                bridge_depth_penalty=args.bridge_depth_penalty,
                bridge_continuity_weight=args.bridge_continuity_weight,
                tree_expansion_score_mode=args.tree_expansion_score_mode,
                include_embeddings=not args.omit_embeddings,
                similarity_backend=args.similarity_backend,
                dreamsim_type=args.dreamsim_type,
            )
            metrics = build_current_to_target_metrics(result, pano_graph=pano_graph)
            gallery_writer = write_current_to_target_alignment_gallery
        result["metrics"] = metrics
        write_json(output_dir / "alignment.json", result)
        write_json(output_dir / "best_alignment.json", build_best_alignment_payload(result))
        write_json(output_dir / "metrics.json", metrics)
        gallery_writer(output_dir, result)
        summary = {
            "output_dir": str(output_dir),
            "alignment_json": str(output_dir / "alignment.json"),
            "best_alignment_json": str(output_dir / "best_alignment.json"),
            "metrics_json": str(output_dir / "metrics.json"),
            "alignment_gallery": str(output_dir / "alignment_gallery.html"),
            "alignment_gallery_embedded": str(output_dir / "alignment_gallery_embedded.html"),
            "selected_view": result["selected_view"],
            "metrics": metrics,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    current_embedding = (
        encoded[len(room_items)]
        if current_extra_path is not None
        else room_embeddings[int(current_pool_index)]
    )
    result = build_memory_tree(
        room_items=room_items,
        room_embeddings=room_embeddings,
        current_embedding=current_embedding,
        target_index=target_index,
        target_passage_label=target_passage_label,
        current_context=current_context,
        branching_factor=args.branching_factor,
        max_depth=args.max_depth,
        current_similarity_threshold=args.current_similarity_threshold,
        current_rank_guard=args.current_rank_guard,
        near_duplicate_threshold=args.near_duplicate_threshold,
        near_duplicate_penalty_weight=args.near_duplicate_penalty,
        parent_similarity_weight=args.parent_similarity_weight,
        current_similarity_weight=args.current_similarity_weight,
        include_embeddings=not args.omit_embeddings,
        similarity_backend=args.similarity_backend,
        dreamsim_type=args.dreamsim_type,
    )

    metrics = build_metrics(result, pano_graph=pano_graph)
    result["metrics"] = metrics

    write_json(output_dir / "tree.json", result)
    write_json(output_dir / "best_chain.json", build_best_chain_payload(result))
    write_json(output_dir / "metrics.json", metrics)
    write_chain_contact_sheet(output_dir / "chain_contact_sheet.png", result)
    write_tree_gallery(output_dir, result)

    summary = {
        "output_dir": str(output_dir),
        "tree_json": str(output_dir / "tree.json"),
        "best_chain_json": str(output_dir / "best_chain.json"),
        "metrics_json": str(output_dir / "metrics.json"),
        "chain_contact_sheet": str(output_dir / "chain_contact_sheet.png"),
        "tree_gallery": str(output_dir / "tree_gallery.html"),
        "tree_gallery_embedded": str(output_dir / "tree_gallery_embedded.html"),
        "stop": result["stop"],
        "metrics": metrics,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0

def build_memory_tree(
    *,
    room_items: Sequence[dict],
    room_embeddings,
    current_embedding,
    target_index: int,
    target_passage_label: str,
    current_context: dict,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    current_similarity_threshold: float = DEFAULT_CURRENT_SIMILARITY_THRESHOLD,
    current_rank_guard: int = DEFAULT_CURRENT_RANK_GUARD,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    near_duplicate_penalty_weight: float = DEFAULT_NEAR_DUPLICATE_PENALTY,
    parent_similarity_weight: float = DEFAULT_PARENT_SIMILARITY_WEIGHT,
    current_similarity_weight: float = DEFAULT_CURRENT_SIMILARITY_WEIGHT,
    include_embeddings: bool = True,
    similarity_backend: str = "dreamsim",
    dreamsim_type: str = "ensemble",
) -> dict:
    if not room_items:
        raise ValueError("room_items must not be empty.")
    if target_index < 0 or target_index >= len(room_items):
        raise ValueError("target_index is out of range.")
    if branching_factor < 1:
        raise ValueError("branching_factor must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    if parent_similarity_weight < 0.0 or current_similarity_weight < 0.0:
        raise ValueError("similarity weights must be non-negative.")
    if parent_similarity_weight == 0.0 and current_similarity_weight == 0.0:
        raise ValueError("At least one similarity weight must be positive.")

    embeddings = normalize_rows(np.asarray(room_embeddings, dtype=np.float32))
    current_vector = normalize_rows(np.asarray(current_embedding, dtype=np.float32).reshape(1, -1))[0]
    if embeddings.shape[0] != len(room_items):
        raise ValueError("room_embeddings and room_items must have the same length.")

    pairwise = embeddings @ embeddings.T
    current_similarities = embeddings @ current_vector
    current_ranks = rank_by_current_similarity(current_similarities, room_items)
    nodes: list[dict] = []
    edges: list[dict] = []

    def add_node(
        *,
        item_index: int,
        depth: int,
        parent_id: str | None,
        path_indices: list[int],
        sim_to_parent: float | None,
        near_duplicate_count: int,
        near_duplicate_penalty: float,
        score: float,
    ) -> dict:
        node_id = f"n{len(nodes)}"
        item = room_items[item_index]
        sim_to_current = float(current_similarities[item_index])
        node = {
            "id": node_id,
            "depth": int(depth),
            "parent_id": parent_id,
            "memory_key": memory_key(item),
            "target_passage_label": target_passage_label if item_index == target_index else None,
            **public_memory_item(item),
            "dreamsim_embedding": (
                [float(value) for value in embeddings[item_index].tolist()]
                if include_embeddings
                else None
            ),
            "sim_to_parent": sim_to_parent,
            "sim_to_current": sim_to_current,
            "top_current_rank": int(current_ranks[memory_key(item)]),
            "near_duplicate_count": int(near_duplicate_count),
            "near_duplicate_penalty": float(near_duplicate_penalty),
            "score": float(score),
        }
        nodes.append(node)
        return node

    root = add_node(
        item_index=target_index,
        depth=0,
        parent_id=None,
        path_indices=[target_index],
        sim_to_parent=None,
        near_duplicate_count=0,
        near_duplicate_penalty=0.0,
        score=float(current_similarities[target_index]),
    )
    frontier: list[tuple[dict, list[int]]] = [(root, [target_index])]
    best_node = root
    stop_node = root if is_stop_node(root, current_similarity_threshold, current_rank_guard) else None

    for depth in range(1, max_depth + 1):
        if stop_node is not None:
            break
        next_frontier: list[tuple[dict, list[int]]] = []
        depth_nodes: list[dict] = []
        for parent_node, path_indices in frontier:
            parent_index = index_for_memory_key(room_items, parent_node["memory_key"])
            scored_candidates = []
            for candidate_index, candidate in enumerate(room_items):
                if candidate_index in path_indices:
                    continue
                candidate_score = score_candidate(
                    candidate_index=candidate_index,
                    parent_index=parent_index,
                    path_indices=path_indices,
                    pairwise_similarities=pairwise,
                    current_similarities=current_similarities,
                    near_duplicate_threshold=near_duplicate_threshold,
                    near_duplicate_penalty_weight=near_duplicate_penalty_weight,
                    parent_similarity_weight=parent_similarity_weight,
                    current_similarity_weight=current_similarity_weight,
                )
                scored_candidates.append(candidate_score)
            scored_candidates.sort(
                key=lambda candidate: (
                    -candidate["score"],
                    -candidate["sim_to_parent"],
                    -candidate["sim_to_current"],
                    memory_key(room_items[int(candidate["candidate_index"])]),
                )
            )
            for candidate_score in scored_candidates[:branching_factor]:
                candidate_index = int(candidate_score["candidate_index"])
                child_path = [*path_indices, candidate_index]
                child_node = add_node(
                    item_index=candidate_index,
                    depth=depth,
                    parent_id=parent_node["id"],
                    path_indices=child_path,
                    sim_to_parent=float(candidate_score["sim_to_parent"]),
                    near_duplicate_count=int(candidate_score["near_duplicate_count"]),
                    near_duplicate_penalty=float(candidate_score["near_duplicate_penalty"]),
                    score=float(candidate_score["score"]),
                )
                edge = {
                    "source": parent_node["id"],
                    "target": child_node["id"],
                    "source_memory_key": parent_node["memory_key"],
                    "target_memory_key": child_node["memory_key"],
                    "sim_to_parent": child_node["sim_to_parent"],
                    "sim_to_current": child_node["sim_to_current"],
                    "near_duplicate_penalty": child_node["near_duplicate_penalty"],
                    "score": child_node["score"],
                }
                edges.append(edge)
                next_frontier.append((child_node, child_path))
                depth_nodes.append(child_node)
                if better_leaf(child_node, best_node):
                    best_node = child_node

        stop_candidates = [
            node
            for node in depth_nodes
            if is_stop_node(node, current_similarity_threshold, current_rank_guard)
        ]
        if stop_candidates:
            stop_candidates.sort(
                key=lambda node: (
                    -float(node["sim_to_current"]),
                    int(node["top_current_rank"]),
                    -float(node["score"]),
                    str(node["memory_key"]),
                )
            )
            stop_node = stop_candidates[0]
            best_node = stop_node
            break
        frontier = next_frontier
        if not frontier:
            break

    selected_node = stop_node or best_node
    stop_reason = "current_similarity_threshold_reached" if stop_node else "max_depth_not_close_enough"
    if not stop_node and len(nodes) == 1 and max_depth == 0:
        stop_reason = "max_depth_zero_not_close_enough"
    tree_path = path_to_root(selected_node, nodes)
    navigation_chain = list(reversed(tree_path))

    return {
        "method": f"{similarity_method_prefix(similarity_backend)}_passage_memory_tree",
        "configuration": {
            **similarity_configuration(
                similarity_backend=similarity_backend,
                dreamsim_type=dreamsim_type,
            ),
            "target_passage_label": target_passage_label,
            "room_id": room_items[target_index].get("room_id"),
            "branching_factor": int(branching_factor),
            "max_depth": int(max_depth),
            "current_similarity_threshold": float(current_similarity_threshold),
            "current_rank_guard": int(current_rank_guard),
            "near_duplicate_threshold": float(near_duplicate_threshold),
            "near_duplicate_penalty": float(near_duplicate_penalty_weight),
            "parent_similarity_weight": float(parent_similarity_weight),
            "current_similarity_weight": float(current_similarity_weight),
            "scoring_formula": (
                "parent_similarity_weight * sim_to_parent + "
                "current_similarity_weight * sim_to_current - near_duplicate_penalty"
            ),
            "current_similarity_default_role": "stop_condition_only",
            "include_embeddings": bool(include_embeddings),
        },
        "current_view": dict(current_context),
        "target_node_id": root["id"],
        "best_node_id": selected_node["id"],
        "stop": {
            "found": bool(stop_node),
            "reason": stop_reason,
            "node_id": selected_node["id"],
            "sim_to_current": float(selected_node["sim_to_current"]),
            "top_current_rank": int(selected_node["top_current_rank"]),
            "threshold": float(current_similarity_threshold),
            "rank_guard": int(current_rank_guard),
        },
        "current_nearest_neighbors": current_nearest_neighbors(
            room_items=room_items,
            current_similarities=current_similarities,
            limit=max(int(current_rank_guard), 10),
        ),
        "nodes": nodes,
        "edges": edges,
        "best_chain_root_to_current": tree_path,
        "navigation_chain_current_to_target": navigation_chain,
    }


def score_candidate(
    *,
    candidate_index: int,
    parent_index: int,
    path_indices: Sequence[int],
    pairwise_similarities,
    current_similarities,
    near_duplicate_threshold: float,
    near_duplicate_penalty_weight: float,
    parent_similarity_weight: float,
    current_similarity_weight: float,
) -> dict:
    sim_to_parent = float(pairwise_similarities[candidate_index, parent_index])
    sim_to_current = float(current_similarities[candidate_index])
    near_duplicate_count = sum(
        1
        for path_index in path_indices
        if float(pairwise_similarities[candidate_index, path_index]) >= near_duplicate_threshold
    )
    near_duplicate_penalty = float(near_duplicate_count) * float(near_duplicate_penalty_weight)
    score = (
        float(parent_similarity_weight) * sim_to_parent
        + float(current_similarity_weight) * sim_to_current
        - near_duplicate_penalty
    )
    return {
        "candidate_index": int(candidate_index),
        "sim_to_parent": sim_to_parent,
        "sim_to_current": sim_to_current,
        "near_duplicate_count": int(near_duplicate_count),
        "near_duplicate_penalty": near_duplicate_penalty,
        "score": float(score),
    }


def is_stop_node(node: dict, current_similarity_threshold: float, current_rank_guard: int) -> bool:
    return (
        float(node["sim_to_current"]) >= float(current_similarity_threshold)
        and int(node["top_current_rank"]) <= int(current_rank_guard)
    )


def better_leaf(candidate: dict, incumbent: dict) -> bool:
    return (
        float(candidate["sim_to_current"]),
        -int(candidate["top_current_rank"]),
        float(candidate["score"]),
    ) > (
        float(incumbent["sim_to_current"]),
        -int(incumbent["top_current_rank"]),
        float(incumbent["score"]),
    )


def path_to_root(node: dict, nodes: Sequence[dict]) -> list[dict]:
    by_id = {item["id"]: item for item in nodes}
    path = []
    cursor = node
    while cursor is not None:
        path.append(cursor)
        parent_id = cursor.get("parent_id")
        cursor = by_id.get(parent_id) if parent_id else None
    return list(reversed(path))


def build_best_chain_payload(result: dict) -> dict:
    return {
        "method": result["method"],
        "configuration": result["configuration"],
        "current_view": result["current_view"],
        "stop": result["stop"],
        "target_node_id": result["target_node_id"],
        "best_node_id": result["best_node_id"],
        "memory_chain_current_to_target": result["navigation_chain_current_to_target"],
        "navigation_chain": [
            {"kind": "current_view", **result["current_view"]},
            *[
                {"kind": "memory_node", **node}
                for node in result["navigation_chain_current_to_target"]
            ],
        ],
    }



def build_current_to_target_chain_alignment(
    *,
    room_items: Sequence[dict],
    room_embeddings,
    target_index: int,
    target_passage_label: str,
    current_view_indices: Sequence[int],
    current_context: dict,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    near_duplicate_penalty_weight: float = DEFAULT_NEAR_DUPLICATE_PENALTY,
    parent_similarity_weight: float = DEFAULT_PARENT_SIMILARITY_WEIGHT,
    bridge_depth_penalty: float = DEFAULT_BRIDGE_DEPTH_PENALTY,
    bridge_continuity_weight: float = DEFAULT_BRIDGE_CONTINUITY_WEIGHT,
    tree_expansion_score_mode: str = "parent",
    include_embeddings: bool = True,
    similarity_backend: str = "dreamsim",
    dreamsim_type: str = "ensemble",
) -> dict:
    if not current_view_indices:
        raise ValueError("current_view_indices must not be empty.")
    if tree_expansion_score_mode not in TREE_EXPANSION_SCORE_MODES:
        raise ValueError(f"tree_expansion_score_mode must be one of {TREE_EXPANSION_SCORE_MODES}.")
    if target_index < 0 or target_index >= len(room_items):
        raise ValueError("target_index is out of range.")
    if branching_factor < 1:
        raise ValueError("branching_factor must be positive.")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative.")
    embeddings = normalize_rows(np.asarray(room_embeddings, dtype=np.float32))
    if embeddings.shape[0] != len(room_items):
        raise ValueError("room_embeddings and room_items must have the same length.")
    pairwise = embeddings @ embeddings.T

    current_roots = [int(index) for index in current_view_indices]
    view_alignments = []
    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    for view_order, root_index in enumerate(current_roots):
        root_item = room_items[root_index]
        capture_index = int(root_item.get("capture_index") or view_order)
        root_target_similarity = float(pairwise[root_index, target_index])
        root_target_direct_match = root_target_similarity >= float(near_duplicate_threshold)
        current_tree = expand_parent_tree(
            room_items=room_items,
            embeddings=embeddings,
            pairwise_similarities=pairwise,
            root_index=root_index,
            node_prefix=f"v{capture_index}_n",
            tree_role="current_to_target",
            branching_factor=branching_factor,
            max_depth=max_depth,
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_penalty_weight=near_duplicate_penalty_weight,
            parent_similarity_weight=parent_similarity_weight,
            include_embeddings=include_embeddings,
            tree_expansion_score_mode=tree_expansion_score_mode,
            exclude_indices=set(current_roots) - {root_index},
            extra_root_fields={"current_view_order": view_order},
        )
        for node in current_tree["nodes"]:
            node["current_view_order"] = view_order
            node["current_capture_index"] = capture_index
            node["target_similarity"] = float(pairwise[int(node["item_index"]), target_index])
            if int(node["item_index"]) == int(target_index):
                node["target_passage_label"] = target_passage_label
        target_nodes = [
            node for node in current_tree["nodes"] if int(node["item_index"]) == int(target_index)
        ]

        if root_target_direct_match:
            selected_node = current_tree["root"]
            chain = [selected_node]
            bottleneck_similarity = 1.0
            mean_chain_similarity = 1.0
            target_similarity = root_target_similarity
            path_length = 0
            chain_score = root_target_similarity + float(bridge_continuity_weight)
            target_found = True
            view_selection_reason = "root_target_direct_match"
        elif target_nodes:
            target_candidates = []
            for node in target_nodes:
                chain = node_path(current_tree, node["id"])
                bottleneck_similarity, mean_chain_similarity = chain_similarity_stats(chain)
                path_length = max(len(chain) - 1, 0)
                chain_score = current_to_target_chain_score(
                    bottleneck_similarity=bottleneck_similarity,
                    mean_chain_similarity=mean_chain_similarity,
                    path_length=path_length,
                    depth_penalty=bridge_depth_penalty,
                    continuity_weight=bridge_continuity_weight,
                )
                target_candidates.append(
                    {
                        "node": node,
                        "chain": chain,
                        "bottleneck_similarity": bottleneck_similarity,
                        "mean_chain_similarity": mean_chain_similarity,
                        "path_length": path_length,
                        "chain_score": chain_score,
                    }
                )
            target_candidates.sort(key=target_chain_sort_key)
            best_target = target_candidates[0]
            selected_node = best_target["node"]
            chain = best_target["chain"]
            bottleneck_similarity = best_target["bottleneck_similarity"]
            mean_chain_similarity = best_target["mean_chain_similarity"]
            path_length = best_target["path_length"]
            chain_score = best_target["chain_score"]
            target_similarity = 1.0
            target_found = True
            view_selection_reason = "target_chain_found"
        else:
            selected_node = best_target_progress_node(
                current_tree=current_tree,
                pairwise_similarities=pairwise,
                target_index=target_index,
            )
            chain = node_path(current_tree, selected_node["id"])
            bottleneck_similarity, mean_chain_similarity = chain_similarity_stats(chain)
            path_length = max(len(chain) - 1, 0)
            target_similarity = float(pairwise[int(selected_node["item_index"]), target_index])
            chain_score = (
                target_similarity
                + float(bridge_continuity_weight) * bottleneck_similarity
                - float(bridge_depth_penalty) * path_length
            )
            target_found = False
            view_selection_reason = "target_not_found_fallback"

        view_alignment = {
            "view_order": int(view_order),
            "capture_index": int(capture_index),
            "root": current_tree["root"],
            "selected_node_id": selected_node["id"],
            "target_node_id": selected_node["id"] if target_found else None,
            "target_found": bool(target_found),
            "selection_reason": view_selection_reason,
            "root_target_similarity": root_target_similarity,
            "root_target_direct_match": bool(root_target_direct_match),
            "target_similarity": float(target_similarity),
            "bottleneck_similarity": float(bottleneck_similarity),
            "mean_chain_similarity": float(mean_chain_similarity),
            "path_length": int(path_length),
            "chain_score": float(chain_score),
            "view_score": float(chain_score),
            "total_score": float(chain_score),
            "memory_chain_current_to_target": chain,
            "current_tree": current_tree,
        }
        view_alignments.append(view_alignment)
        all_nodes.extend(dict(node) for node in current_tree["nodes"])
        all_edges.extend(dict(edge) for edge in current_tree["edges"])

    view_alignments.sort(key=current_to_target_alignment_sort_key)
    selected = view_alignments[0]
    if selected.get("root_target_direct_match"):
        selection_reason = "root_target_direct_match"
    elif selected.get("target_found"):
        selection_reason = "target_chain_found"
    else:
        selection_reason = "target_not_found_fallback"
    for rank, item in enumerate(view_alignments, start=1):
        item["rank"] = rank
        item["selected"] = item is selected

    selected_view = {
        "view_order": selected["view_order"],
        "capture_index": selected["capture_index"],
        **public_memory_item(room_items[current_roots[selected["view_order"]]]),
        "view_score": float(selected["view_score"]),
        "chain_score": float(selected["chain_score"]),
        "total_score": float(selected["total_score"]),
        "target_found": bool(selected["target_found"]),
        "target_similarity": float(selected["target_similarity"]),
        "bottleneck_similarity": float(selected["bottleneck_similarity"]),
        "mean_chain_similarity": float(selected["mean_chain_similarity"]),
        "path_length": int(selected["path_length"]),
        "root_target_similarity": float(selected["root_target_similarity"]),
        "root_target_direct_match": bool(selected.get("root_target_direct_match")),
        "selection_reason": selection_reason,
    }
    return {
        "method": f"{similarity_method_prefix(similarity_backend)}_current_to_target_memory_chain",
        "configuration": {
            **similarity_configuration(
                similarity_backend=similarity_backend,
                dreamsim_type=dreamsim_type,
            ),
            "target_passage_label": target_passage_label,
            "room_id": room_items[target_index].get("room_id"),
            "alignment_mode": "current_to_target",
            "branching_factor": int(branching_factor),
            "max_depth": int(max_depth),
            "near_duplicate_threshold": float(near_duplicate_threshold),
            "near_duplicate_penalty": float(near_duplicate_penalty_weight),
            "parent_similarity_weight": float(parent_similarity_weight),
            "tree_expansion_score_mode": tree_expansion_score_mode,
            "tree_expansion_scoring_formula": (
                "mean(path_edge_similarities + [sim_to_parent]) - near_duplicate_penalty"
                if tree_expansion_score_mode == "path_continuity"
                else "parent_similarity_weight * sim_to_parent - near_duplicate_penalty"
            ),
            "bridge_depth_penalty": float(bridge_depth_penalty),
            "bridge_continuity_weight": float(bridge_continuity_weight),
            "root_target_direct_match_threshold": float(near_duplicate_threshold),
            "current_expansion_excludes_other_current_views": True,
            "current_expansion_allows_target_passage": True,
            "view_scoring_formula": (
                "target_found first; then bottleneck_similarity + "
                "bridge_continuity_weight * mean_chain_similarity - "
                "bridge_depth_penalty * path_length"
            ),
            "fallback_scoring_formula": (
                "target_similarity + bridge_continuity_weight * bottleneck_similarity - "
                "bridge_depth_penalty * path_length"
            ),
            "include_embeddings": bool(include_embeddings),
        },
        "current_view": dict(current_context),
        "selection_reason": selection_reason,
        "target_node_id": memory_key(room_items[target_index]),
        "view_alignments": view_alignments,
        "selected_view": selected_view,
        "selected_alignment": selected,
        "nodes": all_nodes,
        "edges": all_edges,
    }


def chain_similarity_stats(chain: Sequence[dict]) -> tuple[float, float]:
    values = [
        float(node["sim_to_parent"])
        for node in chain[1:]
        if node.get("sim_to_parent") is not None
    ]
    if not values:
        return 1.0, 1.0
    return min(values), sum(values) / len(values)


def current_to_target_chain_score(
    *,
    bottleneck_similarity: float,
    mean_chain_similarity: float,
    path_length: int,
    depth_penalty: float,
    continuity_weight: float,
) -> float:
    return float(bottleneck_similarity) + float(continuity_weight) * float(mean_chain_similarity) - float(depth_penalty) * int(path_length)


def target_chain_sort_key(candidate: dict) -> tuple:
    return (
        -float(candidate["chain_score"]),
        -float(candidate["bottleneck_similarity"]),
        -float(candidate["mean_chain_similarity"]),
        int(candidate["path_length"]),
        str(candidate["node"].get("memory_key")),
    )


def current_to_target_alignment_sort_key(item: dict) -> tuple:
    direct_match = bool(item.get("root_target_direct_match"))
    return (
        -int(direct_match),
        -int(bool(item.get("target_found"))),
        -float(item.get("chain_score", 0.0)),
        -float(item.get("bottleneck_similarity", 0.0)),
        -float(item.get("mean_chain_similarity", 0.0)),
        -float(item.get("target_similarity", 0.0)),
        int(item.get("path_length", 0)),
        int(item.get("capture_index", 0)),
    )


def best_target_progress_node(
    *,
    current_tree: dict,
    pairwise_similarities,
    target_index: int,
) -> dict:
    nodes = list(current_tree.get("nodes", []))
    if not nodes:
        raise RuntimeError("Current-to-target tree has no nodes.")

    def progress_key(node: dict) -> tuple:
        chain = node_path(current_tree, node["id"])
        bottleneck_similarity, mean_chain_similarity = chain_similarity_stats(chain)
        return (
            float(pairwise_similarities[int(node["item_index"]), int(target_index)]),
            bottleneck_similarity,
            mean_chain_similarity,
            -int(node.get("depth", 0)),
            str(node.get("memory_key")),
        )

    return max(nodes, key=progress_key)


def build_bidirectional_alignment(
    *,
    room_items: Sequence[dict],
    room_embeddings,
    target_index: int,
    target_passage_label: str,
    current_view_indices: Sequence[int],
    current_context: dict,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    max_depth: int = DEFAULT_MAX_DEPTH,
    near_duplicate_threshold: float = DEFAULT_NEAR_DUPLICATE_THRESHOLD,
    near_duplicate_penalty_weight: float = DEFAULT_NEAR_DUPLICATE_PENALTY,
    parent_similarity_weight: float = DEFAULT_PARENT_SIMILARITY_WEIGHT,
    bridge_depth_penalty: float = DEFAULT_BRIDGE_DEPTH_PENALTY,
    bridge_continuity_weight: float = DEFAULT_BRIDGE_CONTINUITY_WEIGHT,
    bridge_similarity_tie_margin: float = DEFAULT_BRIDGE_SIMILARITY_TIE_MARGIN,
    bridge_selection_mode: str = "weighted",
    exclude_same_bridge_item: bool = True,
    tree_expansion_score_mode: str = "parent",
    include_embeddings: bool = True,
    similarity_backend: str = "dreamsim",
    dreamsim_type: str = "ensemble",
    pairwise_similarities=None,
    dinov2_patch_model: str = DEFAULT_DINOV2_PATCH_MODEL,
    dinov2_patch_top_k: int = DEFAULT_DINOV2_PATCH_TOP_K,
    dinov2_patch_max_patches: int = DEFAULT_DINOV2_PATCH_MAX_PATCHES,
) -> dict:
    if not current_view_indices:
        raise ValueError("current_view_indices must not be empty.")
    if tree_expansion_score_mode not in TREE_EXPANSION_SCORE_MODES:
        raise ValueError(f"tree_expansion_score_mode must be one of {TREE_EXPANSION_SCORE_MODES}.")
    if bridge_selection_mode not in BRIDGE_SELECTION_MODES:
        raise ValueError(f"bridge_selection_mode must be one of {BRIDGE_SELECTION_MODES}.")
    embeddings = normalize_rows(np.asarray(room_embeddings, dtype=np.float32))
    if embeddings.shape[0] != len(room_items):
        raise ValueError("room_embeddings and room_items must have the same length.")
    if pairwise_similarities is None:
        pairwise = embeddings @ embeddings.T
    else:
        pairwise = np.asarray(pairwise_similarities, dtype=np.float32)
        expected_shape = (len(room_items), len(room_items))
        if pairwise.shape != expected_shape:
            raise ValueError(
                f"pairwise_similarities must have shape {expected_shape}, got {pairwise.shape}."
            )

    passage_tree = expand_parent_tree(
        room_items=room_items,
        embeddings=embeddings,
        pairwise_similarities=pairwise,
        root_index=target_index,
        node_prefix="p",
        tree_role="passage",
        branching_factor=branching_factor,
        max_depth=max_depth,
        near_duplicate_threshold=near_duplicate_threshold,
        near_duplicate_penalty_weight=near_duplicate_penalty_weight,
        parent_similarity_weight=parent_similarity_weight,
        include_embeddings=include_embeddings,
        exclude_indices=set(current_view_indices),
        extra_root_fields={"target_passage_label": target_passage_label},
    )
    passage_leaves = leaf_nodes(passage_tree)

    view_alignments = []
    all_nodes = [dict(node) for node in passage_tree["nodes"]]
    all_edges = [dict(edge) for edge in passage_tree["edges"]]
    for view_order, root_index in enumerate(current_view_indices):
        root_item = room_items[root_index]
        capture_index = int(root_item.get("capture_index") or view_order)
        current_tree = expand_parent_tree(
            room_items=room_items,
            embeddings=embeddings,
            pairwise_similarities=pairwise,
            root_index=root_index,
            node_prefix=f"v{capture_index}_n",
            tree_role="current",
            branching_factor=branching_factor,
            max_depth=max_depth,
            near_duplicate_threshold=near_duplicate_threshold,
            near_duplicate_penalty_weight=near_duplicate_penalty_weight,
            parent_similarity_weight=parent_similarity_weight,
            include_embeddings=include_embeddings,
            tree_expansion_score_mode=tree_expansion_score_mode,
            exclude_indices={target_index, *current_view_indices},
            extra_root_fields={"current_view_order": view_order},
        )
        for node in current_tree["nodes"]:
            node["current_view_order"] = view_order
            node["current_capture_index"] = capture_index
        current_leaves = leaf_nodes(current_tree)
        best_bridge = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=current_leaves,
            passage_leaves=passage_leaves,
            pairwise_similarities=pairwise,
            bridge_depth_penalty=bridge_depth_penalty,
            bridge_continuity_weight=bridge_continuity_weight,
            bridge_selection_mode=bridge_selection_mode,
            exclude_same_bridge_item=exclude_same_bridge_item,
        )
        root_target_similarity = float(pairwise[root_index, target_index])
        root_target_direct_match = root_target_similarity >= float(near_duplicate_threshold)
        view_score = float(best_bridge["total_score"])
        current_chain = node_path(current_tree, best_bridge["current_node_id"])
        passage_chain_root_to_bridge = node_path(passage_tree, best_bridge["passage_node_id"])
        passage_chain_bridge_to_target = list(reversed(passage_chain_root_to_bridge))
        view_alignment = {
            "view_order": view_order,
            "capture_index": capture_index,
            "root": current_tree["root"],
            "best_bridge": best_bridge,
            "root_target_similarity": root_target_similarity,
            "root_target_direct_match": bool(root_target_direct_match),
            "view_score": view_score,
            "current_chain_root_to_bridge": current_chain,
            "passage_chain_root_to_bridge": passage_chain_root_to_bridge,
            "passage_chain_bridge_to_target": passage_chain_bridge_to_target,
            "current_tree": current_tree,
        }
        view_alignments.append(view_alignment)
        all_nodes.extend(dict(node) for node in current_tree["nodes"])
        all_edges.extend(dict(edge) for edge in current_tree["edges"])

    view_alignments.sort(
        key=lambda item: _view_alignment_sort_key(
            item,
            bridge_similarity_tie_margin=bridge_similarity_tie_margin,
        )
    )
    selected = view_alignments[0]
    selection_reason = (
        "root_target_direct_match"
        if selected.get("root_target_direct_match")
        else "best_bridge_score"
    )
    for rank, item in enumerate(view_alignments, start=1):
        item["rank"] = rank
        item["selected"] = item is selected

    selected_view = {
        "view_order": selected["view_order"],
        "capture_index": selected["capture_index"],
        **public_memory_item(room_items[current_view_indices[selected["view_order"]]]),
        "view_score": float(selected["view_score"]),
        "root_target_similarity": float(selected["root_target_similarity"]),
        "root_target_direct_match": bool(selected.get("root_target_direct_match")),
        "selection_reason": selection_reason,
        "total_score": float(selected["best_bridge"]["total_score"]),
        "bridge_similarity": float(selected["best_bridge"]["bridge_similarity"]),
        "continuity_score": float(selected["best_bridge"].get("continuity_score", selected["best_bridge"]["total_score"])),
        "chain_mean_continuity": float(selected["best_bridge"].get("chain_mean_continuity", 0.0)),
        "chain_bottleneck_similarity": float(selected["best_bridge"].get("chain_bottleneck_similarity", 0.0)),
        "bridge_selection_mode": selected["best_bridge"].get("bridge_selection_mode", bridge_selection_mode),
    }
    return {
        "method": f"{similarity_method_prefix(similarity_backend)}_bidirection_passage_alignment",
        "configuration": {
            **similarity_configuration(
                similarity_backend=similarity_backend,
                dreamsim_type=dreamsim_type,
                dinov2_patch_model=dinov2_patch_model,
                dinov2_patch_top_k=dinov2_patch_top_k,
                dinov2_patch_max_patches=dinov2_patch_max_patches,
            ),
            "target_passage_label": target_passage_label,
            "room_id": room_items[target_index].get("room_id"),
            "branching_factor": int(branching_factor),
            "max_depth": int(max_depth),
            "near_duplicate_threshold": float(near_duplicate_threshold),
            "near_duplicate_penalty": float(near_duplicate_penalty_weight),
            "parent_similarity_weight": float(parent_similarity_weight),
            "bridge_depth_penalty": float(bridge_depth_penalty),
            "bridge_continuity_weight": float(bridge_continuity_weight),
            "bridge_similarity_tie_margin": float(bridge_similarity_tie_margin),
            "bridge_selection_mode": bridge_selection_mode,
            "exclude_same_bridge_item": bool(exclude_same_bridge_item),
            "bridge_score_mode": (
                "bidirection" if bridge_selection_mode == "weighted" else "bridge_then_continuity"
            ),
            "root_target_direct_match_threshold": float(near_duplicate_threshold),
            "passage_expansion_excludes_current_views": True,
            "current_expansion_excludes_target_passage_and_current_views": True,
            "bridge_scoring_formula": bridge_scoring_formula(bridge_selection_mode),
            "view_scoring_formula": (
                "bridge_similarity first; root_target_similarity and continuity tie-breaks"
                if bridge_selection_mode == "bridge_then_continuity"
                else "bridge_total_score"
            ),
            "include_embeddings": bool(include_embeddings),
        },
        "current_view": dict(current_context),
        "selection_reason": selection_reason,
        "target_node_id": passage_tree["root"]["id"],
        "passage_tree": passage_tree,
        "view_alignments": view_alignments,
        "selected_view": selected_view,
        "selected_alignment": selected,
        "nodes": all_nodes,
        "edges": all_edges,
    }


def expand_parent_tree(
    *,
    room_items: Sequence[dict],
    embeddings,
    pairwise_similarities,
    root_index: int,
    node_prefix: str,
    tree_role: str,
    branching_factor: int,
    max_depth: int,
    near_duplicate_threshold: float,
    near_duplicate_penalty_weight: float,
    parent_similarity_weight: float,
    include_embeddings: bool,
    tree_expansion_score_mode: str = "parent",
    exclude_indices: set[int] | None = None,
    extra_root_fields: dict | None = None,
) -> dict:
    if tree_expansion_score_mode not in TREE_EXPANSION_SCORE_MODES:
        raise ValueError(f"tree_expansion_score_mode must be one of {TREE_EXPANSION_SCORE_MODES}.")
    nodes: list[dict] = []
    edges: list[dict] = []
    excluded = set(exclude_indices or set())

    def add_node(
        *,
        item_index: int,
        depth: int,
        parent_id: str | None,
        path_indices: list[int],
        sim_to_parent: float | None,
        near_duplicate_count: int,
        near_duplicate_penalty: float,
        score: float,
        path_edge_similarities: list[float] | None = None,
        path_mean_continuity: float = 1.0,
        path_bottleneck_similarity: float = 1.0,
        extra_fields: dict | None = None,
    ) -> dict:
        node_id = f"{node_prefix}{len(nodes)}"
        item = room_items[item_index]
        node = {
            "id": node_id,
            "tree_role": tree_role,
            "item_index": int(item_index),
            "depth": int(depth),
            "parent_id": parent_id,
            "memory_key": memory_key(item),
            **public_memory_item(item),
            "dreamsim_embedding": (
                [float(value) for value in embeddings[item_index].tolist()]
                if include_embeddings
                else None
            ),
            "sim_to_parent": sim_to_parent,
            "near_duplicate_count": int(near_duplicate_count),
            "near_duplicate_penalty": float(near_duplicate_penalty),
            "score": float(score),
            "path_edge_similarities": [float(value) for value in (path_edge_similarities or [])],
            "path_mean_continuity": float(path_mean_continuity),
            "path_bottleneck_similarity": float(path_bottleneck_similarity),
        }
        if extra_fields:
            node.update(extra_fields)
        nodes.append(node)
        return node

    root = add_node(
        item_index=root_index,
        depth=0,
        parent_id=None,
        path_indices=[root_index],
        sim_to_parent=None,
        near_duplicate_count=0,
        near_duplicate_penalty=0.0,
        score=1.0,
        path_edge_similarities=[],
        path_mean_continuity=1.0,
        path_bottleneck_similarity=1.0,
        extra_fields=extra_root_fields,
    )
    frontier: list[tuple[dict, list[int]]] = [(root, [root_index])]
    for depth in range(1, max_depth + 1):
        next_frontier: list[tuple[dict, list[int]]] = []
        for parent_node, path_indices in frontier:
            parent_index = int(parent_node["item_index"])
            scored_candidates = []
            for candidate_index in range(len(room_items)):
                if candidate_index in path_indices or candidate_index in excluded:
                    continue
                scored_candidates.append(
                    score_candidate(
                        candidate_index=candidate_index,
                        parent_index=parent_index,
                        path_indices=path_indices,
                        pairwise_similarities=pairwise_similarities,
                        current_similarities=np.zeros(len(room_items), dtype=np.float32),
                        near_duplicate_threshold=near_duplicate_threshold,
                        near_duplicate_penalty_weight=near_duplicate_penalty_weight,
                        parent_similarity_weight=parent_similarity_weight,
                        current_similarity_weight=0.0,
                    )
                )
            for candidate in scored_candidates:
                parent_edges = list(parent_node.get("path_edge_similarities") or [])
                candidate_edges = [*parent_edges, float(candidate["sim_to_parent"])]
                path_mean = mean_similarity(candidate_edges, default=float(candidate["sim_to_parent"]))
                path_bottleneck = min(candidate_edges) if candidate_edges else float(candidate["sim_to_parent"])
                candidate["path_edge_similarities"] = candidate_edges
                candidate["path_mean_continuity"] = float(path_mean)
                candidate["path_bottleneck_similarity"] = float(path_bottleneck)
                if tree_expansion_score_mode == "path_continuity":
                    candidate["score"] = float(path_mean) - float(candidate["near_duplicate_penalty"])
            scored_candidates.sort(
                key=lambda candidate: (
                    -candidate["score"],
                    -candidate.get("path_bottleneck_similarity", 0.0),
                    -candidate["sim_to_parent"],
                    memory_key(room_items[int(candidate["candidate_index"])]),
                )
            )
            for candidate_score in scored_candidates[:branching_factor]:
                candidate_index = int(candidate_score["candidate_index"])
                child_path = [*path_indices, candidate_index]
                child = add_node(
                    item_index=candidate_index,
                    depth=depth,
                    parent_id=parent_node["id"],
                    path_indices=child_path,
                    sim_to_parent=float(candidate_score["sim_to_parent"]),
                    near_duplicate_count=int(candidate_score["near_duplicate_count"]),
                    near_duplicate_penalty=float(candidate_score["near_duplicate_penalty"]),
                    score=float(candidate_score["score"]),
                    path_edge_similarities=list(candidate_score.get("path_edge_similarities") or []),
                    path_mean_continuity=float(candidate_score.get("path_mean_continuity", 1.0)),
                    path_bottleneck_similarity=float(candidate_score.get("path_bottleneck_similarity", 1.0)),
                )
                edges.append(
                    {
                        "source": parent_node["id"],
                        "target": child["id"],
                        "source_memory_key": parent_node["memory_key"],
                        "target_memory_key": child["memory_key"],
                        "sim_to_parent": child["sim_to_parent"],
                        "near_duplicate_penalty": child["near_duplicate_penalty"],
                        "score": child["score"],
                        "path_mean_continuity": child["path_mean_continuity"],
                        "path_bottleneck_similarity": child["path_bottleneck_similarity"],
                    }
                )
                next_frontier.append((child, child_path))
        frontier = next_frontier
        if not frontier:
            break
    return {"root": root, "nodes": nodes, "edges": edges}


def _view_alignment_sort_key(item: dict, *, bridge_similarity_tie_margin: float = 0.0) -> tuple:
    direct_match = bool(item.get("root_target_direct_match"))
    root_target_similarity = float(item.get("root_target_similarity", 0.0))
    direct_similarity = root_target_similarity if direct_match else 0.0
    bridge = item["best_bridge"]
    if bridge.get("bridge_selection_mode") == "bridge_then_continuity":
        bridge_similarity = float(bridge.get("bridge_similarity", 0.0))
        margin = max(float(bridge_similarity_tie_margin), 0.0)
        bridge_bucket = int(bridge_similarity / margin) if margin > 0.0 else bridge_similarity
        return (
            -int(direct_match),
            -direct_similarity,
            -bridge_bucket,
            -root_target_similarity,
            -bridge_similarity,
            -float(bridge.get("chain_bottleneck_similarity", 0.0)),
            -float(bridge.get("chain_mean_continuity", 0.0)),
            -float(bridge.get("current_chain_min_parent_similarity", 0.0)),
            -float(bridge.get("passage_chain_min_parent_similarity", 0.0)),
            int(bridge.get("total_bridge_depth", 0)),
            int(item["capture_index"]),
        )
    return (
        -int(direct_match),
        -direct_similarity,
        -float(bridge["total_score"]),
        -float(bridge["bridge_similarity"]),
        -float(bridge.get("current_chain_min_parent_similarity", 0.0)),
        -float(bridge.get("passage_chain_min_parent_similarity", 0.0)),
        int(item["capture_index"]),
    )


def choose_best_bridge(
    *,
    current_tree: dict,
    passage_tree: dict,
    current_leaves: Sequence[dict],
    passage_leaves: Sequence[dict],
    pairwise_similarities,
    bridge_depth_penalty: float,
    bridge_continuity_weight: float,
    bridge_selection_mode: str = "weighted",
    exclude_same_bridge_item: bool = True,
) -> dict:
    if bridge_selection_mode not in BRIDGE_SELECTION_MODES:
        raise ValueError(f"bridge_selection_mode must be one of {BRIDGE_SELECTION_MODES}.")
    best: dict | None = None
    skipped_same_item_count = 0
    for current_node in current_leaves:
        current_chain = node_path(current_tree, current_node["id"])
        current_edges = chain_parent_similarities(current_chain)
        current_min = min(current_edges) if current_edges else 1.0
        current_mean = mean_similarity(current_edges, default=1.0)
        for passage_node in passage_leaves:
            passage_chain = node_path(passage_tree, passage_node["id"])
            passage_edges = chain_parent_similarities(passage_chain)
            passage_min = min(passage_edges) if passage_edges else 1.0
            passage_mean = mean_similarity(passage_edges, default=1.0)
            same_bridge_item = int(current_node["item_index"]) == int(passage_node["item_index"])
            if exclude_same_bridge_item and same_bridge_item:
                skipped_same_item_count += 1
                continue
            bridge_similarity = float(
                pairwise_similarities[int(current_node["item_index"]), int(passage_node["item_index"])]
            )
            total_depth = int(current_node["depth"]) + int(passage_node["depth"])
            continuity_edges = [*current_edges, bridge_similarity, *passage_edges]
            chain_mean_continuity = mean_similarity(continuity_edges, default=bridge_similarity)
            chain_bottleneck_similarity = min(continuity_edges) if continuity_edges else bridge_similarity
            continuity_score = chain_mean_continuity
            bidirection_score = (
                bridge_similarity
                + float(bridge_continuity_weight) * (current_min + passage_min)
                - float(bridge_depth_penalty) * total_depth
            )
            total_score = (
                bridge_similarity
                if bridge_selection_mode == "bridge_then_continuity"
                else bidirection_score
            )
            candidate = {
                "current_node_id": current_node["id"],
                "passage_node_id": passage_node["id"],
                "current_memory_key": current_node["memory_key"],
                "passage_memory_key": passage_node["memory_key"],
                "bridge_similarity": bridge_similarity,
                "current_chain_min_parent_similarity": float(current_min),
                "passage_chain_min_parent_similarity": float(passage_min),
                "current_chain_mean_parent_similarity": float(current_mean),
                "passage_chain_mean_parent_similarity": float(passage_mean),
                "chain_mean_continuity": float(chain_mean_continuity),
                "chain_bottleneck_similarity": float(chain_bottleneck_similarity),
                "continuity_score": float(continuity_score),
                "continuity_edge_similarities": [float(value) for value in continuity_edges],
                "same_bridge_item": bool(same_bridge_item),
                "exclude_same_bridge_item": bool(exclude_same_bridge_item),
                "bridge_selection_mode": bridge_selection_mode,
                "bridge_score_mode": (
                    "bidirection" if bridge_selection_mode == "weighted" else "bridge_then_continuity"
                ),
                "bidirection_score": float(bidirection_score),
                "current_depth": int(current_node["depth"]),
                "passage_depth": int(passage_node["depth"]),
                "total_bridge_depth": total_depth,
                "total_score": float(total_score),
                "current_chain_node_ids": [node["id"] for node in current_chain],
                "passage_chain_node_ids": [node["id"] for node in passage_chain],
            }
            if best is None or bridge_sort_key(candidate) > bridge_sort_key(best):
                best = candidate
    if best is None and exclude_same_bridge_item and skipped_same_item_count > 0:
        fallback = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=current_leaves,
            passage_leaves=passage_leaves,
            pairwise_similarities=pairwise_similarities,
            bridge_depth_penalty=bridge_depth_penalty,
            bridge_continuity_weight=bridge_continuity_weight,
            bridge_selection_mode=bridge_selection_mode,
            exclude_same_bridge_item=False,
        )
        fallback["same_bridge_item_fallback"] = True
        fallback["skipped_same_item_bridge_count"] = int(skipped_same_item_count)
        return fallback
    if best is None:
        raise RuntimeError("No bridge candidates were produced.")
    best["same_bridge_item_fallback"] = False
    best["skipped_same_item_bridge_count"] = int(skipped_same_item_count)
    return best


def bridge_sort_key(candidate: dict) -> tuple:
    if candidate.get("bridge_selection_mode") == "bridge_then_continuity":
        return (
            float(candidate["bridge_similarity"]),
            float(candidate.get("chain_bottleneck_similarity", 0.0)),
            float(candidate.get("chain_mean_continuity", 0.0)),
            float(candidate["current_chain_min_parent_similarity"]),
            float(candidate["passage_chain_min_parent_similarity"]),
            -int(candidate["total_bridge_depth"]),
            str(candidate["current_memory_key"]),
            str(candidate["passage_memory_key"]),
        )
    return (
        float(candidate["total_score"]),
        float(candidate["bridge_similarity"]),
        float(candidate["current_chain_min_parent_similarity"]),
        float(candidate["passage_chain_min_parent_similarity"]),
        -int(candidate["total_bridge_depth"]),
        str(candidate["current_memory_key"]),
        str(candidate["passage_memory_key"]),
    )


def bridge_scoring_formula(bridge_selection_mode: str) -> str:
    if bridge_selection_mode == "bridge_then_continuity":
        return (
            "bridge_similarity first; tie-break by chain_bottleneck_similarity, "
            "chain_mean_continuity, side minimum continuity, then shorter total_bridge_depth"
        )
    return (
        "bridge_similarity + bridge_continuity_weight * "
        "(current_chain_min_parent_similarity + passage_chain_min_parent_similarity) "
        "- bridge_depth_penalty * total_bridge_depth"
    )


def chain_parent_similarities(chain: Sequence[dict]) -> list[float]:
    return [
        float(node["sim_to_parent"])
        for node in chain[1:]
        if node.get("sim_to_parent") is not None
    ]


def mean_similarity(values: Sequence[float], *, default: float) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else float(default)


def leaf_nodes(tree: dict) -> list[dict]:
    parents = {edge["source"] for edge in tree.get("edges", [])}
    leaves = [node for node in tree.get("nodes", []) if node["id"] not in parents]
    return leaves or list(tree.get("nodes", []))


def node_path(tree: dict, node_id: str) -> list[dict]:
    by_id = {node["id"]: node for node in tree.get("nodes", [])}
    path = []
    cursor = by_id[node_id]
    while cursor is not None:
        path.append(cursor)
        parent_id = cursor.get("parent_id")
        cursor = by_id.get(parent_id) if parent_id else None
    return list(reversed(path))


def chain_min_parent_similarity(tree: dict, node_id: str) -> float:
    path = node_path(tree, node_id)
    values = [float(node["sim_to_parent"]) for node in path[1:] if node.get("sim_to_parent") is not None]
    return min(values) if values else 1.0


def build_best_alignment_payload(result: dict) -> dict:
    selected = result["selected_alignment"]
    if "memory_chain_current_to_target" in selected:
        return {
            "method": result["method"],
            "configuration": result["configuration"],
            "current_view": result["current_view"],
            "selected_view": result["selected_view"],
            "selection_reason": result.get("selection_reason"),
            "target_found": bool(selected.get("target_found")),
            "chain_score": float(selected.get("chain_score", 0.0)),
            "bottleneck_similarity": float(selected.get("bottleneck_similarity", 0.0)),
            "mean_chain_similarity": float(selected.get("mean_chain_similarity", 0.0)),
            "path_length": int(selected.get("path_length", 0)),
            "memory_chain_current_to_target": selected["memory_chain_current_to_target"],
        }
    return {
        "method": result["method"],
        "configuration": result["configuration"],
        "current_view": result["current_view"],
        "selected_view": result["selected_view"],
        "best_bridge": selected["best_bridge"],
        "current_chain_root_to_bridge": selected["current_chain_root_to_bridge"],
        "passage_chain_bridge_to_target": selected["passage_chain_bridge_to_target"],
    }


def build_current_to_target_metrics(result: dict, *, pano_graph: dict | None) -> dict:
    del pano_graph
    selected = result["selected_alignment"]
    return {
        "selected_capture_index": int(result["selected_view"].get("capture_index") or 0),
        "selected_view_score": float(result["selected_view"].get("view_score", result["selected_view"].get("chain_score", 0.0))),
        "selection_reason": result.get("selection_reason"),
        "selected_target_found": bool(selected.get("target_found")),
        "selected_chain_score": float(selected.get("chain_score", 0.0)),
        "selected_bottleneck_similarity": float(selected.get("bottleneck_similarity", 0.0)),
        "selected_mean_chain_similarity": float(selected.get("mean_chain_similarity", 0.0)),
        "selected_target_similarity": float(selected.get("target_similarity", 0.0)),
        "selected_root_target_similarity": float(selected.get("root_target_similarity", 0.0)),
        "selected_root_target_direct_match": bool(selected.get("root_target_direct_match", False)),
        "selected_path_length": int(selected.get("path_length", 0)),
        "view_count": len(result.get("view_alignments", [])),
        "view_rankings": [
            {
                "rank": item["rank"],
                "selected": item["selected"],
                "capture_index": item["capture_index"],
                "target_found": bool(item.get("target_found", False)),
                "view_score": float(item.get("view_score", item.get("chain_score", 0.0))),
                "chain_score": float(item.get("chain_score", 0.0)),
                "bottleneck_similarity": float(item.get("bottleneck_similarity", 0.0)),
                "mean_chain_similarity": float(item.get("mean_chain_similarity", 0.0)),
                "target_similarity": float(item.get("target_similarity", 0.0)),
                "root_target_similarity": float(item.get("root_target_similarity", 0.0)),
                "root_target_direct_match": bool(item.get("root_target_direct_match", False)),
                "path_length": int(item.get("path_length", 0)),
                "selection_reason": item.get("selection_reason"),
            }
            for item in result.get("view_alignments", [])
        ],
        "selected_chain_length": len(selected.get("memory_chain_current_to_target", [])),
    }

def build_bidirectional_metrics(result: dict, *, pano_graph: dict | None) -> dict:
    del pano_graph
    selected = result["selected_alignment"]
    return {
        "selected_capture_index": int(result["selected_view"].get("capture_index") or 0),
        "selected_view_score": float(result["selected_view"].get("view_score", result["selected_view"]["total_score"])),
        "selected_root_target_similarity": float(result["selected_view"].get("root_target_similarity", 0.0)),
        "selected_root_target_direct_match": bool(result["selected_view"].get("root_target_direct_match", False)),
        "selection_reason": result.get("selection_reason"),
        "selected_total_score": float(result["selected_view"]["total_score"]),
        "selected_bridge_similarity": float(result["selected_view"]["bridge_similarity"]),
        "selected_continuity_score": float(selected["best_bridge"].get("continuity_score", 0.0)),
        "selected_chain_mean_continuity": float(selected["best_bridge"].get("chain_mean_continuity", 0.0)),
        "selected_chain_bottleneck_similarity": float(selected["best_bridge"].get("chain_bottleneck_similarity", 0.0)),
        "view_count": len(result.get("view_alignments", [])),
        "view_rankings": [
            {
                "rank": item["rank"],
                "selected": item["selected"],
                "capture_index": item["capture_index"],
                "view_score": float(item.get("view_score", item["best_bridge"]["total_score"])),
                "root_target_similarity": float(item.get("root_target_similarity", 0.0)),
                "root_target_direct_match": bool(item.get("root_target_direct_match", False)),
                "total_score": float(item["best_bridge"]["total_score"]),
                "bridge_similarity": float(item["best_bridge"]["bridge_similarity"]),
                "continuity_score": float(item["best_bridge"].get("continuity_score", 0.0)),
                "chain_mean_continuity": float(item["best_bridge"].get("chain_mean_continuity", 0.0)),
                "chain_bottleneck_similarity": float(item["best_bridge"].get("chain_bottleneck_similarity", 0.0)),
                "current_chain_min_parent_similarity": float(
                    item["best_bridge"]["current_chain_min_parent_similarity"]
                ),
                "passage_chain_min_parent_similarity": float(
                    item["best_bridge"]["passage_chain_min_parent_similarity"]
                ),
                "current_chain_mean_parent_similarity": float(
                    item["best_bridge"].get("current_chain_mean_parent_similarity", 0.0)
                ),
                "passage_chain_mean_parent_similarity": float(
                    item["best_bridge"].get("passage_chain_mean_parent_similarity", 0.0)
                ),
            }
            for item in result.get("view_alignments", [])
        ],
        "selected_current_chain_length": len(selected["current_chain_root_to_bridge"]),
        "selected_passage_chain_length": len(selected["passage_chain_bridge_to_target"]),
    }

def build_metrics(result: dict, *, pano_graph: dict | None) -> dict:
    chain = result["navigation_chain_current_to_target"]
    target = chain[-1] if chain else None
    target_pano_id = target.get("pano_id") if isinstance(target, dict) else None
    graph_hops = []
    if pano_graph and isinstance(target_pano_id, str):
        for node in chain:
            source_pano_id = node.get("pano_id")
            hops = (
                shortest_pano_hops(pano_graph, str(source_pano_id), target_pano_id)
                if isinstance(source_pano_id, str)
                else None
            )
            graph_hops.append(
                {
                    "node_id": node["id"],
                    "memory_key": node["memory_key"],
                    "pano_id": source_pano_id,
                    "hops_to_target_pano": hops,
                }
            )
    hop_values = [
        item["hops_to_target_pano"]
        for item in graph_hops
        if isinstance(item.get("hops_to_target_pano"), int)
    ]
    monotonic = all(
        int(left) >= int(right)
        for left, right in zip(hop_values, hop_values[1:], strict=False)
    )
    direct_similarity = (
        float(result["nodes"][0]["sim_to_current"])
        if result.get("nodes")
        else None
    )
    stop_similarity = float(result["stop"]["sim_to_current"])
    return {
        "stop_found": bool(result["stop"]["found"]),
        "stop_reason": result["stop"]["reason"],
        "chain_memory_node_count": len(chain),
        "direct_p0_to_current_similarity": direct_similarity,
        "tree_stop_similarity": stop_similarity,
        "similarity_improvement_over_direct": (
            stop_similarity - direct_similarity if direct_similarity is not None else None
        ),
        "graph_hops_to_target_pano": graph_hops,
        "graph_hops_monotonic_nonincreasing": monotonic if graph_hops else None,
    }


def shortest_pano_hops(pano_graph: dict, source_pano_id: str, target_pano_id: str) -> int | None:
    if source_pano_id == target_pano_id:
        return 0
    if source_pano_id not in pano_graph or target_pano_id not in pano_graph:
        return None
    queue: deque[tuple[str, int]] = deque([(source_pano_id, 0)])
    visited = {source_pano_id}
    while queue:
        pano_id, distance = queue.popleft()
        for edge in pano_graph.get(pano_id, {}).get("neighbors", []):
            if not isinstance(edge, dict):
                continue
            neighbor = edge.get("target_pano_id")
            if not isinstance(neighbor, str) or neighbor in visited:
                continue
            if neighbor == target_pano_id:
                return distance + 1
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def current_nearest_neighbors(
    *,
    room_items: Sequence[dict],
    current_similarities,
    limit: int,
) -> list[dict]:
    ranked = sorted(
        range(len(room_items)),
        key=lambda index: (
            -float(current_similarities[index]),
            memory_key(room_items[index]),
        ),
    )
    neighbors = []
    for rank, index in enumerate(ranked[: max(int(limit), 1)], start=1):
        neighbors.append(
            {
                "rank": rank,
                "similarity": float(current_similarities[index]),
                **public_memory_item(room_items[index]),
            }
        )
    return neighbors


def rank_by_current_similarity(current_similarities, room_items: Sequence[dict]) -> dict[str, int]:
    ranked = sorted(
        range(len(room_items)),
        key=lambda index: (
            -float(current_similarities[index]),
            memory_key(room_items[index]),
        ),
    )
    return {
        memory_key(room_items[index]): rank
        for rank, index in enumerate(ranked, start=1)
    }



def load_or_encode_dinov2_patch_features(
    *,
    image_paths: Sequence[Path],
    memory_keys: Sequence[str],
    output_dir: Path,
    model_name: str,
    max_patches: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if len(image_paths) != len(memory_keys):
        raise ValueError("image_paths and memory_keys must have the same length.")
    cache_path = (
        output_dir
        / "embedding_cache"
        / f"dinov2_patch_topk_{safe_name(model_name)}_p{int(max_patches)}.npz"
    )
    cached_records = load_dinov2_patch_cache(
        cache_path,
        model_name=model_name,
        max_patches=max_patches,
    )
    requested_features: list[np.ndarray | None] = [None] * len(image_paths)
    missing_indices: list[int] = []
    records = dict(cached_records)

    for index, (image_path, key) in enumerate(zip(image_paths, memory_keys, strict=True)):
        resolved_path = Path(image_path).resolve()
        signature = dinov2_patch_file_signature(resolved_path)
        record_key = dinov2_patch_record_key(key, resolved_path)
        record = records.get(record_key)
        if record and record.get("file_signature") == signature:
            requested_features[index] = np.asarray(record["patch_features"], dtype=np.float32)
        else:
            missing_indices.append(index)

    if missing_indices:
        encoded = encode_dinov2_patch_paths(
            [Path(image_paths[index]) for index in missing_indices],
            model_name=model_name,
            max_patches=max_patches,
            device=device,
            batch_size=batch_size,
        )
        for offset, index in enumerate(missing_indices):
            resolved_path = Path(image_paths[index]).resolve()
            key = str(memory_keys[index])
            signature = dinov2_patch_file_signature(resolved_path)
            features = np.asarray(encoded[offset], dtype=np.float32)
            requested_features[index] = features
            records[dinov2_patch_record_key(key, resolved_path)] = {
                "memory_key": key,
                "image_path": str(resolved_path),
                "file_signature": signature,
                "patch_features": features,
            }
        save_dinov2_patch_cache(
            cache_path,
            model_name=model_name,
            max_patches=max_patches,
            records=records,
        )

    if any(features is None for features in requested_features):
        raise RuntimeError("Failed to load or encode all DINOv2 patch features.")
    return np.stack([np.asarray(features, dtype=np.float32) for features in requested_features], axis=0)


def load_dinov2_patch_cache(
    path: Path,
    *,
    model_name: str,
    max_patches: int,
) -> dict[str, dict]:
    if not path.exists():
        return {}
    resolved_path = path.resolve()
    cache_key = (str(resolved_path), str(model_name), int(max_patches))
    path_stat = resolved_path.stat()
    path_signature = (int(path_stat.st_mtime_ns), int(path_stat.st_size))
    memory_entry = _DINOV2_PATCH_RECORD_MEMORY_CACHE.get(cache_key)
    if memory_entry is not None and memory_entry[0] == path_signature:
        return memory_entry[1]
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception:
        return {}
    try:
        cached_model = str(payload["model_name"].tolist())
        cached_max_patches = int(payload["max_patches"].tolist())
        if cached_model != str(model_name) or cached_max_patches != int(max_patches):
            return {}
        required_fields = {"memory_keys", "image_paths", "file_signatures", "patch_features"}
        if not required_fields.issubset(set(payload.files)):
            return {}
        memory_keys = [str(value) for value in payload["memory_keys"].tolist()]
        image_paths = [str(value) for value in payload["image_paths"].tolist()]
        signatures = [str(value) for value in payload["file_signatures"].tolist()]
        patch_features = np.asarray(payload["patch_features"], dtype=np.float32)
    except Exception:
        return {}
    finally:
        payload.close()
    if not (len(memory_keys) == len(image_paths) == len(signatures) == int(patch_features.shape[0])):
        return {}
    records: dict[str, dict] = {}
    for key, image_path, signature, features in zip(
        memory_keys,
        image_paths,
        signatures,
        patch_features,
        strict=True,
    ):
        records[dinov2_patch_record_key(key, Path(image_path))] = {
            "memory_key": key,
            "image_path": image_path,
            "file_signature": signature,
            "patch_features": np.asarray(features, dtype=np.float32),
        }
    _DINOV2_PATCH_RECORD_MEMORY_CACHE[cache_key] = (path_signature, records)
    return records


def save_dinov2_patch_cache(
    path: Path,
    *,
    model_name: str,
    max_patches: int,
    records: dict[str, dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_records = sorted(
        records.values(),
        key=lambda record: (str(record["memory_key"]), str(record["image_path"])),
    )
    if ordered_records:
        patch_features = np.stack(
            [np.asarray(record["patch_features"], dtype=np.float32) for record in ordered_records],
            axis=0,
        )
    else:
        patch_features = np.zeros((0, 0, 0), dtype=np.float32)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                memory_keys=np.asarray(
                    [str(record["memory_key"]) for record in ordered_records]
                ),
                image_paths=np.asarray(
                    [str(record["image_path"]) for record in ordered_records]
                ),
                file_signatures=np.asarray(
                    [str(record["file_signature"]) for record in ordered_records]
                ),
                model_name=np.asarray(str(model_name)),
                max_patches=np.asarray(int(max_patches)),
                patch_features=np.asarray(patch_features, dtype=np.float32),
            )
        temporary_path.replace(path)
        resolved_path = path.resolve()
        path_stat = resolved_path.stat()
        cache_key = (str(resolved_path), str(model_name), int(max_patches))
        path_signature = (int(path_stat.st_mtime_ns), int(path_stat.st_size))
        _DINOV2_PATCH_RECORD_MEMORY_CACHE[cache_key] = (
            path_signature,
            dict(records),
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def dinov2_patch_record_key(memory_key_value: object, image_path: Path) -> str:
    del memory_key_value
    return str(Path(image_path).resolve())


def dinov2_patch_file_signature(image_path: Path) -> str:
    stat = Path(image_path).stat()
    return f"{int(stat.st_mtime_ns)}:{int(stat.st_size)}"


def encode_dinov2_patch_paths(
    paths: Sequence[Path],
    *,
    model_name: str,
    max_patches: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = resolve_torch_device(torch, device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(resolved_device)
    batches = []
    step = max(int(batch_size), 1)
    for start in range(0, len(paths), step):
        batch_paths = [Path(path) for path in paths[start : start + step]]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in dict(inputs).items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError(f"DINOv2 model {model_name!r} did not return last_hidden_state.")
            patches = hidden[:, 1:, :]
            patches = patches / patches.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            patch_array = patches.detach().cpu().to(torch.float32).numpy()
            batches.append(select_salient_patch_features(patch_array, max_patches=max_patches))
        finally:
            for image in images:
                image.close()
    if not batches:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def resolve_torch_device(torch_module, requested_device: str) -> str:
    normalized = (requested_device or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def select_salient_patch_features(features: np.ndarray, *, max_patches: int) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("features must have shape (image, patch, dim).")
    limit = max(int(max_patches), 1)
    if array.shape[1] <= limit:
        return array
    selected = []
    for image_features in array:
        image_features = normalize_rows(image_features)
        mean_feature = normalize_rows(image_features.mean(axis=0, keepdims=True))[0]
        saliency = 1.0 - (image_features @ mean_feature)
        indices = np.argsort(-saliency, kind="stable")[:limit]
        selected.append(image_features[indices])
    return np.stack(selected, axis=0).astype(np.float32)


def patch_topk_similarity_matrix(patch_features: np.ndarray, *, top_k: int) -> np.ndarray:
    features = np.asarray(patch_features, dtype=np.float32)
    count = int(features.shape[0])
    matrix = np.eye(count, dtype=np.float32)
    for source in range(count):
        for target in range(source + 1, count):
            score = patch_topk_similarity(features[source], features[target], top_k=top_k)
            matrix[source, target] = score
            matrix[target, source] = score
    return matrix


def load_or_compute_dinov2_patch_similarity_matrix(
    *,
    patch_features: np.ndarray,
    image_paths: Sequence[Path],
    memory_keys: Sequence[str],
    output_dir: Path,
    model_name: str,
    max_patches: int,
    top_k: int,
) -> np.ndarray:
    signatures = [dinov2_patch_file_signature(Path(path).resolve()) for path in image_paths]
    resolved_paths = [str(Path(path).resolve()) for path in image_paths]
    request_digest = dinov2_patch_similarity_request_digest(
        memory_keys=memory_keys,
        image_paths=resolved_paths,
        file_signatures=signatures,
        model_name=model_name,
        max_patches=max_patches,
        top_k=top_k,
    )
    cache_path = (
        output_dir
        / "embedding_cache"
        / f"dinov2_patch_similarity_{safe_name(model_name)}_p{int(max_patches)}_k{int(top_k)}_{request_digest}.npz"
    )
    cached = load_dinov2_patch_similarity_cache(
        cache_path,
        memory_keys=memory_keys,
        image_paths=resolved_paths,
        file_signatures=signatures,
        model_name=model_name,
        max_patches=max_patches,
        top_k=top_k,
    )
    if cached is not None:
        return cached
    matrix = patch_topk_similarity_matrix(patch_features, top_k=top_k)
    save_dinov2_patch_similarity_cache(
        cache_path,
        memory_keys=memory_keys,
        image_paths=resolved_paths,
        file_signatures=signatures,
        model_name=model_name,
        max_patches=max_patches,
        top_k=top_k,
        pairwise_similarities=matrix,
    )
    return matrix



def dinov2_patch_similarity_request_digest(
    *,
    memory_keys: Sequence[str],
    image_paths: Sequence[str],
    file_signatures: Sequence[str],
    model_name: str,
    max_patches: int,
    top_k: int,
) -> str:
    digest = hashlib.sha1()
    digest.update(str(model_name).encode("utf-8"))
    digest.update(f"|p{int(max_patches)}|k{int(top_k)}".encode("utf-8"))
    for key, image_path, signature in zip(memory_keys, image_paths, file_signatures, strict=True):
        digest.update(b"\0")
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(image_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(signature).encode("utf-8"))
    return digest.hexdigest()[:16]


def load_dinov2_patch_similarity_cache(
    path: Path,
    *,
    memory_keys: Sequence[str],
    image_paths: Sequence[str],
    file_signatures: Sequence[str],
    model_name: str,
    max_patches: int,
    top_k: int,
) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        payload = np.load(path, allow_pickle=False)
    except Exception:
        return None
    try:
        if str(payload["model_name"].tolist()) != str(model_name):
            return None
        if int(payload["max_patches"].tolist()) != int(max_patches):
            return None
        if int(payload["top_k"].tolist()) != int(top_k):
            return None
        cached_keys = [str(value) for value in payload["memory_keys"].tolist()]
        cached_paths = [str(value) for value in payload["image_paths"].tolist()]
        cached_signatures = [str(value) for value in payload["file_signatures"].tolist()]
        if (
            cached_keys != [str(value) for value in memory_keys]
            or cached_paths != [str(value) for value in image_paths]
            or cached_signatures != [str(value) for value in file_signatures]
        ):
            return None
        matrix = np.asarray(payload["pairwise_similarities"], dtype=np.float32)
    finally:
        payload.close()
    expected = len(memory_keys)
    if matrix.shape != (expected, expected):
        return None
    return matrix


def save_dinov2_patch_similarity_cache(
    path: Path,
    *,
    memory_keys: Sequence[str],
    image_paths: Sequence[str],
    file_signatures: Sequence[str],
    model_name: str,
    max_patches: int,
    top_k: int,
    pairwise_similarities: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_keys=np.asarray([str(value) for value in memory_keys]),
        image_paths=np.asarray([str(value) for value in image_paths]),
        file_signatures=np.asarray([str(value) for value in file_signatures]),
        model_name=np.asarray(str(model_name)),
        max_patches=np.asarray(int(max_patches)),
        top_k=np.asarray(int(top_k)),
        pairwise_similarities=np.asarray(pairwise_similarities, dtype=np.float32),
    )


def patch_topk_similarity(source_features: np.ndarray, target_features: np.ndarray, *, top_k: int) -> float:
    source = normalize_rows(np.asarray(source_features, dtype=np.float32))
    target = normalize_rows(np.asarray(target_features, dtype=np.float32))
    if source.size == 0 or target.size == 0:
        return 0.0
    similarities = source @ target.T
    k_source = min(max(int(top_k), 1), int(similarities.shape[0]))
    k_target = min(max(int(top_k), 1), int(similarities.shape[1]))
    source_best = similarities.max(axis=1)
    target_best = similarities.max(axis=0)
    source_score = topk_mean(source_best, k_source)
    target_score = topk_mean(target_best, k_target)
    return float((source_score + target_score) / 2.0)


def topk_mean(values: np.ndarray, k: int) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return 0.0
    limit = min(max(int(k), 1), int(array.size))
    if limit == int(array.size):
        return float(array.mean())
    top_values = np.partition(array, -limit)[-limit:]
    return float(top_values.mean())

def normalize_rows(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def load_memory_metadata(metadata_path: Path, render_root: Path) -> list[dict]:
    payload = load_json(metadata_path)
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Metadata contains no items: {metadata_path}")
    resolved = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image_path = resolve_capture_path(item, render_root)
        if image_path is None or not image_path.exists():
            continue
        normalized = dict(item)
        normalized["image_path"] = str(image_path)
        resolved.append(normalized)
    return resolved


def load_target_representative(*, representatives_path: Path, target_passage_label: str) -> dict:
    payload = load_json(representatives_path)
    representatives = payload.get("representatives")
    if not isinstance(representatives, list):
        raise RuntimeError(f"Representatives JSON is missing representatives: {representatives_path}")
    for representative in representatives:
        if isinstance(representative, dict) and representative.get("label") == target_passage_label:
            return dict(representative)
    raise RuntimeError(
        f"Target passage label {target_passage_label!r} was not found in {representatives_path}"
    )


def find_matching_item_index(items: Sequence[dict], query: dict) -> int | None:
    image_path = query.get("image_path") or query.get("capture_path")
    if isinstance(image_path, str) and image_path:
        target_path = resolve_project_path(image_path)
        for index, item in enumerate(items):
            raw_item_path = item.get("image_path") or item.get("capture_path")
            if not isinstance(raw_item_path, str) or not raw_item_path:
                continue
            if resolve_project_path(raw_item_path) == target_path:
                return index
    memory_index = query.get("memory_index")
    if isinstance(memory_index, int):
        for index, item in enumerate(items):
            if item.get("memory_index") == memory_index:
                return index
    pano_id = query.get("pano_id")
    capture_index = query.get("capture_index")
    for index, item in enumerate(items):
        if item.get("pano_id") == pano_id and item.get("capture_index") == capture_index:
            return index
    return None


def index_for_memory_key(items: Sequence[dict], key: str) -> int:
    for index, item in enumerate(items):
        if memory_key(item) == key:
            return index
    raise KeyError(key)


def public_memory_item(item: dict) -> dict:
    payload = {
        "memory_index": item.get("memory_index"),
        "room_id": item.get("room_id"),
        "pano_id": item.get("pano_id"),
        "capture_index": item.get("capture_index"),
        "capture_label": item.get("capture_label"),
        "capture_heading": item.get("capture_heading"),
        "image_path": item.get("image_path") or item.get("capture_path"),
    }
    if item.get("external_image"):
        payload["external_image"] = True
    return payload


def memory_key(item: dict) -> str:
    return f"{item.get('pano_id')}:{item.get('capture_index')}"


def external_target_item(
    image_path: Path,
    *,
    room_id: str,
    target_passage_label: str,
) -> dict:
    return {
        "memory_index": None,
        "room_id": room_id,
        "pano_id": f"external_{safe_name(image_path.stem)}",
        "capture_index": 0,
        "capture_label": target_passage_label,
        "capture_heading": None,
        "image_path": str(image_path.resolve()),
        "external_image": True,
    }


def resolve_capture_path(item: dict, render_root: Path) -> Path | None:
    raw_path = item.get("capture_path") or item.get("image_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    original = Path(raw_path)
    if original.exists():
        return original.resolve()
    pano_id = item.get("pano_id")
    if isinstance(pano_id, str):
        candidate = render_root / pano_id / original.name
        if candidate.exists():
            return candidate.resolve()
    return None


def default_representatives_path(room_id: str) -> Path:
    return resolve_project_path(
        f"outputs/passage_clustering/{room_token(room_id)}/dreamsim_cluster8/representatives.json"
    )


def default_output_dir(
    *,
    room_id: str,
    target_passage_label: str,
    current_image: str | None,
    current_pano_id: str | None,
    current_capture_index: int | None,
) -> Path:
    if current_image:
        current_token = safe_name(Path(current_image).stem)
    elif current_capture_index is None:
        current_token = f"{safe_name(current_pano_id)}_8views"
    else:
        current_token = f"{safe_name(current_pano_id)}_{current_capture_index}"
    return resolve_project_path(
        Path("outputs")
        / "passage_memory_tree"
        / f"{room_token(room_id)}_{safe_name(target_passage_label)}_{current_token}"
    )


def room_token(room_id: str) -> str:
    return safe_name(str(room_id).lower().replace(" ", ""))


def safe_name(value: object) -> str:
    text = str(value or "").strip()
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "item"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_optional_pano_graph(artifacts_dir: Path) -> dict | None:
    path = artifacts_dir / "pano_graph.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_chain_contact_sheet(path: Path, result: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    chain = [
        {"kind": "current", **result["current_view"]},
        *[
            {"kind": "memory", **node}
            for node in result["navigation_chain_current_to_target"]
        ],
    ]
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    card_w, card_h = 340, 292
    thumb_w, thumb_h = 320, 210
    margin, title_h = 24, 64
    columns = min(4, max(len(chain), 1))
    rows = (len(chain) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (margin * 2 + columns * card_w, title_h + margin + rows * card_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 18), "Passage Memory Tree Best Chain", font=title_font, fill="black")
    for index, item in enumerate(chain):
        row, col = divmod(index, columns)
        x = margin + col * card_w
        y = title_h + row * card_h
        image_path = item.get("image_path")
        paste_thumbnail(sheet, draw, image_path, x, y, thumb_w, thumb_h, font)
        if item["kind"] == "current":
            label = f"Current view mode={item.get('mode')}"
        else:
            label = (
                f"{item['id']} rank={item['top_current_rank']} "
                f"sim_current={float(item['sim_to_current']):.3f} "
                f"sim_parent={item.get('sim_to_parent')}"
            )
        draw_wrapped_text(draw, (x, y + thumb_h + 8), label, font, max_width=thumb_w)
    sheet.save(path)


def write_tree_gallery(output_dir: Path, result: dict) -> None:
    gallery_path = output_dir / "tree_gallery.html"
    asset_dir = output_dir / "tree_gallery_assets"
    asset_map = copy_gallery_images(asset_dir, result)
    edge_by_target = {edge["target"]: edge for edge in result.get("edges", [])}
    chain_ids = {node["id"] for node in result.get("navigation_chain_current_to_target", [])}
    stop_id = result.get("stop", {}).get("node_id")
    target_id = result.get("target_node_id")

    depth_groups: dict[int, list[dict]] = {}
    for node in result["nodes"]:
        depth_groups.setdefault(int(node["depth"]), []).append(node)

    chain_cards = [current_view_card(result.get("current_view", {}), asset_map.get("current"))]
    chain_cards.extend(
        node_card(
            node,
            image_src=asset_map.get(node["id"]),
            parent_edge=edge_by_target.get(node["id"]),
            is_chain=True,
            is_stop=node["id"] == stop_id,
            is_target=node["id"] == target_id,
        )
        for node in result.get("navigation_chain_current_to_target", [])
    )

    sections = [
        "<section class='panel'>",
        "<h2>Best Chain: Current -> Target Passage</h2>",
        "<div class='chain'>" + "<span class='arrow'>→</span>".join(chain_cards) + "</div>",
        "</section>",
    ]
    for depth in sorted(depth_groups):
        cards = "\n".join(
            node_card(
                node,
                image_src=asset_map.get(node["id"]),
                parent_edge=edge_by_target.get(node["id"]),
                is_chain=node["id"] in chain_ids,
                is_stop=node["id"] == stop_id,
                is_target=node["id"] == target_id,
            )
            for node in depth_groups[depth]
        )
        sections.append(
            "\n".join(
                [
                    "<section class='panel'>",
                    f"<h2>Depth {depth}</h2>",
                    f"<div class='grid'>{cards}</div>",
                    "</section>",
                ]
            )
        )

    config = result.get("configuration", {})
    stop = result.get("stop", {})
    document = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Passage Memory Tree</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8fafc;color:#111827}",
            ".summary,.panel{background:white;border:1px solid #d1d5db;border-radius:8px;padding:16px;margin-bottom:18px}",
            ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}",
            ".metric{background:#f3f4f6;border-radius:6px;padding:10px}",
            ".metric b{display:block;font-size:12px;color:#4b5563;text-transform:uppercase;letter-spacing:.04em}",
            ".metric span{font-size:18px;font-weight:700}",
            ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}",
            ".chain{display:flex;gap:12px;align-items:stretch;overflow-x:auto;padding-bottom:8px}",
            ".chain .card,.chain .current-card{min-width:280px;max-width:280px}",
            ".arrow{align-self:center;font-size:28px;color:#6b7280}",
            ".card,.current-card{background:white;border:1px solid #d1d5db;border-radius:6px;padding:12px;box-shadow:0 1px 2px rgba(15,23,42,.06)}",
            ".card.chain-node{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.12)}",
            ".card.stop-node{border-color:#16a34a;box-shadow:0 0 0 2px rgba(22,163,74,.14)}",
            ".card.target-node{border-color:#dc2626}",
            "img{width:100%;aspect-ratio:16/10;object-fit:contain;background:#f9fafb;border:1px solid #e5e7eb;border-radius:4px}",
            ".missing{height:160px;display:grid;place-items:center;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:4px;color:#6b7280}",
            "h1{margin-top:0} h2{margin:0 0 12px} h3{font-size:16px;margin:10px 0 6px;line-height:1.25}",
            "p{margin:4px 0}.small{font-size:12px;color:#4b5563}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}",
            ".pills{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.pill{font-size:12px;font-weight:700;border-radius:999px;padding:3px 8px;background:#e5e7eb;color:#374151}",
            ".pill.root{background:#fee2e2;color:#991b1b}.pill.stop{background:#dcfce7;color:#166534}.pill.chain{background:#dbeafe;color:#1d4ed8}",
            ".scores{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.scores div{background:#f8fafc;border-radius:4px;padding:6px}",
            "</style></head><body>",
            "<h1>Passage Memory Tree</h1>",
            "<section class='summary'>",
            "<div class='summary-grid'>",
            metric_html("Stop reason", stop.get("reason")),
            metric_html("Stop similarity", f"{float(stop.get('sim_to_current') or 0.0):.3f}"),
            metric_html("Stop rank", stop.get("top_current_rank")),
            metric_html("Score formula", config.get("scoring_formula")),
            metric_html("Parent weight", config.get("parent_similarity_weight")),
            metric_html("Current weight", config.get("current_similarity_weight")),
            "</div>",
            "<p class='small'>Tree expansion is puzzle-style when current weight is 0: candidates are ranked by parent continuity, while current similarity only decides whether the tree has reached the current view neighborhood.</p>",
            "</section>",
            "\n".join(sections),
            "</body></html>",
        ]
    )
    gallery_path.write_text(document, encoding="utf-8")
    embedded_document = embed_document_asset_refs(document, asset_dir, asset_map)
    (output_dir / "tree_gallery_embedded.html").write_text(
        embedded_document, encoding="utf-8"
    )



def write_current_to_target_alignment_gallery(output_dir: Path, result: dict) -> None:
    gallery_path = output_dir / "alignment_gallery.html"
    asset_dir = output_dir / "alignment_gallery_assets"
    image_items = current_to_target_gallery_image_items(result)
    asset_map = copy_gallery_images(asset_dir, result, image_items=image_items)
    selected = result["selected_alignment"]
    config = result.get("configuration", {})

    ranking_rows = []
    for item in result.get("view_alignments", []):
        mark = "selected" if item.get("selected") else ""
        ranking_rows.append(
            "".join(
                [
                    f"<tr class='{html.escape(mark)}'>",
                    f"<td>{item.get('rank')}</td>",
                    f"<td>{item.get('capture_index')}</td>",
                    f"<td>{html.escape(str(item.get('target_found')))}</td>",
                    f"<td>{format_score(item.get('chain_score'))}</td>",
                    f"<td>{format_score(item.get('bottleneck_similarity'))}</td>",
                    f"<td>{format_score(item.get('mean_chain_similarity'))}</td>",
                    f"<td>{format_score(item.get('target_similarity'))}</td>",
                    f"<td>{item.get('path_length')}</td>",
                    "</tr>",
                ]
            )
        )

    view_panels = []
    for item in result.get("view_alignments", []):
        view_panels.append(current_to_target_view_panel(item, asset_map))

    document = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Current-to-Target Passage Chain</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8fafc;color:#111827}",
            ".summary,.panel{background:white;border:1px solid #d1d5db;border-radius:8px;padding:16px;margin-bottom:18px}",
            ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}",
            ".metric{background:#f3f4f6;border-radius:6px;padding:10px;min-height:58px}",
            ".metric b{display:block;font-size:12px;color:#4b5563;text-transform:uppercase;letter-spacing:.04em}",
            ".metric span{font-size:18px;font-weight:700;word-break:break-word}",
            ".chain{display:flex;gap:12px;align-items:stretch;overflow-x:auto;padding-bottom:8px}",
            ".chain .align-card{min-width:280px;max-width:280px}",
            ".arrow{align-self:center;font-size:26px;color:#6b7280;font-weight:800;white-space:nowrap}",
            ".align-card{background:white;border:1px solid #d1d5db;border-radius:6px;padding:12px;box-shadow:0 1px 2px rgba(15,23,42,.06)}",
            ".align-card.selected-current{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.14)}",
            ".align-card.target-node{border-color:#dc2626;box-shadow:0 0 0 2px rgba(220,38,38,.10)}",
            ".align-card.bridge-node{border-color:#9333ea;box-shadow:0 0 0 2px rgba(147,51,234,.12)}",
            "img{width:100%;aspect-ratio:16/10;object-fit:contain;background:#f9fafb;border:1px solid #e5e7eb;border-radius:4px}",
            ".missing{height:160px;display:grid;place-items:center;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:4px;color:#6b7280}",
            "h1{margin-top:0} h2{margin:0 0 12px} h3{font-size:16px;margin:10px 0 6px;line-height:1.25}",
            "p{margin:4px 0}.small{font-size:12px;color:#4b5563}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}",
            ".pills{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.pill{font-size:12px;font-weight:700;border-radius:999px;padding:3px 8px;background:#e5e7eb;color:#374151}",
            ".pill.current{background:#dbeafe;color:#1d4ed8}.pill.passage{background:#fee2e2;color:#991b1b}.pill.bridge{background:#f3e8ff;color:#7e22ce}.pill.selected{background:#dcfce7;color:#166534}",
            ".scores{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.scores div{background:#f8fafc;border-radius:4px;padding:6px}",
            "table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:8px}th{color:#4b5563;font-size:12px;text-transform:uppercase;letter-spacing:.04em}tr.selected{background:#ecfdf5;font-weight:700}",
            ".view-panel{border:1px solid #d1d5db;border-radius:8px;padding:12px;margin:12px 0;background:#fff}",
            ".view-panel.selected{border-color:#16a34a;box-shadow:0 0 0 2px rgba(22,163,74,.12)}",
            ".view-header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between}",
            "</style></head><body>",
            "<h1>Current-to-Target Passage Chain</h1>",
            "<section class='summary'>",
            "<div class='summary-grid'>",
            metric_html("Selected capture", result.get("selected_view", {}).get("capture_index")),
            metric_html("Selection reason", result.get("selection_reason")),
            metric_html("Target found", selected.get("target_found")),
            metric_html("Chain score", format_score(selected.get("chain_score"))),
            metric_html("Bottleneck", format_score(selected.get("bottleneck_similarity"))),
            metric_html("Path length", selected.get("path_length")),
            metric_html("Branching", config.get("branching_factor")),
            metric_html("Max depth", config.get("max_depth")),
            "</div>",
            "<p class='small'>This mode grows each current pano capture toward the selected passage source image. A view that reaches the target passage is preferred, then ranked by chain continuity and path length.</p>",
            "</section>",
            "<section class='panel'>",
            "<h2>Selected Chain: Current View -> Target Passage</h2>",
            current_to_target_alignment_chain_html(selected, asset_map),
            "</section>",
            "<section class='panel'>",
            "<h2>8-View Ranking</h2>",
            "<table><thead><tr><th>rank</th><th>capture</th><th>found</th><th>score</th><th>bottleneck</th><th>mean</th><th>target sim</th><th>length</th></tr></thead><tbody>",
            "\n".join(ranking_rows),
            "</tbody></table>",
            "</section>",
            "<section class='panel'>",
            "<h2>Per-View Debug</h2>",
            "\n".join(view_panels),
            "</section>",
            "</body></html>",
        ]
    )
    gallery_path.write_text(document, encoding="utf-8")
    embedded_document = embed_document_asset_refs(document, asset_dir, asset_map)
    (output_dir / "alignment_gallery_embedded.html").write_text(
        embedded_document, encoding="utf-8"
    )


def current_to_target_gallery_image_items(result: dict) -> list[tuple[str, dict]]:
    image_items: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def add_node(node: dict) -> None:
        node_id = str(node.get("id"))
        if node_id in seen:
            return
        seen.add(node_id)
        image_items.append((node_id, node))

    selected = result.get("selected_alignment", {})
    for node in selected.get("memory_chain_current_to_target", []):
        add_node(node)
    for alignment in result.get("view_alignments", []):
        chain = alignment.get("memory_chain_current_to_target", [])
        if not chain:
            continue
        add_node(chain[0])
        add_node(chain[-1])
    return image_items


def current_to_target_alignment_chain_html(alignment: dict, asset_map: dict[str, str]) -> str:
    chain = alignment.get("memory_chain_current_to_target", [])
    parts = []
    for index, node in enumerate(chain):
        is_first = index == 0
        is_last = index == len(chain) - 1
        found_target = bool(alignment.get("target_found")) and is_last
        parts.append(
            alignment_node_card(
                node,
                asset_map.get(node["id"]),
                role_label="Current chain",
                extra_class="selected-current" if is_first else ("target-node" if found_target else "bridge-node"),
                extra_pill="Selected view" if is_first else ("Target passage" if found_target else ("Best progress" if is_last else None)),
            )
        )
    return "<div class='chain'>" + "<span class='arrow'>→</span>".join(parts) + "</div>"


def current_to_target_view_panel(alignment: dict, asset_map: dict[str, str]) -> str:
    classes = ["view-panel"]
    if alignment.get("selected"):
        classes.append("selected")
    return "\n".join(
        [
            f"<div class='{html.escape(' '.join(classes))}'>",
            "<div class='view-header'>",
            f"<h3>rank {alignment.get('rank')} / capture {alignment.get('capture_index')}</h3>",
            f"<p>score={format_score(alignment.get('chain_score'))} found={html.escape(str(alignment.get('target_found')))} length={alignment.get('path_length')}</p>",
            "</div>",
            current_to_target_alignment_chain_html(alignment, asset_map),
            "</div>",
        ]
    )


def write_bidirectional_gallery(output_dir: Path, result: dict) -> None:
    gallery_path = output_dir / "alignment_gallery.html"
    asset_dir = output_dir / "alignment_gallery_assets"
    image_items = bidirectional_gallery_image_items(result)
    asset_map = copy_gallery_images(asset_dir, result, image_items=image_items)
    selected = result["selected_alignment"]
    bridge = selected["best_bridge"]
    config = result.get("configuration", {})

    ranking_rows = []
    for item in result.get("view_alignments", []):
        mark = "selected" if item.get("selected") else ""
        item_bridge = item["best_bridge"]
        ranking_rows.append(
            "".join(
                [
                    f"<tr class='{mark}'>",
                    f"<td>{item.get('rank')}</td>",
                    f"<td>{item.get('capture_index')}</td>",
                    f"<td>{format_score(item_bridge.get('total_score'))}</td>",
                    f"<td>{format_score(item_bridge.get('continuity_score'))}</td>",
                    f"<td>{format_score(item_bridge.get('chain_mean_continuity'))}</td>",
                    f"<td>{format_score(item_bridge.get('chain_bottleneck_similarity'))}</td>",
                    f"<td>{format_score(item_bridge.get('bridge_similarity'))}</td>",
                    f"<td>{item_bridge.get('current_depth')} + {item_bridge.get('passage_depth')}</td>",
                    "</tr>",
                ]
            )
        )

    view_panels = []
    for item in result.get("view_alignments", []):
        view_panels.append(view_alignment_panel(item, asset_map))

    document = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Bidirection Passage Alignment</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;background:#f8fafc;color:#111827}",
            ".summary,.panel{background:white;border:1px solid #d1d5db;border-radius:8px;padding:16px;margin-bottom:18px}",
            ".summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}",
            ".metric{background:#f3f4f6;border-radius:6px;padding:10px;min-height:58px}",
            ".metric b{display:block;font-size:12px;color:#4b5563;text-transform:uppercase;letter-spacing:.04em}",
            ".metric span{font-size:18px;font-weight:700;word-break:break-word}",
            ".chain{display:flex;gap:12px;align-items:stretch;overflow-x:auto;padding-bottom:8px}",
            ".chain .align-card{min-width:280px;max-width:280px}",
            ".arrow,.bridge-separator{align-self:center;font-size:26px;color:#6b7280;font-weight:800;white-space:nowrap}",
            ".bridge-separator{color:#9333ea}",
            ".align-card{background:white;border:1px solid #d1d5db;border-radius:6px;padding:12px;box-shadow:0 1px 2px rgba(15,23,42,.06)}",
            ".align-card.selected-current{border-color:#2563eb;box-shadow:0 0 0 2px rgba(37,99,235,.14)}",
            ".align-card.bridge-node{border-color:#9333ea;box-shadow:0 0 0 2px rgba(147,51,234,.12)}",
            ".align-card.target-node{border-color:#dc2626;box-shadow:0 0 0 2px rgba(220,38,38,.10)}",
            "img{width:100%;aspect-ratio:16/10;object-fit:contain;background:#f9fafb;border:1px solid #e5e7eb;border-radius:4px}",
            ".missing{height:160px;display:grid;place-items:center;background:#f3f4f6;border:1px solid #e5e7eb;border-radius:4px;color:#6b7280}",
            "h1{margin-top:0} h2{margin:0 0 12px} h3{font-size:16px;margin:10px 0 6px;line-height:1.25}",
            "p{margin:4px 0}.small{font-size:12px;color:#4b5563}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;word-break:break-all}",
            ".pills{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.pill{font-size:12px;font-weight:700;border-radius:999px;padding:3px 8px;background:#e5e7eb;color:#374151}",
            ".pill.current{background:#dbeafe;color:#1d4ed8}.pill.passage{background:#fee2e2;color:#991b1b}.pill.bridge{background:#f3e8ff;color:#7e22ce}.pill.selected{background:#dcfce7;color:#166534}",
            ".scores{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}.scores div{background:#f8fafc;border-radius:4px;padding:6px}",
            "table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;border-bottom:1px solid #e5e7eb;padding:8px}th{color:#4b5563;font-size:12px;text-transform:uppercase;letter-spacing:.04em}tr.selected{background:#ecfdf5;font-weight:700}",
            ".view-panel{border:1px solid #d1d5db;border-radius:8px;padding:12px;margin:12px 0;background:#fff}",
            ".view-panel.selected{border-color:#16a34a;box-shadow:0 0 0 2px rgba(22,163,74,.12)}",
            ".view-header{display:flex;flex-wrap:wrap;gap:12px;align-items:baseline;justify-content:space-between}",
            "</style></head><body>",
            "<h1>Bidirection Passage Alignment</h1>",
            "<section class='summary'>",
            "<div class='summary-grid'>",
            metric_html("Selected capture", result.get("selected_view", {}).get("capture_index")),
            metric_html("Total score", format_score(bridge.get("total_score"))),
            metric_html("Continuity score", format_score(bridge.get("continuity_score"))),
            metric_html("Mean continuity", format_score(bridge.get("chain_mean_continuity"))),
            metric_html("Bottleneck", format_score(bridge.get("chain_bottleneck_similarity"))),
            metric_html("Bridge similarity", format_score(bridge.get("bridge_similarity"))),
            metric_html("Bridge depth", bridge.get("total_bridge_depth")),
            metric_html("Branching", config.get("branching_factor")),
            metric_html("Max depth", config.get("max_depth")),
            "</div>",
            "<p class='small'>This mode treats each current pano capture as a possible action view. Bidirection mode uses the weighted bridge formula with current and passage minimum continuity.</p>",
            "</section>",
            "<section class='panel'>",
            "<h2>Selected Alignment: Current Side -> Bridge Pair -> Passage Side -> P0</h2>",
            selected_alignment_chain_html(selected, asset_map),
            "</section>",
            "<section class='panel'>",
            "<h2>8-View Ranking</h2>",
            "<table><thead><tr><th>rank</th><th>capture</th><th>total</th><th>continuity</th><th>mean</th><th>bottleneck</th><th>bridge</th><th>depth</th></tr></thead><tbody>",
            "\n".join(ranking_rows),
            "</tbody></table>",
            "</section>",
            "<section class='panel'>",
            "<h2>Per-View Debug</h2>",
            "\n".join(view_panels),
            "</section>",
            "</body></html>",
        ]
    )
    gallery_path.write_text(document, encoding="utf-8")
    embedded_document = embed_document_asset_refs(document, asset_dir, asset_map)
    (output_dir / "alignment_gallery_embedded.html").write_text(
        embedded_document, encoding="utf-8"
    )


def selected_alignment_chain_html(alignment: dict, asset_map: dict[str, str]) -> str:
    current_chain = alignment.get("current_chain_root_to_bridge", [])
    passage_chain = alignment.get("passage_chain_bridge_to_target", [])
    parts = []
    for index, node in enumerate(current_chain):
        parts.append(
            alignment_node_card(
                node,
                asset_map.get(node["id"]),
                role_label="Current side",
                extra_class="selected-current" if index == 0 else ("bridge-node" if index == len(current_chain) - 1 else ""),
                extra_pill="Selected view" if index == 0 else ("Current bridge" if index == len(current_chain) - 1 else None),
            )
        )
    parts.append("<span class='bridge-separator'>bridge</span>")
    for index, node in enumerate(passage_chain):
        parts.append(
            alignment_node_card(
                node,
                asset_map.get(node["id"]),
                role_label="Passage side",
                extra_class="bridge-node" if index == 0 else ("target-node" if index == len(passage_chain) - 1 else ""),
                extra_pill="Passage bridge" if index == 0 else ("P0 target" if index == len(passage_chain) - 1 else None),
            )
        )
    return "<div class='chain'>" + "<span class='arrow'>→</span>".join(parts) + "</div>"


def view_alignment_panel(alignment: dict, asset_map: dict[str, str]) -> str:
    bridge = alignment["best_bridge"]
    classes = ["view-panel"]
    if alignment.get("selected"):
        classes.append("selected")
    preview_nodes = compact_alignment_preview_nodes(alignment)
    cards = []
    for node, label in preview_nodes:
        cards.append(
            alignment_node_card(
                node,
                asset_map.get(node["id"]),
                role_label=label,
                extra_class="bridge-node" if "bridge" in label.lower() else "",
                extra_pill=label,
            )
        )
    return "\n".join(
        [
            f"<div class='{html.escape(' '.join(classes))}'>",
            "<div class='view-header'>",
            f"<h3>rank {alignment.get('rank')} / capture {alignment.get('capture_index')}</h3>",
            f"<p>total={format_score(bridge.get('total_score'))} continuity={format_score(bridge.get('continuity_score'))} bottleneck={format_score(bridge.get('chain_bottleneck_similarity'))} bridge={format_score(bridge.get('bridge_similarity'))} depth={bridge.get('current_depth')}+{bridge.get('passage_depth')}</p>",
            "</div>",
            "<div class='chain'>" + "<span class='arrow'>→</span>".join(cards) + "</div>",
            "</div>",
        ]
    )


def compact_alignment_preview_nodes(alignment: dict) -> list[tuple[dict, str]]:
    current_chain = alignment.get("current_chain_root_to_bridge", [])
    passage_chain = alignment.get("passage_chain_bridge_to_target", [])
    candidates: list[tuple[dict, str]] = []
    if current_chain:
        candidates.append((current_chain[0], "current root"))
        if current_chain[-1]["id"] != current_chain[0]["id"]:
            candidates.append((current_chain[-1], "current bridge"))
    if passage_chain:
        candidates.append((passage_chain[0], "passage bridge"))
        if passage_chain[-1]["id"] != passage_chain[0]["id"]:
            candidates.append((passage_chain[-1], "P0 target"))
    return candidates


def alignment_node_card(
    node: dict,
    image_src: str | None,
    *,
    role_label: str,
    extra_class: str = "",
    extra_pill: str | None = None,
) -> str:
    classes = ["align-card"]
    if extra_class:
        classes.append(extra_class)
    role_class = "current" if "current" in role_label.lower() else "passage"
    pills = [f"<span class='pill {role_class}'>{html.escape(role_label)}</span>"]
    if extra_pill:
        pill_class = "selected" if "selected" in extra_pill.lower() else "bridge"
        pills.append(f"<span class='pill {pill_class}'>{html.escape(extra_pill)}</span>")
    return "\n".join(
        [
            f"<article class='{html.escape(' '.join(classes))}'>",
            image_tag(image_src, str(node.get("id"))),
            "<div class='pills'>" + "".join(pills) + "</div>",
            f"<h3>{html.escape(str(node.get('id')))} {html.escape(str(node.get('memory_key')))}</h3>",
            f"<p>depth={node.get('depth')} parent={html.escape(str(node.get('parent_id')))}</p>",
            "<div class='scores'>",
            score_box("parent", node.get("sim_to_parent")),
            score_box("expand", node.get("score")),
            score_box("path mean", node.get("path_mean_continuity")),
            score_box("path bottleneck", node.get("path_bottleneck_similarity")),
            score_box("near dup", node.get("near_duplicate_count")),
            score_box("penalty", node.get("near_duplicate_penalty")),
            "</div>",
            f"<p class='mono'>{html.escape(str(node.get('pano_id')))} #{node.get('capture_index')} {html.escape(str(node.get('capture_label')))}</p>",
            "</article>",
        ]
    )


def format_score(value) -> str:
    if isinstance(value, (float, int)):
        return f"{float(value):.3f}"
    if value is None:
        return "-"
    return str(value)


def copy_gallery_images(
    asset_dir: Path,
    result: dict,
    *,
    image_items: Sequence[tuple[str, dict]] | None = None,
) -> dict[str, str]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    for key, item in image_items or default_gallery_image_items(result):
        image_path = item.get("image_path") if isinstance(item, dict) else None
        if not isinstance(image_path, str) or not image_path:
            continue
        source = Path(image_path)
        if not source.exists():
            continue
        filename = f"{safe_name(key)}_{safe_name(source.stem)}{source.suffix or '.png'}"
        target = asset_dir / filename
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        mapping[key] = str(Path(asset_dir.name) / filename)
    return mapping


def default_gallery_image_items(result: dict) -> list[tuple[str, dict]]:
    image_items = [("current", result.get("current_view", {}))]
    image_items.extend((node["id"], node) for node in result.get("nodes", []))
    return image_items


def bidirectional_gallery_image_items(result: dict) -> list[tuple[str, dict]]:
    image_items: list[tuple[str, dict]] = []
    seen: set[str] = set()

    def add_node(node: dict) -> None:
        node_id = str(node.get("id"))
        if node_id in seen:
            return
        seen.add(node_id)
        image_items.append((node_id, node))

    selected = result.get("selected_alignment", {})
    for node in selected.get("current_chain_root_to_bridge", []):
        add_node(node)
    for node in selected.get("passage_chain_bridge_to_target", []):
        add_node(node)
    for alignment in result.get("view_alignments", []):
        for node, _label in compact_alignment_preview_nodes(alignment):
            add_node(node)
    return image_items


def embed_document_asset_refs(document: str, asset_dir: Path, asset_map: dict[str, str]) -> str:
    embedded = document
    for relative_src in sorted(set(asset_map.values()), key=len, reverse=True):
        source = asset_dir.parent / relative_src
        data_uri = image_data_uri(source)
        if data_uri is None:
            continue
        escaped_src = html.escape(relative_src)
        embedded = embedded.replace(f"src='{escaped_src}'", f"src='{data_uri}'")
    return embedded


def image_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def current_view_card(current_view: dict, image_src: str | None) -> str:
    image_html = image_tag(image_src, "current view")
    return "\n".join(
        [
            "<article class='current-card'>",
            image_html,
            "<div class='pills'><span class='pill chain'>Current</span></div>",
            "<h3>Current View</h3>",
            f"<p>mode={html.escape(str(current_view.get('mode')))}</p>",
            f"<p class='mono'>{html.escape(str(current_view.get('pano_id') or current_view.get('image_path') or ''))}</p>",
            "</article>",
        ]
    )


def node_card(
    node: dict,
    *,
    image_src: str | None,
    parent_edge: dict | None,
    is_chain: bool,
    is_stop: bool,
    is_target: bool,
) -> str:
    classes = ["card"]
    if is_chain:
        classes.append("chain-node")
    if is_stop:
        classes.append("stop-node")
    if is_target:
        classes.append("target-node")
    pills = []
    if is_target:
        pills.append("<span class='pill root'>P0 / Root</span>")
    if is_stop:
        pills.append("<span class='pill stop'>Stop</span>")
    if is_chain:
        pills.append("<span class='pill chain'>Best chain</span>")
    parent_similarity = None if parent_edge is None else parent_edge.get("sim_to_parent")
    return "\n".join(
        [
            f"<article class='{html.escape(' '.join(classes))}'>",
            image_tag(image_src, str(node.get("id"))),
            "<div class='pills'>" + "".join(pills) + "</div>",
            f"<h3>{html.escape(str(node.get('id')))} {html.escape(str(node.get('memory_key')))}</h3>",
            f"<p>depth={node.get('depth')} parent={html.escape(str(node.get('parent_id')))}</p>",
            "<div class='scores'>",
            score_box("parent edge", parent_similarity),
            score_box("to current", node.get("sim_to_current")),
            score_box("current rank", node.get("top_current_rank")),
            score_box("expand score", node.get("score")),
            score_box("near dup", node.get("near_duplicate_count")),
            score_box("penalty", node.get("near_duplicate_penalty")),
            "</div>",
            f"<p class='mono'>{html.escape(str(node.get('pano_id')))} #{node.get('capture_index')} {html.escape(str(node.get('capture_label')))}</p>",
            "</article>",
        ]
    )


def image_tag(image_src: str | None, alt: str) -> str:
    if not image_src:
        return "<div class='missing'>image unavailable</div>"
    return f"<img src='{html.escape(image_src)}' alt='{html.escape(alt)}' loading='lazy'>"


def metric_html(label: str, value) -> str:
    return f"<div class='metric'><b>{html.escape(str(label))}</b><span>{html.escape(str(value))}</span></div>"


def score_box(label: str, value) -> str:
    if isinstance(value, float):
        display = f"{value:.3f}"
    elif value is None:
        display = "-"
    else:
        display = str(value)
    return f"<div><b>{html.escape(label)}</b><br>{html.escape(display)}</div>"

def paste_thumbnail(sheet, draw, image_path, x: int, y: int, width: int, height: int, font) -> None:
    from PIL import Image

    path = Path(str(image_path)) if image_path else None
    if path is None or not path.exists():
        draw.rectangle((x, y, x + width, y + height), fill="#eeeeee", outline="#bbbbbb")
        draw.text((x + 78, y + 96), "image unavailable", font=font, fill="#777777")
        return
    with Image.open(path).convert("RGB") as image:
        image.thumbnail((width, height))
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        sheet.paste(canvas, (x, y))
    draw.rectangle((x, y, x + width, y + height), outline="#bbbbbb", width=1)


def draw_wrapped_text(draw, position: tuple[int, int], text: str, font, *, max_width: int) -> int:
    x, y = position
    words = str(text).split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines[:4]:
        draw.text((x, y), line, font=font, fill="black")
        y += 16
    return y


if __name__ == "__main__":
    raise SystemExit(main())
