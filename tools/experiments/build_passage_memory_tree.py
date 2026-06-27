from __future__ import annotations

import argparse
import base64
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
from memory_nav.data.memory_localization import load_json  # noqa: E402


DEFAULT_METADATA_PATH = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
DEFAULT_RENDER_ROOT = "renders/room_grounding_fov90"
DEFAULT_ARTIFACTS_DIR = "dataset/sites/british_museum/normalized"
DEFAULT_BRANCHING_FACTOR = 5
DEFAULT_MAX_DEPTH = 4
DEFAULT_CURRENT_SIMILARITY_THRESHOLD = 0.78
DEFAULT_CURRENT_RANK_GUARD = 10
DEFAULT_NEAR_DUPLICATE_THRESHOLD = 0.92
DEFAULT_NEAR_DUPLICATE_PENALTY = 0.15
DEFAULT_PARENT_SIMILARITY_WEIGHT = 1.0
DEFAULT_CURRENT_SIMILARITY_WEIGHT = 0.0
DEFAULT_BRIDGE_DEPTH_PENALTY = 0.02
DEFAULT_BRIDGE_CONTINUITY_WEIGHT = 0.25


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline DreamSim Passage Memory Tree from a target passage "
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
            "When omitted with --current-pano-id, runs 8-view bidirectional "
            "current/passage alignment."
        ),
    )
    parser.add_argument("--representatives-path")
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--output-dir")
    parser.add_argument("--dreamsim-type", default="ensemble")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--branching-factor", type=int, default=DEFAULT_BRANCHING_FACTOR)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
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
    parser.add_argument("--bridge-depth-penalty", type=float, default=DEFAULT_BRIDGE_DEPTH_PENALTY)
    parser.add_argument(
        "--bridge-continuity-weight",
        type=float,
        default=DEFAULT_BRIDGE_CONTINUITY_WEIGHT,
        help="Weight for each side's minimum chain continuity in bidirectional mode.",
    )
    parser.add_argument(
        "--omit-embeddings",
        action="store_true",
        help="Do not write DreamSim embedding vectors into tree.json.",
    )
    return parser


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

    embedder = DreamSimImageEmbedder(
        dreamsim_type=args.dreamsim_type,
        device=args.device,
        batch_size=args.batch_size,
    )
    room_image_paths = [Path(str(item["image_path"])) for item in room_items]
    encode_paths = list(room_image_paths)
    if current_extra_path is not None:
        encode_paths.append(current_extra_path)
    encoded = normalize_rows(np.asarray(embedder.encode_image_paths(encode_paths), dtype=np.float32))
    room_embeddings = encoded[: len(room_items)]
    pano_graph = load_optional_pano_graph(resolve_project_path(args.artifacts_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if bidirectional_mode:
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
            include_embeddings=not args.omit_embeddings,
            dreamsim_type=args.dreamsim_type,
        )
        metrics = build_bidirectional_metrics(result, pano_graph=pano_graph)
        result["metrics"] = metrics
        write_json(output_dir / "alignment.json", result)
        write_json(output_dir / "best_alignment.json", build_best_alignment_payload(result))
        write_json(output_dir / "metrics.json", metrics)
        write_bidirectional_gallery(output_dir, result)
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
        "method": "dreamsim_passage_memory_tree",
        "configuration": {
            "dreamsim_type": dreamsim_type,
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
    include_embeddings: bool = True,
    dreamsim_type: str = "ensemble",
) -> dict:
    if not current_view_indices:
        raise ValueError("current_view_indices must not be empty.")
    embeddings = normalize_rows(np.asarray(room_embeddings, dtype=np.float32))
    if embeddings.shape[0] != len(room_items):
        raise ValueError("room_embeddings and room_items must have the same length.")
    pairwise = embeddings @ embeddings.T

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
        )
        current_chain = node_path(current_tree, best_bridge["current_node_id"])
        passage_chain_root_to_bridge = node_path(passage_tree, best_bridge["passage_node_id"])
        passage_chain_bridge_to_target = list(reversed(passage_chain_root_to_bridge))
        view_alignment = {
            "view_order": view_order,
            "capture_index": capture_index,
            "root": current_tree["root"],
            "best_bridge": best_bridge,
            "current_chain_root_to_bridge": current_chain,
            "passage_chain_root_to_bridge": passage_chain_root_to_bridge,
            "passage_chain_bridge_to_target": passage_chain_bridge_to_target,
            "current_tree": current_tree,
        }
        view_alignments.append(view_alignment)
        all_nodes.extend(dict(node) for node in current_tree["nodes"])
        all_edges.extend(dict(edge) for edge in current_tree["edges"])

    view_alignments.sort(
        key=lambda item: (
            -float(item["best_bridge"]["total_score"]),
            -float(item["best_bridge"]["bridge_similarity"]),
            int(item["capture_index"]),
        )
    )
    selected = view_alignments[0]
    for rank, item in enumerate(view_alignments, start=1):
        item["rank"] = rank
        item["selected"] = item is selected

    selected_view = {
        "view_order": selected["view_order"],
        "capture_index": selected["capture_index"],
        **public_memory_item(room_items[current_view_indices[selected["view_order"]]]),
        "total_score": float(selected["best_bridge"]["total_score"]),
        "bridge_similarity": float(selected["best_bridge"]["bridge_similarity"]),
    }
    return {
        "method": "dreamsim_bidirectional_passage_alignment",
        "configuration": {
            "dreamsim_type": dreamsim_type,
            "target_passage_label": target_passage_label,
            "room_id": room_items[target_index].get("room_id"),
            "branching_factor": int(branching_factor),
            "max_depth": int(max_depth),
            "near_duplicate_threshold": float(near_duplicate_threshold),
            "near_duplicate_penalty": float(near_duplicate_penalty_weight),
            "parent_similarity_weight": float(parent_similarity_weight),
            "bridge_depth_penalty": float(bridge_depth_penalty),
            "bridge_continuity_weight": float(bridge_continuity_weight),
            "passage_expansion_excludes_current_views": True,
            "current_expansion_excludes_target_passage_and_current_views": True,
            "bridge_scoring_formula": (
                "bridge_similarity + bridge_continuity_weight * "
                "(current_chain_min_parent_similarity + passage_chain_min_parent_similarity) "
                "- bridge_depth_penalty * total_bridge_depth"
            ),
            "include_embeddings": bool(include_embeddings),
        },
        "current_view": dict(current_context),
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
    exclude_indices: set[int] | None = None,
    extra_root_fields: dict | None = None,
) -> dict:
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
            scored_candidates.sort(
                key=lambda candidate: (
                    -candidate["score"],
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
                    }
                )
                next_frontier.append((child, child_path))
        frontier = next_frontier
        if not frontier:
            break
    return {"root": root, "nodes": nodes, "edges": edges}


def choose_best_bridge(
    *,
    current_tree: dict,
    passage_tree: dict,
    current_leaves: Sequence[dict],
    passage_leaves: Sequence[dict],
    pairwise_similarities,
    bridge_depth_penalty: float,
    bridge_continuity_weight: float,
) -> dict:
    best: dict | None = None
    for current_node in current_leaves:
        current_min = chain_min_parent_similarity(current_tree, current_node["id"])
        current_chain = node_path(current_tree, current_node["id"])
        for passage_node in passage_leaves:
            passage_min = chain_min_parent_similarity(passage_tree, passage_node["id"])
            passage_chain = node_path(passage_tree, passage_node["id"])
            bridge_similarity = float(
                pairwise_similarities[int(current_node["item_index"]), int(passage_node["item_index"])]
            )
            total_depth = int(current_node["depth"]) + int(passage_node["depth"])
            total_score = (
                bridge_similarity
                + float(bridge_continuity_weight) * (current_min + passage_min)
                - float(bridge_depth_penalty) * total_depth
            )
            candidate = {
                "current_node_id": current_node["id"],
                "passage_node_id": passage_node["id"],
                "current_memory_key": current_node["memory_key"],
                "passage_memory_key": passage_node["memory_key"],
                "bridge_similarity": bridge_similarity,
                "current_chain_min_parent_similarity": float(current_min),
                "passage_chain_min_parent_similarity": float(passage_min),
                "current_depth": int(current_node["depth"]),
                "passage_depth": int(passage_node["depth"]),
                "total_bridge_depth": total_depth,
                "total_score": float(total_score),
                "current_chain_node_ids": [node["id"] for node in current_chain],
                "passage_chain_node_ids": [node["id"] for node in passage_chain],
            }
            if best is None or bridge_sort_key(candidate) > bridge_sort_key(best):
                best = candidate
    if best is None:
        raise RuntimeError("No bridge candidates were produced.")
    return best


def bridge_sort_key(candidate: dict) -> tuple:
    return (
        float(candidate["total_score"]),
        float(candidate["bridge_similarity"]),
        float(candidate["current_chain_min_parent_similarity"]),
        float(candidate["passage_chain_min_parent_similarity"]),
        -int(candidate["total_bridge_depth"]),
        str(candidate["current_memory_key"]),
        str(candidate["passage_memory_key"]),
    )


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
    return {
        "method": result["method"],
        "configuration": result["configuration"],
        "current_view": result["current_view"],
        "selected_view": result["selected_view"],
        "best_bridge": selected["best_bridge"],
        "current_chain_root_to_bridge": selected["current_chain_root_to_bridge"],
        "passage_chain_bridge_to_target": selected["passage_chain_bridge_to_target"],
    }


def build_bidirectional_metrics(result: dict, *, pano_graph: dict | None) -> dict:
    del pano_graph
    selected = result["selected_alignment"]
    return {
        "selected_capture_index": int(result["selected_view"].get("capture_index") or 0),
        "selected_total_score": float(result["selected_view"]["total_score"]),
        "selected_bridge_similarity": float(result["selected_view"]["bridge_similarity"]),
        "view_count": len(result.get("view_alignments", [])),
        "view_rankings": [
            {
                "rank": item["rank"],
                "selected": item["selected"],
                "capture_index": item["capture_index"],
                "total_score": float(item["best_bridge"]["total_score"]),
                "bridge_similarity": float(item["best_bridge"]["bridge_similarity"]),
                "current_chain_min_parent_similarity": float(
                    item["best_bridge"]["current_chain_min_parent_similarity"]
                ),
                "passage_chain_min_parent_similarity": float(
                    item["best_bridge"]["passage_chain_min_parent_similarity"]
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
    return {
        "memory_index": item.get("memory_index"),
        "room_id": item.get("room_id"),
        "pano_id": item.get("pano_id"),
        "capture_index": item.get("capture_index"),
        "capture_label": item.get("capture_label"),
        "capture_heading": item.get("capture_heading"),
        "image_path": item.get("image_path") or item.get("capture_path"),
    }


def memory_key(item: dict) -> str:
    return f"{item.get('pano_id')}:{item.get('capture_index')}"


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
                    f"<td>{format_score(item_bridge.get('bridge_similarity'))}</td>",
                    f"<td>{format_score(item_bridge.get('current_chain_min_parent_similarity'))}</td>",
                    f"<td>{format_score(item_bridge.get('passage_chain_min_parent_similarity'))}</td>",
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
            "<title>Bidirectional Passage Alignment</title>",
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
            "<h1>Bidirectional Passage Alignment</h1>",
            "<section class='summary'>",
            "<div class='summary-grid'>",
            metric_html("Selected capture", result.get("selected_view", {}).get("capture_index")),
            metric_html("Total score", format_score(bridge.get("total_score"))),
            metric_html("Bridge similarity", format_score(bridge.get("bridge_similarity"))),
            metric_html("Current continuity", format_score(bridge.get("current_chain_min_parent_similarity"))),
            metric_html("Passage continuity", format_score(bridge.get("passage_chain_min_parent_similarity"))),
            metric_html("Bridge depth", bridge.get("total_bridge_depth")),
            metric_html("Branching", config.get("branching_factor")),
            metric_html("Max depth", config.get("max_depth")),
            "</div>",
            "<p class='small'>This mode treats each current pano capture as a possible action view. The current-side tree excludes the target passage root and the other current observation views; the passage-side tree excludes current observation views. The bridge score then chooses the view whose expanded chain best aligns with the passage-expanded chain.</p>",
            "</section>",
            "<section class='panel'>",
            "<h2>Selected Alignment: Current Side -> Bridge Pair -> Passage Side -> P0</h2>",
            selected_alignment_chain_html(selected, asset_map),
            "</section>",
            "<section class='panel'>",
            "<h2>8-View Ranking</h2>",
            "<table><thead><tr><th>rank</th><th>capture</th><th>total</th><th>bridge</th><th>current min</th><th>passage min</th><th>depth</th></tr></thead><tbody>",
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
            f"<p>total={format_score(bridge.get('total_score'))} bridge={format_score(bridge.get('bridge_similarity'))} depth={bridge.get('current_depth')}+{bridge.get('passage_depth')}</p>",
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
