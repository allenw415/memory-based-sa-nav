from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.build_passage_memory_tree import (
    load_memory_metadata,
    memory_key,
    resolve_project_path,
    safe_name,
    write_json,
)
from tools.experiments.eval_memory_tree_child_similarity import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DREAMSIM_INDEX_PATH,
    DEFAULT_DREAMSIM_METADATA_PATH,
    DEFAULT_RENDER_ROOT,
    DEFAULT_SALAD_INDEX_PATH,
    DEFAULT_SALAD_METADATA_PATH,
    RecipeScores,
)
from tools.experiments.select_view_by_visual_chain import (
    DEFAULT_CURRENT_PANO_ID,
    DEFAULT_DINOV2_PATCH_MAX_PATCHES,
    DEFAULT_DINOV2_PATCH_MODEL,
    DEFAULT_DINOV2_PATCH_TOP_K,
    DEFAULT_DINOV2_TARGET_MATCH_MODE,
    DEFAULT_SIGLIP2_INDEX_PATH,
    DEFAULT_SIGLIP2_METADATA_PATH,
    VISUAL_CHAIN_RECIPES,
    build_recipe_with_target,
    current_pano_view_indices,
    find_matching_image_index,
    infer_room_id,
)


DEFAULT_RECIPE = "salad_full"
DEFAULT_MAX_DEPTH = 6
DEFAULT_BEAM_WIDTH = 4
DEFAULT_RETRIEVAL_TOP_K = 20
DEFAULT_RECIPROCAL_TOP_K = 20
DEFAULT_TARGET_HIT_PERCENTILE = 0.98
DEFAULT_MIN_TARGET_PROGRESS = 0.02
DEFAULT_PLATEAU_TOLERANCE = 0.01
DEFAULT_MAX_PLATEAU_STEPS = 1
DEFAULT_DUPLICATE_RANK = 2


@dataclass(frozen=True)
class DynamicChainPath:
    root_index: int
    item_indices: tuple[int, ...]
    edge_raw_similarities: tuple[float, ...] = ()
    edge_strengths: tuple[float, ...] = ()
    edge_forward_ranks: tuple[int, ...] = ()
    edge_reverse_ranks: tuple[int, ...] = ()
    edge_plateaus: tuple[bool, ...] = ()
    stop_reason: str = "searching"


@dataclass(frozen=True)
class DynamicChainSummary:
    view_order: int
    capture_index: int
    selected: bool
    target_hit: bool
    hit_depth: int | None
    hit_target_percentile: float | None
    final_target_percentile: float
    max_target_percentile: float
    chain_min_parent_similarity: float
    chain_mean_parent_similarity: float
    chain_bottleneck_strength: float
    chain_mean_strength: float
    plateau_steps: int
    stop_reason: str
    nodes: tuple[dict, ...]


class SimilarityRankCache:
    """Cache deterministic neighbor orders and inverse ranks for one recipe pool."""

    def __init__(self, *, recipe: RecipeScores, pool_indices: Sequence[int]) -> None:
        self.recipe = recipe
        self.pool_indices = tuple(int(index) for index in pool_indices)
        self._row_by_item_index = recipe.row_by_item_index
        self._pool_array = np.asarray(self.pool_indices, dtype=np.int64)
        self._pool_rows = np.asarray(
            [self._row_by_item_index[index] for index in self.pool_indices],
            dtype=np.int64,
        )
        self._pool_position_by_item_index = {
            item_index: position
            for position, item_index in enumerate(self.pool_indices)
        }
        self._ordered_by_query: dict[int, np.ndarray] = {}
        self._rank_by_query: dict[int, np.ndarray] = {}

    @property
    def cached_query_count(self) -> int:
        return len(self._ordered_by_query)

    def similarity(self, source_index: int, candidate_index: int) -> float:
        return float(
            self.recipe.similarity_matrix[
                self._row_by_item_index[int(source_index)],
                self._row_by_item_index[int(candidate_index)],
            ]
        )

    def ranked_candidates(
        self,
        *,
        query_index: int,
        allowed_indices: set[int] | frozenset[int],
        excluded_indices: Sequence[int],
        limit: int,
    ) -> list[int]:
        ordered, _ranks = self._ensure_query(int(query_index))
        excluded = {int(index) for index in excluded_indices}
        selected = []
        for candidate_index in ordered:
            index = int(candidate_index)
            if index not in allowed_indices or index in excluded:
                continue
            selected.append(index)
            if len(selected) >= max(int(limit), 1):
                break
        return selected

    def rank(self, *, query_index: int, candidate_index: int) -> int:
        ordered, ranks = self._ensure_query(int(query_index))
        pool_position = self._pool_position_by_item_index.get(int(candidate_index))
        if pool_position is None:
            return len(ordered) + 1
        return int(ranks[pool_position])

    def _ensure_query(self, query_index: int) -> tuple[np.ndarray, np.ndarray]:
        cached_order = self._ordered_by_query.get(int(query_index))
        if cached_order is not None:
            return cached_order, self._rank_by_query[int(query_index)]

        query_row = self._row_by_item_index[int(query_index)]
        keep = self._pool_array != int(query_index)
        candidate_indices = self._pool_array[keep]
        candidate_rows = self._pool_rows[keep]
        similarities = self.recipe.similarity_matrix[query_row, candidate_rows]
        order = np.lexsort((candidate_indices, -similarities))
        ordered = candidate_indices[order]

        pool_positions = np.flatnonzero(keep)[order]
        missing_rank = len(ordered) + 1
        ranks = np.full(len(self.pool_indices), missing_rank, dtype=np.int32)
        ranks[pool_positions] = np.arange(1, missing_rank, dtype=np.int32)
        self._ordered_by_query[int(query_index)] = ordered
        self._rank_by_query[int(query_index)] = ranks
        return ordered, ranks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one of eight panorama views with an on-demand, target-conditioned "
            "retrieval chain. This experiment does not build a graph or modify navigation."
        )
    )
    parser.add_argument("--current-pano-id", default=DEFAULT_CURRENT_PANO_ID)
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--room-id")
    parser.add_argument("--expected-view", choices=[f"V{index}" for index in range(1, 9)])
    parser.add_argument("--recipe", choices=VISUAL_CHAIN_RECIPES, default=DEFAULT_RECIPE)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--beam-width", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--retrieval-top-k", type=int, default=DEFAULT_RETRIEVAL_TOP_K)
    parser.add_argument("--reciprocal-top-k", type=int, default=DEFAULT_RECIPROCAL_TOP_K)
    parser.add_argument(
        "--target-hit-percentile",
        type=float,
        default=DEFAULT_TARGET_HIT_PERCENTILE,
    )
    parser.add_argument(
        "--min-target-progress",
        type=float,
        default=DEFAULT_MIN_TARGET_PROGRESS,
    )
    parser.add_argument(
        "--plateau-tolerance",
        type=float,
        default=DEFAULT_PLATEAU_TOLERANCE,
    )
    parser.add_argument("--max-plateau-steps", type=int, default=DEFAULT_MAX_PLATEAU_STEPS)
    parser.add_argument("--duplicate-rank", type=int, default=DEFAULT_DUPLICATE_RANK)
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--salad-index-path", default=DEFAULT_SALAD_INDEX_PATH)
    parser.add_argument("--salad-metadata-path", default=DEFAULT_SALAD_METADATA_PATH)
    parser.add_argument("--siglip2-index-path", default=DEFAULT_SIGLIP2_INDEX_PATH)
    parser.add_argument("--siglip2-metadata-path", default=DEFAULT_SIGLIP2_METADATA_PATH)
    parser.add_argument("--dreamsim-index-path", default=DEFAULT_DREAMSIM_INDEX_PATH)
    parser.add_argument("--dreamsim-metadata-path", default=DEFAULT_DREAMSIM_METADATA_PATH)
    parser.add_argument("--dinov2-patch-model", default=DEFAULT_DINOV2_PATCH_MODEL)
    parser.add_argument("--dinov2-patch-top-k", type=int, default=DEFAULT_DINOV2_PATCH_TOP_K)
    parser.add_argument(
        "--dinov2-patch-max-patches",
        type=int,
        default=DEFAULT_DINOV2_PATCH_MAX_PATCHES,
    )
    parser.add_argument(
        "--dinov2-target-match-mode",
        choices=("target_to_candidate", "symmetric"),
        default=DEFAULT_DINOV2_TARGET_MATCH_MODE,
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    target_image = resolve_project_path(args.target_image)
    if not target_image.exists():
        raise SystemExit(f"Target image does not exist: {target_image}")

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            current_pano_id=args.current_pano_id,
            target_image=target_image,
            recipe=args.recipe,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    render_root = resolve_project_path(args.render_root)
    items = load_memory_metadata(resolve_project_path(args.salad_metadata_path), render_root)
    room_id = args.room_id or infer_room_id(
        current_pano_id=args.current_pano_id,
        items=items,
        artifacts_dir=resolve_project_path(args.artifacts_dir),
    )
    room_indices = tuple(
        index for index, item in enumerate(items) if item.get("room_id") == room_id
    )
    if not room_indices:
        raise SystemExit(f"No memory items found for room: {room_id}")
    current_view_indices = current_pano_view_indices(
        items=items,
        current_pano_id=args.current_pano_id,
        room_indices=room_indices,
    )

    recipe = build_recipe_with_target(
        recipe_name=args.recipe,
        items=items,
        item_indices=room_indices,
        target_image=target_image,
        output_dir=output_dir,
        render_root=render_root,
        salad_index_path=resolve_project_path(args.salad_index_path),
        salad_metadata_path=resolve_project_path(args.salad_metadata_path),
        siglip2_index_path=resolve_project_path(args.siglip2_index_path),
        siglip2_metadata_path=resolve_project_path(args.siglip2_metadata_path),
        dreamsim_index_path=resolve_project_path(args.dreamsim_index_path),
        dreamsim_metadata_path=resolve_project_path(args.dreamsim_metadata_path),
        dinov2_patch_model=args.dinov2_patch_model,
        dinov2_patch_top_k=args.dinov2_patch_top_k,
        dinov2_patch_max_patches=args.dinov2_patch_max_patches,
        dinov2_target_match_mode=args.dinov2_target_match_mode,
        device=args.device,
        batch_size=args.batch_size,
    )
    target_percentiles = target_similarity_percentiles(
        recipe=recipe.recipe_scores,
        target_similarities=recipe.target_similarities,
    )
    candidate_indices = [
        index for index in room_indices if items[index].get("pano_id") != args.current_pano_id
    ]
    target_match_index = find_matching_image_index(items, target_image)
    if target_match_index is not None:
        candidate_indices = [index for index in candidate_indices if index != target_match_index]

    chains = build_dynamic_view_chains(
        items=items,
        recipe=recipe.recipe_scores,
        target_similarities=recipe.target_similarities,
        target_percentiles=target_percentiles,
        current_view_indices=current_view_indices,
        candidate_indices=candidate_indices,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        retrieval_top_k=args.retrieval_top_k,
        reciprocal_top_k=args.reciprocal_top_k,
        target_hit_percentile=args.target_hit_percentile,
        min_target_progress=args.min_target_progress,
        plateau_tolerance=args.plateau_tolerance,
        max_plateau_steps=args.max_plateau_steps,
        duplicate_rank=args.duplicate_rank,
    )
    selected = select_dynamic_chain(chains)
    chains = [
        DynamicChainSummary(
            **{
                **chain.__dict__,
                "selected": (
                    selected is not None and chain.view_order == selected.view_order
                ),
            }
        )
        for chain in chains
    ]
    chains = sorted(chains, key=dynamic_chain_sort_key)
    selected = next((chain for chain in chains if chain.selected), None)
    selected_view_label = None if selected is None else f"V{selected.view_order}"
    direction_correct = (
        selected_view_label == args.expected_view
        if selected_view_label is not None and args.expected_view is not None
        else None
    )
    relative_confidence = (
        0.0
        if selected is None
        else dynamic_relative_confidence(chains, selected)
    )

    selection_path = output_dir / "selection.json"
    chains_path = output_dir / "chains.json"
    gallery_path = output_dir / "gallery.html"
    payload = selection_payload(
        args=args,
        room_id=room_id,
        target_image=target_image,
        selected=selected,
        chains=chains,
        candidate_count=len(candidate_indices),
        relative_confidence=relative_confidence,
        direction_correct=direction_correct,
    )
    payload["outputs"] = {
        "selection_json": str(selection_path),
        "chains_json": str(chains_path),
        "gallery_html": str(gallery_path),
    }
    write_json(selection_path, payload)
    write_json(
        chains_path,
        {
            "target_image": str(target_image),
            "target_percentile_definition": "rank of target similarity within each image's room-memory similarities",
            "chains": [chain_to_dict(chain) for chain in chains],
        },
    )
    write_gallery(
        gallery_path,
        target_image=target_image,
        chains=chains,
        selected_view_label=selected_view_label,
        expected_view_label=args.expected_view,
        direction_correct=direction_correct,
        relative_confidence=relative_confidence,
        configuration=payload["configuration"],
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def validate_args(args) -> None:
    if int(args.max_depth) < 0:
        raise SystemExit("--max-depth must be non-negative.")
    for name in ("beam_width", "retrieval_top_k", "reciprocal_top_k", "duplicate_rank"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    for name in ("target_hit_percentile", "min_target_progress", "plateau_tolerance"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if int(args.max_plateau_steps) < 0:
        raise SystemExit("--max-plateau-steps must be non-negative.")


def default_output_dir(*, current_pano_id: str, target_image: Path, recipe: str) -> Path:
    return resolve_project_path(
        Path("outputs")
        / "dynamic_retrieval_chain_direction"
        / f"{safe_name(current_pano_id)}_{safe_name(target_image.stem)}_{safe_name(recipe)}"
    )


def target_similarity_percentiles(
    *,
    recipe: RecipeScores,
    target_similarities: Mapping[int, float],
) -> dict[int, float]:
    row_by_index = recipe.row_by_item_index
    percentiles: dict[int, float] = {}
    for item_index in recipe.item_indices:
        row = row_by_index[int(item_index)]
        reference = np.delete(recipe.similarity_matrix[row], row)
        if reference.size == 0:
            percentiles[int(item_index)] = 1.0
            continue
        target_value = float(target_similarities[int(item_index)])
        less = int(np.count_nonzero(reference < target_value))
        equal = int(np.count_nonzero(np.isclose(reference, target_value, atol=1e-8)))
        percentiles[int(item_index)] = float((less + 0.5 * equal) / int(reference.size))
    return percentiles


def build_dynamic_view_chains(
    *,
    items: Sequence[dict],
    recipe: RecipeScores,
    target_similarities: Mapping[int, float],
    target_percentiles: Mapping[int, float],
    current_view_indices: Sequence[int],
    candidate_indices: Sequence[int],
    max_depth: int = DEFAULT_MAX_DEPTH,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    reciprocal_top_k: int = DEFAULT_RECIPROCAL_TOP_K,
    target_hit_percentile: float = DEFAULT_TARGET_HIT_PERCENTILE,
    min_target_progress: float = DEFAULT_MIN_TARGET_PROGRESS,
    plateau_tolerance: float = DEFAULT_PLATEAU_TOLERANCE,
    max_plateau_steps: int = DEFAULT_MAX_PLATEAU_STEPS,
    duplicate_rank: int = DEFAULT_DUPLICATE_RANK,
) -> list[DynamicChainSummary]:
    summaries = []
    rank_cache = SimilarityRankCache(
        recipe=recipe,
        pool_indices=recipe.item_indices,
    )
    for view_order, root_index in enumerate(current_view_indices, start=1):
        path = best_dynamic_chain_for_root(
            recipe=recipe,
            target_percentiles=target_percentiles,
            root_index=int(root_index),
            candidate_indices=candidate_indices,
            max_depth=max_depth,
            beam_width=beam_width,
            retrieval_top_k=retrieval_top_k,
            reciprocal_top_k=reciprocal_top_k,
            target_hit_percentile=target_hit_percentile,
            min_target_progress=min_target_progress,
            plateau_tolerance=plateau_tolerance,
            max_plateau_steps=max_plateau_steps,
            duplicate_rank=duplicate_rank,
            rank_cache=rank_cache,
        )
        summaries.append(
            summarize_dynamic_chain(
                items=items,
                path=path,
                target_similarities=target_similarities,
                target_percentiles=target_percentiles,
                view_order=view_order,
                target_hit_percentile=target_hit_percentile,
            )
        )
    return summaries


def best_dynamic_chain_for_root(
    *,
    recipe: RecipeScores,
    target_percentiles: Mapping[int, float],
    root_index: int,
    candidate_indices: Sequence[int],
    max_depth: int,
    beam_width: int,
    retrieval_top_k: int,
    reciprocal_top_k: int,
    target_hit_percentile: float,
    min_target_progress: float,
    plateau_tolerance: float,
    max_plateau_steps: int,
    duplicate_rank: int,
    rank_cache: SimilarityRankCache | None = None,
) -> DynamicChainPath:
    width = max(int(beam_width), 1)
    candidate_set = frozenset(int(index) for index in candidate_indices)
    reciprocal_pool = tuple(int(index) for index in recipe.item_indices)
    active_rank_cache = rank_cache or SimilarityRankCache(
        recipe=recipe,
        pool_indices=reciprocal_pool,
    )
    if active_rank_cache.pool_indices != reciprocal_pool:
        raise ValueError("rank_cache pool must match recipe.item_indices.")
    beam = [DynamicChainPath(root_index=root_index, item_indices=(root_index,))]
    seen_paths = list(beam)

    for _depth in range(max(int(max_depth), 0)):
        hit_paths = [
            path
            for path in beam
            if path_target_hit(
                path,
                target_percentiles=target_percentiles,
                threshold=target_hit_percentile,
            )
        ]
        if hit_paths:
            return with_stop_reason(select_best_path(hit_paths, target_percentiles), "target_hit")

        expanded: list[DynamicChainPath] = []
        for path in beam:
            parent_index = int(path.item_indices[-1])
            ranked = active_rank_cache.ranked_candidates(
                query_index=parent_index,
                allowed_indices=candidate_set,
                excluded_indices=path.item_indices,
                limit=retrieval_top_k,
            )
            accepted: list[DynamicChainPath] = []
            for forward_rank, child_index in enumerate(ranked, start=1):
                reverse_rank = rank_of_candidate(
                    recipe=recipe,
                    query_index=child_index,
                    candidate_index=parent_index,
                    pool_indices=reciprocal_pool,
                    rank_cache=active_rank_cache,
                )
                if reverse_rank > max(int(reciprocal_top_k), 1):
                    continue
                if visually_revisits_path(
                    recipe=recipe,
                    child_index=child_index,
                    path=path,
                    duplicate_rank=duplicate_rank,
                    pool_indices=reciprocal_pool,
                    rank_cache=active_rank_cache,
                ):
                    continue

                parent_progress = float(target_percentiles[parent_index])
                child_progress = float(target_percentiles[child_index])
                delta = child_progress - parent_progress
                plateau_count = sum(bool(value) for value in path.edge_plateaus)
                plateau = False
                if delta >= float(min_target_progress):
                    pass
                elif (
                    plateau_count < max(int(max_plateau_steps), 0)
                    and delta >= -float(plateau_tolerance)
                ):
                    plateau = True
                else:
                    continue

                raw_similarity = active_rank_cache.similarity(parent_index, child_index)
                strength = reciprocal_rank_strength(
                    forward_rank=forward_rank,
                    reverse_rank=reverse_rank,
                    retrieval_top_k=retrieval_top_k,
                    reciprocal_top_k=reciprocal_top_k,
                )
                accepted.append(
                    DynamicChainPath(
                        root_index=path.root_index,
                        item_indices=(*path.item_indices, int(child_index)),
                        edge_raw_similarities=(*path.edge_raw_similarities, raw_similarity),
                        edge_strengths=(*path.edge_strengths, strength),
                        edge_forward_ranks=(*path.edge_forward_ranks, int(forward_rank)),
                        edge_reverse_ranks=(*path.edge_reverse_ranks, int(reverse_rank)),
                        edge_plateaus=(*path.edge_plateaus, bool(plateau)),
                    )
                )
            accepted.sort(key=lambda path: generation_sort_key(path, target_percentiles))
            expanded.extend(accepted[:width])

        if not expanded:
            best = select_best_path(seen_paths, target_percentiles)
            return with_stop_reason(best, "no_valid_reciprocal_progress")
        hit_paths = [
            path
            for path in expanded
            if path_target_hit(
                path,
                target_percentiles=target_percentiles,
                threshold=target_hit_percentile,
            )
        ]
        if hit_paths:
            return with_stop_reason(select_best_path(hit_paths, target_percentiles), "target_hit")
        beam = sorted(
            expanded,
            key=lambda path: generation_sort_key(path, target_percentiles),
        )[:width]
        seen_paths.extend(beam)

    best = select_best_path(seen_paths, target_percentiles)
    return with_stop_reason(best, "max_depth")


def rank_of_candidate(
    *,
    recipe: RecipeScores,
    query_index: int,
    candidate_index: int,
    pool_indices: Sequence[int],
    rank_cache: SimilarityRankCache | None = None,
) -> int:
    active_rank_cache = rank_cache or SimilarityRankCache(
        recipe=recipe,
        pool_indices=pool_indices,
    )
    return active_rank_cache.rank(
        query_index=query_index,
        candidate_index=candidate_index,
    )


def visually_revisits_path(
    *,
    recipe: RecipeScores,
    child_index: int,
    path: DynamicChainPath,
    duplicate_rank: int,
    pool_indices: Sequence[int],
    rank_cache: SimilarityRankCache | None = None,
) -> bool:
    earlier_nodes = path.item_indices[:-1]
    return any(
        rank_of_candidate(
            recipe=recipe,
            query_index=child_index,
            candidate_index=int(previous_index),
            pool_indices=pool_indices,
            rank_cache=rank_cache,
        )
        <= max(int(duplicate_rank), 1)
        for previous_index in earlier_nodes
    )


def reciprocal_rank_strength(
    *,
    forward_rank: int,
    reverse_rank: int,
    retrieval_top_k: int,
    reciprocal_top_k: int,
) -> float:
    forward_denominator = max(int(retrieval_top_k) - 1, 1)
    reverse_denominator = max(int(reciprocal_top_k) - 1, 1)
    forward = 1.0 - min(max(int(forward_rank) - 1, 0), forward_denominator) / forward_denominator
    reverse = 1.0 - min(max(int(reverse_rank) - 1, 0), reverse_denominator) / reverse_denominator
    return float((forward + reverse) / 2.0)


def path_target_hit(
    path: DynamicChainPath,
    *,
    target_percentiles: Mapping[int, float],
    threshold: float,
) -> bool:
    return float(target_percentiles[int(path.item_indices[-1])]) >= float(threshold)


def path_metrics(
    path: DynamicChainPath,
    target_percentiles: Mapping[int, float],
) -> dict:
    progress_values = [float(target_percentiles[int(index)]) for index in path.item_indices]
    parent_similarities = [float(value) for value in path.edge_raw_similarities]
    strengths = [float(value) for value in path.edge_strengths]
    return {
        "final_target_percentile": progress_values[-1],
        "max_target_percentile": max(progress_values),
        "min_parent_similarity": min(parent_similarities) if parent_similarities else 1.0,
        "mean_parent_similarity": (
            sum(parent_similarities) / len(parent_similarities)
            if parent_similarities
            else 1.0
        ),
        "bottleneck": min(strengths) if strengths else 1.0,
        "mean_strength": sum(strengths) / len(strengths) if strengths else 1.0,
        "plateau_steps": sum(bool(value) for value in path.edge_plateaus),
        "depth": len(path.item_indices) - 1,
    }


def generation_sort_key(
    path: DynamicChainPath,
    target_percentiles: Mapping[int, float],
) -> tuple:
    metrics = path_metrics(path, target_percentiles)
    return (
        -float(metrics["final_target_percentile"]),
        -float(metrics["bottleneck"]),
        -float(metrics["mean_strength"]),
        int(metrics["plateau_steps"]),
        int(metrics["depth"]),
        tuple(int(index) for index in path.item_indices),
    )


def select_best_path(
    paths: Sequence[DynamicChainPath],
    target_percentiles: Mapping[int, float],
) -> DynamicChainPath:
    if not paths:
        raise ValueError("At least one path is required.")
    return sorted(paths, key=lambda path: generation_sort_key(path, target_percentiles))[0]


def with_stop_reason(path: DynamicChainPath, stop_reason: str) -> DynamicChainPath:
    return DynamicChainPath(
        root_index=path.root_index,
        item_indices=path.item_indices,
        edge_raw_similarities=path.edge_raw_similarities,
        edge_strengths=path.edge_strengths,
        edge_forward_ranks=path.edge_forward_ranks,
        edge_reverse_ranks=path.edge_reverse_ranks,
        edge_plateaus=path.edge_plateaus,
        stop_reason=stop_reason,
    )


def summarize_dynamic_chain(
    *,
    items: Sequence[dict],
    path: DynamicChainPath,
    target_similarities: Mapping[int, float],
    target_percentiles: Mapping[int, float],
    view_order: int,
    target_hit_percentile: float,
) -> DynamicChainSummary:
    metrics = path_metrics(path, target_percentiles)
    hit_depth = next(
        (
            depth
            for depth, item_index in enumerate(path.item_indices)
            if float(target_percentiles[int(item_index)]) >= float(target_hit_percentile)
        ),
        None,
    )
    nodes = []
    for depth, item_index in enumerate(path.item_indices):
        item = items[int(item_index)]
        previous_percentile = (
            None
            if depth == 0
            else float(target_percentiles[int(path.item_indices[depth - 1])])
        )
        percentile = float(target_percentiles[int(item_index)])
        nodes.append(
            {
                "depth": int(depth),
                "item_index": int(item_index),
                "memory_key": memory_key(item),
                "room_id": item.get("room_id"),
                "pano_id": item.get("pano_id"),
                "capture_index": item.get("capture_index"),
                "capture_label": item.get("capture_label"),
                "capture_heading": item.get("capture_heading"),
                "image_path": item.get("image_path") or item.get("capture_path"),
                "target_similarity": float(target_similarities[int(item_index)]),
                "target_percentile": percentile,
                "target_progress_delta": (
                    None if previous_percentile is None else percentile - previous_percentile
                ),
                "target_hit": percentile >= float(target_hit_percentile),
                "parent_similarity": (
                    None if depth == 0 else float(path.edge_raw_similarities[depth - 1])
                ),
                "edge_strength": (
                    None if depth == 0 else float(path.edge_strengths[depth - 1])
                ),
                "forward_rank": (
                    None if depth == 0 else int(path.edge_forward_ranks[depth - 1])
                ),
                "reverse_rank": (
                    None if depth == 0 else int(path.edge_reverse_ranks[depth - 1])
                ),
                "plateau": None if depth == 0 else bool(path.edge_plateaus[depth - 1]),
            }
        )
    root = items[int(path.root_index)]
    return DynamicChainSummary(
        view_order=int(view_order),
        capture_index=int(root.get("capture_index") or 0),
        selected=False,
        target_hit=hit_depth is not None,
        hit_depth=hit_depth,
        hit_target_percentile=(
            None
            if hit_depth is None
            else float(target_percentiles[int(path.item_indices[hit_depth])])
        ),
        final_target_percentile=float(metrics["final_target_percentile"]),
        max_target_percentile=float(metrics["max_target_percentile"]),
        chain_min_parent_similarity=float(metrics["min_parent_similarity"]),
        chain_mean_parent_similarity=float(metrics["mean_parent_similarity"]),
        chain_bottleneck_strength=float(metrics["bottleneck"]),
        chain_mean_strength=float(metrics["mean_strength"]),
        plateau_steps=int(metrics["plateau_steps"]),
        stop_reason=path.stop_reason,
        nodes=tuple(nodes),
    )


def dynamic_chain_sort_key(chain: DynamicChainSummary) -> tuple:
    if chain.target_hit:
        return (
            0,
            -float(chain.chain_mean_parent_similarity),
            -float(chain.chain_min_parent_similarity),
            -float(chain.chain_mean_strength),
            -float(chain.chain_bottleneck_strength),
            int(chain.plateau_steps),
            int(chain.capture_index),
        )
    return (
        1,
        -float(chain.max_target_percentile),
        -float(chain.final_target_percentile),
        -float(chain.chain_mean_parent_similarity),
        -float(chain.chain_min_parent_similarity),
        -float(chain.chain_mean_strength),
        -float(chain.chain_bottleneck_strength),
        int(chain.plateau_steps),
        int(chain.capture_index),
    )


def select_dynamic_chain(
    chains: Sequence[DynamicChainSummary],
) -> DynamicChainSummary | None:
    if not chains:
        raise ValueError("At least one chain is required.")
    hit_chains = [chain for chain in chains if chain.target_hit]
    if not hit_chains:
        return None
    return sorted(hit_chains, key=dynamic_chain_sort_key)[0]


def dynamic_relative_confidence(
    chains: Sequence[DynamicChainSummary],
    selected: DynamicChainSummary,
) -> float:
    ranked = sorted(chains, key=dynamic_chain_sort_key)
    runner_up = next((chain for chain in ranked if chain.view_order != selected.view_order), None)
    if runner_up is None:
        return parent_continuity_strength(selected.chain_min_parent_similarity)
    if selected.target_hit and not runner_up.target_hit:
        uniqueness = 1.0
    else:
        comparable = [
            chain
            for chain in chains
            if bool(chain.target_hit) == bool(selected.target_hit)
        ]
        parent_means = [float(chain.chain_mean_parent_similarity) for chain in comparable]
        parent_span = max(max(parent_means) - min(parent_means), 1e-6)
        uniqueness = max(
            float(selected.chain_mean_parent_similarity)
            - float(runner_up.chain_mean_parent_similarity),
            0.0,
        ) / parent_span
    return float(
        min(max(uniqueness, 0.0), 1.0)
        * parent_continuity_strength(selected.chain_min_parent_similarity)
    )


def parent_continuity_strength(parent_similarity: float) -> float:
    return min(max(float(parent_similarity), 0.0), 1.0)


def selection_payload(
    *,
    args,
    room_id: str,
    target_image: Path,
    selected: DynamicChainSummary | None,
    chains: Sequence[DynamicChainSummary],
    candidate_count: int,
    relative_confidence: float,
    direction_correct: bool | None,
) -> dict:
    return {
        "method": "target_conditioned_dynamic_retrieval_chain",
        "scope": "standalone_panorama_passage_direction_experiment",
        "uses_prebuilt_graph": False,
        "uses_topology_for_selection": False,
        "configuration": {
            "current_pano_id": args.current_pano_id,
            "target_image": str(target_image),
            "room_id": room_id,
            "recipe": args.recipe,
            "candidate_count": int(candidate_count),
            "max_depth": int(args.max_depth),
            "beam_width": int(args.beam_width),
            "retrieval_top_k": int(args.retrieval_top_k),
            "reciprocal_top_k": int(args.reciprocal_top_k),
            "target_hit_percentile": float(args.target_hit_percentile),
            "min_target_progress": float(args.min_target_progress),
            "plateau_tolerance": float(args.plateau_tolerance),
            "max_plateau_steps": int(args.max_plateau_steps),
            "duplicate_rank": int(args.duplicate_rank),
            "selection_rule": (
                "filter target-hit chains first; rank raw parent-child mean similarity, then raw "
                "parent-child minimum similarity; reciprocal continuity is tie-break evidence; "
                "abstain when no chain hits the target"
            ),
        },
        "selection_status": (
            "no_target_hit" if selected is None else "selected_target_hit_chain"
        ),
        "abstained": selected is None,
        "selected_view_label": None if selected is None else f"V{selected.view_order}",
        "selected_capture_index": None if selected is None else int(selected.capture_index),
        "relative_confidence": float(relative_confidence),
        "expected_view_label": args.expected_view,
        "direction_correct": direction_correct,
        "selected_chain": None if selected is None else chain_to_dict(selected),
        "chains": [chain_summary_row(chain) for chain in chains],
    }


def chain_summary_row(chain: DynamicChainSummary) -> dict:
    return {
        "view_label": f"V{chain.view_order}",
        "view_order": int(chain.view_order),
        "capture_index": int(chain.capture_index),
        "selected": bool(chain.selected),
        "target_hit": bool(chain.target_hit),
        "hit_depth": chain.hit_depth,
        "hit_target_percentile": chain.hit_target_percentile,
        "final_target_percentile": float(chain.final_target_percentile),
        "max_target_percentile": float(chain.max_target_percentile),
        "chain_min_parent_similarity": float(chain.chain_min_parent_similarity),
        "chain_mean_parent_similarity": float(chain.chain_mean_parent_similarity),
        "chain_bottleneck_strength": float(chain.chain_bottleneck_strength),
        "chain_mean_strength": float(chain.chain_mean_strength),
        "plateau_steps": int(chain.plateau_steps),
        "stop_reason": chain.stop_reason,
    }


def chain_to_dict(chain: DynamicChainSummary) -> dict:
    return {
        **chain_summary_row(chain),
        "nodes": [dict(node) for node in chain.nodes],
    }


def image_data_uri(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    media_type = image_media_type(payload)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def image_media_type(payload: bytes) -> str:
    if payload.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return "image/webp"
    if payload.startswith(b"BM"):
        return "image/bmp"
    return "application/octet-stream"


def write_gallery(
    path: Path,
    *,
    target_image: Path,
    chains: Sequence[DynamicChainSummary],
    selected_view_label: str | None,
    expected_view_label: str | None,
    direction_correct: bool | None,
    relative_confidence: float,
    configuration: Mapping[str, object],
) -> None:
    target_asset = image_data_uri(target_image)
    node_assets: dict[str, str] = {}
    for chain in chains:
        for node in chain.nodes:
            image_path = node.get("image_path")
            if isinstance(image_path, str):
                embedded = image_data_uri(Path(image_path))
                if embedded is not None:
                    node_assets[str(node["memory_key"])] = embedded

    if selected_view_label is None:
        verdict = "INCONCLUSIVE: NO TARGET HIT"
        verdict_class = "inconclusive"
    elif direction_correct is True:
        verdict = "CORRECT"
        verdict_class = "correct"
    elif direction_correct is False:
        verdict = "INCORRECT"
        verdict_class = "incorrect"
    else:
        verdict = "UNLABELED"
        verdict_class = "unlabeled"
    selected_text = selected_view_label or "none"
    expected_text = expected_view_label or "not provided"
    configuration_html = "".join(
        f"<div class='metric'><b>{html.escape(str(key))}</b> {html.escape(str(value))}</div>"
        for key, value in configuration.items()
        if key
        in {
            "recipe",
            "candidate_count",
            "max_depth",
            "beam_width",
            "retrieval_top_k",
            "reciprocal_top_k",
            "target_hit_percentile",
            "min_target_progress",
            "plateau_tolerance",
            "max_plateau_steps",
        }
    )
    chain_sections = [chain_gallery_section(chain, node_assets) for chain in chains]
    target_html = (
        f"<img src='{html.escape(target_asset)}' alt='target passage'>"
        if target_asset is not None
        else "<div style='width:220px;height:180px;background:#eee'>missing target</div>"
    )
    document = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Dynamic retrieval-chain direction</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,sans-serif;margin:24px;background:#f7f7f4;color:#202020}",
            ".hero,.chain{background:white;border:1px solid #ddd;border-radius:10px;padding:14px;margin-bottom:18px}",
            ".hero{display:flex;gap:18px;align-items:flex-start}.hero img{width:220px;height:180px;object-fit:cover;border:1px solid #bbb}",
            ".verdict{display:inline-block;border-radius:999px;padding:5px 10px;font-weight:750}.correct{background:#dcfce7;color:#166534}.incorrect{background:#fee2e2;color:#991b1b}.unlabeled{background:#e5e7eb;color:#374151}.inconclusive{background:#fef3c7;color:#92400e}",
            ".chain.selected{border-color:#126b38;box-shadow:0 0 0 2px rgba(18,107,56,.15)}",
            ".frames{display:flex;gap:10px;overflow-x:auto;padding-bottom:4px}.frame{min-width:176px;max-width:176px}",
            ".frame img{width:176px;height:150px;object-fit:cover;border:1px solid #ccc;background:#eee}",
            ".label{font-size:12px;line-height:1.4;margin-top:4px}.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px}",
            ".hit{color:#126b38;font-weight:700}.plateau{color:#a16207;font-weight:700}.selected-badge{color:#126b38;font-weight:700}",
            ".metrics{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}.metric{background:#f1f1ed;border:1px solid #ddd;border-radius:6px;padding:6px 8px;font-size:12px}",
            "</style></head><body>",
            "<h1>Dynamic Target-Conditioned Retrieval Chain</h1>",
            "<p>Each chain is built on demand from the eight current views. No prebuilt graph or topology is used. Images are embedded for offline viewing.</p>",
            "<section class='hero'>",
            target_html,
            "<div>",
            "<h2>Direction result</h2>",
            f"<p><span class='verdict {verdict_class}'>{verdict}</span></p>",
            f"<p><b>selected</b> {html.escape(selected_text)} &nbsp; <b>expected</b> {html.escape(expected_text)} &nbsp; <b>relative confidence</b> {relative_confidence:.3f}</p>",
            "<div class='metrics'>",
            configuration_html,
            "</div></div></section>",
            *chain_sections,
            "</body></html>",
        ]
    )
    path.write_text(document, encoding="utf-8")


def chain_gallery_section(
    chain: DynamicChainSummary,
    node_assets: Mapping[str, str],
) -> str:
    classes = "chain selected" if chain.selected else "chain"
    selected = " <span class='selected-badge'>selected</span>" if chain.selected else ""
    frames = []
    for node in chain.nodes:
        image_src = node_assets.get(str(node["memory_key"]))
        image_html = (
            f"<img src='{html.escape(image_src)}' alt='{html.escape(str(node['memory_key']))}'>"
            if image_src
            else "<div style='width:176px;height:150px;background:#eee'>missing</div>"
        )
        progress_delta = node.get("target_progress_delta")
        delta_text = "-" if progress_delta is None else f"{float(progress_delta):+.3f}"
        parent_similarity = node.get("parent_similarity")
        parent_text = "-" if parent_similarity is None else f"{float(parent_similarity):.3f}"
        strength = node.get("edge_strength")
        strength_text = "-" if strength is None else f"{float(strength):.3f}"
        forward_rank = node.get("forward_rank")
        reverse_rank = node.get("reverse_rank")
        reciprocal_text = "-" if forward_rank is None else f"{forward_rank}/{reverse_rank}"
        status_class = " hit" if node.get("target_hit") else (" plateau" if node.get("plateau") else "")
        frames.append(
            "\n".join(
                [
                    "<div class='frame'>",
                    image_html,
                    f"<div class='label{status_class}'>depth={node['depth']} target percentile={float(node['target_percentile']):.3f} ({delta_text})</div>",
                    f"<div class='mono'>target sim={float(node['target_similarity']):.3f}</div>",
                    f"<div class='mono'>parent sim={parent_text} edge={strength_text}</div>",
                    f"<div class='mono'>forward/reverse rank={reciprocal_text}</div>",
                    f"<div class='mono'>{html.escape(str(node.get('pano_id')))} #{node.get('capture_index')}</div>",
                    "</div>",
                ]
            )
        )
    return "\n".join(
        [
            f"<section class='{classes}'>",
            f"<h2>V{chain.view_order} capture {chain.capture_index}{selected}</h2>",
            "<div class='metrics'>",
            metric_html("target_hit", chain.target_hit),
            metric_html("hit_depth", chain.hit_depth),
            metric_html("final percentile", f"{chain.final_target_percentile:.3f}"),
            metric_html("parent mean (raw)", f"{chain.chain_mean_parent_similarity:.3f}"),
            metric_html("parent min (raw)", f"{chain.chain_min_parent_similarity:.3f}"),
            metric_html("reciprocal mean", f"{chain.chain_mean_strength:.3f}"),
            metric_html("reciprocal min", f"{chain.chain_bottleneck_strength:.3f}"),
            metric_html("plateau", chain.plateau_steps),
            metric_html("stop", chain.stop_reason),
            "</div><div class='frames'>",
            *frames,
            "</div></section>",
        ]
    )


def metric_html(label: str, value) -> str:
    return f"<div class='metric'><b>{html.escape(str(label))}</b> {html.escape(str(value))}</div>"


if __name__ == "__main__":
    raise SystemExit(main())
