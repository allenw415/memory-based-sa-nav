from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from tools.experiments.eval_memory_tree_child_similarity import RecipeScores
from tools.experiments.select_view_by_dynamic_retrieval_chain import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_MAX_DEPTH,
    DEFAULT_RECIPROCAL_TOP_K,
    DEFAULT_RETRIEVAL_TOP_K,
    DynamicChainSummary,
    SimilarityRankCache,
    build_dynamic_view_chains,
    build_parser,
    image_data_uri,
    reciprocal_rank_strength,
    select_dynamic_chain,
    target_similarity_percentiles,
)


def _item(name: str, capture_index: int) -> dict:
    return {
        "room_id": "Room 8",
        "pano_id": name,
        "capture_index": capture_index,
        "capture_label": f"view_{capture_index}",
        "capture_heading": float(capture_index * 45),
        "image_path": f"/tmp/{name}_{capture_index}.png",
    }


def _summary(
    *,
    view_order: int,
    target_hit: bool,
    hit_depth: int | None,
    final_percentile: float,
    bottleneck: float,
    mean_parent_similarity: float | None = None,
    min_parent_similarity: float | None = None,
) -> DynamicChainSummary:
    parent_mean = bottleneck if mean_parent_similarity is None else mean_parent_similarity
    parent_min = parent_mean if min_parent_similarity is None else min_parent_similarity
    return DynamicChainSummary(
        view_order=view_order,
        capture_index=view_order - 1,
        selected=False,
        target_hit=target_hit,
        hit_depth=hit_depth,
        hit_target_percentile=0.99 if target_hit else None,
        final_target_percentile=final_percentile,
        max_target_percentile=final_percentile,
        chain_min_parent_similarity=parent_min,
        chain_mean_parent_similarity=parent_mean,
        chain_bottleneck_strength=bottleneck,
        chain_mean_strength=bottleneck,
        plateau_steps=0,
        stop_reason="test",
        nodes=(),
    )


class DynamicRetrievalChainDirectionTests(unittest.TestCase):
    def test_image_data_uri_sniffs_content_instead_of_misleading_suffix(self) -> None:
        with TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "actually_jpeg.png"
            image_path.write_bytes(b"\xff\xd8\xff\xe0test")

            uri = image_data_uri(image_path)

        self.assertEqual(uri, "data:image/jpeg;base64,/9j/4HRlc3Q=")

    def test_parser_matches_standalone_experiment_defaults(self) -> None:
        args = build_parser().parse_args(
            [
                "--target-image",
                "/tmp/passage.png",
                "--expected-view",
                "V3",
            ]
        )

        self.assertEqual(args.max_depth, DEFAULT_MAX_DEPTH)
        self.assertEqual(args.beam_width, DEFAULT_BEAM_WIDTH)
        self.assertEqual(args.retrieval_top_k, DEFAULT_RETRIEVAL_TOP_K)
        self.assertEqual(args.reciprocal_top_k, DEFAULT_RECIPROCAL_TOP_K)
        self.assertEqual(args.expected_view, "V3")

    def test_target_similarity_uses_per_image_room_percentile(self) -> None:
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(0, 1, 2),
            similarity_matrix=np.asarray(
                [
                    [1.0, 0.2, 0.4],
                    [0.2, 1.0, 0.8],
                    [0.4, 0.8, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        percentiles = target_similarity_percentiles(
            recipe=recipe,
            target_similarities={0: 0.5, 1: 0.1, 2: 0.6},
        )

        self.assertEqual(percentiles[0], 1.0)
        self.assertEqual(percentiles[1], 0.0)
        self.assertAlmostEqual(percentiles[2], 0.5)

    def test_similarity_rank_cache_preserves_order_and_reuses_query(self) -> None:
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(10, 20, 30, 40),
            similarity_matrix=np.asarray(
                [
                    [1.0, 0.8, 0.8, 0.4],
                    [0.8, 1.0, 0.7, 0.5],
                    [0.8, 0.7, 1.0, 0.6],
                    [0.4, 0.5, 0.6, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        cache = SimilarityRankCache(recipe=recipe, pool_indices=recipe.item_indices)

        ranked = cache.ranked_candidates(
            query_index=10,
            allowed_indices=frozenset({20, 30, 40}),
            excluded_indices=(20,),
            limit=3,
        )

        self.assertEqual(ranked, [30, 40])
        self.assertEqual(cache.rank(query_index=10, candidate_index=20), 1)
        self.assertEqual(cache.rank(query_index=10, candidate_index=30), 2)
        self.assertEqual(cache.rank(query_index=10, candidate_index=40), 3)
        self.assertEqual(cache.rank(query_index=10, candidate_index=10), 4)
        self.assertEqual(cache.cached_query_count, 1)
        self.assertAlmostEqual(cache.similarity(10, 20), 0.8)

    def test_dynamic_search_selects_root_with_progressive_target_chain(self) -> None:
        items = [
            _item("current_a", 0),
            _item("current_b", 1),
            _item("a_step", 2),
            _item("a_target", 3),
            _item("b_step", 4),
        ]
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(0, 1, 2, 3, 4),
            similarity_matrix=np.asarray(
                [
                    [1.00, 0.05, 0.95, 0.10, 0.20],
                    [0.05, 1.00, 0.20, 0.10, 0.96],
                    [0.95, 0.20, 1.00, 0.94, 0.10],
                    [0.10, 0.10, 0.94, 1.00, 0.05],
                    [0.20, 0.96, 0.10, 0.05, 1.00],
                ],
                dtype=np.float32,
            ),
        )
        target_percentiles = {0: 0.10, 1: 0.20, 2: 0.60, 3: 0.99, 4: 0.25}

        chains = build_dynamic_view_chains(
            items=items,
            recipe=recipe,
            target_similarities=target_percentiles,
            target_percentiles=target_percentiles,
            current_view_indices=(0, 1),
            candidate_indices=(2, 3, 4),
            max_depth=2,
            beam_width=2,
            retrieval_top_k=1,
            reciprocal_top_k=4,
            target_hit_percentile=0.98,
            min_target_progress=0.02,
            plateau_tolerance=0.01,
            max_plateau_steps=1,
            duplicate_rank=1,
        )
        selected = select_dynamic_chain(chains)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.view_order, 1)
        self.assertTrue(selected.target_hit)
        self.assertEqual([node["item_index"] for node in selected.nodes], [0, 2, 3])

    def test_reverse_rank_filter_rejects_one_way_visual_hub(self) -> None:
        items = [_item("current", 0), _item("hub", 1), _item("hub_favorite", 2)]
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(0, 1, 2),
            similarity_matrix=np.asarray(
                [
                    [1.00, 0.90, 0.10],
                    [0.90, 1.00, 0.99],
                    [0.10, 0.99, 1.00],
                ],
                dtype=np.float32,
            ),
        )

        chains = build_dynamic_view_chains(
            items=items,
            recipe=recipe,
            target_similarities={0: 0.1, 1: 0.8, 2: 0.0},
            target_percentiles={0: 0.1, 1: 0.8, 2: 0.0},
            current_view_indices=(0,),
            candidate_indices=(1,),
            max_depth=1,
            beam_width=1,
            retrieval_top_k=1,
            reciprocal_top_k=1,
            target_hit_percentile=0.98,
            min_target_progress=0.02,
            plateau_tolerance=0.01,
            max_plateau_steps=0,
            duplicate_rank=1,
        )

        self.assertEqual(len(chains[0].nodes), 1)
        self.assertEqual(chains[0].stop_reason, "no_valid_reciprocal_progress")

    def test_search_allows_one_plateau_before_progress(self) -> None:
        items = [_item("current", 0), _item("plateau", 1), _item("progress", 2)]
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(0, 1, 2),
            similarity_matrix=np.asarray(
                [
                    [1.00, 0.95, 0.10],
                    [0.95, 1.00, 0.94],
                    [0.10, 0.94, 1.00],
                ],
                dtype=np.float32,
            ),
        )

        chains = build_dynamic_view_chains(
            items=items,
            recipe=recipe,
            target_similarities={0: 0.20, 1: 0.195, 2: 0.99},
            target_percentiles={0: 0.20, 1: 0.195, 2: 0.99},
            current_view_indices=(0,),
            candidate_indices=(1, 2),
            max_depth=2,
            beam_width=1,
            retrieval_top_k=1,
            reciprocal_top_k=2,
            target_hit_percentile=0.98,
            min_target_progress=0.02,
            plateau_tolerance=0.01,
            max_plateau_steps=1,
            duplicate_rank=1,
        )

        self.assertTrue(chains[0].target_hit)
        self.assertEqual(chains[0].plateau_steps, 1)
        self.assertTrue(chains[0].nodes[1]["plateau"])
        self.assertAlmostEqual(chains[0].chain_mean_parent_similarity, 0.945)
        self.assertAlmostEqual(chains[0].chain_min_parent_similarity, 0.94)

    def test_target_hit_chains_prefer_parent_mean_not_root_evidence_or_hit_depth(self) -> None:
        shallow = _summary(
            view_order=2,
            target_hit=True,
            hit_depth=1,
            final_percentile=0.99,
            bottleneck=0.90,
            mean_parent_similarity=0.60,
            min_parent_similarity=0.60,
        )
        deep = _summary(
            view_order=3,
            target_hit=True,
            hit_depth=3,
            final_percentile=0.99,
            bottleneck=0.60,
            mean_parent_similarity=0.80,
            min_parent_similarity=0.40,
        )
        shallow = DynamicChainSummary(
            **{
                **shallow.__dict__,
                "nodes": ({"target_similarity": 0.30}, {"edge_strength": 0.90}),
            }
        )
        deep = DynamicChainSummary(
            **{
                **deep.__dict__,
                "nodes": ({"target_similarity": 0.50}, {"edge_strength": 0.80}),
            }
        )
        selected = select_dynamic_chain(
            [
                _summary(
                    view_order=1,
                    target_hit=False,
                    hit_depth=None,
                    final_percentile=0.97,
                    bottleneck=0.95,
                ),
                shallow,
                deep,
            ]
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.view_order, 3)

    def test_equal_parent_mean_uses_parent_minimum_as_tie_break(self) -> None:
        weak_edge = _summary(
            view_order=1,
            target_hit=True,
            hit_depth=2,
            final_percentile=0.99,
            bottleneck=0.95,
            mean_parent_similarity=0.75,
            min_parent_similarity=0.40,
        )
        smooth = _summary(
            view_order=2,
            target_hit=True,
            hit_depth=4,
            final_percentile=0.99,
            bottleneck=0.50,
            mean_parent_similarity=0.75,
            min_parent_similarity=0.65,
        )

        selected = select_dynamic_chain([weak_edge, smooth])

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.view_order, 2)

    def test_no_target_hit_abstains_instead_of_forcing_a_direction(self) -> None:
        smooth_but_wrong = _summary(
            view_order=1,
            target_hit=False,
            hit_depth=None,
            final_percentile=0.97,
            bottleneck=0.99,
            mean_parent_similarity=0.95,
            min_parent_similarity=0.94,
        )
        closer_to_target = _summary(
            view_order=2,
            target_hit=False,
            hit_depth=None,
            final_percentile=0.979,
            bottleneck=0.60,
            mean_parent_similarity=0.70,
            min_parent_similarity=0.55,
        )

        selected = select_dynamic_chain([smooth_but_wrong, closer_to_target])

        self.assertIsNone(selected)

    def test_reciprocal_strength_is_one_for_mutual_top_one(self) -> None:
        self.assertEqual(
            reciprocal_rank_strength(
                forward_rank=1,
                reverse_rank=1,
                retrieval_top_k=20,
                reciprocal_top_k=20,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
