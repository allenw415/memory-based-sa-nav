from __future__ import annotations

import unittest

import numpy as np

from tools.experiments.eval_memory_tree_child_similarity import RecipeScores
from tools.experiments.select_view_by_visual_chain import (
    DEFAULT_DINOV2_PATCH_MAX_PATCHES,
    DEFAULT_DINOV2_PATCH_MODEL,
    DEFAULT_DINOV2_PATCH_TOP_K,
    DEFAULT_DINOV2_TARGET_MATCH_MODE,
    DEFAULT_SIGLIP2_INDEX_PATH,
    DEFAULT_SIGLIP2_METADATA_PATH,
    ChainSummary,
    build_parser,
    build_view_chains,
    patch_target_similarity,
    patch_topk_similarity,
    patch_topk_similarity_matrix,
    select_chain,
)


def _chain(
    *,
    view_order: int,
    capture_index: int,
    target_hit: bool,
    hit_depth: int | None,
    hit_similarity: float | None,
    max_target_similarity: float,
    bottleneck: float,
) -> ChainSummary:
    return ChainSummary(
        view_order=view_order,
        capture_index=capture_index,
        selected=False,
        target_hit=target_hit,
        hit_depth=hit_depth,
        hit_similarity=hit_similarity,
        max_target_similarity=max_target_similarity,
        chain_bottleneck_similarity=bottleneck,
        chain_mean_similarity=bottleneck,
        stop_reason="test",
        nodes=(),
    )


def _item(pano_id: str, capture_index: int, heading: float) -> dict:
    return {
        "room_id": "Room 23",
        "pano_id": pano_id,
        "capture_index": capture_index,
        "capture_label": f"view_{capture_index}",
        "capture_heading": heading,
        "image_path": f"/tmp/{pano_id}_{capture_index}.png",
    }


class VisualChainDirectionTests(unittest.TestCase):
    def test_hit_chains_prefer_shallowest_hit_depth(self) -> None:
        selected = select_chain(
            [
                _chain(
                    view_order=1,
                    capture_index=0,
                    target_hit=True,
                    hit_depth=3,
                    hit_similarity=0.99,
                    max_target_similarity=0.99,
                    bottleneck=0.9,
                ),
                _chain(
                    view_order=2,
                    capture_index=1,
                    target_hit=True,
                    hit_depth=1,
                    hit_similarity=0.98,
                    max_target_similarity=0.98,
                    bottleneck=0.6,
                ),
            ],
            continuity_threshold=0.0,
        )

        self.assertEqual(selected.view_order, 2)

    def test_equal_hit_depth_prefers_higher_target_similarity(self) -> None:
        selected = select_chain(
            [
                _chain(
                    view_order=1,
                    capture_index=0,
                    target_hit=True,
                    hit_depth=2,
                    hit_similarity=0.91,
                    max_target_similarity=0.91,
                    bottleneck=0.8,
                ),
                _chain(
                    view_order=2,
                    capture_index=1,
                    target_hit=True,
                    hit_depth=2,
                    hit_similarity=0.96,
                    max_target_similarity=0.96,
                    bottleneck=0.7,
                ),
            ],
            continuity_threshold=0.0,
        )

        self.assertEqual(selected.view_order, 2)

    def test_same_hit_depth_and_target_similarity_prefers_higher_bottleneck(self) -> None:
        selected = select_chain(
            [
                _chain(
                    view_order=1,
                    capture_index=0,
                    target_hit=True,
                    hit_depth=2,
                    hit_similarity=0.95,
                    max_target_similarity=0.95,
                    bottleneck=0.5,
                ),
                _chain(
                    view_order=2,
                    capture_index=1,
                    target_hit=True,
                    hit_depth=2,
                    hit_similarity=0.95,
                    max_target_similarity=0.95,
                    bottleneck=0.8,
                ),
            ],
            continuity_threshold=0.0,
        )

        self.assertEqual(selected.view_order, 2)

    def test_without_hit_prefers_target_similarity_then_continuity(self) -> None:
        selected = select_chain(
            [
                _chain(
                    view_order=1,
                    capture_index=0,
                    target_hit=False,
                    hit_depth=None,
                    hit_similarity=None,
                    max_target_similarity=0.7,
                    bottleneck=0.9,
                ),
                _chain(
                    view_order=2,
                    capture_index=1,
                    target_hit=False,
                    hit_depth=None,
                    hit_similarity=None,
                    max_target_similarity=0.8,
                    bottleneck=0.4,
                ),
                _chain(
                    view_order=3,
                    capture_index=2,
                    target_hit=False,
                    hit_depth=None,
                    hit_similarity=None,
                    max_target_similarity=0.8,
                    bottleneck=0.7,
                ),
            ],
            continuity_threshold=0.0,
        )

        self.assertEqual(selected.view_order, 3)

    def test_chain_generation_ranks_by_similarity_only(self) -> None:
        items = [
            _item("current", 0, 0.0),
            _item("metadata_preferred", 0, 0.0),
            _item("similarity_preferred", 7, 180.0),
        ]
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(0, 1, 2),
            similarity_matrix=np.asarray(
                [
                    [1.0, 0.2, 0.9],
                    [0.2, 1.0, 0.1],
                    [0.9, 0.1, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        chains = build_view_chains(
            items=items,
            recipe=recipe,
            target_similarities={0: 0.0, 1: 0.1, 2: 0.2},
            current_view_indices=(0,),
            candidate_indices=(1, 2),
            max_depth=1,
            branching_factor=1,
            target_hit_threshold=0.98,
        )

        self.assertEqual(chains[0].nodes[1]["item_index"], 2)

    def test_chain_generation_stops_when_target_is_hit(self) -> None:
        items = [
            _item("current", 0, 0.0),
            _item("step", 1, 45.0),
            _item("target_like", 2, 90.0),
            _item("after_hit", 3, 135.0),
        ]
        recipe = RecipeScores(
            name="synthetic",
            item_indices=(0, 1, 2, 3),
            similarity_matrix=np.asarray(
                [
                    [1.0, 0.9, 0.1, 0.0],
                    [0.9, 1.0, 0.85, 0.0],
                    [0.1, 0.85, 1.0, 0.99],
                    [0.0, 0.0, 0.99, 1.0],
                ],
                dtype=np.float32,
            ),
        )

        chains = build_view_chains(
            items=items,
            recipe=recipe,
            target_similarities={0: 0.0, 1: 0.1, 2: 0.99, 3: 0.2},
            current_view_indices=(0,),
            candidate_indices=(1, 2, 3),
            max_depth=0,
            branching_factor=1,
            target_hit_threshold=0.98,
        )

        self.assertEqual([node["item_index"] for node in chains[0].nodes], [0, 1, 2])
        self.assertEqual(chains[0].stop_reason, "target_hit")

    def test_parser_defaults_match_visual_chain_experiment(self) -> None:
        args = build_parser().parse_args(
            [
                "--target-image",
                "/tmp/target.png",
            ]
        )

        self.assertEqual(args.recipe, "salad_full")
        self.assertEqual(args.max_depth, 0)
        self.assertEqual(args.branching_factor, 1)
        self.assertAlmostEqual(args.target_hit_threshold, 0.9)

    def test_parser_accepts_siglip2_full_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "--target-image",
                "/tmp/target.png",
                "--recipe",
                "siglip2_full",
            ]
        )

        self.assertEqual(args.recipe, "siglip2_full")
        self.assertEqual(args.siglip2_index_path, DEFAULT_SIGLIP2_INDEX_PATH)
        self.assertEqual(args.siglip2_metadata_path, DEFAULT_SIGLIP2_METADATA_PATH)

    def test_parser_accepts_dinov2_patch_recipe(self) -> None:
        args = build_parser().parse_args(
            [
                "--target-image",
                "/tmp/target.png",
                "--recipe",
                "dinov2_patch_topk",
            ]
        )

        self.assertEqual(args.recipe, "dinov2_patch_topk")
        self.assertEqual(args.dinov2_patch_model, DEFAULT_DINOV2_PATCH_MODEL)
        self.assertEqual(args.dinov2_patch_top_k, DEFAULT_DINOV2_PATCH_TOP_K)
        self.assertEqual(args.dinov2_patch_max_patches, DEFAULT_DINOV2_PATCH_MAX_PATCHES)
        self.assertEqual(args.dinov2_target_match_mode, DEFAULT_DINOV2_TARGET_MATCH_MODE)

    def test_target_to_candidate_mode_rewards_contained_target_patch(self) -> None:
        candidate = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        target = np.asarray(
            [
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

        containment = patch_target_similarity(
            candidate_features=candidate,
            target_features=target,
            top_k=3,
            target_match_mode="target_to_candidate",
        )
        symmetric = patch_target_similarity(
            candidate_features=candidate,
            target_features=target,
            top_k=3,
            target_match_mode="symmetric",
        )

        self.assertGreater(containment, symmetric)
        self.assertAlmostEqual(containment, 1.0)

    def test_patch_topk_similarity_rewards_shared_patch_matches(self) -> None:
        source = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        target_with_overlap = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.7, 0.7],
            ],
            dtype=np.float32,
        )
        target_without_overlap = np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )

        self.assertGreater(
            patch_topk_similarity(source, target_with_overlap, top_k=2),
            patch_topk_similarity(source, target_without_overlap, top_k=2),
        )

    def test_patch_topk_similarity_matrix_is_symmetric(self) -> None:
        features = np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.5, 0.5]],
                [[-1.0, 0.0], [0.0, -1.0]],
            ],
            dtype=np.float32,
        )

        matrix = patch_topk_similarity_matrix(features, top_k=1)

        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), np.ones(3, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
