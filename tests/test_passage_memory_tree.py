from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.experiments.build_passage_memory_tree import (
    build_parser,
    build_bidirectional_alignment,
    build_memory_tree,
    choose_best_bridge,
    expand_parent_tree,
    external_target_item,
    load_or_compute_dinov2_patch_similarity_matrix,
    load_or_encode_dinov2_patch_features,
    memory_key,
    public_memory_item,
    score_candidate,
)


class Dinov2PatchCacheTests(unittest.TestCase):
    def test_patch_feature_cache_reuses_existing_images_and_encodes_only_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a = root / "a.png"
            image_b = root / "b.png"
            image_c = root / "c.png"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            image_c.write_bytes(b"c")
            output_dir = root / "cache"
            calls: list[list[str]] = []

            def fake_encode(paths, *, model_name, max_patches, device, batch_size):
                calls.append([Path(path).name for path in paths])
                base = len(calls) * 10
                return np.asarray(
                    [np.full((2, 3), base + offset, dtype=np.float32) for offset, _ in enumerate(paths)],
                    dtype=np.float32,
                )

            with patch(
                "tools.experiments.build_passage_memory_tree.encode_dinov2_patch_paths",
                side_effect=fake_encode,
            ):
                first = load_or_encode_dinov2_patch_features(
                    image_paths=[image_a, image_b],
                    memory_keys=["a:0", "b:0"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=2,
                )
                second = load_or_encode_dinov2_patch_features(
                    image_paths=[image_a, image_b],
                    memory_keys=["a:0", "b:0"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=2,
                )
                third = load_or_encode_dinov2_patch_features(
                    image_paths=[image_a, image_b, image_c],
                    memory_keys=["a:0", "b:0", "c:0"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=2,
                )
                same_image_new_label = load_or_encode_dinov2_patch_features(
                    image_paths=[image_a],
                    memory_keys=["renamed-passage:99"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=2,
                )

        self.assertEqual(calls, [["a.png", "b.png"], ["c.png"]])
        np.testing.assert_allclose(second, first)
        np.testing.assert_allclose(third[:2], first)
        np.testing.assert_allclose(same_image_new_label[0], first[0])
        self.assertEqual(third.shape, (3, 2, 3))

    def test_corrupt_patch_feature_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "a.png"
            image.write_bytes(b"a")
            output_dir = root / "cache"
            cache_path = (
                output_dir
                / "embedding_cache"
                / "dinov2_patch_topk_facebook_dinov2-base_p2.npz"
            )
            cache_path.parent.mkdir(parents=True)
            cache_path.write_bytes(b"not-a-valid-npz")

            with patch(
                "tools.experiments.build_passage_memory_tree.encode_dinov2_patch_paths",
                return_value=np.ones((1, 2, 3), dtype=np.float32),
            ):
                features = load_or_encode_dinov2_patch_features(
                    image_paths=[image],
                    memory_keys=["a:0"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=1,
                )

            self.assertEqual(features.shape, (1, 2, 3))
            with np.load(cache_path, allow_pickle=False) as payload:
                self.assertEqual(payload["patch_features"].shape, (1, 2, 3))
            self.assertFalse(cache_path.with_name(cache_path.name + ".tmp").exists())

    def test_patch_feature_records_stay_in_memory_within_process(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = root / "a.png"
            image.write_bytes(b"a")
            output_dir = root / "cache"

            with patch(
                "tools.experiments.build_passage_memory_tree.encode_dinov2_patch_paths",
                return_value=np.ones((1, 2, 3), dtype=np.float32),
            ):
                first = load_or_encode_dinov2_patch_features(
                    image_paths=[image],
                    memory_keys=["a:0"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=1,
                )

            with patch(
                "tools.experiments.build_passage_memory_tree.np.load",
                side_effect=AssertionError("disk cache should not be decompressed twice"),
            ):
                second = load_or_encode_dinov2_patch_features(
                    image_paths=[image],
                    memory_keys=["renamed:1"],
                    output_dir=output_dir,
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    device="cpu",
                    batch_size=1,
                )

            np.testing.assert_allclose(first, second)

    def test_patch_similarity_matrix_cache_reuses_exact_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_a = root / "a.png"
            image_b = root / "b.png"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            features = np.asarray(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.5, 0.5]],
                ],
                dtype=np.float32,
            )
            calls = 0

            def fake_matrix(_features, *, top_k):
                nonlocal calls
                calls += 1
                return np.asarray([[1.0, 0.25], [0.25, 1.0]], dtype=np.float32)

            with patch(
                "tools.experiments.build_passage_memory_tree.patch_topk_similarity_matrix",
                side_effect=fake_matrix,
            ):
                first = load_or_compute_dinov2_patch_similarity_matrix(
                    patch_features=features,
                    image_paths=[image_a, image_b],
                    memory_keys=["a:0", "b:0"],
                    output_dir=root / "cache",
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    top_k=1,
                )
                second = load_or_compute_dinov2_patch_similarity_matrix(
                    patch_features=features,
                    image_paths=[image_a, image_b],
                    memory_keys=["a:0", "b:0"],
                    output_dir=root / "cache",
                    model_name="facebook/dinov2-base",
                    max_patches=2,
                    top_k=1,
                )

        self.assertEqual(calls, 1)
        np.testing.assert_allclose(first, second)


class PassageMemoryTreeParserTests(unittest.TestCase):
    def test_eight_view_defaults_to_bidirection(self) -> None:
        args = build_parser().parse_args(
            [
                "--room-id",
                "Room 23",
                "--target-passage-image",
                "/tmp/target.png",
                "--current-pano-id",
                "pano",
            ]
        )

        self.assertEqual(args.alignment_mode, "bidirection")
        self.assertEqual(args.tree_expansion_score_mode, "path_continuity")
        self.assertEqual(args.batch_size, 32)

    def test_parser_accepts_dinov2_patch_bidirection_options(self) -> None:
        args = build_parser().parse_args(
            [
                "--room-id",
                "Room 23",
                "--target-passage-image",
                "/tmp/target.png",
                "--current-pano-id",
                "pano",
                "--similarity-backend",
                "dinov2_patch_topk",
                "--bridge-selection-mode",
                "bridge_then_continuity",
                "--branching-factor",
                "3",
                "--max-depth",
                "5",
            ]
        )

        self.assertEqual(args.similarity_backend, "dinov2_patch_topk")
        self.assertEqual(args.bridge_selection_mode, "bridge_then_continuity")
        self.assertEqual(args.dinov2_patch_model, "facebook/dinov2-base")
        self.assertEqual(args.dinov2_patch_top_k, 5)
        self.assertEqual(args.dinov2_patch_max_patches, 24)
        self.assertEqual(args.branching_factor, 3)
        self.assertEqual(args.max_depth, 5)
        self.assertEqual(args.bridge_similarity_tie_margin, 0.01)
        self.assertFalse(args.allow_same_bridge_item)


class PassageMemoryTreeTests(unittest.TestCase):
    def test_external_target_item_keeps_external_image_path(self) -> None:
        item = external_target_item(
            Path("/tmp/R8P7 target.png"),
            room_id="Room 8",
            target_passage_label="image_R8P7_target",
        )

        self.assertEqual(memory_key(item), "external_R8P7_target:0")
        self.assertEqual(item["image_path"], "/tmp/R8P7 target.png")
        self.assertTrue(public_memory_item(item)["external_image"])

    def test_tree_stops_when_frontier_node_is_close_to_current_view(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("mid", "Room 8", 0),
            _item("current-like", "Room 8", 0),
        ]
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        result = build_memory_tree(
            room_items=items,
            room_embeddings=embeddings,
            current_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            target_index=0,
            target_passage_label="R8P7",
            current_context={"mode": "memory_capture", **items[2]},
            branching_factor=1,
            max_depth=3,
            current_similarity_threshold=0.98,
            current_rank_guard=1,
            include_embeddings=False,
        )

        self.assertTrue(result["stop"]["found"])
        self.assertEqual(result["stop"]["reason"], "current_similarity_threshold_reached")
        self.assertEqual(
            [node["memory_key"] for node in result["navigation_chain_current_to_target"]],
            ["current-like:0", "mid:0", "target:0"],
        )

    def test_near_duplicate_penalty_does_not_penalize_same_pano_by_itself(self) -> None:
        pairwise = np.asarray(
            [
                [1.0, 0.95, 0.1],
                [0.95, 1.0, 0.1],
                [0.1, 0.1, 1.0],
            ],
            dtype=np.float32,
        )
        current_similarities = np.asarray([0.0, 0.2, 0.9], dtype=np.float32)

        duplicate = score_candidate(
            candidate_index=1,
            parent_index=0,
            path_indices=[0],
            pairwise_similarities=pairwise,
            current_similarities=current_similarities,
            near_duplicate_threshold=0.82,
            near_duplicate_penalty_weight=0.15,
            parent_similarity_weight=1.0,
            current_similarity_weight=0.0,
        )
        same_pano_different_view = score_candidate(
            candidate_index=2,
            parent_index=0,
            path_indices=[0],
            pairwise_similarities=pairwise,
            current_similarities=current_similarities,
            near_duplicate_threshold=0.82,
            near_duplicate_penalty_weight=0.15,
            parent_similarity_weight=1.0,
            current_similarity_weight=0.0,
        )

        self.assertEqual(duplicate["near_duplicate_count"], 1)
        self.assertAlmostEqual(duplicate["near_duplicate_penalty"], 0.15)
        self.assertEqual(same_pano_different_view["near_duplicate_count"], 0)
        self.assertAlmostEqual(same_pano_different_view["near_duplicate_penalty"], 0.0)

    def test_outputs_best_leaf_when_threshold_is_not_reached(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("near", "Room 8", 0),
        ]
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.2],
            ],
            dtype=np.float32,
        )

        result = build_memory_tree(
            room_items=items,
            room_embeddings=embeddings,
            current_embedding=np.asarray([0.0, 1.0], dtype=np.float32),
            target_index=0,
            target_passage_label="R8P7",
            current_context={"mode": "image", "image_path": "/tmp/current.png"},
            branching_factor=1,
            max_depth=1,
            current_similarity_threshold=0.99,
            current_rank_guard=1,
            include_embeddings=False,
        )

        self.assertFalse(result["stop"]["found"])
        self.assertEqual(result["stop"]["reason"], "max_depth_not_close_enough")
        self.assertEqual(result["best_node_id"], "n1")

    def test_bidirectional_alignment_directly_selects_near_duplicate_current_view(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("passage-leaf", "Room 8", 1),
            _item("current-direct", "Room 8", 2),
            _item("direct-leaf", "Room 8", 3),
            _item("current-bridge", "Room 8", 4),
            _item("bridge-leaf", "Room 8", 5),
        ]
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.96, 0.28, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = build_bidirectional_alignment(
            room_items=items,
            room_embeddings=embeddings,
            target_index=0,
            target_passage_label="R8P7",
            current_view_indices=[2, 4],
            current_context={"mode": "pano_8_views", "pano_id": "current", "view_count": 2},
            branching_factor=1,
            max_depth=1,
            near_duplicate_threshold=0.82,
            include_embeddings=False,
        )

        self.assertEqual(result["selection_reason"], "root_target_direct_match")
        self.assertEqual(result["selected_view"]["capture_index"], 2)
        self.assertTrue(result["selected_view"]["root_target_direct_match"])
        self.assertGreaterEqual(result["selected_view"]["root_target_similarity"], 0.82)

    def test_bidirectional_alignment_ignores_root_target_similarity_without_direct_match(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("passage-leaf", "Room 8", 1),
            _item("current-bridge", "Room 8", 2),
            _item("bridge-leaf", "Room 8", 3),
            _item("current-similar", "Room 8", 4),
            _item("similar-leaf", "Room 8", 5),
        ]
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.0, 0.6],
                [0.0, 0.95, 0.31],
            ],
            dtype=np.float32,
        )

        result = build_bidirectional_alignment(
            room_items=items,
            room_embeddings=embeddings,
            target_index=0,
            target_passage_label="R8P7",
            current_view_indices=[2, 4],
            current_context={"mode": "pano_8_views", "pano_id": "current", "view_count": 2},
            branching_factor=1,
            max_depth=1,
            include_embeddings=False,
        )

        self.assertEqual(result["selection_reason"], "best_bridge_score")
        self.assertEqual(result["selected_view"]["capture_index"], 2)
        self.assertFalse(result["selected_view"]["root_target_direct_match"])
        self.assertEqual(result["selected_view"]["view_score"], result["selected_view"]["total_score"])
        self.assertEqual(result["configuration"]["view_scoring_formula"], "bridge_total_score")
        similar_alignment = next(item for item in result["view_alignments"] if item["capture_index"] == 4)
        self.assertGreater(similar_alignment["root_target_similarity"], result["selected_view"]["root_target_similarity"])

    def test_bidirectional_alignment_selects_view_with_best_bridge(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("passage-leaf", "Room 8", 1),
            _item("current-bad", "Room 8", 2),
            _item("bad-leaf", "Room 8", 3),
            _item("current-good", "Room 8", 4),
            _item("good-leaf", "Room 8", 5),
        ]
        embeddings = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.6, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.2, 0.98],
                [0.0, 1.0, 0.0],
                [0.7, 0.7, 0.0],
            ],
            dtype=np.float32,
        )

        result = build_bidirectional_alignment(
            room_items=items,
            room_embeddings=embeddings,
            target_index=0,
            target_passage_label="R8P7",
            current_view_indices=[2, 4],
            current_context={"mode": "pano_8_views", "pano_id": "current", "view_count": 2},
            branching_factor=1,
            max_depth=1,
            include_embeddings=False,
        )

        self.assertEqual(result["selected_view"]["capture_index"], 4)
        self.assertEqual(result["selected_alignment"]["rank"], 1)
        self.assertEqual(result["selected_alignment"]["best_bridge"]["current_memory_key"], "good-leaf:5")
        self.assertEqual(result["selected_alignment"]["best_bridge"]["passage_memory_key"], "passage-leaf:1")
        selected_current_keys = {
            node["memory_key"] for node in result["selected_alignment"]["current_tree"]["nodes"]
        }
        passage_keys = {node["memory_key"] for node in result["passage_tree"]["nodes"]}
        self.assertNotIn("target:0", selected_current_keys)
        self.assertNotIn("current-bad:2", selected_current_keys)
        self.assertNotIn("current-bad:2", passage_keys)
        self.assertNotIn("current-good:4", passage_keys)
        self.assertTrue(
            result["configuration"]["current_expansion_excludes_target_passage_and_current_views"]
        )


    def test_path_continuity_expansion_prefers_smoother_full_path(self) -> None:
        items = [
            _item("root", "Room 8", 0),
            _item("smooth-mid", "Room 8", 1),
            _item("spiky-mid", "Room 8", 2),
            _item("smooth-next", "Room 8", 3),
            _item("spiky-next", "Room 8", 4),
        ]
        pairwise = np.eye(5, dtype=np.float32)
        pairwise[0, 1] = pairwise[1, 0] = 0.8
        pairwise[0, 2] = pairwise[2, 0] = 0.4
        pairwise[1, 3] = pairwise[3, 1] = 0.8
        pairwise[2, 4] = pairwise[4, 2] = 1.0

        tree = expand_parent_tree(
            room_items=items,
            embeddings=np.eye(5, dtype=np.float32),
            pairwise_similarities=pairwise,
            root_index=0,
            node_prefix="n",
            tree_role="test",
            branching_factor=2,
            max_depth=2,
            near_duplicate_threshold=1.1,
            near_duplicate_penalty_weight=0.0,
            parent_similarity_weight=1.0,
            include_embeddings=False,
            tree_expansion_score_mode="path_continuity",
        )

        depth_two = [node for node in tree["nodes"] if node["depth"] == 2]
        selected_keys = {node["memory_key"] for node in depth_two[:2]}
        self.assertIn("smooth-next:3", selected_keys)
        smooth = next(node for node in depth_two if node["memory_key"] == "smooth-next:3")
        spiky = next(node for node in depth_two if node["memory_key"] == "spiky-next:4")
        self.assertGreater(smooth["path_mean_continuity"], spiky["path_mean_continuity"])
        self.assertAlmostEqual(spiky["sim_to_parent"], 1.0)

    def test_bidirection_bridge_excludes_same_memory_item_by_default(self) -> None:
        current_tree = _fake_tree(
            [
                _fake_node("c0", 0, None, 0, None, "current-root"),
                _fake_node("c1", 1, "c0", 1, 0.95, "shared-leaf"),
            ]
        )
        passage_tree = _fake_tree(
            [
                _fake_node("p0", 0, None, 2, None, "passage-root"),
                _fake_node("p1", 1, "p0", 1, 0.95, "shared-leaf"),
                _fake_node("p2", 1, "p0", 3, 0.90, "real-bridge"),
            ]
        )
        pairwise = np.eye(4, dtype=np.float32)
        pairwise[1, 3] = pairwise[3, 1] = 0.88

        result = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=[current_tree["nodes"][1]],
            passage_leaves=passage_tree["nodes"][1:],
            pairwise_similarities=pairwise,
            bridge_depth_penalty=0.02,
            bridge_continuity_weight=0.25,
        )

        self.assertEqual(result["passage_memory_key"], "real-bridge:3")
        self.assertFalse(result["same_bridge_item"])
        self.assertTrue(result["exclude_same_bridge_item"])
        self.assertAlmostEqual(result["bridge_similarity"], 0.88)

    def test_bidirection_bridge_can_allow_same_memory_item_for_ablation(self) -> None:
        current_tree = _fake_tree(
            [
                _fake_node("c0", 0, None, 0, None, "current-root"),
                _fake_node("c1", 1, "c0", 1, 0.95, "shared-leaf"),
            ]
        )
        passage_tree = _fake_tree(
            [
                _fake_node("p0", 0, None, 2, None, "passage-root"),
                _fake_node("p1", 1, "p0", 1, 0.95, "shared-leaf"),
                _fake_node("p2", 1, "p0", 3, 0.90, "real-bridge"),
            ]
        )
        pairwise = np.eye(4, dtype=np.float32)
        pairwise[1, 3] = pairwise[3, 1] = 0.88

        result = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=[current_tree["nodes"][1]],
            passage_leaves=passage_tree["nodes"][1:],
            pairwise_similarities=pairwise,
            bridge_depth_penalty=0.02,
            bridge_continuity_weight=0.25,
            exclude_same_bridge_item=False,
        )

        self.assertEqual(result["passage_memory_key"], "shared-leaf:1")
        self.assertTrue(result["same_bridge_item"])
        self.assertFalse(result["exclude_same_bridge_item"])
        self.assertAlmostEqual(result["bridge_similarity"], 1.0)

    def test_bidirection_bridge_uses_weighted_bridge_formula(self) -> None:
        current_tree = _fake_tree(
            [
                _fake_node("c0", 0, None, 0, None, "current-root"),
                _fake_node("c1", 1, "c0", 1, 0.8, "current-leaf"),
            ]
        )
        passage_tree = _fake_tree(
            [
                _fake_node("p0", 0, None, 2, None, "passage-root"),
                _fake_node("p1", 1, "p0", 3, 0.5, "high-bridge-low-chain"),
                _fake_node("p2", 1, "p0", 4, 0.85, "lower-bridge-smooth-chain"),
            ]
        )
        pairwise = np.eye(5, dtype=np.float32)
        pairwise[1, 3] = pairwise[3, 1] = 1.0
        pairwise[1, 4] = pairwise[4, 1] = 0.85

        result = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=[current_tree["nodes"][1]],
            passage_leaves=passage_tree["nodes"][1:],
            pairwise_similarities=pairwise,
            bridge_depth_penalty=0.02,
            bridge_continuity_weight=0.25,
        )

        self.assertEqual(result["passage_memory_key"], "high-bridge-low-chain:3")
        self.assertAlmostEqual(result["total_score"], 1.0 + 0.25 * (0.8 + 0.5) - 0.02 * 2)
        self.assertAlmostEqual(result["bidirection_score"], result["total_score"])
        self.assertEqual(result["bridge_score_mode"], "bidirection")

    def test_bidirection_bridge_uses_bridge_similarity_as_tiebreaker(self) -> None:
        current_tree = _fake_tree(
            [
                _fake_node("c0", 0, None, 0, None, "current-root"),
                _fake_node("c1", 1, "c0", 1, 0.8, "current-leaf"),
            ]
        )
        passage_tree = _fake_tree(
            [
                _fake_node("p0", 0, None, 2, None, "passage-root"),
                _fake_node("p1", 1, "p0", 3, 0.8, "same-bottleneck-higher-mean"),
                _fake_node("p2", 1, "p0", 4, 0.8, "same-bottleneck-lower-mean"),
            ]
        )
        pairwise = np.eye(5, dtype=np.float32)
        pairwise[1, 3] = pairwise[3, 1] = 0.9
        pairwise[1, 4] = pairwise[4, 1] = 0.85

        result = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=[current_tree["nodes"][1]],
            passage_leaves=passage_tree["nodes"][1:],
            pairwise_similarities=pairwise,
            bridge_depth_penalty=0.02,
            bridge_continuity_weight=0.25,
        )

        self.assertEqual(result["passage_memory_key"], "same-bottleneck-higher-mean:3")
        self.assertAlmostEqual(result["total_score"], 0.9 + 0.25 * (0.8 + 0.8) - 0.02 * 2)
        self.assertAlmostEqual(result["bridge_similarity"], 0.9)
        self.assertEqual(result["bridge_score_mode"], "bidirection")


    def test_bidirection_bridge_then_continuity_prefers_higher_bridge_first(self) -> None:
        current_tree = _fake_tree(
            [
                _fake_node("c0", 0, None, 0, None, "current-root"),
                _fake_node("c1", 1, "c0", 1, 0.95, "current-leaf"),
            ]
        )
        passage_tree = _fake_tree(
            [
                _fake_node("p0", 0, None, 2, None, "passage-root"),
                _fake_node("p1", 1, "p0", 3, 0.95, "smooth-lower-bridge"),
                _fake_node("p2", 1, "p0", 4, 0.40, "rough-higher-bridge"),
            ]
        )
        pairwise = np.eye(5, dtype=np.float32)
        pairwise[1, 3] = pairwise[3, 1] = 0.89
        pairwise[1, 4] = pairwise[4, 1] = 0.90

        result = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=[current_tree["nodes"][1]],
            passage_leaves=passage_tree["nodes"][1:],
            pairwise_similarities=pairwise,
            bridge_depth_penalty=0.02,
            bridge_continuity_weight=0.25,
            bridge_selection_mode="bridge_then_continuity",
        )

        self.assertEqual(result["passage_memory_key"], "rough-higher-bridge:4")
        self.assertAlmostEqual(result["total_score"], 0.90)
        self.assertEqual(result["bridge_score_mode"], "bridge_then_continuity")

    def test_bidirection_bridge_then_continuity_uses_continuity_for_tied_bridge(self) -> None:
        current_tree = _fake_tree(
            [
                _fake_node("c0", 0, None, 0, None, "current-root"),
                _fake_node("c1", 1, "c0", 1, 0.95, "current-leaf"),
            ]
        )
        passage_tree = _fake_tree(
            [
                _fake_node("p0", 0, None, 2, None, "passage-root"),
                _fake_node("p1", 1, "p0", 3, 0.95, "smooth-tied-bridge"),
                _fake_node("p2", 1, "p0", 4, 0.40, "rough-tied-bridge"),
            ]
        )
        pairwise = np.eye(5, dtype=np.float32)
        pairwise[1, 3] = pairwise[3, 1] = 0.90
        pairwise[1, 4] = pairwise[4, 1] = 0.90

        result = choose_best_bridge(
            current_tree=current_tree,
            passage_tree=passage_tree,
            current_leaves=[current_tree["nodes"][1]],
            passage_leaves=passage_tree["nodes"][1:],
            pairwise_similarities=pairwise,
            bridge_depth_penalty=0.02,
            bridge_continuity_weight=0.25,
            bridge_selection_mode="bridge_then_continuity",
        )

        self.assertEqual(result["passage_memory_key"], "smooth-tied-bridge:3")
        self.assertGreater(result["chain_bottleneck_similarity"], 0.89)
        self.assertAlmostEqual(result["total_score"], 0.90)

    def test_bridge_then_continuity_uses_root_target_similarity_for_tied_bridge(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("passage-leaf", "Room 8", 1),
            _item("current-low-target", "Room 8", 2),
            _item("low-leaf", "Room 8", 3),
            _item("current-high-target", "Room 8", 4),
            _item("high-leaf", "Room 8", 5),
        ]
        embeddings = np.zeros((len(items), 1), dtype=np.float32)
        pairwise = np.eye(len(items), dtype=np.float32)
        pairwise[0, 1] = pairwise[1, 0] = 0.95
        pairwise[2, 3] = pairwise[3, 2] = 0.95
        pairwise[4, 5] = pairwise[5, 4] = 0.95
        pairwise[3, 1] = pairwise[1, 3] = 0.90
        pairwise[5, 1] = pairwise[1, 5] = 0.90
        pairwise[0, 2] = pairwise[2, 0] = 0.20
        pairwise[0, 4] = pairwise[4, 0] = 0.80

        result = build_bidirectional_alignment(
            room_items=items,
            room_embeddings=embeddings,
            pairwise_similarities=pairwise,
            target_index=0,
            target_passage_label="R8P7",
            current_view_indices=[2, 4],
            current_context={"mode": "pano_8_views", "pano_id": "current", "view_count": 2},
            branching_factor=1,
            max_depth=1,
            near_duplicate_threshold=0.99,
            bridge_selection_mode="bridge_then_continuity",
            include_embeddings=False,
            similarity_backend="dinov2_patch_topk",
        )

        self.assertEqual(result["selected_view"]["capture_index"], 4)
        self.assertAlmostEqual(result["selected_alignment"]["best_bridge"]["bridge_similarity"], 0.90)
        self.assertEqual(
            result["configuration"]["view_scoring_formula"],
            "bridge_similarity first; root_target_similarity and continuity tie-breaks",
        )

    def test_bidirectional_alignment_can_rank_from_pairwise_matrix_override(self) -> None:
        items = [
            _item("target", "Room 8", 0),
            _item("passage-leaf", "Room 8", 1),
            _item("current-bad", "Room 8", 2),
            _item("bad-leaf", "Room 8", 3),
            _item("current-good", "Room 8", 4),
            _item("good-leaf", "Room 8", 5),
        ]
        embeddings = np.zeros((len(items), 1), dtype=np.float32)
        pairwise = np.eye(len(items), dtype=np.float32)
        pairwise[0, 1] = pairwise[1, 0] = 0.95
        pairwise[2, 3] = pairwise[3, 2] = 0.95
        pairwise[4, 5] = pairwise[5, 4] = 0.95
        pairwise[3, 1] = pairwise[1, 3] = 0.70
        pairwise[5, 1] = pairwise[1, 5] = 0.92

        result = build_bidirectional_alignment(
            room_items=items,
            room_embeddings=embeddings,
            pairwise_similarities=pairwise,
            target_index=0,
            target_passage_label="R8P7",
            current_view_indices=[2, 4],
            current_context={"mode": "pano_8_views", "pano_id": "current", "view_count": 2},
            branching_factor=1,
            max_depth=1,
            near_duplicate_threshold=0.99,
            bridge_selection_mode="bridge_then_continuity",
            include_embeddings=False,
            similarity_backend="dinov2_patch_topk",
        )

        self.assertEqual(result["selected_view"]["capture_index"], 4)
        self.assertEqual(result["configuration"]["similarity_backend"], "dinov2_patch_topk")
        self.assertEqual(result["configuration"]["bridge_selection_mode"], "bridge_then_continuity")
        self.assertEqual(result["selected_alignment"]["best_bridge"]["current_memory_key"], "good-leaf:5")

def _item(pano_id: str, room_id: str, capture_index: int) -> dict:
    return {
        "memory_index": len(pano_id) + capture_index,
        "room_id": room_id,
        "pano_id": pano_id,
        "capture_index": capture_index,
        "capture_label": "view",
        "capture_heading": float(capture_index * 45),
        "image_path": f"/tmp/{pano_id}_{capture_index}.png",
    }


def _fake_node(
    node_id: str,
    depth: int,
    parent_id: str | None,
    item_index: int,
    sim_to_parent: float | None,
    pano_id: str,
) -> dict:
    return {
        "id": node_id,
        "depth": depth,
        "parent_id": parent_id,
        "item_index": item_index,
        "sim_to_parent": sim_to_parent,
        "memory_key": f"{pano_id}:{item_index}",
    }


def _fake_tree(nodes: list[dict]) -> dict:
    return {
        "nodes": nodes,
        "edges": [
            {"source": node["parent_id"], "target": node["id"]}
            for node in nodes
            if node.get("parent_id") is not None
        ],
    }


if __name__ == "__main__":
    unittest.main()
