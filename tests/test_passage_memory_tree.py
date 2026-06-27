from __future__ import annotations

import unittest

import numpy as np

from tools.experiments.build_passage_memory_tree import (
    build_bidirectional_alignment,
    build_memory_tree,
    score_candidate,
)


class PassageMemoryTreeTests(unittest.TestCase):
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
            near_duplicate_threshold=0.92,
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
            near_duplicate_threshold=0.92,
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


if __name__ == "__main__":
    unittest.main()
