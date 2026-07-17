from __future__ import annotations

import unittest

from tools.experiments.eval_memory_tree_child_similarity import (
    EvaluationCase,
    expected_target_for_source,
    rank_candidates_by_similarity,
    summarize_expected_ranks,
)


def _item(
    pano_id: str,
    capture_index: int,
    heading: float,
    *,
    room_id: str = "Room 23",
) -> dict:
    return {
        "memory_index": capture_index,
        "room_id": room_id,
        "pano_id": pano_id,
        "capture_index": capture_index,
        "capture_label": f"view_{capture_index}",
        "capture_heading": heading,
        "image_path": f"/tmp/{pano_id}_{capture_index}.png",
    }


class MemoryTreeChildSimilarityTests(unittest.TestCase):
    def test_expected_target_uses_nearest_heading_label(self) -> None:
        items = [
            _item("source", 0, 58.0),
            _item("target", 0, 330.0),
            _item("target", 1, 60.0),
            _item("target", 2, 120.0),
        ]

        expected = expected_target_for_source(
            items=items,
            source_index=0,
            target_pano_id="target",
        )

        self.assertEqual(expected, 2)

    def test_branch_one_ranking_is_similarity_only(self) -> None:
        case = EvaluationCase(
            stage="case",
            case_id="source:0->target",
            source_index=0,
            expected_index=2,
            candidate_indices=(1, 2, 3),
        )
        similarities_by_index = {
            1: 0.9,
            2: 0.7,
            3: 0.8,
        }

        ranked = rank_candidates_by_similarity(
            case.candidate_indices,
            similarities_by_index,
        )

        self.assertEqual([item.candidate_index for item in ranked], [1, 3, 2])
        self.assertEqual(ranked[0].candidate_index, 1)

    def test_summarize_expected_ranks_reports_topk_and_mrr(self) -> None:
        summary = summarize_expected_ranks([1, 2, 4], branching_factor=1)

        self.assertEqual(summary["case_count"], 3)
        self.assertAlmostEqual(summary["top1_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["top3_accuracy"], 2 / 3)
        self.assertAlmostEqual(summary["branch_accuracy"], 1 / 3)
        self.assertAlmostEqual(summary["mrr"], (1.0 + 0.5 + 0.25) / 3)
        self.assertAlmostEqual(summary["mean_expected_rank"], 7 / 3)


if __name__ == "__main__":
    unittest.main()
