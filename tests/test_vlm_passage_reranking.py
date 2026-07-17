from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.experiments.rerank_detected_passages_with_vlm import (
    assign_public_labels,
    build_vlm_request,
    collapse_same_source_candidates,
    deduplicate_candidates_by_similarity,
    index_result_paths,
    validate_vlm_choice,
)


class VlmPassageRerankingTests(unittest.TestCase):
    def test_same_source_detections_collapse_to_highest_ranked_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.png"
            comparison_a = root / "a.png"
            comparison_b = root / "b.png"
            for path in (source, comparison_a, comparison_b):
                path.write_bytes(b"image")
            ranking = [
                {
                    "rank": 1,
                    "label": "R4P25D2",
                    "source_image_path": str(source),
                    "comparison_image_path": str(comparison_a),
                },
                {
                    "rank": 2,
                    "label": "R4P25D1",
                    "source_image_path": str(source),
                    "comparison_image_path": str(comparison_b),
                },
            ]

            representatives, decisions, skipped = collapse_same_source_candidates(
                ranking,
                pool_size=8,
            )

        self.assertEqual(len(representatives), 1)
        self.assertEqual(representatives[0]["label"], "R4P25D2")
        self.assertEqual(
            representatives[0]["same_source_duplicate_labels"],
            ["R4P25D1"],
        )
        self.assertEqual(decisions[1]["action"], "merged_same_source")
        self.assertEqual(skipped, [])

    def test_similarity_dedup_keeps_later_candidate_to_fill_top_k(self) -> None:
        candidates = [
            {"label": "A", "source_image_path": "/tmp/a.png"},
            {"label": "B", "source_image_path": "/tmp/b.png"},
            {"label": "C", "source_image_path": "/tmp/c.png"},
        ]
        embeddings = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        selected, decisions = deduplicate_candidates_by_similarity(
            candidates,
            embeddings,
            threshold=0.95,
            top_k=2,
        )

        self.assertEqual([item["label"] for item in selected], ["A", "C"])
        self.assertEqual(decisions[1]["action"], "merged_visual_duplicate")
        self.assertEqual(decisions[1]["representative_label"], "A")

    def test_vlm_request_uses_original_images_without_scores_or_crop_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current = []
            target = []
            for prefix, collection in (("current", current), ("target", target)):
                for index in range(2):
                    path = root / f"{prefix}_{index}.png"
                    path.write_bytes(b"image")
                    collection.append(
                        {
                            "public_label": f"{'C' if prefix == 'current' else 'T'}{index + 1}",
                            "source_image_path": str(path),
                            "comparison_image_path": f"/secret/{prefix}_masked.png",
                            "selection_score": 0.99,
                            "capture_heading": 90.0,
                        }
                    )

            request = build_vlm_request(
                model="gemma-4-31b-it",
                detail="high",
                reasoning_effort="low",
                current_room_id="Room 4",
                target_room_id="Room 8",
                current_candidates=current,
                target_references=target,
            )

        serialized = json.dumps(request)
        self.assertEqual(
            sum(
                1
                for block in request["input"][0]["content"]
                if block.get("type") == "input_image"
            ),
            4,
        )
        self.assertNotIn("selection_score", serialized)
        self.assertNotIn("capture_heading", serialized)
        self.assertNotIn("masked.png", serialized)

    def test_validate_choice_requires_every_current_candidate_assessment(self) -> None:
        parsed = {
            "candidate_assessments": [
                {
                    "label": "C1",
                    "status": "valid_passage",
                    "passage_confidence": 0.8,
                    "reason": "doorway",
                }
            ],
            "chosen_label": "C1",
            "target_reference_labels_used": ["T1"],
        }

        with self.assertRaisesRegex(ValueError, "missing=.*C2"):
            validate_vlm_choice(
                parsed,
                current_labels=["C1", "C2"],
                target_labels=["T1"],
            )

    def test_result_index_uses_directed_room_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_dir = root / "room4_to_room8"
            result_dir.mkdir()
            result_path = result_dir / "result.json"
            result_path.write_text(
                json.dumps(
                    {
                        "configuration": {
                            "current_room_id": "Room 4",
                            "subgoal_room_id": "Room 8",
                        }
                    }
                ),
                encoding="utf-8",
            )

            indexed = index_result_paths(root)

        self.assertEqual(indexed[("Room 4", "Room 8")], result_path.resolve())

    def test_public_labels_hide_original_ranking_labels(self) -> None:
        labeled = assign_public_labels(
            [{"label": "R4P25D2"}, {"label": "R4P18D2"}],
            prefix="C",
        )
        self.assertEqual([item["public_label"] for item in labeled], ["C1", "C2"])


if __name__ == "__main__":
    unittest.main()
