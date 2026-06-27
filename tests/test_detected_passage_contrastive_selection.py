from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from memory_nav.cli.run_detected_passage_contrastive_selection import (
    DetectionCandidate,
    run_detected_passage_contrastive_selection,
)
from memory_nav.memory.retrieval import MemoryImageRetriever


class FakeDetector:
    def __init__(self, detections_by_label: dict[str, list[DetectionCandidate]]):
        self.detections_by_label = detections_by_label
        self.prompts: list[str] = []

    def detect(self, image_path, prompt):
        self.prompts.append(prompt)
        name = Path(image_path).name
        for label, detections in self.detections_by_label.items():
            if label in name:
                return list(detections)
        return []


class FakeImageEmbedder:
    def encode_image_paths(self, image_paths):
        embeddings = []
        for image_path in image_paths:
            name = Path(image_path).name
            if "R23P1D2" in name or "target" in name:
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return np.asarray(embeddings, dtype=np.float32)


class DetectedPassageContrastiveSelectionTests(unittest.TestCase):
    def test_detected_openings_are_scored_as_independent_contrastive_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_passages = [
                _current_passage(root, "R23P1", 1),
                _current_passage(root, "R23P2", 2),
            ]
            visual_retriever = _visual_retriever(
                root,
                [
                    _metadata_item(root, "target-a", "Room 22", 0),
                    _metadata_item(root, "target-b", "Room 22", 1),
                    _metadata_item(root, "negative-a", "Room 17", 0),
                    _metadata_item(root, "negative-b", "Room 17", 1),
                ],
            )
            output_dir = root / "detected_contrastive"
            detector = FakeDetector(
                {
                    "R23P1": [
                        DetectionCandidate((5.0, 5.0, 35.0, 45.0), 0.92, "doorway"),
                        DetectionCandidate((50.0, 10.0, 90.0, 60.0), 0.88, "gallery entrance"),
                    ],
                    "R23P2": [
                        DetectionCandidate((10.0, 10.0, 40.0, 50.0), 0.8, "opening"),
                    ],
                }
            )

            payload = run_detected_passage_contrastive_selection(
                current_room_id="Room 23",
                subgoal_room_id="Room 22",
                current_passages=current_passages,
                visual_retriever=visual_retriever,
                room_graph={
                    "Room 23": {
                        "neighbors": [
                            {"target_room_id": "Room 22"},
                            {"target_room_id": "Room 17"},
                        ]
                    }
                },
                detector=detector,
                image_embedder=FakeImageEmbedder(),
                output_dir=output_dir,
                configuration=_configuration(),
            )

            ranking = payload["passage_choice"]["passage_ranking"]

            self.assertEqual(payload["passage_choice"]["chosen_label"], "R23P1D2")
            self.assertEqual(payload["passage_choice"]["chosen_source_label"], "R23P1")
            self.assertEqual(len(payload["detected_passage_candidates"]), 3)
            self.assertEqual(
                {item["source_label"] for item in payload["detected_passage_candidates"]},
                {"R23P1", "R23P2"},
            )
            self.assertGreater(ranking[0]["selection_score"], ranking[1]["selection_score"])
            self.assertEqual(ranking[0]["target_mean_similarity"], 1.0)
            self.assertEqual(ranking[0]["negative_mean_similarity"], 0.0)
            self.assertEqual(ranking[0]["source_passage"]["label"], "R23P1")
            self.assertTrue((output_dir / "detected_passage_overlays").exists())
            self.assertTrue((output_dir / "top_k_detected_passage_candidates").exists())
            self.assertTrue((output_dir / "top_k_source_passages").exists())
            self.assertTrue(Path(ranking[0]["top_k_detected_candidate_image_path"]).exists())
            self.assertTrue(Path(ranking[0]["top_k_source_passage_image_path"]).exists())
            self.assertEqual(detector.prompts, ["doorway . gallery entrance .", "doorway . gallery entrance ."])

    def test_no_detection_falls_back_to_full_image_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_passages = [_current_passage(root, "R23P1", 1)]
            visual_retriever = _visual_retriever(
                root,
                [
                    _metadata_item(root, "target-a", "Room 22", 0),
                    _metadata_item(root, "negative-a", "Room 17", 0),
                ],
            )
            output_dir = root / "detected_contrastive"

            payload = run_detected_passage_contrastive_selection(
                current_room_id="Room 23",
                subgoal_room_id="Room 22",
                current_passages=current_passages,
                visual_retriever=visual_retriever,
                room_graph={
                    "Room 23": {
                        "neighbors": [
                            {"target_room_id": "Room 22"},
                            {"target_room_id": "Room 17"},
                        ]
                    }
                },
                detector=FakeDetector({}),
                image_embedder=FakeImageEmbedder(),
                output_dir=output_dir,
                configuration=_configuration(),
            )

            candidate = payload["detected_passage_candidates"][0]

            self.assertEqual(payload["passage_choice"]["chosen_label"], "R23P1D0")
            self.assertEqual(candidate["source_passage"]["label"], "R23P1")
            self.assertEqual(candidate["detection_status"], "no_detection")
            self.assertEqual(candidate["crop_box_xyxy"], [0, 0, 100, 80])
            self.assertTrue(Path(candidate["crop_image_path"]).exists())
            self.assertTrue(Path(payload["current_room_passages"][0]["overlay_image_path"]).exists())


def _configuration() -> dict:
    return {
        "seed": 0,
        "passage_query": "passage",
        "passage_top_k": 2,
        "passage_clustering": False,
        "detector_prompt": "doorway . gallery entrance .",
        "detector_backend": "fake",
        "detector_model": "fake",
        "box_threshold": 0.25,
        "text_threshold": 0.25,
        "min_box_area_ratio": 0.01,
        "max_box_area_ratio": 0.95,
        "crop_padding_ratio": 0.0,
        "current_image_mode": "mask",
        "mask_background_brightness": 0.0,
        "max_detections_per_image": 3,
        "target_sample_count": 2,
        "negative_sample_count": 2,
        "similarity_top_m": 1,
        "target_scoring": "contrastive_neighbor_mean",
        "contrastive_negative_weight": 1.0,
        "similarity_backend": "fake",
        "dreamsim_type": None,
        "semantic_embedding_model": "fake",
        "visual_similarity_model": "fake",
    }


def _current_passage(root: Path, label: str, rank: int) -> dict:
    image_path = root / "current" / f"{label}.png"
    _write_image(image_path, color=(rank * 20, 80, 120))
    return {
        "label": label,
        "room_id": "Room 23",
        "pano_id": f"current-{rank}",
        "capture_index": rank - 1,
        "image_path": str(image_path),
        "semantic_score": 1.0 / rank,
    }


def _metadata_item(root: Path, pano_id: str, room_id: str, capture_index: int) -> dict:
    image_path = root / "metadata" / f"{pano_id}_{capture_index}.png"
    color = (20, 140, 40) if room_id == "Room 22" else (140, 20, 40)
    _write_image(image_path, color=color)
    return {
        "pano_id": pano_id,
        "room_id": room_id,
        "capture_index": capture_index,
        "capture_label": "view",
        "capture_path": str(image_path),
    }


def _visual_retriever(root: Path, items: list[dict]) -> MemoryImageRetriever:
    return MemoryImageRetriever(
        metadata_items=items,
        use_faiss=False,
        project_root=root,
        render_root=root,
    )


def _write_image(path: Path, *, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), color=color).save(path)


if __name__ == "__main__":
    unittest.main()
