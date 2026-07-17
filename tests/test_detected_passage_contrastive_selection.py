from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from memory_nav.cli.run_detected_passage_contrastive_selection import (
    DetectionCandidate,
    build_parser,
    _filter_detections,
    _passage_candidate_limit,
    _passage_queries,
    _retrieve_passages_with_queries,
    run_detected_passage_contrastive_selection,
)
from memory_nav.memory.retrieval import MemoryImageRetriever
from memory_nav.navigation import DynamicPassageRetriever, strict_detected_passage_configuration


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


class FakeQueryPassageRetriever:
    def __init__(self):
        self.query = "original query"
        self.calls: list[tuple[str, str]] = []

    def retrieve_with_query_embedding(self, room_id, _embedding):
        self.calls.append((self.query, room_id))
        results_by_query = {
            "doorway": [
                _retrieved_passage("old-label-a", "pano-a", 0, 0.91),
            ],
            "corridor between exhibits": [
                _retrieved_passage("old-label-b", "pano-b", 1, 0.86),
                _retrieved_passage("old-label-a2", "pano-a", 0, 0.72),
            ],
        }
        return [dict(item) for item in results_by_query.get(self.query, [])]


class StrictDetectedPassageDefaultsTests(unittest.TestCase):
    def test_defaults_match_latest_strict_salad_experiment(self) -> None:
        config = strict_detected_passage_configuration(seed=7)

        self.assertEqual(config["seed"], 7)
        self.assertEqual(config["passage_query_fusion"], "single_text")
        self.assertEqual(config["passage_candidate_limit"], 64)
        self.assertFalse(config["passage_clustering"])
        self.assertEqual(config["detector_model"], "IDEA-Research/grounding-dino-base")
        self.assertEqual(config["similarity_backend"], "salad")
        self.assertEqual(config["target_scoring"], "contrastive_neighbor_mean")
        self.assertEqual(config["contrastive_negative_weight"], 0.5)
        self.assertEqual(config["max_detections_per_image"], 5)


class DetectedPassageContrastiveSelectionTests(unittest.TestCase):
    def test_size_filters_are_disabled_by_default(self) -> None:
        args = build_parser().parse_args(
            [
                "--current-room-id",
                "Room 8",
                "--subgoal-room-id",
                "Room 9",
                "--output-path",
                "out",
            ]
        )

        self.assertEqual(args.min_box_area_ratio, 0.0)
        self.assertEqual(args.min_box_width_ratio, 0.0)
        self.assertEqual(args.min_box_height_ratio, 0.0)
        self.assertEqual(args.max_box_area_ratio, 1.0)
        self.assertEqual(args.min_crop_area_ratio, 0.0)
        self.assertEqual(args.max_crop_area_ratio, 1.0)

    def test_expanded_text_queries_merge_unique_candidates(self) -> None:
        queries = _passage_queries(
            "doorway",
            mode="expanded",
            extra_queries=["corridor between exhibits", "doorway"],
        )
        self.assertIn("corridor between exhibits", queries)
        self.assertTrue(any("columns" in query for query in queries))
        self.assertEqual(queries.count("doorway"), 1)
        self.assertEqual(_passage_candidate_limit(None, passage_top_k=20, query_count=len(queries)), 40)

        retriever = FakeQueryPassageRetriever()
        merged = _retrieve_passages_with_queries(
            retriever,
            room_id="Room 23",
            queries=["doorway", "corridor between exhibits"],
            query_embeddings=np.asarray([[1.0], [2.0]], dtype=np.float32),
            candidate_limit=3,
        )

        self.assertEqual(retriever.query, "original query")
        self.assertEqual(retriever.calls, [("doorway", "Room 23"), ("corridor between exhibits", "Room 23")])
        self.assertEqual([item["label"] for item in merged], ["R23P1", "R23P2"])
        self.assertEqual({item["pano_id"] for item in merged}, {"pano-a", "pano-b"})
        duplicate = next(item for item in merged if item["pano_id"] == "pano-a")
        corridor_only = next(item for item in merged if item["pano_id"] == "pano-b")
        self.assertEqual(len(duplicate["retrieval_query_sources"]), 2)
        self.assertEqual(corridor_only["best_retrieval_query"], "corridor between exhibits")
        self.assertEqual(corridor_only["retrieval_source_label"], "old-label-b")

    def test_max_score_fusion_scores_room_images_without_per_query_top_k_cutoff(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = []
            semantic_vectors = [
                [0.92, 0.08],
                [0.05, 0.99],
                [0.65, 0.65],
            ]
            for index, pano_id in enumerate(["doorway", "columns", "generic"]):
                image_path = root / pano_id / f"{pano_id}_{index:02d}_view_000deg.png"
                _write_image(image_path, color=(40 + index * 30, 90, 120))
                metadata.append(
                    {
                        "pano_id": pano_id,
                        "room_id": "Room 23",
                        "capture_index": index,
                        "capture_label": "view",
                        "capture_heading": float(index * 45),
                        "capture_path": str(image_path),
                    }
                )

            retriever = DynamicPassageRetriever(
                render_root=root,
                retrieval_top_k=1,
                target_clusters=1,
                cluster_candidates=False,
                semantic_embeddings=np.asarray(semantic_vectors, dtype=np.float32),
                semantic_metadata_items=metadata,
                visual_embeddings=np.asarray(semantic_vectors, dtype=np.float32),
                visual_metadata_items=metadata,
            )

            merged = _retrieve_passages_with_queries(
                retriever,
                room_id="Room 23",
                queries=["doorway", "opening between columns"],
                query_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                candidate_limit=2,
                fusion_mode="max_score",
            )

        self.assertEqual(len(merged), 2)
        self.assertEqual([item["label"] for item in merged], ["R23P1", "R23P2"])
        self.assertEqual(merged[0]["pano_id"], "columns")
        self.assertEqual(merged[0]["best_retrieval_query"], "opening between columns")
        self.assertEqual(merged[0]["best_query_index"], 2)
        self.assertEqual(merged[0]["retrieval_query_fusion"], "max_score")
        self.assertEqual(len(merged[0]["semantic_scores_by_query"]), 2)
        self.assertEqual(retriever.retrieval_top_k, 1)

    def test_geometry_filters_reject_column_like_detections(self) -> None:
        kept = _filter_detections(
            [
                DetectionCandidate((84.0, 0.0, 96.0, 95.0), 0.95, "opening between columns"),
                DetectionCandidate((25.0, 20.0, 75.0, 95.0), 0.88, "open doorway"),
                DetectionCandidate((0.0, 58.0, 95.0, 70.0), 0.86, "walkable opening"),
            ],
            image_size=(100, 100),
            min_area_ratio=0.0,
            min_width_ratio=0.0,
            min_height_ratio=0.20,
            min_aspect_ratio=0.30,
            max_aspect_ratio=3.0,
            min_bottom_ratio=0.40,
            max_center_x_distance_ratio=1.0,
            max_area_ratio=0.95,
        )

        self.assertEqual([item.label for item in kept], ["open doorway"])

    def test_crop_area_filter_rejects_near_full_image_crops(self) -> None:
        kept = _filter_detections(
            [
                DetectionCandidate((2.0, 2.0, 95.0, 95.0), 0.95, "gallery opening"),
                DetectionCandidate((25.0, 25.0, 75.0, 90.0), 0.88, "open doorway"),
            ],
            image_size=(100, 100),
            min_area_ratio=0.0,
            min_width_ratio=0.0,
            min_height_ratio=0.0,
            min_aspect_ratio=0.0,
            max_aspect_ratio=999.0,
            min_bottom_ratio=0.0,
            max_center_x_distance_ratio=1.0,
            max_area_ratio=0.95,
            crop_padding_ratio=0.06,
            max_crop_area_ratio=0.90,
        )

        self.assertEqual([item.label for item in kept], ["open doorway"])

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

    def test_leaf_room_without_non_target_neighbor_scores_with_target_only(self) -> None:
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
                room_graph={"Room 23": {"neighbors": [{"target_room_id": "Room 22"}]}},
                detector=detector,
                image_embedder=FakeImageEmbedder(),
                output_dir=output_dir,
                configuration=_configuration(),
            )

            ranking = payload["passage_choice"]["passage_ranking"]

            self.assertTrue(payload["success"])
            self.assertEqual(payload["configuration"]["negative_room_ids"], [])
            self.assertEqual(payload["configuration"]["negative_sampling_policy"], "target_only_leaf_room")
            self.assertEqual(payload["passage_choice"]["negative_visual_clues"], [])
            self.assertEqual(payload["passage_choice"]["negative_room_ids"], [])
            self.assertEqual(payload["passage_choice"]["negative_sampling_policy"], "target_only_leaf_room")
            self.assertEqual(payload["passage_choice"]["chosen_label"], "R23P1D2")
            self.assertEqual(ranking[0]["negative_mean_similarity"], 0.0)
            self.assertEqual(ranking[0]["hard_negative_similarity"], 0.0)
            self.assertIsNone(ranking[0]["hard_negative_room_id"])
            self.assertEqual(ranking[0]["matched_negative_samples"], [])
            self.assertEqual(ranking[0]["selection_score"], ranking[0]["target_mean_similarity"])

    def test_filtered_small_detections_do_not_fall_back_to_full_image(self) -> None:
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
                    _metadata_item(root, "negative-a", "Room 17", 0),
                ],
            )
            output_dir = root / "detected_contrastive"
            detector = FakeDetector(
                {
                    "R23P1": [DetectionCandidate((2.0, 2.0, 5.0, 5.0), 0.95, "doorway")],
                    "R23P2": [DetectionCandidate((10.0, 10.0, 55.0, 60.0), 0.75, "gallery entrance")],
                }
            )

            configuration = _configuration()
            configuration["min_box_area_ratio"] = 0.01
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
                configuration=configuration,
            )

            records_by_label = {item["label"]: item for item in payload["current_room_passages"]}
            self.assertEqual(records_by_label["R23P1"]["detection_status"], "filtered_out")
            self.assertLess(
                records_by_label["R23P1"]["raw_detections"][0]["area_ratio"],
                payload["configuration"]["min_box_area_ratio"],
            )
            self.assertEqual(records_by_label["R23P2"]["detection_status"], "detected")
            self.assertEqual([item["source_label"] for item in payload["detected_passage_candidates"]], ["R23P2"])
            self.assertFalse(any(item["label"].startswith("R23P1D0") for item in payload["detected_passage_candidates"]))

    def test_no_detection_is_excluded_from_candidates_by_default(self) -> None:
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
                    _metadata_item(root, "negative-a", "Room 17", 0),
                ],
            )
            output_dir = root / "detected_contrastive"
            detector = FakeDetector(
                {
                    "R23P2": [DetectionCandidate((10.0, 10.0, 55.0, 60.0), 0.75, "gallery entrance")],
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

            records_by_label = {item["label"]: item for item in payload["current_room_passages"]}

            self.assertEqual(records_by_label["R23P1"]["detection_status"], "no_detection")
            self.assertEqual(records_by_label["R23P2"]["detection_status"], "detected")
            self.assertEqual([item["source_label"] for item in payload["detected_passage_candidates"]], ["R23P2"])
            self.assertFalse(any(item["detection_status"] == "no_detection" for item in payload["detected_passage_candidates"]))
            self.assertFalse(any(item["label"].endswith("D0") for item in payload["detected_passage_candidates"]))

    def test_enable_full_image_fallback_keeps_legacy_full_image_candidate(self) -> None:
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
            configuration = _configuration()
            configuration["fallback_to_full_image_on_no_detection"] = True

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
                configuration=configuration,
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
        "passage_queries": ["passage"],
        "passage_query_mode": "single",
        "passage_query_fusion": "rrf",
        "combined_passage_query": "passage",
        "extra_passage_queries": [],
        "passage_top_k": 2,
        "passage_top_k_per_query": 2,
        "passage_candidate_limit": 2,
        "passage_clustering": False,
        "detector_prompt": "doorway . gallery entrance .",
        "detector_backend": "fake",
        "detector_model": "fake",
        "box_threshold": 0.25,
        "text_threshold": 0.25,
        "min_box_area_ratio": 0.0,
        "min_box_width_ratio": 0.0,
        "min_box_height_ratio": 0.0,
        "min_box_aspect_ratio": 0.0,
        "max_box_aspect_ratio": 999.0,
        "min_box_bottom_ratio": 0.0,
        "max_box_center_x_distance_ratio": 1.0,
        "max_box_area_ratio": 1.0,
        "min_crop_area_ratio": 0.0,
        "max_crop_area_ratio": 1.0,
        "fallback_to_full_image_on_no_detection": False,
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


def _retrieved_passage(label: str, pano_id: str, capture_index: int, score: float) -> dict:
    return {
        "label": label,
        "room_id": "Room 23",
        "pano_id": pano_id,
        "capture_index": capture_index,
        "image_path": f"/tmp/{pano_id}_{capture_index}.png",
        "semantic_score": score,
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
