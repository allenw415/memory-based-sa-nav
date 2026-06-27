from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from memory_nav.memory.retrieval import MemoryImageRetriever
from memory_nav.cli.run_similarity_passage_selection import (
    _copy_retrieved_images,
    _resolve_output_dir,
)
from memory_nav.navigation import DynamicPassageRetriever, SimilarityPassageSelector


class FakeTextEmbedder:
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def encode_texts(self, _texts):
        return np.asarray([self.embedding], dtype=np.float32)


class FakeImageEmbedder:
    def __init__(self, embeddings_by_pano):
        self.embeddings_by_pano = embeddings_by_pano

    def encode_image_paths(self, image_paths):
        embeddings = []
        for image_path in image_paths:
            path = Path(image_path)
            embeddings.append(self.embeddings_by_pano[path.parent.name])
        return np.asarray(embeddings, dtype=np.float32)


class SimilarityPassageSelectionTests(unittest.TestCase):
    def test_dynamic_passage_retriever_can_keep_unclustered_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = []
            for index in range(3):
                image_path = _write_image(root, f"p{index}", index)
                metadata.append(
                    {
                        "pano_id": f"p{index}",
                        "room_id": "Room 8",
                        "capture_index": index,
                        "capture_label": "view",
                        "capture_heading": float(index * 45),
                        "capture_path": str(image_path),
                    }
                )

            retriever = DynamicPassageRetriever(
                render_root=root,
                retrieval_top_k=3,
                target_clusters=1,
                cluster_candidates=False,
                text_embedder=FakeTextEmbedder([1.0, 0.0]),
                semantic_embeddings=np.asarray(
                    [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]],
                    dtype=np.float32,
                ),
                semantic_metadata_items=metadata,
                visual_embeddings=np.asarray(
                    [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                    dtype=np.float32,
                ),
                visual_metadata_items=metadata,
            )

            passages = retriever.retrieve("Room 8")

        self.assertEqual([item["label"] for item in passages], ["R8P1", "R8P2", "R8P3"])
        self.assertEqual([item["cluster_size"] for item in passages], [1, 1, 1])
        self.assertEqual(
            [item["cluster_member_memory_indices"] for item in passages],
            [[0], [1], [2]],
        )

    def test_similarity_selector_chooses_highest_mean_over_all_sampled_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "current-b", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                    _item(root, "target-b", "Room 23", 1),
                    _item(root, "target-c", "Room 23", 2),
                    _item(root, "target-d", "Room 23", 3),
                    _item(root, "wrong-room", "Room 9", 0),
                ],
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 0.0],
                    [0.1, 0.995],
                    [0.1, 0.995],
                    [0.1, 0.995],
                    [1.0, 0.0],
                ],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=8,
                top_m=1,
                seed=7,
            )

            choice = selector.choose(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {
                        "label": "R8P1",
                        "room_id": "Room 8",
                        "pano_id": "current-a",
                        "capture_index": 0,
                        "image_path": str(root / "current-a" / "current-a_00_view_000deg.png"),
                    },
                    {
                        "label": "R8P2",
                        "room_id": "Room 8",
                        "pano_id": "current-b",
                        "capture_index": 0,
                        "image_path": str(root / "current-b" / "current-b_00_view_000deg.png"),
                    },
                ],
                subgoal_candidates=[],
            )

        self.assertEqual(choice["chosen_label"], "R8P2")
        self.assertEqual(choice["selector_source"], "similarity")
        self.assertEqual(choice["score_field"], "mean_similarity")
        self.assertEqual(choice["passage_ranking"][0]["label"], "R8P2")
        self.assertEqual(choice["request_summary"]["target_sample_count"], 4)
        self.assertEqual({item["room_id"] for item in choice["target_visual_clues"]}, {"Room 23"})
        self.assertGreater(
            choice["passage_ranking"][0]["mean_similarity"],
            choice["passage_ranking"][1]["mean_similarity"],
        )
        self.assertGreater(
            choice["passage_ranking"][1]["top_m_mean_similarity"],
            choice["passage_ranking"][0]["top_m_mean_similarity"],
        )
        self.assertEqual(
            choice["request_summary"]["score_aggregation"],
            "mean_all_sampled_target_similarities",
        )

    def test_similarity_selector_breaks_ties_by_label(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "current-b", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                ],
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=4,
                top_m=1,
                seed=0,
            )

            choice = selector.choose(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {"label": "B", "room_id": "Room 8", "pano_id": "current-b", "capture_index": 0},
                    {"label": "A", "room_id": "Room 8", "pano_id": "current-a", "capture_index": 0},
                ],
                subgoal_candidates=[],
            )

        self.assertEqual(choice["chosen_label"], "A")
        self.assertEqual([item["label"] for item in choice["passage_ranking"]], ["A", "B"])

    def test_similarity_selector_can_use_live_dreamsim_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "current-b", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                ],
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=4,
                top_m=1,
                seed=0,
                similarity_backend="dreamsim",
                image_embedder=FakeImageEmbedder(
                    {
                        "current-a": [1.0, 0.0],
                        "current-b": [0.0, 1.0],
                        "target-a": [0.0, 1.0],
                    }
                ),
            )

            choice = selector.choose(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {
                        "label": "R8P1",
                        "room_id": "Room 8",
                        "pano_id": "current-a",
                        "capture_index": 0,
                        "image_path": str(root / "current-a" / "current-a_00_view_000deg.png"),
                    },
                    {
                        "label": "R8P2",
                        "room_id": "Room 8",
                        "pano_id": "current-b",
                        "capture_index": 0,
                        "image_path": str(root / "current-b" / "current-b_00_view_000deg.png"),
                    },
                ],
                subgoal_candidates=[],
            )

        self.assertEqual(choice["chosen_label"], "R8P2")
        self.assertEqual(choice["similarity_backend"], "dreamsim")
        self.assertEqual(choice["score_field"], "mean_similarity")

    def test_target_sampling_is_room_scoped_and_pano_balanced(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            items = []
            embeddings = []
            for pano_id in ["target-a", "target-b", "target-c"]:
                for capture_index in range(3):
                    items.append(_item(root, pano_id, "Room 23", capture_index))
                    embeddings.append([1.0, 0.0])
            items.append(_item(root, "other", "Room 9", 0))
            embeddings.append([0.0, 1.0])
            retriever = _visual_retriever(root, items, embeddings)
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=3,
                top_m=1,
                seed=3,
            )

            samples = selector._sample_target_room("Room 23")

        self.assertEqual(len(samples), 3)
        self.assertEqual({sample["room_id"] for sample in samples}, {"Room 23"})
        self.assertEqual(len({sample["pano_id"] for sample in samples}), 3)


    def test_contrastive_neighbor_mean_penalizes_non_target_neighbor_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "current-b", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                    _item(root, "target-b", "Room 23", 1),
                    _item(root, "negative-a", "Room 4", 0),
                    _item(root, "not-neighbor", "Room 99", 0),
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=8,
                negative_sample_count=8,
                top_m=1,
                seed=0,
                target_scoring="contrastive_neighbor_mean",
                room_graph={
                    "Room 8": {
                        "neighbors": [
                            {"target_room_id": "Room 23"},
                            {"target_room_id": "Room 4"},
                        ]
                    }
                },
            )

            choice = selector.choose(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {"label": "R8P1", "room_id": "Room 8", "pano_id": "current-a", "capture_index": 0},
                    {"label": "R8P2", "room_id": "Room 8", "pano_id": "current-b", "capture_index": 0},
                ],
                subgoal_candidates=[],
            )

        ranking = choice["passage_ranking"]
        self.assertEqual(choice["chosen_label"], "R8P2")
        self.assertEqual(choice["score_field"], "selection_score")
        self.assertEqual(choice["target_scoring"], "contrastive_neighbor_mean")
        self.assertEqual(choice["request_summary"]["negative_room_ids"], ["Room 4"])
        self.assertEqual({item["room_id"] for item in choice["negative_visual_clues"]}, {"Room 4"})
        self.assertGreater(ranking[0]["selection_score"], ranking[1]["selection_score"])
        self.assertGreater(ranking[1]["negative_mean_similarity"], ranking[0]["negative_mean_similarity"])


    def test_contrastive_neighbor_room_max_mean_uses_per_room_hard_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "current-b", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                    _item(root, "target-b", "Room 23", 1),
                    _item(root, "negative4-a", "Room 4", 0),
                    _item(root, "negative4-b", "Room 4", 1),
                    _item(root, "negative5-a", "Room 5", 0),
                    _item(root, "negative5-b", "Room 5", 1),
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [-1.0, 0.0],
                    [-1.0, 0.0],
                ],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=2,
                negative_sample_count=2,
                top_m=1,
                seed=0,
                target_scoring="contrastive_neighbor_room_max_mean",
                room_graph={
                    "Room 8": {
                        "neighbors": [
                            {"target_room_id": "Room 23"},
                            {"target_room_id": "Room 4"},
                            {"target_room_id": "Room 5"},
                        ]
                    }
                },
            )

            choice = selector.choose(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {"label": "R8P1", "room_id": "Room 8", "pano_id": "current-a", "capture_index": 0},
                    {"label": "R8P2", "room_id": "Room 8", "pano_id": "current-b", "capture_index": 0},
                ],
                subgoal_candidates=[],
            )

        ranking = choice["passage_ranking"]
        penalized = next(item for item in ranking if item["label"] == "R8P1")

        self.assertEqual(choice["chosen_label"], "R8P2")
        self.assertEqual(choice["score_field"], "selection_score")
        self.assertEqual(choice["target_scoring"], "contrastive_neighbor_room_max_mean")
        self.assertEqual(choice["request_summary"]["negative_room_ids"], ["Room 4", "Room 5"])
        self.assertEqual(
            choice["request_summary"]["negative_sample_count_by_room"],
            {"Room 4": 2, "Room 5": 2},
        )
        self.assertEqual(len(choice["negative_visual_clues"]), 4)
        self.assertEqual(penalized["hard_negative_room_id"], "Room 4")
        self.assertAlmostEqual(penalized["hard_negative_similarity"], 1.0)
        self.assertAlmostEqual(
            penalized["selection_score"],
            penalized["target_mean_similarity"] - penalized["hard_negative_similarity"],
        )
        self.assertEqual([item["room_id"] for item in penalized["negative_room_scores"]], ["Room 4", "Room 5"])

    def test_contrastive_neighbor_room_average_mean_averages_per_room_negatives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "current-b", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                    _item(root, "target-b", "Room 23", 1),
                    _item(root, "negative4-a", "Room 4", 0),
                    _item(root, "negative4-b", "Room 4", 1),
                    _item(root, "negative5-a", "Room 5", 0),
                    _item(root, "negative5-b", "Room 5", 1),
                ],
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=2,
                negative_sample_count=2,
                top_m=1,
                seed=0,
                target_scoring="contrastive_neighbor_room_average_mean",
                room_graph={
                    "Room 8": {
                        "neighbors": [
                            {"target_room_id": "Room 23"},
                            {"target_room_id": "Room 4"},
                            {"target_room_id": "Room 5"},
                        ]
                    }
                },
            )

            choice = selector.choose(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {"label": "R8P1", "room_id": "Room 8", "pano_id": "current-a", "capture_index": 0},
                    {"label": "R8P2", "room_id": "Room 8", "pano_id": "current-b", "capture_index": 0},
                ],
                subgoal_candidates=[],
            )

        ranking = choice["passage_ranking"]
        selected = next(item for item in ranking if item["label"] == "R8P1")

        self.assertEqual(choice["chosen_label"], "R8P1")
        self.assertEqual(choice["target_scoring"], "contrastive_neighbor_room_average_mean")
        self.assertEqual(choice["request_summary"]["negative_sample_count_by_room"], {"Room 4": 2, "Room 5": 2})
        self.assertAlmostEqual(selected["target_mean_similarity"], 1.0)
        self.assertAlmostEqual(selected["hard_negative_similarity"], 1.0)
        self.assertAlmostEqual(selected["per_room_average_negative_similarity"], 0.5)
        self.assertAlmostEqual(selected["negative_mean_similarity"], 0.5)
        self.assertAlmostEqual(
            selected["selection_score"],
            selected["target_mean_similarity"] - selected["per_room_average_negative_similarity"],
        )
        self.assertNotAlmostEqual(
            selected["selection_score"],
            selected["target_mean_similarity"] - selected["hard_negative_similarity"],
        )

    def test_contrastive_neighbor_mean_errors_without_non_target_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            retriever = _visual_retriever(
                root,
                [
                    _item(root, "current-a", "Room 8", 0),
                    _item(root, "target-a", "Room 23", 0),
                ],
                [[1.0, 0.0], [1.0, 0.0]],
            )
            selector = SimilarityPassageSelector(
                visual_retriever=retriever,
                target_sample_count=4,
                top_m=1,
                target_scoring="contrastive_neighbor_mean",
                room_graph={"Room 8": {"neighbors": [{"target_room_id": "Room 23"}]}},
            )

            with self.assertRaisesRegex(ValueError, "No non-target neighbor rooms"):
                selector.choose(
                    current_room_id="Room 8",
                    subgoal_room_id="Room 23",
                    current_candidates=[
                        {"label": "R8P1", "room_id": "Room 8", "pano_id": "current-a", "capture_index": 0}
                    ],
                    subgoal_candidates=[],
                )

    def test_cli_image_export_uses_structured_experiment_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_image = _write_image(root, "current-a", 0)
            target_image = _write_image(root, "target-a", 0)
            output_path = root / "experiment"
            payload = {
                "current_room_passages": [
                    {
                        "label": "R8P1",
                        "room_id": "Room 8",
                        "pano_id": "current-a",
                        "capture_index": 0,
                        "image_path": str(current_image),
                    }
                ],
                "passage_choice": {
                    "target_visual_clues": [
                        {
                            "memory_index": 2,
                            "room_id": "Room 23",
                            "pano_id": "target-a",
                            "capture_index": 0,
                            "image_path": str(target_image),
                        }
                    ],
                    "passage_ranking": [
                        {
                            "label": "R8P1",
                            "matched_target_samples": [
                                {
                                    "memory_index": 2,
                                    "pano_id": "target-a",
                                    "capture_index": 0,
                                }
                            ],
                        }
                    ],
                },
            }

            exports = _copy_retrieved_images(payload, output_path)
            copied_paths = [Path(item["copied_image_path"]) for item in exports]

            self.assertEqual(_resolve_output_dir(output_path), output_path.resolve())
            self.assertEqual(len(exports), 3)
            self.assertTrue(all(path.exists() for path in copied_paths))
            self.assertEqual(
                {path.parent.name for path in copied_paths},
                {"current_room_passages", "target_room_visual_clues", "top_k_passages"},
            )
            self.assertIn("copied_image_path", payload["current_room_passages"][0])
            self.assertIn("copied_image_path", payload["passage_choice"]["target_visual_clues"][0])
            self.assertIn("current_room_passage_image_path", payload["passage_choice"]["passage_ranking"][0])
            self.assertIn("top_k_passage_image_path", payload["passage_choice"]["passage_ranking"][0])
            self.assertIn(
                "copied_image_path",
                payload["passage_choice"]["passage_ranking"][0]["matched_target_samples"][0],
            )


    def test_cli_image_export_copies_negative_visual_clues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_image = _write_image(root, "current-a", 0)
            target_image = _write_image(root, "target-a", 0)
            negative_image = _write_image(root, "negative-a", 0)
            output_path = root / "experiment"
            payload = {
                "current_room_passages": [
                    {
                        "label": "R8P1",
                        "room_id": "Room 8",
                        "pano_id": "current-a",
                        "capture_index": 0,
                        "image_path": str(current_image),
                    }
                ],
                "passage_choice": {
                    "target_visual_clues": [
                        {
                            "memory_index": 2,
                            "room_id": "Room 23",
                            "pano_id": "target-a",
                            "capture_index": 0,
                            "image_path": str(target_image),
                        }
                    ],
                    "negative_visual_clues": [
                        {
                            "memory_index": 3,
                            "room_id": "Room 4",
                            "pano_id": "negative-a",
                            "capture_index": 0,
                            "image_path": str(negative_image),
                        }
                    ],
                    "passage_ranking": [
                        {
                            "label": "R8P1",
                            "matched_target_samples": [
                                {"memory_index": 2, "pano_id": "target-a", "capture_index": 0}
                            ],
                            "matched_negative_samples": [
                                {"memory_index": 3, "pano_id": "negative-a", "capture_index": 0}
                            ],
                        }
                    ],
                },
            }

            exports = _copy_retrieved_images(payload, output_path)
            copied_paths = [Path(item["copied_image_path"]) for item in exports]

            self.assertEqual(len(exports), 4)
            self.assertIn(output_path / "negative_room_visual_clues", {path.parent for path in copied_paths})
            self.assertIn(
                "copied_image_path",
                payload["passage_choice"]["passage_ranking"][0]["matched_negative_samples"][0],
            )

    def test_cli_image_export_overwrites_in_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            current_image = _write_image(root, "current-a", 0)
            output_path = root / "experiment"
            payload = {
                "current_room_passages": [
                    {
                        "label": "R8P1",
                        "room_id": "Room 8",
                        "pano_id": "current-a",
                        "capture_index": 0,
                        "image_path": str(current_image),
                    }
                ],
                "passage_choice": {"target_visual_clues": [], "passage_ranking": []},
            }

            exports = _copy_retrieved_images(payload, output_path)
            repeated_exports = _copy_retrieved_images(payload, output_path)

            self.assertEqual(len(exports), 1)
            self.assertEqual(len(repeated_exports), 1)
            self.assertEqual(repeated_exports[0]["copied_image_path"], exports[0]["copied_image_path"])
            self.assertEqual(
                Path(exports[0]["copied_image_path"]).parent,
                output_path / "current_room_passages",
            )

    def test_output_path_rejects_explicit_json_file(self) -> None:
        with self.assertRaisesRegex(ValueError, "experiment directory"):
            _resolve_output_dir("outputs/example/result.json")


def _write_image(root: Path, pano_id: str, capture_index: int) -> Path:
    image_dir = root / pano_id
    image_dir.mkdir(exist_ok=True)
    image_path = image_dir / f"{pano_id}_{capture_index:02d}_view_000deg.png"
    image_path.write_bytes(b"image")
    return image_path


def _item(root: Path, pano_id: str, room_id: str, capture_index: int) -> dict:
    image_path = _write_image(root, pano_id, capture_index)
    return {
        "pano_id": pano_id,
        "room_id": room_id,
        "capture_index": capture_index,
        "capture_label": "view",
        "capture_heading": float(capture_index * 45),
        "capture_path": str(image_path),
    }


def _visual_retriever(root: Path, items: list[dict], embeddings: list[list[float]]) -> MemoryImageRetriever:
    return MemoryImageRetriever(
        metadata_items=items,
        image_embeddings=np.asarray(embeddings, dtype=np.float32),
        use_faiss=False,
        project_root=root,
        render_root=root,
    )


if __name__ == "__main__":
    unittest.main()
