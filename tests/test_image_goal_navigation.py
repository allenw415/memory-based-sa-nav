from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from memory_nav.navigation import (
    IndexedPanoramaViewStore,
    PanoramaGraphImageGoalSimulator,
    PureVisualDirectionPolicy,
    VisualView,
    angular_distance_deg,
    resolve_goal_label,
)
from memory_nav.navigation.image_goal import ImagePathSimilarityDirectionPolicy


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def view(
    index: int,
    heading: float,
    embedding,
    auxiliary_embedding=None,
    path: str | Path | None = None,
) -> VisualView:
    return VisualView(
        capture_index=index,
        label=f"view_{index}",
        heading=heading,
        embedding=np.asarray(embedding, dtype=np.float32),
        auxiliary_embedding=(
            np.asarray(auxiliary_embedding, dtype=np.float32)
            if auxiliary_embedding is not None
            else None
        ),
        path=str(path) if path is not None else None,
    )


class FakePathEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode_image_paths(self, image_paths):
        self.calls.append([Path(path).name for path in image_paths])
        return np.asarray(
            [self.vectors[Path(path).name] for path in image_paths],
            dtype=np.float32,
        )


class ImagePathSimilarityDirectionPolicyTests(unittest.TestCase):
    def test_encodes_goal_and_current_view_paths_with_live_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = root / "goal.png"
            left = root / "left.png"
            right = root / "right.png"
            for path in [goal, left, right]:
                path.write_bytes(b"image")

            embedder = FakePathEmbedder(
                {
                    "goal.png": [0.0, 1.0],
                    "left.png": [1.0, 0.0],
                    "right.png": [0.0, 1.0],
                }
            )
            decision = ImagePathSimilarityDirectionPolicy(
                image_embedder=embedder,
                similarity_backend="dreamsim:ensemble",
            ).choose_action(
                goal_embedding=goal,
                views=[
                    view(0, 0.0, [0.0, 0.0], path=left),
                    view(1, 90.0, [0.0, 0.0], path=right),
                ],
                legal_action_headings=[10.0, 100.0],
            )

        self.assertEqual(decision.selected_capture_index, 1)
        self.assertEqual(decision.selected_action_index, 1)
        self.assertEqual(decision.scoring["similarity_backend"], "dreamsim:ensemble")
        self.assertEqual(
            decision.view_scores[1]["similarity_backend"], "dreamsim:ensemble"
        )
        self.assertEqual(embedder.calls, [["goal.png", "left.png", "right.png"]])

class PureVisualDirectionPolicyTests(unittest.TestCase):
    def test_selects_highest_similarity_and_breaks_ties_by_capture_index(self) -> None:
        decision = PureVisualDirectionPolicy().choose_action(
            goal_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            views=[
                view(2, 90.0, [1.0, 0.0]),
                view(1, 45.0, [1.0, 0.0]),
                view(0, 0.0, [0.0, 1.0]),
            ],
            legal_action_headings=[50.0, 180.0],
        )

        self.assertEqual(decision.selected_capture_index, 1)
        self.assertEqual(decision.selected_action_index, 0)
        self.assertAlmostEqual(decision.similarity, 1.0)
        self.assertAlmostEqual(decision.margin, 0.0)

    def test_circular_heading_distance_wraps_at_north(self) -> None:
        self.assertAlmostEqual(angular_distance_deg(359.0, 1.0), 2.0)
        self.assertAlmostEqual(angular_distance_deg(1.0, 359.0), 2.0)

    def test_maps_selected_view_to_nearest_legal_heading(self) -> None:
        decision = PureVisualDirectionPolicy().choose_action(
            goal_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            views=[view(0, 359.0, [1.0, 0.0]), view(1, 180.0, [0.0, 1.0])],
            legal_action_headings=[180.0, 1.0],
        )

        self.assertEqual(decision.selected_action_index, 1)
        self.assertAlmostEqual(decision.selected_action_heading, 1.0)

    def test_fuses_salad_and_siglip_scores(self) -> None:
        decision = PureVisualDirectionPolicy(salad_alpha=0.5).choose_action(
            goal_embedding={
                "salad": np.asarray([1.0, 0.0], dtype=np.float32),
                "siglip": np.asarray([1.0, 0.0], dtype=np.float32),
            },
            views=[
                view(0, 150.0, [1.0, 0.0], [0.0, 1.0]),
                view(1, 240.0, [0.9, 0.1], [1.0, 0.0]),
            ],
            legal_action_headings=[145.0, 256.0],
        )

        self.assertEqual(decision.selected_capture_index, 1)
        self.assertEqual(decision.selected_action_index, 1)
        self.assertEqual(decision.scoring["mode"], "salad_siglip_fusion")
        self.assertIsNotNone(decision.view_scores[0]["siglip_similarity"])


class PanoramaGraphImageGoalSimulatorTests(unittest.TestCase):
    def test_avoids_immediate_backtrack_when_an_alternative_exists(self) -> None:
        graph = {
            "A": {
                "neighbors": [
                    {"target_pano_id": "B", "geocentric_heading_deg": 10.0},
                    {"target_pano_id": "C", "geocentric_heading_deg": 100.0},
                ]
            },
            "B": {"neighbors": [{"target_pano_id": "A", "geocentric_heading_deg": 190.0}]},
            "C": {"neighbors": []},
        }
        simulator = PanoramaGraphImageGoalSimulator(
            pano_graph=graph,
            pano_room_mappings={"A": "Room 8", "B": "Room 8", "C": "Room 8"},
            observation_provider=lambda _pano_id: [view(0, 10.0, [1.0, 0.0])],
        )

        result = simulator.run(
            start_pano_id="B",
            goal_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            target_room_id="Room 23",
            max_steps=2,
        )

        self.assertEqual(result.pano_path, ["B", "A", "C"])
        self.assertTrue(result.trajectory[1]["anti_backtrack_applied"])

    def test_allows_backtrack_when_it_is_the_only_edge(self) -> None:
        graph = {
            "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 10.0}]},
            "B": {"neighbors": [{"target_pano_id": "A", "geocentric_heading_deg": 190.0}]},
        }
        simulator = PanoramaGraphImageGoalSimulator(
            pano_graph=graph,
            pano_room_mappings={"A": "Room 8", "B": "Room 8"},
            observation_provider=lambda _pano_id: [view(0, 10.0, [1.0, 0.0])],
        )

        result = simulator.run(
            start_pano_id="A",
            goal_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            target_room_id="Room 23",
            max_steps=2,
        )

        self.assertEqual(result.pano_path, ["A", "B", "A"])
        self.assertFalse(result.trajectory[1]["anti_backtrack_applied"])


class BritishMuseumImageGoalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_path = (
            PROJECT_ROOT / "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.npz"
        )
        cls.metadata_path = (
            PROJECT_ROOT
            / "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.metadata.json"
        )
        cls.manifest_root = PROJECT_ROOT / "renders/room_grounding_fov90"
        cls.representatives_path = (
            PROJECT_ROOT / "outputs/passage_clustering/room8/salad_cluster8/representatives.json"
        )
        required = [
            cls.index_path,
            cls.metadata_path,
            cls.manifest_root,
            cls.representatives_path,
        ]
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("British Museum DINOv2-SALAD artifacts are not available.")

        cls.view_store = IndexedPanoramaViewStore(
            index_path=cls.index_path,
            metadata_path=cls.metadata_path,
            manifest_root=cls.manifest_root,
        )
        representative = resolve_goal_label(
            "R8P7",
            representatives_path=cls.representatives_path,
        )
        cls.goal_embedding = cls.view_store.embedding_for_capture(
            representative["pano_id"],
            representative["capture_index"],
        )
        cls.pano_graph = json.loads(
            (PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json").read_text(
                encoding="utf-8"
            )
        )
        grounding = json.loads(
            (
                PROJECT_ROOT
                / "dataset/sites/british_museum/normalized/pano_room_grounding.json"
            ).read_text(encoding="utf-8")
        )
        cls.simulator = PanoramaGraphImageGoalSimulator(
            pano_graph=cls.pano_graph,
            pano_room_mappings=grounding["mappings"],
            observation_provider=cls.view_store.load_views,
        )

    def test_room8_to_room23_from_erp(self) -> None:
        expected = [
            "ERpTO5uJ6-_RB4dSoaicGQ",
            "WW6UK09Lg3p4Jpxsf_54mg",
            "7grGsbOXqpEMDLgTG6VfmQ",
            "yhefW17T4Ru4HFlymjRMWQ",
            "JK44-iXUvic1mv3GUGMaUQ",
            "6NL7LiZ1lgnAM6AGrACYyw",
            "cz-P-2bFqhRZa9h9b8mMFg",
            "Li54te8XaSyXgj2x_c2msA",
        ]
        result = self.simulator.run(
            start_pano_id=expected[0],
            goal_embedding=self.goal_embedding,
            target_room_id="Room 23",
            max_steps=20,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.pano_path, expected)
        self.assertEqual(result.step_count, 7)
        self._assert_actions_are_graph_edges(result.trajectory)

    def test_room8_to_room23_from_nc6(self) -> None:
        expected = [
            "nc6tPn6x91aocmWAz__Xsw",
            "5j-j28-T36sl6IE5n1W64A",
            "6NL7LiZ1lgnAM6AGrACYyw",
            "cz-P-2bFqhRZa9h9b8mMFg",
            "Li54te8XaSyXgj2x_c2msA",
        ]
        result = self.simulator.run(
            start_pano_id=expected[0],
            goal_embedding=self.goal_embedding,
            target_room_id="Room 23",
            max_steps=20,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.pano_path, expected)
        self.assertEqual(result.step_count, 4)
        self._assert_actions_are_graph_edges(result.trajectory)

    def _assert_actions_are_graph_edges(self, trajectory: list[dict]) -> None:
        for step in trajectory:
            neighbors = {
                edge["target_pano_id"]
                for edge in self.pano_graph[step["current_pano_id"]]["neighbors"]
            }
            self.assertIn(step["next_pano_id"], neighbors)


if __name__ == "__main__":
    unittest.main()
