from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from memory_nav.memory.retrieval import MemoryLocalizationResult, MemoryRoomLocalizer
from memory_nav.navigation import (
    DynamicPassageRetriever,
    IndexedPanoramaViewStore,
    NavigationEpisodeRunner,
    PassageVLMSelector,
    RecordedPassageSelector,
    VisualActionDecision,
    VisualView,
    resolve_goal_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeTextEmbedder:
    def __init__(self, embedding):
        self.embedding = np.asarray(embedding, dtype=np.float32)

    def encode_texts(self, _texts):
        return np.asarray([self.embedding], dtype=np.float32)


class DynamicPassageRetrieverTests(unittest.TestCase):
    def test_retrieves_top_images_and_uses_stable_cluster_representatives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata = []
            for index in range(4):
                pano_id = f"p{index}"
                image_dir = root / pano_id
                image_dir.mkdir()
                image_path = image_dir / f"p{index}_{index:02d}_view_000deg.png"
                image_path.write_bytes(b"image")
                metadata.append(
                    {
                        "pano_id": pano_id,
                        "room_id": "Room 8",
                        "capture_index": index,
                        "capture_label": "view",
                        "capture_heading": float(index * 10),
                        "capture_path": str(image_path),
                    }
                )

            retriever = DynamicPassageRetriever(
                render_root=root,
                retrieval_top_k=4,
                target_clusters=2,
                text_embedder=FakeTextEmbedder([1.0, 0.0]),
                semantic_embeddings=np.asarray(
                    [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
                    dtype=np.float32,
                ),
                semantic_metadata_items=metadata,
                visual_embeddings=np.asarray(
                    [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
                    dtype=np.float32,
                ),
                visual_metadata_items=metadata,
            )

            first = retriever.retrieve("Room 8")
            second = retriever.retrieve("Room 8")

        self.assertEqual(first, second)
        self.assertEqual([item["label"] for item in first], ["R8P1", "R8P2"])
        self.assertEqual([item["memory_index"] for item in first], [0, 2])
        self.assertEqual([item["cluster_size"] for item in first], [2, 2])

    def test_returns_empty_for_unknown_room(self) -> None:
        retriever = DynamicPassageRetriever(
            text_embedder=FakeTextEmbedder([1.0, 0.0]),
            semantic_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            semantic_metadata_items=[
                {"pano_id": "p", "room_id": "Room 8", "capture_index": 0}
            ],
            visual_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
            visual_metadata_items=[
                {"pano_id": "p", "room_id": "Room 8", "capture_index": 0}
            ],
        )
        self.assertEqual(retriever.retrieve("Room 99"), [])


class LocalizationExclusionTests(unittest.TestCase):
    def test_localizer_forwards_same_pano_exclusion(self) -> None:
        class FakeRetriever:
            def __init__(self):
                self.excluded = None

            def query_image_paths(self, _paths, *, top_k, exclude_same_pano_ids):
                self.excluded = set(exclude_same_pano_ids)
                return [
                    {
                        "candidate_pano_id": "other",
                        "candidate_capture_index": 0,
                        "room_id": "Room 8",
                        "score": 1.0,
                    }
                ]

        retriever = FakeRetriever()
        localizer = MemoryRoomLocalizer(
            retriever,
            confidence_threshold=0.0,
            margin_threshold=0.0,
        )
        result = localizer.localize_from_images(
            ["photo.png"],
            exclude_same_pano_ids={"hidden-current"},
        )
        self.assertEqual(retriever.excluded, {"hidden-current"})
        self.assertEqual(result.predicted_room_id, "Room 8")


class PassageVLMRequestTests(unittest.TestCase):
    def test_request_does_not_expose_scores_headings_or_graph_direction(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image = Path(temp_dir) / "image.png"
            image.write_bytes(b"image")
            selector = PassageVLMSelector(
                model_client=None,  # type: ignore[arg-type]
                model="test-model",
            )
            request = selector.build_request(
                current_room_id="Room 8",
                subgoal_room_id="Room 23",
                current_candidates=[
                    {
                        "label": "R8P7",
                        "image_path": str(image),
                        "semantic_score": 0.99,
                        "capture_heading": 270.0,
                    }
                ],
                subgoal_candidates=[
                    {
                        "label": "R23P1",
                        "image_path": str(image),
                        "semantic_score": 0.98,
                        "capture_heading": 90.0,
                    }
                ],
            )
        serialized = json.dumps(request)
        self.assertNotIn("semantic_score", serialized)
        self.assertNotIn("capture_heading", serialized)
        self.assertNotIn("allocentric", serialized)


class FakeViewStore:
    def __init__(self, root: Path, headings: dict[str, float]):
        self.root = root
        self.headings = headings
        for pano_id in headings:
            (root / f"{pano_id}.png").write_bytes(b"image")

    def load_views(self, pano_id: str):
        return [
            VisualView(
                capture_index=0,
                label="front",
                heading=self.headings[pano_id],
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                path=str(self.root / f"{pano_id}.png"),
            )
        ]

    def embedding_for_capture(self, _pano_id: str, _capture_index: int):
        return np.asarray([1.0, 0.0], dtype=np.float32)


class MappingLocalizer:
    def __init__(self, path_to_room: dict[str, str]):
        self.path_to_room = path_to_room
        self.exclusions: list[set[str]] = []

    def localize_from_images(self, image_paths, *, exclude_same_pano_ids):
        self.exclusions.append(set(exclude_same_pano_ids))
        room_id = self.path_to_room[Path(image_paths[0]).stem]
        return confident_localization(room_id)


class FirstCandidateSelector:
    def choose(self, *, current_candidates, **_kwargs):
        return {
            "chosen_label": current_candidates[0]["label"],
            "navigation_confidence": 1.0,
            "selector_source": "fake",
        }


class FakePassageRetriever:
    def retrieve(self, room_id: str):
        digits = "".join(char for char in room_id if char.isdigit())
        return [
            {
                "label": f"R{digits}P1",
                "room_id": room_id,
                "pano_id": room_id,
                "capture_index": 0,
                "image_path": f"{room_id}.png",
            }
        ]


class NavigationEpisodeWaypointTests(unittest.TestCase):
    def test_room_transition_uses_simulator_localization(self) -> None:
        class SequencedLocalizer:
            def localize_from_images(self, image_paths, *, exclude_same_pano_ids):
                pano_id = Path(image_paths[0]).stem
                if pano_id not in exclude_same_pano_ids:
                    raise AssertionError("Current pano must be excluded")
                if pano_id == "B":
                    return MemoryLocalizationResult(
                        predicted_room_id="Room 1",
                        confidence=0.3,
                        margin=0.1,
                        is_confident=False,
                    )
                if pano_id == "C":
                    return MemoryLocalizationResult(
                        predicted_room_id="Room 2",
                        confidence=0.3,
                        margin=0.1,
                        is_confident=False,
                    )
                return confident_localization("Room 1")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = NavigationEpisodeRunner(
                room_graph={
                    "Room 1": {"neighbors": [{"target_room_id": "Room 2"}]},
                    "Room 2": {"neighbors": [{"target_room_id": "Room 1"}]},
                },
                pano_graph={
                    "A": {
                        "lat": 0.0,
                        "lng": 0.0,
                        "neighbors": [
                            {"target_pano_id": "B", "geocentric_heading_deg": 0.0}
                        ],
                    },
                    "B": {
                        "lat": 0.0,
                        "lng": 0.0001,
                        "neighbors": [
                            {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                            {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                        ],
                    },
                    "C": {
                        "lat": 0.0,
                        "lng": 0.0002,
                        "neighbors": [
                            {"target_pano_id": "B", "geocentric_heading_deg": 180.0}
                        ],
                    },
                },
                pano_room_mappings={"A": "Room 1", "B": "Room 2", "C": "Room 2"},
                view_store=FakeViewStore(root, {"A": 0.0, "B": 0.0, "C": 180.0}),
                localizer=SequencedLocalizer(),
                passage_retriever=FakePassageRetriever(),
                passage_selector=FirstCandidateSelector(),
            )
            result = runner.run(
                start_pano_id="A",
                target_room_id="Room 2",
                max_total_steps=4,
                max_local_steps=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.reason, "target_room_relocalized")
        self.assertEqual(result.pano_path, ["A", "B"])
        self.assertEqual(
            result.rounds[1]["localization_source"],
            "simulator_room_transition",
        )
        self.assertEqual(result.rounds[1]["localization"]["predicted_room_id"], "Room 2")
        metrics = result.to_dict()["navigation_metrics"]
        self.assertEqual(metrics["room_sequence"], ["Room 1", "Room 2"])
        self.assertEqual(metrics["visited_room_ids"], ["Room 1", "Room 2"])
        self.assertEqual(metrics["visited_room_count"], 2)
        self.assertEqual(metrics["room_transition_count"], 1)
        self.assertEqual(metrics["pano_step_count"], 1)
        self.assertAlmostEqual(
            metrics["start_to_final_straight_line_distance_m"],
            11.12,
            places=2,
        )
        self.assertAlmostEqual(metrics["pano_path_distance_m"], 11.12, places=2)
        self.assertEqual(metrics["distance_missing_segment_count"], 0)

    def test_low_confidence_passage_choice_stops_before_moving(self) -> None:
        class LowConfidenceSelector:
            def choose(self, *, current_candidates, **_kwargs):
                return {
                    "chosen_label": current_candidates[0]["label"],
                    "navigation_confidence": 0.49,
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runner = NavigationEpisodeRunner(
                room_graph={
                    "Room 1": {"neighbors": [{"target_room_id": "Room 2"}]},
                    "Room 2": {"neighbors": [{"target_room_id": "Room 1"}]},
                },
                pano_graph={
                    "A": {
                        "neighbors": [
                            {"target_pano_id": "B", "geocentric_heading_deg": 0.0}
                        ]
                    },
                    "B": {
                        "neighbors": [
                            {"target_pano_id": "A", "geocentric_heading_deg": 180.0}
                        ]
                    },
                },
                pano_room_mappings={"A": "Room 1", "B": "Room 2"},
                view_store=FakeViewStore(root, {"A": 0.0, "B": 180.0}),
                localizer=MappingLocalizer({"A": "Room 1", "B": "Room 2"}),
                passage_retriever=FakePassageRetriever(),
                passage_selector=LowConfidenceSelector(),
            )
            result = runner.run(
                start_pano_id="A",
                target_room_id="Room 2",
                max_total_steps=2,
                max_local_steps=2,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.reason, "passage_choice_failed")
        self.assertEqual(result.step_count, 0)
        self.assertEqual(result.pano_path, ["A"])

    def test_image_path_direction_policy_receives_chosen_passage_image(self) -> None:
        class PathGoalPolicy:
            requires_goal_image_path = True

            def __init__(self):
                self.goal_paths: list[Path] = []

            def choose_action(self, *, goal_embedding, views, legal_action_headings):
                self.goal_paths.append(Path(goal_embedding).resolve())
                return VisualActionDecision(
                    selected_capture_index=views[0].capture_index,
                    selected_view_label=views[0].label,
                    selected_view_heading=float(views[0].heading),
                    similarity=1.0,
                    second_similarity=1.0,
                    margin=0.0,
                    selected_action_index=0,
                    selected_action_heading=float(legal_action_headings[0]),
                    scoring={"mode": "image_path_similarity"},
                )

        class PathPassageRetriever:
            def __init__(self, image_path: Path):
                self.image_path = image_path

            def retrieve(self, room_id: str):
                digits = "".join(char for char in room_id if char.isdigit())
                return [
                    {
                        "label": f"R{digits}P1",
                        "room_id": room_id,
                        "pano_id": room_id,
                        "capture_index": 0,
                        "image_path": str(self.image_path),
                    }
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            passage_image = root / "chosen_passage.png"
            passage_image.write_bytes(b"image")
            policy = PathGoalPolicy()
            runner = NavigationEpisodeRunner(
                room_graph={
                    "Room 1": {"neighbors": [{"target_room_id": "Room 2"}]},
                    "Room 2": {"neighbors": [{"target_room_id": "Room 1"}]},
                },
                pano_graph={
                    "A": {
                        "neighbors": [
                            {"target_pano_id": "B", "geocentric_heading_deg": 0.0}
                        ]
                    },
                    "B": {
                        "neighbors": [
                            {"target_pano_id": "A", "geocentric_heading_deg": 180.0}
                        ]
                    },
                },
                pano_room_mappings={"A": "Room 1", "B": "Room 2"},
                view_store=FakeViewStore(root, {"A": 0.0, "B": 180.0}),
                localizer=MappingLocalizer({"A": "Room 1", "B": "Room 2"}),
                passage_retriever=PathPassageRetriever(passage_image),
                passage_selector=FirstCandidateSelector(),
                image_goal_policy=policy,
            )
            result = runner.run(
                start_pano_id="A",
                target_room_id="Room 2",
                max_total_steps=2,
                max_local_steps=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(policy.goal_paths, [passage_image.resolve()])

    def test_waypoint_is_completed_only_after_relocalization(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pano_graph = {
                "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                    ]
                },
                "C": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 180.0}]},
            }
            room_graph = {
                "Room 1": {"neighbors": [{"target_room_id": "Room 2"}]},
                "Room 2": {
                    "neighbors": [
                        {"target_room_id": "Room 1"},
                        {"target_room_id": "Room 3"},
                    ]
                },
                "Room 3": {"neighbors": [{"target_room_id": "Room 2"}]},
            }
            localizer = MappingLocalizer({"A": "Room 1", "B": "Room 2", "C": "Room 3"})
            runner = NavigationEpisodeRunner(
                room_graph=room_graph,
                pano_graph=pano_graph,
                pano_room_mappings={"A": "Room 1", "B": "Room 2", "C": "Room 3"},
                view_store=FakeViewStore(root, {"A": 0.0, "B": 0.0, "C": 180.0}),
                localizer=localizer,
                passage_retriever=FakePassageRetriever(),
                passage_selector=FirstCandidateSelector(),
            )
            result = runner.run(
                start_pano_id="A",
                target_room_id="Room 3",
                waypoint_room_ids=["Room 2"],
                max_total_steps=5,
                max_local_steps=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.completed_waypoints, ["Room 2"])
        self.assertEqual(result.pano_path, ["A", "B", "C"])
        self.assertEqual(
            [item.get("active_target_room_id") for item in result.rounds[:-1]],
            ["Room 2", "Room 3"],
        )
        self.assertEqual(localizer.exclusions, [{"A"}])
        self.assertEqual(
            [item["localization_source"] for item in result.rounds],
            ["image_retrieval", "simulator_room_transition", "simulator_room_transition"],
        )

    def test_future_waypoint_or_goal_passage_does_not_skip_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pano_graph = {
                "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                    ]
                },
                "C": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 180.0}]},
            }
            room_graph = {
                "Room 1": {"neighbors": [{"target_room_id": "Room 3"}]},
                "Room 3": {
                    "neighbors": [
                        {"target_room_id": "Room 1"},
                        {"target_room_id": "Room 2"},
                    ]
                },
                "Room 2": {"neighbors": [{"target_room_id": "Room 3"}]},
            }
            runner = NavigationEpisodeRunner(
                room_graph=room_graph,
                pano_graph=pano_graph,
                pano_room_mappings={"A": "Room 1", "B": "Room 3", "C": "Room 2"},
                view_store=FakeViewStore(root, {"A": 0.0, "B": 0.0, "C": 180.0}),
                localizer=MappingLocalizer({"A": "Room 1", "B": "Room 3", "C": "Room 2"}),
                passage_retriever=FakePassageRetriever(),
                passage_selector=FirstCandidateSelector(),
            )
            result = runner.run(
                start_pano_id="A",
                target_room_id="Room 3",
                waypoint_room_ids=["Room 2"],
                max_total_steps=6,
                max_local_steps=2,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.pano_path, ["A", "B", "C", "B"])
        self.assertEqual(result.completed_waypoints, ["Room 2"])
        self.assertNotIn("completed_target_this_round", result.rounds[1])
        self.assertEqual(result.rounds[1]["active_target_room_id"], "Room 2")


def confident_localization(room_id: str) -> MemoryLocalizationResult:
    return MemoryLocalizationResult(
        predicted_room_id=room_id,
        confidence=1.0,
        margin=1.0,
        is_confident=True,
        room_scores={room_id: 1.0},
        room_distribution={room_id: 1.0},
        top_rooms=[room_id],
        top_matches=[],
    )


class GroundTruthImageLocalizer:
    def __init__(self, mappings: dict[str, str | None]):
        self.mappings = mappings

    def localize_from_images(self, image_paths, *, exclude_same_pano_ids):
        pano_id = Path(image_paths[0]).parent.name
        if pano_id not in exclude_same_pano_ids:
            raise AssertionError("Current pano must be excluded from localization retrieval")
        room_id = self.mappings[pano_id]
        if not isinstance(room_id, str):
            return MemoryLocalizationResult(None, 0.0, 0.0, False)
        return confident_localization(room_id)


class ExistingRepresentativeRetriever:
    def __init__(self, room8: list[dict], room23: list[dict]):
        self.items = {"Room 8": room8, "Room 23": room23}

    def retrieve(self, room_id: str):
        return self.items.get(room_id, [])


class BritishMuseumEpisodeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index_path = PROJECT_ROOT / "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz"
        cls.metadata_path = PROJECT_ROOT / "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
        cls.manifest_root = PROJECT_ROOT / "renders/room_grounding_fov90"
        cls.room8_reps_path = PROJECT_ROOT / "outputs/passage_clustering/room8/salad_cluster8/representatives.json"
        cls.room23_reps_path = PROJECT_ROOT / "outputs/passage_clustering/room23/salad_cluster8/representatives.json"
        required = [
            cls.index_path,
            cls.metadata_path,
            cls.manifest_root,
            cls.room8_reps_path,
            cls.room23_reps_path,
        ]
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("British Museum navigation artifacts are unavailable.")

        cls.view_store = IndexedPanoramaViewStore(
            index_path=cls.index_path,
            metadata_path=cls.metadata_path,
            manifest_root=cls.manifest_root,
        )
        cls.pano_graph = json.loads(
            (PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json").read_text(encoding="utf-8")
        )
        cls.room_graph = json.loads(
            (PROJECT_ROOT / "dataset/sites/british_museum/normalized/room_graph.json").read_text(encoding="utf-8")
        )
        grounding = json.loads(
            (PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_room_grounding.json").read_text(encoding="utf-8")
        )
        cls.mappings = grounding["mappings"]
        room8 = resolve_goal_label("R8P7", representatives_path=cls.room8_reps_path)
        room23 = resolve_goal_label("R23P1", representatives_path=cls.room23_reps_path)
        for item in (room8, room23):
            raw_path = Path(str(item["image_path"]))
            if not raw_path.exists():
                item["image_path"] = str(
                    cls.manifest_root / item["pano_id"] / raw_path.name
                )
        cls.passage_retriever = ExistingRepresentativeRetriever([room8], [room23])
        cls.selector = RecordedPassageSelector(
            {"Room 8->Room 23": {"chosen_label": "R8P7", "navigation_confidence": 0.72}}
        )

    def _run(self, start_pano_id: str):
        runner = NavigationEpisodeRunner(
            room_graph=self.room_graph,
            pano_graph=self.pano_graph,
            pano_room_mappings=self.mappings,
            view_store=self.view_store,
            localizer=GroundTruthImageLocalizer(self.mappings),
            passage_retriever=self.passage_retriever,
            passage_selector=self.selector,
            seed=0,
        )
        return runner.run(
            start_pano_id=start_pano_id,
            target_room_id="Room 23",
            max_total_steps=20,
            max_local_steps=20,
        )

    def test_erp_to_room23(self) -> None:
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
        result = self._run(expected[0])
        self.assertTrue(result.success)
        self.assertEqual(result.pano_path, expected)
        self._assert_legal_actions(result)

    def test_nc6_to_room23(self) -> None:
        expected = [
            "nc6tPn6x91aocmWAz__Xsw",
            "5j-j28-T36sl6IE5n1W64A",
            "6NL7LiZ1lgnAM6AGrACYyw",
            "cz-P-2bFqhRZa9h9b8mMFg",
            "Li54te8XaSyXgj2x_c2msA",
        ]
        result = self._run(expected[0])
        self.assertTrue(result.success)
        self.assertEqual(result.pano_path, expected)
        self._assert_legal_actions(result)

    def _assert_legal_actions(self, result) -> None:
        for round_payload in result.rounds:
            for step in round_payload["movement_steps"]:
                neighbors = {
                    item["target_pano_id"]
                    for item in self.pano_graph[step["current_pano_id"]]["neighbors"]
                }
                self.assertIn(step["next_pano_id"], neighbors)


if __name__ == "__main__":
    unittest.main()
