from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from memory_nav.navigation import (
    EightViewVLMDirectionSelector,
    IndexedPanoramaViewStore,
    MemoryTreeDirectionSelector,
    RecordedDirectionSelector,
    SparseVLMDirectionSimulator,
    VisualView,
    angular_distance_deg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_views(root: Path, pano_id: str, headings: list[float]) -> list[VisualView]:
    views = []
    for index, heading in enumerate(headings):
        path = root / f"{pano_id}_{index}.png"
        path.write_bytes(b"image")
        views.append(
            VisualView(
                capture_index=index,
                label=f"view_{index}",
                heading=heading,
                embedding=np.asarray([1.0, 0.0], dtype=np.float32),
                path=str(path),
            )
        )
    return views


def scored_direction_response(
    *,
    chosen_view: str,
    scores: dict[str, float],
) -> dict:
    labels = [f"V{index}" for index in range(1, 9)]
    ordered = sorted(labels, key=lambda label: (-float(scores.get(label, 0.0)), label))
    return {
        "chosen_view": chosen_view,
        "confidence": 1.0,
        "reason": "scored test response",
        "memory_tree": {
            "view_scores": [
                {
                    "rank": rank,
                    "view_label": label,
                    "root_target_similarity": float(scores.get(label, 0.0)),
                }
                for rank, label in enumerate(ordered, 1)
            ]
        },
    }


def _metadata(
    path: Path,
    pano_id: str,
    capture_index: int,
    capture_label: str = "view",
    capture_heading: float = 0.0,
) -> dict:
    return {
        "memory_index": capture_index,
        "room_id": "Room 1",
        "pano_id": pano_id,
        "capture_index": capture_index,
        "capture_label": capture_label,
        "capture_heading": capture_heading,
        "image_path": str(path.resolve()),
    }


class EightViewVLMSelectorTests(unittest.TestCase):
    def test_request_contains_goal_and_all_eight_views_without_headings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = root / "goal.png"
            goal.write_bytes(b"goal")
            views = make_views(root, "A", [0, 45, 90, 135, 180, 225, 270, 315])
            selector = EightViewVLMDirectionSelector(
                model_client=None,  # type: ignore[arg-type]
                model="test-model",
            )
            request = selector.build_request(goal_image_path=goal, views=views)

        content = request["input"][0]["content"]
        self.assertEqual(sum(item["type"] == "input_image" for item in content), 9)
        serialized = json.dumps(request)
        for index in range(1, 9):
            self.assertIn(f"V{index}", serialized)
        self.assertNotIn("capture_index", serialized)
        self.assertNotIn("heading", serialized)
        self.assertNotIn("pano_id", serialized)

    def test_recorded_selector_rejects_an_unknown_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            views = make_views(
                Path(temp_dir),
                "A",
                [0, 45, 90, 135, 180, 225, 270, 315],
            )
            selector = RecordedDirectionSelector(["V9"])
            with self.assertRaises(RuntimeError):
                selector.choose(goal_image_path="goal.png", views=views)


class MemoryTreeDirectionSelectorTests(unittest.TestCase):
    def test_memory_tree_selector_uses_bidirection_alignment(self) -> None:
        class FakeEmbedder:
            def __init__(self, vectors):
                self.vectors = vectors

            def encode_image_paths(self, paths):
                return np.asarray([self.vectors[str(Path(path).resolve())] for path in paths], dtype=np.float32)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = root / "goal.png"
            goal.write_bytes(b"goal")
            good_mid = root / "good_mid.png"
            good_mid.write_bytes(b"mid")
            bad_mid = root / "bad_mid.png"
            bad_mid.write_bytes(b"bad")
            views = make_views(root, "A", [0, 45, 90, 135, 180, 225, 270, 315])
            vectors = {
                str(goal.resolve()): [1.0, 0.0, 0.0],
                str(good_mid.resolve()): [0.6, 0.8, 0.0],
                str(bad_mid.resolve()): [0.2, -0.98, 0.0],
            }
            for view in views:
                vectors[str(Path(view.path).resolve())] = [0.0, -1.0, 0.0]
            vectors[str(Path(views[2].path).resolve())] = [0.0, 1.0, 0.0]
            metadata = [
                _metadata(goal, "target", 0),
                _metadata(good_mid, "good-mid", 1),
                _metadata(bad_mid, "bad-mid", 2),
                *[
                    _metadata(Path(view.path), "A", int(view.capture_index), view.label, view.heading)
                    for view in views
                ],
            ]
            selector = MemoryTreeDirectionSelector(
                metadata_items=metadata,
                image_embedder=FakeEmbedder(vectors),
                branching_factor=1,
                max_depth=2,
            )
            result = selector.choose(goal_image_path=goal, views=views)

        self.assertEqual(result["chosen_view"], "V3")
        self.assertEqual(result["selector_source"], "memory_tree")
        self.assertEqual(result["memory_tree"]["mode"], "bidirection_passage_alignment")
        self.assertEqual(result["memory_tree"]["selection_reason"], "best_bridge_score")
        self.assertEqual(result["memory_tree"]["method"], "dreamsim_bidirection_passage_alignment")
        self.assertEqual(result["memory_tree"]["best_bridge"]["bridge_score_mode"], "bidirection")
        self.assertIn("continuity_score", result["memory_tree"]["best_bridge"])
        self.assertIn("current_chain_root_to_bridge", result["memory_tree"])
        self.assertIn("passage_chain_bridge_to_target", result["memory_tree"])
        self.assertEqual(result["memory_tree"]["view_scores"][0]["view_label"], "V3")


    def test_memory_tree_selector_can_use_dinov2_patch_pairwise_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            goal = root / "goal.png"
            goal.write_bytes(b"goal")
            bridge = root / "bridge.png"
            bridge.write_bytes(b"bridge")
            distractor = root / "distractor.png"
            distractor.write_bytes(b"distractor")
            views = make_views(root, "A", [0, 45, 90, 135, 180, 225, 270, 315])
            metadata = [
                _metadata(goal, "target", 0),
                _metadata(bridge, "bridge", 1),
                _metadata(distractor, "distractor", 2),
                *[
                    _metadata(Path(view.path), "A", int(view.capture_index), view.label, view.heading)
                    for view in views
                ],
            ]
            pairwise = np.eye(len(metadata), dtype=np.float32)
            pairwise[0, 1] = pairwise[1, 0] = 0.96
            for view_index in range(3, len(metadata)):
                pairwise[view_index, 2] = pairwise[2, view_index] = 0.95
                pairwise[view_index, 1] = pairwise[1, view_index] = 0.10
            pairwise[5, 1] = pairwise[1, 5] = 0.98
            selector = MemoryTreeDirectionSelector(
                metadata_items=metadata,
                similarity_backend="dinov2_patch_topk",
                branching_factor=1,
                max_depth=1,
                bridge_selection_mode="bridge_then_continuity",
                near_duplicate_threshold=1.1,
                patch_cache_dir=root / "cache",
            )
            with patch(
                "tools.experiments.build_passage_memory_tree.load_or_encode_dinov2_patch_features",
                return_value=np.zeros((len(metadata), 2, 3), dtype=np.float32),
            ), patch(
                "tools.experiments.build_passage_memory_tree.patch_topk_similarity_matrix",
                return_value=pairwise,
            ):
                result = selector.choose(goal_image_path=goal, views=views)

        self.assertEqual(result["chosen_view"], "V3")
        self.assertEqual(result["selector_source"], "memory_tree")
        self.assertEqual(result["request_summary"]["similarity_backend"], "dinov2_patch_topk")
        self.assertEqual(result["request_summary"]["bridge_selection_mode"], "bridge_then_continuity")
        self.assertEqual(result["memory_tree"]["best_bridge"]["bridge_score_mode"], "bridge_then_continuity")

    def test_direct_current_view_match_selects_matching_view_without_embedding(self) -> None:
        class FakeEmbedder:
            def encode_image_paths(self, _paths):
                raise AssertionError("Direct match should not encode images")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            views = make_views(root, "A", [0, 45, 90, 135, 180, 225, 270, 315])
            selector = MemoryTreeDirectionSelector(
                metadata_items=[],
                image_embedder=FakeEmbedder(),
                branching_factor=3,
                max_depth=5,
            )
            result = selector.choose(goal_image_path=views[2].path, views=views)

        self.assertEqual(result["chosen_view"], "V3")
        self.assertEqual(result["selector_source"], "memory_tree")
        self.assertEqual(result["memory_tree"]["mode"], "direct_current_view_match")
        self.assertEqual(result["request_summary"]["branching_factor"], 3)
        self.assertEqual(result["request_summary"]["max_depth"], 5)


class SparseVLMDirectionSimulatorTests(unittest.TestCase):
    def test_maps_selected_view_across_north_and_avoids_backtrack(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            views = make_views(root, "A", [45, 90, 135, 180, 225, 270, 315, 359])
            simulator = SparseVLMDirectionSimulator(
                pano_graph={
                    "A": {
                        "neighbors": [
                            {"target_pano_id": "B", "geocentric_heading_deg": 1.0},
                            {"target_pano_id": "P", "geocentric_heading_deg": 359.0},
                            {"target_pano_id": "C", "geocentric_heading_deg": 180.0},
                        ]
                    },
                    "B": {"neighbors": []},
                    "C": {"neighbors": []},
                    "P": {"neighbors": []},
                },
                pano_room_mappings={"A": "Room 1", "B": "Room 2", "C": "Room 1", "P": "Room 1"},
                observation_provider=lambda _pano_id: views,
                direction_selector=RecordedDirectionSelector(["V8"]),
            )
            result = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id="P",
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"P", "A"},
                max_steps=1,
            )

        self.assertEqual(result.actions[0]["next_pano_id"], "B")
        self.assertTrue(result.actions[0]["anti_backtrack_applied"])
        self.assertAlmostEqual(result.actions[0]["heading_difference"], 2.0)

    def test_vlm_heading_continues_through_aligned_branches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {"neighbors": [
                    {"target_pano_id": "X", "geocentric_heading_deg": 300.0},
                    {"target_pano_id": "B", "geocentric_heading_deg": 145.0},
                ]},
                "B": {"neighbors": [
                    {"target_pano_id": "A", "geocentric_heading_deg": 325.0},
                    {"target_pano_id": "Y", "geocentric_heading_deg": 80.0},
                    {"target_pano_id": "C", "geocentric_heading_deg": 143.0},
                ]},
                "C": {"neighbors": [
                    {"target_pano_id": "B", "geocentric_heading_deg": 323.0},
                    {"target_pano_id": "Z", "geocentric_heading_deg": 55.0},
                    {"target_pano_id": "D", "geocentric_heading_deg": 146.0},
                ]},
                "D": {"neighbors": []},
                "X": {"neighbors": []},
                "Y": {"neighbors": []},
                "Z": {"neighbors": []},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 135, 150, 225, 270, 315])
                for pano_id in graph
            }
            selector = RecordedDirectionSelector(["V5"])
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={pano_id: "Room 1" for pano_id in graph},
                observation_provider=lambda pano_id: views[pano_id],
                direction_selector=selector,
                burst_steps=3,
                max_turn_deg=45.0,
            )
            result = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A"},
            )

        self.assertEqual([item["next_pano_id"] for item in result.actions], ["B", "C", "D"])
        self.assertEqual([item["decision_source"] for item in result.actions], ["vlm_decision", "auto_follow", "auto_follow"])
        self.assertEqual(selector.call_index, 1)

    def test_vlm_command_preserves_selected_view_heading_into_null_connector(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {
                    "neighbors": [
                        {"target_pano_id": "X", "geocentric_heading_deg": 120.0},
                        {"target_pano_id": "B", "geocentric_heading_deg": 290.0},
                    ]
                },
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 110.0},
                        {"target_pano_id": "Y", "geocentric_heading_deg": 60.0},
                        {"target_pano_id": "N", "geocentric_heading_deg": 238.0},
                    ]
                },
                "N": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 58.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 200.0},
                    ]
                },
                "C": {"neighbors": []},
                "X": {"neighbors": []},
                "Y": {"neighbors": []},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 135, 180, 225, 240, 315])
                for pano_id in ["A", "B"]
            }

            def observation_provider(pano_id):
                if pano_id == "N":
                    raise AssertionError("null connector should not require indexed views")
                return views[pano_id]

            selector = RecordedDirectionSelector(["V7"])
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={
                    "A": "Room 1",
                    "B": "Room 1",
                    "N": "null",
                    "C": "Room 2",
                    "X": "Room 1",
                    "Y": "Room 1",
                },
                observation_provider=observation_provider,
                direction_selector=selector,
                burst_steps=3,
                max_turn_deg=45.0,
            )
            result = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A"},
            )

        self.assertEqual([item["next_pano_id"] for item in result.actions], ["B", "N", "C"])
        self.assertEqual(
            [item["decision_source"] for item in result.actions],
            ["vlm_decision", "auto_follow", "null_connector_follow"],
        )
        self.assertEqual(result.stop_reason, "room_transition")
        self.assertAlmostEqual(result.actions[1]["heading_difference"], 2.0)
        self.assertEqual(selector.call_index, 1)

    def test_null_connector_keeps_following_original_heading_without_views(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {"neighbors": [{"target_pano_id": "N", "geocentric_heading_deg": 0.0}]},
                "N": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "B", "geocentric_heading_deg": 4.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 85.0},
                    ]
                },
                "B": {"neighbors": []},
                "C": {"neighbors": []},
            }
            views = {"A": make_views(root, "A", [0, 45, 90, 135, 180, 225, 270, 315])}

            def observation_provider(pano_id):
                if pano_id == "N":
                    raise AssertionError("null connector should not require indexed views")
                return views[pano_id]

            selector = RecordedDirectionSelector([])
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={"A": "Room 1", "N": "null", "B": "Room 2", "C": "Room 3"},
                observation_provider=observation_provider,
                direction_selector=selector,
                burst_steps=3,
            )
            result = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A"},
            )

        self.assertEqual([item["next_pano_id"] for item in result.actions], ["N", "B"])
        self.assertEqual([item["decision_source"] for item in result.actions], ["auto_follow", "null_connector_follow"])
        self.assertEqual(result.stop_reason, "room_transition")
        self.assertEqual(result.actions[1]["current_views"], [])
        self.assertEqual(selector.call_index, 0)

    def test_auto_follows_unique_edges_for_at_most_three_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 5.0},
                    ]
                },
                "C": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 185.0},
                        {"target_pano_id": "D", "geocentric_heading_deg": 10.0},
                    ]
                },
                "D": {"neighbors": [{"target_pano_id": "C", "geocentric_heading_deg": 190.0}]},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 135, 180, 225, 270, 315])
                for pano_id in graph
            }
            selector = RecordedDirectionSelector([])
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={pano_id: "Room 1" for pano_id in graph},
                observation_provider=lambda pano_id: views[pano_id],
                direction_selector=selector,
                burst_steps=3,
            )
            result = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A"},
            )

        self.assertEqual([item["next_pano_id"] for item in result.actions], ["B", "C", "D"])
        self.assertEqual(result.stop_reason, "max_burst_steps")
        self.assertEqual(selector.call_index, 0)
        self.assertTrue(all(item["decision_source"] == "auto_follow" for item in result.actions))

    def test_stops_before_branch_turn_cycle_and_at_room_transition(self) -> None:
        cases = [
            (
                {
                    "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                    "B": {
                        "neighbors": [
                            {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                            {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                            {"target_pano_id": "D", "geocentric_heading_deg": 90.0},
                        ]
                    },
                    "C": {"neighbors": []},
                    "D": {"neighbors": []},
                },
                {"A": "R1", "B": "R1", "C": "R1", "D": "R1"},
                "branching_point",
            ),
            (
                {
                    "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                    "B": {
                        "neighbors": [
                            {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                            {"target_pano_id": "C", "geocentric_heading_deg": 90.0},
                        ]
                    },
                    "C": {"neighbors": []},
                },
                {"A": "R1", "B": "R1", "C": "R1"},
                "turn_exceeds_threshold",
            ),
            (
                {
                    "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                    "B": {
                        "neighbors": [
                            {"target_pano_id": "A", "geocentric_heading_deg": 180.0}
                        ]
                    },
                },
                {"A": "R1", "B": "R1"},
                "cycle_detected",
            ),
            (
                {
                    "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                    "B": {"neighbors": []},
                },
                {"A": "R1", "B": "R2"},
                "room_transition",
            ),
        ]
        for graph, rooms, expected_reason in cases:
            with self.subTest(expected_reason=expected_reason), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                views = {
                    pano_id: make_views(
                        root,
                        pano_id,
                        [0, 45, 90, 135, 180, 225, 270, 315],
                    )
                    for pano_id in graph
                }
                simulator = SparseVLMDirectionSimulator(
                    pano_graph=graph,
                    pano_room_mappings=rooms,
                    observation_provider=lambda pano_id: views[pano_id],
                    direction_selector=RecordedDirectionSelector([]),
                    burst_steps=3,
                    max_turn_deg=45.0,
                )
                result = simulator.run_burst(
                    current_pano_id="A",
                    previous_pano_id=None,
                    goal_image_path=root / "A_0.png",
                    step_index=0,
                    visited_pano_ids={"A"},
                )
            self.assertEqual(result.stop_reason, expected_reason)


    def test_memory_tree_falls_back_when_top_view_enters_cycle_trap(self) -> None:
        class RankedMemoryTreeSelector:
            def choose(self, *, goal_image_path, views):
                return {
                    "chosen_view": "V1",
                    "confidence": 1.0,
                    "reason": "test ranking",
                    "selector_source": "memory_tree",
                    "memory_tree": {
                        "view_scores": [
                            {"rank": 1, "view_label": "V1"},
                            {"rank": 2, "view_label": "V3"},
                        ]
                    },
                }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {
                    "neighbors": [
                        {"target_pano_id": "Dead", "geocentric_heading_deg": 0.0},
                        {"target_pano_id": "Good", "geocentric_heading_deg": 90.0},
                    ]
                },
                "Dead": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0}
                    ]
                },
                "Good": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 270.0},
                        {"target_pano_id": "Next", "geocentric_heading_deg": 90.0},
                    ]
                },
                "Next": {"neighbors": []},
            }
            views = {
                pano_id: make_views(
                    root,
                    pano_id,
                    [0, 45, 90, 135, 180, 225, 270, 315],
                )
                for pano_id in graph
            }
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={pano_id: "Room 1" for pano_id in graph},
                observation_provider=lambda pano_id: views[pano_id],
                direction_selector=RankedMemoryTreeSelector(),
                burst_steps=2,
            )
            result = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A"},
            )

        self.assertEqual(result.actions[0]["next_pano_id"], "Good")
        self.assertEqual(result.actions[0]["selected_view_label"], "V3")
        decision = result.actions[0]["direction_decision"]
        self.assertEqual(decision["chosen_view"], "V3")
        self.assertEqual(decision["cycle_avoidance"]["original_view"], "V1")
        self.assertEqual(
            decision["cycle_avoidance"]["skipped_candidates"][0]["reason"],
            "projected_cycle",
        )


    def test_visual_hysteresis_keeps_near_tie_and_switches_on_clear_advantage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                        {"target_pano_id": "D", "geocentric_heading_deg": 90.0},
                    ]
                },
                "C": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "E", "geocentric_heading_deg": 5.0},
                    ]
                },
                "D": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 270.0},
                        {"target_pano_id": "F", "geocentric_heading_deg": 90.0},
                    ]
                },
                "E": {"neighbors": []},
                "F": {"neighbors": []},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 135, 180, 225, 270, 315])
                for pano_id in graph
            }
            cases = [
                (0.92, "C", "kept"),
                (0.95, "D", "switched"),
            ]
            for best_score, expected_target, expected_decision in cases:
                with self.subTest(best_score=best_score):
                    selector = RecordedDirectionSelector(
                        [
                            scored_direction_response(
                                chosen_view="V3",
                                scores={"V3": best_score, "V1": 0.90},
                            )
                        ]
                    )
                    simulator = SparseVLMDirectionSimulator(
                        pano_graph=graph,
                        pano_room_mappings={pano_id: "Room 1" for pano_id in graph},
                        observation_provider=lambda pano_id: views[pano_id],
                        direction_selector=selector,
                        burst_steps=1,
                        commitment_mode="visual_hysteresis",
                        switch_margin=0.03,
                    )
                    state = simulator.create_commitment_state(current_pano_id="B")
                    state.last_action_heading = 0.0
                    result = simulator.run_burst(
                        current_pano_id="B",
                        previous_pano_id="A",
                        goal_image_path=root / "A_0.png",
                        step_index=0,
                        visited_pano_ids={"A", "B"},
                        commitment_state=state,
                    )

                    self.assertEqual(result.actions[0]["next_pano_id"], expected_target)
                    commitment = result.actions[0]["direction_commitment"]
                    self.assertEqual(commitment["decision"], expected_decision)
                    self.assertAlmostEqual(commitment["score_gap"], best_score - 0.90)

    def test_visual_hysteresis_allows_curved_continuation_and_rejects_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 0.0}]},
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 50.0},
                        {"target_pano_id": "D", "geocentric_heading_deg": 120.0},
                    ]
                },
                "C": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 230.0},
                        {"target_pano_id": "E", "geocentric_heading_deg": 55.0},
                    ]
                },
                "D": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 300.0},
                        {"target_pano_id": "F", "geocentric_heading_deg": 120.0},
                    ]
                },
                "E": {"neighbors": []},
                "F": {"neighbors": []},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 120, 180, 225, 270, 315])
                for pano_id in graph
            }
            selector = RecordedDirectionSelector(
                [
                    scored_direction_response(
                        chosen_view="V4",
                        scores={"V4": 0.92, "V2": 0.90},
                    ),
                    scored_direction_response(
                        chosen_view="V4",
                        scores={"V4": 0.92, "V2": 0.90},
                    ),
                ]
            )
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={pano_id: "Room 1" for pano_id in graph},
                observation_provider=lambda pano_id: views[pano_id],
                direction_selector=selector,
                burst_steps=1,
                commitment_mode="visual_hysteresis",
            )
            state = simulator.create_commitment_state(current_pano_id="B")
            state.last_action_heading = 0.0
            result = simulator.run_burst(
                current_pano_id="B",
                previous_pano_id="A",
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A", "B"},
                commitment_state=state,
            )
            self.assertEqual(result.actions[0]["next_pano_id"], "C")
            self.assertEqual(result.actions[0]["direction_commitment"]["decision"], "kept")
            self.assertAlmostEqual(state.last_action_heading, 50.0)

            state = simulator.create_commitment_state(current_pano_id="B")
            state.last_action_heading = 0.0
            result = simulator.run_burst(
                current_pano_id="B",
                previous_pano_id="A",
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids={"A", "B", "C"},
                commitment_state=state,
            )
            self.assertEqual(result.actions[0]["next_pano_id"], "D")
            self.assertEqual(
                result.actions[0]["direction_commitment"]["continuation_block_reason"],
                "visited_pano",
            )

    def test_recovery_retraces_to_branch_blocks_failed_edge_and_uses_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 0.0},
                        {"target_pano_id": "D", "geocentric_heading_deg": 90.0},
                    ]
                },
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                    ]
                },
                "C": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 180.0}]},
                "D": {"neighbors": [{"target_pano_id": "A", "geocentric_heading_deg": 270.0}]},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 135, 180, 225, 270, 315])
                for pano_id in graph
            }
            selector = RecordedDirectionSelector(
                [scored_direction_response(chosen_view="V1", scores={"V1": 0.95, "V3": 0.80})]
            )
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={"A": "Room 1", "B": "Room 1", "C": "Room 1", "D": "Room 2"},
                observation_provider=lambda pano_id: views[pano_id],
                direction_selector=selector,
                burst_steps=3,
                commitment_mode="visual_hysteresis",
                recovery_budget=1,
            )
            state = simulator.create_commitment_state(current_pano_id="A")
            visited = {"A"}
            first = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids=visited,
                max_steps=2,
                commitment_state=state,
            )
            self.assertEqual([item["next_pano_id"] for item in first.actions], ["B", "C"])

            second = simulator.run_burst(
                current_pano_id="C",
                previous_pano_id="B",
                goal_image_path=root / "A_0.png",
                step_index=2,
                visited_pano_ids=visited,
                commitment_state=state,
            )

            self.assertEqual(
                [item["decision_source"] for item in second.actions],
                ["recovery_backtrack", "recovery_backtrack", "auto_follow"],
            )
            self.assertEqual([item["next_pano_id"] for item in second.actions], ["B", "A", "D"])
            self.assertEqual(second.stop_reason, "room_transition")
            self.assertEqual(state.recovery_events_used, 1)
            self.assertIn(("A", "B"), state.blocked_edges)
            self.assertEqual(state.recovery_history[0]["status"], "completed")

    def test_cycle_terminates_after_recovery_budget_is_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph = {
                "A": {
                    "neighbors": [
                        {"target_pano_id": "B", "geocentric_heading_deg": 0.0},
                        {"target_pano_id": "D", "geocentric_heading_deg": 90.0},
                    ]
                },
                "B": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 180.0},
                        {"target_pano_id": "C", "geocentric_heading_deg": 0.0},
                    ]
                },
                "C": {"neighbors": [{"target_pano_id": "B", "geocentric_heading_deg": 180.0}]},
                "D": {
                    "neighbors": [
                        {"target_pano_id": "A", "geocentric_heading_deg": 270.0},
                        {"target_pano_id": "E", "geocentric_heading_deg": 90.0},
                    ]
                },
                "E": {"neighbors": []},
            }
            views = {
                pano_id: make_views(root, pano_id, [0, 45, 90, 135, 180, 225, 270, 315])
                for pano_id in graph
            }
            simulator = SparseVLMDirectionSimulator(
                pano_graph=graph,
                pano_room_mappings={pano_id: "Room 1" for pano_id in graph},
                observation_provider=lambda pano_id: views[pano_id],
                direction_selector=RecordedDirectionSelector(
                    [scored_direction_response(chosen_view="V1", scores={"V1": 0.95})]
                ),
                burst_steps=2,
                commitment_mode="visual_hysteresis",
                recovery_budget=1,
            )
            state = simulator.create_commitment_state(current_pano_id="A")
            visited = {"A"}
            first = simulator.run_burst(
                current_pano_id="A",
                previous_pano_id=None,
                goal_image_path=root / "A_0.png",
                step_index=0,
                visited_pano_ids=visited,
                commitment_state=state,
            )
            self.assertEqual([item["next_pano_id"] for item in first.actions], ["B", "C"])
            state.recovery_events_used = state.recovery_budget

            second = simulator.run_burst(
                current_pano_id="C",
                previous_pano_id="B",
                goal_image_path=root / "A_0.png",
                step_index=2,
                visited_pano_ids=visited,
                commitment_state=state,
            )

            self.assertEqual(second.stop_reason, "cycle_detected")
            self.assertEqual(second.actions, [])
            self.assertEqual(state.recovery_history, [])

    def test_new_commitment_state_does_not_leak_previous_room_state(self) -> None:
        simulator = SparseVLMDirectionSimulator(
            pano_graph={"A": {"neighbors": []}, "B": {"neighbors": []}},
            pano_room_mappings={"A": "Room 1", "B": "Room 2"},
            observation_provider=lambda _pano_id: [],
            direction_selector=RecordedDirectionSelector([]),
            commitment_mode="visual_hysteresis",
        )
        first = simulator.create_commitment_state(current_pano_id="A")
        first.blocked_edges.add(("A", "X"))
        first.recovery_events_used = 1
        second = simulator.create_commitment_state(current_pano_id="B")

        self.assertEqual(second.room_path, ["B"])
        self.assertEqual(second.blocked_edges, set())
        self.assertEqual(second.recovery_events_used, 0)


class BritishMuseumSparseDirectionTests(unittest.TestCase):
    def test_room4_v5c_commitment_reaches_room8_without_cycle(self) -> None:
        index_path = (
            PROJECT_ROOT / "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.npz"
        )
        metadata_path = (
            PROJECT_ROOT
            / "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.metadata.json"
        )
        manifest_root = PROJECT_ROOT / "renders/room_grounding_fov90"
        goal_image = (
            manifest_root
            / "1VaXjbDOjNCp09wI7ErFEg"
            / "1VaXjbDOjNCp09wI7ErFEg_06_west_240deg.png"
        )
        required = [index_path, metadata_path, manifest_root, goal_image]
        if not all(path.exists() for path in required):
            self.skipTest("British Museum Room 4 direction artifacts are unavailable.")

        graph = json.loads(
            (PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json").read_text()
        )
        rooms = json.loads(
            (
                PROJECT_ROOT
                / "dataset/sites/british_museum/normalized/pano_room_grounding.json"
            ).read_text()
        )["mappings"]
        store = IndexedPanoramaViewStore(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_root=manifest_root,
        )
        desired_targets = {
            "V5C-BZZ8zVYaqHr29nbBgg": "eBlyjXFWD1GpZeZzAYNdvA",
            "qJDhoHccxGag27ICHzdNeA": "rkyezsZZtdjxgqnAwgt0Ww",
            "rkyezsZZtdjxgqnAwgt0Ww": "iwnGaONabzJfe_pGR2QDiw",
            "xXRCdcpobXQG1U4kzuS0DQ": "plVMvg3bokHDkkTaG7C__Q",
            "plVMvg3bokHDkkTaG7C__Q": "9GZ_v8bYbDTgV2Md9eV03Q",
        }

        # Test-only recorded evidence; the production controller never receives this route map.
        class RecordedRoom4PassageDirection:
            def choose(self, *, goal_image_path, views):
                del goal_image_path
                ordered = sorted(views, key=lambda item: item.capture_index)
                pano_id = Path(str(ordered[0].path)).parent.name
                target_pano_id = desired_targets[pano_id]
                target_edge = next(
                    edge
                    for edge in graph[pano_id]["neighbors"]
                    if edge["target_pano_id"] == target_pano_id
                )
                selected_index = min(
                    range(8),
                    key=lambda index: angular_distance_deg(
                        ordered[index].heading,
                        float(target_edge["geocentric_heading_deg"]),
                    ),
                )
                view_order = [selected_index] + [
                    index for index in range(8) if index != selected_index
                ]
                return {
                    "chosen_view": f"V{selected_index + 1}",
                    "confidence": 1.0,
                    "reason": "recorded Room 4 passage direction",
                    "selector_source": "memory_tree",
                    "memory_tree": {
                        "view_scores": [
                            {
                                "rank": rank,
                                "view_label": f"V{index + 1}",
                                "root_target_similarity": (
                                    0.95 if index == selected_index else 0.80
                                ),
                            }
                            for rank, index in enumerate(view_order, 1)
                        ]
                    },
                }

        simulator = SparseVLMDirectionSimulator(
            pano_graph=graph,
            pano_room_mappings=rooms,
            observation_provider=store.load_views,
            direction_selector=RecordedRoom4PassageDirection(),
            burst_steps=3,
            max_turn_deg=45.0,
            commitment_mode="visual_hysteresis",
            switch_margin=0.03,
            recovery_budget=1,
        )
        current = "V5C-BZZ8zVYaqHr29nbBgg"
        previous = None
        path = [current]
        visited = {current}
        state = simulator.create_commitment_state(current_pano_id=current)
        for _ in range(10):
            result = simulator.run_burst(
                current_pano_id=current,
                previous_pano_id=previous,
                goal_image_path=goal_image,
                step_index=len(path) - 1,
                visited_pano_ids=visited,
                commitment_state=state,
            )
            for action in result.actions:
                previous, current = action["current_pano_id"], action["next_pano_id"]
                path.append(current)
            if rooms.get(current) == "Room 8":
                break
            self.assertTrue(result.actions, result.stop_reason)

        self.assertEqual(path[1], "eBlyjXFWD1GpZeZzAYNdvA")
        self.assertEqual(rooms.get(current), "Room 8")
        self.assertEqual(len(path), len(set(path)))
        self.assertEqual(state.recovery_events_used, 0)

    def test_recorded_views_reach_room23(self) -> None:
        index_path = (
            PROJECT_ROOT / "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.npz"
        )
        metadata_path = (
            PROJECT_ROOT
            / "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.metadata.json"
        )
        manifest_root = PROJECT_ROOT / "renders/room_grounding_fov90"
        required = [index_path, metadata_path, manifest_root]
        if not all(path.exists() for path in required):
            self.skipTest("British Museum eight-view artifacts are unavailable.")

        graph = json.loads(
            (PROJECT_ROOT / "dataset/sites/british_museum/normalized/pano_graph.json").read_text()
        )
        rooms = json.loads(
            (
                PROJECT_ROOT
                / "dataset/sites/british_museum/normalized/pano_room_grounding.json"
            ).read_text()
        )["mappings"]
        store = IndexedPanoramaViewStore(
            index_path=index_path,
            metadata_path=metadata_path,
            manifest_root=manifest_root,
        )
        selector = RecordedDirectionSelector(["V5", "V6"])
        simulator = SparseVLMDirectionSimulator(
            pano_graph=graph,
            pano_room_mappings=rooms,
            observation_provider=store.load_views,
            direction_selector=selector,
            burst_steps=3,
            max_turn_deg=45.0,
        )
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
        current = expected[0]
        previous = None
        path = [current]
        visited = {current}
        for _ in range(10):
            result = simulator.run_burst(
                current_pano_id=current,
                previous_pano_id=previous,
                goal_image_path=manifest_root
                / "YrClsQHy3IMBgzvZCwCqww"
                / "YrClsQHy3IMBgzvZCwCqww_07_west_to_north_251deg.png",
                step_index=len(path) - 1,
                visited_pano_ids=visited,
            )
            for action in result.actions:
                previous, current = action["current_pano_id"], action["next_pano_id"]
                path.append(current)
            if rooms.get(current) == "Room 23":
                break

        self.assertEqual(path, expected)
        self.assertEqual(selector.call_index, 2)


if __name__ == "__main__":
    unittest.main()
