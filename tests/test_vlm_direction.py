from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from memory_nav.navigation import (
    EightViewVLMDirectionSelector,
    IndexedPanoramaViewStore,
    RecordedDirectionSelector,
    SparseVLMDirectionSimulator,
    VisualView,
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
                pano_room_mappings={"A": "Room 1", "B": "Room 1", "C": "Room 1", "P": "Room 1"},
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


class BritishMuseumSparseDirectionTests(unittest.TestCase):
    def test_recorded_views_reach_room23(self) -> None:
        index_path = (
            PROJECT_ROOT / "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz"
        )
        metadata_path = (
            PROJECT_ROOT
            / "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
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
        selector = RecordedDirectionSelector(["V5", "V6", "V7"])
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
        self.assertEqual(selector.call_index, 3)


if __name__ == "__main__":
    unittest.main()
