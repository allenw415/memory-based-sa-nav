from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.data.build_navigation_paths import (
    build_navigation_paths,
    build_parser,
    build_target_groups,
    constrained_shortest_pano_path,
    difficulty_for_visited_room_count,
    pano_edge_distances,
    ratio_statistics,
    validate_topology_constrained_path,
    write_outputs,
)


def write_artifacts(root: Path) -> None:
    room_graph = {
        "Room A": {
            "room_id": "Room A",
            "floor": "0",
            "category": "Test",
            "synthetic": False,
            "neighbors": [{"target_room_id": "Room B"}],
        },
        "Room B": {
            "room_id": "Room B",
            "floor": "0",
            "category": "Test",
            "synthetic": False,
            "neighbors": [
                {"target_room_id": "Room A"},
                {"target_room_id": "Room C"},
            ],
        },
        "Room C": {
            "room_id": "Room C",
            "floor": "0",
            "category": "Test",
            "synthetic": False,
            "neighbors": [{"target_room_id": "Room B"}],
        },
        "Room X": {
            "room_id": "Room X",
            "floor": "1",
            "category": "Test",
            "synthetic": False,
        },
        "Stairs": {
            "room_id": "Stairs",
            "floor": "0",
            "category": "Circulation",
            "synthetic": True,
        },
    }
    pano_graph = {
        "a0": {
            "pano_id": "a0",
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0,
            "neighbors": [
                {"target_pano_id": "x1"},
                {"target_pano_id": "b0"},
            ],
        },
        "b0": {
            "pano_id": "b0",
            "floor": "0",
            "lat": 0.001,
            "lng": 0.0001,
            "neighbors": [
                {"target_pano_id": "a0"},
                {"target_pano_id": "c0"},
            ],
        },
        "c0": {
            "pano_id": "c0",
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0002,
            "neighbors": [
                {"target_pano_id": "b0"},
                {"target_pano_id": "x1"},
            ],
        },
        "x1": {
            "pano_id": "x1",
            "floor": "1",
            "lat": 0.0,
            "lng": 0.0001,
            "neighbors": [
                {"target_pano_id": "a0"},
                {"target_pano_id": "c0"},
            ],
        },
    }
    grounding = {
        "mappings": {
            "a0": "Room A",
            "b0": "Room B",
            "c0": "Room C",
            "x1": "Room X",
        }
    }
    root.mkdir(parents=True)
    (root / "room_graph.json").write_text(json.dumps(room_graph))
    (root / "pano_graph.json").write_text(json.dumps(pano_graph))
    (root / "pano_room_grounding.json").write_text(json.dumps(grounding))


def write_semantic_artifacts(root: Path) -> Path:
    room_graph = {
        "Room Start": {
            "room_id": "Room Start",
            "floor": "0",
            "title": "Start",
            "category": "Test",
            "synthetic": False,
            "neighbors": [{"target_room_id": "Connector"}],
        },
        "Room Target A": {
            "room_id": "Room Target A",
            "floor": "0",
            "title": "Grouped target",
            "category": "Test",
            "synthetic": False,
            "neighbors": [
                {"target_room_id": "Connector"},
                {"target_room_id": "Room Target B"},
            ],
        },
        "Room Target B": {
            "room_id": "Room Target B",
            "floor": "0",
            "title": "Grouped target",
            "category": "Test",
            "synthetic": False,
            "neighbors": [{"target_room_id": "Room Target A"}],
        },
        "Connector": {
            "room_id": "Connector",
            "floor": "0",
            "title": "Connector",
            "category": "Circulation",
            "synthetic": True,
            "neighbors": [
                {"target_room_id": "Room Start"},
                {"target_room_id": "Room Target A"},
            ],
        },
    }
    pano_graph = {
        "start": {
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0,
            "neighbors": [{"target_pano_id": "connector"}],
        },
        "connector": {
            "floor": "0",
            "lat": 0.001,
            "lng": 0.0001,
            "neighbors": [
                {"target_pano_id": "start"},
                {"target_pano_id": "target-a"},
            ],
        },
        "target-a": {
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0002,
            "neighbors": [
                {"target_pano_id": "connector"},
                {"target_pano_id": "target-b"},
            ],
        },
        "target-b": {
            "floor": "0",
            "lat": 0.0,
            "lng": 0.0004,
            "neighbors": [{"target_pano_id": "target-a"}],
        },
    }
    grounding = {
        "mappings": {
            "start": "Room Start",
            "connector": "Connector",
            "target-a": "Room Target A",
            "target-b": "Room Target B",
        }
    }
    config = {
        "floor": "0",
        "excluded_target_room_ids": ["Room Start"],
        "equivalent_groups": [
            {
                "target_group_id": "grouped_target",
                "target_group_theme": "Grouped target",
                "acceptable_target_room_ids": ["Room Target A", "Room Target B"],
            }
        ],
    }
    root.mkdir(parents=True)
    (root / "room_graph.json").write_text(json.dumps(room_graph))
    (root / "pano_graph.json").write_text(json.dumps(pano_graph))
    (root / "pano_room_grounding.json").write_text(json.dumps(grounding))
    config_path = root / "target_groups.json"
    config_path.write_text(json.dumps(config))
    return config_path


class NavigationPathDifficultyTests(unittest.TestCase):
    def test_default_difficulty_boundaries(self) -> None:
        self.assertEqual(difficulty_for_visited_room_count(2), "easy")
        self.assertEqual(difficulty_for_visited_room_count(3), "easy")
        self.assertEqual(difficulty_for_visited_room_count(4), "medium")
        self.assertEqual(difficulty_for_visited_room_count(5), "medium")
        self.assertEqual(difficulty_for_visited_room_count(6), "hard")

    def test_invalid_thresholds_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            difficulty_for_visited_room_count(
                2,
                easy_max_visited_rooms=1,
                medium_max_visited_rooms=5,
            )
        with self.assertRaises(ValueError):
            difficulty_for_visited_room_count(
                2,
                easy_max_visited_rooms=3,
                medium_max_visited_rooms=3,
            )

    def test_ratio_statistics_keep_raw_tail(self) -> None:
        summary = ratio_statistics([1.0, 2.0, 100.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["max"], 100.0)
        self.assertGreater(summary["p99"], 90.0)


class SemanticTargetNavigationPathTests(unittest.TestCase):
    def test_floor_zero_config_merges_and_excludes_expected_rooms(self) -> None:
        config_path = Path(
            "dataset/sites/british_museum/normalized/navigation_target_groups_floor0.json"
        )
        config = json.loads(config_path.read_text())
        floor_rooms = {
            "Room 1", "Room 2", "Room 4", "Room 6", "Room 7", "Room 8",
            "Room 9", "Room 10", "Room 11", "Room 12", "Room 13", "Room 14",
            "Room 15", "Room 16", "Room 17", "Room 18", "Room 18a", "Room 18b",
            "Room 19", "Room 20", "Room 21", "Room 22", "Room 23", "Room 24",
            "Room 26", "Room 27", "Room 29a", "Room 29b",
        }
        groups = build_target_groups(target_room_ids=floor_rooms, config=config)
        groups_by_id = {group["target_group_id"]: group for group in groups}

        self.assertEqual(len(groups), 24)
        self.assertEqual(
            groups_by_id["assyria_nimrud"]["acceptable_target_room_ids"],
            ["Room 7", "Room 8"],
        )
        self.assertEqual(
            groups_by_id["india"]["acceptable_target_room_ids"],
            ["Room 29a", "Room 29b"],
        )
        target_rooms = {
            room_id
            for group in groups
            for room_id in group["acceptable_target_room_ids"]
        }
        self.assertNotIn("Room 18a", target_rooms)
        self.assertNotIn("Room 18b", target_rooms)

    def test_group_uses_nearest_member_and_counts_circulation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts_dir = root / "artifacts"
            target_groups_path = write_semantic_artifacts(artifacts_dir)
            args = build_parser().parse_args(
                [
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--target-groups-json",
                    str(target_groups_path),
                ]
            )

            records, summary = build_navigation_paths(args)

            self.assertEqual(len(records), 1)
            self.assertEqual(summary["target_group_count"], 1)
            self.assertEqual(summary["target_room_count"], 2)
            self.assertEqual(summary["skipped_same_target_group_count"], 2)
            record = records[0]
            self.assertEqual(record["start_room_id"], "Room Start")
            self.assertEqual(record["target_group_id"], "grouped_target")
            self.assertEqual(
                record["acceptable_target_room_ids"],
                ["Room Target A", "Room Target B"],
            )
            self.assertEqual(record["reference_end_room_id"], "Room Target A")
            self.assertEqual(record["target_room_id"], "Room Target A")
            self.assertEqual(
                record["shortest_path_rooms"],
                ["Room Start", "Connector", "Room Target A"],
            )
            self.assertEqual(record["visited_room_count"], 3)
            self.assertEqual(record["difficulty"], "easy")
            self.assertEqual(
                record["path_constraint_mode"],
                "room_topology_bfs_constrained_dijkstra",
            )
            self.assertEqual(record["topology_validation_status"], "valid")


class FloorFilteredNavigationPathTests(unittest.TestCase):
    def test_build_and_write_floor_zero_candidate_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifacts_dir = root / "artifacts"
            output_dir = root / "outputs"
            write_artifacts(artifacts_dir)
            args = build_parser().parse_args(
                [
                    "--artifacts-dir",
                    str(artifacts_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            records, summary = build_navigation_paths(args)
            output_paths = write_outputs(output_dir, records, summary)

            self.assertEqual(len(records), 6)
            self.assertEqual(summary["floor"], "0")
            self.assertEqual(summary["floor_room_count"], 3)
            self.assertEqual(summary["target_room_count"], 3)
            self.assertEqual(summary["filtered_pano_graph_count"], 3)
            self.assertEqual(summary["topology_room_count"], 3)
            self.assertEqual(summary["topology_validation"]["valid_count"], 6)
            self.assertEqual(summary["skipped_same_room_count"], 3)
            self.assertEqual(summary["skipped_zero_straight_distance_count"], 0)
            self.assertEqual(summary["difficulty_distribution"]["easy"]["count"], 6)
            self.assertEqual(
                sum(
                    item["count"]
                    for item in summary["visited_room_count_distribution"]
                ),
                6,
            )

            a_to_c = next(
                record
                for record in records
                if record["start_room_id"] == "Room A"
                and record["target_room_id"] == "Room C"
            )
            self.assertEqual(a_to_c["shortest_path_panos"], ["a0", "b0", "c0"])
            self.assertEqual(
                a_to_c["shortest_path_rooms"],
                ["Room A", "Room B", "Room C"],
            )
            self.assertGreater(a_to_c["detour_ratio"], 1.0)
            self.assertAlmostEqual(
                a_to_c["detour_ratio"],
                a_to_c["path_distance_m"]
                / a_to_c["straight_line_distance_m"],
            )
            self.assertNotIn("complexity_score", a_to_c)
            self.assertNotIn("detour_ratio_score", a_to_c)
            for record in records:
                self.assertEqual(record["floor"], "0")
                self.assertNotIn("x1", record["shortest_path_panos"])
                self.assertNotEqual(record["start_room_id"], record["target_room_id"])

            jsonl_lines = output_paths["jsonl"].read_text().splitlines()
            with output_paths["csv"].open(newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(jsonl_lines), len(records))
            self.assertEqual(len(csv_rows), len(records))
            self.assertEqual(json.loads(jsonl_lines[0])["floor"], "0")

            for key in (
                "visited_room_count_plot",
                "detour_ratio_histogram_easy",
                "detour_ratio_histogram_medium",
                "detour_ratio_histogram_hard",
                "detour_ratio_ecdf",
                "detour_ratio_heatmap",
                "shortest_path_distance_heatmap",
            ):
                payload = output_paths[key].read_bytes()
                self.assertTrue(payload.startswith(b"\x89PNG"))
                self.assertGreater(len(payload), 100)


class TopologyConstrainedDijkstraTests(unittest.TestCase):
    def test_null_pano_cannot_skip_required_topology_room(self) -> None:
        pano_graph = {
            "a": {
                "lat": 0.0,
                "lng": 0.0,
                "neighbors": [
                    {"target_pano_id": "null"},
                    {"target_pano_id": "b"},
                ],
            },
            "null": {
                "lat": 0.0,
                "lng": 0.00001,
                "neighbors": [
                    {"target_pano_id": "a"},
                    {"target_pano_id": "c"},
                ],
            },
            "b": {
                "lat": 0.0,
                "lng": 0.001,
                "neighbors": [
                    {"target_pano_id": "a"},
                    {"target_pano_id": "c"},
                ],
            },
            "c": {
                "lat": 0.0,
                "lng": 0.002,
                "neighbors": [
                    {"target_pano_id": "null"},
                    {"target_pano_id": "b"},
                ],
            },
        }
        mappings = {"a": "Room A", "b": "Room B", "c": "Room C"}
        route = ["Room A", "Room B", "Room C"]
        result = constrained_shortest_pano_path(
            pano_graph,
            mappings,
            start_pano_id="a",
            target_pano_ids=["c"],
            topology_room_route=route,
            edge_distances=pano_edge_distances(pano_graph),
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result[2], ["a", "b", "c"])
        self.assertTrue(validate_topology_constrained_path(result[2], route, mappings))
        self.assertFalse(
            validate_topology_constrained_path(["a", "null", "c"], route, mappings)
        )


if __name__ == "__main__":
    unittest.main()
