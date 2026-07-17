from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from tools.data.build_room_image_density import (
    EARTH_RADIUS_M,
    build_parser,
    build_room_image_density,
    convex_hull,
    minimum_area_rectangle,
    project_lat_lng_to_local_m,
    write_outputs,
)


def metres_to_latitude_degrees(metres: float) -> float:
    return math.degrees(metres / EARTH_RADIUS_M)


def metres_to_longitude_degrees(metres: float, *, latitude: float = 0.0) -> float:
    return math.degrees(metres / (EARTH_RADIUS_M * math.cos(math.radians(latitude))))


def write_fixture(root: Path) -> tuple[Path, Path, Path]:
    room_graph = {
        "Room 1": {
            "room_id": "Room 1",
            "floor": "0",
            "category": "Test",
            "synthetic": False,
            "title": "Rectangle room",
        },
        "Room 2": {
            "room_id": "Room 2",
            "floor": "0",
            "category": "Test",
            "synthetic": False,
            "title": "Single-pano room",
        },
        "Room 3": {
            "room_id": "Room 3",
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
    local_positions = {
        "p1": (0.0, 0.0, "0"),
        "p2": (10.0, 0.0, "0"),
        "p3": (10.0, 5.0, "0"),
        "p4": (0.0, 5.0, "0"),
        "p5": (20.0, 0.0, "0"),
        "p6": (30.0, 0.0, "1"),
    }
    pano_graph = {
        pano_id: {
            "pano_id": pano_id,
            "floor": floor,
            "lat": metres_to_latitude_degrees(north_m),
            "lng": metres_to_longitude_degrees(east_m),
            "neighbors": [],
        }
        for pano_id, (east_m, north_m, floor) in local_positions.items()
    }
    items = []
    memory_index = 0
    for pano_id in ("p1", "p2", "p3", "p4"):
        for capture_index in range(2):
            items.append(
                {
                    "memory_index": memory_index,
                    "pano_id": pano_id,
                    "room_id": "Room 1",
                    "floor": "0",
                    "capture_index": capture_index,
                }
            )
            memory_index += 1
    for capture_index in range(3):
        items.append(
            {
                "memory_index": memory_index,
                "pano_id": "p5",
                "room_id": "Room 2",
                "floor": "0",
                "capture_index": capture_index,
            }
        )
        memory_index += 1
    items.append(
        {
            "memory_index": memory_index,
            "pano_id": "p6",
            "room_id": "Room 3",
            "floor": "1",
            "capture_index": 0,
        }
    )
    metadata = {
        "summary": {"floor": "0/1", "image_count": len(items)},
        "items": items,
    }

    root.mkdir(parents=True)
    metadata_path = root / "metadata.json"
    pano_graph_path = root / "pano_graph.json"
    room_graph_path = root / "room_graph.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    pano_graph_path.write_text(json.dumps(pano_graph), encoding="utf-8")
    room_graph_path.write_text(json.dumps(room_graph), encoding="utf-8")
    return metadata_path, pano_graph_path, room_graph_path


class LocalProjectionTests(unittest.TestCase):
    def test_projection_reports_east_and_north_metres(self) -> None:
        east_m, north_m = project_lat_lng_to_local_m(
            metres_to_latitude_degrees(5.0),
            metres_to_longitude_degrees(10.0),
            reference_lat=0.0,
            reference_lng=0.0,
        )
        self.assertAlmostEqual(east_m, 10.0, places=6)
        self.assertAlmostEqual(north_m, 5.0, places=6)


class MinimumAreaRectangleTests(unittest.TestCase):
    def test_rotated_rectangle_ignores_duplicates_and_interior_points(self) -> None:
        angle = math.radians(30.0)
        cosine = math.cos(angle)
        sine = math.sin(angle)

        def rotate(point: tuple[float, float]) -> tuple[float, float]:
            x, y = point
            return x * cosine - y * sine, x * sine + y * cosine

        corners = [rotate(point) for point in [(-2, -1), (2, -1), (2, 1), (-2, 1)]]
        points = [*corners, rotate((0, 0)), corners[0]]

        hull = convex_hull(points)
        geometry = minimum_area_rectangle(points)

        self.assertEqual(len(hull), 4)
        self.assertEqual(geometry["geometry_status"], "valid")
        self.assertAlmostEqual(geometry["mbr_area_m2"], 8.0, places=6)
        self.assertAlmostEqual(geometry["mbr_width_m"], 4.0, places=6)
        self.assertAlmostEqual(geometry["mbr_height_m"], 2.0, places=6)
        self.assertAlmostEqual(geometry["mbr_orientation_deg"], 30.0, places=6)

    def test_insufficient_geometry_returns_null_area(self) -> None:
        cases = [
            [(1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0)],
            [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
        ]
        for points in cases:
            with self.subTest(points=points):
                geometry = minimum_area_rectangle(points)
                self.assertEqual(geometry["geometry_status"], "insufficient_geometry")
                self.assertIsNone(geometry["mbr_area_m2"])

    def test_near_collinear_footprint_returns_null_density_geometry(self) -> None:
        geometry = minimum_area_rectangle(
            [(0.0, 0.0), (10.0, 0.0), (10.0, 0.05), (0.0, 0.05)]
        )

        self.assertEqual(geometry["geometry_status"], "near_collinear")
        self.assertIsNone(geometry["mbr_area_m2"])


class RoomImageDensityIntegrationTests(unittest.TestCase):
    def test_counts_images_and_writes_room_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_path, pano_graph_path, room_graph_path = write_fixture(
                root / "inputs"
            )
            output_dir = root / "outputs"
            args = build_parser().parse_args(
                [
                    "--metadata",
                    str(metadata_path),
                    "--pano-graph",
                    str(pano_graph_path),
                    "--room-graph",
                    str(room_graph_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )

            records, summary, configuration = build_room_image_density(args)
            output_paths = write_outputs(output_dir, records, summary, configuration)

            self.assertEqual([record["room_id"] for record in records], ["Room 1", "Room 2"])
            room_1, room_2 = records
            self.assertEqual(room_1["image_count"], 8)
            self.assertEqual(room_1["panorama_count"], 4)
            self.assertAlmostEqual(room_1["mbr_area_m2"], 50.0, places=5)
            self.assertAlmostEqual(room_1["images_per_m2"], 8.0 / 50.0, places=6)
            self.assertAlmostEqual(room_1["panoramas_per_m2"], 4.0 / 50.0, places=6)
            self.assertEqual(room_2["image_count"], 3)
            self.assertEqual(room_2["panorama_count"], 1)
            self.assertEqual(room_2["geometry_status"], "insufficient_geometry")
            self.assertIsNone(room_2["images_per_m2"])

            self.assertEqual(summary["room_count"], 2)
            self.assertEqual(summary["valid_mbr_room_count"], 1)
            self.assertEqual(summary["total_memory_image_count"], 11)
            self.assertEqual(summary["total_unique_panorama_count"], 5)
            self.assertEqual(summary["excluded_metadata_item_count"], 1)

            payload = json.loads(output_paths["json"].read_text(encoding="utf-8"))
            with output_paths["csv"].open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(payload["rooms"]), 2)
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["image_count"], "8")
            self.assertEqual(csv_rows[1]["images_per_m2"], "")
            for key in ("image_count_plot", "image_density_plot"):
                image_payload = output_paths[key].read_bytes()
                self.assertTrue(image_payload.startswith(b"\x89PNG"))
                self.assertGreater(len(image_payload), 100)


if __name__ == "__main__":
    unittest.main()
