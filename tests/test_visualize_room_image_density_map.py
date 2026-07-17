from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.data.visualize_room_image_density_map import (
    EXPECTED_MAP_SIZE,
    ROOM_MAP_REGIONS,
    build_effective_density_records,
    room_region_pixel_area,
    write_outputs,
)


def density_record(
    room_id: str,
    *,
    image_count: int,
    panorama_count: int,
    mbr_area_m2: float | None,
    geometry_status: str,
) -> dict:
    return {
        "room_id": room_id,
        "room_title": room_id,
        "image_count": image_count,
        "panorama_count": panorama_count,
        "geometry_status": geometry_status,
        "mbr_area_m2": mbr_area_m2,
        "images_per_m2": (
            image_count / mbr_area_m2 if mbr_area_m2 is not None else None
        ),
    }


class RoomDensityMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = {
            "rooms": [
                density_record(
                    "Room 1",
                    image_count=100,
                    panorama_count=10,
                    mbr_area_m2=1000.0,
                    geometry_status="valid",
                ),
                density_record(
                    "Room 4",
                    image_count=100,
                    panorama_count=10,
                    mbr_area_m2=1400.0,
                    geometry_status="valid",
                ),
                density_record(
                    "Room 15",
                    image_count=100,
                    panorama_count=10,
                    mbr_area_m2=180.0,
                    geometry_status="valid",
                ),
                density_record(
                    "Room 14",
                    image_count=8,
                    panorama_count=1,
                    mbr_area_m2=None,
                    geometry_status="insufficient_geometry",
                ),
                density_record(
                    "Room 18a",
                    image_count=48,
                    panorama_count=6,
                    mbr_area_m2=None,
                    geometry_status="near_collinear",
                ),
            ]
        }

    def test_map_estimates_use_median_anchor_area_scale(self) -> None:
        records, calibration = build_effective_density_records(self.payload)
        scales = sorted(
            [
                1000.0 / room_region_pixel_area("Room 1"),
                1400.0 / room_region_pixel_area("Room 4"),
                180.0 / room_region_pixel_area("Room 15"),
            ]
        )
        expected_scale = scales[1]
        self.assertAlmostEqual(
            calibration["selected_m2_per_pixel"],
            expected_scale,
        )

        records_by_room = {record["room_id"]: record for record in records}
        for room_id in ("Room 14", "Room 18a"):
            record = records_by_room[room_id]
            expected_area = room_region_pixel_area(room_id) * expected_scale
            self.assertEqual(record["area_source"], "map_relative_estimate")
            self.assertAlmostEqual(record["effective_area_m2"], expected_area)
            self.assertAlmostEqual(
                record["effective_images_per_m2"],
                record["image_count"] / expected_area,
            )
            self.assertLessEqual(
                record["estimated_area_low_m2"],
                record["effective_area_m2"],
            )
            self.assertGreaterEqual(
                record["estimated_area_high_m2"],
                record["effective_area_m2"],
            )

    def test_writes_machine_readable_and_visual_outputs(self) -> None:
        records, calibration = build_effective_density_records(self.payload)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            map_path = root / "map.png"
            output_dir = root / "outputs"
            Image.new("RGB", EXPECTED_MAP_SIZE, color=(99, 101, 102)).save(map_path)
            source_path = root / "source.json"
            source_path.write_text(json.dumps(self.payload), encoding="utf-8")

            paths = write_outputs(
                output_dir,
                map_path,
                records,
                calibration,
                source_density_path=source_path,
            )

            self.assertEqual(set(paths), {"json", "csv", "choropleth_map", "bar_plot"})
            for path in paths.values():
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 100)
            payload = json.loads(paths["json"].read_text(encoding="utf-8"))
            room_14 = next(
                room for room in payload["rooms"] if room["room_id"] == "Room 14"
            )
            self.assertEqual(room_14["area_source"], "map_relative_estimate")
            self.assertTrue(math.isfinite(room_14["effective_images_per_m2"]))

    def test_all_density_rooms_have_map_regions(self) -> None:
        expected_rooms = {
            "Room 1",
            "Room 2",
            "Room 4",
            "Room 6",
            "Room 7",
            "Room 8",
            "Room 9",
            "Room 10",
            "Room 11",
            "Room 12",
            "Room 13",
            "Room 14",
            "Room 15",
            "Room 16",
            "Room 17",
            "Room 18",
            "Room 18a",
            "Room 18b",
            "Room 19",
            "Room 20",
            "Room 21",
            "Room 22",
            "Room 23",
            "Room 24",
            "Room 26",
            "Room 27",
            "Room 29a",
            "Room 29b",
        }
        self.assertEqual(set(ROOM_MAP_REGIONS), expected_rooms)


if __name__ == "__main__":
    unittest.main()
