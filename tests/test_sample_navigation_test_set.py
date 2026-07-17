from __future__ import annotations

import unittest
from collections import Counter

from tools.data.sample_navigation_test_set import (
    DIFFICULTY_ORDER,
    FIXED_RATIO_STRATUM_ORDER,
    RATIO_TERTILE_ORDER,
    add_fixed_ratio_passage_profiles,
    canonical_query_from_theme,
    fixed_ratio_stratum_for_value,
    sample_fixed_ratio_passage_controlled,
    sample_pilot,
)


def synthetic_candidate_pool() -> list[dict]:
    target_groups = [f"target_{index:02d}" for index in range(22)] + [
        "assyria_nimrud",
        "india",
    ]
    start_rooms = [f"start_{index:02d}" for index in range(28)]
    ratio_values = (1.1, 1.5, 2.0)
    records: list[dict] = []
    for difficulty_index, difficulty in enumerate(DIFFICULTY_ORDER):
        for ratio_index, ratio in enumerate(ratio_values):
            for start_index, start_room_id in enumerate(start_rooms):
                for target_index, target_group_id in enumerate(target_groups):
                    if target_group_id == "assyria_nimrud":
                        reference_room_id = (
                            "Room 7"
                            if (start_index + difficulty_index + ratio_index) % 2 == 0
                            else "Room 8"
                        )
                    elif target_group_id == "india":
                        reference_room_id = (
                            "Room 29a"
                            if (start_index + difficulty_index + ratio_index) % 2 == 0
                            else "Room 29b"
                        )
                    else:
                        reference_room_id = target_group_id
                    index = len(records) + 1
                    records.append(
                        {
                            "path_id": f"P{index:06d}",
                            "difficulty": difficulty,
                            "detour_ratio": ratio,
                            "target_group_id": target_group_id,
                            "reference_end_room_id": reference_room_id,
                            "start_room_id": start_room_id,
                            "start_pano_id": (
                                f"p-{difficulty_index}-{ratio_index}-"
                                f"{start_index}-{target_index}"
                            ),
                            "shortest_path_rooms": [
                                start_room_id,
                                (
                                    f"via-{difficulty_index}-{ratio_index}-"
                                    f"{start_index}-{target_index}"
                                ),
                                reference_room_id,
                            ],
                        }
                    )
    return records


class NavigationPilotSamplingTests(unittest.TestCase):
    def test_canonical_query_names_only_the_gallery_theme(self) -> None:
        query = canonical_query_from_theme(
            "Living and Dying; The Wellcome Trust Gallery"
        )

        self.assertEqual(query, "Take me to the Living and Dying gallery.")
        self.assertNotIn("Room", query)
        self.assertNotIn("Wellcome", query)
        self.assertEqual(
            canonical_query_from_theme("The world of Alexander"),
            "Take me to the world of Alexander gallery.",
        )

    def test_seed_42_balanced_sample_is_diverse_and_reproducible(self) -> None:
        candidates = synthetic_candidate_pool()

        first, first_cutpoints = sample_pilot(candidates, seed=42, per_stratum=10)
        second, second_cutpoints = sample_pilot(candidates, seed=42, per_stratum=10)

        self.assertEqual([row["path_id"] for row in first], [row["path_id"] for row in second])
        self.assertEqual(first_cutpoints, second_cutpoints)
        self.assertEqual(len(first), 90)
        strata = Counter((row["difficulty"], row["ratio_tertile"]) for row in first)
        for difficulty in DIFFICULTY_ORDER:
            for tertile in RATIO_TERTILE_ORDER:
                self.assertEqual(strata[(difficulty, tertile)], 10)
        self.assertEqual(len({row["start_pano_id"] for row in first}), 90)
        self.assertEqual(
            len({(row["start_room_id"], row["target_group_id"]) for row in first}),
            90,
        )
        self.assertEqual(
            len({tuple(row["shortest_path_rooms"]) for row in first}),
            90,
        )
        target_counts = Counter(row["target_group_id"] for row in first)
        self.assertEqual(len(target_counts), 24)
        self.assertTrue(all(3 <= count <= 4 for count in target_counts.values()))
        start_counts = Counter(row["start_room_id"] for row in first)
        self.assertEqual(len(start_counts), 28)
        self.assertTrue(all(count <= 4 for count in start_counts.values()))
        self.assertEqual(
            {
                row["reference_end_room_id"]
                for row in first
                if row["target_group_id"] == "assyria_nimrud"
            },
            {"Room 7", "Room 8"},
        )
        self.assertEqual(
            {
                row["reference_end_room_id"]
                for row in first
                if row["target_group_id"] == "india"
            },
            {"Room 29a", "Room 29b"},
        )

    def test_fixed_ratio_boundaries(self) -> None:
        expected = {
            1.0: "1.0-1.5",
            1.499999: "1.0-1.5",
            1.5: "1.5-2.0",
            1.999999: "1.5-2.0",
            2.0: "2.0-2.5",
            2.5: "2.0-2.5",
        }
        for value, label in expected.items():
            self.assertEqual(fixed_ratio_stratum_for_value(value)["label"], label)
        self.assertIsNone(fixed_ratio_stratum_for_value(0.999999))
        self.assertIsNone(fixed_ratio_stratum_for_value(2.500001))

    def test_passage_fail_edges_are_directed_and_all_hits_are_saved(self) -> None:
        records = [
            {
                "path_id": "risk",
                "difficulty": "easy",
                "detour_ratio": 1.2,
                "shortest_path_rooms": [
                    "NORTH STAIRS",
                    "Room 33",
                    "Room 24",
                    "Room 26",
                ],
            },
            {
                "path_id": "reverse",
                "difficulty": "easy",
                "detour_ratio": 1.2,
                "shortest_path_rooms": ["Room 26", "Room 24"],
            },
        ]
        enriched, diagnostics = add_fixed_ratio_passage_profiles(
            records,
            (("North stairs", "Room 33"), ("Room 24", "Room 26")),
        )

        self.assertEqual(diagnostics["eligible_candidate_count"], 2)
        self.assertEqual(enriched[0]["passage_profile"], "risk")
        self.assertEqual(
            enriched[0]["known_failed_passage_edges_on_path"],
            [
                {"from_room_id": "North stairs", "to_room_id": "Room 33"},
                {"from_room_id": "Room 24", "to_room_id": "Room 26"},
            ],
        )
        self.assertEqual(enriched[1]["passage_profile"], "reliable")
        self.assertEqual(enriched[1]["known_failed_passage_edges_on_path"], [])

    def test_fixed_ratio_milp_sample_meets_all_constraints_and_repeats(self) -> None:
        candidates = synthetic_candidate_pool()
        for record in candidates:
            start_index = int(str(record["start_room_id"]).rsplit("_", 1)[1])
            difficulty_index = DIFFICULTY_ORDER.index(record["difficulty"])
            if (start_index + difficulty_index) % 2 == 0:
                original = list(record["shortest_path_rooms"])
                record["shortest_path_rooms"] = [
                    original[0],
                    "Known fail source",
                    "Known fail target",
                    *original[1:],
                ]
        fail_edges = (("Known fail source", "Known fail target"),)

        first, first_diagnostics = sample_fixed_ratio_passage_controlled(
            candidates,
            fail_edges,
            seed=42,
            reliable_per_stratum=8,
            risk_per_stratum=2,
        )
        second, second_diagnostics = sample_fixed_ratio_passage_controlled(
            candidates,
            fail_edges,
            seed=42,
            reliable_per_stratum=8,
            risk_per_stratum=2,
        )

        self.assertEqual(
            [record["path_id"] for record in first],
            [record["path_id"] for record in second],
        )
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertEqual(len(first), 90)
        profiles = Counter(record["passage_profile"] for record in first)
        self.assertEqual(profiles, {"reliable": 72, "risk": 18})
        strata = Counter(
            (
                record["difficulty"],
                record["ratio_stratum"],
                record["passage_profile"],
            )
            for record in first
        )
        for difficulty in DIFFICULTY_ORDER:
            for ratio_stratum in FIXED_RATIO_STRATUM_ORDER:
                self.assertEqual(
                    strata[(difficulty, ratio_stratum, "reliable")],
                    8,
                )
                self.assertEqual(strata[(difficulty, ratio_stratum, "risk")], 2)
        self.assertEqual(len({record["start_pano_id"] for record in first}), 90)
        self.assertEqual(
            len(
                {
                    (record["start_room_id"], record["target_group_id"])
                    for record in first
                }
            ),
            90,
        )
        self.assertEqual(
            len({tuple(record["shortest_path_rooms"]) for record in first}),
            90,
        )
        target_counts = Counter(record["target_group_id"] for record in first)
        self.assertEqual(len(target_counts), 24)
        self.assertTrue(all(3 <= count <= 4 for count in target_counts.values()))
        start_counts = Counter(record["start_room_id"] for record in first)
        self.assertEqual(len(start_counts), 28)
        self.assertTrue(all(count <= 4 for count in start_counts.values()))
        self.assertEqual(
            {
                record["reference_end_room_id"]
                for record in first
                if record["target_group_id"] == "assyria_nimrud"
            },
            {"Room 7", "Room 8"},
        )
        self.assertEqual(
            {
                record["reference_end_room_id"]
                for record in first
                if record["target_group_id"] == "india"
            },
            {"Room 29a", "Room 29b"},
        )


if __name__ == "__main__":
    unittest.main()
