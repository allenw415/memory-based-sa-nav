from __future__ import annotations

import unittest
from types import SimpleNamespace

from tools.evaluation.run_navigation_pilot import (
    aggregate_records,
    build_result_record,
    max_total_steps_for_case,
    ratio_stratum_for_case,
    summarize,
)


class PilotNavigationEvaluationTests(unittest.TestCase):
    def test_dynamic_total_step_budget(self) -> None:
        args = SimpleNamespace(
            step_multiplier=3.0,
            min_total_steps=50,
            max_total_steps=300,
        )
        self.assertEqual(
            max_total_steps_for_case({"pano_step_count": 2}, args),
            50,
        )
        self.assertEqual(
            max_total_steps_for_case({"pano_step_count": 89}, args),
            267,
        )

    def test_success_metrics_use_semantic_group_and_gallery_transitions(self) -> None:
        test_case = {
            "test_id": "TEST001",
            "path_id": "NP1",
            "difficulty": "hard",
            "ratio_stratum": "2.0-2.5",
            "ratio_stratum_lower_bound": 2.0,
            "ratio_stratum_upper_bound": 2.5,
            "ratio_stratum_lower_inclusive": True,
            "ratio_stratum_upper_inclusive": True,
            "passage_profile": "risk",
            "known_failed_passage_edges_on_path": [
                {"from_room_id": "Room 29a", "to_room_id": "Room 29b"}
            ],
            "query": "Take me to the India gallery.",
            "target_group_id": "india",
            "target_group_theme": "India",
            "acceptable_target_room_ids": ["Room 29a", "Room 29b"],
            "reference_end_room_id": "Room 29a",
            "start_pano_id": "A",
            "start_room_id": "Room 1",
            "path_distance_m": 10.0,
            "pano_step_count": 4,
        }
        parsed = SimpleNamespace(target_room_id="Room 29a")
        episode = SimpleNamespace(
            success=True,
            reason="target_room_relocalized",
            final_pano_id="D",
            pano_path=["A", "B", "C", "D"],
            navigation_metrics={
                "pano_path_distance_m": 15.0,
                "pano_step_count": 3,
                "room_sequence": ["Room 1", "East stairs", "Room 29b"],
                "room_transition_count": 2,
            },
        )
        room_graph = {
            "Room 1": {"category": "Gallery"},
            "East stairs": {"category": "Circulation"},
            "Room 29a": {"category": "Gallery"},
            "Room 29b": {"category": "Gallery"},
        }
        record = build_result_record(
            test_case=test_case,
            parsed_query=parsed,
            parse_error=None,
            episode_result=episode,
            episode_error=None,
            elapsed_s=2.5,
            query_counts=(1, 1, 0),
            max_total_steps=50,
            room_graph=room_graph,
            pano_room_mappings={"D": "Room 29b"},
        )

        self.assertTrue(record["success"])
        self.assertTrue(record["query_grounding_correct"])
        self.assertEqual(record["ratio_stratum"], "2.0-2.5")
        self.assertEqual(record["ratio_stratum_type"], "fixed_detour_ratio")
        self.assertIsNone(record["ratio_tertile"])
        self.assertEqual(record["passage_profile"], "risk")
        self.assertEqual(
            record["known_failed_passage_edges_on_path"],
            [{"from_room_id": "Room 29a", "to_room_id": "Room 29b"}],
        )
        self.assertAlmostEqual(record["actual_over_shortest_ratio"], 1.5)
        self.assertAlmostEqual(record["spl"], 2.0 / 3.0)
        self.assertEqual(record["raw_room_transitions"], 2)
        self.assertEqual(record["gallery_room_transitions"], 1)
        self.assertFalse(record["wrong_gallery_terminal"])
        self.assertTrue(record["episode_reported_success"])
        self.assertEqual(
            record["evaluation_outcome"],
            "ground_truth_target_reached",
        )

        summary = aggregate_records([record])
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertAlmostEqual(summary["spl_mean"], 2.0 / 3.0)
        self.assertEqual(summary["query_parser_logical_calls_mean"], 1.0)

    def test_ground_truth_terminal_overrides_false_positive_localization(self) -> None:
        test_case = {
            "test_id": "TEST002",
            "path_id": "NP2",
            "difficulty": "easy",
            "ratio_tertile": "high",
            "query": "Take me to the Egyptian sculpture gallery.",
            "target_group_id": "Room 4",
            "target_group_theme": "Egyptian sculpture",
            "acceptable_target_room_ids": ["Room 4"],
            "reference_end_room_id": "Room 4",
            "start_pano_id": "A",
            "start_room_id": "Room 6",
            "path_distance_m": 10.0,
            "pano_step_count": 4,
        }
        parsed = SimpleNamespace(target_room_id="Room 4")
        episode = SimpleNamespace(
            success=True,
            reason="target_room_relocalized",
            final_pano_id="A",
            pano_path=["A"],
            navigation_metrics={
                "pano_path_distance_m": 0.0,
                "pano_step_count": 0,
                "room_sequence": ["Room 6"],
                "room_transition_count": 0,
            },
        )
        room_graph = {
            "Room 4": {"category": "Gallery"},
            "Room 6": {"category": "Gallery"},
        }
        record = build_result_record(
            test_case=test_case,
            parsed_query=parsed,
            parse_error=None,
            episode_result=episode,
            episode_error=None,
            elapsed_s=1.0,
            query_counts=(1, 1, 0),
            max_total_steps=50,
            room_graph=room_graph,
            pano_room_mappings={"A": "Room 6"},
        )

        self.assertTrue(record["episode_reported_success"])
        self.assertFalse(record["success"])
        self.assertEqual(record["spl"], 0.0)
        self.assertTrue(record["wrong_gallery_terminal"])
        self.assertEqual(
            record["evaluation_outcome"],
            "false_positive_target_relocalization",
        )

    def test_unmapped_null_string_is_not_a_wrong_gallery(self) -> None:
        test_case = {
            "test_id": "TEST003",
            "path_id": "NP3",
            "difficulty": "easy",
            "ratio_tertile": "low",
            "query": "Take me to Room 4.",
            "target_group_id": "Room 4",
            "target_group_theme": "Egyptian sculpture",
            "acceptable_target_room_ids": ["Room 4"],
            "reference_end_room_id": "Room 4",
            "start_pano_id": "A",
            "start_room_id": "Room 6",
            "path_distance_m": 10.0,
            "pano_step_count": 4,
        }
        parsed = SimpleNamespace(target_room_id="Room 4")
        episode = SimpleNamespace(
            success=False,
            reason="max_local_steps_exceeded",
            final_pano_id="B",
            pano_path=["A", "B"],
            navigation_metrics={
                "pano_path_distance_m": 2.0,
                "pano_step_count": 1,
                "room_sequence": ["Room 6"],
                "room_transition_count": 0,
            },
        )
        record = build_result_record(
            test_case=test_case,
            parsed_query=parsed,
            parse_error=None,
            episode_result=episode,
            episode_error=None,
            elapsed_s=1.0,
            query_counts=(1, 1, 0),
            max_total_steps=50,
            room_graph={
                "Room 4": {"category": "Gallery"},
                "Room 6": {"category": "Gallery"},
            },
            pano_room_mappings={"B": "null"},
        )

        self.assertIsNone(record["terminal_room_id"])
        self.assertFalse(record["wrong_gallery_terminal"])

    def test_failed_episode_in_wrong_gallery_is_counted(self) -> None:
        record = {
            "success": False,
            "spl": 0.0,
            "actual_over_shortest_ratio": 0.8,
            "panorama_steps": 4,
            "raw_room_transitions": 2,
            "gallery_room_transitions": 1,
            "cycle_incidence": False,
            "cycle_termination": False,
            "wrong_gallery_terminal": True,
            "query_parse_success": True,
            "query_grounding_correct": False,
            "query_parser_logical_calls": 1,
            "query_parser_http_attempts": 1,
            "query_parser_retries": 0,
            "execution_time_s": 3.0,
            "reason": "unacceptable_target_room_relocalized",
            "terminal_room_id": "Room 10",
        }

        summary = aggregate_records([record])

        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual(summary["wrong_gallery_terminal_rate"], 1.0)
        self.assertEqual(summary["query_grounding_accuracy"], 0.0)

    def test_fixed_ratio_and_passage_profile_are_preserved_and_summarized(self) -> None:
        base = {
            "success": True,
            "spl": 0.75,
            "actual_over_shortest_ratio": 1.3,
            "panorama_steps": 4,
            "raw_room_transitions": 2,
            "gallery_room_transitions": 1,
            "cycle_incidence": False,
            "cycle_termination": False,
            "wrong_gallery_terminal": False,
            "query_parse_success": True,
            "query_grounding_correct": True,
            "query_parser_logical_calls": 1,
            "query_parser_http_attempts": 1,
            "query_parser_retries": 0,
            "execution_time_s": 3.0,
            "reason": "target_room_relocalized",
            "terminal_room_id": "Room 4",
            "difficulty": "easy",
            "ratio_stratum": "1.0-1.5",
            "ratio_stratum_type": "fixed_detour_ratio",
            "ratio_tertile": None,
            "passage_profile": "reliable",
        }
        risk = {
            **base,
            "success": False,
            "spl": 0.0,
            "cycle_incidence": True,
            "wrong_gallery_terminal": True,
            "difficulty": "hard",
            "ratio_stratum": "2.0-2.5",
            "passage_profile": "risk",
            "terminal_room_id": "Room 6",
        }

        summary = summarize([base, risk], {"scope": "test"})

        self.assertEqual(ratio_stratum_for_case(base), "1.0-1.5")
        self.assertEqual(summary["by_ratio_stratum"]["1.0-1.5"]["count"], 1)
        self.assertEqual(summary["by_passage_profile"]["reliable"]["success_rate"], 1.0)
        self.assertEqual(summary["by_passage_profile"]["risk"]["success_rate"], 0.0)
        self.assertEqual(
            summary["by_passage_profile"]["risk"]["cycle_incidence_rate"],
            1.0,
        )
        self.assertNotIn("by_ratio_tertile", summary)

    def test_controlled_recovery_remains_in_raw_cycle_but_not_uncontrolled_cycle(self) -> None:
        test_case = {
            "test_id": "TEST004",
            "path_id": "NP4",
            "difficulty": "medium",
            "ratio_tertile": "middle",
            "query": "Take me to the Egyptian sculpture gallery.",
            "target_group_id": "Room 4",
            "target_group_theme": "Egyptian sculpture",
            "acceptable_target_room_ids": ["Room 4"],
            "reference_end_room_id": "Room 4",
            "start_pano_id": "A",
            "start_room_id": "Room 6",
            "path_distance_m": 10.0,
            "pano_step_count": 3,
        }
        parsed = SimpleNamespace(target_room_id="Room 4")
        movement_steps = [
            {"step_index": 0, "decision_source": "memory_tree_decision", "next_pano_id": "B"},
            {"step_index": 1, "decision_source": "auto_follow", "next_pano_id": "C"},
            {
                "step_index": 2,
                "round_index": 0,
                "decision_source": "recovery_backtrack",
                "recovery_event_index": 0,
                "next_pano_id": "B",
            },
            {
                "step_index": 3,
                "round_index": 0,
                "decision_source": "recovery_backtrack",
                "recovery_event_index": 0,
                "next_pano_id": "A",
            },
            {"step_index": 4, "decision_source": "auto_follow", "next_pano_id": "D"},
        ]
        episode = SimpleNamespace(
            success=True,
            reason="target_room_relocalized",
            final_pano_id="D",
            pano_path=["A", "B", "C", "B", "A", "D"],
            rounds=[
                {
                    "movement_steps": movement_steps,
                    "direction_commitment": {
                        "recovery_history": [{"recovery_event_index": 0, "status": "completed"}]
                    },
                }
            ],
            navigation_metrics={
                "pano_path_distance_m": 15.0,
                "pano_step_count": 5,
                "room_sequence": ["Room 6", "Room 4"],
                "room_transition_count": 1,
            },
        )
        record = build_result_record(
            test_case=test_case,
            parsed_query=parsed,
            parse_error=None,
            episode_result=episode,
            episode_error=None,
            elapsed_s=1.0,
            query_counts=(1, 1, 0),
            max_total_steps=50,
            room_graph={
                "Room 4": {"category": "Gallery"},
                "Room 6": {"category": "Gallery"},
            },
            pano_room_mappings={"D": "Room 4"},
        )

        self.assertTrue(record["cycle_incidence"])
        self.assertFalse(record["uncontrolled_cycle_incidence"])
        self.assertTrue(record["controlled_recovery_used"])
        self.assertEqual(record["controlled_recovery_event_count"], 1)
        self.assertEqual(record["controlled_recovery_step_count"], 2)
        summary = aggregate_records([record])
        self.assertEqual(summary["cycle_incidence_rate"], 1.0)
        self.assertEqual(summary["uncontrolled_cycle_incidence_rate"], 0.0)
        self.assertEqual(summary["controlled_recovery_rate"], 1.0)

    def test_old_ratio_tertile_remains_compatible(self) -> None:
        self.assertEqual(ratio_stratum_for_case({"ratio_tertile": "middle"}), "middle")


if __name__ == "__main__":
    unittest.main()
