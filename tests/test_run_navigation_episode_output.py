from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from memory_nav.cli._common import render_json
from memory_nav.cli.run_navigation_episode import (
    build_parser,
    _build_role_model_client,
    _direction_scoring_label,
    _method_id,
    _navigation_output_dir,
    _prepare_navigation_output_bundle,
    _resolve_navigation_goal,
    _write_navigation_output_bundle,
)


class NavigationEpisodeDefaultPolicyTests(unittest.TestCase):
    def test_parser_defaults_to_detected_contrastive_memory_tree(self) -> None:
        args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--target-room-id",
                "Room 17",
            ]
        )

        self.assertEqual(args.passage_policy, "detected_contrastive")
        self.assertEqual(args.direction_policy, "memory_tree")
        self.assertEqual(args.memory_tree_branching_factor, 3)
        self.assertEqual(args.memory_tree_max_depth, 5)
        self.assertEqual(args.memory_tree_similarity_backend, "dinov2_patch_topk")
        self.assertEqual(args.memory_tree_dinov2_patch_top_k, 5)
        self.assertEqual(args.memory_tree_dinov2_patch_max_patches, 64)
        self.assertFalse(args.memory_tree_allow_same_bridge_item)
        self.assertEqual(args.memory_tree_bridge_similarity_tie_margin, 0.01)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(
            _method_id(args, args.passage_policy),
            "retrieval_localize_plan_detected_contrastive_memory_tree_direction",
        )
        self.assertEqual(_direction_scoring_label(args), "dinov2_patch_topk_passage_memory_tree")

        salad_args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--target-room-id",
                "Room 17",
                "--memory-tree-similarity-backend",
                "salad",
            ]
        )
        self.assertEqual(_direction_scoring_label(salad_args), "salad_passage_memory_tree")


        dinov2_args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--target-room-id",
                "Room 17",
                "--memory-tree-similarity-backend",
                "dinov2_patch_topk",
                "--memory-tree-bridge-selection-mode",
                "bridge_then_continuity",
                "--memory-tree-near-duplicate-threshold",
                "1.1",
            ]
        )
        self.assertEqual(dinov2_args.memory_tree_similarity_backend, "dinov2_patch_topk")
        self.assertEqual(dinov2_args.memory_tree_dinov2_patch_model, "facebook/dinov2-base")
        self.assertEqual(dinov2_args.memory_tree_dinov2_patch_top_k, 5)
        self.assertEqual(dinov2_args.memory_tree_dinov2_patch_max_patches, 64)
        self.assertEqual(dinov2_args.memory_tree_bridge_selection_mode, "bridge_then_continuity")
        self.assertEqual(dinov2_args.memory_tree_near_duplicate_threshold, 1.1)
        self.assertFalse(dinov2_args.memory_tree_allow_same_bridge_item)
        self.assertEqual(dinov2_args.memory_tree_bridge_similarity_tie_margin, 0.01)
        self.assertEqual(_direction_scoring_label(dinov2_args), "dinov2_patch_topk_passage_memory_tree")

    def test_query_mode_does_not_require_manual_target_room(self) -> None:
        args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--query",
                "go to Room 23",
            ]
        )

        self.assertEqual(args.query, "go to Room 23")
        self.assertIsNone(args.target_room_id)
        self.assertEqual(args.waypoint_room_id, [])

    def test_manual_target_mode_still_resolves_without_query_parser(self) -> None:
        args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--target-room-id",
                "Room 17",
                "--waypoint-room-id",
                "Room 23",
            ]
        )

        target_room_id, waypoint_room_ids, parsed_query = _resolve_navigation_goal(
            args,
            room_graph={"Room 17": {}, "Room 23": {}},
        )

        self.assertEqual(target_room_id, "Room 17")
        self.assertEqual(waypoint_room_ids, ["Room 23"])
        self.assertIsNone(parsed_query)

    def test_query_mode_rejects_manual_room_ids(self) -> None:
        args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--query",
                "go to Room 23",
                "--target-room-id",
                "Room 17",
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "cannot be combined"):
            _resolve_navigation_goal(args, room_graph={"Room 17": {}, "Room 23": {}})


class RoleModelClientConfigurationTests(unittest.TestCase):
    def test_passage_and_direction_can_use_different_profiles(self) -> None:
        args = build_parser().parse_args(
            [
                "--start-pano-id",
                "start",
                "--target-room-id",
                "Room 17",
                "--passage-profile",
                "openai",
                "--passage-model",
                "gpt-5.5",
                "--direction-profile",
                "gemini",
                "--direction-model",
                "gemma-4-31b-it",
                "--direction-timeout",
                "300",
            ]
        )
        env = {
            "NAV_OPENAI_KEY": "openai-key",
            "NAV_GEMINI_KEY": "gemini-key",
        }
        with patch.dict(os.environ, env, clear=True):
            passage = _build_role_model_client(args, role="passage")
            direction = _build_role_model_client(args, role="direction")

        self.assertEqual(passage.profile, "openai")
        self.assertEqual(passage.provider, "openai")
        self.assertEqual(passage.model_name, "gpt-5.5")
        self.assertEqual(direction.profile, "gemini")
        self.assertEqual(direction.provider, "gemini")
        self.assertEqual(direction.model_name, "gemma-4-31b-it")
        self.assertEqual(direction.timeout, 300.0)
        self.assertIn("generativelanguage.googleapis.com", direction.api_base)


class NavigationEpisodeOutputBundleTests(unittest.TestCase):
    def test_json_output_path_becomes_directory_and_copies_selected_passage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_image = root / "chosen_passage.png"
            source_image.write_bytes(b"image-bytes")
            payload = {
                "configuration": {},
                "rounds": [
                    {
                        "round_index": 0,
                        "localization": {"predicted_room_id": "Room 8"},
                        "subgoal_room_id": "Room 23",
                        "image_goal": {
                            "label": "R8P7",
                            "image_path": str(source_image),
                            "pano_id": "goal-pano",
                            "capture_index": 7,
                        },
                        "current_room_passages": [
                            {
                                "label": "R8P7",
                                "room_id": "Room 8",
                                "pano_id": "goal-pano",
                                "capture_index": 7,
                                "capture_label": "west_to_north",
                                "image_path": str(source_image),
                            }
                        ],
                    }
                ],
            }

            bundle = _prepare_navigation_output_bundle(payload, root / "episode.json")
            self.assertIsNotNone(bundle)
            assert bundle is not None
            payload["output_bundle"] = bundle
            _write_navigation_output_bundle(render_json(payload), bundle)

            output_dir = root / "episode"
            copied_image = output_dir / "selected_passages" / (
                "round_00_Room_8_to_Room_23_R8P7.png"
            )
            self.assertEqual(Path(bundle["output_dir"]), output_dir)
            self.assertEqual(Path(bundle["episode_json_path"]), output_dir / "episode.json")
            self.assertTrue((output_dir / "episode.json").exists())
            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertEqual(copied_image.read_bytes(), b"image-bytes")
            self.assertEqual(bundle["selected_passages"][0]["copy_status"], "copied")
            self.assertEqual(bundle["selected_passages"][0]["passage"]["pano_id"], "goal-pano")

            episode_payload = json.loads((output_dir / "episode.json").read_text())
            self.assertEqual(episode_payload["output_bundle"]["output_dir"], str(output_dir))

    def test_output_directory_helpers_accept_directory_or_json_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(_navigation_output_dir(root / "episode.json"), root / "episode")
            self.assertEqual(_navigation_output_dir(root / "episode"), root / "episode")


if __name__ == "__main__":
    unittest.main()
