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
    _navigation_output_dir,
    _prepare_navigation_output_bundle,
    _write_navigation_output_bundle,
)


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
