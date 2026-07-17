from __future__ import annotations

import json
import unittest

from memory_nav.common.model_client import ModelResponseClient
from memory_nav.navigation.query_parser import (
    NavigationQueryParser,
    build_gallery_theme_lines,
    build_navigation_query_parse_schema,
    parse_navigation_query_payload,
)


ROOM_GRAPH = {
    "Room 10": {
        "title": "Assyria: Lion hunts, Siege of Lachish and Khorsabad",
        "visual_profile": {
            "short_description": "Assyria gallery focused on lion hunts and palace reliefs.",
            "visual_cues": ["stone reliefs", "lion hunt scenes"],
        },
    },
    "Room 17": {
        "title": "Nereid Monument",
        "visual_profile": {
            "short_description": "Gallery dominated by a reconstructed Lycian monumental tomb.",
            "visual_cues": ["Nereid figures", "monumental tomb architecture"],
        },
    },
    "Room 23": {
        "title": "Greek and Roman sculpture",
        "visual_profile": {
            "short_description": "Greek and Roman sculpture gallery with marble statues.",
            "visual_cues": ["marble statues", "Roman copies"],
        },
    },
}


class NavigationQueryParserTests(unittest.TestCase):
    def _parser_for(self, payload: dict) -> tuple[NavigationQueryParser, list[dict]]:
        calls: list[dict] = []

        def response_client(request_body: dict) -> dict:
            calls.append(request_body)
            return {"output_text": json.dumps(payload)}

        return (
            NavigationQueryParser(
                model_client=ModelResponseClient(response_client=response_client),
                model="test-model",
                room_graph=ROOM_GRAPH,
            ),
            calls,
        )

    def test_gallery_goal_query_resolves_target_room(self) -> None:
        parser, calls = self._parser_for(
            {
                "goal_entities": [
                    {
                        "name": "Room 23",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 23",
                        "confidence": 1.0,
                    }
                ],
                "waypoint_entities": [],
            }
        )

        parsed = parser.parse("go to Room 23")

        self.assertEqual(parsed.target_room_id, "Room 23")
        self.assertEqual(parsed.waypoint_room_ids, [])
        self.assertEqual(parsed.goal_entities[0].entity_type, "gallery")
        schema = calls[0]["text"]["format"]["schema"]
        self.assertNotIn("source_room_id", schema["properties"])
        self.assertNotIn("source_entity", schema["properties"])

    def test_gemma4_request_uses_minimal_thinking_and_counts_calls(self) -> None:
        parser, _calls = self._parser_for(
            {
                "goal_entities": [
                    {
                        "name": "Room 23",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 23",
                        "confidence": 1.0,
                    }
                ],
                "waypoint_entities": [],
            }
        )
        parser.model = "gemma-4-31b-it"
        request = parser.build_request("Take me to the Room 23 gallery.")
        client = ModelResponseClient(
            provider="gemini",
            api_key="test-key",
            api_base="https://generativelanguage.googleapis.com/v1beta",
        )

        payload = client._responses_to_gemini_generate_content_payload(request)

        self.assertEqual(
            payload["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "minimal",
        )

        parsed = parser.parse("Take me to the Room 23 gallery.")
        self.assertEqual(parsed.target_room_id, "Room 23")
        self.assertEqual(parser.model_client.logical_request_count, 1)
        self.assertEqual(parser.model_client.http_attempt_count, 0)

    def test_waypoints_preserve_room_order_before_goal(self) -> None:
        parser, _calls = self._parser_for(
            {
                "goal_entities": [
                    {
                        "name": "Room 10",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 10",
                        "confidence": 1.0,
                    }
                ],
                "waypoint_entities": [
                    {
                        "name": "Room 23",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 23",
                        "confidence": 1.0,
                    },
                    {
                        "name": "Nereid Monument",
                        "entity_type": "artwork",
                        "predicted_room_id": "Room 17",
                        "confidence": 0.82,
                    },
                ],
            }
        )

        parsed = parser.parse("go to Room 10 passing Room 23 and the Nereid Monument")

        self.assertEqual(parsed.target_room_id, "Room 10")
        self.assertEqual(parsed.waypoint_room_ids, ["Room 23", "Room 17"])

    def test_artwork_query_can_resolve_goal_room(self) -> None:
        parser, calls = self._parser_for(
            {
                "goal_entities": [
                    {
                        "name": "Nereid Monument",
                        "entity_type": "artwork",
                        "predicted_room_id": "Room 17",
                        "confidence": 0.84,
                    }
                ],
                "waypoint_entities": [],
            }
        )

        parsed = parser.parse("take me to the Nereid Monument")

        self.assertEqual(parsed.target_room_id, "Room 17")
        self.assertEqual(parsed.goal_entities[0].name, "Nereid Monument")
        self.assertIn("Nereid Monument", calls[0]["input"])

    def test_theme_described_gallery_can_resolve_goal_room(self) -> None:
        parser, calls = self._parser_for(
            {
                "goal_entities": [
                    {
                        "name": "gallery focused on lion hunts and palace reliefs",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 10",
                        "confidence": 0.91,
                    }
                ],
                "waypoint_entities": [],
            }
        )

        parsed = parser.parse(
            "take me to the gallery focused on lion hunts and palace reliefs"
        )

        self.assertEqual(parsed.target_room_id, "Room 10")
        self.assertEqual(parsed.goal_entities[0].entity_type, "gallery")
        self.assertEqual(parsed.goal_entities[0].confidence, 0.91)
        self.assertIn("thematic gallery description", calls[0]["instructions"])
        self.assertIn("lion hunts", calls[0]["input"])

    def test_source_mentions_are_not_part_of_schema_or_output_contract(self) -> None:
        schema = build_navigation_query_parse_schema(list(ROOM_GRAPH))
        self.assertEqual(set(schema["properties"]), {"goal_entities", "waypoint_entities"})

        parsed = parse_navigation_query_payload(
            {
                "goal_entities": [
                    {
                        "name": "Room 23",
                        "entity_type": "gallery",
                        "predicted_room_id": "Room 23",
                        "confidence": 1.0,
                    }
                ],
                "waypoint_entities": [],
            },
            instruction="from Room 8 go to Room 23",
            room_ids=list(ROOM_GRAPH),
        )

        self.assertEqual(parsed.target_room_id, "Room 23")
        self.assertNotIn("source", parsed.to_dict())

    def test_missing_goal_room_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "goal room"):
            parse_navigation_query_payload(
                {"goal_entities": [], "waypoint_entities": []},
                instruction="walk around",
                room_ids=list(ROOM_GRAPH),
            )

    def test_gallery_theme_lines_use_existing_room_profiles(self) -> None:
        theme_lines = build_gallery_theme_lines(ROOM_GRAPH)

        self.assertIn("Room 17: Nereid Monument", theme_lines)
        self.assertIn("monumental tomb", theme_lines)


if __name__ == "__main__":
    unittest.main()
