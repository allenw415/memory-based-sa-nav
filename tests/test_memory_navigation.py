from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from memory_nav import (
    InteractiveMemoryNavigator,
    MemoryImageRetriever,
    MemoryLocalizationResult,
    MemoryRoomLocalizer,
    PassageAlignmentAdvisor,
)
from memory_nav.data.normalize import normalize_room_graph


class MemoryNavigationTests(unittest.TestCase):
    def test_memory_room_localizer_aggregates_matches(self) -> None:
        localizer = MemoryRoomLocalizer(retrieval_top_k=4, confidence_threshold=0.55, margin_threshold=0.15)

        result = localizer.localize_from_matches(
            [
                {"candidate_pano_id": "pano-8a", "room_id": "Room 8", "score": 0.9},
                {"candidate_pano_id": "pano-8b", "room_id": "Room 8", "score": 0.7},
                {"candidate_pano_id": "pano-9", "room_id": "Room 9", "score": 0.2},
            ]
        )

        self.assertEqual(result.predicted_room_id, "Room 8")
        self.assertTrue(result.is_confident)
        self.assertGreater(result.room_distribution["Room 8"], result.room_distribution["Room 9"])
        self.assertEqual(result.top_rooms[0], "Room 8")

    def test_memory_retriever_selects_target_room_memories_by_passage_vector(self) -> None:
        if np is None:
            self.skipTest("numpy is required for embedding similarity tests")
        retriever = MemoryImageRetriever(
            metadata_items=[
                {"pano_id": "room23-a", "room_id": "Room 23", "capture_index": 0, "capture_label": "north"},
                {"pano_id": "room23-b", "room_id": "Room 23", "capture_index": 1, "capture_label": "east"},
                {"pano_id": "room9-a", "room_id": "Room 9", "capture_index": 0, "capture_label": "front"},
            ],
            image_embeddings=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            use_faiss=False,
        )

        memories = retriever.retrieve_room_memories_for_query_embeddings(
            "Room 23",
            np.asarray([[0.0, 1.0]], dtype=np.float32),
            query_image_paths=["front.jpg"],
            passage_labels=["front"],
            top_k_per_query=2,
            max_memories=2,
        )

        self.assertEqual(memories[0]["room_id"], "Room 23")
        self.assertEqual(memories[0]["pano_id"], "room23-b")
        self.assertEqual(memories[0]["matched_passage_label"], "front")
        self.assertEqual(memories[0]["retrieval_mode"], "target_room_vector")
        self.assertGreater(memories[0]["score"], 0.99)
        self.assertNotIn("Room 9", {memory["room_id"] for memory in memories})

    def test_memory_retriever_selects_target_room_memories_by_text_query(self) -> None:
        if np is None:
            self.skipTest("numpy is required for embedding similarity tests")

        class FakeTextEmbedder:
            def encode_texts(self, texts):
                return np.asarray([[0.0, 1.0] for _ in texts], dtype=np.float32)

        retriever = MemoryImageRetriever(
            metadata_items=[
                {"pano_id": "room23-a", "room_id": "Room 23", "capture_index": 0, "capture_label": "north"},
                {"pano_id": "room23-b", "room_id": "Room 23", "capture_index": 1, "capture_label": "east"},
                {"pano_id": "room9-a", "room_id": "Room 9", "capture_index": 0, "capture_label": "front"},
            ],
            image_embeddings=np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            embedder=FakeTextEmbedder(),
            use_faiss=False,
        )

        memories = retriever.retrieve_room_memories_for_text_queries(
            "Room 23",
            ["arched doorway marble sculpture corridor"],
            top_k_per_query=2,
            max_memories=2,
        )

        self.assertEqual(memories[0]["room_id"], "Room 23")
        self.assertEqual(memories[0]["pano_id"], "room23-b")
        self.assertEqual(memories[0]["retrieval_mode"], "target_room_semantic_text")
        self.assertGreater(memories[0]["score"], 0.99)
        self.assertNotIn("Room 9", {memory["room_id"] for memory in memories})

    def test_interactive_memory_navigator_advances_waypoints(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room 8": {
                    "name": "Room 8",
                    "Level": 0,
                    "links": [{"direction": "up", "name": "Room 9"}],
                },
                "Room 9": {
                    "name": "Room 9",
                    "Level": 0,
                    "links": [
                        {"direction": "down", "name": "Room 8"},
                        {"direction": "left", "name": "Room 17"},
                    ],
                },
                "Room 17": {
                    "name": "Room 17",
                    "Level": 0,
                    "links": [
                        {"direction": "right", "name": "Room 9"},
                        {"direction": "left", "name": "Room 23"},
                    ],
                },
                "Room 23": {
                    "name": "Room 23",
                    "Level": 0,
                    "links": [{"direction": "right", "name": "Room 17"}],
                },
            }
        )

        class FakeLocalizer:
            def __init__(self) -> None:
                self.rooms = ["Room 8", "Room 9"]

            def localize_from_images(self, _image_paths):
                room_id = self.rooms.pop(0)
                return MemoryLocalizationResult(
                    predicted_room_id=room_id,
                    confidence=0.9,
                    margin=0.8,
                    is_confident=True,
                    room_scores={room_id: 1.0},
                    room_distribution={room_id: 1.0},
                    top_rooms=[room_id],
                    top_matches=[],
                )

        class FakeAdvisor:
            def advise(self, **kwargs):
                return {
                    "chosen_passage_label": "front",
                    "target_room_id": kwargs["next_room_id"],
                    "direction_hint": "往前方通道走",
                    "confidence": 0.8,
                    "evidence": ["mock"],
                    "rationale_zh": "mock",
                    "message_zh": "請往前方通道走。",
                }

        navigator = InteractiveMemoryNavigator(
            room_graph=room_graph,
            localizer=FakeLocalizer(),
            passage_advisor=FakeAdvisor(),
        )

        first = navigator.guide(
            target_room_id="Room 23",
            waypoint_room_ids=["Room 9", "Room 17"],
            localization_images=["current.jpg"],
        )
        second = navigator.guide(
            target_room_id="Room 23",
            waypoint_room_ids=["Room 9", "Room 17"],
            localization_images=["current.jpg"],
        )

        self.assertEqual(first["action_request"], "capture_passage_views")
        self.assertEqual(first["active_target_room_id"], "Room 9")
        self.assertEqual(first["next_room_id"], "Room 9")
        self.assertEqual(second["active_target_room_id"], "Room 17")
        self.assertEqual(second["next_room_id"], "Room 17")

    def test_passage_alignment_advisor_uses_semantic_target_memories_without_prompt_leak(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room 8": {
                    "name": "Room 8",
                    "Level": 0,
                    "links": [{"direction": "left", "name": "Room 23"}],
                },
                "Room 23": {
                    "name": "Room 23",
                    "Level": 0,
                    "links": [{"direction": "right", "name": "Room 8"}],
                },
            }
        )

        class FakeRetriever:
            def __init__(self) -> None:
                self.calls = []

            def retrieve_room_memories_for_text_queries(
                self,
                room_id,
                text_queries,
                *,
                top_k_per_query=2,
                max_memories=4,
                require_existing_images=False,
            ):
                self.calls.append(
                    {
                        "room_id": room_id,
                        "text_queries": list(text_queries),
                        "top_k_per_query": top_k_per_query,
                        "max_memories": max_memories,
                    }
                )
                return [
                    {
                        "room_id": room_id,
                        "pano_id": "room23-semantic-pano",
                        "capture_label": "east",
                        "capture_heading": 90.0,
                        "capture_path": None,
                        "image_available": False,
                        "score": 0.91,
                        "matched_passage_label": "front",
                        "retrieval_mode": "target_room_semantic_text",
                    }
                ]

            def retrieve_room_memories(self, room_id, *, top_k=4, require_existing_images=False):
                return []

        captured_request = {}

        def fake_response(request_body):
            captured_request["body"] = request_body
            return {
                "output_text": json.dumps(
                    {
                        "chosen_candidate_label": "front",
                        "target_room_id": "Room 23",
                        "direction_hint": "往前方通道走",
                        "confidence": 0.8,
                        "evidence": ["front 的視覺線索最接近 Room 23 target memory"],
                        "rationale_zh": "front 和目標房間記憶的空間線索最一致。",
                        "message_zh": "請往前方通道走，這最可能通往 Room 23。",
                    }
                )
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            front = Path(tmpdir) / "front.png"
            left = Path(tmpdir) / "left.png"
            front.write_bytes(b"fake-front")
            left.write_bytes(b"fake-left")
            retriever = FakeRetriever()
            advisor = PassageAlignmentAdvisor(
                room_graph=room_graph,
                memory_retriever=retriever,
                response_client=fake_response,
                max_memory_images=3,
                passage_semantic_query_provider=lambda _candidates: "arched doorway marble sculpture corridor",
            )
            guidance = advisor.advise(
                current_room_id="Room 8",
                next_room_id="Room 23",
                active_target_room_id="Room 23",
                route=["Room 8", "Room 23"],
                passage_images={"front": front, "left": left},
                localization={"predicted_room_id": "Room 8", "confidence": 0.9},
            )

        self.assertEqual(retriever.calls[0]["room_id"], "Room 23")
        self.assertEqual(retriever.calls[0]["text_queries"], ["arched doorway marble sculpture corridor"])
        prompt_text = "\n".join(
            part["text"]
            for part in captured_request["body"]["input"][0]["content"]
            if part.get("type") == "input_text"
        )
        self.assertNotIn("matched_passage", prompt_text)
        self.assertNotIn("similarity=", prompt_text)
        self.assertNotIn("0.910", prompt_text)
        self.assertEqual(guidance["chosen_passage_label"], "front")
        self.assertEqual(guidance["alignment_mode"], "vlm_with_semantic_memory")

    def test_passage_alignment_advisor_returns_chinese_guidance(self) -> None:
        room_graph = normalize_room_graph(
            {
                "Room 8": {
                    "name": "Room 8",
                    "Level": 0,
                    "category": "Middle East",
                    "title": "Assyria: Nimrud",
                    "links": [{"direction": "left", "name": "Room 23"}],
                },
                "Room 23": {
                    "name": "Room 23",
                    "Level": 0,
                    "category": "Ancient Greece and Rome",
                    "title": "Greek and Roman sculpture",
                    "links": [{"direction": "right", "name": "Room 8"}],
                },
            }
        )

        class FakeRetriever:
            def retrieve_room_memories(self, room_id, *, top_k=4, require_existing_images=False):
                return [
                    {
                        "room_id": room_id,
                        "pano_id": "memory-pano",
                        "capture_label": "west",
                        "capture_heading": 240.0,
                        "capture_path": None,
                        "image_available": False,
                    }
                ]

        responses = [
            {
                "output_text": json.dumps(
                    {
                        "chosen_candidate_label": "front",
                        "target_room_id": "Room 23",
                        "direction_hint": "往前方通道走",
                        "confidence": 0.78,
                        "evidence": ["front 通道可見希臘羅馬雕塑線索"],
                        "rationale_zh": "front 通道最符合 Room 23 的視覺記憶。",
                        "message_zh": "你目前應該在 Room 8。請往前方通道走，這最可能通往 Room 23。",
                    }
                )
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            front = Path(tmpdir) / "front.png"
            left = Path(tmpdir) / "left.png"
            front.write_bytes(b"fake-front")
            left.write_bytes(b"fake-left")
            advisor = PassageAlignmentAdvisor(
                room_graph=room_graph,
                memory_retriever=FakeRetriever(),
                response_client=lambda _: responses.pop(0),
            )
            guidance = advisor.advise(
                current_room_id="Room 8",
                next_room_id="Room 23",
                active_target_room_id="Room 23",
                route=["Room 8", "Room 23"],
                passage_images={"front": front, "left": left},
                localization={"predicted_room_id": "Room 8", "confidence": 0.9},
            )

        self.assertEqual(guidance["chosen_passage_label"], "front")
        self.assertEqual(guidance["target_room_id"], "Room 23")
        self.assertIn("請往前方通道走", guidance["message_zh"])
        self.assertGreater(guidance["confidence"], 0.7)


if __name__ == "__main__":
    unittest.main()
