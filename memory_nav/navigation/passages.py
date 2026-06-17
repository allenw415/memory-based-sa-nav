from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import Protocol, Sequence

from memory_nav.common.model_client import ModelResponseClient, parse_json_output
from memory_nav.data.memory_localization import (
    DEFAULT_SIGLIP2_MODEL,
    brute_force_search,
    create_image_embedder,
    load_image_index_artifacts,
    load_json,
    normalize_rows,
)


DEFAULT_PASSAGE_QUERY = (
    "a walkable doorway, entrance, or opening connecting two museum galleries"
)


class PassageSelector(Protocol):
    def choose(
        self,
        *,
        current_room_id: str,
        subgoal_room_id: str,
        current_candidates: Sequence[dict],
        subgoal_candidates: Sequence[dict],
    ) -> dict: ...


class DynamicPassageRetriever:
    """Retrieve and cluster room passage images without reading prior result JSON."""

    def __init__(
        self,
        *,
        semantic_index_path: str | Path | None = None,
        semantic_metadata_path: str | Path | None = None,
        visual_index_path: str | Path | None = None,
        visual_metadata_path: str | Path | None = None,
        render_root: str | Path = "renders/room_grounding_fov90",
        query: str = DEFAULT_PASSAGE_QUERY,
        retrieval_top_k: int = 20,
        target_clusters: int = 8,
        embedding_model: str = DEFAULT_SIGLIP2_MODEL,
        device: str = "auto",
        batch_size: int = 8,
        text_embedder=None,
        semantic_embeddings=None,
        semantic_metadata_items: Sequence[dict] | None = None,
        visual_embeddings=None,
        visual_metadata_items: Sequence[dict] | None = None,
    ):
        self.semantic_index_path = (
            Path(semantic_index_path).resolve() if semantic_index_path is not None else None
        )
        self.semantic_metadata_path = (
            Path(semantic_metadata_path).resolve() if semantic_metadata_path is not None else None
        )
        self.visual_index_path = (
            Path(visual_index_path).resolve() if visual_index_path is not None else None
        )
        self.visual_metadata_path = (
            Path(visual_metadata_path).resolve() if visual_metadata_path is not None else None
        )
        self.render_root = Path(render_root).resolve()
        self.query = str(query)
        self.retrieval_top_k = max(int(retrieval_top_k), 1)
        self.target_clusters = max(int(target_clusters), 1)
        self.embedding_model = embedding_model
        self.device = device
        self.batch_size = max(int(batch_size), 1)
        self.text_embedder = text_embedder

        self.semantic_embeddings = (
            semantic_embeddings
            if semantic_embeddings is not None
            else load_image_index_artifacts(self._required_path(self.semantic_index_path, "semantic index"))
        )
        self.visual_embeddings = (
            visual_embeddings
            if visual_embeddings is not None
            else load_image_index_artifacts(self._required_path(self.visual_index_path, "visual index"))
        )
        self.semantic_metadata_items = self._load_metadata(
            semantic_metadata_items,
            self.semantic_metadata_path,
            "semantic metadata",
        )
        self.visual_metadata_items = self._load_metadata(
            visual_metadata_items,
            self.visual_metadata_path,
            "visual metadata",
        )
        if len(self.semantic_metadata_items) != int(self.semantic_embeddings.shape[0]):
            raise RuntimeError("Semantic index and metadata counts do not match.")
        if len(self.visual_metadata_items) != int(self.visual_embeddings.shape[0]):
            raise RuntimeError("Visual index and metadata counts do not match.")

        self._visual_index_by_capture: dict[tuple[str, int], int] = {}
        for index, item in enumerate(self.visual_metadata_items):
            pano_id = item.get("pano_id")
            capture_index = item.get("capture_index")
            if isinstance(pano_id, str) and isinstance(capture_index, int):
                self._visual_index_by_capture[(pano_id, capture_index)] = index

    @staticmethod
    def _required_path(path: Path | None, name: str) -> Path:
        if path is None:
            raise ValueError(f"Missing {name} path.")
        return path

    @staticmethod
    def _load_metadata(
        supplied: Sequence[dict] | None,
        path: Path | None,
        name: str,
    ) -> list[dict]:
        if supplied is not None:
            return [dict(item) for item in supplied]
        if path is None:
            raise ValueError(f"Missing {name} path.")
        payload = load_json(path)
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"{name.title()} contains no items: {path}")
        return [dict(item) for item in items if isinstance(item, dict)]

    def retrieve(self, room_id: str) -> list[dict]:
        embedder = self._ensure_text_embedder()
        if not hasattr(embedder, "encode_texts"):
            raise RuntimeError("Passage retrieval requires a text-capable embedding model.")
        query_embeddings = embedder.encode_texts([self.query])
        if len(query_embeddings) != 1:
            raise RuntimeError("Passage text encoder did not return one query embedding.")
        return self.retrieve_with_query_embedding(room_id, query_embeddings[0])

    def retrieve_with_query_embedding(self, room_id: str, query_embedding) -> list[dict]:
        room_indices = [
            index
            for index, item in enumerate(self.semantic_metadata_items)
            if item.get("room_id") == room_id
        ]
        if not room_indices:
            return []

        room_embeddings = self.semantic_embeddings[room_indices]
        ranked = brute_force_search(
            room_embeddings,
            query_embedding,
            top_k=min(self.retrieval_top_k, len(room_indices)),
        )
        candidates: list[dict] = []
        visual_vectors = []
        for local_index, score in ranked:
            semantic_index = room_indices[int(local_index)]
            item = self.semantic_metadata_items[semantic_index]
            pano_id = item.get("pano_id")
            capture_index = item.get("capture_index")
            if not isinstance(pano_id, str) or not isinstance(capture_index, int):
                continue
            visual_index = self._visual_index_by_capture.get((pano_id, capture_index))
            if visual_index is None:
                continue
            image_path = self._resolve_capture_path(item)
            if image_path is None or not image_path.exists():
                continue
            candidates.append(
                {
                    "memory_index": semantic_index,
                    "semantic_score": float(score),
                    "room_id": room_id,
                    "pano_id": pano_id,
                    "capture_index": capture_index,
                    "capture_label": item.get("capture_label"),
                    "capture_heading": item.get("capture_heading"),
                    "image_path": str(image_path),
                }
            )
            visual_vectors.append(self.visual_embeddings[visual_index])

        if not candidates:
            return []

        import numpy as np

        normalized = normalize_rows(np.asarray(visual_vectors, dtype=np.float32))
        similarity = normalized @ normalized.T
        clusters = _agglomerative_clusters(
            similarity,
            target_clusters=min(self.target_clusters, len(candidates)),
        )
        representatives = []
        for cluster in clusters:
            representative_index = max(
                cluster,
                key=lambda index: (
                    float(candidates[index]["semantic_score"]),
                    -int(candidates[index]["memory_index"]),
                ),
            )
            representative = dict(candidates[representative_index])
            representative["cluster_size"] = len(cluster)
            representative["cluster_member_memory_indices"] = [
                int(candidates[index]["memory_index"]) for index in cluster
            ]
            representatives.append(representative)

        representatives.sort(
            key=lambda item: (
                -float(item["semantic_score"]),
                int(item["memory_index"]),
            )
        )
        prefix = _room_label_prefix(room_id)
        for index, representative in enumerate(representatives, start=1):
            representative["cluster_id"] = index
            representative["label"] = f"{prefix}{index}"
            representative["retrieval_query"] = self.query
        return representatives

    def _ensure_text_embedder(self):
        if self.text_embedder is None:
            self.text_embedder = create_image_embedder(
                model_name=self.embedding_model,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self.text_embedder

    def _resolve_capture_path(self, item: dict) -> Path | None:
        raw_path = item.get("capture_path")
        if isinstance(raw_path, str) and raw_path:
            original = Path(raw_path)
            if original.exists():
                return original.resolve()
            pano_id = item.get("pano_id")
            if isinstance(pano_id, str) and pano_id:
                candidate = self.render_root / pano_id / original.name
                if candidate.exists():
                    return candidate.resolve()
        pano_id = item.get("pano_id")
        capture_index = item.get("capture_index")
        capture_label = item.get("capture_label")
        if isinstance(pano_id, str) and isinstance(capture_index, int):
            label = capture_label if isinstance(capture_label, str) else "*"
            matches = sorted(
                (self.render_root / pano_id).glob(
                    f"{pano_id}_{capture_index:02d}_{label}_*.png"
                )
            )
            if matches:
                return matches[0].resolve()
        return None


class PassageVLMSelector:
    """Choose a current-room passage using images and room IDs only."""

    def __init__(
        self,
        *,
        model_client: ModelResponseClient,
        model: str,
        detail: str = "high",
    ):
        self.model_client = model_client
        self.model = model
        self.detail = detail

    def choose(
        self,
        *,
        current_room_id: str,
        subgoal_room_id: str,
        current_candidates: Sequence[dict],
        subgoal_candidates: Sequence[dict],
    ) -> dict:
        request_body = self.build_request(
            current_room_id=current_room_id,
            subgoal_room_id=subgoal_room_id,
            current_candidates=current_candidates,
            subgoal_candidates=subgoal_candidates,
        )
        parsed = parse_json_output(self.model_client.create(request_body))
        _validate_choice(parsed, current_candidates)
        parsed["selector_source"] = "live_vlm"
        parsed["request_summary"] = {
            "current_room_id": current_room_id,
            "subgoal_room_id": subgoal_room_id,
            "current_labels": [item["label"] for item in current_candidates],
            "subgoal_labels": [item["label"] for item in subgoal_candidates],
        }
        return parsed

    def build_request(
        self,
        *,
        current_room_id: str,
        subgoal_room_id: str,
        current_candidates: Sequence[dict],
        subgoal_candidates: Sequence[dict],
    ) -> dict:
        current_labels = [str(item["label"]) for item in current_candidates]
        subgoal_labels = [str(item["label"]) for item in subgoal_candidates]
        content: list[dict] = [
            {
                "type": "input_text",
                "text": "\n".join(
                    [
                        f"Current room: {current_room_id}",
                        f"Immediate subgoal room: {subgoal_room_id}",
                        f"Choose exactly one current-room label from: {current_labels}",
                        f"Subgoal reference labels (never choose these): {subgoal_labels}",
                        "Classify every current candidate as valid_passage, ambiguous, or noise.",
                        "Select the physical exit most likely to cross directly from the current room into the immediate subgoal room.",
                        "Do not choose a passage merely because the visible scene resembles the subgoal references.",
                        "Subgoal reference images may face away from the entrance, show exits, or include neighboring rooms.",
                        "For each current candidate, identify the walkable opening and judge the scene immediately through that opening.",
                        "Prefer evidence that the opening crosses a room boundary into the subgoal room over general visual similarity.",
                        "If a candidate contains multiple plausible openings and the intended one is unclear, classify it as ambiguous.",
                        "Treat views that remain inside the current room, show only exhibits, or lack a clear opening as noise.",
                        "Use only the images and room IDs supplied here.",
                    ]
                ),
            }
        ]
        for candidate in current_candidates:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Current-room candidate {candidate['label']} ({current_room_id}).",
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(Path(candidate["image_path"])),
                        "detail": self.detail,
                    },
                ]
            )
        for candidate in subgoal_candidates:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": f"Subgoal-room reference {candidate['label']} ({subgoal_room_id}); do not choose it.",
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(Path(candidate["image_path"])),
                        "detail": self.detail,
                    },
                ]
            )
        return {
            "model": self.model,
            "instructions": (
                "You are a careful museum navigation assistant. Judge passages visually. "
                "Do not infer direction from labels. Return strict JSON only."
            ),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "visual_passage_choice",
                    "strict": True,
                    "schema": _choice_schema(current_labels),
                }
            },
        }


class RecordedPassageSelector:
    """Deterministic selector used by tests and recorded offline episodes."""

    def __init__(self, responses: dict):
        self.responses = dict(responses)

    @classmethod
    def from_path(cls, path: str | Path) -> "RecordedPassageSelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Recorded VLM response file must contain an object.")
        responses = payload.get("responses", payload)
        if not isinstance(responses, dict):
            raise ValueError("Recorded VLM responses must contain an object mapping.")
        return cls(responses)

    def choose(
        self,
        *,
        current_room_id: str,
        subgoal_room_id: str,
        current_candidates: Sequence[dict],
        subgoal_candidates: Sequence[dict],
    ) -> dict:
        del subgoal_candidates
        key = f"{current_room_id}->{subgoal_room_id}"
        response = self.responses.get(key)
        if isinstance(response, str):
            response = {
                "chosen_label": response,
                "navigation_confidence": 1.0,
            }
        if not isinstance(response, dict):
            raise KeyError(f"No recorded passage response for {key}")
        parsed = dict(response)
        parsed.setdefault("navigation_confidence", 1.0)
        parsed.setdefault("why_this_passage", "recorded response")
        parsed["selector_source"] = "recorded"
        parsed["request_summary"] = {
            "current_room_id": current_room_id,
            "subgoal_room_id": subgoal_room_id,
            "current_labels": [item["label"] for item in current_candidates],
        }
        _validate_choice(parsed, current_candidates)
        return parsed


def _agglomerative_clusters(similarity, *, target_clusters: int) -> list[list[int]]:
    clusters = [[index] for index in range(int(similarity.shape[0]))]
    while len(clusters) > max(int(target_clusters), 1):
        best_pair: tuple[int, int] | None = None
        best_score = -float("inf")
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                values = [
                    float(similarity[a, b])
                    for a in clusters[left]
                    for b in clusters[right]
                ]
                score = sum(values) / max(len(values), 1)
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    return clusters


def _room_label_prefix(room_id: str) -> str:
    digits = "".join(char for char in str(room_id) if char.isdigit())
    return f"R{digits}P" if digits else "P"


def _image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return f"data:{mime_type or 'image/png'};base64,{b64encode(path.read_bytes()).decode('ascii')}"


def _choice_schema(current_labels: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "candidate_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": current_labels},
                        "status": {
                            "type": "string",
                            "enum": ["valid_passage", "ambiguous", "noise"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "status", "reason"],
                },
            },
            "chosen_label": {
                "anyOf": [
                    {"type": "string", "enum": current_labels},
                    {"type": "null"},
                ]
            },
            "navigation_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "why_this_passage": {"type": "string"},
        },
        "required": [
            "candidate_assessments",
            "chosen_label",
            "navigation_confidence",
            "why_this_passage",
        ],
    }


def _validate_choice(parsed: dict, current_candidates: Sequence[dict]) -> None:
    labels = [str(item["label"]) for item in current_candidates]
    chosen = parsed.get("chosen_label")
    if chosen is not None and chosen not in labels:
        raise RuntimeError(f"Passage selector chose {chosen!r}; expected one of {labels}")
