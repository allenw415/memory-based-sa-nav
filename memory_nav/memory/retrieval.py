from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from memory_nav.data.memory_localization import (
    DEFAULT_SIGLIP2_MODEL,
    MissingDependencyError,
    SigLIP2Embedder,
    brute_force_search,
    deduplicate_candidates_by_pano,
    is_valid_room_id,
    load_faiss_index,
    load_image_index_artifacts,
    load_json,
    predict_room_from_candidates,
    resolve_siglip2_model_name,
    search_image_index,
)


@dataclass
class MemoryMatch:
    candidate_index: int
    candidate_pano_id: str
    candidate_capture_index: int
    candidate_capture_label: str | None
    candidate_capture_path: str | None
    room_id: str
    score: float
    matched_query_image_index: int | None = None
    matched_query_image_path: str | None = None
    image_available: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = {
            "candidate_index": self.candidate_index,
            "candidate_pano_id": self.candidate_pano_id,
            "candidate_capture_index": self.candidate_capture_index,
            "candidate_capture_label": self.candidate_capture_label,
            "candidate_capture_path": self.candidate_capture_path,
            "room_id": self.room_id,
            "score": float(self.score),
            "matched_query_image_index": self.matched_query_image_index,
            "matched_query_image_path": self.matched_query_image_path,
            "image_available": bool(self.image_available),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass
class MemoryLocalizationResult:
    predicted_room_id: str | None
    confidence: float
    margin: float
    is_confident: bool
    room_scores: dict[str, float] = field(default_factory=dict)
    room_distribution: dict[str, float] = field(default_factory=dict)
    top_rooms: list[str] = field(default_factory=list)
    top_matches: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "predicted_room_id": self.predicted_room_id,
            "confidence": float(self.confidence),
            "margin": float(self.margin),
            "is_confident": bool(self.is_confident),
            "room_scores": dict(self.room_scores),
            "room_distribution": dict(self.room_distribution),
            "top_rooms": list(self.top_rooms),
            "top_matches": list(self.top_matches),
        }


class MemoryImageRetriever:
    def __init__(
        self,
        *,
        index_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        faiss_path: str | Path | None = None,
        embedding_model: str = DEFAULT_SIGLIP2_MODEL,
        device: str = "auto",
        batch_size: int = 8,
        use_faiss: bool = True,
        project_root: str | Path | None = None,
        metadata_items: Sequence[dict] | None = None,
        image_embeddings=None,
        image_index=None,
        embedder: SigLIP2Embedder | None = None,
    ):
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.index_path = Path(index_path).resolve() if index_path is not None else None
        self.metadata_path = Path(metadata_path).resolve() if metadata_path is not None else None
        self.faiss_path = Path(faiss_path).resolve() if faiss_path is not None else None
        self.embedding_model = resolve_siglip2_model_name(embedding_model)
        self.device = device
        self.batch_size = int(batch_size)
        self.use_faiss = bool(use_faiss)
        self.embedder = embedder
        self.image_embeddings = image_embeddings
        self.image_index = image_index

        if metadata_items is not None:
            self.metadata_items = [dict(item) for item in metadata_items]
        elif self.metadata_path is not None:
            payload = load_json(self.metadata_path)
            raw_items = payload.get("items")
            if not isinstance(raw_items, list) or not raw_items:
                raise RuntimeError(f"Metadata file does not contain indexed images: {self.metadata_path}")
            self.metadata_items = [dict(item) for item in raw_items if isinstance(item, dict)]
        else:
            self.metadata_items = []

        for index, item in enumerate(self.metadata_items):
            item.setdefault("memory_index", index)

        if self.image_embeddings is None and self.index_path is not None:
            self.image_embeddings = load_image_index_artifacts(self.index_path)
        if (
            self.image_index is None
            and self.use_faiss
            and self.faiss_path is not None
            and self.faiss_path.exists()
        ):
            self.image_index = load_faiss_index(self.faiss_path)

    def query_image_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        top_k: int = 10,
        exclude_same_pano_ids: Iterable[str] | None = None,
    ) -> list[dict]:
        if not image_paths:
            return []
        embedder = self._ensure_embedder()
        query_embeddings = embedder.encode_image_paths(image_paths)
        return self.query_embeddings(
            query_embeddings,
            query_image_paths=[str(path) for path in image_paths],
            top_k=top_k,
            exclude_same_pano_ids=set(exclude_same_pano_ids or []),
        )

    def query_embeddings(
        self,
        query_embeddings,
        *,
        query_image_paths: Sequence[str] | None = None,
        top_k: int = 10,
        exclude_same_pano_ids: set[str] | None = None,
    ) -> list[dict]:
        if self.image_embeddings is None and self.image_index is None:
            raise RuntimeError("MemoryImageRetriever requires image embeddings or a FAISS index.")
        if not self.metadata_items:
            return []

        exclude_same_pano_ids = set(exclude_same_pano_ids or set())
        excluded_indices = {
            int(item.get("memory_index", index))
            for index, item in enumerate(self.metadata_items)
            if item.get("pano_id") in exclude_same_pano_ids
        }
        requested_top_k = max(int(top_k), 1)
        raw_top_k = min(len(self.metadata_items), requested_top_k + len(excluded_indices))
        best_by_index: dict[int, MemoryMatch] = {}

        for query_index, query_embedding in enumerate(query_embeddings):
            ranked = self._rank_embedding(query_embedding, top_k=max(raw_top_k, requested_top_k), excluded_indices=excluded_indices)
            for candidate_index, score in ranked[:requested_top_k]:
                if candidate_index < 0 or candidate_index >= len(self.metadata_items):
                    continue
                candidate_meta = self.metadata_items[candidate_index]
                room_id = candidate_meta.get("room_id")
                if not is_valid_room_id(room_id):
                    continue
                resolved_path = self.resolve_capture_path(candidate_meta)
                match = MemoryMatch(
                    candidate_index=int(candidate_index),
                    candidate_pano_id=str(candidate_meta.get("pano_id", "")),
                    candidate_capture_index=int(candidate_meta.get("capture_index", 0)),
                    candidate_capture_label=candidate_meta.get("capture_label")
                    if isinstance(candidate_meta.get("capture_label"), str)
                    else None,
                    candidate_capture_path=resolved_path,
                    room_id=str(room_id),
                    score=max(float(score), 0.0),
                    matched_query_image_index=query_index,
                    matched_query_image_path=query_image_paths[query_index]
                    if query_image_paths is not None and query_index < len(query_image_paths)
                    else None,
                    image_available=bool(resolved_path and Path(resolved_path).exists()),
                    metadata=dict(candidate_meta),
                )
                existing = best_by_index.get(candidate_index)
                if existing is None or match.score > existing.score:
                    best_by_index[candidate_index] = match

        matches = sorted(
            best_by_index.values(),
            key=lambda item: (-item.score, item.candidate_pano_id, item.candidate_capture_index),
        )
        return [match.to_dict() for match in matches]

    def retrieve_room_memories(
        self,
        room_id: str,
        *,
        top_k: int = 4,
        require_existing_images: bool = False,
    ) -> list[dict]:
        memories: list[dict] = []
        seen_panos: set[str] = set()
        for item in self.metadata_items:
            if item.get("room_id") != room_id:
                continue
            pano_id = str(item.get("pano_id", ""))
            if pano_id in seen_panos:
                continue
            resolved_path = self.resolve_capture_path(item)
            image_available = bool(resolved_path and Path(resolved_path).exists())
            if require_existing_images and not image_available:
                continue
            seen_panos.add(pano_id)
            memories.append(
                {
                    "memory_index": int(item.get("memory_index", len(memories))),
                    "pano_id": pano_id,
                    "room_id": room_id,
                    "capture_label": item.get("capture_label"),
                    "capture_heading": item.get("capture_heading"),
                    "capture_path": resolved_path,
                    "original_capture_path": item.get("capture_path"),
                    "image_available": image_available,
                }
            )
            if len(memories) >= max(int(top_k), 0):
                break
        return memories


    def retrieve_room_memories_for_text_queries(
        self,
        room_id: str,
        text_queries: Sequence[str],
        *,
        top_k_per_query: int = 2,
        max_memories: int = 4,
        require_existing_images: bool = False,
    ) -> list[dict]:
        queries = [str(query).strip() for query in text_queries if str(query).strip()]
        if not queries or self.image_embeddings is None:
            return self.retrieve_room_memories(
                room_id,
                top_k=max_memories,
                require_existing_images=require_existing_images,
            )
        embedder = self._ensure_embedder()
        if not hasattr(embedder, "encode_texts"):
            return self.retrieve_room_memories(
                room_id,
                top_k=max_memories,
                require_existing_images=require_existing_images,
            )
        text_embeddings = embedder.encode_texts(queries)
        memories = self.retrieve_room_memories_for_query_embeddings(
            room_id,
            text_embeddings,
            top_k_per_query=top_k_per_query,
            max_memories=max_memories,
            require_existing_images=require_existing_images,
        )
        for memory in memories:
            memory["retrieval_mode"] = "target_room_semantic_text"
            memory["semantic_query_count"] = len(queries)
        return memories

    def retrieve_room_memories_for_query_images(
        self,
        room_id: str,
        query_image_paths: Sequence[str | Path],
        *,
        passage_labels: Sequence[str] | None = None,
        top_k_per_query: int = 2,
        max_memories: int = 4,
        require_existing_images: bool = False,
    ) -> list[dict]:
        """Retrieve target-room memory images that visually match passage photos.

        The room graph decides which room is the subgoal. This method searches
        only that room's memory embeddings, so the VLM receives visually relevant
        evidence without letting retrieval invent an invalid topology step.
        """
        if not query_image_paths or self.image_embeddings is None:
            return self.retrieve_room_memories(
                room_id,
                top_k=max_memories,
                require_existing_images=require_existing_images,
            )
        embedder = self._ensure_embedder()
        query_embeddings = embedder.encode_image_paths(query_image_paths)
        return self.retrieve_room_memories_for_query_embeddings(
            room_id,
            query_embeddings,
            query_image_paths=[str(path) for path in query_image_paths],
            passage_labels=passage_labels,
            top_k_per_query=top_k_per_query,
            max_memories=max_memories,
            require_existing_images=require_existing_images,
        )

    def retrieve_room_memories_for_query_embeddings(
        self,
        room_id: str,
        query_embeddings,
        *,
        query_image_paths: Sequence[str] | None = None,
        passage_labels: Sequence[str] | None = None,
        top_k_per_query: int = 2,
        max_memories: int = 4,
        require_existing_images: bool = False,
    ) -> list[dict]:
        if self.image_embeddings is None:
            return self.retrieve_room_memories(
                room_id,
                top_k=max_memories,
                require_existing_images=require_existing_images,
            )
        room_indices = [
            index
            for index, item in enumerate(self.metadata_items)
            if item.get("room_id") == room_id
        ]
        if not room_indices:
            return []

        top_k_per_query = max(int(top_k_per_query), 1)
        max_memories = max(int(max_memories), 0)
        if max_memories <= 0:
            return []

        room_embeddings = self.image_embeddings[room_indices]
        best_by_memory_index: dict[int, dict] = {}
        local_top_k = min(len(room_indices), max(top_k_per_query, max_memories))

        for query_index, query_embedding in enumerate(query_embeddings):
            ranked = brute_force_search(room_embeddings, query_embedding, top_k=local_top_k)
            for local_index, score in ranked[:top_k_per_query]:
                memory_index = int(room_indices[int(local_index)])
                item = self.metadata_items[memory_index]
                resolved_path = self.resolve_capture_path(item)
                image_available = bool(resolved_path and Path(resolved_path).exists())
                if require_existing_images and not image_available:
                    continue
                passage_label = (
                    passage_labels[query_index]
                    if passage_labels is not None and query_index < len(passage_labels)
                    else None
                )
                matched_query_path = (
                    query_image_paths[query_index]
                    if query_image_paths is not None and query_index < len(query_image_paths)
                    else None
                )
                memory = {
                    "memory_index": memory_index,
                    "pano_id": str(item.get("pano_id", "")),
                    "room_id": room_id,
                    "capture_label": item.get("capture_label"),
                    "capture_heading": item.get("capture_heading"),
                    "capture_path": resolved_path,
                    "original_capture_path": item.get("capture_path"),
                    "image_available": image_available,
                    "score": float(score),
                    "matched_passage_label": passage_label,
                    "matched_passage_image_path": matched_query_path,
                    "retrieval_mode": "target_room_vector",
                }
                existing = best_by_memory_index.get(memory_index)
                if existing is None or float(memory["score"]) > float(existing.get("score", 0.0)):
                    best_by_memory_index[memory_index] = memory

        memories = sorted(
            best_by_memory_index.values(),
            key=lambda item: (
                -float(item.get("score", 0.0)),
                str(item.get("pano_id", "")),
                str(item.get("capture_label") or ""),
                int(item.get("memory_index", 0)),
            ),
        )
        return memories[:max_memories]

    def resolve_capture_path(self, metadata_item: dict) -> str | None:
        raw_path = metadata_item.get("capture_path")
        if not isinstance(raw_path, str) or not raw_path:
            return None
        path = Path(raw_path)
        if path.exists():
            return str(path.resolve())
        if not path.is_absolute():
            candidate = (self.project_root / path).resolve()
            if candidate.exists():
                return str(candidate)
        marker = "renders/room_grounding/"
        if marker in raw_path:
            suffix = raw_path.split(marker, 1)[1]
            candidate = (self.project_root / marker / suffix).resolve()
            if candidate.exists():
                return str(candidate)
            return str(candidate)
        return str(path)

    def _rank_embedding(self, query_embedding, *, top_k: int, excluded_indices: set[int]) -> list[tuple[int, float]]:
        if self.image_index is not None:
            scores, indices = search_image_index(self.image_index, query_embedding, top_k=top_k)
            return [
                (int(candidate_index), float(score))
                for score, candidate_index in zip(scores.tolist(), indices.tolist())
                if int(candidate_index) >= 0 and int(candidate_index) not in excluded_indices
            ]
        return brute_force_search(
            self.image_embeddings,
            query_embedding,
            top_k=top_k,
            exclude_indices=excluded_indices,
        )

    def _ensure_embedder(self) -> SigLIP2Embedder:
        if self.embedder is None:
            try:
                self.embedder = SigLIP2Embedder(
                    model_name=self.embedding_model,
                    device=self.device,
                    batch_size=self.batch_size,
                )
            except MissingDependencyError:
                raise
        return self.embedder


class MemoryRoomLocalizer:
    def __init__(
        self,
        retriever: MemoryImageRetriever | None = None,
        *,
        retrieval_top_k: int = 10,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.15,
        dedup_by_pano: bool = True,
    ):
        self.retriever = retriever
        self.retrieval_top_k = int(retrieval_top_k)
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.dedup_by_pano = bool(dedup_by_pano)

    def localize_from_images(self, image_paths: Sequence[str | Path]) -> MemoryLocalizationResult:
        if self.retriever is None:
            raise RuntimeError("MemoryRoomLocalizer requires a MemoryImageRetriever for image localization.")
        matches = self.retriever.query_image_paths(image_paths, top_k=self.retrieval_top_k)
        return self.localize_from_matches(matches)

    def localize_from_matches(self, matches: Sequence[dict]) -> MemoryLocalizationResult:
        scored_matches = [dict(match) for match in matches if isinstance(match, dict)]
        if self.dedup_by_pano:
            scored_matches = deduplicate_candidates_by_pano(scored_matches)
        predicted_room_id, confidence, room_scores = predict_room_from_candidates(scored_matches)
        room_distribution = self._normalize_distribution(room_scores)
        top_rooms = list(room_scores.keys())
        margin = self._distribution_margin(room_distribution)
        is_confident = bool(
            predicted_room_id
            and confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
        )
        return MemoryLocalizationResult(
            predicted_room_id=predicted_room_id,
            confidence=float(confidence),
            margin=float(margin),
            is_confident=is_confident,
            room_scores=room_scores,
            room_distribution=room_distribution,
            top_rooms=top_rooms,
            top_matches=scored_matches[: self.retrieval_top_k],
        )

    @staticmethod
    def _normalize_distribution(room_scores: dict[str, float]) -> dict[str, float]:
        total = sum(max(float(score), 0.0) for score in room_scores.values())
        if total <= 0.0:
            return {}
        return {
            room_id: max(float(score), 0.0) / total
            for room_id, score in room_scores.items()
        }

    @staticmethod
    def _distribution_margin(room_distribution: dict[str, float]) -> float:
        values = sorted((float(value) for value in room_distribution.values()), reverse=True)
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        return max(0.0, values[0] - values[1])
