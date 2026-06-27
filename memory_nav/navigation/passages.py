from __future__ import annotations

import hashlib
import json
import mimetypes
import random
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


class SimilarityPassageSelector:
    """Choose a current-room passage by comparing it to sampled room views."""

    requires_subgoal_passages = False
    CONTRASTIVE_SCORING_MODES = {
        "contrastive_neighbor_mean",
        "contrastive_neighbor_room_max_mean",
        "contrastive_neighbor_room_average_mean",
    }
    PER_ROOM_CONTRASTIVE_SCORING_MODES = {
        "contrastive_neighbor_room_max_mean",
        "contrastive_neighbor_room_average_mean",
    }

    def __init__(
        self,
        *,
        visual_retriever,
        target_sample_count: int = 64,
        negative_sample_count: int | None = None,
        top_m: int = 5,
        seed: int = 0,
        similarity_backend: str = "salad",
        image_embedder=None,
        target_scoring: str = "sample_mean",
        room_graph: dict | None = None,
        contrastive_negative_weight: float = 1.0,
    ):
        if similarity_backend not in {"salad", "dreamsim"}:
            raise ValueError("Similarity backend must be 'salad' or 'dreamsim'.")
        if target_scoring not in {
            "sample_mean",
            "contrastive_neighbor_mean",
            "contrastive_neighbor_room_max_mean",
            "contrastive_neighbor_room_average_mean",
        }:
            raise ValueError(
                "Target scoring must be 'sample_mean', 'contrastive_neighbor_mean', "
                "'contrastive_neighbor_room_max_mean', or "
                "'contrastive_neighbor_room_average_mean'."
            )
        self.visual_retriever = visual_retriever
        self.target_sample_count = max(int(target_sample_count), 1)
        self.negative_sample_count = (
            self.target_sample_count
            if negative_sample_count is None
            else max(int(negative_sample_count), 1)
        )
        self.top_m = max(int(top_m), 1)
        self.seed = int(seed)
        self.similarity_backend = similarity_backend
        self.image_embedder = image_embedder
        self.target_scoring = target_scoring
        self.room_graph = room_graph or {}
        self.contrastive_negative_weight = float(contrastive_negative_weight)
        self.metadata_items = [dict(item) for item in getattr(visual_retriever, "metadata_items", [])]
        self.image_embeddings = getattr(visual_retriever, "image_embeddings", None)
        if self.similarity_backend == "salad" and self.image_embeddings is None:
            raise RuntimeError("Similarity passage selection requires image embeddings.")
        if self.similarity_backend == "dreamsim" and not hasattr(image_embedder, "encode_image_paths"):
            raise RuntimeError("DreamSim passage selection requires an image embedder.")
        if not self.metadata_items:
            raise RuntimeError("Similarity passage selection requires image metadata.")
        if self.image_embeddings is not None and len(self.metadata_items) != int(self.image_embeddings.shape[0]):
            raise RuntimeError("Similarity image embeddings and metadata counts do not match.")

        self._index_by_capture: dict[tuple[str, int], int] = {}
        for index, item in enumerate(self.metadata_items):
            pano_id = item.get("pano_id")
            capture_index = item.get("capture_index")
            if isinstance(pano_id, str) and isinstance(capture_index, int):
                self._index_by_capture[(pano_id, capture_index)] = index

    def choose(
        self,
        *,
        current_room_id: str,
        subgoal_room_id: str,
        current_candidates: Sequence[dict],
        subgoal_candidates: Sequence[dict],
    ) -> dict:
        del subgoal_candidates
        current_entries = self._current_candidate_entries(current_candidates)
        target_samples = self._sample_target_room(subgoal_room_id)
        if not current_entries or not target_samples:
            return {
                "chosen_label": None,
                "navigation_confidence": 0.0,
                "why_this_passage": "No comparable current passages or target-room samples.",
                "selector_source": "similarity",
                "similarity_backend": self.similarity_backend,
                "target_scoring": self.target_scoring,
                "request_summary": {
                    "current_room_id": current_room_id,
                    "subgoal_room_id": subgoal_room_id,
                    "current_labels": [str(item.get("label")) for item in current_candidates],
                    "target_sample_count": len(target_samples),
                    "negative_sample_count": 0,
                    "negative_room_ids": [],
                },
                "passage_ranking": [],
                "target_visual_clues": [_drop_embedding(sample) for sample in target_samples],
                "negative_visual_clues": [],
            }

        negative_room_ids: tuple[str, ...] = ()
        negative_samples: list[dict] = []
        negative_samples_by_room: dict[str, list[dict]] = {}
        if self.target_scoring in self.CONTRASTIVE_SCORING_MODES:
            negative_room_ids = self._negative_neighbor_room_ids(current_room_id, subgoal_room_id)
            if not negative_room_ids:
                raise ValueError(
                    f"No non-target neighbor rooms found for {current_room_id} excluding {subgoal_room_id}."
                )
            if self.target_scoring in self.PER_ROOM_CONTRASTIVE_SCORING_MODES:
                negative_samples_by_room = self._sample_negative_rooms_by_room(negative_room_ids)
                negative_samples = [
                    sample
                    for room_id in negative_room_ids
                    for sample in negative_samples_by_room.get(room_id, [])
                ]
            else:
                negative_samples = self._sample_negative_rooms(negative_room_ids)
                negative_samples_by_room = {"__mixed__": negative_samples}
            if not negative_samples:
                raise ValueError(
                    f"No negative-room visual samples found for neighbors: {', '.join(negative_room_ids)}."
                )

        if self.similarity_backend == "dreamsim":
            self._attach_live_image_embeddings(current_entries, target_samples, negative_samples)

        import numpy as np

        target_embeddings = normalize_rows(
            np.asarray([sample["embedding"] for sample in target_samples], dtype=np.float32)
        )
        target_public = [_drop_embedding(sample) for sample in target_samples]
        negative_public = [_drop_embedding(sample) for sample in negative_samples]
        negative_public_by_room = {
            room_id: [_drop_embedding(sample) for sample in samples]
            for room_id, samples in negative_samples_by_room.items()
        }
        negative_embeddings = None
        if negative_samples:
            negative_embeddings = normalize_rows(
                np.asarray([sample["embedding"] for sample in negative_samples], dtype=np.float32)
            )
        negative_embeddings_by_room = {
            room_id: normalize_rows(
                np.asarray([sample["embedding"] for sample in samples], dtype=np.float32)
            )
            for room_id, samples in negative_samples_by_room.items()
            if samples
        }

        ranking = []
        for entry in current_entries:
            passage_embedding = normalize_rows(
                np.asarray(entry["embedding"], dtype=np.float32).reshape(1, -1)
            )[0]
            target_scores = target_embeddings @ passage_embedding
            target_values = [float(score) for score in target_scores.tolist()]
            target_mean = sum(target_values) / max(len(target_values), 1)
            scored_targets = sorted(
                ((float(score), target_public[index]) for index, score in enumerate(target_values)),
                key=lambda item: (
                    -item[0],
                    str(item[1].get("pano_id", "")),
                    int(item[1].get("capture_index", 0)),
                ),
            )
            top_targets = scored_targets[: min(self.top_m, len(scored_targets))]
            top_mean = sum(score for score, _ in top_targets) / max(len(top_targets), 1)
            max_score = scored_targets[0][0] if scored_targets else 0.0

            if self.target_scoring in self.PER_ROOM_CONTRASTIVE_SCORING_MODES:
                negative_result = self._score_per_room_negatives(
                    passage_embedding=passage_embedding,
                    negative_embeddings_by_room=negative_embeddings_by_room,
                    negative_public_by_room=negative_public_by_room,
                )
            else:
                negative_result = self._score_mixed_negatives(
                    passage_embedding=passage_embedding,
                    negative_embeddings=negative_embeddings,
                    negative_public=negative_public,
                )

            primary_negative = float(negative_result["primary_negative_similarity"])
            contrastive_score = target_mean - self.contrastive_negative_weight * primary_negative
            selection_score = (
                contrastive_score
                if self.target_scoring in self.CONTRASTIVE_SCORING_MODES
                else target_mean
            )
            ranking.append(
                {
                    "label": entry["label"],
                    "room_id": entry["room_id"],
                    "pano_id": entry["pano_id"],
                    "capture_index": entry["capture_index"],
                    "capture_label": entry.get("capture_label"),
                    "capture_heading": entry.get("capture_heading"),
                    "image_path": entry.get("image_path"),
                    "semantic_score": entry.get("semantic_score"),
                    "selection_score": float(selection_score),
                    "mean_similarity": float(target_mean),
                    "target_mean_similarity": float(target_mean),
                    "negative_mean_similarity": primary_negative,
                    "per_room_average_negative_similarity": float(
                        negative_result["per_room_average_negative_similarity"]
                    ),
                    "hard_negative_similarity": float(negative_result["hard_negative_similarity"]),
                    "hard_negative_room_id": negative_result["hard_negative_room_id"],
                    "negative_room_scores": list(negative_result["negative_room_scores"]),
                    "contrastive_similarity": float(contrastive_score),
                    "contrastive_negative_weight": self.contrastive_negative_weight,
                    "top_m_mean_similarity": float(top_mean),
                    "max_similarity": float(max_score),
                    "negative_room_ids": list(negative_room_ids),
                    "matched_target_samples": [
                        {
                            "rank": index,
                            "similarity": float(score),
                            **dict(sample),
                        }
                        for index, (score, sample) in enumerate(top_targets, start=1)
                    ],
                    "matched_negative_samples": list(negative_result["matched_negative_samples"]),
                }
            )

        if self.target_scoring in self.CONTRASTIVE_SCORING_MODES:
            ranking.sort(
                key=lambda item: (
                    -float(item["selection_score"]),
                    -float(item["target_mean_similarity"]),
                    -float(item["top_m_mean_similarity"]),
                    -float(item["max_similarity"]),
                    str(item["label"]),
                )
            )
        else:
            ranking.sort(
                key=lambda item: (
                    -float(item["mean_similarity"]),
                    -float(item["top_m_mean_similarity"]),
                    -float(item["max_similarity"]),
                    str(item["label"]),
                )
            )
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
            item["selected"] = rank == 1

        score_field = (
            "selection_score"
            if self.target_scoring in self.CONTRASTIVE_SCORING_MODES
            else "mean_similarity"
        )
        score_aggregation = self._score_aggregation_label()
        chosen = ranking[0]
        return {
            "chosen_label": chosen["label"],
            "navigation_confidence": max(0.0, min(1.0, float(chosen[score_field]))),
            "why_this_passage": self._why_this_passage(len(target_samples), len(negative_samples)),
            "selector_source": "similarity",
            "similarity_backend": self.similarity_backend,
            "target_scoring": self.target_scoring,
            "score_field": score_field,
            "request_summary": {
                "current_room_id": current_room_id,
                "subgoal_room_id": subgoal_room_id,
                "current_labels": [str(item.get("label")) for item in current_candidates],
                "target_sample_count": len(target_samples),
                "target_sample_limit": self.target_sample_count,
                "negative_sample_count": len(negative_samples),
                "negative_sample_count_by_room": {
                    room_id: len(samples)
                    for room_id, samples in negative_samples_by_room.items()
                    if room_id != "__mixed__"
                },
                "negative_sample_limit": self.negative_sample_count,
                "negative_room_ids": list(negative_room_ids),
                "contrastive_negative_weight": self.contrastive_negative_weight,
                "score_aggregation": score_aggregation,
                "top_m": self.top_m,
                "seed": self.seed,
            },
            "target_sampling": {
                "mode": "pano_balanced_sample",
                "room_id": subgoal_room_id,
                "sample_count": len(target_samples),
                "sample_limit": self.target_sample_count,
                "seed": self.seed,
            },
            "negative_sampling": {
                "mode": (
                    "neighbor_per_room_pano_balanced_sample"
                    if self.target_scoring in self.PER_ROOM_CONTRASTIVE_SCORING_MODES
                    else "neighbor_pano_balanced_sample"
                ),
                "room_ids": list(negative_room_ids),
                "sample_count": len(negative_samples),
                "sample_count_by_room": {
                    room_id: len(samples)
                    for room_id, samples in negative_samples_by_room.items()
                    if room_id != "__mixed__"
                },
                "sample_limit": self.negative_sample_count,
                "seed": self.seed,
            },
            "target_visual_clues": target_public,
            "negative_visual_clues": negative_public,
            "passage_ranking": ranking,
        }

    def _score_mixed_negatives(
        self,
        *,
        passage_embedding,
        negative_embeddings,
        negative_public: Sequence[dict],
    ) -> dict:
        if negative_embeddings is None or not negative_public:
            return {
                "primary_negative_similarity": 0.0,
                "per_room_average_negative_similarity": 0.0,
                "hard_negative_similarity": 0.0,
                "hard_negative_room_id": None,
                "negative_room_scores": [],
                "matched_negative_samples": [],
            }
        negative_scores = negative_embeddings @ passage_embedding
        negative_values = [float(score) for score in negative_scores.tolist()]
        negative_mean = sum(negative_values) / max(len(negative_values), 1)
        scored_negatives = sorted(
            ((float(score), negative_public[index]) for index, score in enumerate(negative_values)),
            key=lambda item: (
                -item[0],
                str(item[1].get("room_id", "")),
                str(item[1].get("pano_id", "")),
                int(item[1].get("capture_index", 0)),
            ),
        )
        matched_negative_samples = [
            {
                "rank": index,
                "similarity": float(score),
                **dict(sample),
            }
            for index, (score, sample) in enumerate(
                scored_negatives[: min(self.top_m, len(scored_negatives))],
                start=1,
            )
        ]
        return {
            "primary_negative_similarity": float(negative_mean),
            "per_room_average_negative_similarity": float(negative_mean),
            "hard_negative_similarity": float(negative_mean),
            "hard_negative_room_id": None,
            "negative_room_scores": [],
            "matched_negative_samples": matched_negative_samples,
        }

    def _score_per_room_negatives(
        self,
        *,
        passage_embedding,
        negative_embeddings_by_room: dict[str, object],
        negative_public_by_room: dict[str, list[dict]],
    ) -> dict:
        room_scores = []
        for room_id in sorted(negative_embeddings_by_room):
            room_embeddings = negative_embeddings_by_room[room_id]
            room_public = negative_public_by_room.get(room_id, [])
            if room_embeddings is None or not room_public:
                continue
            scores = room_embeddings @ passage_embedding
            values = [float(score) for score in scores.tolist()]
            scored_samples = sorted(
                ((float(score), room_public[index]) for index, score in enumerate(values)),
                key=lambda item: (
                    -item[0],
                    str(item[1].get("pano_id", "")),
                    int(item[1].get("capture_index", 0)),
                ),
            )
            top_samples = scored_samples[: min(self.top_m, len(scored_samples))]
            room_scores.append(
                {
                    "room_id": room_id,
                    "sample_count": len(room_public),
                    "mean_similarity": float(sum(values) / max(len(values), 1)),
                    "top_m_mean_similarity": float(
                        sum(score for score, _ in top_samples) / max(len(top_samples), 1)
                    ),
                    "max_similarity": float(scored_samples[0][0] if scored_samples else 0.0),
                    "matched_negative_samples": [
                        {
                            "rank": index,
                            "similarity": float(score),
                            **dict(sample),
                        }
                        for index, (score, sample) in enumerate(top_samples, start=1)
                    ],
                }
            )
        if not room_scores:
            return {
                "primary_negative_similarity": 0.0,
                "per_room_average_negative_similarity": 0.0,
                "hard_negative_similarity": 0.0,
                "hard_negative_room_id": None,
                "negative_room_scores": [],
                "matched_negative_samples": [],
            }
        sorted_room_scores = sorted(
            room_scores,
            key=lambda item: (
                -float(item["mean_similarity"]),
                -float(item["top_m_mean_similarity"]),
                -float(item["max_similarity"]),
                str(item["room_id"]),
            ),
        )
        hard_negative = sorted_room_scores[0]
        average_negative = sum(float(item["mean_similarity"]) for item in sorted_room_scores) / max(
            len(sorted_room_scores),
            1,
        )
        primary_negative = (
            average_negative
            if self.target_scoring == "contrastive_neighbor_room_average_mean"
            else float(hard_negative["mean_similarity"])
        )
        return {
            "primary_negative_similarity": float(primary_negative),
            "per_room_average_negative_similarity": float(average_negative),
            "hard_negative_similarity": float(hard_negative["mean_similarity"]),
            "hard_negative_room_id": hard_negative["room_id"],
            "negative_room_scores": sorted_room_scores,
            "matched_negative_samples": list(hard_negative["matched_negative_samples"]),
        }

    def _score_aggregation_label(self) -> str:
        if self.target_scoring == "contrastive_neighbor_room_average_mean":
            return "mean_sampled_target_similarities_minus_average_neighbor_room_mean_similarity"
        if self.target_scoring == "contrastive_neighbor_room_max_mean":
            return "mean_sampled_target_similarities_minus_max_neighbor_room_mean_similarity"
        if self.target_scoring == "contrastive_neighbor_mean":
            return "mean_sampled_target_similarities_minus_neighbor_mean_similarity"
        return "mean_all_sampled_target_similarities"

    def _why_this_passage(self, target_count: int, negative_count: int) -> str:
        if self.target_scoring == "contrastive_neighbor_room_average_mean":
            return (
                f"Highest per-room average-negative {self.similarity_backend.upper()} score across "
                f"{target_count} target-room views and {negative_count} non-target neighbor views."
            )
        if self.target_scoring == "contrastive_neighbor_room_max_mean":
            return (
                f"Highest per-room hard-negative {self.similarity_backend.upper()} score across "
                f"{target_count} target-room views and {negative_count} non-target neighbor views."
            )
        if self.target_scoring == "contrastive_neighbor_mean":
            return (
                f"Highest contrastive {self.similarity_backend.upper()} score across "
                f"{target_count} target-room views and {negative_count} non-target neighbor views."
            )
        return (
            f"Highest mean {self.similarity_backend.upper()} similarity across "
            f"{target_count} sampled target-room views."
        )

    def _current_candidate_entries(self, current_candidates: Sequence[dict]) -> list[dict]:
        entries = []
        for candidate in current_candidates:
            row_index = self._row_index_for_candidate(candidate)
            if row_index is None:
                continue
            entry = dict(candidate)
            entry["label"] = str(candidate.get("label"))
            if self.similarity_backend == "salad":
                entry["embedding"] = self.image_embeddings[row_index]
            entries.append(entry)
        return entries

    def _attach_live_image_embeddings(self, *record_groups: Sequence[dict]) -> None:
        import numpy as np

        records = [record for group in record_groups for record in group]
        image_paths = []
        for record in records:
            image_path = record.get("image_path")
            if not isinstance(image_path, (str, Path)) or not image_path:
                raise RuntimeError("DreamSim passage selection requires image paths.")
            path = Path(image_path)
            if not path.exists():
                raise RuntimeError(f"DreamSim image path does not exist: {image_path}")
            image_paths.append(path)
        embeddings = np.asarray(
            self.image_embedder.encode_image_paths(image_paths),
            dtype=np.float32,
        )
        if embeddings.shape[0] != len(records):
            raise RuntimeError("DreamSim image embedder returned the wrong number of embeddings.")
        for record, embedding in zip(records, embeddings, strict=True):
            record["embedding"] = embedding

    def _row_index_for_candidate(self, candidate: dict) -> int | None:
        pano_id = candidate.get("pano_id")
        capture_index = candidate.get("capture_index")
        if isinstance(pano_id, str) and isinstance(capture_index, int):
            row_index = self._index_by_capture.get((pano_id, capture_index))
            if row_index is not None:
                return row_index
        memory_index = candidate.get("memory_index")
        if isinstance(memory_index, int) and 0 <= memory_index < len(self.metadata_items):
            item = self.metadata_items[memory_index]
            if item.get("pano_id") == pano_id and item.get("capture_index") == capture_index:
                return memory_index
        return None

    def _negative_neighbor_room_ids(self, current_room_id: str, subgoal_room_id: str) -> tuple[str, ...]:
        node = self.room_graph.get(current_room_id) if isinstance(self.room_graph, dict) else None
        neighbors = node.get("neighbors") if isinstance(node, dict) else None
        if not isinstance(neighbors, list):
            return ()
        excluded = {current_room_id, subgoal_room_id}
        room_ids = []
        for neighbor in neighbors:
            if not isinstance(neighbor, dict):
                continue
            target_room_id = neighbor.get("target_room_id")
            if not isinstance(target_room_id, str) or target_room_id in excluded:
                continue
            if target_room_id not in room_ids:
                room_ids.append(target_room_id)
        return tuple(sorted(room_ids))

    def _sample_target_room(self, room_id: str) -> list[dict]:
        return self._sample_rooms([room_id], sample_limit=self.target_sample_count, seed_salt=room_id)

    def _sample_negative_rooms(self, room_ids: Sequence[str]) -> list[dict]:
        return self._sample_rooms(
            room_ids,
            sample_limit=self.negative_sample_count,
            seed_salt="negative:" + "|".join(room_ids),
        )

    def _sample_negative_rooms_by_room(self, room_ids: Sequence[str]) -> dict[str, list[dict]]:
        return {
            room_id: self._sample_rooms(
                [room_id],
                sample_limit=self.negative_sample_count,
                seed_salt="negative-room:" + room_id,
            )
            for room_id in room_ids
        }

    def _sample_rooms(self, room_ids: Sequence[str], *, sample_limit: int, seed_salt: str) -> list[dict]:
        room_id_set = {str(room_id) for room_id in room_ids}
        samples = []
        for row_index, item in enumerate(self.metadata_items):
            if item.get("room_id") not in room_id_set:
                continue
            sample = self._sample_from_metadata(row_index, item)
            if sample is not None:
                samples.append(sample)
        return self._pano_balanced_sample(samples, sample_limit=sample_limit, seed_salt=seed_salt)

    def _pano_balanced_sample(
        self,
        samples: Sequence[dict],
        *,
        sample_limit: int,
        seed_salt: str,
    ) -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        for sample in samples:
            pano_id = str(sample.get("pano_id", ""))
            grouped.setdefault(pano_id, []).append(sample)
        if not grouped:
            return []

        rng = random.Random(_stable_seed(self.seed, seed_salt))
        pano_ids = sorted(grouped)
        rng.shuffle(pano_ids)
        for pano_id in pano_ids:
            grouped[pano_id].sort(
                key=lambda sample: (
                    int(sample.get("capture_index", 0)),
                    int(sample.get("memory_index", 0)),
                )
            )
            rng.shuffle(grouped[pano_id])

        selected = []
        active = list(pano_ids)
        while active and len(selected) < max(int(sample_limit), 1):
            next_active = []
            for pano_id in active:
                if len(selected) >= max(int(sample_limit), 1):
                    break
                group = grouped[pano_id]
                if not group:
                    continue
                selected.append(group.pop(0))
                if group:
                    next_active.append(pano_id)
            active = next_active
        return selected

    def _sample_from_metadata(self, row_index: int, item: dict) -> dict | None:
        pano_id = item.get("pano_id")
        capture_index = item.get("capture_index")
        if not isinstance(pano_id, str) or not isinstance(capture_index, int):
            return None
        image_path = None
        if hasattr(self.visual_retriever, "resolve_capture_path"):
            image_path = self.visual_retriever.resolve_capture_path(item)
        if not image_path:
            raw_path = item.get("capture_path")
            image_path = str(raw_path) if isinstance(raw_path, str) else None
        if image_path and not Path(image_path).exists():
            return None
        return {
            "memory_index": int(item.get("memory_index", row_index)),
            "room_id": item.get("room_id"),
            "pano_id": pano_id,
            "capture_index": capture_index,
            "capture_label": item.get("capture_label"),
            "capture_heading": item.get("capture_heading"),
            "image_path": image_path,
            **(
                {"embedding": self.image_embeddings[row_index]}
                if self.similarity_backend == "salad"
                else {}
            ),
        }


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
        cluster_candidates: bool = True,
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
        self.cluster_candidates = bool(cluster_candidates)
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

        if not self.cluster_candidates:
            prefix = _room_label_prefix(room_id)
            for index, candidate in enumerate(candidates, start=1):
                candidate["cluster_size"] = 1
                candidate["cluster_member_memory_indices"] = [int(candidate["memory_index"])]
                candidate["cluster_id"] = index
                candidate["label"] = f"{prefix}{index}"
                candidate["retrieval_query"] = self.query
            return candidates

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


def _drop_embedding(item: dict) -> dict:
    return {key: value for key, value in item.items() if key != "embedding"}


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


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
