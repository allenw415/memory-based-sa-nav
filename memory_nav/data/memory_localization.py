from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence

DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"
SIGLIP2_MODEL_ALIASES = {
    "siglip2": DEFAULT_SIGLIP2_MODEL,
    "siglip2-base": DEFAULT_SIGLIP2_MODEL,
    "siglip2-so400m": "google/siglip2-so400m-patch14-384",
}


class MissingDependencyError(RuntimeError):
    pass


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload: {path}")
    return payload


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_siglip2_model_name(model_name: str | None) -> str:
    normalized = (model_name or "siglip2").strip().lower()
    return SIGLIP2_MODEL_ALIASES.get(normalized, model_name or DEFAULT_SIGLIP2_MODEL)


def parse_csv_argument(value: str | None) -> list[str]:
    if not isinstance(value, str):
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def is_valid_room_id(room_id: object) -> bool:
    return isinstance(room_id, str) and bool(room_id.strip()) and room_id.strip().lower() != "null"


def load_manifest_captures(manifest_path: str | Path, *, max_captures: int | None = None) -> tuple[dict, list[dict]]:
    manifest = load_json(manifest_path)
    captures = [
        capture
        for capture in manifest.get("captures", [])
        if isinstance(capture, dict)
        and isinstance(capture.get("path"), str)
        and capture.get("path")
    ]
    if max_captures is not None and max_captures > 0:
        captures = captures[:max_captures]
    if not captures:
        raise RuntimeError(f"Manifest has no image captures: {manifest_path}")
    return manifest, captures


def select_memory_items(
    *,
    grounding_payload: dict,
    pano_graph: dict[str, dict],
    floor: str | None = None,
    include_sources: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[dict]:
    mappings = grounding_payload.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError("Expected grounding payload to include a 'mappings' object.")
    sources = grounding_payload.get("sources")
    if not isinstance(sources, dict):
        sources = {}

    normalized_floor = str(floor) if floor is not None else None
    allowed_sources = set(str(value) for value in include_sources or [] if isinstance(value, str) and value)

    items: list[dict] = []
    for pano_id in sorted(mappings.keys()):
        room_id = mappings.get(pano_id)
        if not is_valid_room_id(room_id):
            continue
        pano_record = pano_graph.get(pano_id)
        if not isinstance(pano_record, dict):
            continue
        pano_floor = str(pano_record.get("floor")) if pano_record.get("floor") is not None else None
        if normalized_floor is not None and pano_floor != normalized_floor:
            continue
        source = sources.get(pano_id)
        if allowed_sources and source not in allowed_sources:
            continue
        items.append(
            {
                "pano_id": pano_id,
                "room_id": room_id,
                "floor": pano_floor,
                "source": source,
            }
        )
        if limit is not None and len(items) >= max(limit, 0):
            break
    return items


def group_metadata_items_by_pano(metadata_items: Sequence[dict]) -> list[dict]:
    grouped: dict[str, dict] = {}
    for item in metadata_items:
        pano_id = item.get("pano_id")
        room_id = item.get("room_id")
        if not isinstance(pano_id, str) or not pano_id:
            continue
        if not is_valid_room_id(room_id):
            continue
        group = grouped.setdefault(
            pano_id,
            {
                "pano_id": pano_id,
                "room_id": room_id,
                "floor": item.get("floor"),
                "source": item.get("source"),
                "captures": [],
            },
        )
        group["captures"].append(item)
    groups = list(grouped.values())
    for group in groups:
        group["captures"].sort(
            key=lambda record: (
                int(record.get("capture_index", 0)),
                str(record.get("capture_label") or ""),
                str(record.get("capture_path") or ""),
            )
        )
    groups.sort(key=lambda record: record["pano_id"])
    return groups


def select_query_capture_records(
    capture_records: Sequence[dict],
    *,
    query_view_count: int | None,
    selection: str = "evenly-spaced",
    seed: int | None = None,
) -> list[dict]:
    captures = list(capture_records)
    if not captures:
        return []
    count = len(captures)
    requested = count if query_view_count is None or query_view_count <= 0 else min(int(query_view_count), count)
    if requested >= count:
        return captures

    mode = (selection or "evenly-spaced").strip().lower()
    if mode == "first":
        return captures[:requested]
    if mode == "random":
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(count), requested))
        return [captures[index] for index in indices]

    if requested == 1:
        return [captures[count // 2]]
    indices = sorted({round(step * (count - 1) / (requested - 1)) for step in range(requested)})
    if len(indices) < requested:
        for index in range(count):
            if index not in indices:
                indices.append(index)
            if len(indices) >= requested:
                break
        indices = sorted(indices[:requested])
    return [captures[index] for index in indices]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right):
        left_float = float(left_value)
        right_float = float(right_value)
        numerator += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return numerator / math.sqrt(left_norm * right_norm)


def aggregate_room_scores(scored_candidates: Sequence[dict]) -> dict[str, float]:
    room_scores: dict[str, float] = {}
    for candidate in scored_candidates:
        room_id = candidate.get("room_id")
        score = candidate.get("score")
        if not is_valid_room_id(room_id):
            continue
        if not isinstance(score, (int, float)):
            continue
        room_scores[room_id] = room_scores.get(room_id, 0.0) + max(float(score), 0.0)
    return {
        room_id: room_scores[room_id]
        for room_id, _ in sorted(room_scores.items(), key=lambda item: (-item[1], item[0]))
    }


def predict_room_from_candidates(scored_candidates: Sequence[dict]) -> tuple[str | None, float, dict[str, float]]:
    room_scores = aggregate_room_scores(scored_candidates)
    if not room_scores:
        return None, 0.0, {}
    predicted_room_id = next(iter(room_scores.keys()))
    total_score = sum(max(score, 0.0) for score in room_scores.values())
    confidence = (room_scores[predicted_room_id] / total_score) if total_score > 0.0 else 0.0
    return predicted_room_id, confidence, room_scores


def deduplicate_candidates_by_pano(scored_candidates: Sequence[dict]) -> list[dict]:
    best_by_pano: dict[str, dict] = {}
    ordered_fallback: list[dict] = []
    for candidate in scored_candidates:
        pano_id = candidate.get("candidate_pano_id")
        if not isinstance(pano_id, str) or not pano_id:
            ordered_fallback.append(candidate)
            continue
        existing = best_by_pano.get(pano_id)
        score = float(candidate.get("score", 0.0))
        if existing is None or score > float(existing.get("score", 0.0)):
            best_by_pano[pano_id] = candidate
    deduped = list(best_by_pano.values()) + ordered_fallback
    deduped.sort(
        key=lambda record: (
            -float(record.get("score", 0.0)),
            str(record.get("candidate_pano_id", "")),
            int(record.get("candidate_capture_index", 0)),
        )
    )
    return deduped


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Missing numpy. Install numpy, pillow, torch, transformers, and faiss-cpu to build the memory index."
        ) from exc
    return np


def _require_pil_image():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Missing Pillow. Install numpy, pillow, torch, transformers, and faiss-cpu to build the memory index."
        ) from exc
    return Image


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Missing torch. Install numpy, pillow, torch, transformers, and faiss-cpu to build the memory index."
        ) from exc
    return torch


def _require_transformers():
    try:
        from transformers import AutoModel, AutoProcessor
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Missing transformers. Install numpy, pillow, torch, transformers, and faiss-cpu to build the memory index."
        ) from exc
    return AutoModel, AutoProcessor


def require_faiss():
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Missing faiss. Install faiss-cpu (or faiss-gpu) alongside numpy, pillow, torch, and transformers."
        ) from exc
    return faiss


def normalize_rows(matrix):
    np = _require_numpy()
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def save_image_index_artifacts(*, index_path: str | Path, image_embeddings) -> None:
    np = _require_numpy()
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        index_path,
        image_embeddings=image_embeddings.astype(np.float32),
    )


def load_image_index_artifacts(index_path: str | Path):
    np = _require_numpy()
    payload = np.load(Path(index_path))
    return payload["image_embeddings"]


def build_faiss_index(image_embeddings):
    faiss = require_faiss()
    if image_embeddings.ndim != 2:
        raise ValueError("Expected image embeddings with shape [N, D].")
    normalized = normalize_rows(image_embeddings.astype(_require_numpy().float32))
    index = faiss.IndexFlatIP(int(normalized.shape[1]))
    index.add(normalized)
    return index


def save_faiss_index(index, path: str | Path) -> None:
    faiss = require_faiss()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path: str | Path):
    faiss = require_faiss()
    return faiss.read_index(str(Path(path)))


def search_image_index(index, query_embedding, *, top_k: int):
    np = _require_numpy()
    query = normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
    scores, indices = index.search(query, int(top_k))
    return scores[0], indices[0]


def brute_force_search(image_embeddings, query_embedding, *, top_k: int, exclude_indices: set[int] | None = None) -> list[tuple[int, float]]:
    np = _require_numpy()
    normalized_memory = normalize_rows(image_embeddings.astype(np.float32))
    normalized_query = normalize_rows(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))[0]
    scores = normalized_memory @ normalized_query
    blocked = exclude_indices or set()
    ranked = sorted(
        (
            (index, float(score))
            for index, score in enumerate(scores.tolist())
            if index not in blocked
        ),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[: max(top_k, 0)]


def _extract_image_embedding_tensor(model_output):
    if hasattr(model_output, "pooler_output"):
        return model_output.pooler_output
    if hasattr(model_output, "image_embeds"):
        return model_output.image_embeds
    if hasattr(model_output, "last_hidden_state"):
        hidden_state = model_output.last_hidden_state
        if hidden_state is not None:
            return hidden_state[:, 0]
    if isinstance(model_output, (tuple, list)) and model_output:
        first = model_output[0]
        if first is not None:
            return first
    return model_output


class SigLIP2Embedder:
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "auto",
        batch_size: int = 8,
    ):
        self.np = _require_numpy()
        self.Image = _require_pil_image()
        self.torch = _require_torch()
        AutoModel, AutoProcessor = _require_transformers()

        self.model_name = resolve_siglip2_model_name(model_name)
        self.batch_size = max(int(batch_size), 1)
        self.device = self._resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).eval()
        self.model.to(self.device)

    def _resolve_device(self, requested_device: str) -> str:
        normalized = (requested_device or "auto").strip().lower()
        if normalized != "auto":
            return normalized
        if self.torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(self.torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"
        return "cpu"

    def encode_image_paths(self, image_paths: Sequence[str | Path]):
        batches = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start : start + self.batch_size]
            images = [self.Image.open(Path(path)).convert("RGB") for path in batch_paths]
            try:
                inputs = self.processor(images=images, return_tensors="pt")
                if hasattr(inputs, "to"):
                    inputs = inputs.to(self.device)
                else:
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                image_inputs = {
                    key: value
                    for key, value in dict(inputs).items()
                    if str(key).startswith("pixel_") or key == "interpolate_pos_encoding"
                }
                if "pixel_values" not in image_inputs:
                    raise RuntimeError(
                        f"Processor for {self.model_name} did not produce pixel_values. Keys={sorted(dict(inputs).keys())}"
                    )
                with self.torch.inference_mode():
                    model_output = self.model.get_image_features(**image_inputs)
                embeddings = _extract_image_embedding_tensor(model_output)
                if hasattr(embeddings, "ndim") and int(embeddings.ndim) == 3:
                    embeddings = embeddings.mean(dim=1)
                if not hasattr(embeddings, "norm"):
                    raise TypeError(
                        f"Expected tensor-like image embeddings from {self.model_name}, got {type(model_output).__name__}."
                    )
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                batches.append(embeddings.detach().cpu().to(self.torch.float32).numpy())
            finally:
                for image in images:
                    image.close()
        if not batches:
            return self.np.zeros((0, 0), dtype=self.np.float32)
        return self.np.concatenate(batches, axis=0).astype(self.np.float32)

    def encode_texts(self, texts: Sequence[str]):
        batches = []
        cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
        for start in range(0, len(cleaned_texts), self.batch_size):
            batch_texts = cleaned_texts[start : start + self.batch_size]
            inputs = self.processor(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
            text_inputs = {
                key: value
                for key, value in dict(inputs).items()
                if key in {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
            }
            if "input_ids" not in text_inputs:
                raise RuntimeError(
                    f"Processor for {self.model_name} did not produce input_ids. Keys={sorted(dict(inputs).keys())}"
                )
            if not hasattr(self.model, "get_text_features"):
                raise RuntimeError(f"Model {self.model_name} does not expose get_text_features.")
            with self.torch.inference_mode():
                model_output = self.model.get_text_features(**text_inputs)
            embeddings = _extract_image_embedding_tensor(model_output)
            if hasattr(embeddings, "ndim") and int(embeddings.ndim) == 3:
                embeddings = embeddings.mean(dim=1)
            if not hasattr(embeddings, "norm"):
                raise TypeError(
                    f"Expected tensor-like text embeddings from {self.model_name}, got {type(model_output).__name__}."
                )
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            batches.append(embeddings.detach().cpu().to(self.torch.float32).numpy())
        if not batches:
            return self.np.zeros((0, 0), dtype=self.np.float32)
        return self.np.concatenate(batches, axis=0).astype(self.np.float32)
