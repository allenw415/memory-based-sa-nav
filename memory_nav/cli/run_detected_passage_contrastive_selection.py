from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ._common import (
    PROJECT_ROOT,
    ensure_project_root_on_path,
    load_json,
    render_json,
    resolve_project_path,
)

ensure_project_root_on_path()

from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_DINOV2_SALAD_MODEL,
    DEFAULT_SIGLIP2_MODEL,
    create_image_embedder,
    normalize_rows,
)
from memory_nav.memory.retrieval import MemoryImageRetriever  # noqa: E402
from memory_nav.navigation import DEFAULT_PASSAGE_QUERY, DynamicPassageRetriever  # noqa: E402
from memory_nav.cli.run_similarity_passage_selection import (  # noqa: E402
    DreamSimImageEmbedder,
    _resolve_torch_device,
)


DEFAULT_SIGLIP_INDEX = "artifacts/memory_localization/floor0_siglip2_images_fov90.npz"
DEFAULT_SIGLIP_METADATA = "artifacts/memory_localization/floor0_siglip2_images_fov90.metadata.json"
DEFAULT_SALAD_INDEX = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz"
DEFAULT_SALAD_METADATA = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
DEFAULT_MANIFEST_ROOT = "renders/room_grounding_fov90"
DEFAULT_ROOM_GRAPH_PATH = "dataset/sites/british_museum/normalized/room_graph.json"
DEFAULT_RESULT_FILENAME = "result.json"
DEFAULT_DETECTOR_PROMPT = (
    "doorway . room entrance . gallery entrance . large opening . open doorway . walkable opening ."
)
DEFAULT_GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
DEFAULT_OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"
DEFAULT_MODEL_CACHE_DIR = "models/huggingface"


@dataclass(frozen=True)
class DetectionCandidate:
    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve current-room passage images, split them into detected opening candidates, "
            "and rank the detected candidates with contrastive visual similarity."
        )
    )
    parser.add_argument("--current-room-id", required=True)
    parser.add_argument("--subgoal-room-id", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passage-query", default=DEFAULT_PASSAGE_QUERY)
    parser.add_argument("--passage-top-k", type=int, default=20)
    parser.add_argument("--detector-prompt", default=DEFAULT_DETECTOR_PROMPT)
    parser.add_argument("--detector-backend", choices=["groundingdino", "owlv2"], default="groundingdino")
    parser.add_argument("--detector-model", help="Detector model name. Defaults depend on --detector-backend.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.95)
    parser.add_argument("--crop-padding-ratio", type=float, default=0.12)
    parser.add_argument("--current-image-mode", choices=["mask", "crop"], default="mask")
    parser.add_argument("--mask-background-brightness", type=float, default=0.0)
    parser.add_argument("--max-detections-per-image", type=int, default=3)
    parser.add_argument("--target-sample-count", type=int, default=64)
    parser.add_argument("--negative-sample-count", type=int, default=None)
    parser.add_argument("--similarity-top-m", type=int, default=5)
    parser.add_argument(
        "--target-scoring",
        choices=[
            "contrastive_neighbor_mean",
            "contrastive_neighbor_room_max_mean",
            "contrastive_neighbor_room_average_mean",
        ],
        default="contrastive_neighbor_mean",
    )
    parser.add_argument("--contrastive-negative-weight", type=float, default=1.0)
    parser.add_argument("--room-graph-path", default=DEFAULT_ROOM_GRAPH_PATH)
    parser.add_argument("--similarity-backend", choices=["dreamsim", "salad"], default="dreamsim")
    parser.add_argument("--dreamsim-type", default="ensemble")
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--siglip-index-path", default=DEFAULT_SIGLIP_INDEX)
    parser.add_argument("--siglip-metadata-path", default=DEFAULT_SIGLIP_METADATA)
    parser.add_argument("--salad-index-path", default=DEFAULT_SALAD_INDEX)
    parser.add_argument("--salad-metadata-path", default=DEFAULT_SALAD_METADATA)
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-cache-dir", default=DEFAULT_MODEL_CACHE_DIR)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    output_dir = _resolve_output_dir(args.output_path)
    manifest_root = resolve_project_path(args.manifest_root)
    siglip_index = resolve_project_path(args.siglip_index_path)
    siglip_metadata = resolve_project_path(args.siglip_metadata_path)
    salad_index = resolve_project_path(args.salad_index_path)
    salad_metadata = resolve_project_path(args.salad_metadata_path)
    room_graph_path = resolve_project_path(args.room_graph_path)
    room_graph = load_json(room_graph_path)

    text_embedder = create_image_embedder(
        model_name=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    passage_retriever = DynamicPassageRetriever(
        semantic_index_path=siglip_index,
        semantic_metadata_path=siglip_metadata,
        visual_index_path=salad_index,
        visual_metadata_path=salad_metadata,
        render_root=manifest_root,
        query=args.passage_query,
        retrieval_top_k=args.passage_top_k,
        target_clusters=args.passage_top_k,
        cluster_candidates=False,
        embedding_model=args.embedding_model,
        device=args.device,
        batch_size=args.batch_size,
        text_embedder=text_embedder,
    )
    visual_retriever = MemoryImageRetriever(
        metadata_path=salad_metadata,
        project_root=PROJECT_ROOT,
        render_root=manifest_root,
        use_faiss=False,
    )
    detector = create_detector(
        backend=args.detector_backend,
        model_name=args.detector_model or _default_detector_model(args.detector_backend),
        device=args.device,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        cache_dir=resolve_project_path(args.model_cache_dir),
    )
    image_embedder = create_similarity_embedder(
        similarity_backend=args.similarity_backend,
        dreamsim_type=args.dreamsim_type,
        device=args.device,
        batch_size=args.batch_size,
    )

    current_passages = passage_retriever.retrieve(args.current_room_id)
    payload = run_detected_passage_contrastive_selection(
        current_room_id=args.current_room_id,
        subgoal_room_id=args.subgoal_room_id,
        current_passages=current_passages,
        visual_retriever=visual_retriever,
        room_graph=room_graph,
        detector=detector,
        image_embedder=image_embedder,
        output_dir=output_dir,
        configuration={
            "seed": args.seed,
            "passage_query": args.passage_query,
            "passage_top_k": args.passage_top_k,
            "passage_clustering": False,
            "detector_prompt": args.detector_prompt,
            "detector_backend": args.detector_backend,
            "detector_model": args.detector_model or _default_detector_model(args.detector_backend),
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "min_box_area_ratio": args.min_box_area_ratio,
            "max_box_area_ratio": args.max_box_area_ratio,
            "crop_padding_ratio": args.crop_padding_ratio,
            "current_image_mode": args.current_image_mode,
            "mask_background_brightness": args.mask_background_brightness,
            "max_detections_per_image": args.max_detections_per_image,
            "target_sample_count": args.target_sample_count,
            "negative_sample_count": args.negative_sample_count or args.target_sample_count,
            "similarity_top_m": args.similarity_top_m,
            "target_scoring": args.target_scoring,
            "contrastive_negative_weight": args.contrastive_negative_weight,
            "similarity_backend": args.similarity_backend,
            "dreamsim_type": args.dreamsim_type if args.similarity_backend == "dreamsim" else None,
            "semantic_embedding_model": args.embedding_model,
            "visual_similarity_model": (
                f"dreamsim:{args.dreamsim_type}"
                if args.similarity_backend == "dreamsim"
                else DEFAULT_DINOV2_SALAD_MODEL
            ),
            "siglip_index_path": str(siglip_index),
            "salad_index_path": str(salad_index),
            "salad_metadata_path": str(salad_metadata),
            "room_graph_path": str(room_graph_path),
            "manifest_root": str(manifest_root),
        },
    )
    output = render_json(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DEFAULT_RESULT_FILENAME).write_text(output, encoding="utf-8")
    print(output)
    return 0 if payload.get("success") else 1


def run_detected_passage_contrastive_selection(
    *,
    current_room_id: str,
    subgoal_room_id: str,
    current_passages: Sequence[dict],
    visual_retriever: MemoryImageRetriever,
    room_graph: dict,
    detector,
    image_embedder,
    output_dir: str | Path,
    configuration: dict,
) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exports: list[dict] = []
    dirs = _prepare_output_dirs(output_dir)

    current_passages = [dict(passage) for passage in current_passages]
    _copy_current_passages(current_passages, dirs["current"], exports)
    detected_candidates, current_records = _detect_passage_candidates(
        current_passages=current_passages,
        detector=detector,
        output_dirs=dirs,
        configuration=configuration,
        exports=exports,
    )
    if not detected_candidates:
        raise ValueError("No detected passage candidates were produced.")

    target_samples = _sample_room_images(
        visual_retriever=visual_retriever,
        room_ids=[subgoal_room_id],
        sample_limit=int(configuration["target_sample_count"]),
        seed=int(configuration["seed"]),
        seed_salt=f"target:{subgoal_room_id}",
    )
    if not target_samples:
        raise ValueError(f"No target-room visual samples found for {subgoal_room_id}.")

    negative_room_ids = _negative_neighbor_room_ids(room_graph, current_room_id, subgoal_room_id)
    if not negative_room_ids:
        raise ValueError(
            f"No non-target neighbor rooms found for {current_room_id} excluding {subgoal_room_id}."
        )
    negative_sample_limit = int(configuration["negative_sample_count"])
    if configuration["target_scoring"] in {
        "contrastive_neighbor_room_max_mean",
        "contrastive_neighbor_room_average_mean",
    }:
        negative_samples_by_room = {
            room_id: _sample_room_images(
                visual_retriever=visual_retriever,
                room_ids=[room_id],
                sample_limit=negative_sample_limit,
                seed=int(configuration["seed"]),
                seed_salt=f"negative-room:{room_id}",
            )
            for room_id in negative_room_ids
        }
        negative_samples = [
            sample
            for room_id in negative_room_ids
            for sample in negative_samples_by_room.get(room_id, [])
        ]
    else:
        negative_samples = _sample_room_images(
            visual_retriever=visual_retriever,
            room_ids=negative_room_ids,
            sample_limit=negative_sample_limit,
            seed=int(configuration["seed"]),
            seed_salt="negative:" + "|".join(negative_room_ids),
        )
        negative_samples_by_room = {"__mixed__": negative_samples}
    if not negative_samples:
        raise ValueError(
            f"No negative-room visual samples found for neighbors: {', '.join(negative_room_ids)}."
        )

    _copy_sample_images(target_samples, dirs["target"], "target_sample", exports)
    _copy_sample_images(negative_samples, dirs["negative"], "negative_sample", exports)

    ranking = _score_detected_candidates(
        candidates=detected_candidates,
        target_samples=target_samples,
        negative_samples=negative_samples,
        negative_samples_by_room=negative_samples_by_room,
        image_embedder=image_embedder,
        top_m=int(configuration["similarity_top_m"]),
        target_scoring=str(configuration["target_scoring"]),
        negative_weight=float(configuration["contrastive_negative_weight"]),
    )
    _export_ranked_images(ranking, dirs, exports)
    chosen = ranking[0]
    payload = {
        "method": "detected_passage_contrastive_selection",
        "configuration": {
            "current_room_id": current_room_id,
            "subgoal_room_id": subgoal_room_id,
            **dict(configuration),
            "negative_room_ids": list(negative_room_ids),
            "negative_sample_count_by_room": {
                room_id: len(samples)
                for room_id, samples in negative_samples_by_room.items()
                if room_id != "__mixed__"
            },
        },
        "output_directory": str(output_dir),
        "result_json_path": str(output_dir / DEFAULT_RESULT_FILENAME),
        "current_room_passages": current_records,
        "detected_passage_candidates": detected_candidates,
        "passage_choice": {
            "chosen_label": chosen["label"],
            "chosen_source_label": chosen["source_label"],
            "selector_source": "detected_passage_contrastive_similarity",
            "target_scoring": configuration["target_scoring"],
            "score_field": "selection_score",
            "similarity_backend": configuration["similarity_backend"],
            "target_visual_clues": target_samples,
            "negative_visual_clues": negative_samples,
            "negative_room_ids": list(negative_room_ids),
            "passage_ranking": ranking,
        },
        "image_exports": exports,
        "success": bool(ranking),
    }
    return payload


def create_detector(
    *,
    backend: str,
    model_name: str,
    device: str,
    box_threshold: float,
    text_threshold: float,
    cache_dir: Path,
):
    if backend == "groundingdino":
        return GroundingDinoPassageDetector(
            model_name=model_name,
            device=device,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            cache_dir=cache_dir,
        )
    if backend == "owlv2":
        return OwlV2PassageDetector(
            model_name=model_name,
            device=device,
            box_threshold=box_threshold,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unsupported detector backend: {backend}")


def create_similarity_embedder(*, similarity_backend: str, dreamsim_type: str, device: str, batch_size: int):
    if similarity_backend == "dreamsim":
        return DreamSimImageEmbedder(dreamsim_type=dreamsim_type, device=device, batch_size=batch_size)
    if similarity_backend == "salad":
        return create_image_embedder(
            model_name=DEFAULT_DINOV2_SALAD_MODEL,
            device=device,
            batch_size=batch_size,
        )
    raise ValueError("Similarity backend must be 'dreamsim' or 'salad'.")


class GroundingDinoPassageDetector:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        box_threshold: float,
        text_threshold: float,
        cache_dir: Path,
    ):
        import torch
        from transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor

        self.torch = torch
        self.device = _resolve_torch_device(device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.processor = GroundingDinoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_name, cache_dir=cache_dir).eval()
        self.model.to(self.device)

    def detect(self, image_path: str | Path, prompt: str) -> list[DetectionCandidate]:
        from PIL import Image

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            inputs = inputs.to(self.device) if hasattr(inputs, "to") else {
                key: value.to(self.device) for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                input_ids=input_ids.to(self.device) if hasattr(input_ids, "to") else input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[(image.height, image.width)],
            )
        return _detections_from_processor_result(results[0] if results else {})


class OwlV2PassageDetector:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        box_threshold: float,
        cache_dir: Path,
    ):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.torch = torch
        self.device = _resolve_torch_device(device)
        self.box_threshold = float(box_threshold)
        self.processor = Owlv2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name, cache_dir=cache_dir).eval()
        self.model.to(self.device)

    def detect(self, image_path: str | Path, prompt: str) -> list[DetectionCandidate]:
        from PIL import Image

        labels = _prompt_to_labels(prompt)
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            inputs = self.processor(text=[labels], images=image, return_tensors="pt")
            inputs = inputs.to(self.device) if hasattr(inputs, "to") else {
                key: value.to(self.device) for key, value in inputs.items()
            }
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                threshold=self.box_threshold,
                target_sizes=[(image.height, image.width)],
                text_labels=[labels],
            )
        return _detections_from_processor_result(results[0] if results else {})


def _prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "current": output_dir / "current_room_passages",
        "overlay": output_dir / "detected_passage_overlays",
        "crop": output_dir / "detected_passage_crops",
        "masked": output_dir / "detected_passage_masked_images",
        "target": output_dir / "target_room_visual_clues",
        "negative": output_dir / "negative_room_visual_clues",
        "ranked_candidates": output_dir / "top_k_detected_passage_candidates",
        "ranked_source": output_dir / "top_k_source_passages",
        "ranked_crops": output_dir / "top_k_detected_crops",
        "ranked_masked": output_dir / "top_k_detected_masked_images",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _detect_passage_candidates(
    *,
    current_passages: Sequence[dict],
    detector,
    output_dirs: dict[str, Path],
    configuration: dict,
    exports: list[dict],
) -> tuple[list[dict], list[dict]]:
    from PIL import Image, ImageDraw

    detected_candidates: list[dict] = []
    current_records: list[dict] = []
    for passage_index, passage in enumerate(current_passages, start=1):
        source_path = _resolve_image_path(passage)
        if source_path is None or not source_path.exists():
            continue
        source_label = str(passage.get("label") or f"passage_{passage_index}")
        raw_detections = detector.detect(source_path, str(configuration["detector_prompt"]))
        with Image.open(source_path) as opened:
            image = opened.convert("RGB")
            image_size = image.size
            kept = _filter_detections(
                raw_detections,
                image_size=image_size,
                min_area_ratio=float(configuration["min_box_area_ratio"]),
                max_area_ratio=float(configuration["max_box_area_ratio"]),
            )
            kept = kept[: max(int(configuration["max_detections_per_image"]), 1)]
            overlay_path = output_dirs["overlay"] / f"semantic_rank_{passage_index:02d}_{_safe_slug(source_label)}.png"
            _save_detection_overlay(
                image=image,
                detections=kept,
                raw_count=len(raw_detections),
                output_path=overlay_path,
            )
            exports.append(
                {
                    "kind": "detection_overlay",
                    "label": source_label,
                    "source_image_path": str(source_path),
                    "copied_image_path": str(overlay_path),
                }
            )
            current_record = dict(passage)
            current_record.update(
                {
                    "source_image_path": str(source_path),
                    "overlay_image_path": str(overlay_path),
                    "raw_detections": [_detection_to_dict(item, image_size=image_size) for item in raw_detections],
                    "kept_detections": [_detection_to_dict(item, image_size=image_size) for item in kept],
                    "detection_status": "detected" if kept else "no_detection",
                }
            )
            current_records.append(current_record)

            detections_for_candidates = kept or [
                DetectionCandidate(
                    box_xyxy=(0.0, 0.0, float(image.width), float(image.height)),
                    score=0.0,
                    label="full_image_fallback",
                )
            ]
            for detection_index, detection in enumerate(detections_for_candidates, start=1):
                detection_box = _clip_box(detection.box_xyxy, image_size)
                crop_box = _expand_box(
                    detection_box,
                    image_size=image_size,
                    padding_ratio=float(configuration["crop_padding_ratio"]),
                )
                detected_label = f"{source_label}D{detection_index}"
                if not kept:
                    detected_label = f"{source_label}D0"
                crop_path = output_dirs["crop"] / (
                    f"semantic_rank_{passage_index:02d}_{_safe_slug(detected_label)}_crop.png"
                )
                masked_path = output_dirs["masked"] / (
                    f"semantic_rank_{passage_index:02d}_{_safe_slug(detected_label)}_masked.png"
                )
                image.crop(crop_box).save(crop_path)
                _masked_full_image(
                    image,
                    focus_box=crop_box,
                    background_brightness=float(configuration["mask_background_brightness"]),
                ).save(masked_path)
                comparison_path = masked_path if configuration["current_image_mode"] == "mask" else crop_path
                candidate = {
                    "label": detected_label,
                    "source_label": source_label,
                    "source_passage": {
                        "label": source_label,
                        "semantic_rank": passage_index,
                        "semantic_score": passage.get("semantic_score"),
                        "room_id": passage.get("room_id"),
                        "pano_id": passage.get("pano_id"),
                        "capture_index": passage.get("capture_index"),
                        "image_path": str(source_path),
                        "copied_image_path": passage.get("copied_image_path"),
                    },
                    "source_semantic_rank": passage_index,
                    "source_semantic_score": passage.get("semantic_score"),
                    "room_id": passage.get("room_id"),
                    "pano_id": passage.get("pano_id"),
                    "capture_index": passage.get("capture_index"),
                    "capture_label": passage.get("capture_label"),
                    "capture_heading": passage.get("capture_heading"),
                    "source_image_path": str(source_path),
                    "detection_status": "detected" if kept else "no_detection",
                    "detection_rank": detection_index if kept else None,
                    "detection_label": detection.label,
                    "detection_score": float(detection.score) if kept else None,
                    "detection_box_xyxy": [float(value) for value in detection_box],
                    "crop_box_xyxy": [int(value) for value in crop_box],
                    "crop_image_path": str(crop_path),
                    "masked_image_path": str(masked_path),
                    "comparison_image_path": str(comparison_path),
                    "current_image_mode": configuration["current_image_mode"],
                }
                detected_candidates.append(candidate)
                exports.extend(
                    [
                        {
                            "kind": "detected_crop",
                            "label": detected_label,
                            "source_image_path": str(source_path),
                            "copied_image_path": str(crop_path),
                        },
                        {
                            "kind": "detected_masked_image",
                            "label": detected_label,
                            "source_image_path": str(source_path),
                            "copied_image_path": str(masked_path),
                        },
                    ]
                )
    return detected_candidates, current_records


def _score_detected_candidates(
    *,
    candidates: Sequence[dict],
    target_samples: Sequence[dict],
    negative_samples: Sequence[dict],
    negative_samples_by_room: dict[str, list[dict]],
    image_embedder,
    top_m: int,
    target_scoring: str,
    negative_weight: float,
) -> list[dict]:
    import numpy as np

    candidate_paths = [Path(candidate["comparison_image_path"]) for candidate in candidates]
    target_paths = [Path(sample["image_path"]) for sample in target_samples]
    negative_paths = [Path(sample["image_path"]) for sample in negative_samples]
    all_paths = candidate_paths + target_paths + negative_paths
    embeddings = normalize_rows(np.asarray(image_embedder.encode_image_paths(all_paths), dtype=np.float32))
    if embeddings.shape[0] != len(all_paths):
        raise RuntimeError("Image embedder returned the wrong number of embeddings.")

    candidate_embeddings = embeddings[: len(candidate_paths)]
    target_start = len(candidate_paths)
    target_end = target_start + len(target_paths)
    target_embeddings = embeddings[target_start:target_end]
    negative_embeddings = embeddings[target_end:]
    negative_offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    for room_id, samples in negative_samples_by_room.items():
        negative_offsets[room_id] = (offset, offset + len(samples))
        offset += len(samples)

    ranking: list[dict] = []
    for candidate_index, candidate in enumerate(candidates):
        query_embedding = candidate_embeddings[candidate_index]
        target_scores = target_embeddings @ query_embedding
        target_values = [float(score) for score in target_scores.tolist()]
        target_mean = _mean(target_values)
        matched_target = _matched_samples(target_values, target_samples, top_m=top_m)

        if target_scoring in {
            "contrastive_neighbor_room_max_mean",
            "contrastive_neighbor_room_average_mean",
        }:
            negative_result = _score_per_room_negatives(
                query_embedding=query_embedding,
                negative_embeddings=negative_embeddings,
                negative_samples_by_room=negative_samples_by_room,
                negative_offsets=negative_offsets,
                top_m=top_m,
                target_scoring=target_scoring,
            )
        else:
            negative_result = _score_mixed_negatives(
                query_embedding=query_embedding,
                negative_embeddings=negative_embeddings,
                negative_samples=negative_samples,
                top_m=top_m,
            )

        primary_negative = float(negative_result["primary_negative_similarity"])
        selection_score = target_mean - float(negative_weight) * primary_negative
        item = dict(candidate)
        item.update(
            {
                "selection_score": float(selection_score),
                "target_mean_similarity": float(target_mean),
                "negative_mean_similarity": primary_negative,
                "contrastive_similarity": float(selection_score),
                "contrastive_negative_weight": float(negative_weight),
                "top_m_mean_similarity": _mean([sample["similarity"] for sample in matched_target]),
                "max_similarity": float(max(target_values) if target_values else 0.0),
                "hard_negative_similarity": float(negative_result["hard_negative_similarity"]),
                "hard_negative_room_id": negative_result["hard_negative_room_id"],
                "per_room_average_negative_similarity": float(
                    negative_result["per_room_average_negative_similarity"]
                ),
                "negative_room_scores": list(negative_result["negative_room_scores"]),
                "matched_target_samples": matched_target,
                "matched_negative_samples": list(negative_result["matched_negative_samples"]),
            }
        )
        ranking.append(item)

    ranking.sort(
        key=lambda item: (
            -float(item["selection_score"]),
            -float(item["target_mean_similarity"]),
            -float(item["top_m_mean_similarity"]),
            -float(item["max_similarity"]),
            str(item["label"]),
        )
    )
    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank
        item["selected"] = rank == 1
    return ranking


def _score_mixed_negatives(
    *,
    query_embedding,
    negative_embeddings,
    negative_samples: Sequence[dict],
    top_m: int,
) -> dict:
    scores = negative_embeddings @ query_embedding
    values = [float(score) for score in scores.tolist()]
    negative_mean = _mean(values)
    return {
        "primary_negative_similarity": float(negative_mean),
        "per_room_average_negative_similarity": float(negative_mean),
        "hard_negative_similarity": float(negative_mean),
        "hard_negative_room_id": None,
        "negative_room_scores": [],
        "matched_negative_samples": _matched_samples(values, negative_samples, top_m=top_m),
    }


def _score_per_room_negatives(
    *,
    query_embedding,
    negative_embeddings,
    negative_samples_by_room: dict[str, list[dict]],
    negative_offsets: dict[str, tuple[int, int]],
    top_m: int,
    target_scoring: str,
) -> dict:
    room_scores = []
    for room_id in sorted(negative_samples_by_room):
        if room_id == "__mixed__":
            continue
        start, end = negative_offsets.get(room_id, (0, 0))
        samples = negative_samples_by_room.get(room_id, [])
        if start == end or not samples:
            continue
        values = [float(score) for score in (negative_embeddings[start:end] @ query_embedding).tolist()]
        matched = _matched_samples(values, samples, top_m=top_m)
        room_scores.append(
            {
                "room_id": room_id,
                "sample_count": len(samples),
                "mean_similarity": float(_mean(values)),
                "top_m_mean_similarity": float(_mean([sample["similarity"] for sample in matched])),
                "max_similarity": float(max(values) if values else 0.0),
                "matched_negative_samples": matched,
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
    room_scores.sort(
        key=lambda item: (
            -float(item["mean_similarity"]),
            -float(item["top_m_mean_similarity"]),
            -float(item["max_similarity"]),
            str(item["room_id"]),
        )
    )
    hard_negative = room_scores[0]
    average_negative = _mean([float(item["mean_similarity"]) for item in room_scores])
    primary = average_negative if target_scoring == "contrastive_neighbor_room_average_mean" else float(
        hard_negative["mean_similarity"]
    )
    return {
        "primary_negative_similarity": float(primary),
        "per_room_average_negative_similarity": float(average_negative),
        "hard_negative_similarity": float(hard_negative["mean_similarity"]),
        "hard_negative_room_id": hard_negative["room_id"],
        "negative_room_scores": room_scores,
        "matched_negative_samples": list(hard_negative["matched_negative_samples"]),
    }


def _matched_samples(values: Sequence[float], samples: Sequence[dict], *, top_m: int) -> list[dict]:
    scored = sorted(
        ((float(score), dict(samples[index])) for index, score in enumerate(values)),
        key=lambda item: (
            -item[0],
            str(item[1].get("room_id", "")),
            str(item[1].get("pano_id", "")),
            int(item[1].get("capture_index", 0)),
        ),
    )
    return [
        {
            "rank": rank,
            "similarity": float(score),
            **sample,
        }
        for rank, (score, sample) in enumerate(scored[: min(int(top_m), len(scored))], start=1)
    ]


def _sample_room_images(
    *,
    visual_retriever: MemoryImageRetriever,
    room_ids: Sequence[str],
    sample_limit: int,
    seed: int,
    seed_salt: str,
) -> list[dict]:
    room_id_set = {str(room_id) for room_id in room_ids}
    samples = []
    for index, item in enumerate(visual_retriever.metadata_items):
        if item.get("room_id") not in room_id_set:
            continue
        sample = _sample_from_metadata(visual_retriever, index, item)
        if sample is not None:
            samples.append(sample)
    return _pano_balanced_sample(samples, sample_limit=sample_limit, seed=seed, seed_salt=seed_salt)


def _sample_from_metadata(visual_retriever: MemoryImageRetriever, index: int, item: dict) -> dict | None:
    pano_id = item.get("pano_id")
    capture_index = item.get("capture_index")
    if not isinstance(pano_id, str) or not isinstance(capture_index, int):
        return None
    image_path = visual_retriever.resolve_capture_path(item)
    if not image_path or not Path(image_path).exists():
        return None
    return {
        "memory_index": int(item.get("memory_index", index)),
        "room_id": item.get("room_id"),
        "pano_id": pano_id,
        "capture_index": capture_index,
        "capture_label": item.get("capture_label"),
        "capture_heading": item.get("capture_heading"),
        "image_path": image_path,
    }


def _pano_balanced_sample(
    samples: Sequence[dict],
    *,
    sample_limit: int,
    seed: int,
    seed_salt: str,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(str(sample.get("pano_id", "")), []).append(dict(sample))
    if not grouped:
        return []
    rng = random.Random(_stable_seed(seed, seed_salt))
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


def _negative_neighbor_room_ids(room_graph: dict, current_room_id: str, subgoal_room_id: str) -> tuple[str, ...]:
    node = room_graph.get(current_room_id) if isinstance(room_graph, dict) else None
    neighbors = node.get("neighbors") if isinstance(node, dict) else None
    if not isinstance(neighbors, list):
        return ()
    excluded = {current_room_id, subgoal_room_id}
    room_ids = []
    for neighbor in neighbors:
        if isinstance(neighbor, dict):
            target_room_id = neighbor.get("target_room_id")
        else:
            target_room_id = neighbor
        if not isinstance(target_room_id, str) or target_room_id in excluded:
            continue
        if target_room_id not in room_ids:
            room_ids.append(target_room_id)
    return tuple(sorted(room_ids))


def _copy_current_passages(passages: Sequence[dict], output_dir: Path, exports: list[dict]) -> None:
    for index, passage in enumerate(passages, start=1):
        source = _resolve_image_path(passage)
        if source is None or not source.exists():
            continue
        label = str(passage.get("label") or f"passage_{index}")
        destination = output_dir / f"semantic_rank_{index:02d}_{_safe_slug(label)}{source.suffix or '.png'}"
        copied = _copy_image(source, destination)
        passage["copied_image_path"] = copied
        exports.append(
            {
                "kind": "current_passage",
                "label": label,
                "source_image_path": str(source),
                "copied_image_path": copied,
            }
        )


def _copy_sample_images(samples: Sequence[dict], output_dir: Path, prefix: str, exports: list[dict]) -> None:
    for index, sample in enumerate(samples, start=1):
        source = _resolve_image_path(sample)
        if source is None or not source.exists():
            continue
        destination = output_dir / (
            f"{prefix}_{index:03d}_{_safe_slug(sample.get('room_id'))}_"
            f"{_safe_slug(sample.get('pano_id'))}_{sample.get('capture_index', 'x')}{source.suffix or '.png'}"
        )
        copied = _copy_image(source, destination)
        sample["copied_image_path"] = copied
        exports.append(
            {
                "kind": prefix,
                "room_id": sample.get("room_id"),
                "pano_id": sample.get("pano_id"),
                "capture_index": sample.get("capture_index"),
                "source_image_path": str(source),
                "copied_image_path": copied,
            }
        )


def _export_ranked_images(ranking: Sequence[dict], dirs: dict[str, Path], exports: list[dict]) -> None:
    for item in ranking:
        rank = int(item["rank"])
        label = str(item["label"])
        for key, directory, output_key, kind in [
            ("comparison_image_path", dirs["ranked_candidates"], "top_k_detected_candidate_image_path", "top_k_detected_candidate"),
            ("source_image_path", dirs["ranked_source"], "top_k_source_passage_image_path", "top_k_source_passage"),
            ("crop_image_path", dirs["ranked_crops"], "top_k_detected_crop_image_path", "top_k_detected_crop"),
            ("masked_image_path", dirs["ranked_masked"], "top_k_detected_masked_image_path", "top_k_detected_masked"),
        ]:
            source = Path(item[key])
            destination = directory / (
                f"rank_{rank:02d}_{_safe_slug(label)}_score_{float(item['selection_score']):.4f}"
                f"{source.suffix or '.png'}"
            )
            copied = _copy_image(source, destination)
            item[output_key] = copied
            exports.append(
                {
                    "kind": kind,
                    "label": label,
                    "source_image_path": str(source),
                    "copied_image_path": copied,
                }
            )


def _filter_detections(
    detections: Sequence[DetectionCandidate],
    *,
    image_size: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[DetectionCandidate]:
    image_area = max(float(image_size[0] * image_size[1]), 1.0)
    kept = []
    for detection in detections:
        box = _clip_box(detection.box_xyxy, image_size)
        area_ratio = _box_area(box) / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        kept.append(DetectionCandidate(box_xyxy=box, score=float(detection.score), label=detection.label))
    kept.sort(
        key=lambda item: (
            0 if _is_opening_like(item.label) else 1,
            -float(item.score),
            -float(_box_area(item.box_xyxy)),
            str(item.label),
        )
    )
    return kept


def _save_detection_overlay(
    *,
    image,
    detections: Sequence[DetectionCandidate],
    raw_count: int,
    output_path: Path,
) -> None:
    from PIL import ImageDraw

    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    if detections:
        for index, detection in enumerate(detections, start=1):
            color = (255, 170, 0) if index == 1 else (0, 255, 80)
            draw.rectangle(_int_box(detection.box_xyxy), outline=color, width=3)
            draw.text(
                (int(detection.box_xyxy[0]) + 4, int(detection.box_xyxy[1]) + 4),
                f"{index}:{detection.label} {float(detection.score):.3f}",
                fill=(255, 255, 255),
            )
    else:
        draw.rectangle((0, 0, image.width - 1, image.height - 1), outline=(255, 0, 0), width=3)
        draw.text((8, 8), f"no detection / raw={raw_count}", fill=(255, 255, 255))
    overlay.save(output_path)


def _masked_full_image(image, *, focus_box: tuple[int, int, int, int], background_brightness: float):
    from PIL import ImageEnhance

    brightness = min(max(float(background_brightness), 0.0), 1.0)
    background = ImageEnhance.Brightness(image).enhance(brightness)
    x0, y0, x1, y1 = focus_box
    background.paste(image.crop(focus_box), (x0, y0, x1, y1))
    return background


def _expand_box(
    box_xyxy: Sequence[float],
    *,
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _clip_box(box_xyxy, image_size)
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    pad_x = width * max(float(padding_ratio), 0.0)
    pad_y = height * max(float(padding_ratio), 0.0)
    image_width, image_height = image_size
    return (
        max(0, int(round(x0 - pad_x))),
        max(0, int(round(y0 - pad_y))),
        min(image_width, int(round(x1 + pad_x))),
        min(image_height, int(round(y1 + pad_y))),
    )


def _clip_box(box_xyxy: Sequence[float], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    values = [float(value) for value in box_xyxy]
    if len(values) != 4:
        raise ValueError(f"Expected xyxy box with 4 values, got {box_xyxy!r}")
    x0, y0, x1, y1 = values
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    width, height = image_size
    return (
        min(max(x0, 0.0), float(width)),
        min(max(y0, 0.0), float(height)),
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
    )


def _box_area(box_xyxy: Sequence[float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in box_xyxy]
    return max(x1 - x0, 0.0) * max(y1 - y0, 0.0)


def _int_box(box_xyxy: Sequence[float]) -> tuple[int, int, int, int]:
    return tuple(int(round(float(value))) for value in box_xyxy)  # type: ignore[return-value]


def _detection_to_dict(detection: DetectionCandidate, *, image_size: tuple[int, int]) -> dict:
    box = _clip_box(detection.box_xyxy, image_size)
    return {
        "label": detection.label,
        "score": float(detection.score),
        "box_xyxy": [float(value) for value in box],
        "area_ratio": float(_box_area(box) / max(float(image_size[0] * image_size[1]), 1.0)),
    }


def _detections_from_processor_result(result: dict) -> list[DetectionCandidate]:
    boxes = _to_list(result.get("boxes", []))
    scores = _to_list(result.get("scores", []))
    labels = _to_list(result.get("text_labels", result.get("labels", [])))
    detections = []
    for index, box in enumerate(boxes):
        if index >= len(scores):
            break
        label = labels[index] if index < len(labels) else ""
        detections.append(
            DetectionCandidate(
                box_xyxy=tuple(float(value) for value in box),
                score=float(scores[index]),
                label=str(label),
            )
        )
    return detections


def _to_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _prompt_to_labels(prompt: str) -> list[str]:
    labels = [part.strip() for part in str(prompt).replace("\n", " ").split(".")]
    return [label for label in labels if label] or ["doorway", "room entrance", "gallery entrance", "opening"]


def _is_opening_like(label: object) -> bool:
    text = str(label or "").lower()
    return any(token in text for token in ["door", "entrance", "opening", "gallery"])


def _copy_image(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def _resolve_image_path(record: dict) -> Path | None:
    raw_path = record.get("image_path") or record.get("capture_path") or record.get("source_image_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return resolve_project_path(path)


def _resolve_output_dir(output_path: str | Path) -> Path:
    resolved = resolve_project_path(output_path)
    if resolved.suffix.lower() == ".json":
        raise ValueError("--output-path expects an experiment directory, not a JSON file path.")
    return resolved


def _default_detector_model(backend: str) -> str:
    if backend == "groundingdino":
        return DEFAULT_GROUNDING_DINO_MODEL
    if backend == "owlv2":
        return DEFAULT_OWLV2_MODEL
    raise ValueError(f"Unsupported detector backend: {backend}")


def _mean(values: Sequence[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / max(len(values), 1)


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _safe_slug(value: object) -> str:
    text = str(value or "image")
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return cleaned.strip("_") or "image"


if __name__ == "__main__":
    raise SystemExit(main())
