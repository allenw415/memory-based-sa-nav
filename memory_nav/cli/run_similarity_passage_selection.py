from __future__ import annotations

import argparse
import shutil
from pathlib import Path

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
)
from memory_nav.memory.retrieval import MemoryImageRetriever  # noqa: E402
from memory_nav.navigation import (  # noqa: E402
    DEFAULT_PASSAGE_QUERY,
    DynamicPassageRetriever,
    SimilarityPassageSelector,
)


DEFAULT_SIGLIP_INDEX = "artifacts/memory_localization/floor0_siglip2_images_fov90.npz"
DEFAULT_SIGLIP_METADATA = "artifacts/memory_localization/floor0_siglip2_images_fov90.metadata.json"
DEFAULT_SALAD_INDEX = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz"
DEFAULT_SALAD_METADATA = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
DEFAULT_MANIFEST_ROOT = "renders/room_grounding_fov90"
DEFAULT_ROOM_GRAPH_PATH = "dataset/sites/british_museum/normalized/room_graph.json"
DEFAULT_RESULT_FILENAME = "result.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run similarity-only passage selection for one current/subgoal room pair."
    )
    parser.add_argument("--current-room-id", required=True)
    parser.add_argument("--subgoal-room-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passage-query", default=DEFAULT_PASSAGE_QUERY)
    parser.add_argument("--passage-top-k", type=int, default=20)
    parser.add_argument("--target-sample-count", type=int, default=64)
    parser.add_argument("--negative-sample-count", type=int, default=None)
    parser.add_argument("--similarity-top-m", type=int, default=5)
    parser.add_argument(
        "--target-scoring",
        choices=[
            "sample_mean",
            "contrastive_neighbor_mean",
            "contrastive_neighbor_room_max_mean",
            "contrastive_neighbor_room_average_mean",
        ],
        default="sample_mean",
    )
    parser.add_argument("--contrastive-negative-weight", type=float, default=1.0)
    parser.add_argument("--room-graph-path", default=DEFAULT_ROOM_GRAPH_PATH)
    parser.add_argument("--similarity-backend", choices=["salad", "dreamsim"], default="salad")
    parser.add_argument(
        "--dreamsim-type",
        default="ensemble",
        help="DreamSim checkpoint type used when --similarity-backend dreamsim.",
    )
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--siglip-index-path", default=DEFAULT_SIGLIP_INDEX)
    parser.add_argument("--siglip-metadata-path", default=DEFAULT_SIGLIP_METADATA)
    parser.add_argument("--salad-index-path", default=DEFAULT_SALAD_INDEX)
    parser.add_argument("--salad-metadata-path", default=DEFAULT_SALAD_METADATA)
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-path",
        help="Experiment output directory. Writes result.json and image subfolders inside it.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest_root = resolve_project_path(args.manifest_root)
    siglip_index = resolve_project_path(args.siglip_index_path)
    siglip_metadata = resolve_project_path(args.siglip_metadata_path)
    salad_index = resolve_project_path(args.salad_index_path)
    salad_metadata = resolve_project_path(args.salad_metadata_path)
    room_graph_path = resolve_project_path(args.room_graph_path)
    room_graph = load_json(room_graph_path) if args.target_scoring != "sample_mean" else {}

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
        index_path=salad_index,
        metadata_path=salad_metadata,
        embedding_model=DEFAULT_DINOV2_SALAD_MODEL,
        device=args.device,
        batch_size=args.batch_size,
        use_faiss=False,
        project_root=PROJECT_ROOT,
        render_root=manifest_root,
    )
    similarity_embedder = None
    visual_similarity_model = DEFAULT_DINOV2_SALAD_MODEL
    if args.similarity_backend == "dreamsim":
        similarity_embedder = DreamSimImageEmbedder(
            dreamsim_type=args.dreamsim_type,
            device=args.device,
            batch_size=args.batch_size,
        )
        visual_similarity_model = f"dreamsim:{args.dreamsim_type}"

    selector = SimilarityPassageSelector(
        visual_retriever=visual_retriever,
        target_sample_count=args.target_sample_count,
        negative_sample_count=args.negative_sample_count,
        top_m=args.similarity_top_m,
        seed=args.seed,
        similarity_backend=args.similarity_backend,
        image_embedder=similarity_embedder,
        target_scoring=args.target_scoring,
        room_graph=room_graph,
        contrastive_negative_weight=args.contrastive_negative_weight,
    )

    current_candidates = passage_retriever.retrieve(args.current_room_id)
    choice = selector.choose(
        current_room_id=args.current_room_id,
        subgoal_room_id=args.subgoal_room_id,
        current_candidates=current_candidates,
        subgoal_candidates=[],
    )
    score_aggregation = {
        "sample_mean": "mean_all_sampled_target_similarities",
        "contrastive_neighbor_mean": "mean_sampled_target_similarities_minus_neighbor_mean_similarity",
        "contrastive_neighbor_room_max_mean": (
            "mean_sampled_target_similarities_minus_max_neighbor_room_mean_similarity"
        ),
        "contrastive_neighbor_room_average_mean": (
            "mean_sampled_target_similarities_minus_average_neighbor_room_mean_similarity"
        ),
    }[args.target_scoring]
    payload = {
        "method": "similarity_only_passage_selection",
        "configuration": {
            "current_room_id": args.current_room_id,
            "subgoal_room_id": args.subgoal_room_id,
            "seed": args.seed,
            "passage_query": args.passage_query,
            "passage_top_k": args.passage_top_k,
            "passage_clustering": False,
            "target_sample_count": args.target_sample_count,
            "negative_sample_count": args.negative_sample_count or args.target_sample_count,
            "similarity_top_m": args.similarity_top_m,
            "target_scoring": args.target_scoring,
            "contrastive_negative_weight": args.contrastive_negative_weight,
            "score_aggregation": score_aggregation,
            "similarity_backend": args.similarity_backend,
            "dreamsim_type": args.dreamsim_type if args.similarity_backend == "dreamsim" else None,
            "semantic_embedding_model": args.embedding_model,
            "visual_similarity_model": visual_similarity_model,
            "siglip_index_path": str(siglip_index),
            "salad_index_path": str(salad_index),
            "room_graph_path": str(room_graph_path),
            "manifest_root": str(manifest_root),
        },
        "current_room_passages": current_candidates,
        "passage_choice": choice,
        "success": choice.get("chosen_label") is not None,
    }
    output_dir = None
    output_json_path = None
    if args.output_path:
        try:
            output_dir = _resolve_output_dir(args.output_path)
        except ValueError as exc:
            parser.error(str(exc))
        output_json_path = output_dir / DEFAULT_RESULT_FILENAME
        payload["output_directory"] = str(output_dir)
        payload["result_json_path"] = str(output_json_path)
        payload["image_exports"] = _copy_retrieved_images(payload, output_dir)
    output = render_json(payload)
    if output_dir is not None and output_json_path is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(output, encoding="utf-8")
    print(output)
    return 0 if payload["success"] else 1


def _resolve_output_dir(output_path: str | Path) -> Path:
    resolved = resolve_project_path(output_path)
    if resolved.suffix.lower() == ".json":
        raise ValueError("--output-path expects an experiment directory, not a JSON file path.")
    return resolved


class DreamSimImageEmbedder:
    def __init__(self, *, dreamsim_type: str, device: str, batch_size: int):
        self.dreamsim_type = dreamsim_type
        self.device = _resolve_torch_device(device)
        self.batch_size = max(int(batch_size), 1)
        from dreamsim import dreamsim

        self.model, self.preprocess = dreamsim(
            pretrained=True,
            device=self.device,
            dreamsim_type=dreamsim_type,
        )
        self.model = self.model.eval()

    def encode_image_paths(self, image_paths):
        import numpy as np
        import torch
        from PIL import Image

        batches = []
        paths = [Path(path) for path in image_paths]
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            try:
                tensors = [self.preprocess(image) for image in images]
                inputs = torch.cat(
                    [tensor if tensor.ndim == 4 else tensor.unsqueeze(0) for tensor in tensors],
                    dim=0,
                ).to(self.device)
                with torch.inference_mode():
                    embeddings = self.model.embed(inputs)
                if embeddings.ndim > 2:
                    embeddings = embeddings.flatten(start_dim=1)
                embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
                batches.append(embeddings.detach().cpu().to(torch.float32).numpy())
            finally:
                for image in images:
                    image.close()
        if not batches:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(batches, axis=0).astype(np.float32)


def _resolve_torch_device(requested_device: str) -> str:
    import torch

    normalized = (requested_device or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def _copy_retrieved_images(payload: dict, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    current_dir = output_dir / "current_room_passages"
    target_dir = output_dir / "target_room_visual_clues"
    negative_dir = output_dir / "negative_room_visual_clues"
    top_k_dir = output_dir / "top_k_passages"
    for directory in [current_dir, target_dir, negative_dir, top_k_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    exports: list[dict] = []
    current_by_label: dict[str, str] = {}
    current_record_by_label: dict[str, dict] = {}
    target_by_key: dict[tuple[object, object, object], str] = {}
    negative_by_key: dict[tuple[object, object, object], str] = {}

    for index, passage in enumerate(payload.get("current_room_passages", []), start=1):
        if not isinstance(passage, dict):
            continue
        if passage.get("label") is not None:
            current_record_by_label[str(passage["label"])] = passage
        copied = _copy_image_record(
            passage,
            current_dir,
            prefix=f"semantic_rank_{index:02d}_{_safe_slug(passage.get('label'))}",
            kind="current_passage",
            exports=exports,
        )
        if copied and passage.get("label") is not None:
            current_by_label[str(passage["label"])] = copied

    choice = payload.get("passage_choice")
    if isinstance(choice, dict):
        target_visual_clues = choice.get("target_visual_clues")
        if isinstance(target_visual_clues, list):
            for index, sample in enumerate(target_visual_clues, start=1):
                if not isinstance(sample, dict):
                    continue
                copied = _copy_image_record(
                    sample,
                    target_dir,
                    prefix=f"target_sample_{index:03d}_{_safe_slug(sample.get('pano_id'))}_{sample.get('capture_index', 'x')}",
                    kind="target_visual_clue",
                    exports=exports,
                )
                if copied:
                    target_by_key[_sample_key(sample)] = copied

        negative_visual_clues = choice.get("negative_visual_clues")
        if isinstance(negative_visual_clues, list):
            for index, sample in enumerate(negative_visual_clues, start=1):
                if not isinstance(sample, dict):
                    continue
                copied = _copy_image_record(
                    sample,
                    negative_dir,
                    prefix=(
                        f"negative_sample_{index:03d}_{_safe_slug(sample.get('room_id'))}_"
                        f"{_safe_slug(sample.get('pano_id'))}_{sample.get('capture_index', 'x')}"
                    ),
                    kind="negative_visual_clue",
                    exports=exports,
                )
                if copied:
                    negative_by_key[_sample_key(sample)] = copied

        ranking = choice.get("passage_ranking")
        if isinstance(ranking, list):
            for index, passage in enumerate(ranking, start=1):
                if not isinstance(passage, dict):
                    continue
                copied = current_by_label.get(str(passage.get("label")))
                if copied:
                    passage["current_room_passage_image_path"] = copied
                source_record = dict(current_record_by_label.get(str(passage.get("label")), {}))
                source_record.update(passage)
                top_k_copied = _copy_image_record(
                    source_record,
                    top_k_dir,
                    prefix=f"similarity_rank_{index:02d}_{_safe_slug(passage.get('label'))}",
                    kind="top_k_passage",
                    exports=exports,
                )
                if top_k_copied:
                    passage["top_k_passage_image_path"] = top_k_copied
                matched = passage.get("matched_target_samples")
                if isinstance(matched, list):
                    for sample in matched:
                        if not isinstance(sample, dict):
                            continue
                        copied = target_by_key.get(_sample_key(sample))
                        if copied:
                            sample["copied_image_path"] = copied
                matched_negative = passage.get("matched_negative_samples")
                if isinstance(matched_negative, list):
                    for sample in matched_negative:
                        if not isinstance(sample, dict):
                            continue
                        copied = negative_by_key.get(_sample_key(sample))
                        if copied:
                            sample["copied_image_path"] = copied

    return exports


def _copy_image_record(
    record: dict,
    output_dir: Path,
    *,
    prefix: str,
    kind: str,
    exports: list[dict],
) -> str | None:
    source = _resolve_image_path(record)
    if source is None or not source.exists():
        return None
    suffix = source.suffix or ".png"
    destination = output_dir / f"{prefix}{suffix}"
    shutil.copy2(source, destination)
    copied_path = str(destination)
    record["copied_image_path"] = copied_path
    exports.append(
        {
            "kind": kind,
            "label": record.get("label"),
            "room_id": record.get("room_id"),
            "pano_id": record.get("pano_id"),
            "capture_index": record.get("capture_index"),
            "source_image_path": str(source),
            "copied_image_path": copied_path,
        }
    )
    return copied_path


def _resolve_image_path(record: dict) -> Path | None:
    raw_path = record.get("image_path") or record.get("capture_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return resolve_project_path(path)


def _sample_key(sample: dict) -> tuple[object, object, object]:
    return (
        sample.get("memory_index"),
        sample.get("pano_id"),
        sample.get("capture_index"),
    )


def _safe_slug(value: object) -> str:
    text = str(value or "image")
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)
    return cleaned.strip("_") or "image"


if __name__ == "__main__":
    raise SystemExit(main())
