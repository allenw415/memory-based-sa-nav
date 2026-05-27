from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav import PanoramaRenderer, get_env_value, load_dotenv
from memory_nav.data.memory_localization import (
    DEFAULT_SIGLIP2_MODEL,
    deduplicate_candidates_by_pano,
    is_valid_room_id,
    MissingDependencyError,
    SigLIP2Embedder,
    brute_force_search,
    load_faiss_index,
    load_image_index_artifacts,
    load_json,
    load_manifest_captures,
    predict_room_from_candidates,
    resolve_siglip2_model_name,
    search_image_index,
    select_query_capture_records,
    write_json,
)


load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run image-memory localization for a single pano id.")
    parser.add_argument("--pano-id", required=True)
    parser.add_argument("--index-path", default="artifacts/memory_localization/floor0_siglip2_images.npz")
    parser.add_argument(
        "--metadata-path",
        default="artifacts/memory_localization/floor0_siglip2_images.metadata.json",
    )
    parser.add_argument("--faiss-path", default="artifacts/memory_localization/floor0_siglip2_images.faiss")
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--render-api-key",
        default=get_env_value("GMAPS_KEY", "NAV_GMAPS_KEY", "GMAPS_API_KEY"),
    )
    parser.add_argument("--render-output-dir", default="renders/room_grounding")
    parser.add_argument("--render-seed", type=int, default=0)
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "grounding", "graph"], default="museum")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=45)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--render-timeout", type=float, default=30.0)
    parser.add_argument("--max-captures", type=int, default=8)
    parser.add_argument("--query-view-count", type=int, default=4)
    parser.add_argument("--query-selection", choices=["evenly-spaced", "first", "random"], default="evenly-spaced")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--dedup-by-pano", action="store_true")
    parser.add_argument("--exclude-same-pano", action="store_true", default=True)
    parser.add_argument("--include-same-pano", action="store_true")
    parser.add_argument("--preview-candidates", type=int, default=5)
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--output-path")
    return parser


def emit_progress(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def ensure_manifest(
    *,
    renderer: PanoramaRenderer,
    artifacts_dir: Path,
    render_api_key: str | None,
    render_output_dir: Path,
    pano_id: str,
    heading_mode: str,
    pitch: float,
    fov: int,
    width: int,
    height: int,
) -> Path:
    if not render_api_key:
        raise RuntimeError("Missing GMAPS_KEY or GMAPS_API_KEY to render pano views.")
    manifest = renderer.render(
        pano_id=pano_id,
        api_key=render_api_key,
        output_dir=str(render_output_dir),
        heading_mode=heading_mode,
        pitch=pitch,
        fov=fov,
        width=width,
        height=height,
        graph_path=str(artifacts_dir / "pano_graph.json"),
    )
    return Path(str(manifest["manifest_path"])).resolve()


def normalize_distribution(room_scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in room_scores.values())
    if total <= 0.0:
        return {}
    return {room_id: score / total for room_id, score in room_scores.items()}


def main() -> int:
    args = build_parser().parse_args()
    if args.include_same_pano:
        args.exclude_same_pano = False

    metadata_path = (PROJECT_ROOT / args.metadata_path).resolve()
    index_path = (PROJECT_ROOT / args.index_path).resolve()
    faiss_path = (PROJECT_ROOT / args.faiss_path).resolve()
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    render_output_dir = (PROJECT_ROOT / args.render_output_dir).resolve()

    metadata_payload = load_json(metadata_path)
    metadata_items = metadata_payload.get("items")
    if not isinstance(metadata_items, list) or not metadata_items:
        raise RuntimeError("Metadata file does not contain any indexed images.")
    for index, item in enumerate(metadata_items):
        if not isinstance(item, dict):
            raise RuntimeError("Metadata items must be objects.")
        item.setdefault("memory_index", index)

    try:
        image_embeddings = load_image_index_artifacts(index_path)
        embedder = SigLIP2Embedder(
            model_name=resolve_siglip2_model_name(args.embedding_model),
            device=args.device,
            batch_size=args.batch_size,
        )
    except MissingDependencyError as exc:
        raise RuntimeError(str(exc)) from exc

    if len(metadata_items) != int(image_embeddings.shape[0]):
        raise RuntimeError("Index artifacts and metadata item count do not match.")

    use_faiss = faiss_path.exists() and not args.no_faiss
    if use_faiss:
        try:
            image_index = load_faiss_index(faiss_path)
        except MissingDependencyError as exc:
            raise RuntimeError(str(exc)) from exc
    else:
        image_index = None

    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    renderer = PanoramaRenderer(
        pano_graph,
        image_timeout=args.render_timeout,
        rng=random.Random(args.render_seed),
    )

    emit_progress(
        f"[demo] pano={args.pano_id} query_views={args.query_view_count} selection={args.query_selection} "
        f"top_k={args.retrieval_top_k} faiss={'yes' if use_faiss else 'no'}"
    )
    manifest_path = ensure_manifest(
        renderer=renderer,
        artifacts_dir=artifacts_dir,
        render_api_key=args.render_api_key,
        render_output_dir=render_output_dir,
        pano_id=args.pano_id,
        heading_mode=args.heading_mode,
        pitch=args.pitch,
        fov=args.fov,
        width=args.width,
        height=args.height,
    )
    _, captures = load_manifest_captures(manifest_path, max_captures=max(args.max_captures, 1))
    query_captures = select_query_capture_records(
        [
            {
                "capture_index": index,
                "capture_label": capture.get("label"),
                "capture_heading": capture.get("heading"),
                "capture_path": capture["path"],
            }
            for index, capture in enumerate(captures)
        ],
        query_view_count=args.query_view_count,
        selection=args.query_selection,
        seed=args.query_seed,
    )
    query_paths = [capture["capture_path"] for capture in query_captures]
    emit_progress(f"[demo] encode selected_images={len(query_paths)}")
    query_embeddings = embedder.encode_image_paths(query_paths)

    excluded_indices = {
        int(item["memory_index"]) for item in metadata_items if args.exclude_same_pano and item.get("pano_id") == args.pano_id
    }
    raw_top_k = min(len(metadata_items), max(args.retrieval_top_k * max(len(query_paths), 1), args.retrieval_top_k) + len(excluded_indices))

    best_candidates: dict[int, dict] = {}
    for query_capture, query_embedding in zip(query_captures, query_embeddings, strict=False):
        if use_faiss:
            scores, indices = search_image_index(image_index, query_embedding, top_k=max(raw_top_k, args.retrieval_top_k))
            ranked = [
                (int(candidate_index), float(score))
                for score, candidate_index in zip(scores.tolist(), indices.tolist())
                if int(candidate_index) >= 0 and int(candidate_index) not in excluded_indices
            ]
        else:
            ranked = brute_force_search(
                image_embeddings,
                query_embedding,
                top_k=max(raw_top_k, args.retrieval_top_k),
                exclude_indices=excluded_indices,
            )

        for candidate_index, score in ranked[: max(args.retrieval_top_k, 1)]:
            candidate_meta = metadata_items[candidate_index]
            if not is_valid_room_id(candidate_meta.get("room_id")):
                continue
            candidate_record = {
                "candidate_index": candidate_index,
                "candidate_pano_id": candidate_meta["pano_id"],
                "candidate_capture_index": int(candidate_meta.get("capture_index", 0)),
                "candidate_capture_label": candidate_meta.get("capture_label"),
                "candidate_capture_path": candidate_meta.get("capture_path"),
                "room_id": candidate_meta["room_id"],
                "score": max(float(score), 0.0),
                "matched_query_capture_index": int(query_capture.get("capture_index", 0)),
                "matched_query_capture_label": query_capture.get("capture_label"),
            }
            existing = best_candidates.get(candidate_index)
            if existing is None or candidate_record["score"] > existing["score"]:
                best_candidates[candidate_index] = candidate_record

    scored_candidates = sorted(
        best_candidates.values(),
        key=lambda record: (-record["score"], record["candidate_pano_id"], record["candidate_capture_index"]),
    )
    if args.dedup_by_pano:
        scored_candidates = deduplicate_candidates_by_pano(scored_candidates)
    top_candidates = scored_candidates[: max(args.retrieval_top_k, 1)]

    predicted_room_id, confidence, room_scores = predict_room_from_candidates(top_candidates)
    room_distribution = normalize_distribution(room_scores)
    payload = {
        "query": {
            "pano_id": args.pano_id,
            "manifest_path": str(manifest_path),
            "query_view_count": len(query_captures),
            "query_selection": args.query_selection,
            "dedup_by_pano": bool(args.dedup_by_pano),
            "exclude_same_pano": bool(args.exclude_same_pano),
            "query_captures": query_captures,
        },
        "prediction": {
            "predicted_room_id": predicted_room_id,
            "confidence": confidence,
            "room_scores": room_scores,
            "room_distribution": room_distribution,
        },
        "top_candidates": top_candidates,
    }

    if args.output_path:
        output_path = (PROJECT_ROOT / args.output_path).resolve()
        write_json(output_path, payload)

    if args.full_output:
        stdout_payload = payload
    else:
        preview_count = max(args.preview_candidates, 0)
        stdout_payload = {
            "query": {
                "pano_id": payload["query"]["pano_id"],
                "query_view_count": payload["query"]["query_view_count"],
                "query_selection": payload["query"]["query_selection"],
                "dedup_by_pano": payload["query"]["dedup_by_pano"],
                "exclude_same_pano": payload["query"]["exclude_same_pano"],
                "query_image_paths": [capture["capture_path"] for capture in payload["query"]["query_captures"]],
            },
            "prediction": {
                "predicted_room_id": payload["prediction"]["predicted_room_id"],
                "confidence": payload["prediction"]["confidence"],
                "room_distribution": payload["prediction"]["room_distribution"],
            },
            "top_k_images": [
                {
                    "candidate_pano_id": candidate["candidate_pano_id"],
                    "room_id": candidate["room_id"],
                    "score": candidate["score"],
                    "candidate_image_path": candidate.get("candidate_capture_path"),
                }
                for candidate in payload["top_candidates"][:preview_count]
            ],
        }

    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
