from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav import PanoramaRenderer, get_env_value, load_dotenv
from memory_nav.data.memory_localization import (
    DEFAULT_EMBEDDING_MODEL,
    MissingDependencyError,
    brute_force_search,
    create_image_embedder,
    deduplicate_candidates_by_pano,
    group_metadata_items_by_pano,
    is_valid_room_id,
    load_faiss_index,
    load_image_index_artifacts,
    load_json,
    predict_room_from_candidates,
    resolve_embedding_model_name,
    search_image_index,
    select_query_capture_records,
)


load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate image-memory localization with pano-level queries.")
    parser.add_argument("--index-path", default="artifacts/memory_localization/floor0_siglip2_images.npz")
    parser.add_argument(
        "--metadata-path",
        default="artifacts/memory_localization/floor0_siglip2_images.metadata.json",
    )
    parser.add_argument("--faiss-path", default="artifacts/memory_localization/floor0_siglip2_images.faiss")
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--query-view-count", type=int, default=8)
    parser.add_argument(
        "--query-view-counts",
        help="Comma-separated query image counts, e.g. 1,2,3,4. Overrides --query-view-count.",
    )
    parser.add_argument("--query-selection", choices=["evenly-spaced", "first", "random"], default="evenly-spaced")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument(
        "--query-random-seeds",
        help="Comma-separated seeds for repeated sampling. Defaults to --query-seed.",
    )
    parser.add_argument(
        "--query-render-mode",
        choices=["index-captures", "rerender"],
        default="index-captures",
        help="Use indexed captures as query embeddings or render a separate query image set.",
    )
    parser.add_argument("--query-output-dir", default="renders/memory_localization_eval_queries")
    parser.add_argument(
        "--query-render-api-key",
        default=get_env_value("GMAPS_KEY", "NAV_GMAPS_KEY", "GMAPS_API_KEY"),
    )
    parser.add_argument("--query-render-seed", type=int, default=0)
    parser.add_argument("--query-render-capture-count", type=int, default=8)
    parser.add_argument("--query-render-fov", type=int, default=90)
    parser.add_argument("--query-render-width", type=int, default=512)
    parser.add_argument("--query-render-height", type=int, default=512)
    parser.add_argument("--query-render-pitch", type=float, default=0.0)
    parser.add_argument("--query-render-timeout", type=float, default=60.0)
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--embedding-model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--dedup-by-pano", action="store_true")
    parser.add_argument("--include-same-pano", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--hard-example-limit", type=int, default=25)
    parser.add_argument("--output-path")
    return parser


def emit_progress(message: str) -> None:
    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def format_duration(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def parse_int_list(value: str | None, *, default: list[int]) -> list[int]:
    if not isinstance(value, str) or not value.strip():
        return list(default)
    parsed = []
    for part in value.split(","):
        cleaned = part.strip()
        if cleaned:
            parsed.append(int(cleaned))
    return parsed or list(default)


def normalize_distribution(room_scores: dict[str, float]) -> dict[str, float]:
    total = sum(max(float(score), 0.0) for score in room_scores.values())
    if total <= 0.0:
        return {}
    return {
        room_id: max(float(score), 0.0) / total
        for room_id, score in room_scores.items()
    }


def distribution_margin(room_distribution: dict[str, float]) -> float:
    values = sorted((float(value) for value in room_distribution.values()), reverse=True)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return max(0.0, values[0] - values[1])


def build_query_captures(*, count: int, seed: int) -> list[tuple[str, float]]:
    rng = random.Random(seed)
    return [
        (f"query{index:02d}", rng.random() * 360.0)
        for index in range(max(int(count), 1))
    ]


def render_query_manifest(
    *,
    renderer: PanoramaRenderer,
    artifacts_dir: Path,
    render_api_key: str | None,
    query_output_dir: Path,
    pano_id: str,
    capture_count: int,
    seed: int,
    pitch: float,
    fov: int,
    width: int,
    height: int,
) -> Path:
    if not render_api_key:
        raise RuntimeError("Missing GMAPS_KEY or GMAPS_API_KEY to rerender query pano views.")
    manifest = renderer.render(
        pano_id=pano_id,
        api_key=render_api_key,
        output_dir=str(query_output_dir),
        heading_mode="custom",
        custom_captures=build_query_captures(count=capture_count, seed=seed),
        pitch=pitch,
        fov=fov,
        width=width,
        height=height,
        graph_path=str(artifacts_dir / "pano_graph.json"),
    )
    return Path(str(manifest["manifest_path"])).resolve()


def rank_query_embedding(
    *,
    query_embedding,
    image_embeddings,
    image_index,
    metadata_items: list[dict],
    retrieval_top_k: int,
    excluded_indices: set[int],
) -> list[tuple[int, float]]:
    raw_top_k = min(len(metadata_items), retrieval_top_k + len(excluded_indices))
    if image_index is not None:
        scores, indices = search_image_index(image_index, query_embedding, top_k=max(raw_top_k, retrieval_top_k))
        return [
            (int(candidate_index), float(score))
            for score, candidate_index in zip(scores.tolist(), indices.tolist())
            if int(candidate_index) >= 0 and int(candidate_index) not in excluded_indices
        ]
    return brute_force_search(
        image_embeddings,
        query_embedding,
        top_k=max(raw_top_k, retrieval_top_k),
        exclude_indices=excluded_indices,
    )


def query_embeddings_for_captures(
    *,
    query_captures: list[dict],
    image_embeddings,
    query_render_mode: str,
    embedder,
    embedding_cache: dict[str, object],
):
    if query_render_mode == "index-captures":
        return [
            (query_capture, image_embeddings[int(query_capture["memory_index"])])
            for query_capture in query_captures
        ]

    pairs = []
    missing_paths = [
        str(query_capture.get("capture_path"))
        for query_capture in query_captures
        if str(query_capture.get("capture_path")) not in embedding_cache
    ]
    if missing_paths:
        encoded = embedder.encode_image_paths(missing_paths)
        for path, embedding in zip(missing_paths, encoded, strict=False):
            embedding_cache[path] = embedding
    for query_capture in query_captures:
        path = str(query_capture.get("capture_path"))
        pairs.append((query_capture, embedding_cache[path]))
    return pairs


def evaluate_trial(
    *,
    pano_groups: list[dict],
    metadata_items: list[dict],
    image_embeddings,
    image_index,
    retrieval_top_k: int,
    query_view_count: int,
    query_seed: int,
    query_selection: str,
    query_render_mode: str,
    include_same_pano: bool,
    dedup_by_pano: bool,
    limit: int,
    embedder=None,
    embedding_cache: dict[str, object] | None = None,
) -> list[dict]:
    embedding_cache = embedding_cache if embedding_cache is not None else {}
    results: list[dict] = []
    for query_offset, query_group in enumerate(pano_groups[:limit]):
        query_pano_id = query_group["pano_id"]
        query_room_id = query_group["room_id"]
        query_captures = select_query_capture_records(
            query_group["captures"],
            query_view_count=query_view_count,
            selection=query_selection,
            seed=query_seed + query_offset,
        )
        if not query_captures:
            continue

        excluded_indices = set()
        if not include_same_pano:
            memory_captures = query_group.get("memory_captures", query_group["captures"])
            excluded_indices = {
                int(capture["memory_index"])
                for capture in memory_captures
                if isinstance(capture, dict) and "memory_index" in capture
            }

        scored_candidates: list[dict] = []
        for query_capture, query_embedding in query_embeddings_for_captures(
            query_captures=query_captures,
            image_embeddings=image_embeddings,
            query_render_mode=query_render_mode,
            embedder=embedder,
            embedding_cache=embedding_cache,
        ):
            ranked = rank_query_embedding(
                query_embedding=query_embedding,
                image_embeddings=image_embeddings,
                image_index=image_index,
                metadata_items=metadata_items,
                retrieval_top_k=retrieval_top_k,
                excluded_indices=excluded_indices,
            )
            for candidate_index, score in ranked[:retrieval_top_k]:
                candidate_meta = metadata_items[candidate_index]
                if not is_valid_room_id(candidate_meta.get("room_id")):
                    continue
                scored_candidates.append(
                    {
                        "query_capture_index": int(query_capture.get("capture_index", 0)),
                        "query_capture_label": query_capture.get("capture_label"),
                        "candidate_index": candidate_index,
                        "candidate_pano_id": candidate_meta["pano_id"],
                        "candidate_capture_index": int(candidate_meta.get("capture_index", 0)),
                        "candidate_capture_label": candidate_meta.get("capture_label"),
                        "room_id": candidate_meta["room_id"],
                        "score": max(float(score), 0.0),
                    }
                )

        scored_candidates.sort(
            key=lambda record: (-record["score"], record["candidate_pano_id"], record["candidate_capture_index"])
        )
        if dedup_by_pano:
            scored_candidates = deduplicate_candidates_by_pano(scored_candidates)
        top_candidates = scored_candidates[:retrieval_top_k]
        predicted_room_id, confidence, room_scores = predict_room_from_candidates(top_candidates)
        room_distribution = normalize_distribution(room_scores)
        margin = distribution_margin(room_distribution)
        top_rooms = list(room_scores.keys())[:3]
        top_candidate = top_candidates[0] if top_candidates else {}
        top_candidate_same_pano = top_candidate.get("candidate_pano_id") == query_pano_id

        results.append(
            {
                "query_pano_id": query_pano_id,
                "query_room_id": query_room_id,
                "query_view_count": query_view_count,
                "query_seed": query_seed,
                "query_capture_count": len(query_captures),
                "query_render_mode": query_render_mode,
                "include_same_pano": bool(include_same_pano),
                "dedup_by_pano": bool(dedup_by_pano),
                "predicted_room_id": predicted_room_id,
                "confidence": confidence,
                "margin": margin,
                "raw_room_score_margin": raw_room_score_margin(room_scores),
                "is_top1_match": predicted_room_id == query_room_id,
                "is_top3_match": query_room_id in top_rooms,
                "top_candidate_same_pano": bool(top_candidate_same_pano),
                "top_rooms": top_rooms,
                "room_scores": room_scores,
                "room_distribution": room_distribution,
                "top_candidates": top_candidates,
            }
        )
    return results


def raw_room_score_margin(room_scores: dict[str, float]) -> float:
    values = list(room_scores.values())
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return max(0.0, float(values[0]) - float(values[1]))


def summarize_results(
    results: list[dict],
    *,
    retrieval_top_k: int,
    query_view_count: int,
    query_selection: str,
    include_same_pano: bool,
    dedup_by_pano: bool,
    use_faiss: bool,
    confidence_threshold: float,
    margin_threshold: float,
) -> dict:
    total = len(results)
    top1_hits = sum(1 for result in results if result["is_top1_match"])
    top3_hits = sum(1 for result in results if result["is_top3_match"])
    same_pano_hits = sum(1 for result in results if result["top_candidate_same_pano"])
    high_confidence = [
        result
        for result in results
        if float(result["confidence"]) >= confidence_threshold and float(result["margin"]) >= margin_threshold
    ]
    high_confidence_correct = sum(1 for result in high_confidence if result["is_top1_match"])
    confidences = [float(result["confidence"]) for result in results]
    margins = [float(result["margin"]) for result in results]
    return {
        "query_count": total,
        "top1_accuracy": (top1_hits / total) if total else 0.0,
        "top3_accuracy": (top3_hits / total) if total else 0.0,
        "same_pano_top_candidate_rate": (same_pano_hits / total) if total else 0.0,
        "retrieval_top_k": retrieval_top_k,
        "query_view_count": query_view_count,
        "query_selection": query_selection,
        "include_same_pano": bool(include_same_pano),
        "dedup_by_pano": bool(dedup_by_pano),
        "faiss_enabled": bool(use_faiss),
        "confidence_mean": mean(confidences) if confidences else 0.0,
        "confidence_median": median(confidences) if confidences else 0.0,
        "margin_mean": mean(margins) if margins else 0.0,
        "margin_median": median(margins) if margins else 0.0,
        "high_confidence_thresholds": {
            "confidence": confidence_threshold,
            "margin": margin_threshold,
        },
        "high_confidence_count": len(high_confidence),
        "high_confidence_correct": high_confidence_correct,
        "high_confidence_correct_rate": (high_confidence_correct / total) if total else 0.0,
        "high_confidence_precision": (high_confidence_correct / len(high_confidence)) if high_confidence else 0.0,
    }


def per_room_accuracy(results: list[dict]) -> dict:
    totals: dict[str, int] = {}
    top1_hits: dict[str, int] = {}
    top3_hits: dict[str, int] = {}
    confidence_values: dict[str, list[float]] = {}
    for result in results:
        room_id = result["query_room_id"]
        totals[room_id] = totals.get(room_id, 0) + 1
        if result["is_top1_match"]:
            top1_hits[room_id] = top1_hits.get(room_id, 0) + 1
        if result["is_top3_match"]:
            top3_hits[room_id] = top3_hits.get(room_id, 0) + 1
        confidence_values.setdefault(room_id, []).append(float(result["confidence"]))
    return {
        room_id: {
            "correct": top1_hits.get(room_id, 0),
            "top3_correct": top3_hits.get(room_id, 0),
            "total": total,
            "top1_accuracy": (top1_hits.get(room_id, 0) / total) if total > 0 else 0.0,
            "top3_accuracy": (top3_hits.get(room_id, 0) / total) if total > 0 else 0.0,
            "confidence_mean": mean(confidence_values.get(room_id, [])) if confidence_values.get(room_id) else 0.0,
        }
        for room_id, total in sorted(totals.items())
    }


def confusion_pairs(results: list[dict]) -> list[dict]:
    counts: dict[tuple[str, str], int] = {}
    for result in results:
        predicted_room_id = result.get("predicted_room_id")
        query_room_id = result.get("query_room_id")
        if isinstance(predicted_room_id, str) and predicted_room_id != query_room_id:
            key = (str(query_room_id), predicted_room_id)
            counts[key] = counts.get(key, 0) + 1
    return [
        {
            "expected_room_id": expected_room_id,
            "predicted_room_id": predicted_room_id,
            "count": count,
        }
        for (expected_room_id, predicted_room_id), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]


def hardest_errors(results: list[dict], *, limit: int) -> list[dict]:
    return [
        result
        for result in sorted(
            (record for record in results if not record["is_top1_match"]),
            key=lambda record: (-record["confidence"], record["margin"], record["query_pano_id"]),
        )[: max(limit, 0)]
    ]


def seed_summary_stats(seed_summaries: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for summary in seed_summaries:
        grouped.setdefault(str(summary["query_view_count"]), []).append(summary)
    metrics = [
        "top1_accuracy",
        "top3_accuracy",
        "same_pano_top_candidate_rate",
        "confidence_mean",
        "margin_mean",
        "high_confidence_correct_rate",
        "high_confidence_precision",
    ]
    output = {}
    for query_view_count, summaries in sorted(grouped.items(), key=lambda item: int(item[0])):
        stats = {}
        for metric in metrics:
            values = [float(summary.get(metric, 0.0)) for summary in summaries]
            stats[f"{metric}_mean"] = mean(values) if values else 0.0
            stats[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        output[query_view_count] = stats
    return output


def prepare_rerendered_query_groups(
    *,
    pano_groups: list[dict],
    limit: int,
    args: argparse.Namespace,
) -> list[dict]:
    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    renderer = PanoramaRenderer(
        pano_graph,
        image_timeout=args.query_render_timeout,
        rng=random.Random(args.query_render_seed),
    )
    query_output_dir = (PROJECT_ROOT / args.query_output_dir).resolve()
    prepared_groups = []
    for query_offset, group in enumerate(pano_groups[:limit], start=1):
        manifest_path = render_query_manifest(
            renderer=renderer,
            artifacts_dir=artifacts_dir,
            render_api_key=args.query_render_api_key,
            query_output_dir=query_output_dir,
            pano_id=group["pano_id"],
            capture_count=args.query_render_capture_count,
            seed=args.query_render_seed + query_offset,
            pitch=args.query_render_pitch,
            fov=args.query_render_fov,
            width=args.query_render_width,
            height=args.query_render_height,
        )
        _, captures = load_rerendered_captures(manifest_path)
        prepared_groups.append(
            {
                **group,
                "memory_captures": group["captures"],
                "captures": captures,
                "query_manifest_path": str(manifest_path),
            }
        )
    return prepared_groups


def load_rerendered_captures(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = load_json(manifest_path)
    captures = []
    for index, capture in enumerate(manifest.get("captures", [])):
        if not isinstance(capture, dict) or not isinstance(capture.get("path"), str):
            continue
        captures.append(
            {
                "capture_index": index,
                "capture_label": capture.get("label"),
                "capture_heading": capture.get("heading"),
                "capture_path": capture["path"],
            }
        )
    if not captures:
        raise RuntimeError(f"Manifest has no query captures: {manifest_path}")
    return manifest, captures


def main() -> int:
    args = build_parser().parse_args()
    index_path = (PROJECT_ROOT / args.index_path).resolve()
    metadata_path = (PROJECT_ROOT / args.metadata_path).resolve()
    faiss_path = (PROJECT_ROOT / args.faiss_path).resolve()

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
    except MissingDependencyError as exc:
        raise RuntimeError(str(exc)) from exc

    if len(metadata_items) != int(image_embeddings.shape[0]):
        raise RuntimeError("Index artifacts and metadata item count do not match.")

    pano_groups = group_metadata_items_by_pano(metadata_items)
    if not pano_groups:
        raise RuntimeError("Metadata does not contain any pano groups.")

    use_faiss = faiss_path.exists() and not args.no_faiss

    limit = min(max(args.limit or len(pano_groups), 0), len(pano_groups))
    retrieval_top_k = max(args.retrieval_top_k, 1)
    query_view_counts = parse_int_list(args.query_view_counts, default=[args.query_view_count])
    query_random_seeds = parse_int_list(args.query_random_seeds, default=[args.query_seed])
    metadata_summary = metadata_payload.get("summary") if isinstance(metadata_payload.get("summary"), dict) else {}
    embedding_model = resolve_embedding_model_name(
        args.embedding_model or metadata_summary.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
    )
    started_at = time.time()

    emit_progress(
        f"[image-memory-eval] queries={limit} retrieval_top_k={retrieval_top_k} "
        f"query_views={query_view_counts} seeds={query_random_seeds} "
        f"selection={args.query_selection} render_mode={args.query_render_mode} "
        f"include_same_pano={'yes' if args.include_same_pano else 'no'} faiss={'yes' if use_faiss else 'no'}"
    )

    eval_groups = pano_groups
    embedder = None
    embedding_cache: dict[str, object] = {}
    if args.query_render_mode == "rerender":
        try:
            embedder = create_image_embedder(
                model_name=embedding_model,
                device=args.device,
                batch_size=args.batch_size,
            )
        except MissingDependencyError as exc:
            raise RuntimeError(str(exc)) from exc
        eval_groups = prepare_rerendered_query_groups(
            pano_groups=pano_groups,
            limit=limit,
            args=args,
        )

    if use_faiss:
        try:
            image_index = load_faiss_index(faiss_path)
        except MissingDependencyError as exc:
            raise RuntimeError(str(exc)) from exc
    else:
        image_index = None

    all_results: list[dict] = []
    seed_summaries: list[dict] = []
    for query_view_count in query_view_counts:
        for query_seed in query_random_seeds:
            trial_results = evaluate_trial(
                pano_groups=eval_groups,
                metadata_items=metadata_items,
                image_embeddings=image_embeddings,
                image_index=image_index,
                retrieval_top_k=retrieval_top_k,
                query_view_count=query_view_count,
                query_seed=query_seed,
                query_selection=args.query_selection,
                query_render_mode=args.query_render_mode,
                include_same_pano=args.include_same_pano,
                dedup_by_pano=args.dedup_by_pano,
                limit=limit,
                embedder=embedder,
                embedding_cache=embedding_cache,
            )
            seed_summaries.append(
                {
                    "query_view_count": query_view_count,
                    "query_seed": query_seed,
                    **summarize_results(
                        trial_results,
                        retrieval_top_k=retrieval_top_k,
                        query_view_count=query_view_count,
                        query_selection=args.query_selection,
                        include_same_pano=args.include_same_pano,
                        dedup_by_pano=args.dedup_by_pano,
                        use_faiss=use_faiss,
                        confidence_threshold=args.confidence_threshold,
                        margin_threshold=args.margin_threshold,
                    ),
                }
            )
            all_results.extend(trial_results)

    results_by_view_count = {
        str(query_view_count): [
            result for result in all_results if int(result["query_view_count"]) == query_view_count
        ]
        for query_view_count in query_view_counts
    }
    view_count_summaries = {
        str(query_view_count): summarize_results(
            results,
            retrieval_top_k=retrieval_top_k,
            query_view_count=int(query_view_count),
            query_selection=args.query_selection,
            include_same_pano=args.include_same_pano,
            dedup_by_pano=args.dedup_by_pano,
            use_faiss=use_faiss,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
        )
        for query_view_count, results in results_by_view_count.items()
    }

    summary = {
        **summarize_results(
            all_results,
            retrieval_top_k=retrieval_top_k,
            query_view_count=query_view_counts[0] if len(query_view_counts) == 1 else 0,
            query_selection=args.query_selection,
            include_same_pano=args.include_same_pano,
            dedup_by_pano=args.dedup_by_pano,
            use_faiss=use_faiss,
            confidence_threshold=args.confidence_threshold,
            margin_threshold=args.margin_threshold,
        ),
        "index_image_count": len(metadata_items),
        "index_pano_count": len(pano_groups),
        "embedding_model": embedding_model,
        "query_view_counts": query_view_counts,
        "query_random_seeds": query_random_seeds,
        "query_render_mode": args.query_render_mode,
        "elapsed": format_duration(time.time() - started_at),
    }
    payload = {
        "summary": summary,
        "view_count_summaries": view_count_summaries,
        "seed_summaries": seed_summaries,
        "seed_summary_stats_by_view_count": seed_summary_stats(seed_summaries),
        "per_room_accuracy_by_view_count": {
            query_view_count: per_room_accuracy(results)
            for query_view_count, results in results_by_view_count.items()
        },
        "confusion_pairs_by_view_count": {
            query_view_count: confusion_pairs(results)
            for query_view_count, results in results_by_view_count.items()
        },
        "hardest_errors_by_view_count": {
            query_view_count: hardest_errors(results, limit=args.hard_example_limit)
            for query_view_count, results in results_by_view_count.items()
        },
        "results": all_results,
    }

    if args.output_path:
        output_path = (PROJECT_ROOT / args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    stdout_payload = {
        "summary": payload["summary"],
        "view_count_summaries": payload["view_count_summaries"],
        "seed_summary_stats_by_view_count": payload["seed_summary_stats_by_view_count"],
        "per_room_accuracy_by_view_count": payload["per_room_accuracy_by_view_count"],
        "confusion_pairs_by_view_count": payload["confusion_pairs_by_view_count"],
        "hardest_errors_by_view_count": payload["hardest_errors_by_view_count"],
    }
    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
