from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.data.memory_localization import (
    deduplicate_candidates_by_pano,
    MissingDependencyError,
    brute_force_search,
    group_metadata_items_by_pano,
    is_valid_room_id,
    load_faiss_index,
    load_image_index_artifacts,
    load_json,
    predict_room_from_candidates,
    search_image_index,
    select_query_capture_records,
)


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
    parser.add_argument("--query-selection", choices=["evenly-spaced", "first", "random"], default="evenly-spaced")
    parser.add_argument("--query-seed", type=int, default=0)
    parser.add_argument("--retrieval-top-k", type=int, default=5)
    parser.add_argument("--dedup-by-pano", action="store_true")
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
    if use_faiss:
        try:
            image_index = load_faiss_index(faiss_path)
        except MissingDependencyError as exc:
            raise RuntimeError(str(exc)) from exc
    else:
        image_index = None

    limit = min(max(args.limit or len(pano_groups), 0), len(pano_groups))
    retrieval_top_k = max(args.retrieval_top_k, 1)
    started_at = time.time()

    emit_progress(
        f"[image-memory-eval] queries={limit} retrieval_top_k={retrieval_top_k} "
        f"query_views={args.query_view_count} selection={args.query_selection} faiss={'yes' if use_faiss else 'no'}"
    )

    per_room_totals: dict[str, int] = {}
    per_room_hits: dict[str, int] = {}
    confusion_counts: dict[tuple[str, str], int] = {}
    results: list[dict] = []
    matched_top1 = 0
    matched_top3 = 0

    for query_offset, query_group in enumerate(pano_groups[:limit]):
        query_pano_id = query_group["pano_id"]
        query_room_id = query_group["room_id"]
        query_captures = select_query_capture_records(
            query_group["captures"],
            query_view_count=args.query_view_count,
            selection=args.query_selection,
            seed=args.query_seed + query_offset,
        )
        if not query_captures:
            continue

        excluded_indices = {int(capture["memory_index"]) for capture in query_group["captures"]}
        per_room_totals[query_room_id] = per_room_totals.get(query_room_id, 0) + 1
        scored_candidates: list[dict] = []

        for query_capture in query_captures:
            query_memory_index = int(query_capture["memory_index"])
            query_embedding = image_embeddings[query_memory_index]
            raw_top_k = min(len(metadata_items), retrieval_top_k + len(excluded_indices))
            if use_faiss:
                scores, indices = search_image_index(image_index, query_embedding, top_k=max(raw_top_k, retrieval_top_k))
                ranked = [
                    (int(candidate_index), float(score))
                    for score, candidate_index in zip(scores.tolist(), indices.tolist())
                    if int(candidate_index) >= 0 and int(candidate_index) not in excluded_indices
                ]
            else:
                ranked = brute_force_search(
                    image_embeddings,
                    query_embedding,
                    top_k=max(raw_top_k, retrieval_top_k),
                    exclude_indices=excluded_indices,
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
        if args.dedup_by_pano:
            scored_candidates = deduplicate_candidates_by_pano(scored_candidates)
        predicted_room_id, confidence, room_scores = predict_room_from_candidates(scored_candidates)
        top_rooms = list(room_scores.keys())[:3]
        is_top1 = predicted_room_id == query_room_id
        is_top3 = query_room_id in top_rooms
        if is_top1:
            matched_top1 += 1
            per_room_hits[query_room_id] = per_room_hits.get(query_room_id, 0) + 1
        if is_top3:
            matched_top3 += 1
        if isinstance(predicted_room_id, str) and predicted_room_id != query_room_id:
            key = (query_room_id, predicted_room_id)
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

        sorted_room_scores = list(room_scores.items())
        margin = 0.0
        if sorted_room_scores:
            margin = sorted_room_scores[0][1]
            if len(sorted_room_scores) > 1:
                margin -= sorted_room_scores[1][1]

        results.append(
            {
                "query_pano_id": query_pano_id,
                "query_room_id": query_room_id,
                "query_capture_count": len(query_captures),
                "dedup_by_pano": bool(args.dedup_by_pano),
                "predicted_room_id": predicted_room_id,
                "confidence": confidence,
                "is_top1_match": is_top1,
                "is_top3_match": is_top3,
                "margin": margin,
                "top_rooms": top_rooms,
                "room_scores": room_scores,
                "top_candidates": scored_candidates[:retrieval_top_k],
            }
        )

    per_room_accuracy = {
        room_id: {
            "correct": per_room_hits.get(room_id, 0),
            "total": total,
            "top1_accuracy": (per_room_hits.get(room_id, 0) / total) if total > 0 else 0.0,
        }
        for room_id, total in sorted(per_room_totals.items())
    }
    hardest_errors = [
        result
        for result in sorted(
            (record for record in results if not record["is_top1_match"]),
            key=lambda record: (record["margin"], record["query_pano_id"]),
        )[: max(args.hard_example_limit, 0)]
    ]
    confusion_pairs = [
        {
            "expected_room_id": expected_room_id,
            "predicted_room_id": predicted_room_id,
            "count": count,
        }
        for (expected_room_id, predicted_room_id), count in sorted(
            confusion_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]

    summary = {
        "query_count": len(results),
        "index_image_count": len(metadata_items),
        "index_pano_count": len(pano_groups),
        "top1_accuracy": (matched_top1 / len(results)) if results else 0.0,
        "top3_accuracy": (matched_top3 / len(results)) if results else 0.0,
        "retrieval_top_k": retrieval_top_k,
        "query_view_count": args.query_view_count,
        "query_selection": args.query_selection,
        "dedup_by_pano": bool(args.dedup_by_pano),
        "faiss_enabled": use_faiss,
        "elapsed": format_duration(time.time() - started_at),
    }
    payload = {
        "summary": summary,
        "per_room_accuracy": per_room_accuracy,
        "confusion_pairs": confusion_pairs,
        "hardest_errors": hardest_errors,
        "results": results,
    }

    if args.output_path:
        output_path = (PROJECT_ROOT / args.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    stdout_payload = {
        "summary": payload["summary"],
        "per_room_accuracy": payload["per_room_accuracy"],
        "confusion_pairs": payload["confusion_pairs"],
        "hardest_errors": payload["hardest_errors"],
    }
    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
