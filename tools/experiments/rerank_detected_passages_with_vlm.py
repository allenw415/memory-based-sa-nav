from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
from base64 import b64encode
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.common.env import (  # noqa: E402
    get_env_value,
    load_dotenv,
    resolve_model_environment,
)
from memory_nav.common.model_client import (  # noqa: E402
    DEFAULT_GEMINI_API_BASE,
    ModelResponseClient,
    parse_json_output,
)
from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_DINOV2_SALAD_MODEL,
    create_image_embedder,
    normalize_rows,
    write_json,
)


DEFAULT_INPUT_ROOT = (
    "outputs/passage_selection/"
    "passage_detection_salad_strict_prompt_v1_neg05_gdino_base"
)
DEFAULT_OUTPUT_ROOT = "outputs/passage_selection_vlm/gemma4_31b"
DEFAULT_MODEL = "gemma-4-31b-it"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate detected-contrastive passage candidates, then use Gemma 4 "
            "to filter noise and rerank one directed room transition."
        )
    )
    parser.add_argument("--current-room-id", required=True)
    parser.add_argument("--target-room-id", required=True)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--current-result")
    parser.add_argument(
        "--target-result",
        help=(
            "Target-room reference result. Defaults to the reverse directed edge "
            "(target room -> current room) under --input-root."
        ),
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-current", type=int, default=8)
    parser.add_argument("--max-target", type=int, default=8)
    parser.add_argument("--dedup-pool-size", type=int, default=32)
    parser.add_argument("--dedup-similarity-threshold", type=float, default=0.75)
    parser.add_argument("--dedup-model", default=DEFAULT_DINOV2_SALAD_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--profile", default="gemini")
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key")
    parser.add_argument("--api-base", default=DEFAULT_GEMINI_API_BASE)
    parser.add_argument("--api-kind", default="responses")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "high"],
        default="low",
        help="Gemma 4 thinking level; low maps to minimal thinking.",
    )
    parser.add_argument(
        "--dry-run-request",
        action="store_true",
        help="Prepare candidates and request_summary.json without calling Gemini.",
    )
    parser.add_argument(
        "--reuse-existing-response",
        action="store_true",
        help="Reuse output_dir/raw_response.json instead of calling Gemini.",
    )
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def result_edge(payload: dict) -> tuple[str, str]:
    configuration = payload.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Passage result does not contain a configuration object.")
    current_room_id = configuration.get("current_room_id")
    target_room_id = configuration.get("subgoal_room_id")
    if not isinstance(current_room_id, str) or not isinstance(target_room_id, str):
        raise ValueError("Passage result does not identify current/subgoal rooms.")
    return current_room_id, target_room_id


def index_result_paths(input_root: str | Path) -> dict[tuple[str, str], Path]:
    root = resolve_project_path(input_root)
    indexed: dict[tuple[str, str], Path] = {}
    for result_path in sorted(root.glob("*/result.json")):
        payload = load_json(result_path)
        edge = result_edge(payload)
        if edge in indexed:
            raise ValueError(f"Duplicate passage result for {edge}: {result_path}")
        indexed[edge] = result_path.resolve()
    if not indexed:
        raise FileNotFoundError(f"No passage result JSON files found under {root}")
    return indexed


def resolve_edge_results(args: argparse.Namespace) -> tuple[Path, Path]:
    indexed = index_result_paths(args.input_root)
    requested_edge = (args.current_room_id, args.target_room_id)
    reverse_edge = (args.target_room_id, args.current_room_id)
    current_result = (
        resolve_project_path(args.current_result)
        if args.current_result
        else indexed.get(requested_edge)
    )
    target_result = (
        resolve_project_path(args.target_result)
        if args.target_result
        else indexed.get(reverse_edge)
    )
    if current_result is None:
        raise FileNotFoundError(f"No detected passage result for {requested_edge}")
    if target_result is None:
        raise FileNotFoundError(f"No reverse-edge target passage result for {reverse_edge}")
    return current_result, target_result


def passage_ranking(payload: dict) -> list[dict]:
    passage_choice = payload.get("passage_choice")
    ranking = passage_choice.get("passage_ranking") if isinstance(passage_choice, dict) else None
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("Passage result contains no passage_choice.passage_ranking.")
    return [dict(item) for item in ranking if isinstance(item, dict)]


def _candidate_path(candidate: dict, *keys: str) -> Path | None:
    for key in keys:
        value = candidate.get(key)
        if isinstance(value, str) and value:
            path = resolve_project_path(value)
            if path.exists():
                return path
    source_passage = candidate.get("source_passage")
    if isinstance(source_passage, dict):
        value = source_passage.get("image_path")
        if isinstance(value, str) and value:
            path = resolve_project_path(value)
            if path.exists():
                return path
    return None


def _candidate_label(candidate: dict) -> str:
    return str(candidate.get("label") or candidate.get("source_label") or "candidate")


def _candidate_rank(candidate: dict) -> int:
    try:
        return int(candidate.get("rank"))
    except (TypeError, ValueError):
        return 10**9


def collapse_same_source_candidates(
    ranking: Sequence[dict],
    *,
    pool_size: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    representatives: list[dict] = []
    representative_by_source: dict[str, dict] = {}
    decisions: list[dict] = []
    skipped: list[dict] = []

    for raw_candidate in sorted(ranking, key=_candidate_rank):
        candidate = dict(raw_candidate)
        source_path = _candidate_path(candidate, "source_image_path")
        comparison_path = _candidate_path(
            candidate,
            "comparison_image_path",
            "masked_image_path",
            "crop_image_path",
        )
        if source_path is None or comparison_path is None:
            skipped.append(
                {
                    "label": _candidate_label(candidate),
                    "reason": "missing_source_or_comparison_image",
                }
            )
            continue

        source_key = str(source_path)
        representative = representative_by_source.get(source_key)
        if representative is not None:
            representative.setdefault("same_source_duplicate_labels", []).append(
                _candidate_label(candidate)
            )
            decisions.append(
                {
                    "candidate_label": _candidate_label(candidate),
                    "action": "merged_same_source",
                    "representative_label": _candidate_label(representative),
                    "source_image_path": str(source_path),
                }
            )
            continue

        candidate["source_image_path"] = str(source_path)
        candidate["comparison_image_path"] = str(comparison_path)
        candidate["same_source_duplicate_labels"] = []
        representative_by_source[source_key] = candidate
        representatives.append(candidate)
        decisions.append(
            {
                "candidate_label": _candidate_label(candidate),
                "action": "kept_source_representative",
                "source_image_path": str(source_path),
            }
        )

    return representatives[: max(int(pool_size), 1)], decisions, skipped


def deduplicate_candidates_by_similarity(
    candidates: Sequence[dict],
    embeddings,
    *,
    threshold: float,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    import numpy as np

    if not candidates:
        return [], []
    matrix = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    if matrix.ndim != 2 or int(matrix.shape[0]) != len(candidates):
        raise ValueError("Dedup embeddings do not match candidate count.")

    kept_indices: list[int] = []
    decisions: list[dict] = []
    merged_by_representative: dict[int, list[dict]] = {}
    limit = max(int(top_k), 1)

    for candidate_index, candidate in enumerate(candidates):
        if kept_indices:
            similarities = [
                (float(matrix[candidate_index] @ matrix[kept_index]), kept_index)
                for kept_index in kept_indices
            ]
            best_similarity, best_index = max(similarities, key=lambda item: item[0])
        else:
            best_similarity, best_index = -1.0, -1

        if best_index >= 0 and best_similarity >= float(threshold):
            member = {
                "label": _candidate_label(candidate),
                "similarity": float(best_similarity),
                "source_image_path": candidate.get("source_image_path"),
                "comparison_image_path": candidate.get("comparison_image_path"),
            }
            merged_by_representative.setdefault(best_index, []).append(member)
            decisions.append(
                {
                    "candidate_label": _candidate_label(candidate),
                    "action": "merged_visual_duplicate",
                    "representative_label": _candidate_label(candidates[best_index]),
                    "similarity": float(best_similarity),
                }
            )
            continue

        kept_indices.append(candidate_index)
        decisions.append(
            {
                "candidate_label": _candidate_label(candidate),
                "action": "kept_visual_representative",
                "max_similarity_to_previous_kept": (
                    None if best_index < 0 else float(best_similarity)
                ),
            }
        )
        if len(kept_indices) >= limit:
            break

    selected = []
    for candidate_index in kept_indices:
        candidate = dict(candidates[candidate_index])
        candidate["visual_duplicate_members"] = merged_by_representative.get(
            candidate_index,
            [],
        )
        selected.append(candidate)
    return selected, decisions


def prepare_candidates(
    payload: dict,
    *,
    embedder,
    max_candidates: int,
    pool_size: int,
    similarity_threshold: float,
) -> tuple[list[dict], dict]:
    source_candidates, source_decisions, skipped = collapse_same_source_candidates(
        passage_ranking(payload),
        pool_size=pool_size,
    )
    if not source_candidates:
        raise ValueError("No usable source-level passage candidates remain after validation.")
    embeddings = embedder.encode_image_paths(
        [candidate["comparison_image_path"] for candidate in source_candidates]
    )
    selected, similarity_decisions = deduplicate_candidates_by_similarity(
        source_candidates,
        embeddings,
        threshold=similarity_threshold,
        top_k=max_candidates,
    )
    if not selected:
        raise ValueError("No passage candidates remain after visual deduplication.")
    return selected, {
        "input_detected_candidate_count": len(passage_ranking(payload)),
        "source_representative_pool_count": len(source_candidates),
        "selected_count": len(selected),
        "pool_size": max(int(pool_size), 1),
        "similarity_threshold": float(similarity_threshold),
        "source_decisions": source_decisions,
        "similarity_decisions": similarity_decisions,
        "skipped_candidates": skipped,
    }


def assign_public_labels(
    candidates: Sequence[dict],
    *,
    prefix: str,
) -> list[dict]:
    labeled = []
    for index, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item["public_label"] = f"{prefix}{index}"
        labeled.append(item)
    return labeled


def image_to_data_url(image_path: str | Path) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    return (
        f"data:{mime_type or 'image/png'};base64,"
        f"{b64encode(path.read_bytes()).decode('ascii')}"
    )


def response_schema(current_labels: Sequence[str], target_labels: Sequence[str]) -> dict:
    current_enum = list(current_labels)
    target_enum = list(target_labels)
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
                        "label": {"type": "string", "enum": current_enum},
                        "status": {
                            "type": "string",
                            "enum": ["valid_passage", "ambiguous", "noise"],
                        },
                        "passage_confidence": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "label",
                        "status",
                        "passage_confidence",
                        "reason",
                    ],
                },
            },
            "valid_passage_labels": {
                "type": "array",
                "items": {"type": "string", "enum": current_enum},
            },
            "invalid_or_noisy_labels": {
                "type": "array",
                "items": {"type": "string", "enum": current_enum},
            },
            "chosen_label": {
                "anyOf": [
                    {"type": "string", "enum": current_enum},
                    {"type": "null"},
                ]
            },
            "navigation_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "target_reference_labels_used": {
                "type": "array",
                "items": {"type": "string", "enum": target_enum},
            },
            "why_this_passage": {"type": "string"},
            "why_not_others": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": current_enum},
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "reason"],
                },
            },
        },
        "required": [
            "candidate_assessments",
            "valid_passage_labels",
            "invalid_or_noisy_labels",
            "chosen_label",
            "navigation_confidence",
            "target_reference_labels_used",
            "why_this_passage",
            "why_not_others",
        ],
    }


def build_vlm_request(
    *,
    model: str,
    detail: str,
    reasoning_effort: str,
    current_room_id: str,
    target_room_id: str,
    current_candidates: Sequence[dict],
    target_references: Sequence[dict],
) -> dict:
    current_labels = [str(item["public_label"]) for item in current_candidates]
    target_labels = [str(item["public_label"]) for item in target_references]
    content: list[dict] = [
        {
            "type": "input_text",
            "text": "\n".join(
                [
                    f"Current room: {current_room_id}",
                    f"Immediate target room: {target_room_id}",
                    f"Choose only from current candidate labels: {current_labels}",
                    f"Target reference labels, never choose these: {target_labels}",
                    "",
                    "Every image is an original full scene. The current candidates were produced by a doorway detector and similarity ranking, but some may still be exhibits, walls, internal gallery views, or incorrect exits.",
                    "Classify every current candidate as valid_passage, ambiguous, or noise before choosing.",
                    "A valid passage must show a physically walkable doorway, archway, portal, wide opening, threshold, or clear route crossing out of the current room.",
                    "The target references come from the reverse directed edge: they are passage candidates viewed from inside the target room toward the current room. They may show the same room boundary from the opposite side, but they may also contain noisy or alternative exits.",
                    "Use target references as supporting visual context, not as guaranteed ground truth and not as labels to choose.",
                    "Choose the current opening most likely to cross directly into the target room. Prefer architectural and scene continuity across the room boundary over general museum style similarity.",
                    "Do not use candidate label order as evidence. No retrieval scores or headings are provided.",
                    "If no current candidate is usable, set chosen_label to null.",
                ]
            ),
        }
    ]
    for candidate in current_candidates:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Current-room candidate {candidate['public_label']} "
                        f"({current_room_id})."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(candidate["source_image_path"]),
                    "detail": detail,
                },
            ]
        )
    for reference in target_references:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Target-room reverse-side reference {reference['public_label']} "
                        f"({target_room_id}); never choose this label."
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(reference["source_image_path"]),
                    "detail": detail,
                },
            ]
        )
    return {
        "model": model,
        "instructions": (
            "You are a careful visual museum navigation assistant. Judge only from "
            "the supplied full-scene images and room IDs. Return strict JSON only."
        ),
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "gemma4_passage_reranking",
                "strict": True,
                "schema": response_schema(current_labels, target_labels),
            }
        },
    }


def validate_vlm_choice(
    parsed: dict,
    *,
    current_labels: Sequence[str],
    target_labels: Sequence[str],
) -> None:
    current_set = set(current_labels)
    target_set = set(target_labels)
    chosen = parsed.get("chosen_label")
    if chosen is not None and chosen not in current_set:
        raise ValueError(f"Gemma chose invalid current label: {chosen!r}")

    assessments = parsed.get("candidate_assessments")
    if not isinstance(assessments, list):
        raise ValueError("Gemma output is missing candidate_assessments.")
    assessed_labels = [
        item.get("label")
        for item in assessments
        if isinstance(item, dict)
    ]
    if len(assessed_labels) != len(set(assessed_labels)):
        raise ValueError("Gemma assessed a current candidate more than once.")
    if set(assessed_labels) != current_set:
        missing = sorted(current_set - set(assessed_labels))
        extra = sorted(set(assessed_labels) - current_set)
        raise ValueError(
            f"Gemma assessments do not match current candidates; missing={missing}, extra={extra}"
        )

    status_by_label = {
        str(item["label"]): item.get("status")
        for item in assessments
        if isinstance(item, dict) and item.get("label") is not None
    }
    if chosen is not None and status_by_label.get(chosen) == "noise":
        raise ValueError("Gemma chose a candidate that it classified as noise.")

    used_targets = parsed.get("target_reference_labels_used")
    if not isinstance(used_targets, list) or any(
        label not in target_set for label in used_targets
    ):
        raise ValueError("Gemma output contains invalid target reference labels.")


def audit_candidate(candidate: dict) -> dict:
    keys = [
        "public_label",
        "label",
        "source_label",
        "rank",
        "selection_score",
        "target_mean_similarity",
        "negative_mean_similarity",
        "detection_score",
        "room_id",
        "pano_id",
        "capture_index",
        "source_image_path",
        "comparison_image_path",
        "crop_image_path",
        "masked_image_path",
        "same_source_duplicate_labels",
        "visual_duplicate_members",
    ]
    return {key: candidate.get(key) for key in keys if key in candidate}


def copy_audit_images(
    candidates: Sequence[dict],
    output_dir: Path,
    *,
    subdirectory: str,
) -> list[dict]:
    destination_dir = output_dir / subdirectory
    destination_dir.mkdir(parents=True, exist_ok=True)
    exports = []
    for candidate in candidates:
        source = Path(candidate["source_image_path"])
        destination = destination_dir / (
            f"{candidate['public_label']}_{_safe_name(candidate.get('label'))}"
            f"{source.suffix or '.png'}"
        )
        shutil.copy2(source, destination)
        exports.append(
            {
                "public_label": candidate["public_label"],
                "candidate_label": candidate.get("label"),
                "source_image_path": str(source),
                "copied_image_path": str(destination),
            }
        )
    return exports


def request_summary(
    *,
    args: argparse.Namespace,
    current_result: Path,
    target_result: Path,
    current_candidates: Sequence[dict],
    target_references: Sequence[dict],
) -> dict:
    return {
        "provider": args.provider,
        "profile": args.profile,
        "model": args.model,
        "api_base": args.api_base,
        "api_kind": args.api_kind,
        "timeout": args.timeout,
        "detail": args.detail,
        "reasoning_effort": args.reasoning_effort,
        "current_room_id": args.current_room_id,
        "target_room_id": args.target_room_id,
        "current_result_path": str(current_result),
        "target_reverse_result_path": str(target_result),
        "current_labels": [item["public_label"] for item in current_candidates],
        "target_labels": [item["public_label"] for item in target_references],
        "current_images": [item["source_image_path"] for item in current_candidates],
        "target_images": [item["source_image_path"] for item in target_references],
        "current_image_mode": "original_full_scene",
        "target_image_mode": "reverse_edge_original_full_scene",
        "input_image_count": len(current_candidates) + len(target_references),
        "scores_exposed_to_vlm": False,
    }


def _safe_name(value: object) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value or "candidate")
    ).strip("_") or "candidate"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return resolve_project_path(args.output_dir)
    return resolve_project_path(args.output_root) / (
        f"{_safe_name(args.current_room_id)}_to_{_safe_name(args.target_room_id)}"
    )


def build_model_client(args: argparse.Namespace) -> ModelResponseClient:
    settings = resolve_model_environment(
        default_model=args.model,
        default_api_base=args.api_base,
        default_api_kind=args.api_kind,
        profile=args.profile,
    )
    provider = args.provider or settings.provider or "gemini"
    api_base = (settings.api_base or args.api_base or DEFAULT_GEMINI_API_BASE).rstrip("/")
    client = ModelResponseClient(
        provider=provider,
        api_key=(
            args.api_key
            or settings.api_key
            or get_env_value("GEMINI_API_KEY", "GOOGLE_API_KEY")
        ),
        api_base=api_base,
        api_kind=settings.api_kind or args.api_kind,
        request_timeout=settings.request_timeout or args.timeout,
        temperature=0.0 if settings.temperature is None else settings.temperature,
    )
    if not client.is_configured():
        raise RuntimeError("Gemini API is not configured.")
    return client


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args()
    current_result_path, target_result_path = resolve_edge_results(args)
    current_payload = load_json(current_result_path)
    target_payload = load_json(target_result_path)
    if result_edge(current_payload) != (args.current_room_id, args.target_room_id):
        raise ValueError("Current result edge does not match requested rooms.")
    if result_edge(target_payload) != (args.target_room_id, args.current_room_id):
        raise ValueError("Target reference result must be the reverse directed edge.")

    embedder = create_image_embedder(
        model_name=args.dedup_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    current_candidates, current_dedup = prepare_candidates(
        current_payload,
        embedder=embedder,
        max_candidates=args.max_current,
        pool_size=args.dedup_pool_size,
        similarity_threshold=args.dedup_similarity_threshold,
    )
    target_references, target_dedup = prepare_candidates(
        target_payload,
        embedder=embedder,
        max_candidates=args.max_target,
        pool_size=args.dedup_pool_size,
        similarity_threshold=args.dedup_similarity_threshold,
    )
    current_candidates = assign_public_labels(current_candidates, prefix="C")
    target_references = assign_public_labels(target_references, prefix="T")

    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_exports = {
        "current_candidates": copy_audit_images(
            current_candidates,
            output_dir,
            subdirectory="current_candidates",
        ),
        "target_references": copy_audit_images(
            target_references,
            output_dir,
            subdirectory="target_references",
        ),
    }
    deduplication = {
        "model": args.dedup_model,
        "similarity_threshold": args.dedup_similarity_threshold,
        "current": current_dedup,
        "target": target_dedup,
    }
    write_json(output_dir / "deduplication.json", deduplication)
    summary = request_summary(
        args=args,
        current_result=current_result_path,
        target_result=target_result_path,
        current_candidates=current_candidates,
        target_references=target_references,
    )
    summary["image_exports"] = image_exports
    write_json(output_dir / "request_summary.json", summary)

    request_body = build_vlm_request(
        model=args.model,
        detail=args.detail,
        reasoning_effort=args.reasoning_effort,
        current_room_id=args.current_room_id,
        target_room_id=args.target_room_id,
        current_candidates=current_candidates,
        target_references=target_references,
    )
    if args.dry_run_request:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Wrote: {output_dir / 'deduplication.json'}", file=sys.stderr)
        print(f"Wrote: {output_dir / 'request_summary.json'}", file=sys.stderr)
        return 0

    raw_response_path = output_dir / "raw_response.json"
    if args.reuse_existing_response:
        raw_response = load_json(raw_response_path)
    else:
        raw_response = build_model_client(args).create(request_body)
        write_json(raw_response_path, raw_response)
    parsed = parse_json_output(raw_response)
    validate_vlm_choice(
        parsed,
        current_labels=[item["public_label"] for item in current_candidates],
        target_labels=[item["public_label"] for item in target_references],
    )

    current_by_public_label = {
        item["public_label"]: audit_candidate(item) for item in current_candidates
    }
    target_by_public_label = {
        item["public_label"]: audit_candidate(item) for item in target_references
    }
    chosen_public_label = parsed.get("chosen_label")
    chosen_candidate = (
        current_by_public_label.get(chosen_public_label)
        if isinstance(chosen_public_label, str)
        else None
    )
    baseline_choice = current_payload.get("passage_choice")
    baseline_label = (
        baseline_choice.get("chosen_label")
        if isinstance(baseline_choice, dict)
        else None
    )
    result = {
        "method": "detected_contrastive_dedup_gemma4_vlm_reranking",
        "configuration": {
            **summary,
            "dedup_model": args.dedup_model,
            "dedup_pool_size": args.dedup_pool_size,
            "dedup_similarity_threshold": args.dedup_similarity_threshold,
            "max_current": args.max_current,
            "max_target": args.max_target,
        },
        "baseline_detected_contrastive_choice": {
            "chosen_label": baseline_label,
            "chosen_source_label": (
                baseline_choice.get("chosen_source_label")
                if isinstance(baseline_choice, dict)
                else None
            ),
        },
        "current_candidates": list(current_by_public_label.values()),
        "target_references": list(target_by_public_label.values()),
        "vlm_choice": {
            **parsed,
            "chosen_candidate": chosen_candidate,
        },
        "choice_changed_from_baseline": (
            chosen_candidate is not None
            and chosen_candidate.get("label") != baseline_label
        ),
        "success": chosen_candidate is not None,
        "output_directory": str(output_dir),
        "result_json_path": str(output_dir / "result.json"),
    }
    write_json(output_dir / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote: {output_dir / 'result.json'}", file=sys.stderr)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
