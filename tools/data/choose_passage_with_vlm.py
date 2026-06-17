from __future__ import annotations

import argparse
import json
import math
import mimetypes
import shutil
import sys
import urllib.error
import urllib.request
from base64 import b64decode, b64encode
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.common.env import get_env_value, resolve_model_environment  # noqa: E402
from memory_nav.common.model_client import (  # noqa: E402
    DEFAULT_OPENAI_API_BASE,
    ModelResponseClient,
    parse_json_output,
    resolve_api_kind,
)
from memory_nav.data.memory_localization import write_json  # noqa: E402


DEFAULT_CURRENT_REPRESENTATIVES = "outputs/passage_clustering/room8/salad_cluster8/representatives.json"
DEFAULT_TARGET_REPRESENTATIVES = "outputs/passage_clustering/room23/salad_cluster8/representatives.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Use a VLM to choose the best current-room passage toward a target room."
    )
    parser.add_argument("--current-room-id", default="Room 8")
    parser.add_argument("--target-room-id", default="Room 23")
    parser.add_argument("--current-representatives", default=DEFAULT_CURRENT_REPRESENTATIVES)
    parser.add_argument("--target-representatives", default=DEFAULT_TARGET_REPRESENTATIVES)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key")
    parser.add_argument("--api-base", default=DEFAULT_OPENAI_API_BASE)
    parser.add_argument("--api-kind", default="responses", choices=["responses", "chat_completions"])
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--detail", default="high", choices=["low", "high", "auto"])
    parser.add_argument("--max-current", type=int, default=8)
    parser.add_argument("--max-target", type=int, default=8)
    parser.add_argument("--output-dir", default="outputs/navigation/room8_to_room23_gpt55")
    parser.add_argument(
        "--dry-run-request",
        action="store_true",
        help="Write request_summary.json without calling the model API.",
    )
    parser.add_argument(
        "--reuse-existing-choice",
        action="store_true",
        help="Skip the model call and redraw from output_dir/chosen_passage.json.",
    )
    parser.add_argument(
        "--draw-mode",
        choices=["pil", "image-edit", "none"],
        default="pil",
        help="How to create the marked navigation image after choosing a passage.",
    )
    parser.add_argument(
        "--no-draw",
        action="store_true",
        help="Deprecated alias for --draw-mode none.",
    )
    parser.add_argument(
        "--image-edit-model",
        default="gpt-image-1.5",
        help="Image model used when --draw-mode image-edit.",
    )
    parser.add_argument(
        "--image-edit-size",
        default="auto",
        help="Image edit output size, e.g. auto, 1024x1024.",
    )
    parser.add_argument(
        "--image-edit-quality",
        default="medium",
        help="Image edit quality when supported by the image model.",
    )
    parser.add_argument(
        "--image-edit-prompt",
        default=None,
        help="Optional override prompt for --draw-mode image-edit.",
    )
    parser.add_argument(
        "--dry-run-image-edit",
        action="store_true",
        help="With --draw-mode image-edit, write image_edit_request_summary.json without calling the image API.",
    )
    parser.add_argument(
        "--line-start",
        default="0.50,0.92",
        help="Normalized x,y start of the red arrow. Default: bottom center.",
    )
    parser.add_argument(
        "--line-end",
        default="0.50,0.48",
        help="Normalized x,y end of the red arrow. Default: center-upper path area.",
    )
    parser.add_argument("--line-width", type=int, default=0, help="Arrow line width in pixels. 0 = auto.")
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"


def representative_image_path(representative: dict, representatives_json: Path) -> Path:
    representative_image = representative.get("representative_image")
    if isinstance(representative_image, str) and representative_image:
        candidate = representatives_json.parent / representative_image
        if candidate.exists():
            return candidate.resolve()
    image_path = representative.get("image_path")
    if isinstance(image_path, str) and image_path:
        candidate = Path(image_path)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Representative image not found for {representative.get('label')}: {representatives_json}")


def load_representatives(path: Path, *, limit: int) -> list[dict]:
    payload = load_json(path)
    reps = payload.get("representatives")
    if not isinstance(reps, list) or not reps:
        raise RuntimeError(f"No representatives in {path}")
    loaded = []
    for rep in reps[: max(limit, 1)]:
        if not isinstance(rep, dict):
            continue
        label = rep.get("label")
        if not isinstance(label, str) or not label:
            continue
        item = dict(rep)
        item["resolved_image_path"] = str(representative_image_path(rep, path))
        loaded.append(item)
    if not loaded:
        raise RuntimeError(f"No usable representatives in {path}")
    return loaded


def build_instructions() -> str:
    return "\n".join(
        [
            "You are a careful museum navigation assistant.",
            "Your job is to choose a current-room passage image that best guides a visitor toward a target room.",
            "Retrieved candidate images may contain noise. Never assume every current-room candidate is a passage.",
            "First classify each current-room candidate as valid_passage, ambiguous, or noise.",
            "Then choose one label only from the current-room labels, unless no valid/ambiguous passage exists.",
            "Do not choose target-room reference labels. Target-room images are visual context only.",
            "Judge visually; do not rely on retrieval scores alone.",
            "Return strict JSON only.",
        ]
    )


def build_task_text(
    *,
    current_room_id: str,
    target_room_id: str,
    current_reps: list[dict],
    target_reps: list[dict],
) -> str:
    current_labels = [rep["label"] for rep in current_reps]
    target_labels = [rep["label"] for rep in target_reps]
    current_meta = [
        {
            "label": rep["label"],
            "room_id": rep.get("room_id"),
            "cluster_id": rep.get("cluster_id"),
            "cluster_size": rep.get("cluster_size"),
            "semantic_score": rep.get("best_score"),
            "capture_label": rep.get("capture_label"),
            "capture_heading": rep.get("capture_heading"),
        }
        for rep in current_reps
    ]
    target_meta = [
        {
            "label": rep["label"],
            "room_id": rep.get("room_id"),
            "cluster_id": rep.get("cluster_id"),
            "semantic_score": rep.get("best_score"),
            "capture_label": rep.get("capture_label"),
        }
        for rep in target_reps
    ]
    return "\n".join(
        [
            f"Current room: {current_room_id}",
            f"Target room: {target_room_id}",
            f"Current-room candidate labels, choose only from these: {current_labels}",
            f"Target-room reference labels, do not choose these: {target_labels}",
            "",
            "Current-room metadata:",
            json.dumps(current_meta, ensure_ascii=False, indent=2),
            "",
            "Target-room reference metadata:",
            json.dumps(target_meta, ensure_ascii=False, indent=2),
            "",
            "A valid passage candidate should show a plausible route a visitor can physically walk through: a doorway, wide opening, entrance/exit, archway, portal, or visible adjacent gallery beyond an opening.",
            "Invalid/noise candidates include close-up exhibits, display cases, walls, sculptures, or general gallery views with no clear walkable opening/path.",
            "Ambiguous candidates may show a possible path but the opening is unclear or visually weak.",
            "",
            "Decision procedure:",
            "1. Assess every current-room candidate label.",
            "2. Filter out noise candidates.",
            "3. Among valid_passage candidates, choose the one most likely to lead from the current room toward the target room using architectural continuity, adjacent-gallery visibility, and similarity to target-room references.",
            "4. If no valid passage exists but an ambiguous candidate is still the best option, choose it with low confidence.",
            "5. If no current-room candidate is usable, set chosen_label to null.",
        ]
    )


def response_schema(current_labels: list[str], target_labels: list[str]) -> dict:
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
                        "status": {"type": "string", "enum": ["valid_passage", "ambiguous", "noise"]},
                        "passage_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "status", "passage_confidence", "reason"],
                },
            },
            "valid_passage_labels": {"type": "array", "items": {"type": "string", "enum": current_labels}},
            "invalid_or_noisy_labels": {"type": "array", "items": {"type": "string", "enum": current_labels}},
            "chosen_label": {"anyOf": [{"type": "string", "enum": current_labels}, {"type": "null"}]},
            "navigation_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "target_reference_labels_used": {"type": "array", "items": {"type": "string", "enum": target_labels}},
            "why_this_passage": {"type": "string"},
            "why_not_others": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "label": {"type": "string", "enum": current_labels},
                        "reason": {"type": "string"},
                    },
                    "required": ["label", "reason"],
                },
            },
            "red_line_instruction": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "image_label": {"anyOf": [{"type": "string", "enum": current_labels}, {"type": "null"}]},
                    "instruction": {"type": "string"},
                },
                "required": ["image_label", "instruction"],
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
            "red_line_instruction",
        ],
    }


def build_request_body(args: argparse.Namespace, current_reps: list[dict], target_reps: list[dict]) -> dict:
    current_labels = [rep["label"] for rep in current_reps]
    target_labels = [rep["label"] for rep in target_reps]
    content: list[dict] = [
        {
            "type": "input_text",
            "text": build_task_text(
                current_room_id=args.current_room_id,
                target_room_id=args.target_room_id,
                current_reps=current_reps,
                target_reps=target_reps,
            ),
        }
    ]
    for rep in current_reps:
        content.append({"type": "input_text", "text": f"Current-room candidate {rep['label']} ({args.current_room_id})."})
        content.append({"type": "input_image", "image_url": image_to_data_url(Path(rep["resolved_image_path"])), "detail": args.detail})
    for rep in target_reps:
        content.append({"type": "input_text", "text": f"Target-room reference {rep['label']} ({args.target_room_id}); do not choose this label."})
        content.append({"type": "input_image", "image_url": image_to_data_url(Path(rep["resolved_image_path"])), "detail": args.detail})

    return {
        "model": args.model,
        "instructions": build_instructions(),
        "input": [{"role": "user", "content": content}],
        "reasoning": {"effort": "low"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "passage_choice_with_noise_filtering",
                "strict": True,
                "schema": response_schema(current_labels, target_labels),
            }
        },
    }


def request_summary(request_body: dict, current_reps: list[dict], target_reps: list[dict]) -> dict:
    return {
        "model": request_body.get("model"),
        "current_labels": [rep["label"] for rep in current_reps],
        "target_labels": [rep["label"] for rep in target_reps],
        "current_images": [rep["resolved_image_path"] for rep in current_reps],
        "target_images": [rep["resolved_image_path"] for rep in target_reps],
        "input_block_count": len(request_body["input"][0]["content"]),
        "schema_name": request_body["text"]["format"]["name"],
        "instructions": request_body["instructions"],
    }


def validate_choice(parsed: dict, current_labels: list[str]) -> None:
    chosen = parsed.get("chosen_label")
    if chosen is not None and chosen not in current_labels:
        raise RuntimeError(f"Model chose invalid label {chosen!r}; expected one of {current_labels}")
    assessments = parsed.get("candidate_assessments")
    if not isinstance(assessments, list):
        raise RuntimeError("Model output missing candidate_assessments list.")
    assessed_labels = {item.get("label") for item in assessments if isinstance(item, dict)}
    missing = [label for label in current_labels if label not in assessed_labels]
    if missing:
        raise RuntimeError(f"Model did not assess all current candidates: {missing}")


def parse_normalized_point(value: str, *, name: str) -> tuple[float, float]:
    try:
        x_text, y_text = value.split(",", 1)
        x = float(x_text)
        y = float(y_text)
    except ValueError as exc:
        raise ValueError(f"{name} must be formatted as x,y with normalized coordinates, e.g. 0.50,0.92") from exc
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return x, y


def chosen_representative(parsed: dict, current_reps: list[dict]) -> dict | None:
    chosen = parsed.get("chosen_label")
    if not isinstance(chosen, str):
        return None
    return {rep["label"]: rep for rep in current_reps}.get(chosen)




def cleanup_previous_chosen_images(output_dir: Path) -> None:
    for pattern in (
        "chosen_passage_original_*",
        "chosen_passage_marked_*",
        "chosen_passage_marked_model_*",
    ):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()

def copy_chosen_image(parsed: dict, current_reps: list[dict], output_dir: Path) -> Path | None:
    representative = chosen_representative(parsed, current_reps)
    if representative is None:
        return None
    source = Path(representative["resolved_image_path"])
    suffix = source.suffix or ".png"
    destination = output_dir / f"chosen_passage_original_{representative['label']}{suffix}"
    shutil.copy2(source, destination)
    return destination


def draw_navigation_arrow(
    *,
    source_image: Path,
    output_image: Path,
    start: tuple[float, float],
    end: tuple[float, float],
    line_width: int,
) -> None:
    from PIL import Image, ImageDraw

    with Image.open(source_image) as opened:
        image = opened.convert("RGB")
    width, height = image.size
    start_px = (int(start[0] * width), int(start[1] * height))
    end_px = (int(end[0] * width), int(end[1] * height))
    stroke_width = line_width if line_width > 0 else max(8, int(min(width, height) * 0.018))
    outline_width = max(stroke_width + 6, int(stroke_width * 1.7))
    draw = ImageDraw.Draw(image)

    # Black outline first, then red arrow, so the navigation mark stays visible on bright museum floors.
    draw.line([start_px, end_px], fill=(20, 20, 20), width=outline_width)
    draw.line([start_px, end_px], fill=(255, 0, 0), width=stroke_width)

    dx = end_px[0] - start_px[0]
    dy = end_px[1] - start_px[1]
    angle = math.atan2(dy, dx)
    arrow_len = max(28, int(stroke_width * 4.0))
    arrow_angle = math.radians(28)
    left = (
        int(end_px[0] - arrow_len * math.cos(angle - arrow_angle)),
        int(end_px[1] - arrow_len * math.sin(angle - arrow_angle)),
    )
    right = (
        int(end_px[0] - arrow_len * math.cos(angle + arrow_angle)),
        int(end_px[1] - arrow_len * math.sin(angle + arrow_angle)),
    )
    draw.polygon([end_px, left, right], fill=(20, 20, 20))
    inner_left = (
        int(end_px[0] - arrow_len * 0.82 * math.cos(angle - arrow_angle)),
        int(end_px[1] - arrow_len * 0.82 * math.sin(angle - arrow_angle)),
    )
    inner_right = (
        int(end_px[0] - arrow_len * 0.82 * math.cos(angle + arrow_angle)),
        int(end_px[1] - arrow_len * 0.82 * math.sin(angle + arrow_angle)),
    )
    draw.polygon([end_px, inner_left, inner_right], fill=(255, 0, 0))
    image.save(output_image)




def build_image_edit_prompt(parsed: dict) -> str:
    instruction = parsed.get("red_line_instruction")
    instruction_text = ""
    if isinstance(instruction, dict) and isinstance(instruction.get("instruction"), str):
        instruction_text = instruction["instruction"].strip()
    if not instruction_text:
        instruction_text = "Mark the visible route the visitor should walk toward."
    return "\n".join(
        [
            "Edit this museum navigation image by adding ONLY a red navigation mark.",
            instruction_text,
            "The mark should look like a subtle floor-following red path or arrow, not a large UI sticker.",
            "Place it on the walkable floor/path area and make it follow the image perspective toward the selected opening.",
            "Use a semi-transparent or natural-looking red line with a small arrowhead near the destination.",
            "Do not cover important exhibits, wall labels, sculptures, or the destination opening.",
            "Do not add text, captions, labels, or extra symbols.",
            "Do not change the museum scene, lighting, objects, walls, floor, exhibits, camera perspective, or composition.",
            "Preserve the original image as much as possible; only add the red navigation path.",
        ]
    )


def extract_image_edit_base64(payload: dict) -> str:
    data = payload.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            b64_json = item.get("b64_json")
            if isinstance(b64_json, str) and b64_json:
                return b64_json
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        result = item.get("result")
        if isinstance(result, str) and result:
            return result
    raise RuntimeError("Image edit response did not contain base64 image data.")




def encode_multipart_form_data(fields: dict[str, str], files: list[tuple[str, Path, str]]) -> tuple[bytes, str]:
    boundary = "----memory-nav-openai-image-edit-boundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for field_name, file_path, mime_type in files:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("utf-8"),
                (
                    f'Content-Disposition: form-data; name="{field_name}"; '
                    f'filename="{file_path.name}"\r\n'
                ).encode("utf-8"),
                f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
                file_path.read_bytes(),
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def read_http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
    return body.strip()

def call_image_edit_api(
    *,
    args: argparse.Namespace,
    source_image: Path,
    output_image: Path,
    prompt: str,
    output_dir: Path,
) -> bool:
    endpoint = f"{args.api_base.rstrip('/')}/images/edits"
    mime_type, _ = mimetypes.guess_type(str(source_image))
    if not mime_type:
        mime_type = "image/png"
    fields = {
        "model": args.image_edit_model,
        "prompt": prompt,
        "n": "1",
        "output_format": "png",
    }
    if args.image_edit_size:
        fields["size"] = args.image_edit_size
    if args.image_edit_quality:
        fields["quality"] = args.image_edit_quality

    write_json(output_dir / "image_edit_request_summary.json", {
        "endpoint": endpoint,
        "request_encoding": "multipart/form-data",
        "model": args.image_edit_model,
        "source_image": str(source_image),
        "source_mime_type": mime_type,
        "output_image": str(output_image),
        "prompt": prompt,
        "size": args.image_edit_size,
        "quality": args.image_edit_quality,
    })
    if args.dry_run_image_edit:
        return False
    api_key = args.api_key or get_env_value("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key for image edit. Set OPENAI_API_KEY or pass --api-key.")

    body, boundary = encode_multipart_form_data(fields, [("image[]", source_image, mime_type)])
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_text = read_http_error_body(exc)
        write_json(output_dir / "image_edit_error.json", {
            "status": exc.code,
            "reason": exc.reason,
            "body": body_text,
            "endpoint": endpoint,
            "model": args.image_edit_model,
        })
        detail = f": {body_text}" if body_text else ""
        raise RuntimeError(f"Image edit request failed with HTTP {exc.code} {exc.reason}{detail}") from exc
    write_json(output_dir / "image_edit_response.json", payload)
    output_image.write_bytes(b64decode(extract_image_edit_base64(payload)))
    return True

def draw_with_image_edit(args: argparse.Namespace, parsed: dict, original: Path, output_dir: Path) -> Path | None:
    chosen = parsed.get("chosen_label")
    if not isinstance(chosen, str):
        raise RuntimeError("Cannot image-edit without a chosen_label.")
    output_image = output_dir / f"chosen_passage_marked_model_{chosen}.png"
    prompt = args.image_edit_prompt or build_image_edit_prompt(parsed)
    did_write = call_image_edit_api(
        args=args,
        source_image=original,
        output_image=output_image,
        prompt=prompt,
        output_dir=output_dir,
    )
    return output_image if did_write else None

def copy_and_draw_chosen_image(args: argparse.Namespace, parsed: dict, current_reps: list[dict], output_dir: Path) -> None:
    for stale_key in ("marked_image", "drawn_line"):
        parsed.pop(stale_key, None)
    original = copy_chosen_image(parsed, current_reps, output_dir)
    if original is None:
        parsed.pop("original_image", None)
        return
    parsed["original_image"] = str(original)
    draw_mode = "none" if args.no_draw else args.draw_mode
    if draw_mode == "none":
        return
    chosen = parsed.get("chosen_label")
    if draw_mode == "image-edit":
        marked = draw_with_image_edit(args, parsed, original, output_dir)
        parsed["drawn_line"] = {
            "mode": "image-edit",
            "image_edit_model": args.image_edit_model,
            "prompt": args.image_edit_prompt or build_image_edit_prompt(parsed),
            "dry_run": bool(args.dry_run_image_edit),
        }
        if marked is not None:
            parsed["marked_image"] = str(marked)
        return

    start = parse_normalized_point(args.line_start, name="--line-start")
    end = parse_normalized_point(args.line_end, name="--line-end")
    marked = output_dir / f"chosen_passage_marked_{chosen}.png"
    draw_navigation_arrow(
        source_image=original,
        output_image=marked,
        start=start,
        end=end,
        line_width=args.line_width,
    )
    parsed["marked_image"] = str(marked)
    parsed["drawn_line"] = {
        "mode": "pil",
        "start_normalized": list(start),
        "end_normalized": list(end),
        "line_width": args.line_width or "auto",
    }


def load_or_call_model(args: argparse.Namespace, request_body: dict, output_dir: Path) -> dict:
    if args.reuse_existing_choice:
        chosen_path = output_dir / "chosen_passage.json"
        if chosen_path.exists():
            return load_json(chosen_path)
        raw_path = output_dir / "raw_response.json"
        if raw_path.exists():
            return parse_json_output(load_json(raw_path))
        raise FileNotFoundError(f"Missing {chosen_path} and {raw_path}; cannot --reuse-existing-choice.")

    settings = resolve_model_environment(
        default_model=args.model,
        default_api_base=args.api_base,
        default_api_kind=args.api_kind,
    )
    client = ModelResponseClient(
        provider=settings.provider,
        api_key=args.api_key or settings.api_key or get_env_value("OPENAI_API_KEY"),
        api_base=(args.api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/"),
        api_kind=resolve_api_kind(args.api_kind or settings.api_kind),
        request_timeout=args.timeout,
        num_ctx=settings.num_ctx,
        temperature=settings.temperature,
    )
    if not client.is_configured():
        raise RuntimeError("Missing API configuration. Set OPENAI_API_KEY or pass --api-key.")

    payload = client.create(request_body)
    write_json(output_dir / "raw_response.json", payload)
    return parse_json_output(payload)


def main() -> int:
    args = build_parser().parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_path = resolve_project_path(args.current_representatives)
    target_path = resolve_project_path(args.target_representatives)
    current_reps = load_representatives(current_path, limit=args.max_current)
    target_reps = load_representatives(target_path, limit=args.max_target)
    request_body = build_request_body(args, current_reps, target_reps)

    write_json(output_dir / "request_summary.json", request_summary(request_body, current_reps, target_reps))
    if args.dry_run_request:
        print(f"Wrote: {output_dir / 'request_summary.json'}")
        return 0

    parsed = load_or_call_model(args, request_body, output_dir)
    validate_choice(parsed, [rep["label"] for rep in current_reps])
    cleanup_previous_chosen_images(output_dir)
    write_json(output_dir / "chosen_passage.json", parsed)
    copy_and_draw_chosen_image(args, parsed, current_reps, output_dir)
    write_json(output_dir / "chosen_passage.json", parsed)

    print(json.dumps(parsed, ensure_ascii=False, indent=2))
    print(f"Wrote: {output_dir / 'chosen_passage.json'}", file=sys.stderr)
    if not args.reuse_existing_choice:
        print(f"Wrote: {output_dir / 'raw_response.json'}", file=sys.stderr)
    if parsed.get("marked_image"):
        print(f"Wrote: {parsed['marked_image']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
