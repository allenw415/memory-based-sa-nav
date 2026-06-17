from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_SIGLIP2_MODEL,
    brute_force_search,
    create_image_embedder,
    load_image_index_artifacts,
    load_json,
    resolve_embedding_model_name,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search memory images with a text query.")
    parser.add_argument("--query", required=True, help='Text query, e.g. "通道" or "a museum corridor".')
    parser.add_argument("--room-id", help='Optional room filter, e.g. "Room 8".')
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--index-path",
        default="artifacts/memory_localization/floor0_siglip2_images_fov90.npz",
    )
    parser.add_argument(
        "--metadata-path",
        default="artifacts/memory_localization/floor0_siglip2_images_fov90.metadata.json",
    )
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--render-root", default="renders/room_grounding_fov90")
    parser.add_argument("--output-dir", default="outputs/semantic_memory_search")
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def safe_name(value: object) -> str:
    text = str(value or "").strip()
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "item"


def resolve_capture_path(item: dict, render_root: Path) -> Path | None:
    raw_path = item.get("capture_path")
    if not isinstance(raw_path, str) or not raw_path:
        return None

    original = Path(raw_path)
    if original.exists():
        return original.resolve()

    pano_id = str(item.get("pano_id") or "")
    if pano_id:
        local_path = render_root / pano_id / original.name
        if local_path.exists():
            return local_path.resolve()
        capture_index = item.get("capture_index")
        capture_label = item.get("capture_label")
        if isinstance(capture_index, int) and isinstance(capture_label, str) and capture_label:
            pattern = f"{pano_id}_{capture_index:02d}_{capture_label}_*.png"
            matches = sorted((render_root / pano_id).glob(pattern))
            if matches:
                return matches[0].resolve()
        return local_path.resolve()

    if not original.is_absolute():
        return (PROJECT_ROOT / original).resolve()
    return original


def copy_result_image(*, source_path: Path | None, output_dir: Path, rank: int, item: dict) -> str | None:
    if source_path is None or not source_path.exists():
        return None
    suffix = source_path.suffix or ".png"
    filename = (
        f"rank_{rank:02d}_"
        f"{safe_name(item.get('room_id'))}_"
        f"{safe_name(item.get('pano_id'))}_"
        f"{safe_name(item.get('capture_label'))}"
        f"{suffix}"
    )
    destination = output_dir / filename
    shutil.copy2(source_path, destination)
    return filename


def write_html(output_dir: Path, payload: dict) -> None:
    cards = []
    for result in payload["results"]:
        image_name = result.get("output_image")
        if image_name:
            image_html = f'<img src="{html.escape(image_name)}" alt="rank {result["rank"]}">'
        else:
            image_html = "<div class='missing'>image unavailable</div>"
        cards.append(
            "\n".join(
                [
                    "<section class='card'>",
                    image_html,
                    "<pre>",
                    html.escape(json.dumps(result, ensure_ascii=False, indent=2)),
                    "</pre>",
                    "</section>",
                ]
            )
        )

    body = "\n".join(cards)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Semantic Memory Search</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f7; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #555; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    .card {{ background: white; padding: 12px; border-radius: 8px; box-shadow: 0 1px 4px #ccc; }}
    img {{ width: 100%; height: auto; border-radius: 6px; }}
    pre {{ white-space: pre-wrap; font-size: 12px; }}
    .missing {{ padding: 48px; text-align: center; background: #eee; color: #777; }}
  </style>
</head>
<body>
  <h1>Semantic Memory Search</h1>
  <div class="meta">
    query={html.escape(str(payload["query"]))}<br>
    room_id={html.escape(str(payload.get("room_id")))}<br>
    top_k={payload["top_k"]}, searched_images={payload["searched_image_count"]}
  </div>
  <div class="grid">
{body}
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    top_k = max(int(args.top_k), 1)
    index_path = resolve_project_path(args.index_path)
    metadata_path = resolve_project_path(args.metadata_path)
    render_root = resolve_project_path(args.render_root)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_image in output_dir.glob("rank_*"):
        if old_image.is_file():
            old_image.unlink()

    metadata_payload = load_json(metadata_path)
    metadata_items = metadata_payload.get("items")
    if not isinstance(metadata_items, list) or not metadata_items:
        raise RuntimeError(f"Metadata contains no indexed images: {metadata_path}")

    image_embeddings = load_image_index_artifacts(index_path)
    if len(metadata_items) != int(image_embeddings.shape[0]):
        raise RuntimeError("Index and metadata image counts do not match.")

    selected_indices = [
        index
        for index, item in enumerate(metadata_items)
        if isinstance(item, dict) and (not args.room_id or item.get("room_id") == args.room_id)
    ]
    if not selected_indices:
        scope = f"room {args.room_id!r}" if args.room_id else "the whole index"
        raise RuntimeError(f"No memory images found in {scope}.")

    embedder = create_image_embedder(
        model_name=resolve_embedding_model_name(args.embedding_model),
        device=args.device,
        batch_size=args.batch_size,
    )
    if not hasattr(embedder, "encode_texts"):
        raise RuntimeError(f"Embedding model {args.embedding_model!r} does not support text queries.")

    query_embedding = embedder.encode_texts([args.query])[0]
    selected_embeddings = image_embeddings[selected_indices]
    ranked = brute_force_search(
        selected_embeddings,
        query_embedding,
        top_k=min(top_k, len(selected_indices)),
    )

    results = []
    for rank, (local_index, score) in enumerate(ranked, start=1):
        memory_index = selected_indices[int(local_index)]
        item = metadata_items[memory_index]
        source_path = resolve_capture_path(item, render_root)
        output_image = copy_result_image(
            source_path=source_path,
            output_dir=output_dir,
            rank=rank,
            item=item,
        )
        results.append(
            {
                "rank": rank,
                "score": float(score),
                "memory_index": memory_index,
                "room_id": item.get("room_id"),
                "pano_id": item.get("pano_id"),
                "capture_index": item.get("capture_index"),
                "capture_label": item.get("capture_label"),
                "capture_heading": item.get("capture_heading"),
                "source_path": str(source_path) if source_path is not None else None,
                "image_available": bool(source_path and source_path.exists()),
                "output_image": output_image,
            }
        )

    payload = {
        "query": args.query,
        "room_id": args.room_id,
        "top_k": top_k,
        "embedding_model": resolve_embedding_model_name(args.embedding_model),
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "searched_image_count": len(selected_indices),
        "output_dir": str(output_dir),
        "results": results,
    }
    write_json(output_dir / "results.json", payload)
    write_html(output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nWrote: {output_dir / 'results.json'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'index.html'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
