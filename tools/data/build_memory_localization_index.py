from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav import PanoramaRenderer, get_env_value, load_dotenv
from memory_nav.data.memory_localization import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_DINOV2_SALAD_MODEL,
    MissingDependencyError,
    build_faiss_index,
    create_image_embedder,
    load_json,
    load_manifest_captures,
    parse_csv_argument,
    resolve_embedding_model_name,
    save_faiss_index,
    save_image_index_artifacts,
    select_memory_items,
    write_json,
)


load_dotenv(PROJECT_ROOT / ".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an image-memory localization index with an image embedder + FAISS.")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument(
        "--grounding-path",
        default="dataset/sites/british_museum/normalized/pano_room_grounding.json",
    )
    parser.add_argument("--floor", default="0")
    parser.add_argument("--include-sources", default="manual:accepted")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="artifacts/memory_localization")
    parser.add_argument("--output-prefix")
    parser.add_argument(
        "--render-api-key",
        default=get_env_value("GMAPS_KEY", "NAV_GMAPS_KEY", "GMAPS_API_KEY"),
    )
    parser.add_argument("--render-output-dir", default="renders/room_grounding")
    parser.add_argument("--render-seed", type=int)
    parser.add_argument("--heading-mode", choices=["museum", "cardinal", "grounding", "graph"], default="museum")
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--fov", type=int, default=45)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--render-timeout", type=float, default=60.0)
    parser.add_argument("--max-captures", type=int, default=8)
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


def default_output_prefix(*, floor: str, embedding_model: str, fov: int) -> str:
    if embedding_model == DEFAULT_DINOV2_SALAD_MODEL:
        return f"floor{floor}_dinov2_salad_images_fov{int(fov)}"
    if int(fov) == 45:
        return f"floor{floor}_siglip2_images"
    return f"floor{floor}_siglip2_images_fov{int(fov)}"


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


def main() -> int:
    args = build_parser().parse_args()

    artifacts_dir = (PROJECT_ROOT / args.artifacts_dir).resolve()
    grounding_path = (PROJECT_ROOT / args.grounding_path).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pano_graph = load_json(artifacts_dir / "pano_graph.json")
    grounding_payload = load_json(grounding_path)
    include_sources = parse_csv_argument(args.include_sources)
    panos = select_memory_items(
        grounding_payload=grounding_payload,
        pano_graph=pano_graph,
        floor=args.floor,
        include_sources=include_sources,
        limit=args.limit,
    )
    if not panos:
        raise RuntimeError("No labeled panos selected for the image memory index.")

    try:
        resolved_embedding_model = resolve_embedding_model_name(args.embedding_model)
        embedder = create_image_embedder(
            model_name=resolved_embedding_model,
            device=args.device,
            batch_size=args.batch_size,
        )
    except MissingDependencyError as exc:
        raise RuntimeError(str(exc)) from exc

    renderer = PanoramaRenderer(
        pano_graph,
        image_timeout=args.render_timeout,
        rng=random.Random(args.render_seed) if args.render_seed is not None else None,
    )

    np = embedder.np
    render_output_dir = (PROJECT_ROOT / args.render_output_dir).resolve()
    metadata_items: list[dict] = []
    all_image_embeddings: list[object] = []
    started_at = time.time()
    total = len(panos)

    emit_progress(
        f"[image-memory-index] floor={args.floor} panos={total} model={embedder.model_name} "
        f"device={embedder.device} heading={args.heading_mode} captures={args.max_captures}"
    )

    for pano_offset, pano_item in enumerate(panos, start=1):
        pano_id = pano_item["pano_id"]
        manifest_path = ensure_manifest(
            renderer=renderer,
            artifacts_dir=artifacts_dir,
            render_api_key=args.render_api_key,
            render_output_dir=render_output_dir,
            pano_id=pano_id,
            heading_mode=args.heading_mode,
            pitch=args.pitch,
            fov=args.fov,
            width=args.width,
            height=args.height,
        )
        _, captures = load_manifest_captures(manifest_path, max_captures=max(args.max_captures, 1))
        image_paths = [capture["path"] for capture in captures]
        emit_progress(
            f"[{pano_offset}/{total}] encode pano={pano_id} room={pano_item['room_id']} images={len(image_paths)}"
        )
        pano_embeddings = embedder.encode_image_paths(image_paths)
        if pano_embeddings.shape[0] != len(image_paths):
            raise RuntimeError(
                f"Expected {len(image_paths)} image embeddings for pano {pano_id}, got {pano_embeddings.shape[0]}."
            )
        for capture_index, (capture, embedding) in enumerate(zip(captures, pano_embeddings, strict=False)):
            memory_index = len(metadata_items)
            metadata_items.append(
                {
                    "memory_index": memory_index,
                    **pano_item,
                    "manifest_path": str(manifest_path),
                    "capture_index": capture_index,
                    "capture_path": capture["path"],
                    "capture_label": capture.get("label"),
                    "capture_heading": capture.get("heading"),
                }
            )
            all_image_embeddings.append(embedding)

    if not metadata_items:
        raise RuntimeError("No capture images were collected for the image memory index.")

    image_embeddings = np.stack(all_image_embeddings, axis=0).astype(np.float32)
    faiss_index = build_faiss_index(image_embeddings)

    output_prefix = args.output_prefix or default_output_prefix(
        floor=str(args.floor),
        embedding_model=resolved_embedding_model,
        fov=args.fov,
    )
    index_path = output_dir / f"{output_prefix}.npz"
    faiss_path = output_dir / f"{output_prefix}.faiss"
    metadata_path = output_dir / f"{output_prefix}.metadata.json"

    save_image_index_artifacts(
        index_path=index_path,
        image_embeddings=image_embeddings,
    )
    save_faiss_index(faiss_index, faiss_path)
    write_json(
        metadata_path,
        {
            "summary": {
                "floor": str(args.floor),
                "pano_count": len(panos),
                "image_count": len(metadata_items),
                "embedding_model": embedder.model_name,
                "device": embedder.device,
                "dimension": int(image_embeddings.shape[1]),
                "heading_mode": args.heading_mode,
                "max_captures": int(args.max_captures),
                "include_sources": include_sources,
                "build_elapsed": format_duration(time.time() - started_at),
            },
            "items": metadata_items,
        },
    )

    print(
        json.dumps(
            {
                "index_path": str(index_path),
                "faiss_path": str(faiss_path),
                "metadata_path": str(metadata_path),
                "pano_count": len(panos),
                "image_count": len(metadata_items),
                "dimension": int(image_embeddings.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
