from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.build_passage_memory_tree import (
    DEFAULT_DINOV2_PATCH_MAX_PATCHES,
    DEFAULT_DINOV2_PATCH_MODEL,
    DEFAULT_METADATA_PATH,
    DEFAULT_RENDER_ROOT,
    DEFAULT_DINOV2_PATCH_TOP_K,
    load_memory_metadata,
    load_or_compute_dinov2_patch_similarity_matrix,
    load_or_encode_dinov2_patch_features,
    memory_key,
    resolve_project_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute persistent DINOv2 patch features for memory-tree navigation."
    )
    parser.add_argument("--room-id", action="append", help="Room id to cache. Repeat for multiple rooms. Omit to cache all rooms.")
    parser.add_argument("--floor", action="append", help="Floor id to cache. Repeat for multiple floors, e.g. --floor 0 --floor 1.")
    parser.add_argument(
        "--gallery-rooms-only",
        action="store_true",
        help="Keep only room ids that start with 'Room ', excluding stairs/lifts/connectors.",
    )
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--output-dir", default="outputs/navigation_memory_tree_cache")
    parser.add_argument("--dinov2-patch-model", default=DEFAULT_DINOV2_PATCH_MODEL)
    parser.add_argument("--dinov2-patch-max-patches", type=int, default=DEFAULT_DINOV2_PATCH_MAX_PATCHES)
    parser.add_argument("--dinov2-patch-top-k", type=int, default=DEFAULT_DINOV2_PATCH_TOP_K)
    parser.add_argument(
        "--compute-pairwise",
        action="store_true",
        help="Also precompute the room-level patch top-k similarity matrix used by memory tree.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata_path = resolve_project_path(args.metadata_path)
    render_root = resolve_project_path(args.render_root)
    output_dir = resolve_project_path(args.output_dir)
    items = load_memory_metadata(metadata_path, render_root)
    room_filter = set(args.room_id or [])
    floor_filter = set(args.floor or [])
    if floor_filter:
        items = [item for item in items if str(item.get("floor")) in floor_filter]
    if room_filter:
        items = [item for item in items if item.get("room_id") in room_filter]
    if args.gallery_rooms_only:
        items = [item for item in items if str(item.get("room_id") or "").startswith("Room ")]
    if not items:
        raise SystemExit("No matching memory images to cache.")

    image_paths = [Path(str(item["image_path"])) for item in items]
    memory_keys = [memory_key(item) for item in items]
    features = load_or_encode_dinov2_patch_features(
        image_paths=image_paths,
        memory_keys=memory_keys,
        output_dir=output_dir,
        model_name=args.dinov2_patch_model,
        max_patches=args.dinov2_patch_max_patches,
        device=args.device,
        batch_size=args.batch_size,
    )
    pairwise_shapes = {}
    if args.compute_pairwise:
        room_to_indices: dict[str, list[int]] = {}
        for index, item in enumerate(items):
            room_id = str(item.get("room_id"))
            room_to_indices.setdefault(room_id, []).append(index)
        for room_id in sorted(room_to_indices):
            indices = room_to_indices[room_id]
            pairwise = load_or_compute_dinov2_patch_similarity_matrix(
                patch_features=features[indices],
                image_paths=[image_paths[index] for index in indices],
                memory_keys=[memory_keys[index] for index in indices],
                output_dir=output_dir,
                model_name=args.dinov2_patch_model,
                max_patches=args.dinov2_patch_max_patches,
                top_k=args.dinov2_patch_top_k,
            )
            pairwise_shapes[room_id] = list(pairwise.shape)
            print(
                json.dumps(
                    {"room_id": room_id, "pairwise_shape": pairwise_shapes[room_id]},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    payload = {
        "cached_image_count": len(items),
        "feature_shape": list(features.shape),
        "pairwise_shapes": pairwise_shapes,
        "room_ids": sorted({str(item.get("room_id")) for item in items}),
        "floor_ids": sorted({str(item.get("floor")) for item in items}),
        "output_dir": str(output_dir),
        "dinov2_patch_model": args.dinov2_patch_model,
        "dinov2_patch_max_patches": int(args.dinov2_patch_max_patches),
        "dinov2_patch_top_k": int(args.dinov2_patch_top_k),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
