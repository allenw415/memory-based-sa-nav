from __future__ import annotations

import argparse

from ._common import (
    ensure_project_root_on_path,
    load_normalized_artifacts,
    render_json,
    resolve_project_path,
    write_text_if_requested,
)

ensure_project_root_on_path()

from memory_nav.data.memory_localization import DEFAULT_DINOV2_SALAD_MODEL  # noqa: E402
from memory_nav.navigation import (  # noqa: E402
    IndexedPanoramaViewStore,
    PanoramaGraphImageGoalSimulator,
    PureVisualDirectionPolicy,
    resolve_goal_label,
)


DEFAULT_INDEX_PATH = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz"
DEFAULT_METADATA_PATH = "artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json"
DEFAULT_MANIFEST_ROOT = "renders/room_grounding_fov90"
DEFAULT_REPRESENTATIVES_PATH = "outputs/passage_clustering/room8/salad_cluster8/representatives.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Navigate a panorama graph using only eight-view image-goal similarity."
    )
    parser.add_argument("--start-pano-id", required=True)
    goal_group = parser.add_mutually_exclusive_group()
    goal_group.add_argument("--goal-image")
    goal_group.add_argument("--goal-label")
    parser.add_argument("--target-room-id", default="Room 23")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    parser.add_argument("--metadata-path", default=DEFAULT_METADATA_PATH)
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--representatives-path", default=DEFAULT_REPRESENTATIVES_PATH)
    parser.add_argument("--embedding-model", default=DEFAULT_DINOV2_SALAD_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = load_normalized_artifacts(
        args.artifacts_dir,
        pano_graph=True,
        pano_room_grounding=True,
    )
    pano_graph = artifacts.pano_graph or {}
    grounding = artifacts.pano_room_grounding or {}
    mappings = grounding.get("mappings")
    if not isinstance(mappings, dict):
        raise RuntimeError("pano_room_grounding.json does not contain a mappings object.")

    view_store = IndexedPanoramaViewStore(
        index_path=resolve_project_path(args.index_path),
        metadata_path=resolve_project_path(args.metadata_path),
        manifest_root=resolve_project_path(args.manifest_root),
    )

    goal_label = args.goal_label or ("R8P7" if not args.goal_image else None)
    goal_context: dict
    if args.goal_image:
        goal_path = resolve_project_path(args.goal_image)
        goal_embedding = view_store.embedding_for_image(
            goal_path,
            embedding_model=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
        )
        goal_context = {"mode": "image", "path": str(goal_path)}
    else:
        representative = resolve_goal_label(
            str(goal_label),
            representatives_path=resolve_project_path(args.representatives_path),
        )
        goal_embedding = view_store.embedding_for_capture(
            str(representative["pano_id"]),
            int(representative["capture_index"]),
        )
        goal_context = {
            "mode": "label",
            "label": goal_label,
            "representative": representative,
        }

    simulator = PanoramaGraphImageGoalSimulator(
        pano_graph=pano_graph,
        pano_room_mappings=mappings,
        observation_provider=view_store.load_views,
        policy=PureVisualDirectionPolicy(),
    )
    result = simulator.run(
        start_pano_id=args.start_pano_id,
        goal_embedding=goal_embedding,
        target_room_id=args.target_room_id,
        max_steps=args.max_steps,
    )
    payload = {
        "method": "eight_view_pure_similarity",
        "goal": goal_context,
        "index_path": str(resolve_project_path(args.index_path)),
        "metadata_path": str(resolve_project_path(args.metadata_path)),
        "manifest_root": str(resolve_project_path(args.manifest_root)),
        **result.to_dict(),
    }
    output = render_json(payload)
    write_text_if_requested(output, args.output_path)
    print(output)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
