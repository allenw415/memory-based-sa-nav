from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUERY_SLUG = "general_opening"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-size passage representative sets per room from semantic search results. "
            "Each room is processed independently with SALAD clustering; representatives are "
            "the highest semantic-score image in each target cluster."
        )
    )
    parser.add_argument(
        "--room-id",
        action="append",
        default=None,
        help='Room id to process, e.g. "Room 8". Repeat for multiple rooms. Defaults to Room 4/8/23.',
    )
    parser.add_argument(
        "--query-slug",
        default=DEFAULT_QUERY_SLUG,
        help="Slug used in result directories: outputs/passage_retrieval/room{n}_{query_slug}/results.json.",
    )
    parser.add_argument(
        "--result-dir-template",
        default="outputs/passage_retrieval/{room_slug}_{query_slug}",
        help="Template for input result directories. Available fields: room_slug, query_slug.",
    )
    parser.add_argument(
        "--output-dir-template",
        default="outputs/passage_clustering/{room_slug}/salad_cluster{target_clusters}",
        help="Template for output directories. Available fields: room_slug, query_slug, target_clusters.",
    )
    parser.add_argument(
        "--embedding-index",
        default="artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.npz",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--target-clusters", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser


def room_slug(room_id: str) -> str:
    return room_id.strip().lower().replace(" ", "")


def representative_prefix(room_id: str) -> str:
    digits = "".join(char for char in room_id if char.isdigit())
    return f"R{digits}P" if digits else "P"


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def build_command(args: argparse.Namespace, room_id: str) -> list[str]:
    slug = room_slug(room_id)
    fields = {
        "room_slug": slug,
        "query_slug": args.query_slug,
        "target_clusters": args.target_clusters,
    }
    result_dir = resolve_project_path(args.result_dir_template.format(**fields))
    output_dir = resolve_project_path(args.output_dir_template.format(**fields))
    result_json = result_dir / "results.json"
    if not result_json.exists():
        raise FileNotFoundError(f"Missing semantic search result for {room_id}: {result_json}")

    return [
        sys.executable,
        str(PROJECT_ROOT / "tools/data/merge_semantic_search_results.py"),
        "--result-json",
        str(result_json),
        "--embedding-index",
        str(resolve_project_path(args.embedding_index)),
        "--similarity-threshold",
        str(args.similarity_threshold),
        "--target-clusters",
        str(args.target_clusters),
        "--representative-prefix",
        representative_prefix(room_id),
        "--output-dir",
        str(output_dir),
        "--quiet",
    ]


def main() -> int:
    args = build_parser().parse_args()
    room_ids = args.room_id or ["Room 4", "Room 8", "Room 23"]

    for room_id in room_ids:
        command = build_command(args, room_id)
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
