from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.experiments.build_passage_memory_tree import (  # noqa: E402
    create_memory_tree_embedder,
    load_memory_metadata,
    load_optional_pano_graph,
    memory_key,
    normalize_rows,
    resolve_project_path,
    room_token,
    safe_name,
    write_json,
)


DEFAULT_ROOM_ID = "Room 23"
DEFAULT_SOURCE_PANO_ID = "Li54te8XaSyXgj2x_c2msA"
DEFAULT_TARGET_PANO_ID = "-tWBcZwQAYJz79VgT61mRg"
DEFAULT_CAPTURE_INDICES = "0,1,2"
DEFAULT_RENDER_ROOT = "renders/room_grounding_fov90"
DEFAULT_ARTIFACTS_DIR = "dataset/sites/british_museum/normalized"
DEFAULT_SALAD_INDEX_PATH = "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.npz"
DEFAULT_SALAD_METADATA_PATH = (
    "artifacts/memory_localization/floor0_1_dinov2_salad_images_fov90.metadata.json"
)
DEFAULT_DREAMSIM_INDEX_PATH = (
    "artifacts/memory_localization/floor0_1_dreamsim_ensemble_images_fov90.npz"
)
DEFAULT_DREAMSIM_METADATA_PATH = (
    "artifacts/memory_localization/floor0_1_dreamsim_ensemble_images_fov90.metadata.json"
)
DEFAULT_RECIPES = (
    "dreamsim_full",
    "salad_full",
    "salad_center_70",
    "salad_lower_walkable",
    "salad_multi_crop_max",
)
TOP_K = 5


@dataclass(frozen=True)
class EvaluationCase:
    stage: str
    case_id: str
    source_index: int
    expected_index: int
    candidate_indices: tuple[int, ...]


@dataclass(frozen=True)
class RankedCandidate:
    candidate_index: int
    similarity: float


@dataclass(frozen=True)
class RecipeScores:
    name: str
    item_indices: tuple[int, ...]
    similarity_matrix: np.ndarray
    component_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.similarity_matrix.shape != (len(self.item_indices), len(self.item_indices)):
            raise ValueError("similarity_matrix shape must match item_indices.")

    @property
    def row_by_item_index(self) -> dict[int, int]:
        return {item_index: row for row, item_index in enumerate(self.item_indices)}

    def similarity(self, source_index: int, candidate_index: int) -> float:
        row_by_item_index = self.row_by_item_index
        return float(
            self.similarity_matrix[
                row_by_item_index[int(source_index)],
                row_by_item_index[int(candidate_index)],
            ]
        )

    @classmethod
    def from_embeddings(
        cls,
        *,
        name: str,
        item_indices: Sequence[int],
        embeddings,
        component_names: Sequence[str] = (),
    ) -> "RecipeScores":
        normalized = normalize_rows(np.asarray(embeddings, dtype=np.float32))
        return cls(
            name=name,
            item_indices=tuple(int(index) for index in item_indices),
            similarity_matrix=normalized @ normalized.T,
            component_names=tuple(component_names),
        )

    @classmethod
    def max_of(cls, *, name: str, recipes: Sequence["RecipeScores"]) -> "RecipeScores":
        if not recipes:
            raise ValueError("At least one recipe is required.")
        first_indices = recipes[0].item_indices
        if any(recipe.item_indices != first_indices for recipe in recipes):
            raise ValueError("Component recipes must share item_indices.")
        matrix = np.maximum.reduce([recipe.similarity_matrix for recipe in recipes])
        return cls(
            name=name,
            item_indices=first_indices,
            similarity_matrix=matrix,
            component_names=tuple(recipe.name for recipe in recipes),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whether memory-tree parent-child image similarity can select "
            "the next pano's nearest-heading child without using pano/heading in ranking."
        )
    )
    parser.add_argument("--room-id", default=DEFAULT_ROOM_ID)
    parser.add_argument("--source-pano-id", default=DEFAULT_SOURCE_PANO_ID)
    parser.add_argument("--target-pano-id", default=DEFAULT_TARGET_PANO_ID)
    parser.add_argument("--capture-indices", default=DEFAULT_CAPTURE_INDICES)
    parser.add_argument("--branching-factor", type=int, default=1)
    parser.add_argument("--stage", choices=["case", "room", "both"], default="both")
    parser.add_argument("--recipes", default=",".join(DEFAULT_RECIPES))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--gallery-stage2-limit", type=int, default=24)
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--salad-index-path", default=DEFAULT_SALAD_INDEX_PATH)
    parser.add_argument("--salad-metadata-path", default=DEFAULT_SALAD_METADATA_PATH)
    parser.add_argument("--dreamsim-index-path", default=DEFAULT_DREAMSIM_INDEX_PATH)
    parser.add_argument("--dreamsim-metadata-path", default=DEFAULT_DREAMSIM_METADATA_PATH)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            room_id=args.room_id,
            source_pano_id=args.source_pano_id,
            target_pano_id=args.target_pano_id,
            capture_indices=parse_capture_indices(args.capture_indices),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    render_root = resolve_project_path(args.render_root)
    items = load_memory_metadata(resolve_project_path(args.salad_metadata_path), render_root)
    room_indices = tuple(
        index for index, item in enumerate(items) if item.get("room_id") == args.room_id
    )
    if not room_indices:
        raise SystemExit(f"No memory items found for room: {args.room_id}")

    pano_graph = load_optional_pano_graph(resolve_project_path(args.artifacts_dir))
    requested_recipes = parse_recipes(args.recipes)
    recipes = build_recipe_scores(
        requested_recipes=requested_recipes,
        items=items,
        item_indices=room_indices,
        output_dir=output_dir,
        render_root=render_root,
        salad_index_path=resolve_project_path(args.salad_index_path),
        salad_metadata_path=resolve_project_path(args.salad_metadata_path),
        dreamsim_index_path=resolve_project_path(args.dreamsim_index_path),
        dreamsim_metadata_path=resolve_project_path(args.dreamsim_metadata_path),
        device=args.device,
        batch_size=args.batch_size,
    )

    stages: dict[str, list[EvaluationCase]] = {}
    if args.stage in {"case", "both"}:
        stages["case"] = build_case_stage(
            items=items,
            room_indices=room_indices,
            source_pano_id=args.source_pano_id,
            target_pano_id=args.target_pano_id,
            capture_indices=parse_capture_indices(args.capture_indices),
        )
    if args.stage in {"room", "both"}:
        if pano_graph is None:
            raise SystemExit(f"No pano_graph.json found under {args.artifacts_dir}")
        stages["room"] = build_room_adjacency_stage(
            items=items,
            room_indices=room_indices,
            pano_graph=pano_graph,
        )

    stage_results: dict[str, dict[str, list[dict]]] = {}
    ranking_rows: list[dict] = []
    summary: dict = {
        "configuration": {
            "room_id": args.room_id,
            "source_pano_id": args.source_pano_id,
            "target_pano_id": args.target_pano_id,
            "capture_indices": parse_capture_indices(args.capture_indices),
            "branching_factor": int(args.branching_factor),
            "recipes": requested_recipes,
            "ranking_policy": (
                "child ranking uses only per-recipe similarity scores; pano and heading "
                "are used only for case construction and evaluation labels"
            ),
        },
        "stages": {},
    }
    for stage_name, cases in stages.items():
        stage_results[stage_name] = {}
        summary["stages"][stage_name] = {"case_count": len(cases), "recipes": {}}
        for recipe in recipes:
            case_results = evaluate_cases(
                recipe=recipe,
                cases=cases,
                items=items,
                branching_factor=args.branching_factor,
                top_k=args.top_k,
                ranking_rows=ranking_rows,
            )
            stage_results[stage_name][recipe.name] = case_results
            metrics = summarize_case_results(
                case_results,
                branching_factor=args.branching_factor,
            )
            summary["stages"][stage_name]["recipes"][recipe.name] = {
                **metrics,
                "component_recipes": list(recipe.component_names),
            }
    if "case" in summary["stages"]:
        summary["stages"]["case"]["passing_recipes"] = [
            name
            for name, metrics in summary["stages"]["case"]["recipes"].items()
            if metrics["top1_accuracy"] == 1.0
        ]
    summary["best_recipes"] = best_recipes_by_stage(summary["stages"])

    summary_path = output_dir / "summary.json"
    rankings_path = output_dir / "rankings.csv"
    gallery_path = output_dir / "gallery.html"
    summary["outputs"] = {
        "summary_json": str(summary_path),
        "rankings_csv": str(rankings_path),
        "gallery_html": str(gallery_path),
    }
    write_json(summary_path, summary)
    write_rankings_csv(rankings_path, ranking_rows)
    write_gallery(
        gallery_path,
        items=items,
        stage_results=stage_results,
        top_k=args.top_k,
        stage2_limit=args.gallery_stage2_limit,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_capture_indices(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("capture-indices must not be empty.")
        return [int(token) for token in tokens]
    return [int(item) for item in value]


def parse_recipes(value: str | Sequence[str]) -> list[str]:
    recipes = [token.strip() for token in str(",".join(value) if not isinstance(value, str) else value).split(",")]
    parsed = [recipe for recipe in recipes if recipe]
    unknown = set(parsed) - set(DEFAULT_RECIPES)
    if unknown:
        raise ValueError(f"Unknown recipe(s): {', '.join(sorted(unknown))}")
    if not parsed:
        raise ValueError("At least one recipe is required.")
    return parsed


def default_output_dir(
    *,
    room_id: str,
    source_pano_id: str,
    target_pano_id: str,
    capture_indices: Sequence[int],
) -> Path:
    captures = "_".join(str(index) for index in capture_indices)
    return resolve_project_path(
        Path("outputs")
        / "passage_memory_tree_child_similarity"
        / (
            f"{room_token(room_id)}_"
            f"{safe_name(source_pano_id)}_to_{safe_name(target_pano_id)}_"
            f"captures_{captures}"
        )
    )


def build_case_stage(
    *,
    items: Sequence[dict],
    room_indices: Sequence[int],
    source_pano_id: str,
    target_pano_id: str,
    capture_indices: Sequence[int],
) -> list[EvaluationCase]:
    cases = []
    for capture_index in capture_indices:
        source_index = find_item_index(
            items,
            pano_id=source_pano_id,
            capture_index=int(capture_index),
        )
        expected_index = expected_target_for_source(
            items=items,
            source_index=source_index,
            target_pano_id=target_pano_id,
        )
        candidates = candidate_pool_for_source(
            items=items,
            room_indices=room_indices,
            source_pano_id=source_pano_id,
        )
        cases.append(
            EvaluationCase(
                stage="case",
                case_id=f"{source_pano_id}:{capture_index}->{target_pano_id}",
                source_index=source_index,
                expected_index=expected_index,
                candidate_indices=tuple(candidates),
            )
        )
    return cases


def build_room_adjacency_stage(
    *,
    items: Sequence[dict],
    room_indices: Sequence[int],
    pano_graph: Mapping[str, dict],
) -> list[EvaluationCase]:
    room_panos = sorted(
        {
            str(items[index].get("pano_id"))
            for index in room_indices
            if isinstance(items[index].get("pano_id"), str)
        }
    )
    room_pano_set = set(room_panos)
    source_indices_by_pano = {
        pano_id: sorted(
            [
                index
                for index in room_indices
                if items[index].get("pano_id") == pano_id
            ],
            key=lambda index: int(items[index].get("capture_index") or 0),
        )
        for pano_id in room_panos
    }
    cases = []
    for source_pano_id in room_panos:
        node = pano_graph.get(source_pano_id)
        if not isinstance(node, dict):
            continue
        for edge in node.get("neighbors", []) or []:
            if not isinstance(edge, dict):
                continue
            target_pano_id = edge.get("target_pano_id")
            if not isinstance(target_pano_id, str) or target_pano_id not in room_pano_set:
                continue
            if target_pano_id == source_pano_id:
                continue
            candidates = candidate_pool_for_source(
                items=items,
                room_indices=room_indices,
                source_pano_id=source_pano_id,
            )
            for source_index in source_indices_by_pano[source_pano_id]:
                expected_index = expected_target_for_source(
                    items=items,
                    source_index=source_index,
                    target_pano_id=target_pano_id,
                )
                cases.append(
                    EvaluationCase(
                        stage="room",
                        case_id=(
                            f"{source_pano_id}:"
                            f"{items[source_index].get('capture_index')}->{target_pano_id}"
                        ),
                        source_index=source_index,
                        expected_index=expected_index,
                        candidate_indices=tuple(candidates),
                    )
                )
    return cases


def find_item_index(items: Sequence[dict], *, pano_id: str, capture_index: int) -> int:
    for index, item in enumerate(items):
        if item.get("pano_id") == pano_id and item.get("capture_index") == capture_index:
            return index
    raise KeyError(f"Missing memory item for pano={pano_id} capture_index={capture_index}")


def expected_target_for_source(
    *,
    items: Sequence[dict],
    source_index: int,
    target_pano_id: str,
) -> int:
    source_heading = float(items[source_index]["capture_heading"])
    target_indices = [
        index for index, item in enumerate(items) if item.get("pano_id") == target_pano_id
    ]
    if not target_indices:
        raise KeyError(f"Target pano has no memory items: {target_pano_id}")
    return min(
        target_indices,
        key=lambda index: (
            angular_distance_deg(source_heading, float(items[index]["capture_heading"])),
            int(items[index].get("capture_index") or 0),
        ),
    )


def candidate_pool_for_source(
    *,
    items: Sequence[dict],
    room_indices: Sequence[int],
    source_pano_id: str,
) -> list[int]:
    return [
        int(index)
        for index in room_indices
        if items[int(index)].get("pano_id") != source_pano_id
    ]


def angular_distance_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def build_recipe_scores(
    *,
    requested_recipes: Sequence[str],
    items: Sequence[dict],
    item_indices: Sequence[int],
    output_dir: Path,
    render_root: Path,
    salad_index_path: Path,
    salad_metadata_path: Path,
    dreamsim_index_path: Path,
    dreamsim_metadata_path: Path,
    device: str,
    batch_size: int,
) -> list[RecipeScores]:
    requested = list(requested_recipes)
    components: dict[str, RecipeScores] = {}

    def require_salad_full() -> RecipeScores:
        if "salad_full" not in components:
            components["salad_full"] = load_full_embedding_recipe(
                name="salad_full",
                items=items,
                item_indices=item_indices,
                index_path=salad_index_path,
                metadata_path=salad_metadata_path,
                render_root=render_root,
            )
        return components["salad_full"]

    def require_dreamsim_full() -> RecipeScores:
        if "dreamsim_full" not in components:
            components["dreamsim_full"] = load_full_embedding_recipe(
                name="dreamsim_full",
                items=items,
                item_indices=item_indices,
                index_path=dreamsim_index_path,
                metadata_path=dreamsim_metadata_path,
                render_root=render_root,
            )
        return components["dreamsim_full"]

    crop_names = {"salad_center_70", "salad_lower_walkable"}
    needs_crop = bool(set(requested) & crop_names or "salad_multi_crop_max" in requested)
    crop_embedder = None
    if needs_crop:
        crop_embedder = create_memory_tree_embedder(
            similarity_backend="salad",
            dreamsim_type="ensemble",
            device=device,
            batch_size=batch_size,
        )

    def require_crop(name: str) -> RecipeScores:
        if name not in components:
            if crop_embedder is None:
                raise RuntimeError("Crop embedder was not initialized.")
            crop_box = {
                "salad_center_70": (0.15, 0.15, 0.85, 0.85),
                "salad_lower_walkable": (0.10, 0.40, 0.90, 1.00),
            }[name]
            components[name] = build_crop_recipe(
                name=name,
                items=items,
                item_indices=item_indices,
                output_dir=output_dir,
                crop_box=crop_box,
                image_embedder=crop_embedder,
            )
        return components[name]

    recipes = []
    for recipe_name in requested:
        if recipe_name == "dreamsim_full":
            recipes.append(require_dreamsim_full())
        elif recipe_name == "salad_full":
            recipes.append(require_salad_full())
        elif recipe_name in crop_names:
            recipes.append(require_crop(recipe_name))
        elif recipe_name == "salad_multi_crop_max":
            recipes.append(
                RecipeScores.max_of(
                    name="salad_multi_crop_max",
                    recipes=[
                        require_salad_full(),
                        require_crop("salad_center_70"),
                        require_crop("salad_lower_walkable"),
                    ],
                )
            )
        else:
            raise ValueError(f"Unsupported recipe: {recipe_name}")
    return recipes


def load_full_embedding_recipe(
    *,
    name: str,
    items: Sequence[dict],
    item_indices: Sequence[int],
    index_path: Path,
    metadata_path: Path,
    render_root: Path,
) -> RecipeScores:
    del render_root
    metadata_items = load_embedding_metadata(metadata_path)
    metadata_row_by_key = {
        memory_key(item): row for row, item in enumerate(metadata_items)
    }
    payload = np.load(index_path)
    embeddings = np.asarray(payload["image_embeddings"], dtype=np.float32)
    aligned = []
    for item_index in item_indices:
        key = memory_key(items[int(item_index)])
        row = metadata_row_by_key.get(key)
        if row is None:
            raise KeyError(f"{name} metadata is missing memory key: {key}")
        aligned.append(embeddings[row])
    return RecipeScores.from_embeddings(
        name=name,
        item_indices=item_indices,
        embeddings=np.asarray(aligned, dtype=np.float32),
    )


def load_embedding_metadata(metadata_path: Path) -> list[dict]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Metadata contains no items: {metadata_path}")
    return [dict(item) for item in items if isinstance(item, dict)]


def build_crop_recipe(
    *,
    name: str,
    items: Sequence[dict],
    item_indices: Sequence[int],
    output_dir: Path,
    crop_box: tuple[float, float, float, float],
    image_embedder,
) -> RecipeScores:
    memory_keys = [memory_key(items[int(index)]) for index in item_indices]
    cache_path = output_dir / "embedding_cache" / f"{name}.npz"
    cached = load_embedding_cache(cache_path, memory_keys)
    if cached is not None:
        return RecipeScores.from_embeddings(
            name=name,
            item_indices=item_indices,
            embeddings=cached,
        )

    crop_dir = output_dir / "crop_cache" / name
    crop_paths = [
        crop_image_for_item(
            item=items[int(index)],
            crop_dir=crop_dir,
            crop_box=crop_box,
        )
        for index in item_indices
    ]
    embeddings = np.asarray(image_embedder.encode_image_paths(crop_paths), dtype=np.float32)
    save_embedding_cache(cache_path, memory_keys, embeddings)
    return RecipeScores.from_embeddings(
        name=name,
        item_indices=item_indices,
        embeddings=embeddings,
    )


def load_embedding_cache(path: Path, memory_keys: Sequence[str]) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    cached_keys = [str(value) for value in payload["memory_keys"].tolist()]
    if cached_keys != list(memory_keys):
        return None
    return np.asarray(payload["image_embeddings"], dtype=np.float32)


def save_embedding_cache(path: Path, memory_keys: Sequence[str], embeddings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_keys=np.asarray(list(memory_keys), dtype=str),
        image_embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def crop_image_for_item(
    *,
    item: dict,
    crop_dir: Path,
    crop_box: tuple[float, float, float, float],
) -> Path:
    from PIL import Image

    source = Path(str(item["image_path"]))
    target = crop_dir / f"{safe_name(memory_key(item))}.png"
    if target.exists():
        return target
    crop_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source).convert("RGB") as image:
        width, height = image.size
        left, top, right, bottom = crop_box
        box = (
            max(0, min(width - 1, int(round(left * width)))),
            max(0, min(height - 1, int(round(top * height)))),
            max(1, min(width, int(round(right * width)))),
            max(1, min(height, int(round(bottom * height)))),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Invalid crop box {crop_box} for {source}")
        image.crop(box).save(target)
    return target


def evaluate_cases(
    *,
    recipe: RecipeScores,
    cases: Sequence[EvaluationCase],
    items: Sequence[dict],
    branching_factor: int,
    top_k: int,
    ranking_rows: list[dict],
) -> list[dict]:
    results = []
    for case in cases:
        scores_by_index = {
            candidate_index: recipe.similarity(case.source_index, candidate_index)
            for candidate_index in case.candidate_indices
        }
        ranked = rank_candidates_by_similarity(case.candidate_indices, scores_by_index)
        expected_rank = next(
            (
                rank
                for rank, candidate in enumerate(ranked, start=1)
                if candidate.candidate_index == case.expected_index
            ),
            None,
        )
        if expected_rank is None:
            raise RuntimeError(f"Expected child is absent from candidate pool: {case.case_id}")
        top_candidates = ranked[: max(int(top_k), 1)]
        result = {
            "stage": case.stage,
            "case_id": case.case_id,
            "source_index": case.source_index,
            "expected_index": case.expected_index,
            "expected_rank": expected_rank,
            "expected_similarity": scores_by_index[case.expected_index],
            "top_candidates": [
                {
                    "candidate_index": candidate.candidate_index,
                    "similarity": candidate.similarity,
                    "expected": candidate.candidate_index == case.expected_index,
                }
                for candidate in top_candidates
            ],
        }
        results.append(result)
        for rank, candidate in enumerate(ranked, start=1):
            row = ranking_row(
                recipe_name=recipe.name,
                case=case,
                items=items,
                rank=rank,
                candidate=candidate,
                branching_factor=branching_factor,
            )
            ranking_rows.append(row)
    return results


def rank_candidates_by_similarity(
    candidate_indices: Sequence[int],
    similarities_by_index: Mapping[int, float],
) -> list[RankedCandidate]:
    ranked = [
        RankedCandidate(candidate_index=int(index), similarity=float(similarities_by_index[int(index)]))
        for index in candidate_indices
    ]
    return sorted(ranked, key=lambda item: (-item.similarity, item.candidate_index))


def ranking_row(
    *,
    recipe_name: str,
    case: EvaluationCase,
    items: Sequence[dict],
    rank: int,
    candidate: RankedCandidate,
    branching_factor: int,
) -> dict:
    source = items[case.source_index]
    expected = items[case.expected_index]
    candidate_item = items[candidate.candidate_index]
    return {
        "stage": case.stage,
        "recipe": recipe_name,
        "case_id": case.case_id,
        "source_pano_id": source.get("pano_id"),
        "source_capture_index": source.get("capture_index"),
        "source_heading": source.get("capture_heading"),
        "expected_pano_id": expected.get("pano_id"),
        "expected_capture_index": expected.get("capture_index"),
        "expected_heading": expected.get("capture_heading"),
        "candidate_rank": rank,
        "candidate_pano_id": candidate_item.get("pano_id"),
        "candidate_capture_index": candidate_item.get("capture_index"),
        "candidate_heading": candidate_item.get("capture_heading"),
        "candidate_similarity": candidate.similarity,
        "expected": candidate.candidate_index == case.expected_index,
        "selected_by_branching_factor": rank <= int(branching_factor),
    }


def summarize_case_results(
    case_results: Sequence[dict],
    *,
    branching_factor: int,
) -> dict:
    ranks = [int(result["expected_rank"]) for result in case_results]
    return summarize_expected_ranks(ranks, branching_factor=branching_factor)


def summarize_expected_ranks(ranks: Sequence[int], *, branching_factor: int) -> dict:
    if not ranks:
        return {
            "case_count": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "branch_accuracy": 0.0,
            "mrr": 0.0,
            "mean_expected_rank": None,
        }
    numeric = [int(rank) for rank in ranks]
    count = len(numeric)
    return {
        "case_count": count,
        "top1_accuracy": sum(rank == 1 for rank in numeric) / count,
        "top3_accuracy": sum(rank <= 3 for rank in numeric) / count,
        "branch_accuracy": sum(rank <= int(branching_factor) for rank in numeric) / count,
        "mrr": sum(1.0 / rank for rank in numeric) / count,
        "mean_expected_rank": sum(numeric) / count,
        "expected_ranks": numeric,
    }


def best_recipes_by_stage(stages: Mapping[str, dict]) -> dict:
    best = {}
    for stage_name, stage_payload in stages.items():
        recipes = stage_payload.get("recipes", {})
        if not isinstance(recipes, dict) or not recipes:
            continue
        best[stage_name] = max(
            recipes.items(),
            key=lambda item: (
                float(item[1].get("top1_accuracy", 0.0)),
                float(item[1].get("top3_accuracy", 0.0)),
                float(item[1].get("mrr", 0.0)),
                -float(item[1].get("mean_expected_rank") or 1e12),
            ),
        )[0]
    return best


def write_rankings_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "stage",
        "recipe",
        "case_id",
        "source_pano_id",
        "source_capture_index",
        "source_heading",
        "expected_pano_id",
        "expected_capture_index",
        "expected_heading",
        "candidate_rank",
        "candidate_pano_id",
        "candidate_capture_index",
        "candidate_heading",
        "candidate_similarity",
        "expected",
        "selected_by_branching_factor",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_gallery(
    path: Path,
    *,
    items: Sequence[dict],
    stage_results: Mapping[str, Mapping[str, Sequence[dict]]],
    top_k: int,
    stage2_limit: int,
) -> None:
    gallery_cases = select_gallery_cases(stage_results, stage2_limit=stage2_limit)
    asset_map = copy_gallery_assets(path.parent / "gallery_assets", items, gallery_cases)
    sections = []
    for stage_name, recipes in gallery_cases.items():
        recipe_html = []
        for recipe_name, case_results in recipes.items():
            cards = [
                gallery_case_card(
                    items=items,
                    asset_map=asset_map,
                    recipe_name=recipe_name,
                    case_result=case_result,
                    top_k=top_k,
                )
                for case_result in case_results
            ]
            recipe_html.append(
                "\n".join(
                    [
                        f"<h3>{html.escape(recipe_name)}</h3>",
                        "<div class='case-grid'>",
                        *cards,
                        "</div>",
                    ]
                )
            )
        sections.append(
            "\n".join(
                [
                    f"<section><h2>{html.escape(stage_name)}</h2>",
                    *recipe_html,
                    "</section>",
                ]
            )
        )
    document = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Memory tree child similarity</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,sans-serif;margin:24px;background:#f7f7f4;color:#202020}",
            "section{margin-bottom:36px}",
            ".case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}",
            ".case{background:white;border:1px solid #ddd;border-radius:8px;padding:12px}",
            ".row{display:flex;gap:10px;align-items:flex-start;overflow-x:auto}",
            ".frame{min-width:140px;max-width:160px}",
            ".frame img{width:140px;height:140px;object-fit:cover;border:1px solid #ccc;background:#eee}",
            ".label{font-size:12px;line-height:1.35;margin-top:4px}",
            ".mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px}",
            ".hit{color:#126b38;font-weight:700}",
            ".miss{color:#9b2c2c;font-weight:700}",
            "</style></head><body>",
            "<h1>Memory tree child similarity</h1>",
            "<p>Ranking uses recipe similarity only. Pano and heading labels are shown for evaluation.</p>",
            *sections,
            "</body></html>",
        ]
    )
    path.write_text(document, encoding="utf-8")


def select_gallery_cases(
    stage_results: Mapping[str, Mapping[str, Sequence[dict]]],
    *,
    stage2_limit: int,
) -> dict[str, dict[str, list[dict]]]:
    selected: dict[str, dict[str, list[dict]]] = {}
    for stage_name, recipes in stage_results.items():
        selected[stage_name] = {}
        for recipe_name, case_results in recipes.items():
            results = list(case_results)
            if stage_name == "room":
                failures = [result for result in results if int(result["expected_rank"]) != 1]
                results = sorted(
                    failures or results,
                    key=lambda result: int(result["expected_rank"]),
                    reverse=True,
                )[: max(int(stage2_limit), 0)]
            selected[stage_name][recipe_name] = results
    return selected


def copy_gallery_assets(
    asset_dir: Path,
    items: Sequence[dict],
    stage_results: Mapping[str, Mapping[str, Sequence[dict]]],
) -> dict[str, str]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    needed: dict[str, dict] = {}
    for recipes in stage_results.values():
        for case_results in recipes.values():
            for result in case_results:
                for index in [result["source_index"], result["expected_index"]]:
                    item = items[int(index)]
                    needed[memory_key(item)] = item
                for candidate in result.get("top_candidates", []):
                    item = items[int(candidate["candidate_index"])]
                    needed[memory_key(item)] = item
    asset_map = {}
    for key, item in needed.items():
        source = Path(str(item.get("image_path") or ""))
        if not source.exists():
            continue
        target = asset_dir / f"{safe_name(key)}{source.suffix or '.png'}"
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        asset_map[key] = str(Path(asset_dir.name) / target.name)
    return asset_map


def gallery_case_card(
    *,
    items: Sequence[dict],
    asset_map: Mapping[str, str],
    recipe_name: str,
    case_result: dict,
    top_k: int,
) -> str:
    del recipe_name
    source = items[int(case_result["source_index"])]
    expected = items[int(case_result["expected_index"])]
    top_candidates = case_result.get("top_candidates", [])[: max(int(top_k), 1)]
    status_class = "hit" if int(case_result["expected_rank"]) == 1 else "miss"
    frames = [
        image_frame("Source", source, asset_map),
        image_frame(
            f"Expected rank {case_result['expected_rank']}",
            expected,
            asset_map,
            extra_class=status_class,
        ),
    ]
    for rank, candidate in enumerate(top_candidates, start=1):
        candidate_item = items[int(candidate["candidate_index"])]
        label = f"Top {rank} sim={float(candidate['similarity']):.3f}"
        if candidate.get("expected"):
            label += " expected"
        frames.append(image_frame(label, candidate_item, asset_map))
    return "\n".join(
        [
            "<article class='case'>",
            f"<h4>{html.escape(str(case_result['case_id']))}</h4>",
            "<div class='row'>",
            *frames,
            "</div>",
            "</article>",
        ]
    )


def image_frame(
    label: str,
    item: dict,
    asset_map: Mapping[str, str],
    *,
    extra_class: str = "",
) -> str:
    key = memory_key(item)
    src = asset_map.get(key)
    image = (
        f"<img src='{html.escape(src)}' alt='{html.escape(label)}'>"
        if src
        else "<div style='width:140px;height:140px;background:#eee'>missing</div>"
    )
    heading = item.get("capture_heading")
    heading_text = f"{float(heading):.1f}" if isinstance(heading, (int, float)) else "-"
    classes = "label " + extra_class
    return "\n".join(
        [
            "<div class='frame'>",
            image,
            f"<div class='{html.escape(classes.strip())}'>{html.escape(label)}</div>",
            f"<div class='mono'>{html.escape(str(item.get('pano_id')))} #{item.get('capture_index')} h={heading_text}</div>",
            "</div>",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
