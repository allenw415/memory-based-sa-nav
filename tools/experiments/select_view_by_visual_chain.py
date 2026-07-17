from __future__ import annotations

import argparse
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
    memory_key,
    normalize_rows,
    resolve_project_path,
    safe_name,
    write_json,
)
from memory_nav.data.memory_localization import (  # noqa: E402
    DEFAULT_SIGLIP2_MODEL,
    create_image_embedder,
)
from tools.experiments.eval_memory_tree_child_similarity import (  # noqa: E402
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_DREAMSIM_INDEX_PATH,
    DEFAULT_DREAMSIM_METADATA_PATH,
    DEFAULT_RECIPES,
    DEFAULT_RENDER_ROOT,
    DEFAULT_SALAD_INDEX_PATH,
    DEFAULT_SALAD_METADATA_PATH,
    RecipeScores,
    crop_image_for_item,
    load_embedding_cache,
    load_embedding_metadata,
    save_embedding_cache,
)


DEFAULT_CURRENT_PANO_ID = "Li54te8XaSyXgj2x_c2msA"
DEFAULT_RECIPE = "salad_full"
DEFAULT_MAX_DEPTH = 0
DEFAULT_BRANCHING_FACTOR = 1
DEFAULT_TARGET_HIT_THRESHOLD = 0.9
DEFAULT_CONTINUITY_THRESHOLD = 0.0
DEFAULT_SIGLIP2_INDEX_PATH = "artifacts/memory_localization/floor0_1_siglip2_images_fov90.npz"
DEFAULT_SIGLIP2_METADATA_PATH = (
    "artifacts/memory_localization/floor0_1_siglip2_images_fov90.metadata.json"
)
DEFAULT_DINOV2_PATCH_MODEL = "facebook/dinov2-base"
DEFAULT_DINOV2_PATCH_TOP_K = 5
DEFAULT_DINOV2_PATCH_MAX_PATCHES = 24
DEFAULT_DINOV2_TARGET_MATCH_MODE = "target_to_candidate"
DINOV2_TARGET_MATCH_MODES = ("target_to_candidate", "symmetric")
VISUAL_CHAIN_RECIPES = (*DEFAULT_RECIPES, "siglip2_full", "dinov2_patch_topk")


@dataclass(frozen=True)
class ChainPath:
    root_index: int
    item_indices: tuple[int, ...]
    edge_similarities: tuple[float, ...]
    stop_reason: str = "searching"


@dataclass(frozen=True)
class ChainSummary:
    view_order: int
    capture_index: int
    selected: bool
    target_hit: bool
    hit_depth: int | None
    hit_similarity: float | None
    max_target_similarity: float
    chain_bottleneck_similarity: float
    chain_mean_similarity: float
    stop_reason: str
    nodes: tuple[dict, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a current panorama view by growing visual-similarity chains "
            "toward a target passage image."
        )
    )
    parser.add_argument("--current-pano-id", default=DEFAULT_CURRENT_PANO_ID)
    parser.add_argument("--target-image", required=True)
    parser.add_argument("--room-id")
    parser.add_argument("--recipe", choices=VISUAL_CHAIN_RECIPES, default=DEFAULT_RECIPE)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help="Maximum chain depth. Use 0 to search the finite room pool until target hit or exhaustion.",
    )
    parser.add_argument("--branching-factor", type=int, default=DEFAULT_BRANCHING_FACTOR)
    parser.add_argument("--target-hit-threshold", type=float, default=DEFAULT_TARGET_HIT_THRESHOLD)
    parser.add_argument("--continuity-threshold", type=float, default=DEFAULT_CONTINUITY_THRESHOLD)
    parser.add_argument("--render-root", default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--artifacts-dir", default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--salad-index-path", default=DEFAULT_SALAD_INDEX_PATH)
    parser.add_argument("--salad-metadata-path", default=DEFAULT_SALAD_METADATA_PATH)
    parser.add_argument("--siglip2-index-path", default=DEFAULT_SIGLIP2_INDEX_PATH)
    parser.add_argument("--siglip2-metadata-path", default=DEFAULT_SIGLIP2_METADATA_PATH)
    parser.add_argument("--dreamsim-index-path", default=DEFAULT_DREAMSIM_INDEX_PATH)
    parser.add_argument("--dreamsim-metadata-path", default=DEFAULT_DREAMSIM_METADATA_PATH)
    parser.add_argument("--dinov2-patch-model", default=DEFAULT_DINOV2_PATCH_MODEL)
    parser.add_argument("--dinov2-patch-top-k", type=int, default=DEFAULT_DINOV2_PATCH_TOP_K)
    parser.add_argument("--dinov2-patch-max-patches", type=int, default=DEFAULT_DINOV2_PATCH_MAX_PATCHES)
    parser.add_argument(
        "--dinov2-target-match-mode",
        choices=DINOV2_TARGET_MATCH_MODES,
        default=DEFAULT_DINOV2_TARGET_MATCH_MODE,
        help=(
            "How DINOv2 patch target-hit similarity is computed. "
            "target_to_candidate asks whether target passage patches appear inside a candidate view; "
            "symmetric uses the same bidirectional patch score as parent-child continuity."
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target_image = resolve_project_path(args.target_image)
    if not target_image.exists():
        raise SystemExit(f"Target image does not exist: {target_image}")

    output_dir = (
        resolve_project_path(args.output_dir)
        if args.output_dir
        else default_output_dir(
            current_pano_id=args.current_pano_id,
            target_image=target_image,
            recipe=args.recipe,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    render_root = resolve_project_path(args.render_root)
    items = load_memory_metadata(resolve_project_path(args.salad_metadata_path), render_root)
    room_id = args.room_id or infer_room_id(
        current_pano_id=args.current_pano_id,
        items=items,
        artifacts_dir=resolve_project_path(args.artifacts_dir),
    )
    room_indices = tuple(
        index for index, item in enumerate(items) if item.get("room_id") == room_id
    )
    if not room_indices:
        raise SystemExit(f"No memory items found for room: {room_id}")
    current_view_indices = current_pano_view_indices(
        items=items,
        current_pano_id=args.current_pano_id,
        room_indices=room_indices,
    )

    recipe = build_recipe_with_target(
        recipe_name=args.recipe,
        items=items,
        item_indices=room_indices,
        target_image=target_image,
        output_dir=output_dir,
        render_root=render_root,
        salad_index_path=resolve_project_path(args.salad_index_path),
        salad_metadata_path=resolve_project_path(args.salad_metadata_path),
        siglip2_index_path=resolve_project_path(args.siglip2_index_path),
        siglip2_metadata_path=resolve_project_path(args.siglip2_metadata_path),
        dreamsim_index_path=resolve_project_path(args.dreamsim_index_path),
        dreamsim_metadata_path=resolve_project_path(args.dreamsim_metadata_path),
        dinov2_patch_model=args.dinov2_patch_model,
        dinov2_patch_top_k=args.dinov2_patch_top_k,
        dinov2_patch_max_patches=args.dinov2_patch_max_patches,
        dinov2_target_match_mode=args.dinov2_target_match_mode,
        device=args.device,
        batch_size=args.batch_size,
    )
    candidate_indices = [
        index for index in room_indices if items[index].get("pano_id") != args.current_pano_id
    ]
    target_match_index = find_matching_image_index(items, target_image)
    if target_match_index is not None:
        candidate_indices = [index for index in candidate_indices if index != target_match_index]

    chains = build_view_chains(
        items=items,
        recipe=recipe.recipe_scores,
        target_similarities=recipe.target_similarities,
        current_view_indices=current_view_indices,
        candidate_indices=candidate_indices,
        max_depth=args.max_depth,
        branching_factor=args.branching_factor,
        target_hit_threshold=args.target_hit_threshold,
    )
    selected = select_chain(
        chains,
        continuity_threshold=args.continuity_threshold,
    )
    chains = [
        ChainSummary(
            **{
                **chain.__dict__,
                "selected": chain.view_order == selected.view_order,
            }
        )
        for chain in chains
    ]
    selected = next(chain for chain in chains if chain.selected)

    payload = selection_payload(
        args=args,
        room_id=room_id,
        target_image=target_image,
        recipe=recipe,
        selected=selected,
        chains=chains,
        candidate_count=len(candidate_indices),
    )
    chains_payload = {
        "target_image": str(target_image),
        "chains": [chain_to_dict(chain) for chain in chains],
    }
    selection_path = output_dir / "selection.json"
    chains_path = output_dir / "chains.json"
    gallery_path = output_dir / "gallery.html"
    payload["outputs"] = {
        "selection_json": str(selection_path),
        "chains_json": str(chains_path),
        "gallery_html": str(gallery_path),
    }
    write_json(selection_path, payload)
    write_json(chains_path, chains_payload)
    write_gallery(
        gallery_path,
        items=items,
        target_image=target_image,
        chains=chains,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


@dataclass(frozen=True)
class RecipeWithTarget:
    recipe_scores: RecipeScores
    target_similarities: dict[int, float]
    component_recipes: tuple[str, ...] = ()


def default_output_dir(*, current_pano_id: str, target_image: Path, recipe: str) -> Path:
    return resolve_project_path(
        Path("outputs")
        / "visual_chain_direction"
        / f"{safe_name(current_pano_id)}_{safe_name(target_image.stem)}_{safe_name(recipe)}"
    )


def infer_room_id(*, current_pano_id: str, items: Sequence[dict], artifacts_dir: Path) -> str:
    grounding_path = artifacts_dir / "pano_room_grounding.json"
    if grounding_path.exists():
        payload = json.loads(grounding_path.read_text(encoding="utf-8"))
        mappings = payload.get("mappings") if isinstance(payload, dict) else None
        room_id = mappings.get(current_pano_id) if isinstance(mappings, dict) else None
        if isinstance(room_id, str) and room_id and room_id.lower() != "null":
            return room_id
    candidates = {
        str(item.get("room_id"))
        for item in items
        if item.get("pano_id") == current_pano_id and item.get("room_id")
    }
    candidates.discard("None")
    candidates.discard("null")
    if len(candidates) == 1:
        return next(iter(candidates))
    raise SystemExit(
        f"Could not infer room for pano {current_pano_id}; pass --room-id explicitly."
    )


def current_pano_view_indices(
    *,
    items: Sequence[dict],
    current_pano_id: str,
    room_indices: Sequence[int],
) -> tuple[int, ...]:
    current = sorted(
        [
            index
            for index in room_indices
            if items[index].get("pano_id") == current_pano_id
        ],
        key=lambda index: int(items[index].get("capture_index") or 0),
    )
    if len(current) != 8:
        raise SystemExit(f"Expected 8 current views for pano {current_pano_id}, got {len(current)}.")
    return tuple(current)


def build_recipe_with_target(
    *,
    recipe_name: str,
    items: Sequence[dict],
    item_indices: Sequence[int],
    target_image: Path,
    output_dir: Path,
    render_root: Path,
    salad_index_path: Path,
    salad_metadata_path: Path,
    siglip2_index_path: Path,
    siglip2_metadata_path: Path,
    dreamsim_index_path: Path,
    dreamsim_metadata_path: Path,
    dinov2_patch_model: str,
    dinov2_patch_top_k: int,
    dinov2_patch_max_patches: int,
    dinov2_target_match_mode: str,
    device: str,
    batch_size: int,
) -> RecipeWithTarget:
    parse_visual_chain_recipe(recipe_name)
    component_cache: dict[str, tuple[RecipeScores, dict[int, float]]] = {}

    def component(name: str) -> tuple[RecipeScores, dict[int, float]]:
        if name not in component_cache:
            component_cache[name] = build_component_recipe_with_target(
                recipe_name=name,
                items=items,
                item_indices=item_indices,
                target_image=target_image,
                output_dir=output_dir,
                render_root=render_root,
                salad_index_path=salad_index_path,
                salad_metadata_path=salad_metadata_path,
                siglip2_index_path=siglip2_index_path,
                siglip2_metadata_path=siglip2_metadata_path,
                dreamsim_index_path=dreamsim_index_path,
                dreamsim_metadata_path=dreamsim_metadata_path,
                dinov2_patch_model=dinov2_patch_model,
                dinov2_patch_top_k=dinov2_patch_top_k,
                dinov2_patch_max_patches=dinov2_patch_max_patches,
                dinov2_target_match_mode=dinov2_target_match_mode,
                device=device,
                batch_size=batch_size,
            )
        return component_cache[name]

    if recipe_name == "salad_multi_crop_max":
        components = [
            component("salad_full"),
            component("salad_center_70"),
            component("salad_lower_walkable"),
        ]
        recipe_scores = RecipeScores.max_of(
            name="salad_multi_crop_max",
            recipes=[item[0] for item in components],
        )
        target_similarities = {
            item_index: max(scores[item_index] for _recipe, scores in components)
            for item_index in item_indices
        }
        return RecipeWithTarget(
            recipe_scores=recipe_scores,
            target_similarities=target_similarities,
            component_recipes=tuple(recipe.name for recipe, _scores in components),
        )

    recipe_scores, target_similarities = component(recipe_name)
    return RecipeWithTarget(
        recipe_scores=recipe_scores,
        target_similarities=target_similarities,
    )


def build_component_recipe_with_target(
    *,
    recipe_name: str,
    items: Sequence[dict],
    item_indices: Sequence[int],
    target_image: Path,
    output_dir: Path,
    render_root: Path,
    salad_index_path: Path,
    salad_metadata_path: Path,
    siglip2_index_path: Path,
    siglip2_metadata_path: Path,
    dreamsim_index_path: Path,
    dreamsim_metadata_path: Path,
    dinov2_patch_model: str,
    dinov2_patch_top_k: int,
    dinov2_patch_max_patches: int,
    dinov2_target_match_mode: str,
    device: str,
    batch_size: int,
) -> tuple[RecipeScores, dict[int, float]]:
    if recipe_name in {"salad_full", "dreamsim_full", "siglip2_full"}:
        full_recipe_config = {
            "salad_full": (salad_index_path, salad_metadata_path, "salad"),
            "dreamsim_full": (dreamsim_index_path, dreamsim_metadata_path, "dreamsim"),
            "siglip2_full": (siglip2_index_path, siglip2_metadata_path, "siglip2"),
        }
        index_path, metadata_path, backend = full_recipe_config[recipe_name]
        embeddings = load_aligned_full_embeddings(
            items=items,
            item_indices=item_indices,
            index_path=index_path,
            metadata_path=metadata_path,
        )
        recipe_scores = RecipeScores.from_embeddings(
            name=recipe_name,
            item_indices=item_indices,
            embeddings=embeddings,
        )
        target_embedding = encode_paths(
            [target_image],
            similarity_backend=backend,
            device=device,
            batch_size=batch_size,
        )[0]
        target_similarities = similarities_to_target(
            item_indices=item_indices,
            embeddings=embeddings,
            target_embedding=target_embedding,
        )
        return recipe_scores, target_similarities

    if recipe_name == "dinov2_patch_topk":
        patch_features = load_or_encode_dinov2_patch_features(
            recipe_name=recipe_name,
            items=items,
            item_indices=item_indices,
            output_dir=output_dir,
            model_name=dinov2_patch_model,
            max_patches=dinov2_patch_max_patches,
            device=device,
            batch_size=batch_size,
        )
        target_features = encode_dinov2_patch_paths(
            [target_image],
            model_name=dinov2_patch_model,
            max_patches=dinov2_patch_max_patches,
            device=device,
            batch_size=batch_size,
        )[0]
        recipe_scores = RecipeScores(
            name=recipe_name,
            item_indices=tuple(int(index) for index in item_indices),
            similarity_matrix=patch_topk_similarity_matrix(
                patch_features,
                top_k=dinov2_patch_top_k,
            ),
        )
        target_similarities = patch_topk_similarities_to_target(
            item_indices=item_indices,
            patch_features=patch_features,
            target_features=target_features,
            top_k=dinov2_patch_top_k,
            target_match_mode=dinov2_target_match_mode,
        )
        return recipe_scores, target_similarities

    if recipe_name in {"salad_center_70", "salad_lower_walkable"}:
        crop_box = {
            "salad_center_70": (0.15, 0.15, 0.85, 0.85),
            "salad_lower_walkable": (0.10, 0.40, 0.90, 1.00),
        }[recipe_name]
        embeddings = load_or_encode_crop_embeddings(
            recipe_name=recipe_name,
            items=items,
            item_indices=item_indices,
            output_dir=output_dir,
            crop_box=crop_box,
            device=device,
            batch_size=batch_size,
        )
        recipe_scores = RecipeScores.from_embeddings(
            name=recipe_name,
            item_indices=item_indices,
            embeddings=embeddings,
        )
        target_item = {
            "pano_id": f"external_{safe_name(target_image.stem)}",
            "capture_index": 0,
            "image_path": str(target_image),
        }
        target_crop = crop_image_for_item(
            item=target_item,
            crop_dir=output_dir / "target_crop_cache" / recipe_name,
            crop_box=crop_box,
        )
        target_embedding = encode_paths(
            [target_crop],
            similarity_backend="salad",
            device=device,
            batch_size=batch_size,
        )[0]
        target_similarities = similarities_to_target(
            item_indices=item_indices,
            embeddings=embeddings,
            target_embedding=target_embedding,
        )
        return recipe_scores, target_similarities

    raise ValueError(f"Unsupported recipe: {recipe_name}")


def parse_visual_chain_recipe(value: str) -> str:
    recipe = str(value).strip()
    if recipe not in VISUAL_CHAIN_RECIPES:
        raise ValueError(f"Unknown recipe: {recipe}")
    return recipe


def load_aligned_full_embeddings(
    *,
    items: Sequence[dict],
    item_indices: Sequence[int],
    index_path: Path,
    metadata_path: Path,
) -> np.ndarray:
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
            raise KeyError(f"Embedding metadata is missing memory key: {key}")
        aligned.append(embeddings[row])
    return np.asarray(aligned, dtype=np.float32)


def load_or_encode_crop_embeddings(
    *,
    recipe_name: str,
    items: Sequence[dict],
    item_indices: Sequence[int],
    output_dir: Path,
    crop_box: tuple[float, float, float, float],
    device: str,
    batch_size: int,
) -> np.ndarray:
    memory_keys = [memory_key(items[int(index)]) for index in item_indices]
    cache_path = output_dir / "embedding_cache" / f"{recipe_name}.npz"
    cached = load_embedding_cache(cache_path, memory_keys)
    if cached is not None:
        return cached
    crop_paths = [
        crop_image_for_item(
            item=items[int(index)],
            crop_dir=output_dir / "crop_cache" / recipe_name,
            crop_box=crop_box,
        )
        for index in item_indices
    ]
    embeddings = encode_paths(
        crop_paths,
        similarity_backend="salad",
        device=device,
        batch_size=batch_size,
    )
    save_embedding_cache(cache_path, memory_keys, embeddings)
    return embeddings


def load_or_encode_dinov2_patch_features(
    *,
    recipe_name: str,
    items: Sequence[dict],
    item_indices: Sequence[int],
    output_dir: Path,
    model_name: str,
    max_patches: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    memory_keys = [memory_key(items[int(index)]) for index in item_indices]
    cache_path = output_dir / "embedding_cache" / f"{recipe_name}_{safe_name(model_name)}_p{int(max_patches)}.npz"
    cached = load_dinov2_patch_cache(
        cache_path,
        memory_keys=memory_keys,
        model_name=model_name,
        max_patches=max_patches,
    )
    if cached is not None:
        return cached
    image_paths = [
        Path(str(items[int(index)].get("image_path") or items[int(index)].get("capture_path")))
        for index in item_indices
    ]
    patch_features = encode_dinov2_patch_paths(
        image_paths,
        model_name=model_name,
        max_patches=max_patches,
        device=device,
        batch_size=batch_size,
    )
    save_dinov2_patch_cache(
        cache_path,
        memory_keys=memory_keys,
        model_name=model_name,
        max_patches=max_patches,
        patch_features=patch_features,
    )
    return patch_features


def load_dinov2_patch_cache(
    path: Path,
    *,
    memory_keys: Sequence[str],
    model_name: str,
    max_patches: int,
) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    cached_keys = [str(value) for value in payload["memory_keys"].tolist()]
    cached_model = str(payload["model_name"].tolist())
    cached_max_patches = int(payload["max_patches"].tolist())
    if (
        cached_keys != list(memory_keys)
        or cached_model != str(model_name)
        or cached_max_patches != int(max_patches)
    ):
        return None
    return np.asarray(payload["patch_features"], dtype=np.float32)


def save_dinov2_patch_cache(
    path: Path,
    *,
    memory_keys: Sequence[str],
    model_name: str,
    max_patches: int,
    patch_features: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        memory_keys=np.asarray(list(memory_keys)),
        model_name=np.asarray(str(model_name)),
        max_patches=np.asarray(int(max_patches)),
        patch_features=np.asarray(patch_features, dtype=np.float32),
    )


def encode_dinov2_patch_paths(
    paths: Sequence[Path],
    *,
    model_name: str,
    max_patches: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    resolved_device = resolve_torch_device(torch, device)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(resolved_device)
    batches = []
    step = max(int(batch_size), 1)
    for start in range(0, len(paths), step):
        batch_paths = [Path(path) for path in paths[start : start + step]]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(resolved_device) for key, value in dict(inputs).items()}
            with torch.inference_mode():
                outputs = model(**inputs)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError(f"DINOv2 model {model_name!r} did not return last_hidden_state.")
            patches = hidden[:, 1:, :]
            patches = patches / patches.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            patch_array = patches.detach().cpu().to(torch.float32).numpy()
            batches.append(select_salient_patch_features(patch_array, max_patches=max_patches))
        finally:
            for image in images:
                image.close()
    if not batches:
        return np.zeros((0, 0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def resolve_torch_device(torch_module, requested_device: str) -> str:
    normalized = (requested_device or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    if torch_module.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch_module.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def select_salient_patch_features(features: np.ndarray, *, max_patches: int) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("features must have shape (image, patch, dim).")
    limit = max(int(max_patches), 1)
    if array.shape[1] <= limit:
        return array
    selected = []
    for image_features in array:
        image_features = normalize_rows(image_features)
        mean_feature = normalize_rows(image_features.mean(axis=0, keepdims=True))[0]
        saliency = 1.0 - (image_features @ mean_feature)
        indices = np.argsort(-saliency, kind="stable")[:limit]
        selected.append(image_features[indices])
    return np.stack(selected, axis=0).astype(np.float32)


def patch_topk_similarity_matrix(patch_features: np.ndarray, *, top_k: int) -> np.ndarray:
    features = np.asarray(patch_features, dtype=np.float32)
    count = int(features.shape[0])
    matrix = np.eye(count, dtype=np.float32)
    for source in range(count):
        for target in range(source + 1, count):
            score = patch_topk_similarity(features[source], features[target], top_k=top_k)
            matrix[source, target] = score
            matrix[target, source] = score
    return matrix


def patch_topk_similarities_to_target(
    *,
    item_indices: Sequence[int],
    patch_features: np.ndarray,
    target_features: np.ndarray,
    top_k: int,
    target_match_mode: str,
) -> dict[int, float]:
    return {
        int(item_index): patch_target_similarity(
            candidate_features=features,
            target_features=target_features,
            top_k=top_k,
            target_match_mode=target_match_mode,
        )
        for item_index, features in zip(item_indices, patch_features, strict=True)
    }


def patch_target_similarity(
    *,
    candidate_features: np.ndarray,
    target_features: np.ndarray,
    top_k: int,
    target_match_mode: str,
) -> float:
    if target_match_mode == "symmetric":
        return patch_topk_similarity(candidate_features, target_features, top_k=top_k)
    if target_match_mode == "target_to_candidate":
        return patch_target_to_candidate_similarity(
            target_features=target_features,
            candidate_features=candidate_features,
            top_k=top_k,
        )
    raise ValueError(f"Unsupported target_match_mode: {target_match_mode}")


def patch_target_to_candidate_similarity(
    *,
    target_features: np.ndarray,
    candidate_features: np.ndarray,
    top_k: int,
) -> float:
    target = normalize_rows(np.asarray(target_features, dtype=np.float32))
    candidate = normalize_rows(np.asarray(candidate_features, dtype=np.float32))
    if target.size == 0 or candidate.size == 0:
        return 0.0
    similarities = target @ candidate.T
    target_best = similarities.max(axis=1)
    k = min(max(int(top_k), 1), int(target_best.shape[0]))
    return topk_mean(target_best, k)


def patch_topk_similarity(source_features: np.ndarray, target_features: np.ndarray, *, top_k: int) -> float:
    source = normalize_rows(np.asarray(source_features, dtype=np.float32))
    target = normalize_rows(np.asarray(target_features, dtype=np.float32))
    if source.size == 0 or target.size == 0:
        return 0.0
    similarities = source @ target.T
    k_source = min(max(int(top_k), 1), int(similarities.shape[0]))
    k_target = min(max(int(top_k), 1), int(similarities.shape[1]))
    source_best = similarities.max(axis=1)
    target_best = similarities.max(axis=0)
    source_score = topk_mean(source_best, k_source)
    target_score = topk_mean(target_best, k_target)
    return float((source_score + target_score) / 2.0)


def topk_mean(values: np.ndarray, k: int) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return 0.0
    limit = min(max(int(k), 1), int(array.size))
    if limit == int(array.size):
        return float(array.mean())
    top_values = np.partition(array, -limit)[-limit:]
    return float(top_values.mean())


def encode_paths(
    paths: Sequence[Path],
    *,
    similarity_backend: str,
    device: str,
    batch_size: int,
) -> np.ndarray:
    if similarity_backend == "siglip2":
        embedder = create_image_embedder(
            model_name=DEFAULT_SIGLIP2_MODEL,
            device=device,
            batch_size=batch_size,
        )
    else:
        embedder = create_memory_tree_embedder(
            similarity_backend=similarity_backend,
            dreamsim_type="ensemble",
            device=device,
            batch_size=batch_size,
        )
    return np.asarray(embedder.encode_image_paths([Path(path) for path in paths]), dtype=np.float32)


def similarities_to_target(
    *,
    item_indices: Sequence[int],
    embeddings,
    target_embedding,
) -> dict[int, float]:
    normalized = normalize_rows(np.asarray(embeddings, dtype=np.float32))
    target_vector = normalize_rows(np.asarray(target_embedding, dtype=np.float32).reshape(1, -1))[0]
    values = normalized @ target_vector
    return {
        int(item_index): float(value)
        for item_index, value in zip(item_indices, values, strict=True)
    }


def find_matching_image_index(items: Sequence[dict], target_image: Path) -> int | None:
    target = target_image.resolve()
    for index, item in enumerate(items):
        raw_path = item.get("image_path") or item.get("capture_path")
        if not isinstance(raw_path, str) or not raw_path:
            continue
        try:
            if Path(raw_path).resolve() == target:
                return index
        except FileNotFoundError:
            continue
    return None


def build_view_chains(
    *,
    items: Sequence[dict],
    recipe: RecipeScores,
    target_similarities: Mapping[int, float],
    current_view_indices: Sequence[int],
    candidate_indices: Sequence[int],
    max_depth: int,
    branching_factor: int,
    target_hit_threshold: float,
) -> list[ChainSummary]:
    chains = []
    for view_order, root_index in enumerate(current_view_indices, start=1):
        best_path = best_chain_for_root(
            recipe=recipe,
            target_similarities=target_similarities,
            root_index=int(root_index),
            candidate_indices=candidate_indices,
            max_depth=max_depth,
            branching_factor=branching_factor,
            target_hit_threshold=target_hit_threshold,
        )
        chains.append(
            summarize_chain_path(
                items=items,
                path=best_path,
                target_similarities=target_similarities,
                view_order=view_order,
                target_hit_threshold=target_hit_threshold,
            )
        )
    return chains


def best_chain_for_root(
    *,
    recipe: RecipeScores,
    target_similarities: Mapping[int, float],
    root_index: int,
    candidate_indices: Sequence[int],
    max_depth: int,
    branching_factor: int,
    target_hit_threshold: float,
) -> ChainPath:
    width = max(int(branching_factor), 1)
    beam = [ChainPath(root_index=root_index, item_indices=(root_index,), edge_similarities=())]
    candidate_set = tuple(int(index) for index in candidate_indices)
    depth_limit = effective_depth_limit(max_depth=max_depth, candidate_count=len(candidate_set))
    for _depth in range(depth_limit):
        hit_paths = target_hit_paths(
            beam,
            target_similarities=target_similarities,
            target_hit_threshold=target_hit_threshold,
        )
        if hit_paths:
            return with_stop_reason(
                select_path_for_root(
                    hit_paths,
                    target_similarities=target_similarities,
                    target_hit_threshold=target_hit_threshold,
                ),
                "target_hit",
            )

        expanded = []
        for path in beam:
            parent_index = path.item_indices[-1]
            available = [index for index in candidate_set if index not in path.item_indices]
            ranked = sorted(
                available,
                key=lambda index: (-recipe.similarity(parent_index, index), index),
            )
            for child_index in ranked[:width]:
                expanded.append(
                    ChainPath(
                        root_index=root_index,
                        item_indices=(*path.item_indices, child_index),
                        edge_similarities=(
                            *path.edge_similarities,
                            recipe.similarity(parent_index, child_index),
                        ),
                    )
                )
        if not expanded:
            break
        hit_paths = target_hit_paths(
            expanded,
            target_similarities=target_similarities,
            target_hit_threshold=target_hit_threshold,
        )
        if hit_paths:
            return with_stop_reason(
                select_path_for_root(
                    hit_paths,
                    target_similarities=target_similarities,
                    target_hit_threshold=target_hit_threshold,
                ),
                "target_hit",
            )

        beam = sorted(
            expanded,
            key=lambda path: generation_sort_key(path),
            reverse=True,
        )[:width]
    stop_reason = "max_depth" if depth_limit < len(candidate_set) else "candidate_exhausted"
    return with_stop_reason(
        select_path_for_root(
            beam,
            target_similarities=target_similarities,
            target_hit_threshold=target_hit_threshold,
        ),
        stop_reason,
    )


def effective_depth_limit(*, max_depth: int, candidate_count: int) -> int:
    if int(max_depth) <= 0:
        return max(int(candidate_count), 0)
    return min(int(max_depth), max(int(candidate_count), 0))


def target_hit_paths(
    paths: Sequence[ChainPath],
    *,
    target_similarities: Mapping[int, float],
    target_hit_threshold: float,
) -> list[ChainPath]:
    return [
        path
        for path in paths
        if summarize_path_metrics(
            path=path,
            target_similarities=target_similarities,
            target_hit_threshold=target_hit_threshold,
        )["target_hit"]
    ]


def select_path_for_root(
    paths: Sequence[ChainPath],
    *,
    target_similarities: Mapping[int, float],
    target_hit_threshold: float,
) -> ChainPath:
    return sorted(
        paths,
        key=lambda path: chain_selection_sort_key(
            summarize_path_metrics(
                path=path,
                target_similarities=target_similarities,
                target_hit_threshold=target_hit_threshold,
            ),
            capture_index=0,
            continuity_threshold=0.0,
        ),
    )[0]


def with_stop_reason(path: ChainPath, stop_reason: str) -> ChainPath:
    return ChainPath(
        root_index=path.root_index,
        item_indices=path.item_indices,
        edge_similarities=path.edge_similarities,
        stop_reason=stop_reason,
    )


def generation_sort_key(path: ChainPath) -> tuple:
    if not path.edge_similarities:
        return (1.0, 1.0, 0.0)
    bottleneck = min(path.edge_similarities)
    mean_similarity = sum(path.edge_similarities) / len(path.edge_similarities)
    return (float(bottleneck), float(mean_similarity), -float(len(path.item_indices)))


def summarize_chain_path(
    *,
    items: Sequence[dict],
    path: ChainPath,
    target_similarities: Mapping[int, float],
    view_order: int,
    target_hit_threshold: float,
) -> ChainSummary:
    metrics = summarize_path_metrics(
        path=path,
        target_similarities=target_similarities,
        target_hit_threshold=target_hit_threshold,
    )
    nodes = []
    for depth, item_index in enumerate(path.item_indices):
        item = items[int(item_index)]
        nodes.append(
            {
                "depth": depth,
                "item_index": int(item_index),
                "memory_key": memory_key(item),
                "room_id": item.get("room_id"),
                "pano_id": item.get("pano_id"),
                "capture_index": item.get("capture_index"),
                "capture_label": item.get("capture_label"),
                "capture_heading": item.get("capture_heading"),
                "image_path": item.get("image_path") or item.get("capture_path"),
                "parent_similarity": (
                    None if depth == 0 else float(path.edge_similarities[depth - 1])
                ),
                "target_similarity": float(target_similarities[int(item_index)]),
                "target_hit": float(target_similarities[int(item_index)])
                >= float(target_hit_threshold),
            }
        )
    root = items[path.root_index]
    return ChainSummary(
        view_order=int(view_order),
        capture_index=int(root.get("capture_index") or 0),
        selected=False,
        target_hit=bool(metrics["target_hit"]),
        hit_depth=metrics["hit_depth"],
        hit_similarity=metrics["hit_similarity"],
        max_target_similarity=float(metrics["max_target_similarity"]),
        chain_bottleneck_similarity=float(metrics["chain_bottleneck_similarity"]),
        chain_mean_similarity=float(metrics["chain_mean_similarity"]),
        stop_reason=path.stop_reason,
        nodes=tuple(nodes),
    )


def summarize_path_metrics(
    *,
    path: ChainPath,
    target_similarities: Mapping[int, float],
    target_hit_threshold: float,
) -> dict:
    target_values = [float(target_similarities[int(index)]) for index in path.item_indices]
    hit_depth = None
    hit_similarity = None
    for depth, value in enumerate(target_values):
        if value >= float(target_hit_threshold):
            hit_depth = depth
            hit_similarity = value
            break
    if path.edge_similarities:
        bottleneck = min(path.edge_similarities)
        mean_similarity = sum(path.edge_similarities) / len(path.edge_similarities)
    else:
        bottleneck = 1.0
        mean_similarity = 1.0
    return {
        "target_hit": hit_depth is not None,
        "hit_depth": hit_depth,
        "hit_similarity": hit_similarity,
        "max_target_similarity": max(target_values) if target_values else 0.0,
        "chain_bottleneck_similarity": float(bottleneck),
        "chain_mean_similarity": float(mean_similarity),
    }


def select_chain(
    chains: Sequence[ChainSummary],
    *,
    continuity_threshold: float,
) -> ChainSummary:
    if not chains:
        raise ValueError("At least one chain is required.")
    hit_chains = [chain for chain in chains if chain.target_hit]
    if hit_chains:
        return sorted(
            hit_chains,
            key=lambda chain: (
                int(chain.hit_depth if chain.hit_depth is not None else 10**9),
                -float(chain.hit_similarity if chain.hit_similarity is not None else 0.0),
                -float(chain.chain_bottleneck_similarity),
                int(chain.capture_index),
            ),
        )[0]

    fallback = [
        chain
        for chain in chains
        if float(chain.chain_bottleneck_similarity) >= float(continuity_threshold)
    ]
    if not fallback:
        fallback = list(chains)
    return sorted(
        fallback,
        key=lambda chain: (
            -float(chain.max_target_similarity),
            -float(chain.chain_bottleneck_similarity),
            int(chain.capture_index),
        ),
    )[0]


def chain_selection_sort_key(
    metrics: Mapping[str, object],
    *,
    capture_index: int,
    continuity_threshold: float,
) -> tuple:
    if metrics["target_hit"]:
        return (
            0,
            int(metrics["hit_depth"] if metrics["hit_depth"] is not None else 10**9),
            -float(metrics["hit_similarity"] if metrics["hit_similarity"] is not None else 0.0),
            -float(metrics["chain_bottleneck_similarity"]),
            int(capture_index),
        )
    continuous = float(metrics["chain_bottleneck_similarity"]) >= float(continuity_threshold)
    return (
        1 if continuous else 2,
        -float(metrics["max_target_similarity"]),
        -float(metrics["chain_bottleneck_similarity"]),
        int(capture_index),
    )


def selection_payload(
    *,
    args,
    room_id: str,
    target_image: Path,
    recipe: RecipeWithTarget,
    selected: ChainSummary,
    chains: Sequence[ChainSummary],
    candidate_count: int,
) -> dict:
    return {
        "method": "visual_chain_view_selector",
        "configuration": {
            "current_pano_id": args.current_pano_id,
            "target_image": str(target_image),
            "room_id": room_id,
            "recipe": args.recipe,
            "component_recipes": list(recipe.component_recipes),
            "dinov2_patch_model": args.dinov2_patch_model,
            "dinov2_patch_top_k": int(args.dinov2_patch_top_k),
            "dinov2_patch_max_patches": int(args.dinov2_patch_max_patches),
            "dinov2_target_match_mode": args.dinov2_target_match_mode,
            "max_depth": int(args.max_depth),
            "branching_factor": int(args.branching_factor),
            "candidate_count": int(candidate_count),
            "effective_depth_limit": effective_depth_limit(
                max_depth=args.max_depth,
                candidate_count=candidate_count,
            ),
            "observed_max_chain_depth": max((len(chain.nodes) - 1 for chain in chains), default=0),
            "target_hit_threshold": float(args.target_hit_threshold),
            "continuity_threshold": float(args.continuity_threshold),
            "selection_rule": (
                "target_hit first: hit_depth asc, hit_similarity desc, bottleneck desc; "
                "fallback: max_target_similarity desc, bottleneck desc"
            ),
            "max_depth_semantics": (
                "max_depth <= 0 searches the finite same-room candidate pool until target hit "
                "or candidate exhaustion"
            ),
        },
        "selected_view_label": f"V{selected.view_order}",
        "selected_capture_index": selected.capture_index,
        "selected_chain": chain_to_dict(selected),
        "chains": [chain_summary_row(chain) for chain in chains],
    }


def chain_summary_row(chain: ChainSummary) -> dict:
    return {
        "view_label": f"V{chain.view_order}",
        "view_order": chain.view_order,
        "capture_index": chain.capture_index,
        "selected": chain.selected,
        "target_hit": chain.target_hit,
        "hit_depth": chain.hit_depth,
        "hit_similarity": chain.hit_similarity,
        "max_target_similarity": chain.max_target_similarity,
        "chain_bottleneck_similarity": chain.chain_bottleneck_similarity,
        "chain_mean_similarity": chain.chain_mean_similarity,
        "stop_reason": chain.stop_reason,
    }


def chain_to_dict(chain: ChainSummary) -> dict:
    return {
        **chain_summary_row(chain),
        "nodes": [dict(node) for node in chain.nodes],
    }


def write_gallery(
    path: Path,
    *,
    items: Sequence[dict],
    target_image: Path,
    chains: Sequence[ChainSummary],
) -> None:
    del items
    asset_dir = path.parent / "gallery_assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    target_asset = copy_image_asset(target_image, asset_dir, "target")
    node_assets = {}
    for chain in chains:
        for node in chain.nodes:
            image_path = node.get("image_path")
            if isinstance(image_path, str):
                node_assets[str(node["memory_key"])] = copy_image_asset(
                    Path(image_path),
                    asset_dir,
                    str(node["memory_key"]),
                )
    chain_sections = [chain_gallery_section(chain, target_asset, node_assets) for chain in chains]
    document = "\n".join(
        [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'>",
            "<title>Visual chain direction</title>",
            "<style>",
            "body{font-family:system-ui,-apple-system,sans-serif;margin:24px;background:#f7f7f4;color:#202020}",
            ".target{display:flex;gap:12px;align-items:center;margin-bottom:24px}",
            ".target img{width:180px;height:180px;object-fit:cover;border:1px solid #bbb}",
            ".chain{background:white;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:18px}",
            ".chain.selected{border-color:#126b38;box-shadow:0 0 0 2px rgba(18,107,56,.15)}",
            ".frames{display:flex;gap:10px;overflow-x:auto;padding-bottom:4px}",
            ".frame{min-width:150px;max-width:160px}",
            ".frame img{width:150px;height:150px;object-fit:cover;border:1px solid #ccc;background:#eee}",
            ".label{font-size:12px;line-height:1.35;margin-top:4px}",
            ".mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11px}",
            ".hit{color:#126b38;font-weight:700}",
            ".selected-badge{color:#126b38;font-weight:700}",
            ".metrics{display:flex;gap:10px;flex-wrap:wrap;margin:8px 0}",
            ".metric{background:#f1f1ed;border:1px solid #ddd;border-radius:6px;padding:6px 8px;font-size:12px}",
            "</style></head><body>",
            "<h1>Visual chain direction</h1>",
            "<div class='target'>",
            f"<img src='{html.escape(target_asset)}' alt='target image'>",
            "<div><h2>Target Passage</h2><p>Chains are generated by parent-child image similarity only.</p></div>",
            "</div>",
            *chain_sections,
            "</body></html>",
        ]
    )
    path.write_text(document, encoding="utf-8")


def chain_gallery_section(
    chain: ChainSummary,
    target_asset: str,
    node_assets: Mapping[str, str],
) -> str:
    del target_asset
    classes = "chain selected" if chain.selected else "chain"
    frames = []
    for node in chain.nodes:
        image_src = node_assets.get(str(node["memory_key"]))
        image_html = (
            f"<img src='{html.escape(image_src)}' alt='{html.escape(str(node['memory_key']))}'>"
            if image_src
            else "<div style='width:150px;height:150px;background:#eee'>missing</div>"
        )
        hit_class = " hit" if node.get("target_hit") else ""
        parent = node.get("parent_similarity")
        parent_text = "-" if parent is None else f"{float(parent):.3f}"
        frames.append(
            "\n".join(
                [
                    "<div class='frame'>",
                    image_html,
                    f"<div class='label{hit_class}'>depth={node['depth']} target={float(node['target_similarity']):.3f}</div>",
                    f"<div class='mono'>{html.escape(str(node.get('pano_id')))} #{node.get('capture_index')}</div>",
                    f"<div class='mono'>parent={parent_text}</div>",
                    "</div>",
                ]
            )
        )
    selected = " <span class='selected-badge'>selected</span>" if chain.selected else ""
    return "\n".join(
        [
            f"<section class='{classes}'>",
            f"<h2>V{chain.view_order} capture {chain.capture_index}{selected}</h2>",
            "<div class='metrics'>",
            metric_html("target_hit", chain.target_hit),
            metric_html("hit_depth", chain.hit_depth),
            metric_html("max_target", f"{chain.max_target_similarity:.3f}"),
            metric_html("bottleneck", f"{chain.chain_bottleneck_similarity:.3f}"),
            metric_html("stop", chain.stop_reason),
            "</div>",
            "<div class='frames'>",
            *frames,
            "</div>",
            "</section>",
        ]
    )


def metric_html(label: str, value) -> str:
    return f"<div class='metric'><b>{html.escape(str(label))}</b> {html.escape(str(value))}</div>"


def copy_image_asset(source: Path, asset_dir: Path, label: str) -> str:
    suffix = source.suffix or ".png"
    target = asset_dir / f"{safe_name(label)}{suffix}"
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return str(Path(asset_dir.name) / target.name)


if __name__ == "__main__":
    raise SystemExit(main())
