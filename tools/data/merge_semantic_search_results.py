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

from memory_nav.data.memory_localization import load_image_index_artifacts, write_json  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge text-image semantic search results with image embedding clustering. "
            "Each cluster representative is the candidate with the highest semantic score."
        )
    )
    parser.add_argument(
        "--result-json",
        action="append",
        required=True,
        help="Path to a semantic_search_memory.py results.json. Repeat for multiple queries.",
    )
    parser.add_argument(
        "--embedding-index",
        default="artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz",
        help="Image embedding npz used when --clustering-model=index.",
    )
    parser.add_argument(
        "--clustering-model",
        choices=["index", "dreamsim", "hybrid"],
        default="index",
        help=(
            "Similarity backend: an existing embedding index, DreamSim embeddings, "
            "or a weighted DINOv2 + SigLIP2 similarity matrix."
        ),
    )
    parser.add_argument(
        "--siglip-index",
        default="artifacts/memory_localization/floor0_siglip2_images_fov90.npz",
        help="SigLIP2 image embedding index used by --clustering-model=hybrid.",
    )
    parser.add_argument(
        "--dinov2-model",
        default="facebook/dinov2-base",
        help="Hugging Face DINOv2 model used by --clustering-model=hybrid.",
    )
    parser.add_argument(
        "--dreamsim-type",
        default="ensemble",
        help="DreamSim checkpoint type, such as ensemble or dinov2_vitb14.",
    )
    parser.add_argument(
        "--dinov2-weight",
        type=float,
        default=0.7,
        help="DINOv2 similarity weight for hybrid clustering.",
    )
    parser.add_argument(
        "--siglip-weight",
        type=float,
        default=0.3,
        help="SigLIP2 similarity weight for hybrid clustering.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device for DreamSim or DINOv2: auto, cuda, mps, or cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Image encoding batch size for DreamSim or DINOv2.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.5,
        help="Cosine threshold for connected-component clustering.",
    )
    parser.add_argument(
        "--target-clusters",
        type=int,
        default=4,
        help="Also run simple agglomerative clustering until this many clusters remain.",
    )
    parser.add_argument(
        "--representative-prefix",
        default="P",
        help="Label prefix for representative images, e.g. P gives P1, P2, ...",
    )
    parser.add_argument("--output-dir", default="outputs/merged_semantic_search_results")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the full JSON payload to stdout.",
    )
    return parser


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON payload: {path}")
    return payload


def safe_name(value: object) -> str:
    text = str(value or "").strip()
    cleaned = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_") or "item"


def normalize_rows(array):
    import numpy as np

    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def resolve_torch_device(requested_device: str) -> str:
    import torch

    normalized = (requested_device or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def candidate_image_paths(candidates: list[dict]) -> list[Path]:
    paths = []
    for candidate in candidates:
        image_path = candidate.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise RuntimeError(
                f"Candidate memory_index={candidate.get('memory_index')} has no readable image path."
            )
        path = Path(image_path)
        if not path.exists():
            raise RuntimeError(f"Candidate image does not exist: {path}")
        paths.append(path)
    return paths


def encode_dinov2_image_paths(
    image_paths: list[Path],
    *,
    model_name: str,
    device: str,
    batch_size: int,
):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    batches = []
    step = max(int(batch_size), 1)
    for start in range(0, len(image_paths), step):
        batch_paths = image_paths[start : start + step]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in dict(inputs).items()}
            with torch.inference_mode():
                output = model(**inputs)
            embeddings = getattr(output, "pooler_output", None)
            if embeddings is None:
                embeddings = output.last_hidden_state[:, 0]
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            batches.append(embeddings.detach().cpu().to(torch.float32).numpy())
        finally:
            for image in images:
                image.close()
    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def encode_dreamsim_image_paths(
    image_paths: list[Path],
    *,
    dreamsim_type: str,
    device: str,
    batch_size: int,
):
    import numpy as np
    import torch
    from dreamsim import dreamsim
    from PIL import Image

    model, preprocess = dreamsim(
        pretrained=True,
        device=device,
        dreamsim_type=dreamsim_type,
    )
    model = model.eval()
    batches = []
    step = max(int(batch_size), 1)
    for start in range(0, len(image_paths), step):
        batch_paths = image_paths[start : start + step]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        try:
            tensors = [preprocess(image) for image in images]
            inputs = torch.cat(
                [tensor if tensor.ndim == 4 else tensor.unsqueeze(0) for tensor in tensors],
                dim=0,
            ).to(device)
            with torch.inference_mode():
                embeddings = model.embed(inputs)
            if embeddings.ndim > 2:
                embeddings = embeddings.flatten(start_dim=1)
            embeddings = embeddings / embeddings.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-12)
            batches.append(embeddings.detach().cpu().to(torch.float32).numpy())
        finally:
            for image in images:
                image.close()
    if not batches:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(batches, axis=0).astype(np.float32)


def build_similarity(args, candidates: list[dict], embedding_index: Path):
    import numpy as np

    memory_indices = [int(candidate["memory_index"]) for candidate in candidates]
    backend = args.clustering_model
    if backend == "index":
        embeddings = load_image_index_artifacts(embedding_index)
        if max(memory_indices) >= int(embeddings.shape[0]):
            raise RuntimeError("A candidate memory_index is outside the embedding index.")
        candidate_embeddings = normalize_rows(embeddings[memory_indices])
        return candidate_embeddings @ candidate_embeddings.T, {
            "clustering_model": backend,
            "embedding_index": str(embedding_index),
        }

    image_paths = candidate_image_paths(candidates)
    device = resolve_torch_device(args.device)
    if backend == "dreamsim":
        embeddings = normalize_rows(
            encode_dreamsim_image_paths(
                image_paths,
                dreamsim_type=args.dreamsim_type,
                device=device,
                batch_size=args.batch_size,
            )
        )
        return embeddings @ embeddings.T, {
            "clustering_model": backend,
            "dreamsim_type": args.dreamsim_type,
            "device": device,
        }

    siglip_index = resolve_project_path(args.siglip_index)
    siglip_embeddings = load_image_index_artifacts(siglip_index)
    if max(memory_indices) >= int(siglip_embeddings.shape[0]):
        raise RuntimeError("A candidate memory_index is outside the SigLIP2 embedding index.")
    siglip_candidates = normalize_rows(siglip_embeddings[memory_indices])
    dinov2_candidates = normalize_rows(
        encode_dinov2_image_paths(
            image_paths,
            model_name=args.dinov2_model,
            device=device,
            batch_size=args.batch_size,
        )
    )
    dinov2_weight = float(args.dinov2_weight)
    siglip_weight = float(args.siglip_weight)
    weight_sum = dinov2_weight + siglip_weight
    if dinov2_weight < 0.0 or siglip_weight < 0.0 or weight_sum <= 0.0:
        raise ValueError("Hybrid weights must be non-negative and have a positive sum.")
    dinov2_weight /= weight_sum
    siglip_weight /= weight_sum
    similarity = (
        dinov2_weight * (dinov2_candidates @ dinov2_candidates.T)
        + siglip_weight * (siglip_candidates @ siglip_candidates.T)
    )
    return np.asarray(similarity, dtype=np.float32), {
        "clustering_model": backend,
        "dinov2_model": args.dinov2_model,
        "siglip_index": str(siglip_index),
        "dinov2_weight": dinov2_weight,
        "siglip_weight": siglip_weight,
        "device": device,
    }


def resolve_result_image(result: dict, result_json_path: Path) -> Path | None:
    source_path = result.get("source_path")
    if isinstance(source_path, str) and source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return candidate.resolve()

    output_image = result.get("output_image")
    if isinstance(output_image, str) and output_image:
        candidate = result_json_path.parent / output_image
        if candidate.exists():
            return candidate.resolve()

    return None


def merge_candidates(result_json_paths: list[Path]) -> list[dict]:
    merged: dict[int, dict] = {}
    for result_json_path in result_json_paths:
        payload = load_json(result_json_path)
        query = payload.get("query")
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            memory_index = result.get("memory_index")
            if not isinstance(memory_index, int):
                continue

            score = float(result.get("score") or 0.0)
            candidate = merged.get(memory_index)
            if candidate is None:
                candidate = dict(result)
                candidate["memory_index"] = memory_index
                candidate["best_score"] = score
                candidate["query_hits"] = []
                candidate["image_path"] = None
                merged[memory_index] = candidate
            elif score > float(candidate.get("best_score") or 0.0):
                for key, value in result.items():
                    if key not in {"rank", "score", "output_image"}:
                        candidate[key] = value
                candidate["best_score"] = score

            image_path = resolve_result_image(result, result_json_path)
            if image_path is not None and candidate.get("image_path") is None:
                candidate["image_path"] = str(image_path)

            candidate["query_hits"].append(
                {
                    "query": query,
                    "rank": result.get("rank"),
                    "score": score,
                    "result_json": str(result_json_path),
                }
            )

    candidates = list(merged.values())
    candidates.sort(key=lambda item: float(item.get("best_score") or 0.0), reverse=True)
    return candidates


def copy_candidate_images(candidates: list[dict], output_dir: Path) -> None:
    for index, candidate in enumerate(candidates, start=1):
        image_path = candidate.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            candidate["output_image"] = None
            continue
        source = Path(image_path)
        if not source.exists():
            candidate["output_image"] = None
            continue
        suffix = source.suffix or ".png"
        filename = (
            f"candidate_{index:02d}_"
            f"{safe_name(candidate.get('room_id'))}_"
            f"{safe_name(candidate.get('pano_id'))}_"
            f"{safe_name(candidate.get('capture_label'))}"
            f"{suffix}"
        )
        shutil.copy2(source, output_dir / filename)
        candidate["output_image"] = filename


def connected_components(similarity, threshold: float) -> list[list[int]]:
    n_items = int(similarity.shape[0])
    seen: set[int] = set()
    clusters: list[list[int]] = []
    for start in range(n_items):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        cluster = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            neighbors = [
                index
                for index in range(n_items)
                if index not in seen and float(similarity[current, index]) >= threshold
            ]
            seen.update(neighbors)
            stack.extend(neighbors)
        clusters.append(sorted(cluster))
    return clusters


def average_link_similarity(cluster_a: list[int], cluster_b: list[int], similarity) -> float:
    values = [float(similarity[a, b]) for a in cluster_a for b in cluster_b]
    return sum(values) / max(len(values), 1)


def agglomerative_clusters(similarity, target_clusters: int) -> list[list[int]]:
    clusters = [[index] for index in range(int(similarity.shape[0]))]
    target_clusters = max(1, int(target_clusters))
    while len(clusters) > target_clusters:
        best_pair: tuple[int, int] | None = None
        best_score = -float("inf")
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                score = average_link_similarity(clusters[left], clusters[right], similarity)
                if score > best_score:
                    best_score = score
                    best_pair = (left, right)
        if best_pair is None:
            break
        left, right = best_pair
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    return clusters


def summarize_candidate(candidate: dict, *, similarity_to_representative: float | None = None) -> dict:
    summary = {
        "memory_index": candidate.get("memory_index"),
        "best_score": candidate.get("best_score"),
        "room_id": candidate.get("room_id"),
        "pano_id": candidate.get("pano_id"),
        "capture_index": candidate.get("capture_index"),
        "capture_label": candidate.get("capture_label"),
        "capture_heading": candidate.get("capture_heading"),
        "image_path": candidate.get("image_path"),
        "output_image": candidate.get("output_image"),
        "query_hits": candidate.get("query_hits", []),
    }
    if similarity_to_representative is not None:
        summary["similarity_to_representative"] = similarity_to_representative
    return summary


def summarize_clusters(clusters: list[list[int]], candidates: list[dict], similarity) -> list[dict]:
    summaries = []
    for cluster in clusters:
        representative_index = max(
            cluster,
            key=lambda index: float(candidates[index].get("best_score") or 0.0),
        )
        representative = candidates[representative_index]
        members = [
            summarize_candidate(
                candidates[index],
                similarity_to_representative=float(similarity[representative_index, index]),
            )
            for index in cluster
        ]
        members.sort(key=lambda item: float(item.get("best_score") or 0.0), reverse=True)
        summaries.append(
            {
                "size": len(cluster),
                "representative": summarize_candidate(representative),
                "members": members,
            }
        )
    summaries.sort(
        key=lambda item: float(item["representative"].get("best_score") or 0.0),
        reverse=True,
    )
    for index, summary in enumerate(summaries, start=1):
        summary["cluster_id"] = index
    return summaries


def write_html(output_dir: Path, payload: dict) -> None:
    def image_tag(candidate: dict, title: str) -> str:
        image_name = candidate.get("output_image")
        if not image_name:
            return "<div class='missing'>image unavailable</div>"
        return (
            f'<img src="{html.escape(str(image_name))}" '
            f'alt="{html.escape(title)}" loading="lazy">'
        )

    def candidate_block(candidate: dict) -> str:
        title = (
            f"idx={candidate.get('memory_index')} "
            f"score={float(candidate.get('best_score') or 0.0):.4f} "
            f"{candidate.get('capture_label')}"
        )
        return "\n".join(
            [
                "<section class='candidate'>",
                image_tag(candidate, title),
                f"<pre>{html.escape(json.dumps(candidate, ensure_ascii=False, indent=2))}</pre>",
                "</section>",
            ]
        )

    def cluster_section(title: str, clusters: list[dict]) -> str:
        cards = []
        for cluster in clusters:
            member_html = "\n".join(candidate_block(member) for member in cluster["members"])
            cards.append(
                "\n".join(
                    [
                        "<section class='cluster'>",
                        f"<h3>{html.escape(title)} #{cluster['cluster_id']} "
                        f"(size={cluster['size']})</h3>",
                        "<h4>Representative</h4>",
                        candidate_block(cluster["representative"]),
                        "<h4>Members</h4>",
                        f"<div class='grid'>{member_html}</div>",
                        "</section>",
                    ]
                )
            )
        return "\n".join(cards)

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Merged Semantic Search Results</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f7; }}
    h1, h2, h3, h4 {{ margin-bottom: 8px; }}
    .meta {{ color: #555; margin-bottom: 24px; }}
    .cluster {{ background: white; padding: 16px; margin-bottom: 20px; border-radius: 10px; box-shadow: 0 1px 4px #ccc; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    .candidate {{ background: #fff; padding: 10px; border: 1px solid #ddd; border-radius: 8px; }}
    img {{ width: 100%; height: auto; border-radius: 6px; }}
    pre {{ white-space: pre-wrap; font-size: 11px; max-height: 260px; overflow: auto; }}
    .missing {{ padding: 48px; text-align: center; background: #eee; color: #777; }}
  </style>
</head>
<body>
  <h1>Merged Semantic Search Results</h1>
  <div class="meta">
    candidate_count={payload["candidate_count"]}<br>
    embedding_index={html.escape(str(payload["embedding_index"]))}<br>
    similarity_threshold={payload["similarity_threshold"]}<br>
    target_clusters={payload["target_clusters"]}
  </div>

  <h2>Threshold Clusters</h2>
{cluster_section("threshold", payload["threshold_clusters"])}

  <h2>Target-K Clusters</h2>
{cluster_section("target", payload["target_clusters_result"])}
</body>
</html>
"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")



def gallery_document(*, title: str, subtitle: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f6f6f6; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 32px; border-top: 1px solid #ddd; padding-top: 20px; }}
    h3 {{ margin: 10px 0 4px; }}
    .subtitle {{ color: #555; margin-bottom: 24px; }}
    .cluster {{ background: white; padding: 18px; margin-bottom: 24px; border-radius: 12px; box-shadow: 0 1px 4px #ccc; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }}
    .grid.large {{ grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }}
    .grid.representative {{ grid-template-columns: minmax(320px, 560px); }}
    .card {{ background: white; padding: 10px; border: 1px solid #ddd; border-radius: 10px; }}
    .cluster .card {{ background: #fbfbfb; }}
    img {{ width: 100%; height: auto; border-radius: 8px; display: block; }}
    p {{ margin: 3px 0; color: #333; font-size: 13px; }}
    .queries {{ color: #666; }}
    .missing {{ padding: 80px 24px; text-align: center; background: #eee; color: #777; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="subtitle">{html.escape(subtitle)}</div>
{body}
</body>
</html>
"""




def load_candidate_image(candidate: dict, *, thumb_width: int, thumb_height: int):
    from PIL import Image

    image_name = candidate.get("output_image")
    if not image_name:
        return None
    image_path = Path(str(candidate.get("_output_dir", ""))) / str(image_name)
    if not image_path.exists():
        image_path = Path(str(image_name))
    if not image_path.exists():
        return None

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((thumb_width, thumb_height))
    canvas = Image.new("RGB", (thumb_width, thumb_height), "white")
    x = (thumb_width - image.width) // 2
    y = (thumb_height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_wrapped_text(draw, position: tuple[int, int], text: str, font, *, max_width: int, fill="black") -> int:
    x, y = position
    words = str(text).split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines[:3]:
        draw.text((x, y), line, font=font, fill=fill)
        y += 16
    return y


def write_contact_sheets(output_dir: Path, payload: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    for candidate in payload["candidates"]:
        candidate["_output_dir"] = str(output_dir)
    for cluster_group in ("threshold_clusters", "target_clusters_result"):
        for cluster in payload[cluster_group]:
            cluster["representative"]["_output_dir"] = str(output_dir)
            for member in cluster["members"]:
                member["_output_dir"] = str(output_dir)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    def candidate_label(candidate: dict, prefix: str) -> str:
        hits = candidate.get("query_hits") or []
        queries = ", ".join(str(hit.get("query")) for hit in hits[:2])
        return (
            f"{prefix} score={float(candidate.get('best_score') or 0.0):.4f} "
            f"label={candidate.get('capture_label')} heading={candidate.get('capture_heading')} "
            f"idx={candidate.get('memory_index')} {queries}"
        )

    def paste_card(sheet, draw, candidate: dict, x: int, y: int, label: str) -> None:
        thumb_w, thumb_h = 320, 220
        image = load_candidate_image(candidate, thumb_width=thumb_w, thumb_height=thumb_h)
        if image is None:
            draw.rectangle((x, y, x + thumb_w, y + thumb_h), fill="#eeeeee", outline="#bbbbbb")
            draw.text((x + 70, y + 100), "image unavailable", font=font, fill="#777777")
        else:
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + thumb_w, y + thumb_h), outline="#bbbbbb", width=1)
        draw_wrapped_text(draw, (x, y + thumb_h + 8), label, font, max_width=thumb_w)

    def write_cluster_sheet(clusters: list[dict], *, title: str, filename: str) -> None:
        margin = 24
        title_h = 56
        max_members = max((len(cluster["members"]) for cluster in clusters), default=1)
        columns = min(max_members + 1, 6)
        card_w, card_h = 350, 290
        cluster_title_h = 38
        rows_per_cluster = max(1, (min(max_members + 1, columns * 2) + columns - 1) // columns)
        cluster_h = cluster_title_h + card_h * rows_per_cluster
        width = margin * 2 + columns * card_w
        height = title_h + margin + len(clusters) * cluster_h
        sheet = Image.new("RGB", (width, max(height, 400)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, 18), title, font=title_font, fill="black")
        y = title_h
        for cluster in clusters:
            draw.text((margin, y), f"Cluster #{cluster['cluster_id']} size={cluster['size']}", font=title_font, fill="black")
            y += cluster_title_h
            representative_index = cluster["representative"].get("memory_index")
            cards = [(cluster["representative"], "Representative")]
            cards.extend(
                (member, f"Member #{idx}")
                for idx, member in enumerate(cluster["members"], start=1)
                if member.get("memory_index") != representative_index
            )
            for index, (candidate, prefix) in enumerate(cards[:columns * 2]):
                row, col = divmod(index, columns)
                paste_card(
                    sheet,
                    draw,
                    candidate,
                    margin + col * card_w,
                    y + row * card_h,
                    candidate_label(candidate, prefix),
                )
            y += card_h * rows_per_cluster
        sheet.save(output_dir / filename)

    write_cluster_sheet(
        payload["threshold_clusters"],
        title=f"Threshold clusters, similarity >= {payload['similarity_threshold']}",
        filename="threshold_clusters_contact_sheet.png",
    )
    write_cluster_sheet(
        payload["target_clusters_result"],
        title="Target cluster passage candidates",
        filename="target_clusters_contact_sheet.png",
    )



def write_representative_outputs(output_dir: Path, payload: dict) -> None:
    representatives_dir = output_dir / "representatives"
    representatives_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in representatives_dir.glob("*"):
        if stale_image.is_file():
            stale_image.unlink()

    representatives = []
    prefix = str(payload["representative_prefix"])
    for index, cluster in enumerate(payload["target_clusters_result"], start=1):
        representative = dict(cluster["representative"])
        label = f"{prefix}{index}"
        copied_image = None
        output_image = representative.get("output_image")
        if isinstance(output_image, str) and output_image:
            source = output_dir / output_image
            if source.exists():
                suffix = source.suffix or ".png"
                copied_image = (
                    f"{label}_cluster_{int(cluster['cluster_id']):02d}_"
                    f"{safe_name(representative.get('room_id'))}_"
                    f"{safe_name(representative.get('pano_id'))}_"
                    f"{safe_name(representative.get('capture_label'))}"
                    f"{suffix}"
                )
                shutil.copy2(source, representatives_dir / copied_image)
        representative["label"] = label
        representative["cluster_id"] = cluster["cluster_id"]
        representative["cluster_size"] = cluster["size"]
        representative["representative_image"] = (
            str(Path("representatives") / copied_image) if copied_image else None
        )
        representatives.append(representative)

    payload["representatives"] = representatives
    write_json(output_dir / "representatives.json", {"representatives": representatives})
    write_representatives_contact_sheet(output_dir, payload)


def write_representatives_contact_sheet(output_dir: Path, payload: dict) -> None:
    from PIL import Image, ImageDraw, ImageFont

    representatives = payload.get("representatives", [])
    if not representatives:
        return

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
        title_font = font

    def load_representative_image(representative: dict):
        image_name = representative.get("representative_image")
        if not isinstance(image_name, str) or not image_name:
            return None
        image_path = output_dir / image_name
        if not image_path.exists():
            return None
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((320, 220))
        canvas = Image.new("RGB", (320, 220), "white")
        canvas.paste(image, ((320 - image.width) // 2, (220 - image.height) // 2))
        return canvas

    columns = min(4, max(len(representatives), 1))
    card_w, card_h = 350, 292
    margin, title_h = 24, 58
    rows = (len(representatives) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (margin * 2 + columns * card_w, title_h + margin + rows * card_h),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    title = (
        f"Target representatives: {payload['target_clusters']} clusters, "
        f"{payload['candidate_count']} candidates"
    )
    draw.text((margin, 18), title, font=title_font, fill="black")

    for index, representative in enumerate(representatives):
        row, col = divmod(index, columns)
        x = margin + col * card_w
        y = title_h + row * card_h
        image = load_representative_image(representative)
        if image is None:
            draw.rectangle((x, y, x + 320, y + 220), fill="#eeeeee", outline="#bbbbbb")
            draw.text((x + 70, y + 100), "image unavailable", font=font, fill="#777777")
        else:
            sheet.paste(image, (x, y))
            draw.rectangle((x, y, x + 320, y + 220), outline="#bbbbbb", width=1)
        label = (
            f"{representative['label']} cluster={representative['cluster_id']} "
            f"size={representative['cluster_size']} "
            f"score={float(representative.get('best_score') or 0.0):.4f} "
            f"{representative.get('room_id')} {representative.get('capture_label')}"
        )
        draw_wrapped_text(draw, (x, y + 228), label, font, max_width=320)

    sheet.save(output_dir / "representatives_contact_sheet.png")


def write_gallery_html(output_dir: Path, payload: dict) -> None:
    def image_card(candidate: dict, label: str) -> str:
        image_name = candidate.get("output_image")
        if image_name:
            image_html = (
                f'<a href="{html.escape(str(image_name))}">'
                f'<img src="{html.escape(str(image_name))}" alt="{html.escape(label)}" loading="lazy">'
                "</a>"
            )
        else:
            image_html = "<div class='missing'>image unavailable</div>"
        hits = candidate.get("query_hits") or []
        query_text = ", ".join(
            f"{hit.get('query')}:{float(hit.get('score') or 0.0):.4f}"
            for hit in hits
        )
        return "\n".join(
            [
                "<section class='card'>",
                image_html,
                f"<h3>{html.escape(label)}</h3>",
                f"<p>score={float(candidate.get('best_score') or 0.0):.4f}</p>",
                f"<p>{html.escape(str(candidate.get('capture_label')))} "
                f"heading={html.escape(str(candidate.get('capture_heading')))}</p>",
                f"<p class='queries'>{html.escape(query_text)}</p>",
                "</section>",
            ]
        )

    cluster_sections = []
    for cluster in payload["target_clusters_result"]:
        representative = cluster["representative"]
        members = cluster["members"]
        member_cards = "\n".join(
            image_card(member, f"Member #{member_index}")
            for member_index, member in enumerate(members, start=1)
        )
        cluster_sections.append(
            "\n".join(
                [
                    "<section class='cluster'>",
                    f"<h2>Cluster #{cluster['cluster_id']} / size={cluster['size']}</h2>",
                    "<h3>Representative</h3>",
                    f"<div class='grid representative'>{image_card(representative, 'Representative')}</div>",
                    "<h3>Members</h3>",
                    f"<div class='grid'>{member_cards}</div>",
                    "</section>",
                ]
            )
        )
    clusters_html = gallery_document(
        title="Target Cluster Passage Candidates",
        subtitle=f"Forced into {payload['target_clusters']} visual clusters.",
        body="\n".join(cluster_sections),
    )
    (output_dir / "target_clusters_gallery.html").write_text(clusters_html, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    result_json_paths = [resolve_project_path(path) for path in args.result_json]
    embedding_index = resolve_project_path(args.embedding_index)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stale_image in output_dir.glob("candidate_*"):
        if stale_image.is_file():
            stale_image.unlink()
    for stale_output in ["diverse_gallery.html", "diverse_contact_sheet.png"]:
        stale_path = output_dir / stale_output
        if stale_path.exists():
            stale_path.unlink()

    candidates = merge_candidates(result_json_paths)
    if not candidates:
        raise RuntimeError("No candidates found in input result JSON files.")

    similarity, clustering_config = build_similarity(args, candidates, embedding_index)
    copy_candidate_images(candidates, output_dir)

    threshold_clusters = connected_components(similarity, args.similarity_threshold)
    target_clusters = agglomerative_clusters(similarity, args.target_clusters)

    payload = {
        "input_result_json": [str(path) for path in result_json_paths],
        "embedding_index": str(embedding_index),
        "clustering_config": clustering_config,
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "similarity_threshold": args.similarity_threshold,
        "target_clusters": args.target_clusters,
        "representative_prefix": args.representative_prefix,
        "candidates": [summarize_candidate(candidate) for candidate in candidates],
        "threshold_clusters": summarize_clusters(threshold_clusters, candidates, similarity),
        "target_clusters_result": summarize_clusters(target_clusters, candidates, similarity),
    }
    write_representative_outputs(output_dir, payload)
    write_json(output_dir / "merged_results.json", payload)
    write_html(output_dir, payload)
    write_gallery_html(output_dir, payload)
    write_contact_sheets(output_dir, payload)

    if not args.quiet:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nWrote: {output_dir / 'merged_results.json'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'index.html'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'target_clusters_gallery.html'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'threshold_clusters_contact_sheet.png'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'target_clusters_contact_sheet.png'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'representatives.json'}", file=sys.stderr)
    print(f"Wrote: {output_dir / 'representatives_contact_sheet.png'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
