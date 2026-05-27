from __future__ import annotations

import json
import math
import random
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

CARDINAL_HEADINGS = (0.0, 90.0, 180.0, 270.0)
GROUNDING_HEADINGS = (330.0, 60.0, 150.0, 240.0)
MUSEUM_HEADINGS = (330.0, 60.0, 150.0, 240.0)
CARDINAL_LABELS = ("north", "east", "south", "west")
MUSEUM_INTERSTITIAL_LABELS = ("north_to_east", "east_to_south", "south_to_west", "west_to_north")


def normalize_heading(heading: float) -> float:
    return float(heading) % 360.0


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value) or "pano"


def circular_mean_headings(headings: list[float]) -> float:
    if not headings:
        return 0.0

    sin_sum = sum(math.sin(math.radians(heading)) for heading in headings)
    cos_sum = sum(math.cos(math.radians(heading)) for heading in headings)
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return normalize_heading(headings[0])
    return normalize_heading(math.degrees(math.atan2(sin_sum, cos_sum)))


def build_streetview_url(
    *,
    api_key: str,
    pano_id: str,
    heading: float,
    pitch: float,
    fov: int,
    width: int,
    height: int,
) -> str:
    query = urllib.parse.urlencode(
        {
            "size": f"{width}x{height}",
            "pano": pano_id,
            "heading": f"{normalize_heading(heading):.6f}",
            "pitch": f"{pitch:.6f}",
            "fov": str(fov),
            "key": api_key,
        }
    )
    return f"https://maps.googleapis.com/maps/api/streetview?{query}"


def _download_image(url: str, output_path: Path, *, timeout: float = 60.0) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "image" not in content_type:
            body = response.read(200).decode("utf-8", errors="replace")
            raise RuntimeError(f"Street View API did not return an image: {content_type} {body}")
        output_path.write_bytes(response.read())


class PanoramaRenderer:
    """
    Perception-side renderer that turns a pano node into multi-view images and a manifest.
    """

    def __init__(
        self,
        pano_graph: dict[str, dict],
        *,
        image_downloader: Callable[[str, Path], None] | None = None,
        image_timeout: float = 60.0,
        rng: random.Random | None = None,
    ):
        self.pano_graph = pano_graph
        self.image_timeout = float(image_timeout)
        self.image_downloader = image_downloader or (
            lambda url, output_path: _download_image(url, output_path, timeout=self.image_timeout)
        )
        self.rng = rng or random.Random()

    def render(
        self,
        *,
        pano_id: str,
        api_key: str,
        output_dir: str | Path,
        heading_mode: str = "cardinal",
        custom_captures: list[tuple[str, float]] | None = None,
        pitch: float = 0.0,
        fov: int = 45,
        width: int = 512,
        height: int = 512,
        graph_path: str | Path | None = None,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        record = self._get_pano_record(pano_id, required=(heading_mode == "graph"))
        captures_to_render = (
            [(str(label), normalize_heading(float(heading))) for label, heading in custom_captures]
            if custom_captures is not None
            else self._resolve_captures(record, heading_mode)
        )

        output_dir = Path(output_dir)
        pano_slug = sanitize_name(pano_id)
        pano_output_dir = output_dir / pano_slug
        pano_output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = pano_output_dir / f"{pano_slug}_manifest.json"

        cached_manifest = self._load_cached_manifest(
            manifest_path,
            pano_id=pano_id,
            heading_mode=heading_mode,
            pitch=pitch,
            fov=fov,
            width=width,
            height=height,
            graph_path=graph_path,
            captures_to_render=captures_to_render,
        )
        if cached_manifest is not None:
            cached_manifest["manifest_path"] = str(manifest_path)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "render_cached",
                        "pano_id": pano_id,
                        "capture_count": len(captures_to_render),
                        "manifest_path": str(manifest_path),
                    }
                )
            return cached_manifest

        captures = []
        for index, (label, heading) in enumerate(captures_to_render):
            filename = f"{pano_slug}_{index:02d}_{label}_{int(round(heading)):03d}deg.png"
            image_path = pano_output_dir / filename
            image_url = build_streetview_url(
                api_key=api_key,
                pano_id=pano_id,
                heading=heading,
                pitch=pitch,
                fov=fov,
                width=width,
                height=height,
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "render_capture_start",
                        "pano_id": pano_id,
                        "capture_index": index + 1,
                        "capture_count": len(captures_to_render),
                        "label": label,
                        "heading": heading,
                        "path": str(image_path),
                    }
                )
            self.image_downloader(image_url, image_path)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "render_capture_done",
                        "pano_id": pano_id,
                        "capture_index": index + 1,
                        "capture_count": len(captures_to_render),
                        "label": label,
                        "heading": heading,
                        "path": str(image_path),
                    }
                )
            captures.append(
                {
                    "label": label,
                    "heading": heading,
                    "path": str(image_path),
                    "url": image_url,
                }
            )

        manifest = {
            "graph_path": str(graph_path) if graph_path is not None else None,
            "pano_id": pano_id,
            "floor": self._record_floor(record),
            "lat": record.get("lat"),
            "lng": record.get("lng"),
            "heading_mode": heading_mode,
            "pitch": pitch,
            "fov": fov,
            "size": {"width": width, "height": height},
            "captures": captures,
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "render_done",
                    "pano_id": pano_id,
                    "capture_count": len(captures_to_render),
                    "manifest_path": str(manifest_path),
                }
            )
        return manifest

    @staticmethod
    def _load_cached_manifest(
        manifest_path: Path,
        *,
        pano_id: str,
        heading_mode: str,
        pitch: float,
        fov: int,
        width: int,
        height: int,
        graph_path: str | Path | None,
        captures_to_render: list[tuple[str, float]],
    ) -> dict | None:
        if not manifest_path.exists():
            return None

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(manifest, dict):
            return None

        if manifest.get("pano_id") != pano_id:
            return None
        if manifest.get("heading_mode") != heading_mode:
            return None
        if manifest.get("pitch") != pitch or manifest.get("fov") != fov:
            return None
        if manifest.get("graph_path") != (str(graph_path) if graph_path is not None else None):
            return None

        size = manifest.get("size")
        if not isinstance(size, dict):
            return None
        if size.get("width") != width or size.get("height") != height:
            return None

        captures = manifest.get("captures")
        if not isinstance(captures, list) or len(captures) != len(captures_to_render):
            return None

        for index, capture in enumerate(captures):
            if not isinstance(capture, dict):
                return None
            expected_label, expected_heading = captures_to_render[index]
            if capture.get("label") != expected_label:
                return None
            if heading_mode != "museum":
                heading = capture.get("heading")
                if not isinstance(heading, (int, float)) or abs(float(heading) - expected_heading) > 1e-6:
                    return None
            image_path = capture.get("path")
            if not isinstance(image_path, str) or not Path(image_path).exists():
                return None
        return manifest

    def _get_pano_record(self, pano_id: str, *, required: bool) -> dict:
        record = self.pano_graph.get(pano_id)
        if not isinstance(record, dict):
            if not required:
                return {}
            raise KeyError(f"pano_id not found in graph: {pano_id}")
        return record

    def _resolve_captures(self, record: dict, heading_mode: str) -> list[tuple[str, float]]:
        if heading_mode == "museum":
            return self._museum_captures()
        if heading_mode == "grounding":
            return list(zip(CARDINAL_LABELS, GROUNDING_HEADINGS))
        if heading_mode == "cardinal":
            return list(zip(CARDINAL_LABELS, CARDINAL_HEADINGS))
        return [(f"view{index}", heading) for index, heading in enumerate(self._graph_aligned_headings(record))]

    def _graph_aligned_headings(self, record: dict) -> list[float]:
        graph_headings = self._neighbor_headings(record)
        if not graph_headings:
            return list(CARDINAL_HEADINGS)

        base_heading = circular_mean_headings(graph_headings)
        return [normalize_heading(base_heading + index * 90.0) for index in range(4)]

    def _museum_captures(self) -> list[tuple[str, float]]:
        captures: list[tuple[str, float]] = []
        for index, heading in enumerate(MUSEUM_HEADINGS):
            captures.append((CARDINAL_LABELS[index], heading))
            next_heading = MUSEUM_HEADINGS[(index + 1) % len(MUSEUM_HEADINGS)]
            captures.append((MUSEUM_INTERSTITIAL_LABELS[index], self._sample_heading_between(heading, next_heading)))
        return captures

    def _sample_heading_between(self, start_heading: float, end_heading: float) -> float:
        span = (normalize_heading(end_heading) - normalize_heading(start_heading)) % 360.0
        if span == 0.0:
            span = 360.0
        fraction = min(max(self.rng.random(), 1e-6), 1.0 - 1e-6)
        return normalize_heading(start_heading + span * fraction)

    @staticmethod
    def _record_floor(record: dict) -> str | None:
        floor = record.get("floor")
        if floor is None:
            return None
        return str(floor)

    @staticmethod
    def _neighbor_headings(record: dict) -> list[float]:
        headings: list[float] = []
        if isinstance(record.get("neighbors"), list):
            for neighbor in record["neighbors"]:
                if isinstance(neighbor, dict) and isinstance(neighbor.get("geocentric_heading_deg"), (int, float)):
                    headings.append(normalize_heading(float(neighbor["geocentric_heading_deg"])))
        if headings:
            return headings

        if isinstance(record.get("links"), list):
            for link in record["links"]:
                if isinstance(link, dict) and isinstance(link.get("heading"), (int, float)):
                    headings.append(normalize_heading(float(link["heading"])))
        return headings
