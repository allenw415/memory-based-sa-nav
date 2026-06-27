from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from memory_nav import load_dotenv
from memory_nav.cli._common import PROJECT_ROOT
from memory_nav.web.memory_guidance import (
    MEMORY_GUIDANCE_STATIC_ROOT,
    MemoryGuidanceConfig,
    MemoryGuidanceWebApp,
)
from memory_nav.web.pano_export import PanoExportConfig, export_pano_viewer, project_path

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only before dependencies are installed.
    raise ModuleNotFoundError(
        "FastAPI web server dependencies are missing. Install requirements.txt or run "
        "`pip install fastapi uvicorn httpx`."
    ) from exc


@dataclass(frozen=True)
class WebConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    pano_export: str = "auto"
    memory_guidance: MemoryGuidanceConfig = field(default_factory=MemoryGuidanceConfig)
    pano_viewer: PanoExportConfig = field(default_factory=PanoExportConfig)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def create_app(config: WebConfig | None = None) -> FastAPI:
    load_dotenv(PROJECT_ROOT / ".env")
    resolved_config = config or WebConfig()
    memory_app = MemoryGuidanceWebApp(resolved_config.memory_guidance)
    pano_status = prepare_pano_viewer(resolved_config)

    app = FastAPI(title="Memory Navigation Web Tools")
    app.state.web_config = resolved_config
    app.state.memory_guidance = memory_app
    app.state.pano_status = pano_status

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return _index_html(resolved_config)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        pano_output_dir = project_path(resolved_config.pano_viewer.output_dir)
        return {
            "status": "ok",
            "server": {"host": resolved_config.host, "port": resolved_config.port},
            "pano_export": app.state.pano_status,
            "tools": {
                "memory_guidance": {
                    "available": MEMORY_GUIDANCE_STATIC_ROOT.exists(),
                    "path": "/memory-guidance/",
                },
                "pano_viewer": {
                    "available": (pano_output_dir / "index.html").exists()
                    and (pano_output_dir / "viewer_data.json").exists(),
                    "path": "/pano-viewer/",
                    "output_dir": str(pano_output_dir),
                },
            },
        }

    @app.post("/memory-guidance/api/guide")
    async def memory_guidance(payload_request: Request) -> JSONResponse:
        try:
            body = await payload_request.body()
            payload = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object.")
            return JSONResponse(memory_app.guide(payload))
        except Exception as exc:
            return JSONResponse(
                {
                    "action_request": "error",
                    "message_zh": "伺服器處理請求時發生錯誤。",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                status_code=500,
            )

    @app.get("/memory-guidance/", include_in_schema=False)
    def memory_guidance_index() -> Response:
        return serve_file(MEMORY_GUIDANCE_STATIC_ROOT, "index.html")

    @app.get("/memory-guidance/{path:path}", include_in_schema=False)
    def memory_guidance_static(path: str) -> Response:
        return serve_file(MEMORY_GUIDANCE_STATIC_ROOT, path)

    @app.get("/pano-viewer/", include_in_schema=False)
    def pano_viewer_index() -> Response:
        return serve_file(project_path(resolved_config.pano_viewer.output_dir), "index.html")

    @app.post("/pano-viewer/api/trajectory-panorama-frames")
    async def pano_viewer_trajectory_panorama_frames(payload_request: Request) -> JSONResponse:
        try:
            body = await payload_request.body()
            payload = json.loads(body.decode("utf-8")) if body else {}
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object.")
            return JSONResponse(build_panorama_frame_manifest(payload))
        except Exception as exc:
            return JSONResponse(
                {
                    "frames": [],
                    "missing_frames": [],
                    "error": f"{type(exc).__name__}: {exc}",
                },
                status_code=400,
            )

    @app.get("/pano-viewer/api/render-image", include_in_schema=False)
    def pano_viewer_render_image(path: str) -> Response:
        try:
            image_path = resolve_project_image_path(path)
        except ValueError as exc:
            return PlainTextResponse(str(exc), status_code=403)
        if not image_path.exists() or not image_path.is_file():
            return PlainTextResponse("Not Found", status_code=404)
        media_type, _ = mimetypes.guess_type(str(image_path))
        return Response(image_path.read_bytes(), media_type=media_type or "application/octet-stream")

    @app.get("/pano-viewer/{path:path}", include_in_schema=False)
    def pano_viewer_static(path: str) -> Response:
        return serve_file(project_path(resolved_config.pano_viewer.output_dir), path)

    return app


def prepare_pano_viewer(config: WebConfig) -> dict[str, Any]:
    mode = config.pano_export
    output_dir = project_path(config.pano_viewer.output_dir)
    status: dict[str, Any] = {
        "mode": mode,
        "ran": False,
        "ok": True,
        "output_dir": str(output_dir),
    }
    if mode not in {"auto", "missing", "never"}:
        status.update({"ok": False, "error": f"Unsupported pano_export mode: {mode}"})
        return status
    should_run = mode == "auto" or (
        mode == "missing"
        and not ((output_dir / "index.html").exists() and (output_dir / "viewer_data.json").exists())
    )
    if not should_run:
        status["skipped_reason"] = "disabled" if mode == "never" else "existing_export"
        return status
    try:
        result = export_pano_viewer(config.pano_viewer)
    except Exception as exc:  # pragma: no cover - depends on local artifact availability.
        status.update({"ran": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return status
    status.update({"ran": True, "result": result})
    return status


def serve_file(root: Path, relative_path: str) -> Response:
    root = root.resolve()
    candidate = (root / relative_path.lstrip("/")).resolve()
    if root not in candidate.parents and candidate != root:
        return PlainTextResponse("Forbidden", status_code=403)
    if candidate.is_dir():
        candidate = candidate / "index.html"
    if not candidate.exists() or not candidate.is_file():
        return PlainTextResponse("Not Found", status_code=404)
    media_type, _ = mimetypes.guess_type(str(candidate))
    return Response(candidate.read_bytes(), media_type=media_type or "application/octet-stream")



def build_panorama_frame_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    movements = _trajectory_movements(payload)
    frames: list[dict[str, Any]] = []
    missing_frames: list[dict[str, Any]] = []

    for frame_index, movement in enumerate(movements):
        pano_id = _string_or_none(movement.get("current_pano_id"))
        if not pano_id:
            missing_frames.append({"frame_index": frame_index, "reason": "missing_current_pano_id"})
            continue
        image_path = _image_path_for_movement(movement)
        if image_path is None:
            missing_frames.append({"frame_index": frame_index, "pano_id": pano_id, "reason": "missing_render"})
            continue
        frames.append(
            _panorama_frame_payload(
                frame_index=frame_index,
                pano_id=pano_id,
                room_id=_string_or_none(movement.get("current_room_id")),
                next_room_id=_string_or_none(movement.get("next_room_id")),
                subgoal_room_id=_string_or_none(movement.get("subgoal_room_id")),
                active_target_room_id=_string_or_none(movement.get("active_target_room_id")),
                heading=_finite_number(movement.get("selected_action_heading")),
                image_path=image_path,
            )
        )

    final_pano_id = _final_pano_id(payload, movements)
    if final_pano_id:
        final_index = len(movements)
        last_movement = movements[-1] if movements else {}
        final_image_path = find_render_capture(
            final_pano_id,
            heading=_finite_number(last_movement.get("selected_action_heading")),
        )
        if final_image_path is None:
            missing_frames.append({"frame_index": final_index, "pano_id": final_pano_id, "reason": "missing_final_render"})
        else:
            frames.append(
                _panorama_frame_payload(
                    frame_index=final_index,
                    pano_id=final_pano_id,
                    room_id=_string_or_none(last_movement.get("next_room_id")),
                    next_room_id=None,
                    subgoal_room_id=_string_or_none(last_movement.get("subgoal_room_id"))
                    or _string_or_none(payload.get("target_room_id")),
                    active_target_room_id=_string_or_none(last_movement.get("active_target_room_id"))
                    or _string_or_none(payload.get("target_room_id")),
                    heading=_finite_number(last_movement.get("selected_action_heading")),
                    image_path=final_image_path,
                )
            )

    return {
        "schema_version": 1,
        "frame_count": len(frames),
        "missing_count": len(missing_frames),
        "frames": frames,
        "missing_frames": missing_frames,
    }


def _trajectory_movements(payload: dict[str, Any]) -> list[dict[str, Any]]:
    movements: list[dict[str, Any]] = []
    for round_payload in payload.get("rounds") or []:
        if not isinstance(round_payload, dict):
            continue
        for step in round_payload.get("movement_steps") or []:
            if isinstance(step, dict):
                movements.append(step)
    return movements


def _image_path_for_movement(movement: dict[str, Any]) -> Path | None:
    pano_id = _string_or_none(movement.get("current_pano_id"))
    if not pano_id:
        return None

    action_heading = _finite_number(movement.get("selected_action_heading"))
    if action_heading is not None:
        action_image = find_render_capture(pano_id, heading=action_heading)
        if action_image is not None:
            return action_image

    selected_capture_index = _integer_or_none(movement.get("selected_capture_index"))
    for score in _candidate_view_scores(movement, selected_capture_index):
        raw_path = _string_or_none(score.get("path")) if isinstance(score, dict) else None
        if not raw_path:
            continue
        try:
            image_path = resolve_project_image_path(raw_path)
        except ValueError:
            continue
        if image_path.exists():
            return image_path

    return find_render_capture(
        pano_id,
        capture_index=selected_capture_index,
        heading=_finite_number(movement.get("selected_view_heading")),
    )


def _candidate_view_scores(movement: dict[str, Any], selected_capture_index: int | None) -> list[dict[str, Any]]:
    scores = [score for score in movement.get("view_scores") or [] if isinstance(score, dict)]
    selected = [score for score in scores if score.get("selected") is True]
    if selected:
        return selected + [score for score in scores if score not in selected]
    if selected_capture_index is not None:
        matching = [score for score in scores if _integer_or_none(score.get("capture_index")) == selected_capture_index]
        if matching:
            return matching + [score for score in scores if score not in matching]
    return scores


def _panorama_frame_payload(
    *,
    frame_index: int,
    pano_id: str,
    room_id: str | None,
    next_room_id: str | None,
    subgoal_room_id: str | None,
    active_target_room_id: str | None,
    heading: float | None,
    image_path: Path,
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "pano_id": pano_id,
        "room_id": room_id,
        "next_room_id": next_room_id,
        "subgoal_room_id": subgoal_room_id,
        "active_target_room_id": active_target_room_id,
        "heading": heading,
        "duration_ms": 900,
        "image_url": "/pano-viewer/api/render-image?path=" + quote(str(image_path), safe=""),
    }


def _final_pano_id(payload: dict[str, Any], movements: list[dict[str, Any]]) -> str | None:
    final_pano_id = _string_or_none(payload.get("final_pano_id"))
    if final_pano_id:
        return final_pano_id
    pano_path = payload.get("pano_path")
    if isinstance(pano_path, list) and pano_path:
        return _string_or_none(pano_path[-1])
    if movements:
        return _string_or_none(movements[-1].get("next_pano_id"))
    return None


def resolve_project_image_path(raw_path: str) -> Path:
    if not raw_path:
        raise ValueError("Empty image path.")
    candidate = Path(raw_path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if not _is_allowed_image_path(resolved):
            marker = f"{PROJECT_ROOT.name}/"
            text = str(raw_path)
            if marker in text:
                resolved = (PROJECT_ROOT / text.split(marker, 1)[1]).resolve()
    else:
        resolved = (PROJECT_ROOT / candidate).resolve()
    if not _is_allowed_image_path(resolved):
        raise ValueError("Image path is outside the allowed project image directories.")
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Unsupported image type.")
    return resolved


def find_render_capture(pano_id: str, capture_index: int | None = None, heading: float | None = None) -> Path | None:
    pano_dir = (PROJECT_ROOT / "renders" / "room_grounding_fov90" / pano_id).resolve()
    if not _path_is_relative_to(pano_dir, PROJECT_ROOT / "renders") or not pano_dir.is_dir():
        return None
    candidates = _render_candidates(pano_dir, pano_id, capture_index)
    if not candidates:
        candidates = _render_candidates(pano_dir, pano_id, None)
    if not candidates:
        return None
    if heading is None:
        return candidates[0]
    return min(candidates, key=lambda path: _heading_distance(_capture_heading(path), heading))


def _render_candidates(pano_dir: Path, pano_id: str, capture_index: int | None) -> list[Path]:
    candidates: list[Path] = []
    capture_prefix = None if capture_index is None else f"{pano_id}_{capture_index:02d}_"
    for path in sorted(pano_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if capture_prefix is not None and not path.name.startswith(capture_prefix):
            continue
        if capture_prefix is None and not path.name.startswith(f"{pano_id}_"):
            continue
        candidates.append(path)
    return candidates


def _capture_heading(path: Path) -> float | None:
    match = re.search(r"_([0-9]{3})deg\.[^.]+$", path.name)
    return float(match.group(1)) if match else None


def _heading_distance(left: float | None, right: float) -> float:
    if left is None:
        return 360.0
    return abs(((left - right + 180.0) % 360.0) - 180.0)


def _is_allowed_image_path(path: Path) -> bool:
    return any(_path_is_relative_to(path, root) for root in (PROJECT_ROOT / "renders", PROJECT_ROOT / "outputs"))


def _path_is_relative_to(path: Path, root: Path) -> bool:
    resolved_root = root.resolve()
    return path == resolved_root or resolved_root in path.parents


def _string_or_none(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _finite_number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _index_html(config: WebConfig) -> str:
    return f"""<!doctype html>
<html lang="zh-Hant">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Memory Navigation Web Tools</title>
    <style>
      body {{
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #f5f3ee;
        color: #212121;
      }}
      main {{
        max-width: 760px;
        margin: 0 auto;
        padding: 56px 24px;
      }}
      h1 {{ margin: 0 0 12px; font-size: 34px; }}
      p {{ color: #5a5852; line-height: 1.6; }}
      nav {{ display: grid; gap: 12px; margin-top: 28px; }}
      a {{
        display: block;
        padding: 18px 20px;
        border: 1px solid #d5d0c5;
        border-radius: 8px;
        color: #1f4b45;
        background: #fffdf8;
        text-decoration: none;
        font-weight: 700;
      }}
      a span {{ display: block; margin-top: 4px; color: #6b6860; font-weight: 400; }}
      code {{ background: #ebe6da; border-radius: 4px; padding: 2px 5px; }}
    </style>
  </head>
  <body>
    <main>
      <h1>Memory Navigation Web Tools</h1>
      <p>Unified FastAPI server running on <code>{config.host}:{config.port}</code>.</p>
      <nav>
        <a href="/memory-guidance/">Memory Guidance<span>Upload localization and passage images for navigation guidance.</span></a>
        <a href="/pano-viewer/">Pano Viewer<span>Explore the exported panorama graph and Street View links.</span></a>
      </nav>
    </main>
  </body>
</html>
"""
