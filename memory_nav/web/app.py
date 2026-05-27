from __future__ import annotations

import json
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
