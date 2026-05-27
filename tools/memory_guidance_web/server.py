from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent / "web"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory_nav import get_env_value, load_dotenv, resolve_model_environment  # noqa: E402
from memory_nav.cli._common import load_normalized_artifacts, resolve_project_path  # noqa: E402
from memory_nav.memory import (  # noqa: E402
    InteractiveMemoryNavigator,
    MemoryImageRetriever,
    MemoryRoomLocalizer,
    PassageAlignmentAdvisor,
)
from memory_nav.data.memory_localization import DEFAULT_SIGLIP2_MODEL  # noqa: E402


load_dotenv(PROJECT_ROOT / ".env")


class MemoryGuidanceWebApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.upload_root = resolve_project_path(args.upload_dir)
        self.upload_root.mkdir(parents=True, exist_ok=True)
        self._navigator: InteractiveMemoryNavigator | None = None

    def guide(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_room_id = str(payload.get("target_room_id") or "").strip()
        waypoint_room_ids = self._parse_waypoints(payload.get("waypoint_room_ids"))
        if not target_room_id:
            return {
                "action_request": "missing_target",
                "message_zh": "請先輸入目標展廳，例如 Room 23。",
            }

        batch_dir = self.upload_root / time.strftime("%Y%m%d-%H%M%S")
        localization_images = self._save_image_list(
            payload.get("localization_images"),
            batch_dir=batch_dir / "localization",
        )
        passage_images = self._save_passage_images(
            payload.get("passage_images"),
            batch_dir=batch_dir / "passages",
        )
        if not localization_images:
            return {
                "action_request": "capture_more_localization_views",
                "target_room_id": target_room_id,
                "waypoint_room_ids": waypoint_room_ids,
                "message_zh": "請先上傳一張你現在看到的展廳照片，我會用這張照片判斷你目前在哪裡。",
            }

        try:
            result = self._ensure_navigator().guide(
                target_room_id=target_room_id,
                waypoint_room_ids=waypoint_room_ids,
                localization_images=localization_images,
                passage_images=passage_images,
            )
        except Exception as exc:  # pragma: no cover - exercised manually in the browser demo.
            return {
                "action_request": "error",
                "message_zh": "執行互動式記憶導航時發生錯誤。請確認 memory index、模型依賴與圖片路徑。",
                "error": f"{type(exc).__name__}: {exc}",
            }
        result["uploaded_files"] = {
            "localization_images": [str(path) for path in localization_images],
            "passage_images": {label: str(path) for label, path in passage_images.items()},
        }
        return result

    def _ensure_navigator(self) -> InteractiveMemoryNavigator:
        if self._navigator is not None:
            return self._navigator
        model_env = resolve_model_environment(
            default_model=self.args.llm_model,
            default_api_base=self.args.llm_api_base,
            default_api_kind=self.args.llm_api_kind,
        )
        artifacts = load_normalized_artifacts(self.args.artifacts_dir, room_graph=True)
        room_graph = artifacts.room_graph or {}
        retriever = MemoryImageRetriever(
            index_path=resolve_project_path(self.args.index_path),
            metadata_path=resolve_project_path(self.args.metadata_path),
            faiss_path=resolve_project_path(self.args.faiss_path),
            embedding_model=self.args.embedding_model,
            device=self.args.device,
            batch_size=self.args.batch_size,
            use_faiss=not self.args.no_faiss,
            project_root=PROJECT_ROOT,
        )
        localizer = MemoryRoomLocalizer(
            retriever,
            retrieval_top_k=self.args.retrieval_top_k,
            confidence_threshold=self.args.confidence_threshold,
            margin_threshold=self.args.margin_threshold,
        )
        advisor = PassageAlignmentAdvisor(
            room_graph=room_graph,
            memory_retriever=retriever,
            model=model_env.model_name,
            api_key=self.args.llm_api_key
            or model_env.api_key
            or get_env_value("NAV_KEY", "NAV_API_KEY", "ST_NAV_API_KEY", "OPENAI_API_KEY"),
            api_base=model_env.api_base,
            api_kind=model_env.api_kind,
            request_timeout=self.args.llm_timeout,
        )
        self._navigator = InteractiveMemoryNavigator(
            room_graph=room_graph,
            localizer=localizer,
            passage_advisor=advisor,
        )
        return self._navigator

    @staticmethod
    def _parse_waypoints(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    @classmethod
    def _save_image_list(cls, values: object, *, batch_dir: Path) -> list[Path]:
        if not isinstance(values, list):
            return []
        saved = []
        for index, record in enumerate(values):
            if not isinstance(record, dict):
                continue
            data_url = record.get("data_url")
            if not isinstance(data_url, str) or not data_url:
                continue
            name = cls._safe_filename(str(record.get("name") or f"image_{index}.png"))
            saved.append(cls._write_data_url(data_url, batch_dir / name))
        return saved

    @classmethod
    def _save_passage_images(cls, values: object, *, batch_dir: Path) -> dict[str, Path]:
        if not isinstance(values, dict):
            return {}
        saved: dict[str, Path] = {}
        for label, record in values.items():
            if not isinstance(label, str) or not label.strip() or not isinstance(record, dict):
                continue
            data_url = record.get("data_url")
            if not isinstance(data_url, str) or not data_url:
                continue
            name = cls._safe_filename(str(record.get("name") or f"{label}.png"))
            saved[label.strip()] = cls._write_data_url(data_url, batch_dir / name)
        return saved

    @staticmethod
    def _write_data_url(data_url: str, output_path: Path) -> Path:
        if "," not in data_url:
            raise ValueError("Expected data URL with base64 payload.")
        header, encoded = data_url.split(",", 1)
        suffix = ".png"
        if header.startswith("data:") and ";" in header:
            mime = header[5:].split(";", 1)[0]
            suffix = mimetypes.guess_extension(mime) or suffix
        if not output_path.suffix:
            output_path = output_path.with_suffix(suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(encoded))
        return output_path.resolve()

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
        return cleaned or "image.png"


class MemoryGuidanceRequestHandler(BaseHTTPRequestHandler):
    app: MemoryGuidanceWebApp

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in {"", "/"}:
            self._serve_file(WEB_ROOT / "index.html")
            return
        candidate = (WEB_ROOT / path.lstrip("/")).resolve()
        if WEB_ROOT.resolve() not in candidate.parents and candidate != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not candidate.exists() or not candidate.is_file():
            self.send_error(404)
            return
        self._serve_file(candidate)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] != "/api/guide":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            if not isinstance(payload, dict):
                raise ValueError("Expected JSON object.")
            result = self.app.guide(payload)
            self._send_json(result)
        except Exception as exc:
            self._send_json(
                {
                    "action_request": "error",
                    "message_zh": "伺服器處理請求時發生錯誤。",
                    "error": f"{type(exc).__name__}: {exc}",
                },
                status=500,
            )

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[memory-guidance-web] " + format % args + "\n")

    def _serve_file(self, path: Path) -> None:
        mime_type, _ = mimetypes.guess_type(str(path))
        content = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the interactive RAG memory-navigation web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--upload-dir", default="outputs/memory_guidance_web/uploads")
    parser.add_argument("--artifacts-dir", default="dataset/sites/british_museum/normalized")
    parser.add_argument("--index-path", default="artifacts/memory_localization/floor0_siglip2_images.npz")
    parser.add_argument("--metadata-path", default="artifacts/memory_localization/floor0_siglip2_images.metadata.json")
    parser.add_argument("--faiss-path", default="artifacts/memory_localization/floor0_siglip2_images.faiss")
    parser.add_argument("--no-faiss", action="store_true")
    parser.add_argument("--embedding-model", default=DEFAULT_SIGLIP2_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--llm-api-key")
    parser.add_argument("--llm-api-kind", default="responses")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--llm-timeout", type=float, default=60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    MemoryGuidanceRequestHandler.app = MemoryGuidanceWebApp(args)
    server = ThreadingHTTPServer((args.host, args.port), MemoryGuidanceRequestHandler)
    print(f"Serving memory guidance web demo at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping memory guidance web demo.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
