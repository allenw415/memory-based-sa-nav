from __future__ import annotations

import base64
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_nav import get_env_value, resolve_model_environment
from memory_nav.cli._common import PROJECT_ROOT, load_normalized_artifacts, resolve_project_path
from memory_nav.data.memory_localization import DEFAULT_SIGLIP2_MODEL
from memory_nav.memory import (
    InteractiveMemoryNavigator,
    MemoryImageRetriever,
    MemoryRoomLocalizer,
    PassageAlignmentAdvisor,
)


MEMORY_GUIDANCE_STATIC_ROOT = Path(__file__).resolve().parent / "static/memory_guidance"


@dataclass(frozen=True)
class MemoryGuidanceConfig:
    upload_dir: str = "outputs/memory_guidance_web/uploads"
    artifacts_dir: str = "dataset/sites/british_museum/normalized"
    index_path: str = "artifacts/memory_localization/floor0_siglip2_images.npz"
    metadata_path: str = "artifacts/memory_localization/floor0_siglip2_images.metadata.json"
    faiss_path: str = "artifacts/memory_localization/floor0_siglip2_images.faiss"
    no_faiss: bool = False
    embedding_model: str = DEFAULT_SIGLIP2_MODEL
    device: str = "auto"
    batch_size: int = 8
    retrieval_top_k: int = 10
    confidence_threshold: float = 0.55
    margin_threshold: float = 0.15
    llm_model: str = "gpt-5-mini"
    llm_api_key: str | None = None
    llm_api_kind: str = "responses"
    llm_api_base: str = "https://api.openai.com/v1"
    llm_timeout: float = 60.0


class MemoryGuidanceWebApp:
    def __init__(self, config: MemoryGuidanceConfig):
        self.config = config
        self.upload_root = resolve_project_path(config.upload_dir)
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
        except Exception as exc:  # pragma: no cover - exercised by browser/API smoke tests.
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
            default_model=self.config.llm_model,
            default_api_base=self.config.llm_api_base,
            default_api_kind=self.config.llm_api_kind,
        )
        artifacts = load_normalized_artifacts(self.config.artifacts_dir, room_graph=True)
        room_graph = artifacts.room_graph or {}
        retriever = MemoryImageRetriever(
            index_path=resolve_project_path(self.config.index_path),
            metadata_path=resolve_project_path(self.config.metadata_path),
            faiss_path=resolve_project_path(self.config.faiss_path),
            embedding_model=self.config.embedding_model,
            device=self.config.device,
            batch_size=self.config.batch_size,
            use_faiss=not self.config.no_faiss,
            project_root=PROJECT_ROOT,
        )
        localizer = MemoryRoomLocalizer(
            retriever,
            retrieval_top_k=self.config.retrieval_top_k,
            confidence_threshold=self.config.confidence_threshold,
            margin_threshold=self.config.margin_threshold,
        )
        advisor = PassageAlignmentAdvisor(
            room_graph=room_graph,
            memory_retriever=retriever,
            model=model_env.model_name,
            api_key=self.config.llm_api_key
            or model_env.api_key
            or get_env_value("NAV_KEY", "NAV_API_KEY", "ST_NAV_API_KEY", "OPENAI_API_KEY"),
            api_base=model_env.api_base,
            api_kind=model_env.api_kind,
            request_timeout=self.config.llm_timeout,
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
