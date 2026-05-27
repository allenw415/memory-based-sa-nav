from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from pathlib import Path
from typing import Callable

from ..common.env import resolve_model_environment
from ..common.model_client import DEFAULT_OPENAI_API_BASE, ModelResponseClient, parse_json_output, resolve_api_kind
from ..common.room_profiles import compact_visual_profile
from .retrieval import MemoryImageRetriever


class PassageAlignmentAdvisor:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        memory_retriever: MemoryImageRetriever | None = None,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_kind: str | None = None,
        request_timeout: float | None = None,
        response_client: Callable[[dict], dict] | None = None,
        max_memory_images: int = 3,
        evidence_selection_mode: str = "semantic",
        passage_semantic_query_provider: Callable[[list[tuple[str, Path]]], object] | None = None,
    ):
        settings = resolve_model_environment(
            default_model="gpt-5-mini",
            default_api_base=DEFAULT_OPENAI_API_BASE,
            default_api_kind="responses",
        )
        self.room_graph = room_graph
        self.memory_retriever = memory_retriever
        self.model = model or settings.model_name or "gpt-5-mini"
        self.max_memory_images = int(max_memory_images)
        self.evidence_selection_mode = (evidence_selection_mode or "semantic").strip().lower()
        self.passage_semantic_query_provider = passage_semantic_query_provider
        self.last_memory_selection_mode: str | None = None
        self.last_semantic_query: str | None = None
        self.model_client = ModelResponseClient(
            provider=settings.provider,
            api_key=api_key if api_key is not None else settings.api_key,
            api_base=(api_base or settings.api_base or DEFAULT_OPENAI_API_BASE).rstrip("/"),
            api_kind=resolve_api_kind(api_kind or settings.api_kind),
            request_timeout=float(request_timeout if request_timeout is not None else (settings.request_timeout or 60.0)),
            num_ctx=settings.num_ctx,
            temperature=settings.temperature,
            response_client=response_client,
        )
        self.last_request_body: dict | None = None
        self.last_response_payload: dict | None = None

    def advise(
        self,
        *,
        current_room_id: str,
        next_room_id: str,
        active_target_room_id: str,
        route: list[str],
        passage_images: dict[str, str | Path],
        localization: dict | None = None,
    ) -> dict:
        self.last_memory_selection_mode = None
        self.last_semantic_query = None
        candidates = self._normalize_passage_images(passage_images)
        if not candidates:
            return {
                "chosen_passage_label": None,
                "target_room_id": next_room_id,
                "direction_hint": "",
                "confidence": 0.0,
                "evidence": [],
                "rationale_zh": "沒有可用的通道照片。",
                "message_zh": "請拍一張包含附近主要通道或出口的照片。",
            }
        if not self.model_client.is_configured():
            return self._fallback_guidance(
                current_room_id=current_room_id,
                next_room_id=next_room_id,
                active_target_room_id=active_target_room_id,
                route=route,
                candidates=candidates,
            )

        request_body = self._build_request_body(
            current_room_id=current_room_id,
            next_room_id=next_room_id,
            active_target_room_id=active_target_room_id,
            route=route,
            candidates=candidates,
            localization=localization or {},
        )
        self.last_request_body = self._clone_json(request_body)
        payload = self.model_client.create(request_body)
        self.last_response_payload = self._clone_json(payload)
        parsed = parse_json_output(payload)
        return self._normalize_response(
            parsed,
            labels=[label for label, _ in candidates],
            next_room_id=next_room_id,
        )

    def _build_request_body(
        self,
        *,
        current_room_id: str,
        next_room_id: str,
        active_target_room_id: str,
        route: list[str],
        candidates: list[tuple[str, Path]],
        localization: dict,
    ) -> dict:
        labels = [label for label, _ in candidates]
        content: list[dict] = [
            {
                "type": "input_text",
                "text": self._alignment_text(
                    current_room_id=current_room_id,
                    next_room_id=next_room_id,
                    active_target_room_id=active_target_room_id,
                    route=route,
                    candidate_labels=labels,
                    localization=localization,
                ),
            }
        ]
        for label, image_path in candidates:
            content.append({"type": "input_text", "text": f"候選通道照片 label={label}。"})
            content.append({"type": "input_image", "image_url": self._image_to_data_url(image_path), "detail": "high"})

        for memory in self._target_memories(next_room_id, candidates=candidates):
            content.append({"type": "input_text", "text": self._memory_context_text(memory)})
            capture_path = memory.get("capture_path")
            if memory.get("image_available") and isinstance(capture_path, str):
                content.append({"type": "input_image", "image_url": self._image_to_data_url(Path(capture_path)), "detail": "high"})

        return {
            "model": self.model,
            "instructions": self._instructions(),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "passage_alignment_guidance",
                    "strict": True,
                    "schema": self._schema(labels, next_room_id),
                }
            },
        }

    def _alignment_text(
        self,
        *,
        current_room_id: str,
        next_room_id: str,
        active_target_room_id: str,
        route: list[str],
        candidate_labels: list[str],
        localization: dict,
    ) -> str:
        current_node = self.room_graph.get(current_room_id, {})
        next_node = self.room_graph.get(next_room_id, {})
        active_node = self.room_graph.get(active_target_room_id, {})
        transition = self._transition_context(current_room_id, next_room_id)
        return "\n".join(
            [
                "互動式博物館導航任務。",
                f"目前定位房間：{current_room_id}",
                f"下一個要走向的房間：{next_room_id}",
                f"目前 waypoint/goal：{active_target_room_id}",
                f"完整 room route：{route}",
                f"候選通道 labels：{candidate_labels}",
                "",
                "目前房間語意：",
                json.dumps(self._room_context(current_room_id, current_node), ensure_ascii=False),
                "下一個房間語意：",
                json.dumps(self._room_context(next_room_id, next_node), ensure_ascii=False),
                "目前 active target 語意：",
                json.dumps(self._room_context(active_target_room_id, active_node), ensure_ascii=False),
                "room graph transition：",
                json.dumps(transition, ensure_ascii=False),
                "RAG 定位摘要：",
                json.dumps(localization, ensure_ascii=False),
                "",
                "下一個房間的 memory evidence 會優先用候選通道的中性語意描述，在目標房間影像記憶中檢索取得。",
                "memory evidence 不會提供 matched passage label 或 similarity score，避免提示洩漏。",
                "請只根據候選通道照片、目標房間記憶照片與 room graph 語意做空間對齊，選出使用者應該走的通道。",
                "請用繁體中文產生給使用者的自然語言導航訊息。",
            ]
        )

    @staticmethod
    def _instructions() -> str:
        return " ".join(
            [
                "你是互動式博物館導航助手。",
                "你會收到使用者拍攝的候選通道照片、目前房間、下一個房間、room graph 方向關係，以及下一個房間的 RAG 記憶線索。",
                "請選出最可能通往下一個房間的通道。",
                "不要跳過 waypoint；如果 active target 不是 final goal，也只能引導到 route 上的下一個房間。",
                "如果圖片證據不足，仍需選出最合理候選，但降低 confidence 並在 evidence 說明不確定性。",
                "回覆必須是 JSON。",
            ]
        )

    @staticmethod
    def _schema(labels: list[str], next_room_id: str) -> dict:
        return {
            "type": "object",
            "properties": {
                "chosen_candidate_label": {"type": "string", "enum": labels},
                "target_room_id": {"type": "string", "enum": [next_room_id]},
                "direction_hint": {"type": "string"},
                "confidence": {"type": "number"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "rationale_zh": {"type": "string"},
                "message_zh": {"type": "string"},
            },
            "required": [
                "chosen_candidate_label",
                "target_room_id",
                "direction_hint",
                "confidence",
                "evidence",
                "rationale_zh",
                "message_zh",
            ],
            "additionalProperties": False,
        }

    def _passage_semantic_query(self, candidates: list[tuple[str, Path]]) -> str | None:
        if self.passage_semantic_query_provider is not None:
            return self._coerce_semantic_query(self.passage_semantic_query_provider(candidates))
        if not self.model_client.is_configured():
            return None
        try:
            payload = self.model_client.create(self._semantic_query_request_body(candidates))
            parsed = parse_json_output(payload)
        except (RuntimeError, OSError, ValueError, TypeError):
            return None
        return self._coerce_semantic_query(parsed)

    def _semantic_query_request_body(self, candidates: list[tuple[str, Path]]) -> dict:
        content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    "請觀察以下候選通道照片，產生一段中性的視覺語意檢索 query。"
                    "只描述可見的建築、通道、展品材質、雕像、牆面、燈光、空間結構等線索；"
                    "不要包含 front/left/right、候選編號、方向、Room 編號、目標房間名稱或任何導航答案。"
                ),
            }
        ]
        for index, (_, image_path) in enumerate(candidates, start=1):
            content.append({"type": "input_text", "text": f"候選通道照片 {index}。"})
            content.append({"type": "input_image", "image_url": self._image_to_data_url(image_path), "detail": "high"})
        return {
            "model": self.model,
            "instructions": self._semantic_query_instructions(),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "passage_semantic_query",
                    "strict": True,
                    "schema": self._semantic_query_schema(),
                }
            },
        }

    @staticmethod
    def _semantic_query_instructions() -> str:
        return " ".join(
            [
                "你會為博物館通道照片產生影像檢索用的中性語意描述。",
                "描述只能包含視覺元素，不可包含候選 label、左右前後方向、Room ID、目標房間名稱或行動建議。",
                "輸出 JSON。",
            ]
        )

    @staticmethod
    def _semantic_query_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "semantic_query": {"type": "string"},
                "visual_keywords": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["semantic_query", "visual_keywords"],
            "additionalProperties": False,
        }

    @staticmethod
    def _coerce_semantic_query(payload: object) -> str | None:
        if isinstance(payload, str):
            query = payload.strip()
            return query or None
        if not isinstance(payload, dict):
            return None
        fragments: list[str] = []
        semantic_query = payload.get("semantic_query")
        if isinstance(semantic_query, str) and semantic_query.strip():
            fragments.append(semantic_query.strip())
        keywords = payload.get("visual_keywords")
        if isinstance(keywords, list):
            fragments.extend(str(keyword).strip() for keyword in keywords if str(keyword).strip())
        query = ", ".join(fragments).strip()
        return query or None

    def _target_memories(self, room_id: str, *, candidates: list[tuple[str, Path]] | None = None) -> list[dict]:
        if self.memory_retriever is None:
            self.last_memory_selection_mode = "vlm_without_memory"
            return []

        mode = self.evidence_selection_mode
        if candidates and mode in {"semantic", "auto"} and hasattr(self.memory_retriever, "retrieve_room_memories_for_text_queries"):
            semantic_query = self._passage_semantic_query(candidates)
            self.last_semantic_query = semantic_query
            if semantic_query:
                try:
                    memories = self.memory_retriever.retrieve_room_memories_for_text_queries(
                        room_id,
                        [semantic_query],
                        top_k_per_query=self.max_memory_images,
                        max_memories=self.max_memory_images,
                    )
                except (RuntimeError, OSError, ValueError, TypeError):
                    memories = []
                if memories:
                    self.last_memory_selection_mode = "vlm_with_semantic_memory"
                    return memories

        if candidates and mode == "image_vector" and hasattr(self.memory_retriever, "retrieve_room_memories_for_query_images"):
            passage_labels = [label for label, _ in candidates]
            passage_paths = [path for _, path in candidates]
            try:
                memories = self.memory_retriever.retrieve_room_memories_for_query_images(
                    room_id,
                    passage_paths,
                    passage_labels=passage_labels,
                    top_k_per_query=2,
                    max_memories=self.max_memory_images,
                )
            except (RuntimeError, OSError, ValueError, TypeError):
                memories = []
            if memories:
                self.last_memory_selection_mode = "vlm_with_vector_selected_memory"
                return memories

        self.last_memory_selection_mode = "vlm_with_room_memory"
        return self.memory_retriever.retrieve_room_memories(room_id, top_k=self.max_memory_images)

    @staticmethod
    def _memory_context_text(memory: dict) -> str:
        # Do not expose matched_passage_label or similarity to the VLM. Those
        # values are useful for debugging retrieval, but leaking them turns the
        # alignment task into a prompted answer rather than visual reasoning.
        parts = [
            "下一個展廳的記憶照片：",
            f"room={memory.get('room_id')}",
            f"pano={memory.get('pano_id')}",
            f"capture={memory.get('capture_label')}",
            f"heading={memory.get('capture_heading')}",
        ]
        retrieval_mode = memory.get("retrieval_mode")
        if isinstance(retrieval_mode, str) and retrieval_mode:
            parts.append("retrieval_note=target_room_memory_preselected_without_candidate_label")
        return " ".join(parts)

    def _transition_context(self, current_room_id: str, next_room_id: str) -> dict:
        for neighbor in self.room_graph.get(current_room_id, {}).get("neighbors", []):
            if isinstance(neighbor, dict) and neighbor.get("target_room_id") == next_room_id:
                return {
                    "target_room_id": next_room_id,
                    "allocentric_direction": neighbor.get("allocentric_direction"),
                    "allocentric_heading_deg": neighbor.get("allocentric_heading_deg"),
                    "transition_type": neighbor.get("transition_type"),
                }
        return {"target_room_id": next_room_id}

    @staticmethod
    def _room_context(room_id: str, node: dict) -> dict:
        payload = {
            "room_id": room_id,
            "title": node.get("title"),
            "category": node.get("category"),
            "aliases": list(node.get("aliases", [])) if isinstance(node.get("aliases"), list) else [],
        }
        profile = compact_visual_profile(node)
        if profile:
            payload["visual_profile"] = profile
        return payload

    @staticmethod
    def _normalize_passage_images(passage_images: dict[str, str | Path]) -> list[tuple[str, Path]]:
        candidates = []
        for label, path in passage_images.items():
            if not isinstance(label, str) or not label:
                continue
            image_path = Path(path)
            if image_path.exists():
                candidates.append((label, image_path))
        return candidates

    def _normalize_response(self, parsed: dict, *, labels: list[str], next_room_id: str) -> dict:
        chosen = parsed.get("chosen_candidate_label")
        if chosen not in labels:
            chosen = labels[0] if labels else None
        confidence = parsed.get("confidence")
        evidence = parsed.get("evidence")
        return {
            "chosen_passage_label": chosen,
            "target_room_id": next_room_id,
            "direction_hint": parsed.get("direction_hint") if isinstance(parsed.get("direction_hint"), str) else "",
            "confidence": max(0.0, min(1.0, float(confidence))) if isinstance(confidence, (int, float)) else 0.0,
            "evidence": [value for value in evidence if isinstance(value, str) and value] if isinstance(evidence, list) else [],
            "rationale_zh": parsed.get("rationale_zh") if isinstance(parsed.get("rationale_zh"), str) else "",
            "message_zh": parsed.get("message_zh") if isinstance(parsed.get("message_zh"), str) else "",
            "alignment_mode": self.last_memory_selection_mode or "vlm_with_semantic_memory",
        }

    def _fallback_guidance(
        self,
        *,
        current_room_id: str,
        next_room_id: str,
        active_target_room_id: str,
        route: list[str],
        candidates: list[tuple[str, Path]],
    ) -> dict:
        label = candidates[0][0]
        direction = self._transition_context(current_room_id, next_room_id).get("allocentric_direction")
        direction_hint = self._direction_hint_zh(direction)
        return {
            "chosen_passage_label": label,
            "target_room_id": next_room_id,
            "direction_hint": direction_hint,
            "confidence": 0.35,
            "evidence": ["目前沒有可用的 VLM 設定，因此使用 room graph 方向與第一個候選通道作為保守建議。"],
            "rationale_zh": f"route={route}，active target={active_target_room_id}，下一個房間={next_room_id}。",
            "message_zh": f"我目前只能根據地圖方向保守判斷。請往 {label} 這個通道前進，方向提示：{direction_hint}。",
            "alignment_mode": "room_graph_fallback",
        }

    @staticmethod
    def _direction_hint_zh(direction: object) -> str:
        mapping = {
            "north": "往北側通道走",
            "east": "往東側通道走",
            "south": "往南側通道走",
            "west": "往西側通道走",
        }
        return mapping.get(direction, "往最符合下一個展廳線索的通道走")

    @staticmethod
    def _image_to_data_url(image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        if not mime_type:
            mime_type = "image/png"
        return f"data:{mime_type};base64,{b64encode(image_path.read_bytes()).decode('ascii')}"

    @staticmethod
    def _clone_json(payload: dict | None) -> dict | None:
        if payload is None:
            return None
        return json.loads(json.dumps(payload))
