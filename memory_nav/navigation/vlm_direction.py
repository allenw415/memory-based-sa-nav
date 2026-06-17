from __future__ import annotations

import json
import mimetypes
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from memory_nav.common.model_client import ModelResponseClient, parse_json_output

from .image_goal import VisualView, angular_distance_deg


class DirectionSelector(Protocol):
    def choose(
        self,
        *,
        goal_image_path: str | Path,
        views: Sequence[VisualView],
    ) -> dict: ...


class EightViewVLMDirectionSelector:
    """Choose one of eight current panorama views without exposing graph metadata."""

    def __init__(
        self,
        *,
        model_client: ModelResponseClient,
        model: str,
        detail: str = "high",
    ):
        self.model_client = model_client
        self.model = model
        self.detail = detail

    def choose(
        self,
        *,
        goal_image_path: str | Path,
        views: Sequence[VisualView],
    ) -> dict:
        ordered_views = _ordered_eight_views(views)
        request = self.build_request(
            goal_image_path=goal_image_path,
            views=ordered_views,
        )
        parsed = parse_json_output(self.model_client.create(request))
        _validate_direction_choice(parsed, ordered_views)
        parsed["selector_source"] = "live_vlm"
        parsed["request_summary"] = {
            "view_labels": _view_labels(ordered_views),
            "goal_image_path": str(goal_image_path),
        }
        return parsed

    def build_request(
        self,
        *,
        goal_image_path: str | Path,
        views: Sequence[VisualView],
    ) -> dict:
        ordered_views = _ordered_eight_views(views)
        labels = _view_labels(ordered_views)
        content: list[dict] = [
            {
                "type": "input_text",
                "text": "\n".join(
                    [
                        "The first image is a target passage that the user must walk toward.",
                        f"The following eight current views are labeled: {labels}.",
                        "Choose the current view whose direction most likely makes progress toward the target passage.",
                        "Reason about walkable geometry, corridor continuation, doorway position, and scene layout.",
                        "Do not simply choose the image with the most similar colors, objects, or architectural style.",
                        "Return exactly one current-view label.",
                    ]
                ),
            },
            {
                "type": "input_text",
                "text": "Target passage image:",
            },
            {
                "type": "input_image",
                "image_url": _image_to_data_url(Path(goal_image_path)),
                "detail": self.detail,
            },
        ]
        for label, view in zip(labels, ordered_views, strict=True):
            if not view.path:
                raise ValueError(f"Current view {label} has no image path.")
            content.extend(
                [
                    {"type": "input_text", "text": f"Current view {label}:"},
                    {
                        "type": "input_image",
                        "image_url": _image_to_data_url(Path(view.path)),
                        "detail": self.detail,
                    },
                ]
            )
        return {
            "model": self.model,
            "instructions": (
                "You are a visual navigation controller. Choose a direction from current images. "
                "You are not doing image retrieval. Return strict JSON only."
            ),
            "input": [{"role": "user", "content": content}],
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "eight_view_direction_choice",
                    "strict": True,
                    "schema": _direction_schema(labels),
                }
            },
        }


class RecordedDirectionSelector:
    """Replay direction responses in call order without invoking an API."""

    def __init__(self, responses: Sequence[dict | str]):
        self.responses = list(responses)
        self.call_index = 0

    @classmethod
    def from_path(cls, path: str | Path) -> "RecordedDirectionSelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("responses")
        if not isinstance(payload, list):
            raise ValueError("Recorded direction responses must contain a responses array.")
        return cls(payload)

    def choose(
        self,
        *,
        goal_image_path: str | Path,
        views: Sequence[VisualView],
    ) -> dict:
        del goal_image_path
        ordered_views = _ordered_eight_views(views)
        if self.call_index >= len(self.responses):
            raise KeyError(f"No recorded direction response for call {self.call_index}.")
        response = self.responses[self.call_index]
        self.call_index += 1
        if isinstance(response, str):
            response = {
                "chosen_view": response,
                "confidence": 1.0,
                "reason": "recorded response",
            }
        if not isinstance(response, dict):
            raise ValueError("Each recorded direction response must be an object or view label.")
        parsed = dict(response)
        parsed.setdefault("confidence", 1.0)
        parsed.setdefault("reason", "recorded response")
        _validate_direction_choice(parsed, ordered_views)
        parsed["selector_source"] = "recorded"
        parsed["call_index"] = self.call_index - 1
        return parsed


@dataclass(frozen=True)
class DirectionBurstResult:
    actions: list[dict] = field(default_factory=list)
    stop_reason: str = "max_burst_steps"
    current_pano_id: str = ""
    previous_pano_id: str | None = None
    direction_decision: dict | None = None


class SparseVLMDirectionSimulator:
    """Use one VLM view choice, then auto-follow unambiguous graph edges."""

    def __init__(
        self,
        *,
        pano_graph: dict[str, dict],
        pano_room_mappings: dict[str, str | None],
        observation_provider: Callable[[str], Sequence[VisualView]],
        direction_selector: DirectionSelector,
        confidence_threshold: float = 0.5,
        burst_steps: int = 3,
        max_turn_deg: float = 45.0,
    ):
        self.pano_graph = pano_graph
        self.pano_room_mappings = pano_room_mappings
        self.observation_provider = observation_provider
        self.direction_selector = direction_selector
        self.confidence_threshold = float(confidence_threshold)
        self.burst_steps = max(int(burst_steps), 1)
        self.max_turn_deg = float(max_turn_deg)

    def run_burst(
        self,
        *,
        current_pano_id: str,
        previous_pano_id: str | None,
        goal_image_path: str | Path,
        step_index: int,
        visited_pano_ids: set[str],
        max_steps: int | None = None,
    ) -> DirectionBurstResult:
        actions: list[dict] = []
        start_room_id = self.pano_room_mappings.get(current_pano_id)
        direction_decision: dict | None = None
        command_heading: float | None = None
        has_vlm_command = False
        limit = min(self.burst_steps, max(int(max_steps), 0)) if max_steps is not None else self.burst_steps

        for burst_step_index in range(limit):
            legal_edges, anti_backtrack_applied = self._legal_edges(
                current_pano_id=current_pano_id,
                previous_pano_id=previous_pano_id,
            )
            if not legal_edges:
                return DirectionBurstResult(
                    actions=actions,
                    stop_reason="no_legal_actions",
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    direction_decision=direction_decision,
                )

            views = list(self.observation_provider(current_pano_id))
            if len(views) != 8:
                raise RuntimeError(f"Expected 8 current views, got {len(views)}.")

            heading_change = 0.0
            if burst_step_index == 0 and len(legal_edges) > 1:
                direction_decision = self.direction_selector.choose(
                    goal_image_path=goal_image_path,
                    views=views,
                )
                confidence = float(direction_decision.get("confidence") or 0.0)
                if confidence < self.confidence_threshold:
                    return DirectionBurstResult(
                        actions=actions,
                        stop_reason="direction_confidence_below_threshold",
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                selected_view = _view_from_choice(direction_decision, views)
                selected_edge_index, selected_edge = _nearest_edge(
                    selected_view.heading,
                    legal_edges,
                )
                command_heading = float(selected_edge["geocentric_heading_deg"])
                has_vlm_command = True
                decision_source = "vlm_decision"
            elif burst_step_index == 0:
                selected_edge_index, selected_edge = 0, legal_edges[0]
                selected_view = None
                command_heading = float(selected_edge["geocentric_heading_deg"])
                decision_source = "auto_follow"
            else:
                if len(legal_edges) > 1 and not has_vlm_command:
                    return DirectionBurstResult(
                        actions=actions,
                        stop_reason="branching_point",
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                if len(legal_edges) > 1:
                    selected_edge_index, selected_edge = _nearest_edge(
                        float(command_heading),
                        legal_edges,
                    )
                else:
                    selected_edge_index, selected_edge = 0, legal_edges[0]
                selected_view = None
                edge_heading = float(selected_edge["geocentric_heading_deg"])
                if str(selected_edge["target_pano_id"]) in visited_pano_ids:
                    return DirectionBurstResult(
                        actions=actions,
                        stop_reason="cycle_detected",
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                heading_change = (
                    angular_distance_deg(command_heading, edge_heading)
                    if command_heading is not None
                    else 0.0
                )
                allowed_turn = (
                    min(self.max_turn_deg, 20.0)
                    if len(legal_edges) > 1
                    else self.max_turn_deg
                )
                if heading_change > allowed_turn:
                    return DirectionBurstResult(
                        actions=actions,
                        stop_reason="turn_exceeds_threshold",
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                command_heading = edge_heading
                decision_source = "auto_follow"

            next_pano_id = str(selected_edge["target_pano_id"])
            if next_pano_id in visited_pano_ids:
                return DirectionBurstResult(
                    actions=actions,
                    stop_reason="cycle_detected",
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    direction_decision=direction_decision,
                )

            action_heading = float(selected_edge["geocentric_heading_deg"])
            action = {
                "step_index": int(step_index + len(actions)),
                "burst_step_index": burst_step_index,
                "decision_source": decision_source,
                "current_pano_id": current_pano_id,
                "current_room_id": self.pano_room_mappings.get(current_pano_id),
                "previous_pano_id": previous_pano_id,
                "anti_backtrack_applied": anti_backtrack_applied,
                "current_views": [_view_summary(index, view) for index, view in enumerate(views)],
                "direction_decision": dict(direction_decision) if direction_decision else None,
                "selected_view_label": (
                    f"V{sorted(views, key=lambda item: item.capture_index).index(selected_view) + 1}"
                    if selected_view is not None
                    else None
                ),
                "selected_capture_index": (
                    int(selected_view.capture_index) if selected_view is not None else None
                ),
                "selected_view_heading": (
                    float(selected_view.heading) if selected_view is not None else None
                ),
                "selected_action_index": selected_edge_index,
                "selected_action_heading": action_heading,
                "heading_difference": (
                    angular_distance_deg(selected_view.heading, action_heading)
                    if selected_view is not None
                    else heading_change
                ),
                "legal_actions": [
                    {
                        "action_index": index,
                        "target_pano_id": edge["target_pano_id"],
                        "heading": float(edge["geocentric_heading_deg"]),
                    }
                    for index, edge in enumerate(legal_edges)
                ],
                "next_pano_id": next_pano_id,
                "next_room_id": self.pano_room_mappings.get(next_pano_id),
            }
            actions.append(action)
            visited_pano_ids.add(next_pano_id)
            previous_pano_id, current_pano_id = current_pano_id, next_pano_id

            if self.pano_room_mappings.get(current_pano_id) != start_room_id:
                return DirectionBurstResult(
                    actions=actions,
                    stop_reason="room_transition",
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    direction_decision=direction_decision,
                )

        return DirectionBurstResult(
            actions=actions,
            stop_reason="max_burst_steps",
            current_pano_id=current_pano_id,
            previous_pano_id=previous_pano_id,
            direction_decision=direction_decision,
        )

    def _legal_edges(
        self,
        *,
        current_pano_id: str,
        previous_pano_id: str | None,
    ) -> tuple[list[dict], bool]:
        raw_edges = [
            edge
            for edge in self.pano_graph.get(current_pano_id, {}).get("neighbors", [])
            if isinstance(edge, dict)
            and isinstance(edge.get("target_pano_id"), str)
            and isinstance(edge.get("geocentric_heading_deg"), (int, float))
            and edge["target_pano_id"] in self.pano_graph
        ]
        non_backtracking = [
            edge for edge in raw_edges if edge["target_pano_id"] != previous_pano_id
        ]
        anti_backtrack = bool(
            previous_pano_id
            and len(non_backtracking) < len(raw_edges)
            and non_backtracking
        )
        return (non_backtracking if anti_backtrack else raw_edges), anti_backtrack


def _ordered_eight_views(views: Sequence[VisualView]) -> list[VisualView]:
    ordered = sorted(views, key=lambda item: item.capture_index)
    if len(ordered) != 8:
        raise ValueError(f"Expected exactly 8 current views, got {len(ordered)}.")
    return ordered


def _view_labels(views: Sequence[VisualView]) -> list[str]:
    return [f"V{index}" for index in range(1, len(views) + 1)]


def _view_from_choice(choice: dict, views: Sequence[VisualView]) -> VisualView:
    ordered = _ordered_eight_views(views)
    chosen = choice.get("chosen_view")
    labels = _view_labels(ordered)
    if chosen not in labels:
        raise RuntimeError(f"Direction selector chose {chosen!r}; expected one of {labels}.")
    return ordered[labels.index(chosen)]


def _nearest_edge(view_heading: float, edges: Sequence[dict]) -> tuple[int, dict]:
    return min(
        enumerate(edges),
        key=lambda item: (
            angular_distance_deg(view_heading, float(item[1]["geocentric_heading_deg"])),
            item[0],
        ),
    )


def _validate_direction_choice(parsed: dict, views: Sequence[VisualView]) -> None:
    labels = _view_labels(_ordered_eight_views(views))
    if parsed.get("chosen_view") not in labels:
        raise RuntimeError(
            f"Direction selector chose {parsed.get('chosen_view')!r}; expected one of {labels}."
        )
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise RuntimeError("Direction selector confidence must be between 0 and 1.")
    if not isinstance(parsed.get("reason"), str):
        raise RuntimeError("Direction selector response must include a reason.")


def _direction_schema(labels: list[str]) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "chosen_view": {"type": "string", "enum": labels},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
        "required": ["chosen_view", "confidence", "reason"],
    }


def _view_summary(index: int, view: VisualView) -> dict:
    return {
        "view_label": f"V{index + 1}",
        "capture_index": int(view.capture_index),
        "capture_label": str(view.label),
        "heading": float(view.heading),
        "path": view.path,
    }


def _image_to_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return f"data:{mime_type or 'image/png'};base64,{b64encode(path.read_bytes()).decode('ascii')}"
