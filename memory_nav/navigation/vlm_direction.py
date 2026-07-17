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


class MemoryTreeDirectionSelector:
    """Choose an eight-view direction with bidirection passage memory alignment."""

    def __init__(
        self,
        *,
        metadata_items: Sequence[dict],
        image_embedder=None,
        render_root: str | Path | None = None,
        branching_factor: int = 3,
        max_depth: int = 5,
        similarity_backend: str = "dreamsim",
        dreamsim_type: str = "ensemble",
        bridge_selection_mode: str = "weighted",
        exclude_same_bridge_item: bool = True,
        bridge_similarity_tie_margin: float = 0.01,
        near_duplicate_threshold: float = 0.82,
        dinov2_patch_model: str = "facebook/dinov2-base",
        dinov2_patch_top_k: int = 5,
        dinov2_patch_max_patches: int = 24,
        patch_cache_dir: str | Path = "outputs/navigation_memory_tree_cache",
        device: str = "auto",
        batch_size: int = 32,
    ):
        self.similarity_backend = str(similarity_backend)
        if self.similarity_backend != "dinov2_patch_topk" and not hasattr(image_embedder, "encode_image_paths"):
            raise RuntimeError("Memory tree direction requires an image-path embedder.")
        self.image_embedder = image_embedder
        self.render_root = Path(render_root).resolve() if render_root is not None else None
        self.branching_factor = max(int(branching_factor), 1)
        self.max_depth = max(int(max_depth), 0)
        self.dreamsim_type = str(dreamsim_type)
        self.bridge_selection_mode = str(bridge_selection_mode)
        self.exclude_same_bridge_item = bool(exclude_same_bridge_item)
        self.bridge_similarity_tie_margin = max(float(bridge_similarity_tie_margin), 0.0)
        self.bridge_score_mode = (
            "bidirection" if self.bridge_selection_mode == "weighted" else self.bridge_selection_mode
        )
        self.near_duplicate_threshold = float(near_duplicate_threshold)
        self.dinov2_patch_model = str(dinov2_patch_model)
        self.dinov2_patch_top_k = max(int(dinov2_patch_top_k), 1)
        self.dinov2_patch_max_patches = max(int(dinov2_patch_max_patches), 1)
        self.patch_cache_dir = Path(patch_cache_dir)
        self.device = str(device)
        self.batch_size = max(int(batch_size), 1)
        self.metadata_items = self._memory_items(metadata_items)

    def choose(
        self,
        *,
        goal_image_path: str | Path,
        views: Sequence[VisualView],
    ) -> dict:
        ordered_views = _ordered_eight_views(views)
        labels = _view_labels(ordered_views)
        goal_path = _resolve_existing_path(goal_image_path, "Goal image")
        for index, view in enumerate(ordered_views):
            if view.path and _same_path(goal_path, Path(view.path)):
                return {
                    "chosen_view": labels[index],
                    "confidence": 1.0,
                    "reason": "The selected passage source image is one of the current views.",
                    "selector_source": "memory_tree",
                    "request_summary": {
                        "goal_image_path": str(goal_path),
                        "view_labels": labels,
                        "branching_factor": self.branching_factor,
                        "max_depth": self.max_depth,
                        "similarity_backend": self.similarity_backend,
                        "bridge_score_mode": self.bridge_score_mode,
                        "bridge_selection_mode": self.bridge_selection_mode,
                        "exclude_same_bridge_item": self.exclude_same_bridge_item,
                        "bridge_similarity_tie_margin": self.bridge_similarity_tie_margin,
                        "near_duplicate_threshold": self.near_duplicate_threshold,
                        "dinov2_patch_model": (
                            self.dinov2_patch_model if self.similarity_backend == "dinov2_patch_topk" else None
                        ),
                        "dinov2_patch_top_k": (
                            self.dinov2_patch_top_k if self.similarity_backend == "dinov2_patch_topk" else None
                        ),
                        "dinov2_patch_max_patches": (
                            self.dinov2_patch_max_patches if self.similarity_backend == "dinov2_patch_topk" else None
                        ),
                    },
                    "memory_tree": {
                        "mode": "direct_current_view_match",
                        "selected_view": _view_summary(index, view),
                    },
                }

        room_items = self._room_items_for_goal(goal_path, ordered_views)
        target_index = _find_matching_image_path(room_items, goal_path)
        if target_index is None:
            room_id = _first_room_id(room_items) or "external_room"
            target_index = len(room_items)
            room_items.append(
                _external_memory_item(
                    goal_path,
                    room_id=room_id,
                    pano_id="external_target",
                    capture_index=0,
                )
            )
        current_view_indices = []
        for view in ordered_views:
            view_path = _resolve_existing_path(view.path, f"Current view {view.capture_index}")
            view_index = _find_matching_image_path(room_items, view_path)
            if view_index is None:
                view_index = len(room_items)
                room_items.append(
                    _external_memory_item(
                        view_path,
                        room_id=str(room_items[target_index].get("room_id") or "external_room"),
                        pano_id=f"external_current_{view.capture_index}",
                        capture_index=int(view.capture_index),
                        capture_label=view.label,
                        capture_heading=float(view.heading),
                    )
                )
            current_view_indices.append(view_index)

        import numpy as np
        from tools.experiments.build_passage_memory_tree import (
            build_bidirectional_alignment,
            load_or_compute_dinov2_patch_similarity_matrix,
            load_or_encode_dinov2_patch_features,
            memory_key,
        )

        image_paths = [Path(str(item["image_path"])) for item in room_items]
        pairwise_similarities = None
        if self.similarity_backend == "dinov2_patch_topk":
            patch_features = load_or_encode_dinov2_patch_features(
                image_paths=image_paths,
                memory_keys=[memory_key(item) for item in room_items],
                output_dir=self.patch_cache_dir,
                model_name=self.dinov2_patch_model,
                max_patches=self.dinov2_patch_max_patches,
                device=self.device,
                batch_size=self.batch_size,
            )
            pairwise_similarities = load_or_compute_dinov2_patch_similarity_matrix(
                patch_features=patch_features,
                image_paths=image_paths,
                memory_keys=[memory_key(item) for item in room_items],
                output_dir=self.patch_cache_dir,
                model_name=self.dinov2_patch_model,
                max_patches=self.dinov2_patch_max_patches,
                top_k=self.dinov2_patch_top_k,
            )
            embeddings = np.zeros((len(room_items), 1), dtype=np.float32)
        else:
            embeddings = np.asarray(self.image_embedder.encode_image_paths(image_paths), dtype=np.float32)
            if embeddings.shape[0] != len(room_items):
                raise RuntimeError("Memory tree embedder returned the wrong number of embeddings.")
        result = build_bidirectional_alignment(
            room_items=room_items,
            room_embeddings=embeddings,
            target_index=int(target_index),
            target_passage_label=Path(goal_path).stem,
            current_view_indices=current_view_indices,
            current_context={
                "mode": "pano_8_views",
                "view_count": len(ordered_views),
                "views": [_view_summary(index, view) for index, view in enumerate(ordered_views)],
            },
            branching_factor=self.branching_factor,
            max_depth=self.max_depth,
            near_duplicate_threshold=self.near_duplicate_threshold,
            bridge_selection_mode=self.bridge_selection_mode,
            exclude_same_bridge_item=self.exclude_same_bridge_item,
            bridge_similarity_tie_margin=self.bridge_similarity_tie_margin,
            tree_expansion_score_mode="path_continuity",
            include_embeddings=False,
            similarity_backend=self.similarity_backend,
            dreamsim_type=self.dreamsim_type,
            pairwise_similarities=pairwise_similarities,
            dinov2_patch_model=self.dinov2_patch_model,
            dinov2_patch_top_k=self.dinov2_patch_top_k,
            dinov2_patch_max_patches=self.dinov2_patch_max_patches,
        )
        selected_view = result["selected_view"]
        view_order = int(selected_view.get("view_order") or 0)
        if view_order < 0 or view_order >= len(labels):
            raise RuntimeError(f"Memory tree selected invalid view order: {view_order}")
        selected_alignment = result["selected_alignment"]
        selection_reason = str(result.get("selection_reason") or "best_bridge_score")
        if selection_reason == "root_target_direct_match":
            reason = "Selected because a current view directly matches the target passage source image."
        elif selection_reason == "best_bridge_score":
            reason = "Selected by bidirection passage-memory alignment with the strongest continuity bridge."
        else:
            reason = "Selected by bidirection passage-memory alignment."
        return {
            "chosen_view": labels[view_order],
            "confidence": 1.0,
            "reason": reason,
            "selector_source": "memory_tree",
            "request_summary": {
                "goal_image_path": str(goal_path),
                "view_labels": labels,
                "branching_factor": self.branching_factor,
                "max_depth": self.max_depth,
                "similarity_backend": self.similarity_backend,
                "dreamsim_type": self.dreamsim_type if self.similarity_backend == "dreamsim" else None,
                "bridge_score_mode": self.bridge_score_mode,
                "bridge_selection_mode": self.bridge_selection_mode,
                "exclude_same_bridge_item": self.exclude_same_bridge_item,
                "bridge_similarity_tie_margin": self.bridge_similarity_tie_margin,
                "near_duplicate_threshold": self.near_duplicate_threshold,
                "dinov2_patch_model": (
                    self.dinov2_patch_model if self.similarity_backend == "dinov2_patch_topk" else None
                ),
                "dinov2_patch_top_k": (
                    self.dinov2_patch_top_k if self.similarity_backend == "dinov2_patch_topk" else None
                ),
                "dinov2_patch_max_patches": (
                    self.dinov2_patch_max_patches if self.similarity_backend == "dinov2_patch_topk" else None
                ),
            },
            "memory_tree": {
                "mode": "bidirection_passage_alignment",
                "method": result["method"],
                "selected_view": selected_view,
                "selection_reason": selection_reason,
                "best_bridge": _compact_bridge(selected_alignment.get("best_bridge", {})),
                "current_chain_root_to_bridge": _compact_chain(
                    selected_alignment.get("current_chain_root_to_bridge", [])
                ),
                "passage_chain_bridge_to_target": _compact_chain(
                    selected_alignment.get("passage_chain_bridge_to_target", [])
                ),
                "view_scores": [
                    {
                        "rank": int(item["rank"]),
                        "view_label": labels[int(item["view_order"])],
                        "capture_index": int(item["capture_index"]),
                        "selected": bool(item["selected"]),
                        "view_score": float(item.get("view_score", 0.0)),
                        "root_target_similarity": float(item.get("root_target_similarity", 0.0)),
                        "root_target_direct_match": bool(item.get("root_target_direct_match", False)),
                        "bridge_similarity": float(
                            item.get("best_bridge", {}).get("bridge_similarity", 0.0)
                        ),
                        "continuity_score": float(
                            item.get("best_bridge", {}).get("continuity_score", 0.0)
                        ),
                        "chain_mean_continuity": float(
                            item.get("best_bridge", {}).get("chain_mean_continuity", 0.0)
                        ),
                        "chain_bottleneck_similarity": float(
                            item.get("best_bridge", {}).get("chain_bottleneck_similarity", 0.0)
                        ),
                        "total_bridge_depth": int(
                            item.get("best_bridge", {}).get("total_bridge_depth", 0)
                        ),
                    }
                    for item in result["view_alignments"]
                ],
            },
        }

    def _memory_items(self, metadata_items: Sequence[dict]) -> list[dict]:
        items = []
        for index, item in enumerate(metadata_items):
            image_path = self._resolve_metadata_image_path(item)
            pano_id = item.get("pano_id")
            capture_index = item.get("capture_index")
            if image_path is None or not isinstance(pano_id, str) or not isinstance(capture_index, int):
                continue
            items.append(
                {
                    "memory_index": int(item.get("memory_index", index)),
                    "room_id": item.get("room_id"),
                    "pano_id": pano_id,
                    "capture_index": capture_index,
                    "capture_label": item.get("capture_label"),
                    "capture_heading": item.get("capture_heading"),
                    "image_path": str(image_path),
                }
            )
        return items

    def _resolve_metadata_image_path(self, item: dict) -> Path | None:
        raw_path = item.get("image_path") or item.get("capture_path")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if path.exists():
                return path.resolve()
            if self.render_root is not None:
                pano_id = item.get("pano_id")
                if isinstance(pano_id, str):
                    candidate = self.render_root / pano_id / path.name
                    if candidate.exists():
                        return candidate.resolve()
        return None

    def _room_items_for_goal(self, goal_path: Path, views: Sequence[VisualView]) -> list[dict]:
        target_index = _find_matching_image_path(self.metadata_items, goal_path)
        room_id = self.metadata_items[target_index].get("room_id") if target_index is not None else None
        if room_id is None:
            for view in views:
                if not view.path:
                    continue
                view_index = _find_matching_image_path(self.metadata_items, Path(view.path))
                if view_index is not None:
                    room_id = self.metadata_items[view_index].get("room_id")
                    break
        if room_id is None:
            return []
        return [dict(item) for item in self.metadata_items if item.get("room_id") == room_id]


@dataclass(frozen=True)
class DirectionBurstResult:
    actions: list[dict] = field(default_factory=list)
    stop_reason: str = "max_burst_steps"
    current_pano_id: str = ""
    previous_pano_id: str | None = None
    direction_decision: dict | None = None


@dataclass
class DirectionBranchState:
    pano_id: str
    path_index: int
    incoming_heading: float | None
    selected_target_pano_id: str
    candidate_target_pano_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pano_id": self.pano_id,
            "path_index": self.path_index,
            "incoming_heading": self.incoming_heading,
            "selected_target_pano_id": self.selected_target_pano_id,
            "candidate_target_pano_ids": list(self.candidate_target_pano_ids),
        }


@dataclass
class DirectionCommitmentState:
    """Per-room state for cross-burst direction continuity and bounded recovery."""

    mode: str = "off"
    switch_margin: float = 0.03
    recovery_budget: int = 1
    last_action_heading: float | None = None
    room_path: list[str] = field(default_factory=list)
    branches: list[DirectionBranchState] = field(default_factory=list)
    blocked_edges: set[tuple[str, str]] = field(default_factory=set)
    recovery_events_used: int = 0
    recovery_target_pano_id: str | None = None
    recovery_branch_heading: float | None = None
    recovery_history: list[dict] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return self.mode == "visual_hysteresis"

    @property
    def recovery_in_progress(self) -> bool:
        return self.recovery_target_pano_id is not None

    def summary_dict(self) -> dict:
        return {
            "mode": self.mode,
            "switch_margin": self.switch_margin,
            "recovery_budget": self.recovery_budget,
            "last_action_heading": self.last_action_heading,
            "branch_count": len(self.branches),
            "blocked_edge_count": len(self.blocked_edges),
            "recovery_events_used": self.recovery_events_used,
            "recovery_in_progress": self.recovery_in_progress,
            "recovery_target_pano_id": self.recovery_target_pano_id,
        }

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "switch_margin": self.switch_margin,
            "recovery_budget": self.recovery_budget,
            "last_action_heading": self.last_action_heading,
            "room_path": list(self.room_path),
            "branches": [item.to_dict() for item in self.branches],
            "blocked_edges": [
                {"source_pano_id": source, "target_pano_id": target}
                for source, target in sorted(self.blocked_edges)
            ],
            "recovery_events_used": self.recovery_events_used,
            "recovery_in_progress": self.recovery_in_progress,
            "recovery_target_pano_id": self.recovery_target_pano_id,
            "recovery_history": [dict(item) for item in self.recovery_history],
        }


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
        commitment_mode: str = "off",
        switch_margin: float = 0.03,
        recovery_budget: int = 1,
    ):
        self.pano_graph = pano_graph
        self.pano_room_mappings = pano_room_mappings
        self.observation_provider = observation_provider
        self.direction_selector = direction_selector
        self.confidence_threshold = float(confidence_threshold)
        self.burst_steps = max(int(burst_steps), 1)
        self.max_turn_deg = float(max_turn_deg)
        if commitment_mode not in {"off", "visual_hysteresis"}:
            raise ValueError(f"Unsupported direction commitment mode: {commitment_mode}")
        self.commitment_mode = str(commitment_mode)
        self.switch_margin = max(float(switch_margin), 0.0)
        self.recovery_budget = max(int(recovery_budget), 0)

    def create_commitment_state(self, *, current_pano_id: str) -> DirectionCommitmentState:
        return DirectionCommitmentState(
            mode=self.commitment_mode,
            switch_margin=self.switch_margin,
            recovery_budget=self.recovery_budget,
            room_path=[current_pano_id],
        )

    def run_burst(
        self,
        *,
        current_pano_id: str,
        previous_pano_id: str | None,
        goal_image_path: str | Path,
        step_index: int,
        visited_pano_ids: set[str],
        max_steps: int | None = None,
        commitment_state: DirectionCommitmentState | None = None,
    ) -> DirectionBurstResult:
        actions: list[dict] = []
        start_room_id = _normalized_room_id(self.pano_room_mappings.get(current_pano_id))
        direction_decision: dict | None = None
        state = commitment_state
        if state is None and self.commitment_mode == "visual_hysteresis":
            state = self.create_commitment_state(current_pano_id=current_pano_id)
        if state is not None and state.enabled:
            if not state.room_path:
                state.room_path.append(current_pano_id)
            if state.room_path[-1] != current_pano_id:
                raise RuntimeError(
                    "Direction commitment state does not match current panorama: "
                    f"{state.room_path[-1]} != {current_pano_id}"
                )
        if state is not None and state.enabled and state.last_action_heading is not None:
            command_heading: float | None = state.last_action_heading
        elif start_room_id is None:
            command_heading = _incoming_edge_heading(
                self.pano_graph,
                previous_pano_id=previous_pano_id,
                current_pano_id=current_pano_id,
            )
        else:
            command_heading = None
        has_vlm_command = command_heading is not None
        limit = (
            min(self.burst_steps, max(int(max_steps), 0))
            if max_steps is not None
            else self.burst_steps
        )

        while len(actions) < limit:
            burst_step_index = len(actions)
            if state is not None and state.enabled and state.recovery_in_progress:
                recovery_action = self._recovery_action(
                    state=state,
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    step_index=step_index + len(actions),
                    burst_step_index=burst_step_index,
                )
                if recovery_action is None:
                    return DirectionBurstResult(
                        actions=actions,
                        stop_reason="cycle_detected",
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                actions.append(recovery_action)
                next_pano_id = str(recovery_action["next_pano_id"])
                visited_pano_ids.add(next_pano_id)
                previous_pano_id, current_pano_id = current_pano_id, next_pano_id
                command_heading = state.last_action_heading
                has_vlm_command = command_heading is not None
                continue

            blocked_edges = state.blocked_edges if state is not None and state.enabled else None
            legal_edges, anti_backtrack_applied = self._legal_edges(
                current_pano_id=current_pano_id,
                previous_pano_id=previous_pano_id,
                blocked_edges=blocked_edges,
            )
            if not legal_edges:
                if self._begin_recovery(
                    state=state,
                    current_pano_id=current_pano_id,
                    visited_pano_ids=visited_pano_ids,
                ):
                    continue
                return DirectionBurstResult(
                    actions=actions,
                    stop_reason=(
                        "cycle_detected"
                        if state is not None and state.enabled
                        else "no_legal_actions"
                    ),
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    direction_decision=direction_decision,
                )

            current_room_id = _normalized_room_id(self.pano_room_mappings.get(current_pano_id))
            passthrough_null_connector = current_room_id is None and command_heading is not None
            views: list[VisualView] = []
            commitment_payload: dict | None = None
            if not passthrough_null_connector:
                views = list(self.observation_provider(current_pano_id))
                if len(views) != 8:
                    raise RuntimeError(f"Expected 8 current views, got {len(views)}.")

            heading_change = 0.0
            if passthrough_null_connector:
                selected_edge_index, selected_edge = _nearest_edge(
                    float(command_heading),
                    legal_edges,
                )
                selected_view = None
                edge_heading = float(selected_edge["geocentric_heading_deg"])
                heading_change = angular_distance_deg(float(command_heading), edge_heading)
                command_heading = edge_heading
                decision_source = "null_connector_follow"
            elif burst_step_index == 0 and len(legal_edges) > 1:
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
                (
                    selected_view,
                    selected_edge_index,
                    selected_edge,
                    cycle_override,
                ) = self._cycle_safe_view_edge(
                    direction_decision=direction_decision,
                    views=views,
                    legal_edges=legal_edges,
                    current_pano_id=current_pano_id,
                    visited_pano_ids=visited_pano_ids,
                    avoid_dead_end_traps=True,
                    blocked_edges=blocked_edges,
                )
                if selected_view is None or selected_edge is None:
                    if self._begin_recovery(
                        state=state,
                        current_pano_id=current_pano_id,
                        visited_pano_ids=visited_pano_ids,
                    ):
                        continue
                    return DirectionBurstResult(
                        actions=actions,
                        stop_reason="cycle_detected",
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                if cycle_override:
                    direction_decision = self._with_cycle_override(
                        direction_decision,
                        cycle_override,
                    )
                if state is not None and state.enabled:
                    (
                        direction_decision,
                        selected_view,
                        selected_edge_index,
                        selected_edge,
                        commitment_payload,
                    ) = self._apply_visual_hysteresis(
                        state=state,
                        direction_decision=direction_decision,
                        selected_view=selected_view,
                        selected_edge_index=selected_edge_index,
                        selected_edge=selected_edge,
                        views=views,
                        legal_edges=legal_edges,
                        current_pano_id=current_pano_id,
                        visited_pano_ids=visited_pano_ids,
                    )
                command_heading = float(selected_view.heading)
                has_vlm_command = True
                decision_source = (
                    "memory_tree_decision"
                    if direction_decision.get("selector_source") == "memory_tree"
                    else "vlm_decision"
                )
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
                    if self._begin_recovery(
                        state=state,
                        current_pano_id=current_pano_id,
                        visited_pano_ids=visited_pano_ids,
                    ):
                        continue
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
                        stop_reason=(
                            "replan_required"
                            if state is not None and state.enabled
                            else "turn_exceeds_threshold"
                        ),
                        current_pano_id=current_pano_id,
                        previous_pano_id=previous_pano_id,
                        direction_decision=direction_decision,
                    )
                command_heading = edge_heading
                decision_source = "auto_follow"

            next_pano_id = str(selected_edge["target_pano_id"])
            if next_pano_id in visited_pano_ids:
                if self._begin_recovery(
                    state=state,
                    current_pano_id=current_pano_id,
                    visited_pano_ids=visited_pano_ids,
                ):
                    continue
                return DirectionBurstResult(
                    actions=actions,
                    stop_reason="cycle_detected",
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    direction_decision=direction_decision,
                )

            if state is not None and state.enabled and len(legal_edges) > 1:
                self._record_branch(
                    state=state,
                    current_pano_id=current_pano_id,
                    incoming_heading=state.last_action_heading,
                    selected_target_pano_id=next_pano_id,
                    legal_edges=legal_edges,
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
                "direction_commitment": dict(commitment_payload) if commitment_payload else None,
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
            if state is not None and state.enabled:
                state.room_path.append(next_pano_id)
                state.last_action_heading = action_heading
            previous_pano_id, current_pano_id = current_pano_id, next_pano_id

            current_room_after = _normalized_room_id(self.pano_room_mappings.get(current_pano_id))
            if current_room_after is not None and current_room_after != start_room_id:
                return DirectionBurstResult(
                    actions=actions,
                    stop_reason="room_transition",
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    direction_decision=direction_decision,
                )

        stop_reason = (
            "recovery_in_progress"
            if state is not None and state.enabled and state.recovery_in_progress
            else "max_burst_steps"
        )
        return DirectionBurstResult(
            actions=actions,
            stop_reason=stop_reason,
            current_pano_id=current_pano_id,
            previous_pano_id=previous_pano_id,
            direction_decision=direction_decision,
        )

    @staticmethod
    def _with_cycle_override(direction_decision: dict, cycle_override: dict) -> dict:
        updated = dict(direction_decision)
        updated["chosen_view"] = cycle_override["selected_view"]
        updated["cycle_avoidance"] = cycle_override
        updated["reason"] = (
            f"{updated.get('reason', '')} "
            f"Cycle avoidance skipped {cycle_override['original_view']}."
        ).strip()
        if isinstance(updated.get("memory_tree"), dict):
            updated["memory_tree"] = dict(updated["memory_tree"])
            updated["memory_tree"]["cycle_avoidance"] = cycle_override
        return updated

    def _apply_visual_hysteresis(
        self,
        *,
        state: DirectionCommitmentState,
        direction_decision: dict,
        selected_view: VisualView,
        selected_edge_index: int,
        selected_edge: dict,
        views: Sequence[VisualView],
        legal_edges: Sequence[dict],
        current_pano_id: str,
        visited_pano_ids: set[str],
    ) -> tuple[dict, VisualView, int, dict, dict]:
        memory_tree = direction_decision.get("memory_tree")
        direct_match = bool(
            isinstance(memory_tree, dict)
            and memory_tree.get("mode") == "direct_current_view_match"
        )
        best_target = str(selected_edge["target_pano_id"])
        base_payload = {
            "mode": state.mode,
            "switch_margin": state.switch_margin,
            "best_target_pano_id": best_target,
        }
        if direct_match:
            payload = {**base_payload, "decision": "direct_match"}
            return self._with_commitment(direction_decision, payload), selected_view, selected_edge_index, selected_edge, payload
        if state.last_action_heading is None:
            payload = {**base_payload, "decision": "initialized"}
            return self._with_commitment(direction_decision, payload), selected_view, selected_edge_index, selected_edge, payload

        continuation_index, continuation_edge = _nearest_edge(
            float(state.last_action_heading),
            legal_edges,
        )
        continuation_target = str(continuation_edge["target_pano_id"])
        continuation_block_reason = self._cycle_block_reason(
            continuation_edge,
            current_pano_id=current_pano_id,
            visited_pano_ids=visited_pano_ids,
            avoid_dead_end_traps=True,
            follow_heading=float(state.last_action_heading),
            blocked_edges=state.blocked_edges,
        )
        scores = self._visual_scores_by_edge(
            direction_decision=direction_decision,
            views=views,
            legal_edges=legal_edges,
        )
        best_score = scores.get(best_target, {}).get("score")
        continuation_record = scores.get(continuation_target)
        continuation_score = (
            continuation_record.get("score") if continuation_record is not None else None
        )
        score_gap = (
            float(best_score) - float(continuation_score)
            if best_score is not None and continuation_score is not None
            else None
        )
        keep_continuation = bool(
            continuation_block_reason is None
            and score_gap is not None
            and score_gap <= state.switch_margin
        )
        if keep_continuation and continuation_record is not None:
            selected_view = continuation_record["view"]
            selected_edge_index = continuation_index
            selected_edge = continuation_edge
            selected_label = str(continuation_record["view_label"])
            payload = {
                **base_payload,
                "decision": "kept",
                "selected_target_pano_id": continuation_target,
                "original_selected_view": direction_decision.get("chosen_view"),
                "selected_view": selected_label,
                "best_root_target_similarity": best_score,
                "continuation_root_target_similarity": continuation_score,
                "score_gap": score_gap,
                "continuation_block_reason": None,
            }
            updated = dict(direction_decision)
            updated["chosen_view"] = selected_label
            updated["reason"] = (
                f"{updated.get('reason', '')} Direction commitment retained the local continuation."
            ).strip()
            return self._with_commitment(updated, payload), selected_view, selected_edge_index, selected_edge, payload

        payload = {
            **base_payload,
            "decision": "switched",
            "selected_target_pano_id": best_target,
            "selected_view": direction_decision.get("chosen_view"),
            "best_root_target_similarity": best_score,
            "continuation_target_pano_id": continuation_target,
            "continuation_root_target_similarity": continuation_score,
            "score_gap": score_gap,
            "continuation_block_reason": continuation_block_reason,
        }
        return self._with_commitment(direction_decision, payload), selected_view, selected_edge_index, selected_edge, payload

    @staticmethod
    def _with_commitment(direction_decision: dict, payload: dict) -> dict:
        updated = dict(direction_decision)
        updated["commitment"] = dict(payload)
        if isinstance(updated.get("memory_tree"), dict):
            updated["memory_tree"] = dict(updated["memory_tree"])
            updated["memory_tree"]["commitment"] = dict(payload)
        return updated

    @staticmethod
    def _visual_scores_by_edge(
        *,
        direction_decision: dict,
        views: Sequence[VisualView],
        legal_edges: Sequence[dict],
    ) -> dict[str, dict]:
        memory_tree = direction_decision.get("memory_tree")
        view_scores = memory_tree.get("view_scores") if isinstance(memory_tree, dict) else None
        if not isinstance(view_scores, list):
            return {}
        ordered_views = _ordered_eight_views(views)
        labels = _view_labels(ordered_views)
        result: dict[str, dict] = {}
        for item in view_scores:
            if not isinstance(item, dict):
                continue
            label = item.get("view_label")
            score = item.get("root_target_similarity")
            if not isinstance(label, str) or label not in labels:
                continue
            if not isinstance(score, (int, float)):
                continue
            view = ordered_views[labels.index(label)]
            _, edge = _nearest_edge(float(view.heading), legal_edges)
            target = str(edge["target_pano_id"])
            rank = int(item.get("rank", len(labels) + 1))
            previous = result.get(target)
            if previous is None or float(score) > float(previous["score"]) or (
                float(score) == float(previous["score"]) and rank < int(previous["rank"])
            ):
                result[target] = {
                    "score": float(score),
                    "rank": rank,
                    "view": view,
                    "view_label": label,
                }
        return result

    def _record_branch(
        self,
        *,
        state: DirectionCommitmentState,
        current_pano_id: str,
        incoming_heading: float | None,
        selected_target_pano_id: str,
        legal_edges: Sequence[dict],
    ) -> None:
        path_index = len(state.room_path) - 1
        state.branches = [item for item in state.branches if item.path_index < path_index]
        candidates: list[str] = []
        for edge in legal_edges:
            target = str(edge["target_pano_id"])
            if target not in candidates:
                candidates.append(target)
        state.branches.append(
            DirectionBranchState(
                pano_id=current_pano_id,
                path_index=path_index,
                incoming_heading=incoming_heading,
                selected_target_pano_id=selected_target_pano_id,
                candidate_target_pano_ids=candidates,
            )
        )

    def _begin_recovery(
        self,
        *,
        state: DirectionCommitmentState | None,
        current_pano_id: str,
        visited_pano_ids: set[str],
    ) -> bool:
        if state is None or not state.enabled or state.recovery_in_progress:
            return False
        if state.recovery_events_used >= state.recovery_budget:
            return False
        if not state.room_path or state.room_path[-1] != current_pano_id:
            return False
        for branch in reversed(state.branches):
            branch_index = branch.path_index
            if branch_index < 0 or branch_index >= len(state.room_path) - 1:
                continue
            if state.room_path[branch_index] != branch.pano_id:
                continue
            failed_target = state.room_path[branch_index + 1]
            available_targets: list[str] = []
            for target in branch.candidate_target_pano_ids:
                if target == failed_target or target in visited_pano_ids:
                    continue
                if (branch.pano_id, target) in state.blocked_edges:
                    continue
                edge = self._edge_to_target(branch.pano_id, target)
                if edge is None:
                    continue
                if self._cycle_block_reason(
                    edge,
                    current_pano_id=branch.pano_id,
                    visited_pano_ids=visited_pano_ids,
                    avoid_dead_end_traps=True,
                    blocked_edges=state.blocked_edges,
                ) is None:
                    available_targets.append(target)
            if not available_targets:
                continue
            blocked_edge = (branch.pano_id, failed_target)
            state.blocked_edges.add(blocked_edge)
            state.recovery_events_used += 1
            state.recovery_target_pano_id = branch.pano_id
            state.recovery_branch_heading = branch.incoming_heading
            state.branches = [
                item for item in state.branches if item.path_index < branch_index
            ]
            state.recovery_history.append(
                {
                    "recovery_event_index": state.recovery_events_used - 1,
                    "status": "in_progress",
                    "trigger_pano_id": current_pano_id,
                    "target_branch_pano_id": branch.pano_id,
                    "blocked_edge": {
                        "source_pano_id": blocked_edge[0],
                        "target_pano_id": blocked_edge[1],
                    },
                    "available_alternative_target_pano_ids": available_targets,
                    "planned_backtrack_steps": len(state.room_path) - 1 - branch_index,
                    "completed_backtrack_steps": 0,
                }
            )
            return True
        return False

    def _recovery_action(
        self,
        *,
        state: DirectionCommitmentState,
        current_pano_id: str,
        previous_pano_id: str | None,
        step_index: int,
        burst_step_index: int,
    ) -> dict | None:
        if not state.recovery_in_progress or len(state.room_path) < 2:
            return None
        if state.room_path[-1] != current_pano_id:
            return None
        next_pano_id = state.room_path[-2]
        raw_edges = self._raw_edges(current_pano_id)
        selected = next(
            (
                (index, edge)
                for index, edge in enumerate(raw_edges)
                if str(edge["target_pano_id"]) == next_pano_id
            ),
            None,
        )
        if selected is None:
            return None
        selected_edge_index, selected_edge = selected
        event = state.recovery_history[-1]
        action_heading = float(selected_edge["geocentric_heading_deg"])
        action = {
            "step_index": int(step_index),
            "burst_step_index": int(burst_step_index),
            "decision_source": "recovery_backtrack",
            "current_pano_id": current_pano_id,
            "current_room_id": self.pano_room_mappings.get(current_pano_id),
            "previous_pano_id": previous_pano_id,
            "anti_backtrack_applied": False,
            "current_views": [],
            "direction_decision": None,
            "direction_commitment": {
                "mode": state.mode,
                "decision": "recovery_backtrack",
                "recovery_event_index": event["recovery_event_index"],
                "target_branch_pano_id": state.recovery_target_pano_id,
                "blocked_edge": dict(event["blocked_edge"]),
            },
            "selected_view_label": None,
            "selected_capture_index": None,
            "selected_view_heading": None,
            "selected_action_index": selected_edge_index,
            "selected_action_heading": action_heading,
            "heading_difference": 0.0,
            "legal_actions": [
                {
                    "action_index": index,
                    "target_pano_id": edge["target_pano_id"],
                    "heading": float(edge["geocentric_heading_deg"]),
                }
                for index, edge in enumerate(raw_edges)
            ],
            "next_pano_id": next_pano_id,
            "next_room_id": self.pano_room_mappings.get(next_pano_id),
            "recovery_event_index": event["recovery_event_index"],
        }
        state.room_path.pop()
        event["completed_backtrack_steps"] = int(event["completed_backtrack_steps"]) + 1
        if next_pano_id == state.recovery_target_pano_id:
            event["status"] = "completed"
            state.recovery_target_pano_id = None
            state.last_action_heading = state.recovery_branch_heading
            state.recovery_branch_heading = None
        return action

    def _raw_edges(self, pano_id: str) -> list[dict]:
        return [
            edge
            for edge in self.pano_graph.get(pano_id, {}).get("neighbors", [])
            if isinstance(edge, dict)
            and isinstance(edge.get("target_pano_id"), str)
            and isinstance(edge.get("geocentric_heading_deg"), (int, float))
            and edge["target_pano_id"] in self.pano_graph
        ]

    def _edge_to_target(self, source_pano_id: str, target_pano_id: str) -> dict | None:
        return next(
            (
                edge
                for edge in self._raw_edges(source_pano_id)
                if str(edge["target_pano_id"]) == target_pano_id
            ),
            None,
        )

    def _legal_edges(
        self,
        *,
        current_pano_id: str,
        previous_pano_id: str | None,
        blocked_edges: set[tuple[str, str]] | None = None,
    ) -> tuple[list[dict], bool]:
        blocked = blocked_edges or set()
        raw_edges = [
            edge
            for edge in self._raw_edges(current_pano_id)
            if (current_pano_id, str(edge["target_pano_id"])) not in blocked
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

    def _cycle_safe_view_edge(
        self,
        *,
        direction_decision: dict,
        views: Sequence[VisualView],
        legal_edges: Sequence[dict],
        current_pano_id: str,
        visited_pano_ids: set[str],
        avoid_dead_end_traps: bool,
        blocked_edges: set[tuple[str, str]] | None = None,
    ) -> tuple[VisualView | None, int | None, dict | None, dict | None]:
        ordered_views = _ordered_eight_views(views)
        labels = _view_labels(ordered_views)
        chosen_view = str(direction_decision.get("chosen_view"))
        candidates = _ranked_choice_labels(direction_decision, labels)
        skipped: list[dict] = []
        for label in candidates:
            view = ordered_views[labels.index(label)]
            edge_index, edge = _nearest_edge(view.heading, legal_edges)
            block_reason = self._cycle_block_reason(
                edge,
                current_pano_id=current_pano_id,
                visited_pano_ids=visited_pano_ids,
                avoid_dead_end_traps=avoid_dead_end_traps,
                follow_heading=float(view.heading),
                blocked_edges=blocked_edges,
            )
            if block_reason:
                skipped.append(
                    {
                        "view": label,
                        "target_pano_id": str(edge["target_pano_id"]),
                        "reason": block_reason,
                    }
                )
                continue
            if label == chosen_view:
                return view, edge_index, edge, None
            return (
                view,
                edge_index,
                edge,
                {
                    "original_view": chosen_view,
                    "selected_view": label,
                    "selected_target_pano_id": str(edge["target_pano_id"]),
                    "skipped_candidates": skipped,
                },
            )
        return (
            None,
            None,
            None,
            {
                "original_view": chosen_view,
                "selected_view": None,
                "skipped_candidates": skipped,
            },
        )

    def _cycle_block_reason(
        self,
        edge: dict,
        *,
        current_pano_id: str,
        visited_pano_ids: set[str],
        avoid_dead_end_traps: bool,
        follow_heading: float | None = None,
        blocked_edges: set[tuple[str, str]] | None = None,
    ) -> str | None:
        target_pano_id = str(edge["target_pano_id"])
        if (current_pano_id, target_pano_id) in (blocked_edges or set()):
            return "blocked_edge"
        if target_pano_id in visited_pano_ids:
            return "visited_pano"
        if not avoid_dead_end_traps:
            return None

        current_room_id = _normalized_room_id(self.pano_room_mappings.get(current_pano_id))
        target_room_id = _normalized_room_id(self.pano_room_mappings.get(target_pano_id))
        if (
            current_room_id is not None
            and target_room_id is not None
            and target_room_id != current_room_id
        ):
            return None

        continuation_edges, _ = self._legal_edges(
            current_pano_id=target_pano_id,
            previous_pano_id=current_pano_id,
            blocked_edges=blocked_edges,
        )
        if not continuation_edges:
            return "dead_end_cycle_trap"

        if follow_heading is not None:
            _, projected_edge = _nearest_edge(float(follow_heading), continuation_edges)
            projected_target = str(projected_edge["target_pano_id"])
            if projected_target == current_pano_id or projected_target in visited_pano_ids:
                return "projected_cycle"
            return None

        for continuation in continuation_edges:
            continuation_target = str(continuation["target_pano_id"])
            if continuation_target == current_pano_id:
                continue
            if continuation_target in visited_pano_ids:
                continue
            return None
        return "dead_end_cycle_trap"


def _ordered_eight_views(views: Sequence[VisualView]) -> list[VisualView]:
    ordered = sorted(views, key=lambda item: item.capture_index)
    if len(ordered) != 8:
        raise ValueError(f"Expected exactly 8 current views, got {len(ordered)}.")
    return ordered


def _ranked_choice_labels(direction_decision: dict, labels: Sequence[str]) -> list[str]:
    candidates: list[str] = []
    chosen = direction_decision.get("chosen_view")
    if isinstance(chosen, str) and chosen in labels:
        candidates.append(chosen)

    memory_tree = direction_decision.get("memory_tree")
    view_scores = memory_tree.get("view_scores") if isinstance(memory_tree, dict) else None
    if isinstance(view_scores, list):
        ranked = sorted(
            [item for item in view_scores if isinstance(item, dict)],
            key=lambda item: int(item.get("rank", len(labels) + 1)),
        )
        for item in ranked:
            label = item.get("view_label")
            if isinstance(label, str) and label in labels and label not in candidates:
                candidates.append(label)

    for label in labels:
        if label not in candidates:
            candidates.append(label)
    return candidates


def _view_labels(views: Sequence[VisualView]) -> list[str]:
    return [f"V{index}" for index in range(1, len(views) + 1)]


def _normalized_room_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"null", "none", "nan"}:
        return None
    return normalized


def _incoming_edge_heading(
    pano_graph: dict[str, dict],
    *,
    previous_pano_id: str | None,
    current_pano_id: str,
) -> float | None:
    if not previous_pano_id:
        return None
    previous_node = pano_graph.get(previous_pano_id)
    if not isinstance(previous_node, dict):
        return None
    for edge in previous_node.get("neighbors", []) or []:
        if isinstance(edge, dict) and edge.get("target_pano_id") == current_pano_id:
            try:
                return float(edge["geocentric_heading_deg"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


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



def _resolve_existing_path(value, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{label} path is required.")
    path = Path(value).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except FileNotFoundError:
        return left == right


def _find_matching_image_path(items: Sequence[dict], image_path: str | Path) -> int | None:
    target = Path(image_path).resolve()
    for index, item in enumerate(items):
        raw_path = item.get("image_path")
        if isinstance(raw_path, str) and _same_path(Path(raw_path), target):
            return index
    return None


def _external_memory_item(
    image_path: Path,
    *,
    room_id: str,
    pano_id: str,
    capture_index: int,
    capture_label: str | None = None,
    capture_heading: float | None = None,
) -> dict:
    return {
        "memory_index": -1,
        "room_id": room_id,
        "pano_id": pano_id,
        "capture_index": int(capture_index),
        "capture_label": capture_label,
        "capture_heading": capture_heading,
        "image_path": str(image_path),
        "external_image": True,
    }


def _first_room_id(items: Sequence[dict]) -> str | None:
    for item in items:
        room_id = item.get("room_id")
        if isinstance(room_id, str):
            return room_id
    return None


def _compact_bridge(bridge: dict) -> dict:
    keys = [
        "current_node_id",
        "passage_node_id",
        "current_memory_key",
        "passage_memory_key",
        "bridge_similarity",
        "current_chain_min_parent_similarity",
        "passage_chain_min_parent_similarity",
        "current_chain_mean_parent_similarity",
        "passage_chain_mean_parent_similarity",
        "bridge_score_mode",
        "bridge_selection_mode",
        "same_bridge_item",
        "exclude_same_bridge_item",
        "same_bridge_item_fallback",
        "skipped_same_item_bridge_count",
        "continuity_score",
        "chain_mean_continuity",
        "chain_bottleneck_similarity",
        "continuity_edge_similarities",
        "total_bridge_depth",
        "total_score",
    ]
    return {key: bridge[key] for key in keys if key in bridge}


def _compact_chain(chain: Sequence[dict]) -> list[dict]:
    keys = [
        "id",
        "memory_key",
        "room_id",
        "pano_id",
        "capture_index",
        "capture_label",
        "capture_heading",
        "image_path",
        "depth",
        "sim_to_parent",
        "score",
        "target_similarity",
        "target_passage_label",
    ]
    return [
        {key: node[key] for key in keys if key in node}
        for node in chain
        if isinstance(node, dict)
    ]


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
