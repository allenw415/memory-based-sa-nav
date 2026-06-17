from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from memory_nav.data.memory_localization import (
    DEFAULT_DINOV2_SALAD_MODEL,
    create_image_embedder,
    load_image_index_artifacts,
    normalize_rows,
)


def angular_distance_deg(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class VisualView:
    capture_index: int
    label: str
    heading: float
    embedding: object
    auxiliary_embedding: object | None = None
    path: str | None = None


@dataclass(frozen=True)
class VisualActionDecision:
    selected_capture_index: int
    selected_view_label: str
    selected_view_heading: float
    similarity: float
    second_similarity: float
    margin: float
    selected_action_index: int
    selected_action_heading: float
    view_scores: list[dict] = field(default_factory=list)
    scoring: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "selected_capture_index": self.selected_capture_index,
            "selected_view_label": self.selected_view_label,
            "selected_view_heading": self.selected_view_heading,
            "similarity": self.similarity,
            "second_similarity": self.second_similarity,
            "margin": self.margin,
            "selected_action_index": self.selected_action_index,
            "selected_action_heading": self.selected_action_heading,
            "view_scores": list(self.view_scores),
            "scoring": dict(self.scoring),
        }


@dataclass(frozen=True)
class ImageGoalStepResult:
    moved: bool
    reason: str
    current_pano_id: str
    next_pano_id: str
    trajectory_entry: dict | None = None


@dataclass(frozen=True)
class ImageGoalNavigationResult:
    success: bool
    reason: str
    start_pano_id: str
    final_pano_id: str
    target_room_id: str
    step_count: int
    pano_path: list[str]
    trajectory: list[dict]

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "reason": self.reason,
            "start_pano_id": self.start_pano_id,
            "final_pano_id": self.final_pano_id,
            "target_room_id": self.target_room_id,
            "step_count": self.step_count,
            "pano_path": list(self.pano_path),
            "trajectory": list(self.trajectory),
        }


class PureVisualDirectionPolicy:
    """Choose a view by weighted image similarity, then map its heading to an action."""

    def __init__(self, *, salad_alpha: float = 1.0):
        if not 0.0 <= float(salad_alpha) <= 1.0:
            raise ValueError("salad_alpha must be between 0 and 1.")
        self.salad_alpha = float(salad_alpha)

    def choose_action(
        self,
        *,
        goal_embedding,
        views: Sequence[VisualView],
        legal_action_headings: Sequence[float],
    ) -> VisualActionDecision:
        if not views:
            raise ValueError("At least one current view is required.")
        if not legal_action_headings:
            raise ValueError("At least one legal action heading is required.")

        import numpy as np

        primary_goal, auxiliary_goal = _split_goal_embeddings(goal_embedding)
        normalized_goal = normalize_rows(
            np.asarray(primary_goal, dtype=np.float32).reshape(1, -1)
        )[0]
        normalized_auxiliary_goal = (
            normalize_rows(np.asarray(auxiliary_goal, dtype=np.float32).reshape(1, -1))[0]
            if auxiliary_goal is not None
            else None
        )
        use_fusion = normalized_auxiliary_goal is not None and all(
            view.auxiliary_embedding is not None for view in views
        )
        scored_views: list[tuple[float, float, float | None, VisualView]] = []
        for view in views:
            normalized_view = normalize_rows(
                np.asarray(view.embedding, dtype=np.float32).reshape(1, -1)
            )[0]
            primary_score = float(normalized_view @ normalized_goal)
            auxiliary_score = None
            fused_score = primary_score
            if use_fusion:
                normalized_auxiliary_view = normalize_rows(
                    np.asarray(view.auxiliary_embedding, dtype=np.float32).reshape(1, -1)
                )[0]
                auxiliary_score = float(normalized_auxiliary_view @ normalized_auxiliary_goal)
                fused_score = (
                    self.salad_alpha * primary_score
                    + (1.0 - self.salad_alpha) * auxiliary_score
                )
            scored_views.append((fused_score, primary_score, auxiliary_score, view))

        scored_views.sort(key=lambda item: (-item[0], item[3].capture_index))
        best_score, _, _, best_view = scored_views[0]
        second_score = scored_views[1][0] if len(scored_views) > 1 else best_score
        ranked_actions = sorted(
            enumerate(legal_action_headings),
            key=lambda item: (
                angular_distance_deg(best_view.heading, float(item[1])),
                item[0],
            ),
        )
        selected_action_index, selected_action_heading = ranked_actions[0]
        scores_by_capture = {
            view.capture_index: (score, primary_score, auxiliary_score)
            for score, primary_score, auxiliary_score, view in scored_views
        }
        view_scores = []
        for view in sorted(views, key=lambda item: item.capture_index):
            score, primary_score, auxiliary_score = scores_by_capture[view.capture_index]
            view_scores.append(
                {
                    "capture_index": view.capture_index,
                    "label": view.label,
                    "heading": float(view.heading),
                    "path": view.path,
                    "similarity": float(score),
                    "salad_similarity": float(primary_score),
                    "siglip_similarity": (
                        float(auxiliary_score) if auxiliary_score is not None else None
                    ),
                    "selected": view.capture_index == best_view.capture_index,
                }
            )
        return VisualActionDecision(
            selected_capture_index=best_view.capture_index,
            selected_view_label=best_view.label,
            selected_view_heading=float(best_view.heading),
            similarity=float(best_score),
            second_similarity=float(second_score),
            margin=float(best_score - second_score),
            selected_action_index=int(selected_action_index),
            selected_action_heading=float(selected_action_heading),
            view_scores=view_scores,
            scoring={
                "mode": "salad_siglip_fusion" if use_fusion else "salad_only",
                "salad_alpha": self.salad_alpha if use_fusion else 1.0,
                "siglip_alpha": (1.0 - self.salad_alpha) if use_fusion else 0.0,
            },
        )


class IndexedPanoramaViewStore:
    """Load eight-view observations from manifests and a precomputed image index."""

    def __init__(
        self,
        *,
        index_path: str | Path,
        metadata_path: str | Path,
        manifest_root: str | Path,
        auxiliary_index_path: str | Path | None = None,
        auxiliary_metadata_path: str | Path | None = None,
    ):
        self.index_path = Path(index_path).resolve()
        self.metadata_path = Path(metadata_path).resolve()
        self.manifest_root = Path(manifest_root).resolve()
        self.image_embeddings = load_image_index_artifacts(self.index_path)
        self.auxiliary_embeddings = None
        self.auxiliary_metadata_items: list[dict] = []
        self._auxiliary_index_by_capture: dict[tuple[str, int], int] = {}
        if auxiliary_index_path is not None or auxiliary_metadata_path is not None:
            if auxiliary_index_path is None or auxiliary_metadata_path is None:
                raise ValueError("Both auxiliary index and metadata paths are required.")
            self.auxiliary_embeddings = load_image_index_artifacts(
                Path(auxiliary_index_path).resolve()
            )
            auxiliary_payload = _load_json(Path(auxiliary_metadata_path).resolve())
            auxiliary_items = auxiliary_payload.get("items")
            if not isinstance(auxiliary_items, list) or not auxiliary_items:
                raise RuntimeError("Auxiliary metadata file contains no indexed images.")
            self.auxiliary_metadata_items = [
                dict(item) for item in auxiliary_items if isinstance(item, dict)
            ]
            if len(self.auxiliary_metadata_items) != int(self.auxiliary_embeddings.shape[0]):
                raise RuntimeError("Auxiliary embeddings and metadata counts do not match.")
            for row_index, item in enumerate(self.auxiliary_metadata_items):
                pano_id = item.get("pano_id")
                capture_index = item.get("capture_index")
                if isinstance(pano_id, str) and isinstance(capture_index, int):
                    self._auxiliary_index_by_capture[(pano_id, capture_index)] = row_index

        metadata_payload = _load_json(self.metadata_path)
        raw_items = metadata_payload.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise RuntimeError(f"Metadata file contains no indexed images: {self.metadata_path}")
        self.metadata_items = [dict(item) for item in raw_items if isinstance(item, dict)]
        if len(self.metadata_items) != int(self.image_embeddings.shape[0]):
            raise RuntimeError("Index embeddings and metadata item counts do not match.")

        self._index_by_capture: dict[tuple[str, int], int] = {}
        self._index_by_path: dict[Path, int] = {}
        for row_index, item in enumerate(self.metadata_items):
            pano_id = item.get("pano_id")
            capture_index = item.get("capture_index")
            if isinstance(pano_id, str) and isinstance(capture_index, int):
                self._index_by_capture[(pano_id, capture_index)] = row_index
            capture_path = item.get("capture_path")
            if isinstance(capture_path, str) and capture_path:
                self._index_by_path[Path(capture_path).resolve()] = row_index

    def load_views(self, pano_id: str) -> list[VisualView]:
        manifest_path = self.manifest_root / pano_id / f"{pano_id}_manifest.json"
        manifest = _load_json(manifest_path)
        captures = manifest.get("captures")
        if not isinstance(captures, list) or len(captures) != 8:
            raise RuntimeError(f"Expected exactly 8 captures in {manifest_path}.")
        if int(manifest.get("fov", 0)) != 90:
            raise RuntimeError(f"Expected FOV 90 manifest: {manifest_path}")

        views: list[VisualView] = []
        for capture_index, capture in enumerate(captures):
            if not isinstance(capture, dict):
                raise RuntimeError(f"Invalid capture record in {manifest_path}.")
            row_index = self._index_by_capture.get((pano_id, capture_index))
            if row_index is None:
                raise KeyError(f"Missing indexed embedding for pano={pano_id} capture={capture_index}.")
            auxiliary_embedding = None
            if self.auxiliary_embeddings is not None:
                auxiliary_row = self._auxiliary_index_by_capture.get((pano_id, capture_index))
                if auxiliary_row is None:
                    raise KeyError(
                        f"Missing auxiliary embedding for pano={pano_id} capture={capture_index}."
                    )
                auxiliary_embedding = self.auxiliary_embeddings[auxiliary_row]
            views.append(
                VisualView(
                    capture_index=capture_index,
                    label=str(capture.get("label") or f"view_{capture_index}"),
                    heading=float(capture["heading"]),
                    embedding=self.image_embeddings[row_index],
                    auxiliary_embedding=auxiliary_embedding,
                    path=str(capture.get("path")) if capture.get("path") else None,
                )
            )
        return views

    def embedding_for_capture(self, pano_id: str, capture_index: int):
        key = (pano_id, int(capture_index))
        row_index = self._index_by_capture.get(key)
        if row_index is None:
            raise KeyError(f"Missing indexed embedding for pano={pano_id} capture={capture_index}.")
        primary = self.image_embeddings[row_index]
        if self.auxiliary_embeddings is None:
            return primary
        auxiliary_row = self._auxiliary_index_by_capture.get(key)
        if auxiliary_row is None:
            raise KeyError(
                f"Missing auxiliary embedding for pano={pano_id} capture={capture_index}."
            )
        return {"salad": primary, "siglip": self.auxiliary_embeddings[auxiliary_row]}

    def embedding_for_image(
        self,
        image_path: str | Path,
        *,
        embedding_model: str = DEFAULT_DINOV2_SALAD_MODEL,
        device: str = "auto",
        batch_size: int = 8,
    ):
        resolved_path = Path(image_path).resolve()
        row_index = self._index_by_path.get(resolved_path)
        if row_index is not None:
            return self.image_embeddings[row_index]
        for (pano_id, capture_index), candidate_index in self._index_by_capture.items():
            item = self.metadata_items[candidate_index]
            raw_path = item.get("capture_path")
            if isinstance(raw_path, str) and Path(raw_path).name == resolved_path.name:
                return self.image_embeddings[candidate_index]
        if not resolved_path.exists():
            raise FileNotFoundError(f"Goal image not found: {resolved_path}")
        embedder = create_image_embedder(
            model_name=embedding_model,
            device=device,
            batch_size=batch_size,
        )
        return embedder.encode_image_paths([resolved_path])[0]


class PanoramaGraphImageGoalSimulator:
    """Execute a visual policy while keeping pano identity inside the simulator."""

    def __init__(
        self,
        *,
        pano_graph: dict[str, dict],
        pano_room_mappings: dict[str, str | None],
        observation_provider: Callable[[str], Sequence[VisualView]],
        policy: PureVisualDirectionPolicy | None = None,
    ):
        self.pano_graph = pano_graph
        self.pano_room_mappings = pano_room_mappings
        self.observation_provider = observation_provider
        self.policy = policy or PureVisualDirectionPolicy()

    def step(
        self,
        *,
        current_pano_id: str,
        previous_pano_id: str | None,
        goal_embedding,
        step_index: int = 0,
    ) -> ImageGoalStepResult:
        if current_pano_id not in self.pano_graph:
            raise KeyError(f"Pano is not in the graph: {current_pano_id}")
        raw_edges = [
            edge
            for edge in self.pano_graph.get(current_pano_id, {}).get("neighbors", [])
            if isinstance(edge, dict)
            and isinstance(edge.get("target_pano_id"), str)
            and isinstance(edge.get("geocentric_heading_deg"), (int, float))
            and edge["target_pano_id"] in self.pano_graph
        ]
        if not raw_edges:
            return ImageGoalStepResult(
                moved=False,
                reason="no_legal_actions",
                current_pano_id=current_pano_id,
                next_pano_id=current_pano_id,
            )

        has_backtrack_edge = bool(
            previous_pano_id
            and any(edge["target_pano_id"] == previous_pano_id for edge in raw_edges)
        )
        non_backtracking_edges = [
            edge for edge in raw_edges if edge["target_pano_id"] != previous_pano_id
        ]
        anti_backtrack_applied = bool(has_backtrack_edge and non_backtracking_edges)
        legal_edges = non_backtracking_edges if anti_backtrack_applied else raw_edges
        decision = self.policy.choose_action(
            goal_embedding=goal_embedding,
            views=list(self.observation_provider(current_pano_id)),
            legal_action_headings=[
                float(edge["geocentric_heading_deg"]) for edge in legal_edges
            ],
        )
        selected_edge = legal_edges[decision.selected_action_index]
        next_pano_id = str(selected_edge["target_pano_id"])
        entry = {
            "step_index": int(step_index),
            "current_pano_id": current_pano_id,
            "current_room_id": self.pano_room_mappings.get(current_pano_id),
            "previous_pano_id": previous_pano_id,
            "anti_backtrack_applied": anti_backtrack_applied,
            **decision.to_dict(),
            "legal_actions": [
                {
                    "action_index": action_index,
                    "target_pano_id": edge["target_pano_id"],
                    "heading": float(edge["geocentric_heading_deg"]),
                }
                for action_index, edge in enumerate(legal_edges)
            ],
            "next_pano_id": next_pano_id,
            "next_room_id": self.pano_room_mappings.get(next_pano_id),
        }
        return ImageGoalStepResult(
            moved=True,
            reason="moved",
            current_pano_id=current_pano_id,
            next_pano_id=next_pano_id,
            trajectory_entry=entry,
        )

    def run(
        self,
        *,
        start_pano_id: str,
        goal_embedding,
        target_room_id: str,
        max_steps: int = 20,
    ) -> ImageGoalNavigationResult:
        if start_pano_id not in self.pano_graph:
            raise KeyError(f"Start pano is not in the graph: {start_pano_id}")
        current_pano_id = start_pano_id
        previous_pano_id: str | None = None
        pano_path = [current_pano_id]
        trajectory: list[dict] = []

        for step_index in range(max(int(max_steps), 0) + 1):
            if self.pano_room_mappings.get(current_pano_id) == target_room_id:
                return ImageGoalNavigationResult(
                    success=True,
                    reason="target_room_reached",
                    start_pano_id=start_pano_id,
                    final_pano_id=current_pano_id,
                    target_room_id=target_room_id,
                    step_count=len(trajectory),
                    pano_path=pano_path,
                    trajectory=trajectory,
                )
            if step_index >= max(int(max_steps), 0):
                break
            step = self.step(
                current_pano_id=current_pano_id,
                previous_pano_id=previous_pano_id,
                goal_embedding=goal_embedding,
                step_index=step_index,
            )
            if not step.moved:
                return ImageGoalNavigationResult(
                    success=False,
                    reason=step.reason,
                    start_pano_id=start_pano_id,
                    final_pano_id=current_pano_id,
                    target_room_id=target_room_id,
                    step_count=len(trajectory),
                    pano_path=pano_path,
                    trajectory=trajectory,
                )
            if step.trajectory_entry is not None:
                trajectory.append(step.trajectory_entry)
            previous_pano_id, current_pano_id = current_pano_id, step.next_pano_id
            pano_path.append(current_pano_id)

        return ImageGoalNavigationResult(
            success=False,
            reason="max_steps_exceeded",
            start_pano_id=start_pano_id,
            final_pano_id=current_pano_id,
            target_room_id=target_room_id,
            step_count=len(trajectory),
            pano_path=pano_path,
            trajectory=trajectory,
        )


def _split_goal_embeddings(goal_embedding):
    if isinstance(goal_embedding, dict):
        primary = goal_embedding.get("salad")
        auxiliary = goal_embedding.get("siglip")
        if primary is None:
            raise ValueError("Fused goal embedding is missing SALAD data.")
        return primary, auxiliary
    return goal_embedding, None


def resolve_goal_label(label: str, *, representatives_path: str | Path) -> dict:
    payload = _load_json(representatives_path)
    representatives = payload.get("representatives")
    if not isinstance(representatives, list):
        raise RuntimeError(f"Missing representatives list: {representatives_path}")
    for representative in representatives:
        if isinstance(representative, dict) and representative.get("label") == label:
            pano_id = representative.get("pano_id")
            capture_index = representative.get("capture_index")
            if not isinstance(pano_id, str) or not isinstance(capture_index, int):
                raise RuntimeError(f"Representative {label} lacks pano/capture metadata.")
            return dict(representative)
    raise KeyError(f"Goal label not found: {label}")


def _load_json(path: str | Path) -> dict:
    resolved_path = Path(path)
    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {resolved_path}")
    return payload
