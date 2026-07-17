from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Callable, Sequence

from memory_nav.spatial.routing import RoutePlanner

from .image_goal import (
    IndexedPanoramaViewStore,
    PanoramaGraphImageGoalSimulator,
    PureVisualDirectionPolicy,
    VisualDirectionPolicy,
)
from .passages import PassageSelector
from .vlm_direction import DirectionSelector, SparseVLMDirectionSimulator


@dataclass(frozen=True)
class NavigationEpisodeResult:
    success: bool
    reason: str
    start_pano_id: str
    final_pano_id: str
    target_room_id: str
    waypoint_room_ids: list[str]
    ordered_targets: list[str]
    completed_waypoints: list[str]
    step_count: int
    pano_path: list[str]
    rounds: list[dict] = field(default_factory=list)
    navigation_metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "reason": self.reason,
            "start_pano_id": self.start_pano_id,
            "final_pano_id": self.final_pano_id,
            "target_room_id": self.target_room_id,
            "waypoint_room_ids": list(self.waypoint_room_ids),
            "ordered_targets": list(self.ordered_targets),
            "completed_waypoints": list(self.completed_waypoints),
            "step_count": self.step_count,
            "pano_path": list(self.pano_path),
            "navigation_metrics": dict(self.navigation_metrics),
            "rounds": list(self.rounds),
        }


class NavigationEpisodeRunner:
    """Connect localization, room routing, passage selection, and pano actions."""

    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        pano_graph: dict[str, dict],
        pano_room_mappings: dict[str, str | None],
        view_store: IndexedPanoramaViewStore,
        localizer,
        passage_retriever,
        passage_selector: PassageSelector,
        passage_confidence_threshold: float = 0.5,
        direction_salad_alpha: float = 1.0,
        image_goal_policy: VisualDirectionPolicy | None = None,
        direction_selector: DirectionSelector | None = None,
        direction_confidence_threshold: float = 0.5,
        direction_burst_steps: int = 3,
        direction_max_turn_deg: float = 45.0,
        direction_commitment_mode: str = "off",
        direction_switch_margin: float = 0.03,
        direction_recovery_budget: int = 1,
        progress_callback: Callable[[str], None] | None = None,
        seed: int = 0,
    ):
        self.room_graph = room_graph
        self.pano_graph = pano_graph
        self.pano_room_mappings = pano_room_mappings
        self.view_store = view_store
        self.localizer = localizer
        self.passage_retriever = passage_retriever
        self.passage_selector = passage_selector
        self.passage_confidence_threshold = float(passage_confidence_threshold)
        self.progress_callback = progress_callback
        self.seed = int(seed)
        self.route_planner = RoutePlanner(room_graph=room_graph)
        self.simulator = PanoramaGraphImageGoalSimulator(
            pano_graph=pano_graph,
            pano_room_mappings=pano_room_mappings,
            observation_provider=view_store.load_views,
            policy=image_goal_policy
            or PureVisualDirectionPolicy(salad_alpha=direction_salad_alpha),
        )
        self.vlm_direction_simulator = (
            SparseVLMDirectionSimulator(
                pano_graph=pano_graph,
                pano_room_mappings=pano_room_mappings,
                observation_provider=view_store.load_views,
                direction_selector=direction_selector,
                confidence_threshold=direction_confidence_threshold,
                burst_steps=direction_burst_steps,
                max_turn_deg=direction_max_turn_deg,
                commitment_mode=direction_commitment_mode,
                switch_margin=direction_switch_margin,
                recovery_budget=direction_recovery_budget,
            )
            if direction_selector is not None
            else None
        )

    def run(
        self,
        *,
        start_pano_id: str,
        target_room_id: str,
        acceptable_target_room_ids: Sequence[str] | None = None,
        waypoint_room_ids: Sequence[str] | None = None,
        max_total_steps: int = 100,
        max_local_steps: int = 20,
    ) -> NavigationEpisodeResult:
        if start_pano_id not in self.pano_graph:
            raise KeyError(f"Start pano is not in the graph: {start_pano_id}")
        if target_room_id not in self.room_graph:
            raise KeyError(f"Target room is not in the room graph: {target_room_id}")

        acceptable_targets = {
            str(room_id).strip()
            for room_id in (acceptable_target_room_ids or [target_room_id])
            if str(room_id).strip()
        }
        if not acceptable_targets:
            acceptable_targets = {target_room_id}
        unknown_acceptable_targets = acceptable_targets.difference(self.room_graph)
        if unknown_acceptable_targets:
            unknown = ", ".join(sorted(unknown_acceptable_targets))
            raise KeyError(f"Acceptable target rooms are not in the room graph: {unknown}")

        waypoints = _clean_waypoints(waypoint_room_ids or [], target_room_id)
        ordered_targets = [*waypoints, target_room_id]
        target_index = 0
        completed_waypoints: list[str] = []
        rng = random.Random(self.seed)
        current_pano_id = start_pano_id
        previous_pano_id: str | None = None
        pano_path = [current_pano_id]
        rounds: list[dict] = []
        passage_cache: dict[str, list[dict]] = {}
        total_steps = 0
        last_confirmed_room_id: str | None = None
        simulated_room_override: str | None = None

        self._progress(
            f"start pano={start_pano_id} target={target_room_id} "
            f"waypoints={waypoints or 'none'}"
        )

        def finish(success: bool, reason: str) -> NavigationEpisodeResult:
            self._progress(
                f"finish success={str(success).lower()} reason={reason} "
                f"pano={current_pano_id} steps={total_steps}"
            )
            return NavigationEpisodeResult(
                success=success,
                reason=reason,
                start_pano_id=start_pano_id,
                final_pano_id=current_pano_id,
                target_room_id=target_room_id,
                waypoint_room_ids=waypoints,
                ordered_targets=ordered_targets,
                completed_waypoints=completed_waypoints,
                step_count=total_steps,
                pano_path=pano_path,
                navigation_metrics=_navigation_metrics(
                    pano_graph=self.pano_graph,
                    pano_room_mappings=self.pano_room_mappings,
                    pano_path=pano_path,
                ),
                rounds=rounds,
            )

        while True:
            hidden_room_id = self.pano_room_mappings.get(current_pano_id)
            if simulated_room_override is not None:
                self._progress(
                    f"localization simulator pano={current_pano_id} "
                    f"room={simulated_room_override}"
                )
                current_room_id = simulated_room_override
                localization_is_confident = True
                localization_payload = {
                    "predicted_room_id": current_room_id,
                    "confidence": 1.0,
                    "margin": 1.0,
                    "is_confident": True,
                    "room_scores": {current_room_id: 1.0},
                    "room_distribution": {current_room_id: 1.0},
                    "top_rooms": [current_room_id],
                    "top_matches": [],
                    "source": "simulator_room_transition",
                }
                localization_source = "simulator_room_transition"
                localization_observation = _simulated_localization_observation(
                    current_pano_id=current_pano_id,
                    room_id=current_room_id,
                )
                simulated_room_override = None
            else:
                views = self.view_store.load_views(current_pano_id)
                if not views:
                    return finish(False, "localization_observation_missing")
                observation = views[rng.randrange(len(views))]
                if not observation.path or not Path(observation.path).exists():
                    return finish(False, "localization_observation_missing")
                localization_observation = _view_summary(observation)

                self._progress(
                    f"localization image pano={current_pano_id} "
                    f"view={observation.capture_index}"
                )
                try:
                    localization = self.localizer.localize_from_images(
                        [observation.path],
                        exclude_same_pano_ids={current_pano_id},
                    )
                except Exception as exc:
                    rounds.append(
                        {
                            "round_index": len(rounds),
                            "start_pano_id": current_pano_id,
                            "localization_observation": localization_observation,
                            "failure": {
                                "reason": "localization_failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        }
                    )
                    return finish(False, "localization_failed")
                localization_payload = localization.to_dict()
                current_room_id = localization.predicted_room_id
                localization_is_confident = localization.is_confident
                localization_source = "image_retrieval"
                self._progress(
                    f"localized room={current_room_id} "
                    f"confidence={localization.confidence:.3f} "
                    f"confirmed={str(localization_is_confident).lower()}"
                )

            round_payload = {
                "round_index": len(rounds),
                "start_pano_id": current_pano_id,
                "hidden_start_room_id": hidden_room_id,
                "localization_observation": localization_observation,
                "localization_observations": [localization_observation],
                "localization_source": localization_source,
                "localization_excluded_pano_ids": [current_pano_id],
                "localization": localization_payload,
                "movement_steps": [],
            }
            rounds.append(round_payload)
            if not current_room_id:
                round_payload["failure"] = {"reason": "localization_failed"}
                return finish(False, "localization_failed")

            if current_room_id not in self.room_graph:
                round_payload["failure"] = {"reason": "localized_room_not_in_graph"}
                return finish(False, "localized_room_not_in_graph")

            if localization_is_confident:
                last_confirmed_room_id = current_room_id
            else:
                round_payload["localization_unconfirmed"] = True
                round_payload["continued_navigation_after_unconfirmed_localization"] = True

            if localization_is_confident and target_index < len(ordered_targets):
                is_final_target = target_index >= len(waypoints)
                expected_target = ordered_targets[target_index]
                if is_final_target and current_room_id in acceptable_targets:
                    target_index += 1
                    round_payload["completed_target_this_round"] = expected_target
                    round_payload["reached_acceptable_target_room_id"] = current_room_id
                elif current_room_id == expected_target:
                    if is_final_target:
                        round_payload["failure"] = {
                            "reason": "unacceptable_target_room_relocalized",
                            "parsed_target_room_id": target_room_id,
                            "reached_room_id": current_room_id,
                            "acceptable_target_room_ids": sorted(acceptable_targets),
                        }
                        return finish(False, "unacceptable_target_room_relocalized")
                    completed_waypoints.append(expected_target)
                    target_index += 1
                    round_payload["completed_target_this_round"] = expected_target

            if target_index >= len(ordered_targets):
                round_payload["goal_reached"] = True
                return finish(True, "target_room_relocalized")

            active_target_room_id = ordered_targets[target_index]
            route = self.route_planner.shortest_room_path(
                current_room_id,
                active_target_room_id,
            )
            self._progress(
                f"route current={current_room_id} target={active_target_room_id} "
                f"path={' -> '.join(route) if route else 'not found'}"
            )
            round_payload["active_target_room_id"] = active_target_room_id
            round_payload["route"] = route
            round_payload["completed_waypoints"] = list(completed_waypoints)
            if len(route) < 2:
                round_payload["failure"] = {"reason": "route_not_found"}
                return finish(False, "route_not_found")
            subgoal_room_id = route[1]
            round_payload["subgoal_room_id"] = subgoal_room_id

            requires_subgoal_passages = bool(
                getattr(self.passage_selector, "requires_subgoal_passages", True)
            )
            try:
                self._progress(
                    f"passages retrieve current={current_room_id} subgoal={subgoal_room_id} "
                    f"requires_subgoal={str(requires_subgoal_passages).lower()}"
                )
                current_candidates = self._room_passages(current_room_id, passage_cache)
                subgoal_candidates = (
                    self._room_passages(subgoal_room_id, passage_cache)
                    if requires_subgoal_passages
                    else []
                )
            except Exception as exc:
                round_payload["failure"] = {
                    "reason": "passage_retrieval_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return finish(False, "passage_retrieval_failed")
            round_payload["current_room_passages"] = current_candidates
            round_payload["subgoal_room_passages"] = subgoal_candidates
            round_payload["passage_requires_subgoal_passages"] = requires_subgoal_passages
            if not current_candidates or (requires_subgoal_passages and not subgoal_candidates):
                round_payload["failure"] = {"reason": "passage_retrieval_failed"}
                return finish(False, "passage_retrieval_failed")

            try:
                selector_source = type(self.passage_selector).__name__
                self._progress(
                    f"passage choosing selector={selector_source} current={current_room_id} "
                    f"subgoal={subgoal_room_id}"
                )
                choice = self.passage_selector.choose(
                    current_room_id=current_room_id,
                    subgoal_room_id=subgoal_room_id,
                    current_candidates=current_candidates,
                    subgoal_candidates=subgoal_candidates,
                )
            except Exception as exc:
                round_payload["failure"] = {
                    "reason": "passage_choice_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return finish(False, "passage_choice_failed")
            round_payload["passage_choice"] = choice
            if isinstance(choice.get("current_room_passage_records"), list):
                round_payload["current_room_passages"] = list(
                    choice["current_room_passage_records"]
                )
            if isinstance(choice.get("detected_passage_candidates"), list):
                round_payload["detected_passage_candidates"] = list(
                    choice["detected_passage_candidates"]
                )
            chosen_label = choice.get("chosen_label")
            confidence = float(choice.get("navigation_confidence") or 0.0)
            chosen = _resolve_chosen_passage_goal(current_candidates, choice)
            if chosen is None or confidence < self.passage_confidence_threshold:
                round_payload["failure"] = {
                    "reason": "passage_choice_failed",
                    "confidence_threshold": self.passage_confidence_threshold,
                }
                return finish(False, "passage_choice_failed")
            self._progress(
                f"passage selected label={chosen_label} confidence={confidence:.3f}"
            )

            goal_embedding = None
            if self.vlm_direction_simulator is None:
                if getattr(self.simulator.policy, "requires_goal_image_path", False):
                    goal_image_path = chosen.get("image_path")
                    if not isinstance(goal_image_path, str) or not Path(
                        goal_image_path
                    ).exists():
                        round_payload["failure"] = {"reason": "image_goal_missing"}
                        return finish(False, "image_goal_missing")
                    goal_embedding = goal_image_path
                else:
                    goal_embedding = self.view_store.embedding_for_capture(
                        str(chosen["pano_id"]),
                        int(chosen["capture_index"]),
                    )
            round_payload["image_goal"] = {
                "label": chosen_label,
                "source_label": chosen.get("source_label"),
                "image_path": chosen.get("image_path"),
                "pano_id": chosen.get("pano_id"),
                "capture_index": chosen.get("capture_index"),
                "target_source": chosen.get("target_source", "candidate_image"),
            }
            if isinstance(chosen.get("detected_candidate"), dict):
                round_payload["detected_passage_goal"] = dict(chosen["detected_candidate"])

            if total_steps >= max(int(max_total_steps), 0):
                round_payload["failure"] = {"reason": "max_total_steps_exceeded"}
                return finish(False, "max_total_steps_exceeded")

            room_before_movement = self.pano_room_mappings.get(current_pano_id)
            crossed_room_boundary = False
            localization_was_unconfirmed = not localization_is_confident
            moved_before_relocalization = False
            if self.vlm_direction_simulator is not None:
                goal_image_path = chosen.get("image_path")
                if not isinstance(goal_image_path, str) or not Path(goal_image_path).exists():
                    round_payload["failure"] = {"reason": "image_goal_missing"}
                    return finish(False, "image_goal_missing")
                round_payload["direction_bursts"] = []
                visited_pano_ids = {current_pano_id}
                commitment_state = self.vlm_direction_simulator.create_commitment_state(
                    current_pano_id=current_pano_id
                )
                round_payload["direction_commitment"] = commitment_state.to_dict()
                local_step_index = 0
                zero_action_replans = 0
                while local_step_index < max(int(max_local_steps), 0):
                    remaining_total = max(int(max_total_steps), 0) - total_steps
                    remaining_local = max(int(max_local_steps), 0) - local_step_index
                    if remaining_total <= 0:
                        round_payload["failure"] = {"reason": "max_total_steps_exceeded"}
                        return finish(False, "max_total_steps_exceeded")
                    try:
                        self._progress(
                            f"direction burst pano={current_pano_id} "
                            f"step={total_steps} goal={chosen_label}"
                        )
                        burst = self.vlm_direction_simulator.run_burst(
                            current_pano_id=current_pano_id,
                            previous_pano_id=previous_pano_id,
                            goal_image_path=goal_image_path,
                            step_index=total_steps,
                            visited_pano_ids=visited_pano_ids,
                            max_steps=min(remaining_total, remaining_local),
                            commitment_state=commitment_state,
                        )
                    except Exception as exc:
                        round_payload["failure"] = {
                            "reason": "direction_choice_failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        return finish(False, "direction_choice_failed")

                    round_payload["direction_bursts"].append(
                        {
                            "burst_index": len(round_payload["direction_bursts"]),
                            "start_pano_id": current_pano_id,
                            "stop_reason": burst.stop_reason,
                            "step_count": len(burst.actions),
                            "direction_decision": burst.direction_decision,
                            "direction_commitment": commitment_state.summary_dict(),
                        }
                    )
                    round_payload["direction_commitment"] = commitment_state.to_dict()
                    if burst.direction_decision:
                        self._progress(
                            f"direction selected view={burst.direction_decision.get('chosen_view')} "
                            f"confidence={float(burst.direction_decision.get('confidence') or 0.0):.3f}"
                        )
                    for action in burst.actions:
                        action = dict(action)
                        action.update(
                            {
                                "round_index": round_payload["round_index"],
                                "local_step_index": local_step_index,
                                "active_target_room_id": active_target_room_id,
                                "subgoal_room_id": subgoal_room_id,
                                "image_goal_label": chosen_label,
                            }
                        )
                        round_payload["movement_steps"].append(action)
                        previous_pano_id = action["current_pano_id"]
                        current_pano_id = action["next_pano_id"]
                        pano_path.append(current_pano_id)
                        total_steps += 1
                        local_step_index += 1
                        self._progress(
                            f"move step={total_steps} source={action['decision_source']} "
                            f"{action['current_pano_id']} -> {action['next_pano_id']} "
                            f"heading={float(action['selected_action_heading']):.1f}"
                        )
                    self._progress(
                        f"burst stopped reason={burst.stop_reason} "
                        f"steps={len(burst.actions)} pano={current_pano_id}"
                    )

                    if burst.actions:
                        zero_action_replans = 0
                    elif burst.stop_reason == "replan_required":
                        zero_action_replans += 1
                        if zero_action_replans <= 1:
                            continue

                    if localization_was_unconfirmed and burst.actions:
                        moved_before_relocalization = True
                        round_payload["relocalization_after_continuation"] = {
                            "at_pano_id": current_pano_id,
                            "steps_since_unconfirmed_localization": len(burst.actions),
                        }
                        break

                    room_after_movement = self.pano_room_mappings.get(current_pano_id)
                    if _is_effective_room_transition(room_before_movement, room_after_movement):
                        round_payload["room_boundary"] = {
                            "from_room_id": room_before_movement,
                            "to_room_id": room_after_movement,
                            "at_pano_id": current_pano_id,
                        }
                        crossed_room_boundary = True
                        simulated_room_override = room_after_movement
                        break
                    if burst.stop_reason in {
                        "cycle_detected",
                        "no_legal_actions",
                        "direction_confidence_below_threshold",
                    }:
                        round_payload["failure"] = {"reason": burst.stop_reason}
                        return finish(False, burst.stop_reason)
                    if not burst.actions:
                        round_payload["failure"] = {"reason": burst.stop_reason}
                        return finish(False, burst.stop_reason)

                if crossed_room_boundary or moved_before_relocalization:
                    continue
                round_payload["failure"] = {"reason": "max_local_steps_exceeded"}
                return finish(False, "max_local_steps_exceeded")

            for local_step_index in range(max(int(max_local_steps), 0)):
                if total_steps >= max(int(max_total_steps), 0):
                    round_payload["failure"] = {"reason": "max_total_steps_exceeded"}
                    return finish(False, "max_total_steps_exceeded")
                step = self.simulator.step(
                    current_pano_id=current_pano_id,
                    previous_pano_id=previous_pano_id,
                    goal_embedding=goal_embedding,
                    step_index=total_steps,
                )
                if not step.moved:
                    round_payload["failure"] = {"reason": step.reason}
                    return finish(False, step.reason)
                action = dict(step.trajectory_entry or {})
                action.update(
                    {
                        "round_index": round_payload["round_index"],
                        "local_step_index": local_step_index,
                        "active_target_room_id": active_target_room_id,
                        "subgoal_room_id": subgoal_room_id,
                        "image_goal_label": chosen_label,
                    }
                )
                round_payload["movement_steps"].append(action)
                previous_pano_id, current_pano_id = current_pano_id, step.next_pano_id
                pano_path.append(current_pano_id)
                total_steps += 1
                self._progress(
                    f"move step={total_steps} source=image_similarity "
                    f"{action['current_pano_id']} -> {action['next_pano_id']} "
                    f"heading={float(action['selected_action_heading']):.1f}"
                )
                if localization_was_unconfirmed:
                    moved_before_relocalization = True
                    round_payload["relocalization_after_continuation"] = {
                        "at_pano_id": current_pano_id,
                        "steps_since_unconfirmed_localization": 1,
                    }
                    break
                room_after_movement = self.pano_room_mappings.get(current_pano_id)
                if _is_effective_room_transition(room_before_movement, room_after_movement):
                    round_payload["room_boundary"] = {
                        "from_room_id": room_before_movement,
                        "to_room_id": room_after_movement,
                        "at_pano_id": current_pano_id,
                    }
                    crossed_room_boundary = True
                    simulated_room_override = room_after_movement
                    break

            if moved_before_relocalization:
                continue
            if not crossed_room_boundary:
                round_payload["failure"] = {"reason": "max_local_steps_exceeded"}
                return finish(False, "max_local_steps_exceeded")

    def _progress(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(f"[navigation] {message}")

    def _room_passages(self, room_id: str, cache: dict[str, list[dict]]) -> list[dict]:
        if room_id not in cache:
            cache[room_id] = [dict(item) for item in self.passage_retriever.retrieve(room_id)]
        return [dict(item) for item in cache[room_id]]


def _resolve_chosen_passage_goal(
    current_candidates: Sequence[dict],
    choice: dict,
) -> dict | None:
    chosen_label = choice.get("chosen_label")
    direct = next(
        (
            dict(item)
            for item in current_candidates
            if item.get("label") == chosen_label
        ),
        None,
    )
    if direct is not None:
        direct.setdefault("target_source", "candidate_image")
        return direct

    direction_target = choice.get("direction_target")
    if isinstance(direction_target, dict) and direction_target.get("image_path"):
        return _goal_from_direction_target(direction_target, chosen_label)

    ranking = choice.get("passage_ranking")
    if isinstance(ranking, list):
        detected = next(
            (
                item
                for item in ranking
                if isinstance(item, dict)
                and (item.get("selected") or item.get("label") == chosen_label)
            ),
            None,
        )
        if isinstance(detected, dict):
            return _goal_from_detected_candidate(detected, chosen_label)
    return None


def _goal_from_direction_target(direction_target: dict, chosen_label) -> dict:
    source_passage = direction_target.get("source_passage")
    source_passage = dict(source_passage) if isinstance(source_passage, dict) else {}
    goal = dict(source_passage)
    goal.update(
        {
            "label": chosen_label,
            "source_label": direction_target.get("source_label") or source_passage.get("label"),
            "image_path": direction_target.get("image_path"),
            "pano_id": direction_target.get("pano_id") or source_passage.get("pano_id"),
            "capture_index": direction_target.get("capture_index")
            if direction_target.get("capture_index") is not None
            else source_passage.get("capture_index"),
            "target_source": "source_passage_image",
        }
    )
    detected = direction_target.get("detected_candidate")
    if isinstance(detected, dict):
        goal["detected_candidate"] = dict(detected)
    return goal


def _goal_from_detected_candidate(candidate: dict, chosen_label) -> dict:
    source_passage = candidate.get("source_passage")
    source_passage = dict(source_passage) if isinstance(source_passage, dict) else {}
    image_path = source_passage.get("image_path") or candidate.get("source_image_path")
    if not image_path:
        return {}
    goal = dict(source_passage)
    goal.update(
        {
            "label": chosen_label,
            "source_label": candidate.get("source_label") or source_passage.get("label"),
            "image_path": image_path,
            "pano_id": source_passage.get("pano_id") or candidate.get("pano_id"),
            "capture_index": source_passage.get("capture_index")
            if source_passage.get("capture_index") is not None
            else candidate.get("capture_index"),
            "target_source": "source_passage_image",
            "detected_candidate": {
                key: candidate.get(key)
                for key in [
                    "label",
                    "source_label",
                    "crop_image_path",
                    "masked_image_path",
                    "comparison_image_path",
                    "detection_box_xyxy",
                    "crop_box_xyxy",
                    "selection_score",
                    "target_mean_similarity",
                    "negative_mean_similarity",
                ]
                if key in candidate
            },
        }
    )
    return goal


def _navigation_metrics(
    *,
    pano_graph: dict[str, dict],
    pano_room_mappings: dict[str, str | None],
    pano_path: Sequence[str],
) -> dict:
    room_sequence = _room_sequence(pano_room_mappings, pano_path)
    visited_room_ids = _unique_non_null(room_sequence)
    distances = _path_distances_m(pano_graph, pano_path)
    return {
        "room_sequence": room_sequence,
        "visited_room_ids": visited_room_ids,
        "visited_room_count": len(visited_room_ids),
        "room_transition_count": max(len(room_sequence) - 1, 0),
        "pano_step_count": max(len(pano_path) - 1, 0),
        "distance_unit": "meters",
        "distance_source": "pano_graph_lat_lng_haversine",
        **distances,
    }


def _room_sequence(
    pano_room_mappings: dict[str, str | None],
    pano_path: Sequence[str],
) -> list[str]:
    sequence: list[str] = []
    for pano_id in pano_path:
        room_id = _normalized_room_id(pano_room_mappings.get(pano_id))
        if room_id is not None and (not sequence or sequence[-1] != room_id):
            sequence.append(room_id)
    return sequence


def _unique_non_null(values: Sequence[str | None]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        normalized = _normalized_room_id(value)
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _path_distances_m(pano_graph: dict[str, dict], pano_path: Sequence[str]) -> dict:
    missing_pano_ids = [
        pano_id
        for pano_id in dict.fromkeys(pano_path)
        if _lat_lng(pano_graph.get(pano_id)) is None
    ]
    straight_line = None
    if len(pano_path) >= 2:
        straight_line = _distance_between_panos_m(
            pano_graph,
            pano_path[0],
            pano_path[-1],
        )

    path_distance = 0.0
    missing_segments = 0
    for source, target in zip(pano_path, pano_path[1:], strict=False):
        segment_distance = _distance_between_panos_m(pano_graph, source, target)
        if segment_distance is None:
            missing_segments += 1
            continue
        path_distance += segment_distance

    return {
        "start_to_final_straight_line_distance_m": straight_line,
        "pano_path_distance_m": path_distance if missing_segments == 0 else None,
        "distance_missing_pano_ids": missing_pano_ids,
        "distance_missing_segment_count": missing_segments,
    }


def _distance_between_panos_m(
    pano_graph: dict[str, dict],
    source_pano_id: str,
    target_pano_id: str,
) -> float | None:
    source = _lat_lng(pano_graph.get(source_pano_id))
    target = _lat_lng(pano_graph.get(target_pano_id))
    if source is None or target is None:
        return None
    return _haversine_m(source[0], source[1], target[0], target[1])


def _lat_lng(record: dict | None) -> tuple[float, float] | None:
    if not isinstance(record, dict):
        return None
    lat = record.get("lat")
    lng = record.get("lng")
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return None
    return float(lat), float(lng)


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_m = 6_371_000.0
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lng2 - lng1)
    a = (
        sin(delta_phi / 2.0) ** 2
        + cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * earth_radius_m * asin(sqrt(a))


def _clean_waypoints(waypoint_room_ids: Sequence[str], target_room_id: str) -> list[str]:
    cleaned: list[str] = []
    for room_id in waypoint_room_ids:
        if not isinstance(room_id, str):
            continue
        value = room_id.strip()
        if value and value != target_room_id and value not in cleaned:
            cleaned.append(value)
    return cleaned


def _normalized_room_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"null", "none", "nan"}:
        return None
    return normalized


def _is_effective_room_transition(before: object, after: object) -> bool:
    normalized_after = _normalized_room_id(after)
    if normalized_after is None:
        return False
    return normalized_after != _normalized_room_id(before)


def _simulated_localization_observation(*, current_pano_id: str, room_id: str | None) -> dict:
    return {
        "pano_id": current_pano_id,
        "room_id": room_id,
        "source": "simulator_room_transition",
    }


def _view_summary(view) -> dict:
    return {
        "capture_index": int(view.capture_index),
        "label": str(view.label),
        "heading": float(view.heading),
        "path": view.path,
    }
