from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..spatial.routing import RoutePlanner
from .advisor import PassageAlignmentAdvisor
from .retrieval import MemoryLocalizationResult, MemoryRoomLocalizer


class InteractiveMemoryNavigator:
    def __init__(
        self,
        *,
        room_graph: dict[str, dict],
        localizer: MemoryRoomLocalizer,
        passage_advisor: PassageAlignmentAdvisor,
    ):
        self.room_graph = room_graph
        self.localizer = localizer
        self.passage_advisor = passage_advisor
        self.route_planner = RoutePlanner(room_graph=room_graph)

    def guide(
        self,
        *,
        target_room_id: str,
        waypoint_room_ids: Sequence[str] | None = None,
        localization_images: Sequence[str | Path] | None = None,
        passage_images: dict[str, str | Path] | None = None,
    ) -> dict:
        ordered_targets = self._ordered_targets(waypoint_room_ids or [], target_room_id)
        if not ordered_targets:
            return {
                "action_request": "missing_target",
                "message_zh": "請先提供目標展廳。",
            }
        if not localization_images:
            return {
                "action_request": "capture_more_localization_views",
                "ordered_targets": ordered_targets,
                "message_zh": "請先拍一張你現在看到的展廳照片，我會用這張照片判斷你目前在哪裡。",
            }

        localization = self.localizer.localize_from_images(localization_images)
        localization_payload = localization.to_dict()
        if not localization.is_confident or not localization.predicted_room_id:
            return {
                "action_request": "capture_more_localization_views",
                "ordered_targets": ordered_targets,
                "localization": localization_payload,
                "message_zh": "我還不能穩定判斷你在哪個展廳。請原地向左或向右再拍一張，盡量包含牆面、展品或入口標示。",
            }

        current_room_id = localization.predicted_room_id
        remaining_targets = self._remaining_targets(current_room_id, ordered_targets)
        if not remaining_targets:
            return {
                "action_request": "goal_reached",
                "current_room_id": current_room_id,
                "ordered_targets": ordered_targets,
                "localization": localization_payload,
                "message_zh": f"我判斷你已經到達最後目標 {target_room_id}。",
            }

        active_target_room_id = remaining_targets[0]
        route = self._route_through_targets(current_room_id, remaining_targets)
        next_room_id = route[1] if len(route) > 1 else active_target_room_id

        base_payload = {
            "current_room_id": current_room_id,
            "active_target_room_id": active_target_room_id,
            "next_room_id": next_room_id,
            "target_room_id": target_room_id,
            "waypoint_room_ids": list(waypoint_room_ids or []),
            "ordered_targets": ordered_targets,
            "route": route,
            "localization": localization_payload,
        }
        if not passage_images:
            return {
                "action_request": "capture_passage_views",
                **base_payload,
                "message_zh": (
                    f"我判斷你目前可能在 {current_room_id}。"
                    f"下一個 waypoint/目標是 {active_target_room_id}，route 上下一個房間是 {next_room_id}。"
                    "請拍附近主要通道或出口；如果有多個通道，請分別拍左前方、正前方、右前方。"
                ),
            }

        guidance = self.passage_advisor.advise(
            current_room_id=current_room_id,
            next_room_id=next_room_id,
            active_target_room_id=active_target_room_id,
            route=route,
            passage_images=passage_images,
            localization=localization_payload,
        )
        return {
            "action_request": "move",
            **base_payload,
            **guidance,
        }

    @staticmethod
    def _ordered_targets(waypoint_room_ids: Sequence[str], target_room_id: str) -> list[str]:
        ordered = []
        for room_id in list(waypoint_room_ids) + [target_room_id]:
            if isinstance(room_id, str):
                cleaned = room_id.strip()
                if cleaned and cleaned not in ordered:
                    ordered.append(cleaned)
        return ordered

    @staticmethod
    def _remaining_targets(current_room_id: str, ordered_targets: list[str]) -> list[str]:
        if current_room_id in ordered_targets:
            index = ordered_targets.index(current_room_id)
            return ordered_targets[index + 1 :]
        return list(ordered_targets)

    def _route_through_targets(self, current_room_id: str, remaining_targets: list[str]) -> list[str]:
        if not remaining_targets:
            return [current_room_id]
        if len(remaining_targets) == 1:
            route = self.route_planner.shortest_room_path(current_room_id, remaining_targets[0])
        else:
            route = self.route_planner.shortest_room_route(
                current_room_id,
                remaining_targets[-1],
                remaining_targets[:-1],
            )
        return route if route else [current_room_id] + list(remaining_targets)
