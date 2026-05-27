from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ..common.types import TaskSpec


@dataclass
class ParsedRoutePlan:
    instruction: str
    source_room_id: str | None
    target_room_id: str | None
    waypoint_room_ids: list[str] = field(default_factory=list)
    shortest_path: list[str] = field(default_factory=list)
    task: TaskSpec | None = None


class RoutePlanner:
    def __init__(self, *, room_graph: dict[str, dict]):
        self.room_graph = room_graph

    def route_to_goal(self, task: TaskSpec, current_room_id: str | None) -> list[str]:
        if not current_room_id or not task.goal_room_ids:
            return []
        ordered_targets = list(task.waypoint_room_ids) + list(task.goal_room_ids)
        return self._compose_ordered_route(current_room_id, ordered_targets)

    def shortest_room_path(self, source_room_id: str, target_room_id: str) -> list[str]:
        if not source_room_id or not target_room_id:
            return []
        if source_room_id not in self.room_graph or target_room_id not in self.room_graph:
            return []
        return self._bfs_room_path(source_room_id, target_room_id)

    def shortest_room_path_avoiding(
        self,
        source_room_id: str,
        target_room_id: str,
        forbidden_room_ids: set[str] | None,
    ) -> list[str]:
        return self._bfs_room_path(source_room_id, target_room_id, forbidden_room_ids=forbidden_room_ids)

    def shortest_room_route(
        self,
        source_room_id: str,
        target_room_id: str,
        waypoint_room_ids: list[str] | None = None,
    ) -> list[str]:
        ordered_rooms = [source_room_id] + list(waypoint_room_ids or []) + [target_room_id]
        ordered_rooms = [room_id for room_id in ordered_rooms if room_id]
        if len(ordered_rooms) < 2:
            return ordered_rooms
        return self._compose_ordered_route(ordered_rooms[0], ordered_rooms[1:])

    def desired_heading_for_route(self, current_room_id: str | None, route: list[str]) -> float | None:
        if not current_room_id or len(route) < 2:
            return None
        try:
            current_index = route.index(current_room_id)
        except ValueError:
            return None
        if current_index + 1 >= len(route):
            return None
        next_room_id = route[current_index + 1]
        for neighbor in self.room_graph.get(current_room_id, {}).get("neighbors", []):
            if neighbor["target_room_id"] == next_room_id:
                return neighbor.get("allocentric_heading_deg")
        return None

    def _compose_ordered_route(self, start_room_id: str, ordered_targets: list[str]) -> list[str]:
        route: list[str] = []
        cursor = start_room_id
        cleaned_targets = [room_id for room_id in ordered_targets if room_id]
        for target_room_id in cleaned_targets:
            if cursor == target_room_id:
                if not route:
                    route.append(cursor)
                continue
            segment = self.shortest_room_path(cursor, target_room_id)
            if not segment:
                return route
            if not route:
                route.extend(segment)
            else:
                route.extend(segment[1:])
            cursor = target_room_id
        return route

    def _bfs_room_path(
        self,
        start_room_id: str,
        goal_room_id: str,
        forbidden_room_ids: set[str] | None = None,
    ) -> list[str]:
        forbidden_room_ids = set(forbidden_room_ids or set())
        forbidden_room_ids.discard(start_room_id)
        forbidden_room_ids.discard(goal_room_id)
        queue: deque[str] = deque([start_room_id])
        parent = {start_room_id: None}
        while queue:
            room_id = queue.popleft()
            if room_id == goal_room_id:
                break
            for neighbor in self.room_graph.get(room_id, {}).get("neighbors", []):
                target_room_id = neighbor["target_room_id"]
                if target_room_id in forbidden_room_ids:
                    continue
                if target_room_id not in parent and target_room_id in self.room_graph:
                    parent[target_room_id] = room_id
                    queue.append(target_room_id)

        if goal_room_id not in parent:
            return []

        path: list[str] = []
        cursor: str | None = goal_room_id
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        path.reverse()
        return path


class InstructionRoutePlanner:
    """
    Spatial-layer planning flow:
    instruction -> parsed route constraints -> shortest room route
    """

    def __init__(self, *, instruction_parser, spatial_engine):
        self.instruction_parser = instruction_parser
        self.spatial_engine = spatial_engine

    def plan(self, instruction: str) -> ParsedRoutePlan:
        task = self.instruction_parser.parse(instruction)
        target_room_id = task.goal_room_ids[-1] if task.goal_room_ids else None
        shortest_path: list[str] = []
        if task.source_room_id and target_room_id:
            shortest_path = self.spatial_engine.shortest_room_route(
                task.source_room_id,
                target_room_id,
                task.waypoint_room_ids,
            )
        return ParsedRoutePlan(
            instruction=instruction,
            source_room_id=task.source_room_id,
            target_room_id=target_room_id,
            waypoint_room_ids=list(task.waypoint_room_ids),
            shortest_path=shortest_path,
            task=task,
        )
