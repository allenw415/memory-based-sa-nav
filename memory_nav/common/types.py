from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class RoomNode:
    room_id: str
    display_name: str
    floor: str
    category: str | None
    title: str | None
    aliases: list[str] = field(default_factory=list)
    neighbors: list[JsonDict] = field(default_factory=list)
    synthetic: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "room_id": self.room_id,
            "display_name": self.display_name,
            "floor": self.floor,
            "category": self.category,
            "title": self.title,
            "aliases": list(self.aliases),
            "neighbors": list(self.neighbors),
            "synthetic": self.synthetic,
        }


@dataclass
class PanoNode:
    pano_id: str
    floor: str
    lat: float | None
    lng: float | None
    neighbors: list[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "pano_id": self.pano_id,
            "floor": self.floor,
            "lat": self.lat,
            "lng": self.lng,
            "neighbors": list(self.neighbors),
        }


@dataclass
class RoomGroundingEntry:
    room_id: str
    floor: str
    pano_ids: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    anchor_entities: list[str] = field(default_factory=list)
    notes: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "room_id": self.room_id,
            "floor": self.floor,
            "pano_ids": list(self.pano_ids),
            "aliases": list(self.aliases),
            "anchor_entities": list(self.anchor_entities),
            "notes": self.notes,
        }


@dataclass
class RenderedView:
    label: str
    heading: float
    path: str
    url: str | None = None


@dataclass
class EntityDetection:
    name: str
    confidence: float
    kind: str
    source_view: str
    location_scope: str = "inside"
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class ViewDetection:
    capture_label: str
    entities: list[EntityDetection] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class Observation:
    pano_id: str
    views: list[RenderedView] = field(default_factory=list)
    entities: list[EntityDetection] = field(default_factory=list)
    heading_estimate: float | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class BeliefState:
    current_pano_id: str
    previous_pano_id: str | None = None
    current_room_id: str | None = None
    grounded_room_id: str | None = None
    current_heading: float = 0.0
    pano_belief: dict[str, float] = field(default_factory=dict)
    room_belief: dict[str, float] = field(default_factory=dict)
    visited_panos: set[str] = field(default_factory=set)
    visited_rooms: set[str] = field(default_factory=set)
    grounded_entities: dict[str, list[str]] = field(default_factory=dict)
    junction_stack: list[str] = field(default_factory=list)


@dataclass
class CandidateAction:
    target_pano_id: str
    absolute_heading: float
    relative_heading: float | None
    relative_label: str | None
    target_room_id: str | None
    score: float
    reason: str
    route_step_index: int | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class ParsedNavigationEntity:
    name: str
    entity_type: str
    predicted_room_id: str | None = None
    confidence: float | None = None


@dataclass
class SourcePanoResolution:
    source_room_id: str
    pano_id: str | None = None
    candidate_pano_ids: list[str] = field(default_factory=list)
    resolution_method: str = "representative_grounding"


@dataclass
class TaskSpec:
    task_type: str
    raw_instruction: str
    source_room_id: str | None = None
    source_entity: ParsedNavigationEntity | None = None
    goal_room_ids: list[str] = field(default_factory=list)
    waypoint_room_ids: list[str] = field(default_factory=list)
    goal_entities: list[ParsedNavigationEntity] = field(default_factory=list)
    waypoint_entities: list[ParsedNavigationEntity] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class PolicyOutput:
    action: CandidateAction | None
    rationale: str


@dataclass
class ReasoningInput:
    task: TaskSpec
    route: list[str]
    candidates: list[CandidateAction] = field(default_factory=list)
    current_room_id: str | None = None
    subgoal_room_id: str | None = None
    current_room_context: JsonDict = field(default_factory=dict)
    visible_passages: list[JsonDict] = field(default_factory=list)
    spatial_alignment: JsonDict = field(default_factory=dict)
    view_contexts: list[JsonDict] = field(default_factory=list)
