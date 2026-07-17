from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from memory_nav.common.model_client import ModelResponseClient, parse_json_output
from memory_nav.common.room_profiles import visual_profile_anchor_entities


NAVIGATION_QUERY_ENTITY_TYPES = ("gallery", "artwork")


@dataclass(frozen=True)
class ParsedNavigationQueryEntity:
    name: str
    entity_type: str
    predicted_room_id: str | None
    confidence: float

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        allowed_room_ids: set[str],
    ) -> "ParsedNavigationQueryEntity":
        name = str(payload.get("name") or "").strip()
        entity_type = str(payload.get("entity_type") or "").strip()
        predicted_room_id = payload.get("predicted_room_id")
        confidence = payload.get("confidence")

        if not name:
            raise ValueError("Parsed navigation entity is missing a name.")
        if entity_type not in NAVIGATION_QUERY_ENTITY_TYPES:
            raise ValueError(f"Unsupported parsed entity type: {entity_type}")
        if predicted_room_id is not None and predicted_room_id not in allowed_room_ids:
            raise ValueError(f"Parsed entity used unknown room id: {predicted_room_id}")
        if not isinstance(confidence, (int, float)):
            raise ValueError(f"Parsed entity has invalid confidence: {confidence!r}")
        numeric_confidence = float(confidence)
        if numeric_confidence < 0.0 or numeric_confidence > 1.0:
            raise ValueError(f"Parsed entity confidence is out of range: {numeric_confidence}")
        if entity_type == "gallery" and predicted_room_id is None:
            raise ValueError(f"Gallery entity must resolve to a room id: {name}")
        return cls(
            name=name,
            entity_type=entity_type,
            predicted_room_id=predicted_room_id,
            confidence=numeric_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "predicted_room_id": self.predicted_room_id,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ParsedNavigationQuery:
    raw_instruction: str
    target_room_id: str
    waypoint_room_ids: list[str] = field(default_factory=list)
    goal_entities: list[ParsedNavigationQueryEntity] = field(default_factory=list)
    waypoint_entities: list[ParsedNavigationQueryEntity] = field(default_factory=list)
    raw_response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_instruction": self.raw_instruction,
            "target_room_id": self.target_room_id,
            "waypoint_room_ids": list(self.waypoint_room_ids),
            "goal_entities": [entity.to_dict() for entity in self.goal_entities],
            "waypoint_entities": [entity.to_dict() for entity in self.waypoint_entities],
            "raw_response": dict(self.raw_response),
        }


class NavigationQueryParser:
    def __init__(
        self,
        *,
        model_client: ModelResponseClient,
        model: str,
        room_graph: dict[str, dict],
        max_theme_lines: int | None = None,
    ):
        self.model_client = model_client
        self.model = model
        self.room_graph = room_graph
        self.room_ids = list(room_graph)
        self.max_theme_lines = max_theme_lines

    def parse(self, instruction: str) -> ParsedNavigationQuery:
        request_body = self.build_request(instruction)
        parsed = parse_json_output(self.model_client.create(request_body))
        return parse_navigation_query_payload(
            parsed,
            instruction=instruction,
            room_ids=self.room_ids,
        )

    def build_request(self, instruction: str) -> dict[str, Any]:
        return {
            "model": self.model,
            "instructions": build_navigation_query_parse_instructions(),
            "input": build_navigation_query_parse_input(
                instruction=instruction,
                room_ids=self.room_ids,
                theme_lines=build_gallery_theme_lines(
                    self.room_graph,
                    max_lines=self.max_theme_lines,
                ),
            ),
            "reasoning": {"effort": "low"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "navigation_query_parse",
                    "strict": True,
                    "schema": build_navigation_query_parse_schema(self.room_ids),
                }
            },
        }


def parse_navigation_query_payload(
    payload: dict[str, Any],
    *,
    instruction: str,
    room_ids: Sequence[str],
) -> ParsedNavigationQuery:
    allowed_room_ids = set(room_ids)
    goal_entities = _parse_entities(
        payload.get("goal_entities"),
        allowed_room_ids=allowed_room_ids,
        field_name="goal_entities",
    )
    waypoint_entities = _parse_entities(
        payload.get("waypoint_entities"),
        allowed_room_ids=allowed_room_ids,
        field_name="waypoint_entities",
    )
    goal_room_ids = _ordered_unique_room_ids(goal_entities)
    if not goal_room_ids:
        raise ValueError("Navigation query did not resolve any goal room ids.")
    return ParsedNavigationQuery(
        raw_instruction=instruction,
        target_room_id=goal_room_ids[-1],
        waypoint_room_ids=_ordered_unique_room_ids(waypoint_entities),
        goal_entities=goal_entities,
        waypoint_entities=waypoint_entities,
        raw_response=payload,
    )


def build_navigation_query_parse_instructions() -> str:
    return " ".join(
        [
            "You are a museum navigation query parser.",
            "Extract only ordered waypoint entities and ordered goal entities from the user query.",
            "Classify each extracted entity as gallery or artwork.",
            "Do not extract or return source entities, source rooms, current locations, or start locations.",
            "Treat explicit room, gallery, stairs, or lift mentions as gallery entities.",
            "Also treat a gallery described by its collection, culture, period, or visual theme as a gallery entity.",
            "Treat named objects, sculptures, monuments, reliefs, gates, statues, and artworks as artwork entities.",
            "Treat entities mentioned after from, starting at, current, or currently in as source context and ignore them.",
            "Treat entities mentioned after passing, via, through, or by way of as waypoints.",
            "Treat entities mentioned after to, toward, visit, find, go to, navigate to, or take me to as goals.",
            "Preserve entity order from the original query.",
            "Infer gallery and artwork room ids using only the provided gallery themes.",
            "A thematic gallery description may resolve to any allowed room whose theme is the best semantic match.",
            "Do not invent room ids, entities, source fields, or extra steps.",
            "For an explicit Room ID, predicted_room_id must match that ID and confidence must be 1.0.",
            "For a theme-described gallery or artwork, predicted_room_id must be one allowed room id and confidence must be between 0 and 1.",
            "Return JSON only.",
        ]
    )


def build_navigation_query_parse_input(
    *,
    instruction: str,
    room_ids: Sequence[str],
    theme_lines: str,
) -> str:
    return "\n".join(
        [
            "Allowed room ids:",
            ", ".join(room_ids),
            "",
            "Gallery themes:",
            theme_lines,
            "",
            "Task:",
            "1. Identify waypoint and goal spans from the wording.",
            "2. Ignore source/current/start spans.",
            "3. Classify each extracted entity as gallery or artwork; thematic collection descriptions are galleries.",
            "4. Ground each entity to one allowed room id using the gallery themes.",
            "5. Return no source fields.",
            "",
            f"Query: {instruction}",
        ]
    )


def build_navigation_query_parse_schema(room_ids: Sequence[str]) -> dict[str, Any]:
    entity_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "entity_type": {"type": "string", "enum": list(NAVIGATION_QUERY_ENTITY_TYPES)},
            "predicted_room_id": {"type": ["string", "null"], "enum": list(room_ids) + [None]},
            "confidence": {"type": "number"},
        },
        "required": ["name", "entity_type", "predicted_room_id", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "goal_entities": {"type": "array", "items": entity_schema},
            "waypoint_entities": {"type": "array", "items": entity_schema},
        },
        "required": ["goal_entities", "waypoint_entities"],
        "additionalProperties": False,
    }


def build_gallery_theme_lines(
    room_graph: dict[str, dict],
    *,
    max_lines: int | None = None,
    max_anchors: int = 4,
) -> str:
    lines: list[str] = []
    for room_id, node in room_graph.items():
        title = _string_value(node.get("title") or node.get("display_name") or room_id)
        anchors = visual_profile_anchor_entities(node, max_items=max_anchors)
        if not anchors:
            anchors = [
                value
                for value in (_string_value(node.get("category")),)
                if value
            ]
        details = "; ".join(anchors)
        if details:
            lines.append(f"- {room_id}: {title}. {details}")
        else:
            lines.append(f"- {room_id}: {title}")
        if max_lines is not None and len(lines) >= max_lines:
            break
    return "\n".join(lines)


def _parse_entities(
    value: object,
    *,
    allowed_room_ids: set[str],
    field_name: str,
) -> list[ParsedNavigationQueryEntity]:
    if not isinstance(value, list):
        raise ValueError(f"Parsed navigation query field must be a list: {field_name}")
    entities: list[ParsedNavigationQueryEntity] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Parsed navigation query entity must be an object: {field_name}")
        entity = ParsedNavigationQueryEntity.from_payload(
            item,
            allowed_room_ids=allowed_room_ids,
        )
        if entity.predicted_room_id is not None:
            entities.append(entity)
    return entities


def _ordered_unique_room_ids(entities: Sequence[ParsedNavigationQueryEntity]) -> list[str]:
    room_ids: list[str] = []
    for entity in entities:
        room_id = entity.predicted_room_id
        if room_id and room_id not in room_ids:
            room_ids.append(room_id)
    return room_ids


def _string_value(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
