from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, deque
from html import escape
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

ROOM_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#0f766e",
    "#9333ea",
    "#c2410c",
    "#1d4ed8",
    "#65a30d",
    "#db2777",
    "#0e7490",
    "#ca8a04",
    "#4338ca",
    "#16a34a",
    "#e11d48",
    "#0369a1",
    "#a16207",
    "#6d28d9",
    "#15803d",
    "#f97316",
    "#b91c1c",
    "#7e22ce",
    "#0284c7",
    "#84cc16",
    "#c026d3",
    "#ea580c",
    "#047857",
    "#1e40af",
    "#a21caf",
    "#b45309",
    "#be185d",
    "#0d9488",
    "#7c2d12",
    "#3b82f6",
    "#92400e",
    "#2dd4bf",
]

STATUS_COLORS = {
    "room": "#2563eb",
    "null": "#475569",
    "unknown": "#94a3b8",
}


def load_json(path: str | Path) -> JsonDict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict JSON: {path}")
    return data


def extract_grounding_mapping(grounding_payload: JsonDict | None) -> tuple[dict[str, str | None], dict[str, str]]:
    if not grounding_payload:
        return {}, {}
    raw_mappings = grounding_payload.get("mappings", grounding_payload)
    if not isinstance(raw_mappings, dict):
        return {}, {}
    raw_sources = grounding_payload.get("sources", {})
    sources = raw_sources if isinstance(raw_sources, dict) else {}

    mappings: dict[str, str | None] = {}
    source_map: dict[str, str] = {}
    for pano_id, room_id in raw_mappings.items():
        if not isinstance(pano_id, str) or not pano_id:
            continue
        normalized_room_id = str(room_id).strip() if room_id is not None else None
        if normalized_room_id is None or normalized_room_id.lower() == "null":
            mappings[pano_id] = None
        else:
            mappings[pano_id] = normalized_room_id
        source = sources.get(pano_id)
        if isinstance(source, str) and source:
            source_map[pano_id] = source
    return mappings, source_map


def build_room_color_map(room_ids: list[str]) -> dict[str, str]:
    color_map: dict[str, str] = {}
    for index, room_id in enumerate(sorted(set(room_ids), key=_room_sort_key)):
        color_map[room_id] = ROOM_PALETTE[index] if index < len(ROOM_PALETTE) else _generated_room_color(index)
    return color_map


def room_color(room_id: str | None, room_color_map: dict[str, str] | None = None) -> str:
    if not room_id:
        return STATUS_COLORS["unknown"]
    if room_color_map and room_id in room_color_map:
        return room_color_map[room_id]
    fallback_map = build_room_color_map([room_id])
    return fallback_map[room_id]


def build_visualization_payload(
    pano_graph: JsonDict,
    *,
    room_graph: JsonDict | None = None,
    grounding_payload: JsonDict | None = None,
) -> JsonDict:
    room_graph = room_graph or {}
    mappings, sources = extract_grounding_mapping(grounding_payload)
    mapped_room_ids = [room_id for room_id in mappings.values() if isinstance(room_id, str) and room_id]
    room_color_map = build_room_color_map(mapped_room_ids)
    incoming = Counter()
    edge_records: list[JsonDict] = []

    for source_id, source in pano_graph.items():
        if not isinstance(source, dict):
            continue
        for index, neighbor in enumerate(source.get("neighbors", [])):
            if not isinstance(neighbor, dict):
                continue
            target_id = neighbor.get("target_pano_id")
            if not isinstance(target_id, str) or not target_id:
                continue
            incoming[target_id] += 1
            target = pano_graph.get(target_id, {})
            edge_records.append(
                {
                    "id": f"{source_id}->{target_id}#{index}",
                    "source": source_id,
                    "target": target_id,
                    "heading": _float_or_none(neighbor.get("geocentric_heading_deg")),
                    "description": neighbor.get("description"),
                    "source_floor": _string_or_unknown(source.get("floor")),
                    "target_floor": _string_or_unknown(target.get("floor")) if isinstance(target, dict) else "unknown",
                    "same_floor": isinstance(target, dict)
                    and _string_or_unknown(source.get("floor")) == _string_or_unknown(target.get("floor")),
                    "dangling": target_id not in pano_graph,
                }
            )

    nodes: list[JsonDict] = []
    for pano_id, node in sorted(pano_graph.items()):
        if not isinstance(node, dict):
            continue
        room_id = mappings.get(pano_id)
        is_mapped = pano_id in mappings
        status = "room" if room_id else ("null" if is_mapped else "unknown")
        room_node = room_graph.get(room_id or "", {}) if isinstance(room_graph, dict) else {}
        neighbors = [n for n in node.get("neighbors", []) if isinstance(n, dict)]
        nodes.append(
            {
                "id": pano_id,
                "pano_id": pano_id,
                "floor": _string_or_unknown(node.get("floor")),
                "lat": _float_or_none(node.get("lat")),
                "lng": _float_or_none(node.get("lng")),
                "room_id": room_id,
                "room_title": room_node.get("title") if isinstance(room_node, dict) else None,
                "room_category": room_node.get("category") if isinstance(room_node, dict) else None,
                "grounding_status": status,
                "grounding_source": sources.get(pano_id),
                "degree_out": len(neighbors),
                "degree_in": int(incoming[pano_id]),
                "color": room_color(room_id, room_color_map) if status == "room" else STATUS_COLORS[status],
            }
        )

    node_by_id = {node["id"]: node for node in nodes}
    for edge in edge_records:
        source = node_by_id.get(edge["source"])
        target = node_by_id.get(edge["target"])
        edge["coordinates"] = _edge_coordinates(source, target)

    floors = sorted({node["floor"] for node in nodes}, key=_floor_sort_key)
    rooms = sorted(
        {
            node["room_id"]
            for node in nodes
            if isinstance(node.get("room_id"), str) and node.get("room_id")
        },
        key=_room_sort_key,
    )
    status_counts = Counter(node["grounding_status"] for node in nodes)
    floor_counts = Counter(node["floor"] for node in nodes)

    return {
        "schema_version": 1,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edge_records),
            "dangling_edge_count": sum(1 for edge in edge_records if edge["dangling"]),
            "floors": [{"floor": floor, "count": floor_counts[floor]} for floor in floors],
            "grounding_status": dict(sorted(status_counts.items())),
            "room_count": len(rooms),
        },
        "floors": floors,
        "rooms": rooms,
        "room_colors": room_color_map,
        "nodes": nodes,
        "edges": edge_records,
    }


def build_geojson(payload: JsonDict, *, feature_type: str) -> JsonDict:
    features: list[JsonDict] = []
    if feature_type in {"nodes", "all"}:
        for node in payload.get("nodes", []):
            if not isinstance(node, dict):
                continue
            lat = node.get("lat")
            lng = node.get("lng")
            if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
                continue
            props = {key: value for key, value in node.items() if key not in {"lat", "lng"}}
            props["feature_type"] = "node"
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lng, lat]},
                    "properties": props,
                }
            )
    if feature_type in {"edges", "all"}:
        for edge in payload.get("edges", []):
            if not isinstance(edge, dict) or not edge.get("coordinates"):
                continue
            props = {key: value for key, value in edge.items() if key != "coordinates"}
            props["feature_type"] = "edge"
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": edge["coordinates"]},
                    "properties": props,
                }
            )
    return {"type": "FeatureCollection", "features": features}


def build_gexf(payload: JsonDict) -> str:
    ET.register_namespace("", "http://www.gexf.net/1.3")
    ns = "http://www.gexf.net/1.3"
    gexf = ET.Element(f"{{{ns}}}gexf", {"version": "1.3"})
    graph = ET.SubElement(gexf, f"{{{ns}}}graph", {"mode": "static", "defaultedgetype": "directed"})
    node_attrs = ET.SubElement(graph, f"{{{ns}}}attributes", {"class": "node"})
    node_attr_keys = [
        ("floor", "string"),
        ("lat", "double"),
        ("lng", "double"),
        ("room_id", "string"),
        ("room_title", "string"),
        ("room_category", "string"),
        ("grounding_status", "string"),
        ("grounding_source", "string"),
        ("degree_out", "integer"),
        ("degree_in", "integer"),
    ]
    for attr_id, attr_type in node_attr_keys:
        ET.SubElement(node_attrs, f"{{{ns}}}attribute", {"id": attr_id, "title": attr_id, "type": attr_type})

    edge_attrs = ET.SubElement(graph, f"{{{ns}}}attributes", {"class": "edge"})
    for attr_id, attr_type in [("heading", "double"), ("source_floor", "string"), ("target_floor", "string")]:
        ET.SubElement(edge_attrs, f"{{{ns}}}attribute", {"id": attr_id, "title": attr_id, "type": attr_type})

    nodes_el = ET.SubElement(graph, f"{{{ns}}}nodes")
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_el = ET.SubElement(nodes_el, f"{{{ns}}}node", {"id": str(node["id"]), "label": str(node["id"])})
        attvalues = ET.SubElement(node_el, f"{{{ns}}}attvalues")
        for key, _ in node_attr_keys:
            value = node.get(key)
            if value is not None:
                ET.SubElement(attvalues, f"{{{ns}}}attvalue", {"for": key, "value": str(value)})

    valid_nodes = {node["id"] for node in payload.get("nodes", []) if isinstance(node, dict)}
    edges_el = ET.SubElement(graph, f"{{{ns}}}edges")
    for index, edge in enumerate(payload.get("edges", [])):
        if not isinstance(edge, dict) or edge.get("dangling"):
            continue
        if edge.get("source") not in valid_nodes or edge.get("target") not in valid_nodes:
            continue
        edge_el = ET.SubElement(
            edges_el,
            f"{{{ns}}}edge",
            {"id": str(index), "source": str(edge["source"]), "target": str(edge["target"])},
        )
        attvalues = ET.SubElement(edge_el, f"{{{ns}}}attvalues")
        for key in ("heading", "source_floor", "target_floor"):
            value = edge.get(key)
            if value is not None:
                ET.SubElement(attvalues, f"{{{ns}}}attvalue", {"for": key, "value": str(value)})

    return _xml_to_string(gexf)


def build_graphml(payload: JsonDict) -> str:
    ns = "http://graphml.graphdrawing.org/xmlns"
    ET.register_namespace("", ns)
    graphml = ET.Element(f"{{{ns}}}graphml")
    key_specs = [
        ("floor", "node", "string"),
        ("lat", "node", "double"),
        ("lng", "node", "double"),
        ("room_id", "node", "string"),
        ("grounding_status", "node", "string"),
        ("degree_out", "node", "int"),
        ("degree_in", "node", "int"),
        ("heading", "edge", "double"),
        ("source_floor", "edge", "string"),
        ("target_floor", "edge", "string"),
    ]
    for key_id, domain, attr_type in key_specs:
        ET.SubElement(
            graphml,
            f"{{{ns}}}key",
            {"id": key_id, "for": domain, "attr.name": key_id, "attr.type": attr_type},
        )
    graph = ET.SubElement(graphml, f"{{{ns}}}graph", {"id": "pano_graph", "edgedefault": "directed"})
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_el = ET.SubElement(graph, f"{{{ns}}}node", {"id": str(node["id"])})
        for key in ("floor", "lat", "lng", "room_id", "grounding_status", "degree_out", "degree_in"):
            value = node.get(key)
            if value is not None:
                data_el = ET.SubElement(node_el, f"{{{ns}}}data", {"key": key})
                data_el.text = str(value)

    valid_nodes = {node["id"] for node in payload.get("nodes", []) if isinstance(node, dict)}
    for index, edge in enumerate(payload.get("edges", [])):
        if not isinstance(edge, dict) or edge.get("dangling"):
            continue
        if edge.get("source") not in valid_nodes or edge.get("target") not in valid_nodes:
            continue
        edge_el = ET.SubElement(
            graph,
            f"{{{ns}}}edge",
            {"id": f"e{index}", "source": str(edge["source"]), "target": str(edge["target"])},
        )
        for key in ("heading", "source_floor", "target_floor"):
            value = edge.get(key)
            if value is not None:
                data_el = ET.SubElement(edge_el, f"{{{ns}}}data", {"key": key})
                data_el.text = str(value)
    return _xml_to_string(graphml)


def build_dot(
    payload: JsonDict,
    *,
    floor: str | None = None,
    room_ids: set[str] | None = None,
    route_pano_ids: list[str] | None = None,
) -> str:
    room_ids = set(room_ids or set())
    route_pano_ids = list(route_pano_ids or [])
    route_set = set(route_pano_ids)
    node_map = {node["id"]: node for node in payload.get("nodes", []) if isinstance(node, dict)}

    def include_node(node: JsonDict) -> bool:
        if floor is not None and node.get("floor") != floor:
            return False
        if room_ids and node.get("room_id") not in room_ids:
            return node.get("id") in route_set
        return True

    included = {node_id for node_id, node in node_map.items() if include_node(node)}
    lines = [
        "digraph pano_graph {",
        '  graph [rankdir=LR, overlap=false, splines=true, fontsize=11, fontname="Helvetica"];',
        '  node [shape=circle, style=filled, width=0.18, height=0.18, fixedsize=true, fontsize=8, fontname="Helvetica"];',
        '  edge [arrowsize=0.35, color="#94a3b8", penwidth=0.8];',
    ]
    for node_id in sorted(included):
        node = node_map[node_id]
        label = node.get("room_id") or node_id[:6]
        color = "#ef4444" if node_id in route_set else node.get("color", "#94a3b8")
        lines.append(f'  "{_dot_escape(node_id)}" [label="{_dot_escape(str(label))}", fillcolor="{color}"];')
    route_edges = set(zip(route_pano_ids, route_pano_ids[1:], strict=False))
    for edge in payload.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = edge.get("source")
        target = edge.get("target")
        if source not in included or target not in included:
            continue
        attrs = []
        if (source, target) in route_edges:
            attrs.extend(['color="#ef4444"', "penwidth=2.2"])
        if edge.get("heading") is not None:
            attrs.append(f'label="{edge["heading"]:.0f}"')
        attr_text = f" [{', '.join(attrs)}]" if attrs else ""
        lines.append(f'  "{_dot_escape(str(source))}" -> "{_dot_escape(str(target))}"{attr_text};')
    lines.append("}")
    return "\n".join(lines) + "\n"


def shortest_pano_path(payload: JsonDict, source_pano_id: str, target_pano_id: str) -> list[str]:
    node_ids = {node["id"] for node in payload.get("nodes", []) if isinstance(node, dict)}
    if source_pano_id not in node_ids or target_pano_id not in node_ids:
        return []
    adjacency: dict[str, list[str]] = {}
    for edge in payload.get("edges", []):
        if not isinstance(edge, dict) or edge.get("dangling"):
            continue
        adjacency.setdefault(str(edge["source"]), []).append(str(edge["target"]))
    queue: deque[str] = deque([source_pano_id])
    parent: dict[str, str | None] = {source_pano_id: None}
    while queue:
        pano_id = queue.popleft()
        if pano_id == target_pano_id:
            break
        for target in adjacency.get(pano_id, []):
            if target in parent:
                continue
            parent[target] = pano_id
            queue.append(target)
    if target_pano_id not in parent:
        return []
    path = []
    cursor: str | None = target_pano_id
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return path


def build_floor_overview_svg(
    payload: JsonDict,
    *,
    floor: str,
    width: int = 1400,
    height: int = 1000,
    route_pano_ids: list[str] | None = None,
) -> str:
    route_pano_ids = list(route_pano_ids or [])
    route_set = set(route_pano_ids)
    nodes = [
        node
        for node in payload.get("nodes", [])
        if isinstance(node, dict)
        and node.get("floor") == floor
        and isinstance(node.get("lat"), (int, float))
        and isinstance(node.get("lng"), (int, float))
    ]
    if not nodes:
        return _empty_svg(width, height, f"No pano nodes for floor {floor}")
    bounds = _bounds(nodes)
    node_map = {node["id"]: node for node in nodes}
    route_edges = set(zip(route_pano_ids, route_pano_ids[1:], strict=False))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="32" y="42" font-family="Helvetica,Arial,sans-serif" font-size="24" fill="#0f172a">Panorama graph floor {escape(floor)}</text>',
        f'<text x="32" y="72" font-family="Helvetica,Arial,sans-serif" font-size="14" fill="#475569">{len(nodes)} nodes</text>',
    ]
    for edge in payload.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = node_map.get(edge.get("source"))
        target = node_map.get(edge.get("target"))
        if not source or not target:
            continue
        x1, y1 = _project(source, bounds, width, height)
        x2, y2 = _project(target, bounds, width, height)
        is_route = (edge.get("source"), edge.get("target")) in route_edges
        color = "#ef4444" if is_route else "#cbd5e1"
        opacity = "0.95" if is_route else "0.28"
        stroke_width = "3.2" if is_route else "1.0"
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{color}" stroke-opacity="{opacity}" stroke-width="{stroke_width}"/>'
        )
    for node in nodes:
        x, y = _project(node, bounds, width, height)
        radius = 5.2 if node["id"] in route_set else 3.4
        color = "#ef4444" if node["id"] in route_set else node.get("color", "#94a3b8")
        lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" fill-opacity="0.9"/>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _edge_coordinates(source: JsonDict | None, target: JsonDict | None) -> list[list[float]] | None:
    if not source or not target:
        return None
    source_lat = source.get("lat")
    source_lng = source.get("lng")
    target_lat = target.get("lat")
    target_lng = target.get("lng")
    if not all(isinstance(value, (int, float)) for value in (source_lat, source_lng, target_lat, target_lng)):
        return None
    return [[source_lng, source_lat], [target_lng, target_lat]]


def _string_or_unknown(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _floor_sort_key(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _room_sort_key(value: str) -> tuple[int, int | str, str]:
    prefix = "Room "
    if value.startswith(prefix):
        suffix = value[len(prefix) :]
        digits = ""
        rest = ""
        for ch in suffix:
            if ch.isdigit() and not rest:
                digits += ch
            else:
                rest += ch
        if digits:
            return (0, int(digits), rest)
    return (1, value, "")


def _generated_room_color(index: int) -> str:
    hue = (index * 137.508) % 360
    saturation = 0.68
    lightness = 0.42
    red, green, blue = _hsl_to_rgb(hue / 360.0, saturation, lightness)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _hsl_to_rgb(hue: float, saturation: float, lightness: float) -> tuple[int, int, int]:
    if saturation == 0:
        value = round(lightness * 255)
        return value, value, value

    def hue_to_rgb(p: float, q: float, t: float) -> float:
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = lightness * (1 + saturation) if lightness < 0.5 else lightness + saturation - lightness * saturation
    p = 2 * lightness - q
    red = hue_to_rgb(p, q, hue + 1 / 3)
    green = hue_to_rgb(p, q, hue)
    blue = hue_to_rgb(p, q, hue - 1 / 3)
    return round(red * 255), round(green * 255), round(blue * 255)


def _xml_to_string(root: ET.Element) -> str:
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)


def _dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _bounds(nodes: list[JsonDict]) -> tuple[float, float, float, float]:
    lats = [float(node["lat"]) for node in nodes]
    lngs = [float(node["lng"]) for node in nodes]
    min_lat = min(lats)
    max_lat = max(lats)
    min_lng = min(lngs)
    max_lng = max(lngs)
    if max_lat == min_lat:
        max_lat += 0.0001
        min_lat -= 0.0001
    if max_lng == min_lng:
        max_lng += 0.0001
        min_lng -= 0.0001
    return min_lat, max_lat, min_lng, max_lng


def _project(node: JsonDict, bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float]:
    min_lat, max_lat, min_lng, max_lng = bounds
    margin_x = width * 0.055
    margin_top = height * 0.1
    margin_bottom = height * 0.055
    x = margin_x + (float(node["lng"]) - min_lng) / (max_lng - min_lng) * (width - 2 * margin_x)
    y = margin_top + (max_lat - float(node["lat"])) / (max_lat - min_lat) * (height - margin_top - margin_bottom)
    return x, y


def _empty_svg(width: int, height: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#f8fafc"/>'
        f'<text x="32" y="48" font-family="Helvetica,Arial,sans-serif" font-size="20" fill="#0f172a">'
        f"{escape(message)}</text></svg>\n"
    )
