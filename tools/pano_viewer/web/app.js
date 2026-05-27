const state = {
  data: null,
  filteredNodes: [],
  filteredNodeIds: new Set(),
  selectedNodeId: null,
  hoverNodeId: null,
  routeNodeIds: [],
  mode: "map",
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  dragStart: null,
  positions: new Map(),
  streetView: null,
  streetViewLoaded: false,
  syncingStreetView: false,
};

const canvas = document.getElementById("graphCanvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

const els = {
  summaryText: document.getElementById("summaryText"),
  floorOptions: document.getElementById("floorOptions"),
  roomOptions: document.getElementById("roomOptions"),
  clearFloorsButton: document.getElementById("clearFloorsButton"),
  clearRoomsButton: document.getElementById("clearRoomsButton"),
  statusSelect: document.getElementById("statusSelect"),
  searchInput: document.getElementById("searchInput"),
  mapModeButton: document.getElementById("mapModeButton"),
  topologyModeButton: document.getElementById("topologyModeButton"),
  routeSourceInput: document.getElementById("routeSourceInput"),
  routeTargetInput: document.getElementById("routeTargetInput"),
  routeButton: document.getElementById("routeButton"),
  routeText: document.getElementById("routeText"),
  detailsList: document.getElementById("detailsList"),
  neighborList: document.getElementById("neighborList"),
  streetViewButton: document.getElementById("streetViewButton"),
  streetViewPane: document.getElementById("streetViewPane"),
  streetViewStatus: document.getElementById("streetViewStatus"),
};

main().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<pre class="load-error">${escapeHtml(String(error.message || error))}</pre>`;
});

async function main() {
  await loadOptionalScript("./.env.js");
  const response = await fetch("./viewer_data.json");
  if (!response.ok) throw new Error(`Failed to load viewer_data.json (${response.status})`);
  state.data = await response.json();
  initControls();
  resizeCanvas();
  applyFilters();
  bindEvents();
  draw();
}

function initControls() {
  const summary = state.data.summary;
  els.summaryText.textContent = `${summary.node_count.toLocaleString()} panos / ${summary.edge_count.toLocaleString()} links / ${state.data.floors.length} floors`;

  fillChecklist(els.floorOptions, state.data.floors.map((floor) => [floor, `Floor ${floor}`]));
  const roomOptions = [["__ungrounded__", "Ungrounded"], ...state.data.rooms.map((room) => [room, room])];
  fillChecklist(els.roomOptions, roomOptions);
}

function fillChecklist(container, options) {
  container.innerHTML = "";
  for (const [value, label] of options) {
    const optionLabel = document.createElement("label");
    optionLabel.className = "filter-option";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = value;
    const labelText = document.createElement("span");
    labelText.textContent = label;
    optionLabel.append(input, labelText);
    container.appendChild(optionLabel);
  }
}

function bindEvents() {
  window.addEventListener("resize", () => {
    resizeCanvas();
    draw();
  });
  for (const el of [els.floorOptions, els.roomOptions, els.statusSelect, els.searchInput]) {
    el.addEventListener("input", () => {
      applyFilters();
      draw();
    });
  }
  els.clearFloorsButton.addEventListener("click", () => clearChecklist(els.floorOptions));
  els.clearRoomsButton.addEventListener("click", () => clearChecklist(els.roomOptions));
  els.mapModeButton.addEventListener("click", () => setMode("map"));
  els.topologyModeButton.addEventListener("click", () => setMode("topology"));
  els.routeButton.addEventListener("click", findRoute);
  els.streetViewButton.addEventListener("click", loadSelectedStreetView);

  canvas.addEventListener("mousedown", (event) => {
    state.dragging = true;
    state.dragStart = { x: event.clientX, y: event.clientY, offsetX: state.offsetX, offsetY: state.offsetY };
  });
  window.addEventListener("mouseup", () => {
    state.dragging = false;
  });
  window.addEventListener("mousemove", (event) => {
    if (state.dragging && state.dragStart) {
      state.offsetX = state.dragStart.offsetX + event.clientX - state.dragStart.x;
      state.offsetY = state.dragStart.offsetY + event.clientY - state.dragStart.y;
      draw();
      return;
    }
    updateHover(event);
  });
  canvas.addEventListener("click", (event) => {
    const node = findNodeAt(event);
    if (node) selectNode(node.id);
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const before = screenToWorld(event.offsetX, event.offsetY);
    const delta = event.deltaY < 0 ? 1.12 : 0.9;
    state.scale = clamp(state.scale * delta, 0.45, 12);
    const after = worldToScreen(before.x, before.y);
    state.offsetX += event.offsetX - after.x;
    state.offsetY += event.offsetY - after.y;
    draw();
  }, { passive: false });
}

function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  els.mapModeButton.classList.toggle("active", mode === "map");
  els.topologyModeButton.classList.toggle("active", mode === "topology");
  buildPositions();
  fitToFilteredNodes();
  draw();
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  fitToFilteredNodes();
}

function applyFilters() {
  const selectedFloors = checkedValues(els.floorOptions);
  const selectedRoomOptions = checkedValues(els.roomOptions);
  const includeUngrounded = selectedRoomOptions.has("__ungrounded__");
  const selectedRooms = new Set([...selectedRoomOptions].filter((value) => value !== "__ungrounded__"));
  const status = els.statusSelect.value;
  const query = els.searchInput.value.trim().toLowerCase();
  state.filteredNodes = state.data.nodes.filter((node) => {
    if (selectedFloors.size && !selectedFloors.has(node.floor)) return false;
    if (status !== "all" && node.grounding_status !== status) return false;
    if (selectedRoomOptions.size) {
      const roomMatches = Boolean(node.room_id && selectedRooms.has(node.room_id));
      const ungroundedMatches = includeUngrounded && !node.room_id;
      if (!roomMatches && !ungroundedMatches) return false;
    }
    if (query) {
      const haystack = [node.id, node.room_id, node.room_title, node.room_category].filter(Boolean).join(" ").toLowerCase();
      if (!haystack.includes(query)) return false;
    }
    return true;
  });
  state.filteredNodeIds = new Set(state.filteredNodes.map((node) => node.id));
  buildPositions();
  fitToFilteredNodes();
  if (state.selectedNodeId && !state.filteredNodeIds.has(state.selectedNodeId)) {
    state.selectedNodeId = null;
    updateDetails();
  }
}

function checkedValues(container) {
  return new Set([...container.querySelectorAll("input[type='checkbox']:checked")].map((input) => input.value));
}

function clearChecklist(container) {
  for (const input of container.querySelectorAll("input[type='checkbox']")) {
    input.checked = false;
  }
  applyFilters();
  draw();
}

function buildPositions() {
  state.positions.clear();
  if (state.mode === "topology") {
    buildTopologyPositions();
    return;
  }
  const nodesWithCoords = state.filteredNodes.filter((node) => Number.isFinite(node.lat) && Number.isFinite(node.lng));
  if (!nodesWithCoords.length) return;
  const minLat = Math.min(...nodesWithCoords.map((node) => node.lat));
  const maxLat = Math.max(...nodesWithCoords.map((node) => node.lat));
  const minLng = Math.min(...nodesWithCoords.map((node) => node.lng));
  const maxLng = Math.max(...nodesWithCoords.map((node) => node.lng));
  const latSpan = Math.max(maxLat - minLat, 0.00001);
  const lngSpan = Math.max(maxLng - minLng, 0.00001);
  const aspect = Math.max(0.4, Math.min(2.4, lngSpan / latSpan));
  for (const node of nodesWithCoords) {
    state.positions.set(node.id, {
      x: ((node.lng - minLng) / lngSpan - 0.5) * 1200 * aspect,
      y: (0.5 - (node.lat - minLat) / latSpan) * 1200,
    });
  }
}

function buildTopologyPositions() {
  const grouped = new Map();
  for (const node of state.filteredNodes) {
    const key = node.room_id || node.grounding_status || "unknown";
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(node);
  }
  const groups = [...grouped.entries()].sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]));
  const groupRadius = Math.max(220, groups.length * 18);
  groups.forEach(([key, nodes], groupIndex) => {
    nodes.sort((a, b) => b.degree_out + b.degree_in - (a.degree_out + a.degree_in) || a.id.localeCompare(b.id));
    const groupAngle = (Math.PI * 2 * groupIndex) / Math.max(groups.length, 1);
    const cx = Math.cos(groupAngle) * groupRadius;
    const cy = Math.sin(groupAngle) * groupRadius;
    const localRadius = Math.max(22, Math.sqrt(nodes.length) * 12);
    nodes.forEach((node, nodeIndex) => {
      const angle = (Math.PI * 2 * nodeIndex) / Math.max(nodes.length, 1);
      const ring = localRadius + Math.floor(nodeIndex / 72) * 30;
      state.positions.set(node.id, {
        x: cx + Math.cos(angle) * ring,
        y: cy + Math.sin(angle) * ring,
        group: key,
      });
    });
  });
}

function fitToFilteredNodes() {
  if (!state.positions.size || !canvas.clientWidth || !canvas.clientHeight) return;
  const positions = [...state.positions.values()];
  const minX = Math.min(...positions.map((point) => point.x));
  const maxX = Math.max(...positions.map((point) => point.x));
  const minY = Math.min(...positions.map((point) => point.y));
  const maxY = Math.max(...positions.map((point) => point.y));
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const scaleX = (canvas.clientWidth - 80) / width;
  const scaleY = (canvas.clientHeight - 80) / height;
  state.scale = clamp(Math.min(scaleX, scaleY), 0.45, 9);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  state.offsetX = canvas.clientWidth / 2 - centerX * state.scale;
  state.offsetY = canvas.clientHeight / 2 - centerY * state.scale;
}

function draw() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.save();
  ctx.translate(state.offsetX, state.offsetY);
  ctx.scale(state.scale, state.scale);
  drawEdges();
  drawNodes();
  ctx.restore();
}

function drawEdges() {
  const routeEdges = new Set(zipPairs(state.routeNodeIds).map(([source, target]) => `${source}->${target}`));
  for (const edge of state.data.edges) {
    if (!state.filteredNodeIds.has(edge.source) || !state.filteredNodeIds.has(edge.target)) continue;
    const source = state.positions.get(edge.source);
    const target = state.positions.get(edge.target);
    if (!source || !target) continue;
    const isRoute = routeEdges.has(`${edge.source}->${edge.target}`);
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = isRoute ? "#ef4444" : "rgba(100,116,139,0.22)";
    ctx.lineWidth = isRoute ? 3.2 / state.scale : 1.0 / state.scale;
    ctx.stroke();
    if (isRoute) drawArrow(source, target);
  }
}

function drawArrow(source, target) {
  const angle = Math.atan2(target.y - source.y, target.x - source.x);
  const size = 8 / state.scale;
  const x = target.x - Math.cos(angle) * 7 / state.scale;
  const y = target.y - Math.sin(angle) * 7 / state.scale;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - Math.cos(angle - 0.45) * size, y - Math.sin(angle - 0.45) * size);
  ctx.lineTo(x - Math.cos(angle + 0.45) * size, y - Math.sin(angle + 0.45) * size);
  ctx.closePath();
  ctx.fillStyle = "#ef4444";
  ctx.fill();
}

function drawNodes() {
  const routeSet = new Set(state.routeNodeIds);
  for (const node of state.filteredNodes) {
    const point = state.positions.get(node.id);
    if (!point) continue;
    const isSelected = node.id === state.selectedNodeId;
    const isHover = node.id === state.hoverNodeId;
    const isRoute = routeSet.has(node.id);
    const radius = (isSelected || isHover || isRoute ? 7 : 4.2) / state.scale;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isRoute ? "#ef4444" : node.color || "#94a3b8";
    ctx.fill();
    if (isSelected || isHover) {
      ctx.lineWidth = 2 / state.scale;
      ctx.strokeStyle = "#0f172a";
      ctx.stroke();
    }
  }
}

function updateHover(event) {
  const node = findNodeAt(event);
  state.hoverNodeId = node ? node.id : null;
  if (node) {
    tooltip.hidden = false;
    tooltip.style.left = `${event.clientX - canvas.parentElement.getBoundingClientRect().left + 12}px`;
    tooltip.style.top = `${event.clientY - canvas.parentElement.getBoundingClientRect().top + 12}px`;
    tooltip.innerHTML = `<strong>${escapeHtml(node.id)}</strong><br>${escapeHtml(node.room_id || node.grounding_status)}<br>floor ${escapeHtml(node.floor)}`;
  } else {
    tooltip.hidden = true;
  }
  draw();
}

function findNodeAt(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const world = screenToWorld(x, y);
  let best = null;
  let bestDistance = Infinity;
  const hitRadius = 10 / state.scale;
  for (const node of state.filteredNodes) {
    const point = state.positions.get(node.id);
    if (!point) continue;
    const distance = Math.hypot(point.x - world.x, point.y - world.y);
    if (distance <= hitRadius && distance < bestDistance) {
      best = node;
      bestDistance = distance;
    }
  }
  return best;
}

function selectNode(nodeId) {
  state.selectedNodeId = nodeId;
  const node = nodeById(nodeId);
  if (node) {
    els.routeSourceInput.value ||= node.id;
    if (els.routeSourceInput.value && els.routeSourceInput.value !== node.id) {
      els.routeTargetInput.value = node.id;
    }
  }
  updateDetails();
  if (state.streetViewLoaded) updateStreetView();
  draw();
}

function updateDetails() {
  const node = nodeById(state.selectedNodeId);
  els.detailsList.innerHTML = "";
  els.neighborList.innerHTML = "";
  if (!node) {
    els.detailsList.innerHTML = "<dt>Status</dt><dd>No pano selected</dd>";
    return;
  }
  const rows = [
    ["Pano ID", node.id],
    ["Floor", node.floor],
    ["Room", node.room_id || node.grounding_status],
    ["Title", node.room_title || "-"],
    ["Category", node.room_category || "-"],
    ["Degree", `${node.degree_in} in / ${node.degree_out} out`],
    ["Lat/Lng", `${formatNumber(node.lat)}, ${formatNumber(node.lng)}`],
    ["Source", node.grounding_source || "-"],
  ];
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    els.detailsList.append(dt, dd);
  }
  const outgoing = state.data.edges.filter((edge) => edge.source === node.id);
  for (const edge of outgoing) {
    const li = document.createElement("li");
    const target = nodeById(edge.target);
    li.textContent = `${edge.target} (${target?.room_id || target?.grounding_status || "dangling"}) heading ${formatNumber(edge.heading)} deg`;
    els.neighborList.appendChild(li);
  }
}

function findRoute() {
  const source = els.routeSourceInput.value.trim();
  const target = els.routeTargetInput.value.trim();
  const path = shortestPath(source, target);
  state.routeNodeIds = path;
  if (!source || !target) {
    els.routeText.textContent = "Enter source and target pano ids.";
  } else if (!path.length) {
    els.routeText.textContent = "No path found.";
  } else {
    els.routeText.textContent = `${path.length} panos: ${path.slice(0, 5).join(" -> ")}${path.length > 5 ? " ..." : ""}`;
    if (path.some((id) => !state.filteredNodeIds.has(id))) {
      clearChecklistWithoutRedraw(els.floorOptions);
      clearChecklistWithoutRedraw(els.roomOptions);
      els.statusSelect.value = "all";
      els.searchInput.value = "";
      applyFilters();
    }
    selectNode(path[0]);
  }
  draw();
}

function clearChecklistWithoutRedraw(container) {
  for (const input of container.querySelectorAll("input[type='checkbox']")) {
    input.checked = false;
  }
}

function shortestPath(source, target) {
  if (!source || !target || !nodeById(source) || !nodeById(target)) return [];
  const adjacency = new Map();
  for (const edge of state.data.edges) {
    if (edge.dangling) continue;
    if (!adjacency.has(edge.source)) adjacency.set(edge.source, []);
    adjacency.get(edge.source).push(edge.target);
  }
  const queue = [source];
  const parent = new Map([[source, null]]);
  for (let index = 0; index < queue.length; index++) {
    const current = queue[index];
    if (current === target) break;
    for (const next of adjacency.get(current) || []) {
      if (parent.has(next)) continue;
      parent.set(next, current);
      queue.push(next);
    }
  }
  if (!parent.has(target)) return [];
  const path = [];
  let cursor = target;
  while (cursor) {
    path.push(cursor);
    cursor = parent.get(cursor);
  }
  return path.reverse();
}

async function loadSelectedStreetView() {
  state.streetViewLoaded = true;
  const key = window.GMAPS_API_KEY;
  if (!key) {
    els.streetViewStatus.textContent = "Set window.GMAPS_API_KEY in .env.js to enable Street View.";
    return;
  }
  if (!window.google?.maps?.StreetViewPanorama) {
    await loadGoogleMaps(key);
  }
  if (!state.streetView) {
    state.streetView = new google.maps.StreetViewPanorama(els.streetViewPane, {
      disableDefaultUI: false,
      clickToGo: true,
      linksControl: true,
      panControl: true,
      zoomControl: true,
      addressControl: false,
      showRoadLabels: false,
    });
    state.streetView.addListener("pano_changed", syncSelectedNodeFromStreetView);
  }
  updateStreetView();
}

function updateStreetView() {
  const node = nodeById(state.selectedNodeId);
  if (!node || !state.streetView || !window.google?.maps) return;
  state.syncingStreetView = true;
  state.streetView.setPano(node.id);
  state.streetView.setPov({ heading: 330, pitch: 0 });
  state.streetView.setZoom(1);
  state.syncingStreetView = false;
  els.streetViewStatus.textContent = node.id;
}

function syncSelectedNodeFromStreetView() {
  if (state.syncingStreetView || !state.streetView) return;
  const panoId = state.streetView.getPano();
  if (!panoId || panoId === state.selectedNodeId) return;
  const node = nodeById(panoId);
  if (!node) {
    els.streetViewStatus.textContent = `${panoId} (not in current pano graph)`;
    return;
  }
  state.selectedNodeId = panoId;
  if (!state.filteredNodeIds.has(panoId)) {
    clearChecklistWithoutRedraw(els.floorOptions);
    clearChecklistWithoutRedraw(els.roomOptions);
    els.statusSelect.value = "all";
    els.searchInput.value = "";
    applyFilters();
  }
  updateDetails();
  els.streetViewStatus.textContent = panoId;
  draw();
}

function loadGoogleMaps(apiKey) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.async = true;
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}&v=weekly`;
    script.onload = resolve;
    script.onerror = () => reject(new Error("Failed to load Google Maps JavaScript API."));
    document.head.appendChild(script);
  });
}

function loadOptionalScript(src) {
  return new Promise((resolve) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = resolve;
    document.head.appendChild(script);
  });
}

function nodeById(nodeId) {
  if (!nodeId) return null;
  return state.data.nodes.find((node) => node.id === nodeId) || null;
}

function screenToWorld(x, y) {
  return {
    x: (x - state.offsetX) / state.scale,
    y: (y - state.offsetY) / state.scale,
  };
}

function worldToScreen(x, y) {
  return {
    x: x * state.scale + state.offsetX,
    y: y * state.scale + state.offsetY,
  };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function zipPairs(values) {
  const pairs = [];
  for (let index = 0; index + 1 < values.length; index++) {
    pairs.push([values[index], values[index + 1]]);
  }
  return pairs;
}

function formatNumber(value) {
  return Number.isFinite(value) ? value.toFixed(6).replace(/0+$/, "").replace(/\.$/, "") : "-";
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}
