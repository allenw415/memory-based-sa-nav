const state = {
  data: null,
  nodeMap: new Map(),
  filteredNodes: [],
  filteredNodeIds: new Set(),
  selectedNodeId: null,
  hoverNodeId: null,
  routeNodeIds: [],
  trajectory: null,
  trajectoryFileName: null,
  trajectoryIndex: 0,
  playing: false,
  playbackTimer: null,
  mode: "map",
  scale: 1,
  offsetX: 0,
  offsetY: 0,
  dragging: false,
  dragStart: null,
  positions: new Map(),
  streetView: null,
  streetViewGate: null,
  syncingStreetView: false,
  requestedStreetViewPanoId: null,
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
  trajectoryDropZone: document.getElementById("trajectoryDropZone"),
  trajectoryFileInput: document.getElementById("trajectoryFileInput"),
  trajectoryStatus: document.getElementById("trajectoryStatus"),
  clearTrajectoryButton: document.getElementById("clearTrajectoryButton"),
  resetButton: document.getElementById("resetButton"),
  previousButton: document.getElementById("previousButton"),
  playButton: document.getElementById("playButton"),
  nextButton: document.getElementById("nextButton"),
  timelineInput: document.getElementById("timelineInput"),
  timelineText: document.getElementById("timelineText"),
  playbackSpeedSelect: document.getElementById("playbackSpeedSelect"),
  trajectoryDetails: document.getElementById("trajectoryDetails"),
  waypointList: document.getElementById("waypointList"),
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
  if (!window.PanoTrajectory) throw new Error("trajectory.js did not load.");
  await loadOptionalScript("./.env.js");
  const response = await fetch("./viewer_data.json");
  if (!response.ok) throw new Error(`Failed to load viewer_data.json (${response.status})`);
  state.data = await response.json();
  state.nodeMap = new Map(state.data.nodes.map((node) => [node.id, node]));
  state.streetViewGate = window.PanoTrajectory.createManualSyncGate({
    load: initializeStreetView,
    sync: syncStreetViewTarget,
  });
  initControls();
  resizeCanvas();
  applyFilters();
  bindEvents();
  updateDetails();
  updateTrajectoryPanels();
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

  els.trajectoryFileInput.addEventListener("change", () => {
    const file = els.trajectoryFileInput.files && els.trajectoryFileInput.files[0];
    if (file) loadTrajectoryFile(file);
  });
  els.clearTrajectoryButton.addEventListener("click", clearTrajectory);
  els.resetButton.addEventListener("click", () => setTrajectoryIndex(0));
  els.previousButton.addEventListener("click", () => setTrajectoryIndex(state.trajectoryIndex - 1));
  els.nextButton.addEventListener("click", () => setTrajectoryIndex(state.trajectoryIndex + 1));
  els.playButton.addEventListener("click", togglePlayback);
  els.timelineInput.addEventListener("input", () => setTrajectoryIndex(Number(els.timelineInput.value)));
  els.playbackSpeedSelect.addEventListener("change", restartPlaybackTimer);
  for (const eventName of ["dragenter", "dragover"]) {
    els.trajectoryDropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.trajectoryDropZone.classList.add("dragging");
    });
  }
  for (const eventName of ["dragleave", "drop"]) {
    els.trajectoryDropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      els.trajectoryDropZone.classList.remove("dragging");
    });
  }
  els.trajectoryDropZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) loadTrajectoryFile(file);
  });

  canvas.addEventListener("mousedown", (event) => {
    state.dragging = true;
    state.dragStart = { x: event.clientX, y: event.clientY, offsetX: state.offsetX, offsetY: state.offsetY };
  });
  window.addEventListener("mouseup", () => {
    state.dragging = false;
  });
  window.addEventListener("mousemove", (event) => {
    if (!state.dragging || !state.dragStart) return;
    state.offsetX = state.dragStart.offsetX + event.clientX - state.dragStart.x;
    state.offsetY = state.dragStart.offsetY + event.clientY - state.dragStart.y;
    draw();
  });
  canvas.addEventListener("mousemove", updateHover);
  canvas.addEventListener("mouseleave", () => {
    state.hoverNodeId = null;
    tooltip.hidden = true;
    draw();
  });
  canvas.addEventListener("click", (event) => {
    const node = findNodeAt(event);
    if (node) selectNode(node.id, true);
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

async function loadTrajectoryFile(file) {
  stopPlayback();
  els.trajectoryStatus.classList.remove("error");
  els.trajectoryStatus.textContent = `Reading ${file.name}...`;
  try {
    const payload = JSON.parse(await file.text());
    const trajectory = window.PanoTrajectory.parseTrajectory(payload, state.data);
    setTrajectory(trajectory, file.name);
  } catch (error) {
    console.error(error);
    els.trajectoryStatus.classList.add("error");
    els.trajectoryStatus.textContent = String(error.message || error);
  } finally {
    els.trajectoryFileInput.value = "";
  }
}

function setTrajectory(trajectory, fileName) {
  state.trajectory = trajectory;
  state.trajectoryFileName = fileName;
  state.trajectoryIndex = 0;
  state.routeNodeIds = [];
  clearAllFilters();
  applyFilters();
  const controls = [
    els.clearTrajectoryButton,
    els.resetButton,
    els.previousButton,
    els.playButton,
    els.nextButton,
    els.timelineInput,
    els.playbackSpeedSelect,
  ];
  controls.forEach((control) => { control.disabled = false; });
  els.timelineInput.max = String(Math.max(0, trajectory.frames.length - 1));
  els.trajectoryStatus.classList.remove("error");
  els.trajectoryStatus.textContent = `${fileName}: ${trajectory.movements.length} moves, ${trajectory.boundaries.length} room boundaries, success=${trajectory.success}.`;
  showTrajectoryFrame();
  fitToNodeIds(trajectory.panoPath);
  draw();
}

function clearTrajectory() {
  stopPlayback();
  state.trajectory = null;
  state.trajectoryFileName = null;
  state.trajectoryIndex = 0;
  for (const control of [
    els.clearTrajectoryButton,
    els.resetButton,
    els.previousButton,
    els.playButton,
    els.nextButton,
    els.timelineInput,
    els.playbackSpeedSelect,
  ]) {
    control.disabled = true;
  }
  els.timelineInput.max = "0";
  els.timelineInput.value = "0";
  els.trajectoryStatus.classList.remove("error");
  els.trajectoryStatus.textContent = "No trajectory loaded.";
  updateTrajectoryPanels();
  fitToFilteredNodes();
  draw();
}

function togglePlayback() {
  if (!state.trajectory) return;
  if (state.playing) {
    stopPlayback();
    return;
  }
  if (state.trajectoryIndex >= state.trajectory.frames.length - 1) {
    setTrajectoryIndex(0, false);
  }
  state.playing = true;
  updatePlaybackControls();
  schedulePlaybackStep();
}

function stopPlayback() {
  state.playing = false;
  if (state.playbackTimer !== null) {
    window.clearTimeout(state.playbackTimer);
    state.playbackTimer = null;
  }
  updatePlaybackControls();
}

function restartPlaybackTimer() {
  if (!state.playing) return;
  if (state.playbackTimer !== null) window.clearTimeout(state.playbackTimer);
  schedulePlaybackStep();
}

function schedulePlaybackStep() {
  if (!state.playing || !state.trajectory) return;
  const speed = Math.max(0.1, Number(els.playbackSpeedSelect.value) || 1);
  state.playbackTimer = window.setTimeout(() => {
    state.playbackTimer = null;
    if (state.trajectoryIndex >= state.trajectory.frames.length - 1) {
      stopPlayback();
      return;
    }
    setTrajectoryIndex(state.trajectoryIndex + 1, false);
    if (state.trajectoryIndex >= state.trajectory.frames.length - 1) {
      stopPlayback();
    } else {
      schedulePlaybackStep();
    }
  }, 1200 / speed);
}

function setTrajectoryIndex(index, pause = true) {
  if (!state.trajectory) return;
  if (pause) stopPlayback();
  state.trajectoryIndex = clamp(Math.round(index), 0, state.trajectory.frames.length - 1);
  showTrajectoryFrame();
}

function showTrajectoryFrame() {
  const frame = currentTrajectoryFrame();
  if (!frame) return;
  state.selectedNodeId = frame.panoId;
  updateDetails();
  updateTrajectoryPanels();
  requestStreetViewSync();
  draw();
}

function updateTrajectoryPanels() {
  updatePlaybackControls();
  updateTrajectoryDetails();
  updateWaypointList();
}

function updatePlaybackControls() {
  if (!state.trajectory) {
    els.timelineText.textContent = "Step 0 / 0";
    els.playButton.textContent = "Play";
    return;
  }
  const moveCount = state.trajectory.movements.length;
  els.timelineInput.value = String(state.trajectoryIndex);
  els.timelineText.textContent = `Step ${state.trajectoryIndex} / ${moveCount}`;
  els.playButton.textContent = state.playing ? "Pause" : "Play";
  els.previousButton.disabled = state.trajectoryIndex <= 0;
  els.nextButton.disabled = state.trajectoryIndex >= state.trajectory.frames.length - 1;
}

function updateTrajectoryDetails() {
  els.trajectoryDetails.innerHTML = "";
  const frame = currentTrajectoryFrame();
  if (!state.trajectory || !frame) {
    renderDefinitionRows(els.trajectoryDetails, [["Status", "No trajectory loaded"]]);
    return;
  }
  const rows = [
    ["File", state.trajectoryFileName || "-"],
    ["Step", `${frame.index} / ${state.trajectory.movements.length}`],
    ["Pano ID", frame.panoId],
    ["Room", frame.roomId || "-"],
    ["Round", frame.roundIndex === null ? "-" : frame.roundIndex],
    ["Subgoal", frame.subgoalRoomId || "-"],
    ["Target", frame.activeTargetRoomId || state.trajectory.targetRoomId || "-"],
    ["Heading", frame.heading === null ? "-" : `${formatNumber(frame.heading)} deg`],
    ["Similarity", frame.similarity === null ? "-" : formatNumber(frame.similarity)],
    ["Margin", frame.margin === null ? "-" : formatNumber(frame.margin)],
    ["Image goal", frame.imageGoalLabel || "-"],
    ["Episode", state.trajectory.success ? "success" : state.trajectory.reason || "not successful"],
  ];
  if (frame.boundary) {
    rows.push(["Boundary", `${frame.boundary.fromRoomId || "-"} -> ${frame.boundary.toRoomId || "-"}`]);
  }
  renderDefinitionRows(els.trajectoryDetails, rows);
}

function updateWaypointList() {
  els.waypointList.innerHTML = "";
  if (!state.trajectory) {
    const li = document.createElement("li");
    li.textContent = "No trajectory loaded.";
    els.waypointList.appendChild(li);
    return;
  }
  if (!state.trajectory.boundaries.length && !state.trajectory.waypointRoomIds.length) {
    const li = document.createElement("li");
    li.textContent = "No waypoint or room boundary recorded.";
    els.waypointList.appendChild(li);
    return;
  }
  for (const boundary of state.trajectory.boundaries) {
    const li = document.createElement("li");
    const label = boundary.kind === "waypoint" ? "Waypoint" : "Room transition";
    li.textContent = `${label} at step ${boundary.pathIndex}: ${boundary.fromRoomId || "-"} -> ${boundary.toRoomId || "-"}`;
    li.classList.toggle("reached", boundary.pathIndex <= state.trajectoryIndex);
    li.classList.toggle("current", boundary.pathIndex === state.trajectoryIndex);
    els.waypointList.appendChild(li);
  }
  const representedRooms = new Set(state.trajectory.boundaries.map((boundary) => boundary.toRoomId));
  for (const roomId of state.trajectory.waypointRoomIds) {
    if (representedRooms.has(roomId)) continue;
    const li = document.createElement("li");
    const complete = state.trajectory.completedWaypointIds.includes(roomId);
    li.textContent = `Waypoint ${roomId}: ${complete ? "completed" : "not reached"}`;
    li.classList.toggle("reached", complete);
    els.waypointList.appendChild(li);
  }
}

function setMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;
  els.mapModeButton.classList.toggle("active", mode === "map");
  els.topologyModeButton.classList.toggle("active", mode === "topology");
  buildPositions();
  if (state.trajectory) fitToNodeIds(state.trajectory.panoPath);
  else fitToFilteredNodes();
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
  if (state.trajectory) fitToNodeIds(state.trajectory.panoPath);
  else fitToFilteredNodes();
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
  for (const input of container.querySelectorAll("input[type='checkbox']")) input.checked = false;
  applyFilters();
  draw();
}

function clearAllFilters() {
  clearChecklistWithoutRedraw(els.floorOptions);
  clearChecklistWithoutRedraw(els.roomOptions);
  els.statusSelect.value = "all";
  els.searchInput.value = "";
}

function clearChecklistWithoutRedraw(container) {
  for (const input of container.querySelectorAll("input[type='checkbox']")) input.checked = false;
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
  fitToPositions([...state.positions.values()]);
}

function fitToNodeIds(nodeIds) {
  const positions = nodeIds.map((nodeId) => state.positions.get(nodeId)).filter(Boolean);
  if (positions.length) fitToPositions(positions, 120);
}

function fitToPositions(positions, padding = 80) {
  if (!positions.length || !canvas.clientWidth || !canvas.clientHeight) return;
  const minX = Math.min(...positions.map((point) => point.x));
  const maxX = Math.max(...positions.map((point) => point.x));
  const minY = Math.min(...positions.map((point) => point.y));
  const maxY = Math.max(...positions.map((point) => point.y));
  const width = Math.max(12, maxX - minX);
  const height = Math.max(12, maxY - minY);
  const scaleX = (canvas.clientWidth - padding) / width;
  const scaleY = (canvas.clientHeight - padding) / height;
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
  drawBaseEdges();
  drawPath(state.routeNodeIds, "#ef4444", 3.2, true);
  if (state.trajectory) {
    drawPath(state.trajectory.panoPath, "#93c5fd", 4, true);
    drawPath(state.trajectory.panoPath.slice(0, state.trajectoryIndex + 1), "#1d4ed8", 6, true);
  }
  drawNodes();
  drawBoundaryMarkers();
  ctx.restore();
}

function drawBaseEdges() {
  for (const edge of state.data.edges) {
    if (!state.filteredNodeIds.has(edge.source) || !state.filteredNodeIds.has(edge.target)) continue;
    const source = state.positions.get(edge.source);
    const target = state.positions.get(edge.target);
    if (!source || !target) continue;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = "rgba(100,116,139,0.2)";
    ctx.lineWidth = 1 / state.scale;
    ctx.stroke();
  }
}

function drawPath(nodeIds, color, width, arrows) {
  for (const [sourceId, targetId] of zipPairs(nodeIds)) {
    if (!state.filteredNodeIds.has(sourceId) || !state.filteredNodeIds.has(targetId)) continue;
    const source = state.positions.get(sourceId);
    const target = state.positions.get(targetId);
    if (!source || !target) continue;
    ctx.beginPath();
    ctx.moveTo(source.x, source.y);
    ctx.lineTo(target.x, target.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = width / state.scale;
    ctx.stroke();
    if (arrows) drawArrow(source, target, color);
  }
}

function drawArrow(source, target, color) {
  const angle = Math.atan2(target.y - source.y, target.x - source.x);
  const size = 8 / state.scale;
  const x = target.x - Math.cos(angle) * 7 / state.scale;
  const y = target.y - Math.sin(angle) * 7 / state.scale;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x - Math.cos(angle - 0.45) * size, y - Math.sin(angle - 0.45) * size);
  ctx.lineTo(x - Math.cos(angle + 0.45) * size, y - Math.sin(angle + 0.45) * size);
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

function drawNodes() {
  const routeSet = new Set(state.routeNodeIds);
  const trajectorySet = new Set(state.trajectory ? state.trajectory.panoPath : []);
  const traversedSet = new Set(state.trajectory ? state.trajectory.panoPath.slice(0, state.trajectoryIndex + 1) : []);
  const currentPanoId = currentTrajectoryFrame() ? currentTrajectoryFrame().panoId : null;
  for (const node of state.filteredNodes) {
    const point = state.positions.get(node.id);
    if (!point) continue;
    const isSelected = node.id === state.selectedNodeId;
    const isHover = node.id === state.hoverNodeId;
    const isCurrent = node.id === currentPanoId;
    const isTrajectory = trajectorySet.has(node.id);
    const isTraversed = traversedSet.has(node.id);
    const radius = (isCurrent ? 9 : isSelected || isHover || isTrajectory || routeSet.has(node.id) ? 7 : 4.2) / state.scale;
    ctx.beginPath();
    ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = isCurrent
      ? "#f59e0b"
      : isTraversed
        ? "#1d4ed8"
        : isTrajectory
          ? "#93c5fd"
          : routeSet.has(node.id)
            ? "#ef4444"
            : node.color || "#94a3b8";
    ctx.fill();
    if (isCurrent || isSelected || isHover) {
      ctx.lineWidth = (isCurrent ? 3 : 2) / state.scale;
      ctx.strokeStyle = isCurrent ? "#7c2d12" : "#0f172a";
      ctx.stroke();
    }
  }
}

function drawBoundaryMarkers() {
  if (!state.trajectory) return;
  for (const boundary of state.trajectory.boundaries) {
    if (!state.filteredNodeIds.has(boundary.panoId)) continue;
    const point = state.positions.get(boundary.panoId);
    if (!point) continue;
    const size = 11 / state.scale;
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = boundary.pathIndex <= state.trajectoryIndex ? "#0f766e" : "#5eead4";
    ctx.strokeStyle = "#134e4a";
    ctx.lineWidth = 2 / state.scale;
    ctx.fillRect(-size / 2, -size / 2, size, size);
    ctx.strokeRect(-size / 2, -size / 2, size, size);
    ctx.restore();
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
  const world = screenToWorld(event.clientX - rect.left, event.clientY - rect.top);
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

function selectNode(nodeId, updateRouteInputs = false) {
  state.selectedNodeId = nodeId;
  const node = nodeById(nodeId);
  if (node && updateRouteInputs) {
    els.routeSourceInput.value ||= node.id;
    if (els.routeSourceInput.value && els.routeSourceInput.value !== node.id) {
      els.routeTargetInput.value = node.id;
    }
  }
  updateDetails();
  requestStreetViewSync();
  draw();
}

function updateDetails() {
  const node = nodeById(state.selectedNodeId);
  els.detailsList.innerHTML = "";
  els.neighborList.innerHTML = "";
  if (!node) {
    renderDefinitionRows(els.detailsList, [["Status", "No pano selected"]]);
    return;
  }
  renderDefinitionRows(els.detailsList, [
    ["Pano ID", node.id],
    ["Floor", node.floor],
    ["Room", node.room_id || node.grounding_status],
    ["Title", node.room_title || "-"],
    ["Category", node.room_category || "-"],
    ["Degree", `${node.degree_in} in / ${node.degree_out} out`],
    ["Lat/Lng", `${formatNumber(node.lat)}, ${formatNumber(node.lng)}`],
    ["Source", node.grounding_source || "-"],
  ]);
  const outgoing = state.data.edges.filter((edge) => edge.source === node.id);
  for (const edge of outgoing) {
    const li = document.createElement("li");
    const target = nodeById(edge.target);
    li.textContent = `${edge.target} (${target?.room_id || target?.grounding_status || "dangling"}) heading ${formatNumber(edge.heading)} deg`;
    els.neighborList.appendChild(li);
  }
}

function renderDefinitionRows(container, rows) {
  for (const [key, value] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = value;
    container.append(dt, dd);
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
      clearAllFilters();
      applyFilters();
    }
    selectNode(path[0]);
    fitToNodeIds(path);
  }
  draw();
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
  els.streetViewButton.disabled = true;
  els.streetViewStatus.classList.remove("error");
  els.streetViewStatus.textContent = "Loading Google Street View...";
  try {
    await state.streetViewGate.load();
    els.streetViewButton.textContent = "Loaded";
    els.streetViewStatus.textContent = state.selectedNodeId || "Street View loaded. Select a pano.";
  } catch (error) {
    console.error(error);
    els.streetViewButton.disabled = false;
    els.streetViewStatus.classList.add("error");
    els.streetViewStatus.textContent = String(error.message || error);
  }
}

async function initializeStreetView() {
  const key = window.GMAPS_API_KEY;
  if (!key) throw new Error("Set GMAPS_KEY and rerun the viewer export to enable Street View.");
  if (!window.google?.maps?.StreetViewPanorama) await loadGoogleMaps(key);
  if (state.streetView) return;
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

function requestStreetViewSync() {
  if (!state.streetViewGate) return;
  const node = nodeById(state.selectedNodeId);
  if (!node) return;
  const frame = currentTrajectoryFrame();
  const heading = frame && frame.panoId === node.id && Number.isFinite(frame.heading) ? frame.heading : 330;
  state.streetViewGate.update({ panoId: node.id, heading });
}

function syncStreetViewTarget(target) {
  if (!state.streetView || !target) return;
  state.syncingStreetView = true;
  state.requestedStreetViewPanoId = target.panoId;
  state.streetView.setPano(target.panoId);
  state.streetView.setPov({ heading: target.heading, pitch: 0 });
  state.streetView.setZoom(1);
  window.setTimeout(() => { state.syncingStreetView = false; }, 0);
  els.streetViewStatus.classList.remove("error");
  els.streetViewStatus.textContent = `${target.panoId} / heading ${formatNumber(target.heading)} deg`;
}

function syncSelectedNodeFromStreetView() {
  if (!state.streetView) return;
  const panoId = state.streetView.getPano();
  if (!panoId || panoId === state.selectedNodeId || panoId === state.requestedStreetViewPanoId) return;
  if (state.syncingStreetView) return;
  const node = nodeById(panoId);
  if (!node) {
    els.streetViewStatus.textContent = `${panoId} (not in current pano graph)`;
    return;
  }
  state.selectedNodeId = panoId;
  if (!state.filteredNodeIds.has(panoId)) {
    clearAllFilters();
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

function currentTrajectoryFrame() {
  if (!state.trajectory) return null;
  return state.trajectory.frames[state.trajectoryIndex] || null;
}

function nodeById(nodeId) {
  return nodeId ? state.nodeMap.get(nodeId) || null : null;
}

function screenToWorld(x, y) {
  return { x: (x - state.offsetX) / state.scale, y: (y - state.offsetY) / state.scale };
}

function worldToScreen(x, y) {
  return { x: x * state.scale + state.offsetX, y: y * state.scale + state.offsetY };
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function zipPairs(values) {
  const pairs = [];
  for (let index = 0; index + 1 < values.length; index++) pairs.push([values[index], values[index + 1]]);
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
