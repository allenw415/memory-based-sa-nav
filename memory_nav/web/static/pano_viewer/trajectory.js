(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.PanoTrajectory = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function parseTrajectory(payload, graphData) {
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new Error("Trajectory JSON must be an object.");
    }
    const graph = buildGraphIndex(graphData);
    const panoPath = requireStringArray(payload.pano_path, "pano_path");
    if (!panoPath.length) {
      throw new Error("pano_path must contain at least one pano id.");
    }
    for (const panoId of panoPath) {
      if (!graph.nodes.has(panoId)) {
        throw new Error(`Trajectory references unknown pano: ${panoId}`);
      }
    }

    const rounds = Array.isArray(payload.rounds) ? payload.rounds : [];
    const movements = [];
    rounds.forEach((round, roundPosition) => {
      if (!round || typeof round !== "object" || Array.isArray(round)) {
        throw new Error(`rounds[${roundPosition}] must be an object.`);
      }
      const roundIndex = finiteNumber(round.round_index, roundPosition);
      const steps = Array.isArray(round.movement_steps) ? round.movement_steps : [];
      steps.forEach((step, localIndex) => {
        if (!step || typeof step !== "object" || Array.isArray(step)) {
          throw new Error(`rounds[${roundPosition}].movement_steps[${localIndex}] must be an object.`);
        }
        movements.push({ raw: step, round, roundIndex, localIndex });
      });
    });

    const expectedMovementCount = Math.max(0, panoPath.length - 1);
    if (movements.length !== expectedMovementCount) {
      throw new Error(
        `pano_path contains ${expectedMovementCount} moves, but rounds contain ${movements.length} movement steps.`,
      );
    }

    const normalizedMovements = movements.map((entry, index) => {
      const source = panoPath[index];
      const target = panoPath[index + 1];
      const rawSource = stringOrNull(entry.raw.current_pano_id);
      const rawTarget = stringOrNull(entry.raw.next_pano_id);
      if (rawSource !== source || rawTarget !== target) {
        throw new Error(
          `Movement ${index} does not match pano_path: expected ${source} -> ${target}, got ${rawSource || "-"} -> ${rawTarget || "-"}.`,
        );
      }
      const edge = graph.edges.get(edgeKey(source, target));
      if (!edge) {
        throw new Error(`Trajectory move is not a legal graph edge: ${source} -> ${target}`);
      }
      const selectedHeading = finiteNumber(entry.raw.selected_action_heading, null);
      const edgeHeading = finiteNumber(edge.heading, null);
      const actionHeading = selectedHeading === null ? edgeHeading : selectedHeading;
      if (actionHeading === null) {
        throw new Error(`Movement ${index} has no action heading and its graph edge has no heading.`);
      }
      return {
        index,
        source,
        target,
        actionHeading: normalizeHeading(actionHeading),
        selectedActionHeading: selectedHeading === null ? null : normalizeHeading(selectedHeading),
        edgeHeading: edgeHeading === null ? null : normalizeHeading(edgeHeading),
        headingSource: selectedHeading === null ? "graph_edge" : "trajectory",
        roundIndex: entry.roundIndex,
        localStepIndex: finiteNumber(entry.raw.local_step_index, entry.localIndex),
        currentRoomId: stringOrNull(entry.raw.current_room_id),
        nextRoomId: stringOrNull(entry.raw.next_room_id),
        subgoalRoomId:
          stringOrNull(entry.raw.subgoal_room_id) || stringOrNull(entry.round.subgoal_room_id),
        activeTargetRoomId:
          stringOrNull(entry.raw.active_target_room_id) ||
          stringOrNull(entry.round.active_target_room_id),
        similarity: finiteNumber(entry.raw.similarity, null),
        margin: finiteNumber(entry.raw.margin, null),
        imageGoalLabel: stringOrNull(entry.raw.image_goal_label),
        raw: entry.raw,
      };
    });

    const waypointRoomIds = optionalStringArray(payload.waypoint_room_ids);
    const completedWaypointIds = new Set(optionalStringArray(payload.completed_waypoints));
    const boundaries = [];
    rounds.forEach((round, roundPosition) => {
      const boundary = round.room_boundary;
      if (!boundary || typeof boundary !== "object" || Array.isArray(boundary)) return;
      const panoId = stringOrNull(boundary.at_pano_id);
      if (!panoId) return;
      const pathIndex = panoPath.indexOf(panoId);
      if (pathIndex < 0) {
        throw new Error(`Room boundary references pano outside pano_path: ${panoId}`);
      }
      const toRoomId = stringOrNull(boundary.to_room_id);
      boundaries.push({
        roundIndex: finiteNumber(round.round_index, roundPosition),
        panoId,
        pathIndex,
        fromRoomId: stringOrNull(boundary.from_room_id),
        toRoomId,
        kind: toRoomId && waypointRoomIds.includes(toRoomId) ? "waypoint" : "room_transition",
        completed: Boolean(toRoomId && completedWaypointIds.has(toRoomId)),
      });
    });

    const boundaryByPathIndex = new Map(boundaries.map((boundary) => [boundary.pathIndex, boundary]));
    const frames = panoPath.map((panoId, index) => {
      const outgoing = normalizedMovements[index] || null;
      const incoming = index > 0 ? normalizedMovements[index - 1] : null;
      const movement = outgoing || incoming;
      const graphNode = graph.nodes.get(panoId);
      const boundary = boundaryByPathIndex.get(index) || null;
      return {
        index,
        panoId,
        roomId:
          (outgoing && outgoing.currentRoomId) ||
          (incoming && incoming.nextRoomId) ||
          stringOrNull(graphNode.room_id),
        heading: movement ? movement.actionHeading : null,
        roundIndex: movement ? movement.roundIndex : null,
        subgoalRoomId: movement ? movement.subgoalRoomId : null,
        activeTargetRoomId: movement ? movement.activeTargetRoomId : null,
        similarity: outgoing ? outgoing.similarity : incoming ? incoming.similarity : null,
        margin: outgoing ? outgoing.margin : incoming ? incoming.margin : null,
        imageGoalLabel: outgoing ? outgoing.imageGoalLabel : incoming ? incoming.imageGoalLabel : null,
        boundary,
        outgoingMovement: outgoing,
        incomingMovement: incoming,
      };
    });

    return {
      schema: "memory_nav_full_episode",
      panoPath,
      movements: normalizedMovements,
      frames,
      boundaries,
      rounds: rounds.map((round, roundPosition) => ({
        roundIndex: finiteNumber(round.round_index, roundPosition),
        startPanoId: stringOrNull(round.start_pano_id),
        activeTargetRoomId: stringOrNull(round.active_target_room_id),
        subgoalRoomId: stringOrNull(round.subgoal_room_id),
        completedWaypoints: optionalStringArray(round.completed_waypoints),
        movementCount: Array.isArray(round.movement_steps) ? round.movement_steps.length : 0,
      })),
      waypointRoomIds,
      completedWaypointIds: [...completedWaypointIds],
      orderedTargets: optionalStringArray(payload.ordered_targets),
      targetRoomId: stringOrNull(payload.target_room_id),
      startPanoId: stringOrNull(payload.start_pano_id) || panoPath[0],
      finalPanoId: stringOrNull(payload.final_pano_id) || panoPath[panoPath.length - 1],
      success: payload.success === true,
      reason: stringOrNull(payload.reason),
      raw: payload,
    };
  }

  function buildGraphIndex(graphData) {
    if (!graphData || !Array.isArray(graphData.nodes) || !Array.isArray(graphData.edges)) {
      throw new Error("viewer_data.json must contain nodes and edges arrays.");
    }
    const nodes = new Map();
    for (const node of graphData.nodes) {
      if (node && typeof node.id === "string") nodes.set(node.id, node);
    }
    const edges = new Map();
    for (const edge of graphData.edges) {
      if (!edge || typeof edge.source !== "string" || typeof edge.target !== "string") continue;
      const key = edgeKey(edge.source, edge.target);
      if (!edges.has(key)) edges.set(key, edge);
    }
    return { nodes, edges };
  }

  function createManualSyncGate(options) {
    if (!options || typeof options.load !== "function" || typeof options.sync !== "function") {
      throw new Error("Manual sync gate requires load and sync functions.");
    }
    let loaded = false;
    let loading = null;
    let latestValue;

    return {
      update(value) {
        latestValue = value;
        if (loaded) options.sync(value);
      },
      async load() {
        if (loaded) {
          if (latestValue !== undefined) options.sync(latestValue);
          return;
        }
        if (!loading) {
          loading = Promise.resolve()
            .then(() => options.load())
            .then(() => {
              loaded = true;
              if (latestValue !== undefined) options.sync(latestValue);
            })
            .finally(() => {
              loading = null;
            });
        }
        await loading;
      },
      isLoaded() {
        return loaded;
      },
    };
  }

  function requireStringArray(value, name) {
    if (!Array.isArray(value)) throw new Error(`${name} must be an array.`);
    return value.map((item, index) => {
      if (typeof item !== "string" || !item.trim()) {
        throw new Error(`${name}[${index}] must be a non-empty string.`);
      }
      return item.trim();
    });
  }

  function optionalStringArray(value) {
    if (!Array.isArray(value)) return [];
    return value.filter((item) => typeof item === "string" && item.trim()).map((item) => item.trim());
  }

  function finiteNumber(value, fallback) {
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
  }

  function stringOrNull(value) {
    return typeof value === "string" && value.trim() ? value.trim() : null;
  }

  function normalizeHeading(value) {
    return ((value % 360) + 360) % 360;
  }

  function edgeKey(source, target) {
    return `${source}\u0000${target}`;
  }

  return {
    buildGraphIndex,
    createManualSyncGate,
    parseTrajectory,
  };
});
