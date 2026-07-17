const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const {
  createManualSyncGate,
  parseTrajectory,
  resolveEpisodePayload,
} = require("../memory_nav/web/static/pano_viewer/trajectory.js");

function graphFixture() {
  return {
    nodes: [
      { id: "A", room_id: "Room 1" },
      { id: "B", room_id: "Room 2" },
      { id: "C", room_id: "Room 3" },
    ],
    edges: [
      { source: "A", target: "B", heading: 12 },
      { source: "B", target: "C", heading: 359 },
    ],
  };
}

function episodeFixture() {
  return {
    start_pano_id: "A",
    final_pano_id: "C",
    target_room_id: "Room 3",
    pano_path: ["A", "B", "C"],
    waypoint_room_ids: ["Room 2"],
    completed_waypoints: ["Room 2"],
    ordered_targets: ["Room 2", "Room 3"],
    success: true,
    rounds: [
      {
        round_index: 0,
        start_pano_id: "A",
        active_target_room_id: "Room 2",
        subgoal_room_id: "Room 2",
        completed_waypoints: [],
        movement_steps: [
          {
            current_pano_id: "A",
            next_pano_id: "B",
            current_room_id: "Room 1",
            next_room_id: "Room 2",
            selected_action_heading: 372,
            subgoal_room_id: "Room 2",
            active_target_room_id: "Room 2",
          },
        ],
        room_boundary: {
          from_room_id: "Room 1",
          to_room_id: "Room 2",
          at_pano_id: "B",
        },
      },
      {
        round_index: 1,
        start_pano_id: "B",
        active_target_room_id: "Room 3",
        subgoal_room_id: "Room 3",
        completed_waypoints: ["Room 2"],
        movement_steps: [
          {
            current_pano_id: "B",
            next_pano_id: "C",
            current_room_id: "Room 2",
            next_room_id: "Room 3",
          },
        ],
        room_boundary: {
          from_room_id: "Room 2",
          to_room_id: "Room 3",
          at_pano_id: "C",
        },
      },
    ],
  };
}

test("parseTrajectory expands movements, rounds, headings, and boundaries", () => {
  const trajectory = parseTrajectory(episodeFixture(), graphFixture());

  assert.deepEqual(trajectory.panoPath, ["A", "B", "C"]);
  assert.equal(trajectory.frames.length, 3);
  assert.equal(trajectory.movements.length, 2);
  assert.equal(trajectory.movements[0].actionHeading, 12);
  assert.equal(trajectory.movements[0].headingSource, "trajectory");
  assert.equal(trajectory.movements[1].actionHeading, 359);
  assert.equal(trajectory.movements[1].headingSource, "graph_edge");
  assert.equal(trajectory.frames[2].heading, 359);
  assert.equal(trajectory.boundaries[0].kind, "waypoint");
  assert.equal(trajectory.boundaries[0].pathIndex, 1);
  assert.equal(trajectory.boundaries[1].kind, "room_transition");
  assert.equal(trajectory.frames[1].subgoalRoomId, "Room 3");
});

test("parseTrajectory accepts a Pilot evaluator wrapper and preserves test metadata", () => {
  const episode = episodeFixture();
  const wrapper = {
    test_case: {
      test_id: "TEST001",
      query: "Take me to the Greece 1050–520 BC gallery.",
      difficulty: "easy",
      ratio_stratum: "1.0-1.5",
      passage_profile: "reliable",
      target_group_theme: "Greece 1050–520 BC",
    },
    parsed_query: { target_room_id: "Room 3" },
    episode,
    episode_error: null,
    evaluation: {
      test_id: "TEST001",
      success: true,
      reason: "target_room_relocalized",
    },
  };

  const trajectory = parseTrajectory(wrapper, graphFixture());

  assert.equal(trajectory.sourceSchema, "evaluation_wrapper");
  assert.equal(trajectory.raw, episode);
  assert.equal(trajectory.rawEnvelope, wrapper);
  assert.deepEqual(trajectory.panoPath, ["A", "B", "C"]);
  assert.deepEqual(trajectory.evaluationMetadata, {
    testId: "TEST001",
    query: "Take me to the Greece 1050–520 BC gallery.",
    difficulty: "easy",
    ratioStratum: "1.0-1.5",
    passageProfile: "reliable",
    targetGroupTheme: "Greece 1050–520 BC",
  });
});

test("resolveEpisodePayload reports evaluator records without a completed episode", () => {
  assert.throws(
    () =>
      resolveEpisodePayload({
        episode: null,
        episode_error: "RuntimeError: navigation failed",
      }),
    /does not contain a completed episode: RuntimeError: navigation failed/,
  );
});

test("parseTrajectory rejects unknown panos and illegal graph edges", () => {
  const unknown = episodeFixture();
  unknown.pano_path[2] = "missing";
  assert.throws(() => parseTrajectory(unknown, graphFixture()), /unknown pano/);

  const illegal = episodeFixture();
  illegal.pano_path = ["A", "C", "B"];
  illegal.rounds[0].movement_steps[0].next_pano_id = "C";
  illegal.rounds[1].movement_steps[0].current_pano_id = "C";
  illegal.rounds[1].movement_steps[0].next_pano_id = "B";
  illegal.rounds[0].room_boundary.at_pano_id = "C";
  assert.throws(() => parseTrajectory(illegal, graphFixture()), /not a legal graph edge/);
});

test("parseTrajectory rejects movement steps that do not align with pano_path", () => {
  const episode = episodeFixture();
  episode.rounds[0].movement_steps[0].next_pano_id = "C";
  assert.throws(() => parseTrajectory(episode, graphFixture()), /does not match pano_path/);
});

test("manual sync gate never loads until load is explicitly requested", async () => {
  const events = [];
  const gate = createManualSyncGate({
    load: async () => events.push("load"),
    sync: (value) => events.push(["sync", value]),
  });

  gate.update({ panoId: "A", heading: 12 });
  gate.update({ panoId: "B", heading: 20 });
  assert.deepEqual(events, []);
  assert.equal(gate.isLoaded(), false);

  await gate.load();
  assert.deepEqual(events, ["load", ["sync", { panoId: "B", heading: 20 }]]);
  assert.equal(gate.isLoaded(), true);

  gate.update({ panoId: "C", heading: 30 });
  assert.deepEqual(events.at(-1), ["sync", { panoId: "C", heading: 30 }]);
});

for (const [fileName, expectedMoves, expectedFinalPano] of [
  ["full_episode_erp_to_room23.json", 7, "Li54te8XaSyXgj2x_c2msA"],
  ["full_episode_nc6_to_room23.json", 4, "Li54te8XaSyXgj2x_c2msA"],
]) {
  const projectRoot = path.resolve(__dirname, "..");
  const graphPath = path.join(projectRoot, "artifacts/pano_viewer/british_museum/viewer_data.json");
  const episodePath = path.join(projectRoot, "outputs/navigation", fileName);
  const missingFixture = !fs.existsSync(graphPath) || !fs.existsSync(episodePath);
  test(`real episode ${fileName} parses against the exported panorama graph`, { skip: missingFixture }, () => {
    const graph = JSON.parse(fs.readFileSync(graphPath, "utf8"));
    const payload = JSON.parse(fs.readFileSync(episodePath, "utf8"));

    const trajectory = parseTrajectory(payload, graph);
    assert.equal(trajectory.success, true);
    assert.equal(trajectory.movements.length, expectedMoves);
    assert.equal(trajectory.frames.at(-1).panoId, expectedFinalPano);
    assert.equal(trajectory.frames.at(-1).roomId, "Room 23");
    assert.ok(trajectory.movements.every((movement) => Number.isFinite(movement.actionHeading)));
  });
}
