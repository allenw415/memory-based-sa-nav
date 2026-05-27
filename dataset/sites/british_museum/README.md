## British Museum Dataset

This site keeps different data assets in parallel directories.

- `explicit_map/`: manually curated semantic room-to-room map
- `pano_graph/`: Street View pano navigation graph
- `normalized/`: runtime-ready normalized artifacts and grounding files

This directory is data-only. Generation scripts and viewer tooling live under `dataset/pipelines/`.

Typical normalized files used by the runtime:

- `room_graph.json`
- `pano_graph.json`
- `pano_room_grounding.json`

Additional grounding outputs such as Gemini predictions, manual review files,
and compact pano-to-room mappings also live under `normalized/`.
