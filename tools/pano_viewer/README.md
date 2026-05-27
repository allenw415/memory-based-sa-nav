# Pano Viewer

Static viewer and export targets for normalized panorama graphs.

Generate artifacts:

```bash
python3 tools/pano_viewer/export.py
```

The exporter copies viewer assets from `tools/pano_viewer/web`. It also reads
`GMAPS_API_KEY` from the project `.env` and writes a generated `.env.js` next to
the exported viewer files so the Street View panel can load.

Serve the copied viewer:

```bash
python3 -m http.server 8000 --directory artifacts/pano_viewer/british_museum
```

The export writes:

- `viewer_data.json` for the browser viewer
- `pano_nodes.geojson` and `pano_edges.geojson` for map tooling
- `pano_graph.gexf` and `pano_graph.graphml` for Gephi/Sigma-style workflows
- `pano_graph_floor0.dot` for Graphviz
- `publication/floor_*_overview.svg` for report figures
- `.env.js` for the local Street View API key when `GMAPS_API_KEY` is set

Street View uses the generated `.env.js` in the exported viewer folder:

```js
window.GMAPS_API_KEY = "YOUR_KEY";
```

Regenerate it after changing `.env`:

```bash
python3 tools/pano_viewer/export.py
```

Keep `.env.js` out of git.
