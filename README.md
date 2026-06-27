# memory-based-sa-nav

Memory-based spatial-alignment navigation with visual memory retrieval.

This repository contains the memory-based navigation pipeline for interactive museum navigation, including image-memory localization, target-room memory retrieval, and passage-level spatial alignment. The Python package is named `memory_nav`.

## What Is Included

- `memory_nav/memory/`: memory image retrieval, room localization, passage alignment, and interactive guidance orchestration
- `memory_nav/data/`: memory localization utilities, SigLIP2/DINOv2-SALAD embedding helpers, pano visualization helpers, and British Museum graph normalization helpers
- `memory_nav/common/`: minimal model/environment utilities used by the memory advisor
- `memory_nav/perception/renderer.py`: Google Street View pano rendering used to rebuild memory indexes
- `memory_nav/spatial/routing.py`: room graph route planning used by the memory navigator
- `memory_nav/web/`: unified FastAPI web server and static assets for memory guidance and pano viewer
- `tools/data/build_memory_localization_index.py`: render pano views and rebuild SigLIP2 or DINOv2-SALAD/FAISS memory indexes
- `tools/data/demo_memory_localization.py`: single-pano memory localization/debug run
- `tools/data/eval_memory_localization.py`: offline evaluation for the image-memory localization index
- `dataset/sites/british_museum/`: British Museum site assets retained from the original project
- `artifacts/memory_localization/`: prebuilt memory localization indexes

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For model guidance, configure `.env` or your shell with the concise `NAV_<profile>_<field>` keys:

```bash
export NAV_PROFILE=openai
export NAV_OPENAI_KEY=...
export NAV_OPENAI_MODEL=gpt-5-mini
export NAV_OPENAI_API=responses
export NAV_OPENAI_BASE=https://api.openai.com/v1
```

## Run One Guidance Step

```bash
python3 -m memory_nav.cli.run_memory_guidance \
  --target-room-id "Room 23" \
  --localization-images path/to/current_view.jpg \
  --passage-images front=path/to/front.jpg left=path/to/left.jpg
```

## Web Tools

```bash
python3 -m memory_nav.web --port 8765
```

Then open:

- `http://127.0.0.1:8765/memory-guidance/`
- `http://127.0.0.1:8765/pano-viewer/`.

## Rebuild Memory Index

Set `GMAPS_KEY` in your shell or pass `--render-api-key`, then rebuild the default floor-0 image memory index:

```bash
python3 tools/data/build_memory_localization_index.py \
  --floor 0 \
  --include-sources manual:accepted \
  --heading-mode museum \
  --max-captures 8 \
  --render-output-dir renders/room_grounding \
  --output-dir artifacts/memory_localization \
  --output-prefix floor0_siglip2_images
```

The renderer caches pano manifests/images under `renders/`, so repeated rebuilds skip already-rendered captures when settings match.

To build the DINOv2-SALAD floor-0 fov90 index for visual place-recognition experiments:

```bash
python3 tools/data/build_memory_localization_index.py \
  --floor 0 \
  --include-sources manual:accepted \
  --embedding-model dinov2-salad \
  --device cpu \
  --batch-size 1 \
  --heading-mode museum \
  --fov 90 \
  --max-captures 8 \
  --render-output-dir renders/room_grounding_fov90 \
  --output-dir artifacts/memory_localization
```

This writes independent artifacts such as `floor0_dinov2_salad_images_fov90.npz`,
`floor0_dinov2_salad_images_fov90.faiss`, and
`floor0_dinov2_salad_images_fov90.metadata.json`. The first DINOv2-SALAD run
uses Torch Hub to load the official `serizba/salad` model and may need network
access.

## Pano Viewer

The unified web server exports and serves the panorama viewer automatically by
default. Use `--pano-export missing` to export only when the viewer artifact is
missing, or `--pano-export never` to serve an existing export without refreshing
it.

The viewer accepts navigation outputs such as
`outputs/navigation/full_episode_erp_to_room23.json`. Open or drop the JSON in
the **Trajectory** panel to draw the full pano path, inspect room and waypoint
boundaries, and play the episode with step, timeline, and speed controls.

Street View remains unloaded while a trajectory is opened or played. Click the
Street View **Load** button when you want to initialize Google Maps; after that,
the panorama follows the current playback frame and uses each movement's
`selected_action_heading`. If that field is absent, the viewer uses the heading
from the matching panorama graph edge.

Trajectory JSON is parsed locally in the browser and is not uploaded to the web
server. When a trajectory is loaded, the viewer can also prepare local rendered
panorama frames and export the bottom-right panorama playback as a WebM video
with the **Export Video** button.

## Evaluate Memory Localization

```bash
python3 tools/data/eval_memory_localization.py \
  --index-path artifacts/memory_localization/floor0_siglip2_images.npz \
  --metadata-path artifacts/memory_localization/floor0_siglip2_images.metadata.json \
  --faiss-path artifacts/memory_localization/floor0_siglip2_images.faiss
```

For the DINOv2-SALAD experiment with rerendered query images, 1/2/3/4 sampled
views, repeated random seeds, and same-pano retrieval enabled:

```bash
python3 tools/data/eval_memory_localization.py \
  --index-path artifacts/memory_localization/floor0_dinov2_salad_images_fov90.npz \
  --metadata-path artifacts/memory_localization/floor0_dinov2_salad_images_fov90.metadata.json \
  --faiss-path artifacts/memory_localization/floor0_dinov2_salad_images_fov90.faiss \
  --embedding-model dinov2-salad \
  --device cpu \
  --batch-size 1 \
  --query-render-mode rerender \
  --query-output-dir renders/memory_localization_eval_queries_dinov2_salad_fov90 \
  --query-render-fov 90 \
  --query-view-counts 1,2,3,4 \
  --query-selection random \
  --query-random-seeds 0,1,2,3,4 \
  --include-same-pano \
  --output-path outputs/localization/memory_localization_eval_dinov2_salad_fov90_1to4_views.json
```

The report includes overall and per-room top-1/top-3 accuracy, confidence and
margin summaries, high-confidence correctness, hardest errors, confusion pairs,
and the same-pano top-candidate rate so the result is not mistaken for pure
cross-pano generalization.

## Tests

```bash
python3 -m unittest tests/test_memory_navigation.py
```

## Artifact Note

The `.npz` and `.faiss` files are large binary memory indexes. `.gitattributes` marks them for Git LFS; if you publish this repository, initialize Git LFS before committing those files.
