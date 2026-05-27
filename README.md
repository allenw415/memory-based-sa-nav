# memory-based-sa-nav

Memory-based spatial-alignment navigation with visual memory retrieval.

This repository contains the memory-based navigation pipeline for interactive museum navigation, including image-memory localization, target-room memory retrieval, and passage-level spatial alignment. The Python package is named `memory_nav`.

## What Is Included

- `memory_nav/memory/`: memory image retrieval, room localization, passage alignment, and interactive guidance orchestration
- `memory_nav/data/`: memory localization utilities, pano visualization helpers, and British Museum graph normalization helpers
- `memory_nav/common/`: minimal model/environment utilities used by the memory advisor
- `memory_nav/perception/renderer.py`: Google Street View pano rendering used to rebuild memory indexes
- `memory_nav/spatial/routing.py`: room graph route planning used by the memory navigator
- `memory_nav/web/`: unified FastAPI web server and static assets for memory guidance and pano viewer
- `tools/data/build_memory_localization_index.py`: render pano views and rebuild SigLIP2/FAISS memory indexes
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

## Pano Viewer

The unified web server exports and serves the panorama viewer automatically by
default. Use `--pano-export missing` to export only when the viewer artifact is
missing, or `--pano-export never` to serve an existing export without refreshing
it.

## Evaluate Memory Localization

```bash
python3 tools/data/eval_memory_localization.py \
  --index-path artifacts/memory_localization/floor0_siglip2_images.npz \
  --metadata-path artifacts/memory_localization/floor0_siglip2_images.metadata.json \
  --faiss-path artifacts/memory_localization/floor0_siglip2_images.faiss
```

## Tests

```bash
python3 -m unittest tests/test_memory_navigation.py
```

## Artifact Note

The `.npz` and `.faiss` files are large binary memory indexes. `.gitattributes` marks them for Git LFS; if you publish this repository, initialize Git LFS before committing those files.
