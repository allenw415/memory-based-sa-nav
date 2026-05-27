# memory-based-sa-nav

Memory-based spatial-alignment navigation with visual memory retrieval.

This repository contains the memory-based navigation pipeline for interactive museum navigation, including image-memory localization, target-room memory retrieval, and passage-level spatial alignment. The Python package is named `memory_nav`.

## What Is Included

- `memory_nav/memory/`: memory image retrieval, room localization, passage alignment, and interactive guidance orchestration
- `memory_nav/data/`: memory localization utilities and British Museum graph normalization helpers
- `memory_nav/common/`: minimal model/environment utilities used by the memory advisor
- `memory_nav/spatial/routing.py`: room graph route planning used by the memory navigator
- `tools/memory_guidance_web/`: local browser demo for uploading localization and passage images
- `tools/data/eval_memory_localization.py`: offline evaluation for the image-memory localization index
- `dataset/sites/british_museum/`: British Museum site assets retained from the original project
- `artifacts/memory_localization/`: prebuilt memory localization indexes

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For hosted VLM guidance, configure one of these environment variables:

```bash
export ST_NAV_API_KEY=...
# optional
export ST_NAV_MODEL_NAME=gpt-5-mini
export ST_NAV_API_KIND=responses
export ST_NAV_API_BASE=https://api.openai.com/v1
```

## Run One Guidance Step

```bash
python3 -m memory_nav.cli.run_memory_guidance \
  --target-room-id "Room 23" \
  --localization-images path/to/current_view.jpg \
  --passage-images front=path/to/front.jpg left=path/to/left.jpg
```

## Web Demo

```bash
python3 tools/memory_guidance_web/server.py --port 8765
```

Then open `http://127.0.0.1:8765`.

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
