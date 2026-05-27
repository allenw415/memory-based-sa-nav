# Memory Guidance Web Demo

Local browser demo for the interactive RAG memory-navigation prototype.

Run:

```bash
python3 tools/memory_guidance_web/server.py --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The page lets you upload localization images, optional passage images, a target
room, and optional waypoints. The server returns the same guidance JSON shape as
`memory_nav.cli.run_memory_guidance`.

If the memory artifacts are stale, rebuild them first:

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
