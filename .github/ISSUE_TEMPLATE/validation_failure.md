---
name: Validation Failure
about: Report a failure following the quickstart or reproduction steps
title: "[Validation] "
labels: validation
assignees: ''
---

## Which Step Failed

- [ ] `uv sync`
- [ ] `scripts/export_yolo.py`
- [ ] `scripts/init_qdrant.py`
- [ ] `docker compose up --build triton`
- [ ] Triton model loading (which model: ___)
- [ ] `client.py` inference
- [ ] Other: ___

## Environment

| Field | Value |
|-------|-------|
| GPU model | |
| VRAM | |
| NVIDIA driver | |
| CUDA version | |
| OS | |
| Docker version | |
| LLM model ID | |

## Error Output

```
Paste relevant log output here
```

## What You Expected

Describe the expected outcome at this step.

## Additional Context

Network conditions, disk space, whether this is a first run or repeat attempt, etc.
