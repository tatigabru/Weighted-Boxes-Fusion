# weighted-boxes-fusion

Python library implementing box ensemble methods for object detection: NMS, Soft-NMS, NMW, and WBF.

## Project structure

- [ensemble_boxes/](../ensemble_boxes/) — core algorithm implementations
  - [ensemble_boxes_nms.py](../ensemble_boxes/ensemble_boxes_nms.py) — NMS and Soft-NMS
  - [ensemble_boxes_nmw.py](../ensemble_boxes/ensemble_boxes_nmw.py) — Non-maximum weighted
  - [ensemble_boxes_wbf.py](../ensemble_boxes/ensemble_boxes_wbf.py) — Weighted boxes fusion (2D)
  - [ensemble_boxes_wbf_1d.py](../ensemble_boxes/ensemble_boxes_wbf_1d.py) — WBF for 1D (NLP spans)
  - [ensemble_boxes_wbf_3d.py](../ensemble_boxes/ensemble_boxes_wbf_3d.py) — WBF for 3D boxes
  - [ensemble_boxes_wbf_experimental.py](../ensemble_boxes/ensemble_boxes_wbf_experimental.py) — experimental variants
- [tests/](../tests/) — test suite
- [examples/](../examples/) — usage notebooks and scripts
- [benchmark_coco/](../benchmark_coco/), [benchmark_oid/](../benchmark_oid/), [benchmark_nlp/](../benchmark_nlp/) — benchmarks

## Key conventions

- Box coordinates are normalized to `[0, 1]`, in `x1, y1, x2, y2` order.
- Inputs: `boxes_list`, `scores_list`, `labels_list` — one list per model; `weights` is optional.
- All public functions return `(boxes, scores, labels)` as numpy arrays.
- Dependencies: `numpy`, `numba`, `pandas`. Numba is used for performance-critical loops — keep numba-compiled functions free of Python objects.
- Package version is in [setup.py](../setup.py) and [CHANGES.md](../CHANGES.md).

## Development

```bash
pip install -e .          # install in editable mode
python -m pytest tests/   # run tests
```

## Rules

- Do not change the existing public API signatures without asking! 
- Keep algorithms numerically stable; prefer explicit epsilon guards over silent divisions by zero.
- New ensemble methods go in their own `ensemble_boxes_<name>.py` file and must be exported from [ensemble_boxes/__init__.py](../ensemble_boxes/__init__.py).
- Do not add heavy dependencies (e.g. torch, cv2) to the core package.
- Tests live in [tests/](../tests/) and should cover edge cases: empty inputs, single model, all boxes filtered by threshold.
