# File Manifest

This document explains the purpose of the main files in CorrTwin-Net. For a Chinese version, see `项目文件说明.md` in the repository root.

## Project Summary

CorrTwin-Net predicts normalized wireless channel correlation-magnitude curves:

```text
scenario parameters -> |Temporal ACF| / |Spatial CCF| / |Frequency FCF|
```

It is a surrogate modeling project, not CSI estimation and not a neural receiver.

## Top-Level Files

| Path | Purpose |
|---|---|
| `README.md` | Main GitHub project page with motivation, method, results, reproduction commands, and limitations. |
| `requirements.txt` | Python dependencies. |
| `requirements-lock.txt` | Exact dependency snapshot from the local environment. |
| `LICENSE` | MIT license. |
| `.gitignore` | Keeps generated data, model weights, caches, and virtual environments out of Git by default. |
| `.github/workflows/python-ci.yml` | GitHub Actions workflow that installs dependencies and runs pytest. |
| `项目文件说明.md` | Chinese file-by-file explanation for upload/review. |

## Source Directories

| Directory | Purpose |
|---|---|
| `channel_sim/` | Lightweight channel simulator, correlation-magnitude computation, realization averaging, and theory references. |
| `models/` | PyTorch MLP/ResMLP and sklearn baseline models. |
| `experiments/` | Reproducible scripts for data generation, training, evaluation, OOD tests, and full pipelines. |
| `utils/` | Metrics, plotting, normalization, config, seed, and run-directory utilities. |
| `tests/` | Lightweight pytest test suite. |
| `configs/` | YAML experiment settings. |
| `notebooks/` | Small demonstration notebooks. |

## Results

| Directory | Purpose |
|---|---|
| `results/figures/` | Key figures for README and reports. |
| `results/metrics/` | Experiment metrics in JSON/CSV. |
| `results/runs/` | Run configs, metrics, training logs, and local model/baseline artifacts when preserved. |
| `results/checkpoints/` | Best MLP/ResMLP checkpoints from the current local run. |
| `results/reproduction_manifest.json` | Machine-readable record of the current verified benchmark. |

## Documentation

| File | Purpose |
|---|---|
| `docs/project_report.md` | Project report. |
| `docs/experiment_protocol.md` | Experimental protocol. |
| `docs/reproducibility.md` | Environment and reproduction notes. |
| `docs/interview_notes.md` | Interview explanations and Q&A. |
| `docs/resume_description.md` | Resume-ready Chinese and English descriptions. |
| `docs/limitations_and_future_work.md` | Limitations and future work. |
| `docs/github_publish_guide.md` | GitHub upload guide. |
| `docs/for_next_codex_upload.md` | Handoff guide for another Codex instance. |
| `docs/release_notes_v1.0.0.md` | Development-preview notes and future release draft. |
| `docs/artifacts_manifest.json` | Machine-readable artifact list. |

## Data

| File | Purpose |
|---|---|
| `data/sample_dataset.npz` | Tiny demo dataset. |
| `data/train.npz`, `data/val.npz`, `data/test.npz` | Latest generated benchmark splits when preserved locally. |
| `data/formal_dataset_metadata.json` | Metadata for the latest generated dataset. |
| `data/README.md` | Data generation notes. |

Generated datasets may be included in the local package if complete artifacts are desired. For a normal Git repository, they can also be regenerated from scripts instead of committed directly.
