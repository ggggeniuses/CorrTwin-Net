# GitHub Publish Guide

## Files to Upload

Upload the full project directory after applying `.gitignore`.

Important files and folders:

```text
README.md
LICENSE
requirements.txt
channel_sim/
models/
experiments/
utils/
tests/
configs/
docs/
notebooks/
results/figures/
results/metrics/
data/README.md
data/sample_dataset.npz
.github/workflows/python-ci.yml
```

## Files Not to Upload

These files are generated and should not be committed:

```text
data/train.npz
data/val.npz
data/test.npz
results/checkpoints/*.pt
results/runs/**/*.pt
results/runs/**/*.pkl
__pycache__/
.pytest_cache/
.venv/
```

## Recommended Git Commands

```bash
git init
git add README.md LICENSE requirements.txt .gitignore .github configs channel_sim models utils experiments tests docs notebooks data results/figures results/metrics
git commit -m "Initial release of CorrTwin-Net"
git branch -M main
git remote add origin https://github.com/<your-name>/CorrTwin-Net.git
git push -u origin main
```

After pushing, replace the README badge placeholder if you want a CI badge:

```markdown
[![Python CI](https://github.com/<your-name>/CorrTwin-Net/actions/workflows/python-ci.yml/badge.svg)](https://github.com/<your-name>/CorrTwin-Net/actions/workflows/python-ci.yml)
```

## Release Tag

```bash
git tag -a dev-preview-20260606 -m "CorrTwin-Net development preview"
git push origin dev-preview-20260606
```

Use `docs/release_notes_v1.0.0.md` as development-preview notes until the long formal run and realization convergence study are complete.

## Final Checks

```bash
python -m pytest -q
python experiments/run_full_pipeline.py --quick
```

The quick pipeline regenerates small local data and model files. They are ignored by Git, but you can delete them again before upload if using manual web upload.
