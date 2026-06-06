from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def iter_artifacts(data_dir: str, results_dir: str) -> list[Path]:
    patterns = [
        f"{data_dir}/*.npz",
        f"{data_dir}/*metadata*.json",
        f"{results_dir}/checkpoints/*.pt",
        f"{results_dir}/runs/**/*.pt",
        f"{results_dir}/runs/**/*.pkl",
        f"{results_dir}/metrics/*.json",
        f"{results_dir}/metrics/*.csv",
        f"{results_dir}/reproduction_manifest.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(ROOT.glob(pattern))
    return sorted({p for p in paths if p.is_file()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artifact SHA256 manifest.")
    parser.add_argument("--output", default="artifacts_manifest.generated.json")
    parser.add_argument("--dataset-config-version", default="p0_clean_channel_v2")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat()
    commit = git_commit()
    entries = []
    for path in iter_artifacts(args.data_dir, args.results_dir):
        entries.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "file_size": path.stat().st_size,
                "sha256": sha256(path),
                "created_at": created_at,
                "git_commit": commit,
                "dataset_config_version": args.dataset_config_version,
            }
        )
    output = ROOT / args.output
    output.write_text(json.dumps({"artifacts": entries}, indent=2), encoding="utf-8")
    print(f"wrote {output} with {len(entries)} artifacts")


if __name__ == "__main__":
    main()
