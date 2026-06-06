from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify artifact SHA256 manifest.")
    parser.add_argument("--manifest", default="artifacts_manifest.generated.json")
    args = parser.parse_args()
    manifest_path = ROOT / args.manifest
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in payload.get("artifacts", []):
        path = ROOT / item["relative_path"]
        if not path.exists():
            failures.append(f"missing: {item['relative_path']}")
            continue
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        if actual_size != int(item["file_size"]):
            failures.append(f"size mismatch: {item['relative_path']}")
        if actual_hash != item["sha256"]:
            failures.append(f"sha256 mismatch: {item['relative_path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"verified {len(payload.get('artifacts', []))} artifacts")


if __name__ == "__main__":
    main()
