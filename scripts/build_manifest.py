"""Build reproducibility manifest with SHA-256 checksums, byte sizes, CSV row counts, and environment metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PACKAGE_LIST = [
    "numpy",
    "pandas",
    "scikit-learn",
    "matplotlib",
    "requests",
    "python-dotenv",
    "pytest",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_installed_version(pkg_name: str) -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version(pkg_name)
    except Exception:
        return "not installed"


def inspect_file(path: Path, root: Path) -> dict:
    rel_path = str(path.relative_to(root)).replace("\\", "/")
    byte_size = path.stat().st_size
    checksum = sha256_file(path)

    info = {
        "path": rel_path,
        "bytes": byte_size,
        "sha256": checksum,
    }

    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            info["csv_rows"] = len(df)
            info["csv_columns"] = df.columns.tolist()
        except Exception as exc:
            info["csv_read_error"] = str(exc)

    return info


def build_manifest(root_dir: Path, output_file: Path) -> dict:
    scanned_dirs = ["results", "data", "scripts", "src", "tests"]
    file_records = []

    for d in scanned_dirs:
        dir_path = root_dir / d
        if not dir_path.exists():
            continue
        for child in sorted(dir_path.rglob("*")):
            if child.is_file():
                # Skip pycache and temporary files
                if "__pycache__" in child.parts or child.suffix in (".pyc", ".pyo"):
                    continue
                # Skip large local corpora excluded from Git if present
                if child.name == "bitext_customer_support.csv":
                    continue
                if child.name == "run_real_benchmark_backup.py":
                    continue
                file_records.append(inspect_file(child, root_dir))

    # Also include root configuration / doc files
    for root_file in ["pyproject.toml", "README.md", "CONTRIBUTING.md", ".gitignore"]:
        p = root_dir / root_file
        if p.exists() and p.is_file():
            file_records.append(inspect_file(p, root_dir))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {pkg: get_installed_version(pkg) for pkg in PACKAGE_LIST},
        "file_count": len(file_records),
        "files": file_records,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to: {output_file}")
    print(f"Total files cataloged: {len(file_records)}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build reproducibility manifest for adaptive semantic caching experiments."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reproducibility_manifest.json"),
        help="Path where the JSON manifest will be saved.",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    build_manifest(root_dir, args.output)


if __name__ == "__main__":
    main()
