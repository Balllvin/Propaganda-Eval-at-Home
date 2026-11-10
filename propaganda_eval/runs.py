"""Run folders: ``<out>/<label>/<profile>/run<N>/`` with manifest.json, completions.jsonl and grades.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import EvalError

SCHEMA = "propaganda-eval/v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by a crash; the item is redone on resume
                continue
    return rows


def latest(path: Path) -> dict[str, dict[str, Any]]:
    """Per id, the last stored row; a retried error is replaced by its later result."""

    return {str(r["id"]): r for r in read_jsonl(path)}


def run_dir(out: Path, label: str, profile: str, run: int) -> Path:
    return out / label / profile / f"run{run}"


def run_dirs(out: Path, label: str, profile: str) -> list[Path]:
    root = out / label / profile
    return sorted((p for p in root.glob("run[0-9]*") if p.is_dir()), key=lambda p: int(p.name[3:]))


def labels(out: Path) -> list[tuple[str, str]]:
    """Every (label, profile) under out that has at least one run."""

    found = []
    for manifest in sorted(out.glob("*/*/run*/manifest.json")):
        pair = (manifest.parent.parent.parent.name, manifest.parent.parent.name)
        if pair not in found:
            found.append(pair)
    return found


def write_manifest(path: Path, want: dict[str, Any], ignore: tuple[str, ...] = ("base_url", "send_seed", "concurrency")) -> None:
    """Write the run's settings once; a resume with different settings would mix two experiments."""

    file = path / "manifest.json"
    if file.is_file():
        have = json.loads(file.read_text())
        diff = [k for k in want if k not in ignore and have.get(k) != want[k]]
        if diff:
            raise EvalError(f"{path}: resume refused, settings differ: {diff}. Use a new --label.")
        return
    path.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(want, indent=1) + "\n")


def manifest(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text())
