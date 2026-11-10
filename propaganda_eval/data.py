"""The eval items and judge prompt, downloaded from Hugging Face at a pinned revision and checked by sha256.

The data and judge prompt belong to Nous Research (MIT): https://huggingface.co/datasets/NousResearch/Propaganda_Eval
Nothing from it is copied into this repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "NousResearch/Propaganda_Eval"
REVISION = "401223a2bfcdcd99d4650a83c4d20565db0e56b8"  # main on 2026-11-06
FILES = {
    "data/items.jsonl": "76bf1b75337b948d37ab494d7192aade5cc8fefd08c1e170f1925816ac6c4ef6",
    "judge/JUDGE_PROMPT.md": "a391774a99cb9cae9d8c9b6feff347691ddb93b1be14d840ee675c5cb8b9274e",
    "LICENSE": None,
    "README.md": None,
}
ITEMS_SHA256 = FILES["data/items.jsonl"]
PROMPT_SHA256 = FILES["judge/JUDGE_PROMPT.md"]


class EvalError(RuntimeError):
    """A refusal with a reason the user can act on."""


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def cache_root() -> Path:
    env = os.environ.get("PROPAGANDA_EVAL_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "propaganda-eval"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, target: Path, timeout: float = 120) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "propaganda-eval"})
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    tmp.replace(target)


def fetch_dataset(revision: str = REVISION, root: Path | None = None) -> Path:
    """Download the pinned files once and return the folder. A file whose sha256 differs is refused and deleted."""

    folder = (root or cache_root()) / "nous" / revision
    for name, want in FILES.items():
        path = folder / name
        if not path.is_file():
            download(f"{hf_endpoint()}/datasets/{DATASET}/resolve/{revision}/{name}", path)
        if want and revision == REVISION and sha256(path) != want:
            path.unlink()
            raise EvalError(f"{name} at revision {revision} does not match its pinned sha256")
    return folder


def resolve_data(data_dir: Path | None, pinned: bool = True) -> Path:
    """A local copy of the dataset (same layout as on Hugging Face) or the pinned download."""

    if data_dir is None:
        return fetch_dataset()
    if pinned:
        for name in ("data/items.jsonl", "judge/JUDGE_PROMPT.md"):
            if sha256(data_dir / name) != FILES[name]:
                raise EvalError(f"{data_dir / name} does not match the pinned sha256; scores would not be comparable. "
                                "Use --unpinned only for test fixtures.")
    return data_dir


def load_items(data: Path) -> list[dict[str, Any]]:
    path = data / "data" / "items.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_template(data: Path) -> str:
    return (data / "judge" / "JUDGE_PROMPT.md").read_text(encoding="utf-8")


def items_sha(data: Path) -> str:
    return sha256(data / "data" / "items.jsonl")
