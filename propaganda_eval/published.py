"""Reference runs: three public result datasets, downloaded at pinned revisions and written as normal run folders, so
``score``, ``compare`` and ``report`` treat them like your own runs. They used the ``card`` profile and grok-4.7."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import ITEMS_SHA256, EvalError, cache_root, download, hf_endpoint
from .generate import PROFILES
from .runs import SCHEMA, run_dir

PUBLISHED: dict[str, dict[str, str]] = {
    "ref-dory-nvfp4": {"dataset": "balllvin/Propaganda-Eval-Dory-4B-NVFP4", "revision": "a7027ca3778bd55e5550579c67ef939409af4a7c",
                       "model": "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4", "name": "Dory 4B NVFP4"},
    "ref-lfm3-abliterated": {"dataset": "balllvin/Propaganda-Eval-LFM3-1.7B-Abliterated", "revision": "a8040583b300baf05f020bc2ebaa8b08e9209908",
                             "model": "orcarouter/LFM3-1.7B-abliterated", "name": "LFM3 1.7B abliterated"},
    "ref-lfm3-base": {"dataset": "balllvin/Propaganda-Eval-LFM3-1.7B-Base", "revision": "0f7ddc70c784e23a28560e756351f8f24d8c86bb",
                      "model": "LiquidAI/LFM3-1.7B", "name": "LFM3 1.7B base"},
}
GRADE_COUNTS = ("facts_stated_count", "facts_missed_count", "facts_opposed_count")


def convert(rows: Sequence[Mapping[str, Any]], target: Path, model: str, items: Sequence[Mapping[str, Any]] | None = None) -> dict[str, int]:
    """Write completions.jsonl, grades.jsonl and manifest.json from one published dataset's rows."""

    known = {str(i["id"]): i["prompt"] for i in items} if items else None
    target.mkdir(parents=True, exist_ok=True)
    comp, grades = [], []
    for r in rows:
        item = str(r.get("item_id"))
        if known is not None and known.get(item) != r.get("prompt"):
            raise EvalError(f"published row {item} does not match the eval item")
        usage = {k: r.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens") if r.get(k) is not None}
        response = r.get("response") or ""
        comp.append({"id": item, "response": response, "reasoning": r.get("reasoning") or "", "status": r.get("status"),
                     "finish_reason": r.get("finish_reason"), "truncated": bool(response) and r.get("finish_reason") == "length",
                     "usage": usage, "seeded": bool(r.get("seeded")), "error": None})
        if r.get("graded") and r.get("fact_score") is not None:
            grades.append({"id": item, "judge": "grok-4.7", "score": r["fact_score"], "propaganda": r["propaganda"],
                           "refused": bool(r.get("refused")), **{k: r.get(k) for k in GRADE_COUNTS}})
    for name, data in (("completions.jsonl", comp), ("grades.jsonl", grades)):
        (target / name).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data), encoding="utf-8")
    (target / "manifest.json").write_text(json.dumps({"schema": SCHEMA, "model": model, "profile": "card", "settings": PROFILES["card"],
                                                      "send_seed": False, "items_sha256": ITEMS_SHA256, "content_is_final": False,
                                                      "reference": True}, indent=1) + "\n")
    return {"rows": len(comp), "graded": len(grades)}


def fetch(out: Path, which: Sequence[str] = (), items: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    result = {}
    for label, ref in PUBLISHED.items():
        if which and label not in which:
            continue
        cached = cache_root() / "published" / ref["revision"] / "completions.jsonl"
        if not cached.is_file():
            download(f"{hf_endpoint()}/datasets/{ref['dataset']}/resolve/{ref['revision']}/data/completions.jsonl", cached)
        rows = [json.loads(x) for x in cached.read_text(encoding="utf-8").splitlines() if x.strip()]
        result[label] = convert(rows, run_dir(out, label, "card", 1), ref["model"], items)
    return result
