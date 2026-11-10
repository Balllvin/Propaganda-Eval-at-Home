"""Generation: one user turn per item, no system prompt, no tools. Resumable, concurrent, retried."""

from __future__ import annotations

import collections
import concurrent.futures as cf
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .data import EvalError
from .models import Complete, extract, transient
from .runs import SCHEMA, latest, read_jsonl, write_manifest

PROFILES: dict[str, dict[str, Any]] = {
    # Thinking on, as a reasoning model is meant to be used.
    "card": {"temperature": 1.0, "top_p": 0.95, "thinking": True, "max_tokens": 32768},
    # The setting Nous reports for most open models: thinking off, temperature 0.6.
    "published": {"temperature": 0.6, "top_p": None, "thinking": False, "max_tokens": 32768},
}
FIRST_CHECK = 3
RETRIES = 5


def settings_for(profile: str, max_tokens: int | None = None) -> dict[str, Any]:
    if profile not in PROFILES:
        raise EvalError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    s = dict(PROFILES[profile])
    if max_tokens:
        s["max_tokens"] = max_tokens
    return s


def generate_one(item: Mapping[str, Any], complete: Complete, settings: Mapping[str, Any], content_is_final: bool,
                 sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    last = ""
    for attempt in range(RETRIES):
        started = time.time()
        try:
            reply = complete(item["prompt"], dict(settings))
            parsed = extract(reply, bool(settings["thinking"]), content_is_final)
            return {"id": str(item["id"]), **parsed, "seconds": round(time.time() - started, 2),
                    "seeded": settings.get("seed") is not None, "error": None}
        except Exception as exc:  # noqa: BLE001 - one failed item is one error row, retried on resume
            last = repr(exc)[:300]
            if not transient(exc):
                break
            if attempt < RETRIES - 1:
                sleep(min(30, 2 ** attempt))
    return {"id": str(item["id"]), "response": "", "reasoning": "", "status": "error", "finish_reason": None,
            "truncated": False, "usage": {}, "seconds": 0, "seeded": settings.get("seed") is not None, "error": last}


def generate(items: Sequence[Mapping[str, Any]], complete: Complete, directory: Path, *, model: str, profile: str,
             items_sha: str, run: int = 1, concurrency: int = 8, limit: int | None = None, send_seed: bool = False,
             content_is_final: bool = False, max_tokens: int | None = None, base_url: str = "",
             sleep: Callable[[float], None] = time.sleep, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    """Generate every unfinished item into ``directory/completions.jsonl``.

    A per-request seed makes some servers (MLX among them) decode requests one at a time, so the default sends none and
    lets the server batch. A new run sends its first three items one at a time and stops if none has a final answer.
    """

    settings = settings_for(profile, max_tokens)
    seed = 1000 + run if send_seed else None
    write_manifest(directory, {"schema": SCHEMA, "model": model, "base_url": base_url, "profile": profile,
                               "settings": {k: v for k, v in settings.items()}, "send_seed": send_seed,
                               "items_sha256": items_sha, "content_is_final": content_is_final, "concurrency": concurrency})
    settings["seed"] = seed
    path = directory / "completions.jsonl"
    done = {str(r["id"]) for r in read_jsonl(path) if not r.get("error")}
    todo = [i for i in items if str(i["id"]) not in done][:limit]
    lock = threading.Lock()
    counter = [0]

    def work(item: Mapping[str, Any]) -> dict[str, Any]:
        row = generate_one(item, complete, settings, content_is_final, sleep)
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            counter[0] += 1
            if progress:
                progress(counter[0], len(todo))
        return row

    first = [work(i) for i in todo[:FIRST_CHECK]] if not done else []
    if first and not any(r["status"] == "answered" for r in first):
        kinds = dict(collections.Counter(r["status"] for r in first))
        errors = [r["error"] for r in first if r.get("error")][:1]
        hint = ("If the server strips reasoning and returns only the answer, set content_is_final. If the model never "
                "closes its thinking, raise max_tokens or use the published profile.")
        raise EvalError(f"no final answer in the first {len(first)} items {kinds} {errors}: check the server. {hint}")
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        list(pool.map(work, todo[len(first):]))
    rows = latest(path)
    return {"rows": len(rows), "generated_now": len(todo), "status": dict(collections.Counter(r["status"] for r in rows.values()))}
