"""Judging with the published judge prompt, verdict check and retry rule (Nous ``judge/grade.py``, MIT).

The default judge is grok-4.7 at temperature 0 through the xAI API, the judge behind the Nous leaderboard. Any other
OpenAI-compatible judge works, but its scores are not comparable with published ones. Paid calls need an explicit
spend flag and a dollar cap; the cap counts reserved worst case cost for calls in flight, so it is never passed.
Only the final answer goes to the judge, never the reasoning.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import threading
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .data import EvalError
from .models import Post, is_local, post_json
from .runs import latest, read_jsonl

XAI_URL = "https://api.x.ai/v1"
DEFAULT_JUDGE = "grok-4.7"
JUDGE_MAX_TOKENS = 1000  # grade.py default
JUDGE_WORKERS = 8  # grade.py default
# USD per 1M tokens. grok-4.7 under 200k prompt tokens, from docs.x.ai on 2026-11-09. Check before a big run.
PRICES: dict[str, tuple[float, float]] = {"grok-4.7": (2.00, 6.00)}
TYPICAL_VERDICT_TOKENS = 400

Judge = Callable[[str], str]


class BudgetReached(RuntimeError):
    pass


def host(url: str) -> str:
    return (urllib.parse.urlparse(url).hostname or "").lower()


def judge_key_env(url: str, explicit: str | None) -> str:
    """The env var whose key goes to the judge. XAI_API_KEY is the default only for api.x.ai itself, so an xAI key
    never reaches another host unless the user names it."""

    if explicit is not None:
        return explicit
    return "XAI_API_KEY" if host(url) == "api.x.ai" else ""


def comparable(judge_url: str, judge_model: str) -> bool:
    """Scores line up with the Nous table and the reference site only for grok-4.7 at temperature 0."""

    return judge_model.split("/")[-1] == DEFAULT_JUDGE


def build_prompt(template: str, item: Mapping[str, Any], response: str) -> str:
    markers = item.get("official_frame_markers") or []
    return template.format(prompt=item["prompt"], required_facts="\n".join(f"- {f}" for f in item["required_facts"]),
                           markers="\n".join(f"- {m}" for m in markers) if markers else "none",
                           censorship_note="none", propaganda_note="none", response=response)


def parse_verdict(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        o = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(o, dict):
        return None
    if o.get("score") not in (0, 1, 2, 3, 4, 5) or o.get("propaganda") not in (0, 1, 2):
        return None
    if not isinstance(o.get("refused"), bool):
        return None
    if not all(isinstance(o.get(k), list) for k in ("facts_stated", "facts_missed", "facts_opposed")):
        return None
    return o


def pending(directories: Sequence[Path], limit: int | None = None) -> dict[Path, list[dict[str, Any]]]:
    """Per run, the answered rows without a valid grade."""

    out = {}
    for d in directories:
        done = {str(r["id"]) for r in read_jsonl(d / "grades.jsonl") if r.get("score") is not None}
        out[d] = [r for r in latest(d / "completions.jsonl").values() if (r.get("response") or "").strip() and r["id"] not in done][:limit]
    return out


def estimate_usd(rows: Sequence[Mapping[str, Any]], template: str, items: Mapping[str, Mapping[str, Any]],
                 price: tuple[float, float]) -> dict[str, Any]:
    """Dollars for a judge pass from the stored answers: 4 characters per token typical, 3 worst; the verdict about
    400 tokens typical, the 1,000 token cap worst (reasoning tokens bill as output)."""

    p_in, p_out = price
    chars = [len(build_prompt(template, items[r["id"]], r["response"])) for r in rows]
    typical = sum(c / 4 * p_in + TYPICAL_VERDICT_TOKENS * p_out for c in chars) / 1e6
    worst = sum(c / 3 * p_in + JUDGE_MAX_TOKENS * p_out for c in chars) / 1e6
    return {"calls": len(rows), "typical_usd": round(typical, 2), "worst_usd": round(worst, 2),
            "price_per_1m_tokens": {"input": p_in, "output": p_out}}


class Budget:
    """A dollar cap shared by all judge threads. Each call reserves its worst case first and settles to the billed
    usage after; a call that would pass the cap is not made."""

    def __init__(self, cap_usd: float, price: tuple[float, float], ledger: Path | None = None):
        self.cap, self.price, self.ledger = cap_usd, price, ledger
        self.spent = self.reserved = 0.0
        self.lock = threading.Lock()

    def worst(self, prompt: str) -> float:
        return (len(prompt) / 3 * self.price[0] + JUDGE_MAX_TOKENS * self.price[1]) / 1e6

    def reserve(self, prompt: str) -> float:
        cost = self.worst(prompt)
        with self.lock:
            if self.spent + self.reserved + cost > self.cap:
                raise BudgetReached(f"judge cap ${self.cap:.2f} reached (spent ${self.spent:.4f})")
            self.reserved += cost
        return cost

    def settle(self, reserved: float, usage: Mapping[str, Any] | None) -> None:
        if usage and usage.get("prompt_tokens") is not None:
            cost = (usage.get("prompt_tokens", 0) * self.price[0] + usage.get("completion_tokens", 0) * self.price[1]) / 1e6
        else:
            cost = reserved  # no usage reported: count the worst case
        with self.lock:
            self.reserved -= reserved
            self.spent += cost
            if self.ledger:
                with self.ledger.open("a") as fh:
                    fh.write(json.dumps({"t": round(time.time(), 1), "usd": round(cost, 6), "usage": dict(usage or {})}) + "\n")


def openai_judge(base_url: str, model: str, api_key: str = "", budget: Budget | None = None, post: Post = post_json,
                 max_tokens: int = JUDGE_MAX_TOKENS) -> Judge:
    url = base_url.rstrip("/") + "/chat/completions"

    def call(prompt: str) -> str:
        reserved = budget.reserve(prompt) if budget else 0.0
        usage = None
        try:
            body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
            data = post(url, body, api_key, 300)
            usage = data.get("usage")
            return (data["choices"][0]["message"].get("content") or "").strip()
        finally:
            if budget:
                budget.settle(reserved, usage)

    return call


def judge_run(directory: Path, items: Sequence[Mapping[str, Any]], template: str, judge: Judge, judge_name: str,
              workers: int = JUDGE_WORKERS, limit: int | None = None, sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    by_id = {str(i["id"]): i for i in items}
    path, lock = directory / "grades.jsonl", threading.Lock()
    todo = pending([directory], limit)[directory]
    stop = threading.Event()
    counts = {"graded": 0, "failed": 0}

    def work(row: Mapping[str, Any]) -> None:
        if stop.is_set():
            return
        verdict, err = None, ""
        for attempt in range(5):
            try:
                verdict = parse_verdict(judge(build_prompt(template, by_id[row["id"]], row["response"])))
                if verdict:
                    break
                err = "unparseable verdict"
            except BudgetReached:
                stop.set()
                return
            except Exception as exc:  # noqa: BLE001 - one bad call is one failed grade, redone on resume
                err = repr(exc)[:200]
            if attempt < 4:
                sleep(min(60, 2 ** attempt))
        out = {"id": row["id"], "judge": judge_name, **(verdict or {"score": None, "error": err})}
        with lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            counts["graded" if verdict else "failed"] += 1

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, todo))
    return {"to_grade": len(todo), **counts, "stopped_on_budget": stop.is_set()}


def plan(judge_url: str, judge_model: str, spend: bool, max_usd: float | None, price: tuple[float, float] | None) -> dict[str, Any]:
    """Decide whether this judge call may go ahead. Returns the reason when it may not."""

    free = is_local(judge_url)
    if not free and price is None:
        raise EvalError(f"no price known for judge {judge_model!r}; give --judge-price IN OUT (USD per 1M tokens)")
    ok = free or (spend and max_usd is not None and max_usd > 0)
    return {"free": free, "go": ok, "comparable": comparable(judge_url, judge_model)}
