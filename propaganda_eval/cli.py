"""Command line: generate, judge, score, compare, fetch-published, report. Run ``propaganda-eval -h``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import data as D
from . import judge as J
from . import html_report, published, score
from .generate import PROFILES, generate
from .models import Endpoint, make_complete
from .runs import run_dir, run_dirs

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

SMOKE_ITEMS = 5


def load_config(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def pick(flag: Any, cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    return flag if flag is not None else cfg.get(key, default)


def show(obj: Any) -> None:
    print(json.dumps(obj, indent=1, ensure_ascii=False))


def warn(text: str) -> None:
    print("WARNING: " + text, file=sys.stderr)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="propaganda-eval", description=__doc__)
    ap.add_argument("--config", type=Path, help="TOML file with [model], [run] and [judge] tables; flags win")
    ap.add_argument("--out", type=Path, help="runs folder (default ./runs)")
    ap.add_argument("--data-dir", type=Path, help="a local copy of the dataset instead of the pinned download")
    ap.add_argument("--unpinned", action="store_true", help="test fixtures only: skip the sha256 check of --data-dir")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser, runs: bool = True) -> None:
        p.add_argument("--label", help="a name for this model's runs")
        p.add_argument("--profile", choices=sorted(PROFILES))
        if runs:
            p.add_argument("--runs", type=int, help="independent runs (default 1)")
            p.add_argument("--limit", type=int, help="only the first N unfinished items per run")
            p.add_argument("--smoke", action="store_true", help=f"{SMOKE_ITEMS} items, one run, label suffix -smoke")

    g = sub.add_parser("generate", help="ask the model every item")
    common(g)
    g.add_argument("--base-url", help="OpenAI-compatible base URL ending in /v1")
    g.add_argument("--model", help="model name the server expects")
    g.add_argument("--api-key-env", help="name of the env var that holds the key (none for local servers)")
    g.add_argument("--adapter", help="path/to/file.py:function or module:function instead of an endpoint")
    g.add_argument("--concurrency", type=int, help="parallel requests (default 8)")
    g.add_argument("--seed", action="store_true", default=None, help="send a per-request seed (turns off batching on some servers)")
    g.add_argument("--content-is-final", action="store_true", default=None, help="the server strips reasoning; content is the answer")
    g.add_argument("--no-thinking-param", action="store_true", default=None, help="do not send chat_template_kwargs (hosted APIs)")
    g.add_argument("--max-tokens", type=int, help="override the profile's token cap (scores then not comparable)")

    j = sub.add_parser("judge", help="grade answers; shows the cost and spends nothing without --spend")
    common(j)
    j.add_argument("--judge-url", help=f"default {J.XAI_URL}")
    j.add_argument("--judge-model", help=f"default {J.DEFAULT_JUDGE}")
    j.add_argument("--judge-key-env", help="env var with the judge key (default XAI_API_KEY)")
    j.add_argument("--judge-price", type=float, nargs=2, metavar=("IN", "OUT"), help="USD per 1M tokens, for a judge with no known price")
    j.add_argument("--judge-workers", type=int, help=f"parallel judge calls (default {J.JUDGE_WORKERS})")
    j.add_argument("--spend", action="store_true", help="allow paid judge calls")
    j.add_argument("--max-usd", type=float, help="hard dollar cap for this command")

    s = sub.add_parser("score", help="fact mean, propaganda rate and intervals")
    common(s, runs=False)
    s.add_argument("--smoke", action="store_true")
    c = sub.add_parser("compare", help="paired difference of two labels")
    c.add_argument("--base", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--profile", choices=sorted(PROFILES), default="card")
    f = sub.add_parser("fetch-published", help="download the three reference runs as labels ref-*")
    f.add_argument("--only", nargs="*", default=[], choices=sorted(published.PUBLISHED))
    r = sub.add_parser("report", help="write a local static report")
    r.add_argument("--labels", nargs="*", default=[], help="labels to include (default all)")
    r.add_argument("--profile", choices=sorted(PROFILES))
    r.add_argument("--to", type=Path, help="report folder (default <out>/report)")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    cfg = load_config(a.config)
    mc, rc, jc = cfg.get("model", {}), cfg.get("run", {}), cfg.get("judge", {})
    out = a.out or Path(rc.get("out", "runs"))
    try:
        data = D.resolve_data(a.data_dir, pinned=not a.unpinned)
        items = D.load_items(data)
        if a.cmd == "fetch-published":
            show(published.fetch(out, a.only, items))
            return 0
        if a.cmd == "report":
            show(html_report.build(items, out, a.to or out / "report", a.labels, a.profile))
            return 0
        if a.cmd == "compare":
            show(score.compare(items, out, a.base, a.cand, a.profile))
            return 0
        label = pick(a.label, rc, "label")
        if not label:
            raise D.EvalError("give --label or [run] label")
        profile = pick(a.profile, rc, "profile", "card")
        smoke = getattr(a, "smoke", False) or rc.get("smoke", False)
        if smoke:
            label += "-smoke"
        if a.cmd == "score":
            result = score.report(items, out, label, profile)
            show({k: v for k, v in result.items() if k != "per_run"} | {"per_run": [
                {k: r[k] for k in r if k not in ("by_type", "by_category")} for r in result["per_run"]]})
            return 0
        runs = 1 if smoke else pick(a.runs, rc, "runs", 1)
        limit = SMOKE_ITEMS if smoke else pick(a.limit, rc, "limit")
        if a.cmd == "generate":
            ep = Endpoint(base_url=pick(a.base_url, mc, "base_url", ""), model=pick(a.model, mc, "model", ""),
                          api_key_env=pick(a.api_key_env, mc, "api_key_env", ""), adapter=pick(a.adapter, mc, "adapter", ""),
                          thinking_param="none" if pick(a.no_thinking_param, mc, "no_thinking_param", False) else mc.get("thinking_param", "chat_template_kwargs"),
                          omit_params=list(mc.get("omit_params", [])), extra_body=dict(mc.get("extra_body", {})),
                          content_is_final=bool(pick(a.content_is_final, mc, "content_is_final", False)),
                          timeout=float(mc.get("timeout", 3600)))
            complete = make_complete(ep)
            max_tokens = pick(a.max_tokens, rc, "max_tokens")
            if max_tokens:
                warn("a changed token cap makes scores not comparable with published ones")
            for run in range(1, runs + 1):
                d = run_dir(out, label, profile, run)
                tick = lambda n, t: print(f"\r{d.name}: {n}/{t}", end="", file=sys.stderr, flush=True)  # noqa: E731
                res = generate(items, complete, d, model=ep.model or ep.adapter, profile=profile, items_sha=D.items_sha(data),
                               run=run, concurrency=pick(a.concurrency, rc, "concurrency", 8), limit=limit,
                               send_seed=bool(pick(a.seed, rc, "seed", False)), content_is_final=ep.content_is_final,
                               max_tokens=max_tokens, base_url=ep.base_url, progress=tick)
                print(file=sys.stderr)
                show({"run": run, "dir": str(d), **res})
            return 0
        # judge
        url = pick(a.judge_url, jc, "url", J.XAI_URL)
        model = pick(a.judge_model, jc, "model", J.DEFAULT_JUDGE)
        price = tuple(a.judge_price) if a.judge_price else tuple(jc["price"]) if "price" in jc else J.PRICES.get(model.split("/")[-1])
        spend, cap = a.spend or bool(jc.get("spend", False)), pick(a.max_usd, jc, "max_usd")
        dirs = [d for d in run_dirs(out, label, profile) if int(d.name[3:]) <= runs] if a.runs or rc.get("runs") or smoke else run_dirs(out, label, profile)
        if not dirs:
            raise D.EvalError(f"no runs under {out / label / profile}; run generate first")
        template = D.load_template(data)
        by_id = {str(i["id"]): i for i in items}
        todo = J.pending(dirs, limit)
        decision = J.plan(url, model, spend, cap, price)
        if not decision["comparable"]:
            warn(f"judge {model} is not grok-4.7: scores are NOT comparable with the Nous leaderboard or the reference runs")
        estimate = J.estimate_usd([r for rows in todo.values() for r in rows], template, by_id, price or (0.0, 0.0))
        if decision["free"]:
            estimate = {**estimate, "typical_usd": 0.0, "worst_usd": 0.0, "note": "local judge, no charge"}
        show({"judge": model, "at": url, **estimate})
        if not decision["go"]:
            show({"plan_only": True, "to_spend": "add --spend --max-usd N (a hard cap)"})
            return 2
        key_env = pick(a.judge_key_env, jc, "key_env", "" if decision["free"] else "XAI_API_KEY")
        key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not key:
            raise D.EvalError(f"environment variable {key_env} is empty")
        budget = None if decision["free"] else J.Budget(cap, price, out / label / profile / "judge_spend.jsonl")
        call = J.openai_judge(url, model, key, budget)
        for d in dirs:
            show({"run": d.name, **J.judge_run(d, items, template, call, model, workers=pick(a.judge_workers, jc, "workers", J.JUDGE_WORKERS), limit=limit)})
        if budget:
            show({"spent_usd": round(budget.spent, 4), "cap_usd": cap})
        return 0
    except D.EvalError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
