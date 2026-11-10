"""A local static report of your runs: headline numbers with intervals, scores by question type and topic, and every
item with reasoning, answer and verdict. Open ``index.html`` in a browser; nothing is uploaded or deployed."""

from __future__ import annotations

import html
import json
import shutil
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Sequence

from .data import EvalError
from .published import PUBLISHED
from .runs import labels, latest, manifest, run_dirs
from .score import grades, score_run

VERDICT_KEYS = ("score", "propaganda", "refused", "facts_stated", "facts_missed", "facts_opposed", "frame_adopted",
                "facts_stated_count", "facts_missed_count", "facts_opposed_count", "note", "judge")


def _js(name: str, value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"window.__report=window.__report||{{}};window.__report[{json.dumps(name)}]={text};\n"


def model_payload(items: Sequence[Mapping[str, Any]], out: Path, label: str, profile: str, run: int = 1) -> dict[str, Any]:
    dirs = run_dirs(out, label, profile)
    if not dirs:
        raise EvalError(f"no runs for {label}/{profile}")
    d = next((x for x in dirs if x.name == f"run{run}"), dirs[0])
    comp, g = latest(d / "completions.jsonl"), grades(d)
    m = manifest(d)
    summary = score_run(items, d)
    others = [score_run(items, x) for x in dirs if x != d]
    rows = {}
    for item in items:
        i = str(item["id"])
        c = comp.get(i) or {}
        v = g.get(i)
        rows[i] = {"r": c.get("reasoning") or "", "a": c.get("response") or "", "s": c.get("status") or "missing",
                   "f": c.get("finish_reason"), "t": (c.get("usage") or {}).get("completion_tokens"),
                   "v": {k: v[k] for k in VERDICT_KEYS if v and k in v} if v else None}
    name = PUBLISHED[label]["name"] + " (reference)" if label in PUBLISHED else label
    return {"label": label, "name": name, "profile": profile, "model": m.get("model"), "run": d.name, "runs": len(dirs),
            "summary": summary, "fact_by_run": [summary["fact_mean"], *[o["fact_mean"] for o in others]], "rows": rows}


def build(items: Sequence[Mapping[str, Any]], out: Path, target: Path, only: Sequence[str] = (), profile: str | None = None) -> dict[str, Any]:
    pairs = [(lab, prof) for lab, prof in labels(out) if (not only or lab in only) and (profile is None or prof == profile)]
    if not pairs:
        raise EvalError(f"no runs to report under {out}")
    target.mkdir(parents=True, exist_ok=True)
    (target / "data").mkdir(exist_ok=True)
    assets = resources.files("propaganda_eval") / "report_assets"
    for name in ("report.css", "report.js"):
        shutil.copyfile(str(assets / name), target / name)
    index = [{"id": str(i["id"]), "p": i["prompt"], "q": str(i.get("question_description")), "c": [str(c) for c in i.get("categories") or []],
              "u": str(i.get("source_url") or ""), "rf": [str(f) for f in i.get("required_facts") or []]} for i in items]
    scripts = ["data/items.js"]
    (target / "data" / "items.js").write_text(_js("items", index), encoding="utf-8")
    models = []
    for n, (lab, prof) in enumerate(pairs):
        payload = model_payload(items, out, lab, prof)
        file = f"data/model{n}.js"
        (target / file).write_text(f"window.__report=window.__report||{{}};(window.__report.models=window.__report.models||[]).push("
                                   + json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/") + ");\n", encoding="utf-8")
        scripts.append(file)
        models.append(f"{lab}/{prof}")
    tags = "\n".join(f'<script src="{html.escape(s)}"></script>' for s in scripts)
    (target / "index.html").write_text(PAGE.replace("{{scripts}}", tags), encoding="utf-8")
    return {"report": str(target / "index.html"), "models": models}


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Propaganda Eval report</title>
<link rel="stylesheet" href="report.css">
</head>
<body>
<header class="wrap">
<h1>Censorship and Propaganda Eval</h1>
<p class="muted">Local report. Eval and judge prompt by <a href="https://huggingface.co/datasets/NousResearch/Propaganda_Eval">Nous Research</a> (MIT).
Fact score 0 to 5, higher is better. Propaganda rate is the share of items where the answer adopts the official account, lower is better.
Intervals are 95% and resample source documents whole.</p>
</header>
<main class="wrap">
<section><h2>Headline</h2><div id="cards" class="cards"></div><p id="warn" class="warn" hidden></p></section>
<section><h2>By question type</h2><div class="scroll"><table id="bytype"></table></div></section>
<section><h2>By topic</h2><div class="scroll"><table id="bycat"></table></div></section>
<section><h2>Every item</h2>
<form id="filters" class="filters">
<label>Model <select id="f-model"></select></label>
<label>Type <select id="f-type"><option value="">all</option></select></label>
<label>Topic <select id="f-cat"><option value="">all</option></select></label>
<label>Fact score <select id="f-fact"><option value="">all</option><option>0</option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option><option value="u">ungraded</option></select></label>
<label>Propaganda <select id="f-prop"><option value="">all</option><option>0</option><option>1</option><option>2</option></select></label>
<label>Status <select id="f-status"><option value="">all</option></select></label>
<label class="grow">Search <input id="f-q" type="search" placeholder="prompt, answer or id"></label>
</form>
<p id="count" class="muted"></p>
<ol id="list" class="items"></ol>
<button id="more" type="button" hidden>Show more</button>
</section>
</main>
{{scripts}}
<script src="report.js"></script>
</body>
</html>
"""
