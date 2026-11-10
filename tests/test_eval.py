from __future__ import annotations

import json
from pathlib import Path

import pytest

import propaganda_eval as pe
from propaganda_eval import cli, data as D, judge as J, published
from propaganda_eval.models import HTTPStatusError, extract, request_body, Endpoint

from conftest import ITEMS, TEMPLATE, Server


def run(args: list[str]) -> int:
    return cli.main(args)


def base(tmp_path: Path, dataset: Path) -> list[str]:
    return ["--out", str(tmp_path / "runs"), "--data-dir", str(dataset), "--unpinned"]


# ---- reasoning extraction -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reply,thinking,final,want", [
    ({"content": "a", "reasoning": "r"}, True, False, ("a", "r", "answered")),
    ({"content": "<think>r</think> a"}, True, False, ("a", "r", "answered")),
    ({"content": "r</think>a"}, True, False, ("a", "r", "answered")),
    ({"content": "<think>never closed"}, True, False, ("", "never closed", "no_final_answer")),
    ({"content": "only thought", "finish_reason": "length"}, True, False, ("", "only thought", "cap")),
    ({"content": "stripped answer"}, True, True, ("stripped answer", "", "answered")),
    ("plain", False, False, ("plain", "", "answered")),
    ({"content": [{"type": "text", "text": "x"}]}, False, False, ("x", "", "answered")),
])
def test_extract(reply, thinking, final, want):
    got = extract(reply, thinking, final)
    assert (got["response"], got["reasoning"], got["status"]) == want


def test_request_body_options():
    s = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 9, "thinking": True, "seed": None}
    body = request_body("p", "m", s, Endpoint())
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and "seed" not in body
    body = request_body("p", "m", s, Endpoint(thinking_param="none", omit_params=["top_p"], extra_body={"reasoning": {"effort": "high"}}))
    assert "chat_template_kwargs" not in body and "top_p" not in body and body["reasoning"] == {"effort": "high"}


# ---- generation -----------------------------------------------------------------------------------------------------

def test_generate_resume_and_statuses(tmp_path, dataset, model_server):
    args = base(tmp_path, dataset) + ["generate", "--base-url", model_server.url + "/v1", "--model", "m", "--label", "x", "--concurrency", "4"]
    assert run(args) == 0
    d = tmp_path / "runs/x/card/run1"
    rows = pe.runs.latest(d / "completions.jsonl") if hasattr(pe, "runs") else None
    from propaganda_eval.runs import latest
    rows = latest(d / "completions.jsonl")
    assert {k: v["status"] for k, v in rows.items()} == {"a1": "answered", "a2": "answered", "a3": "answered", "a4": "cap", "a5": "answered"}
    assert rows["a3"]["reasoning"] == "t3" and rows["a1"]["reasoning"] == "thought"
    assert all("seed" not in b for b in model_server.bodies)  # no seed by default: lets servers batch
    n = len(model_server.bodies)
    assert run(args) == 0 and len(model_server.bodies) == n  # resume skips finished items
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["settings"]["temperature"] == 1.0 and manifest["send_seed"] is False


def test_resume_refuses_changed_settings(tmp_path, dataset, model_server):
    args = base(tmp_path, dataset) + ["generate", "--base-url", model_server.url + "/v1", "--model", "m", "--label", "x"]
    assert run(args) == 0
    assert run(base(tmp_path, dataset) + ["generate", "--base-url", model_server.url + "/v1", "--model", "other", "--label", "x"]) == 2


def test_first_items_check_stops_a_broken_server(tmp_path):
    items = [{"id": str(i), "prompt": "p"} for i in range(10)]
    calls = []

    def complete(prompt, settings):
        calls.append(1)
        return {"content": "thinking with no end"}

    with pytest.raises(D.EvalError, match="no final answer"):
        pe.generate(items, complete, tmp_path / "r", model="m", profile="card", items_sha="x")
    assert len(calls) == 3


def test_retries_transient_then_records_error(tmp_path):
    attempts = {"n": 0}

    def flaky(prompt, settings):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise HTTPStatusError(503, "busy")
        return "fine"

    res = pe.generate([{"id": "1", "prompt": "p"}], flaky, tmp_path / "r", model="m", profile="published", items_sha="x", sleep=lambda s: None)
    assert res["status"] == {"answered": 1} and attempts["n"] == 3

    def bad(prompt, settings):
        raise HTTPStatusError(400, "bad request")

    calls = []
    row = pe.generate_one if hasattr(pe, "generate_one") else None
    from propaganda_eval.generate import generate_one
    out = generate_one({"id": "2", "prompt": "p"}, lambda p, s: calls.append(1) or bad(p, s), {"thinking": False, "seed": None}, False, lambda s: None)
    assert out["status"] == "error" and len(calls) == 1  # a 400 is not retried


def test_adapter_example(tmp_path, dataset):
    adapter = Path(__file__).resolve().parents[1] / "examples/adapters/echo_adapter.py"
    assert run(base(tmp_path, dataset) + ["generate", "--adapter", f"{adapter}:complete", "--label", "echo", "--profile", "published"]) == 0
    from propaganda_eval.runs import latest
    rows = latest(tmp_path / "runs/echo/published/run1/completions.jsonl")
    assert len(rows) == len(ITEMS) and all(r["status"] == "answered" for r in rows.values())


def test_smoke_mode(tmp_path, dataset, model_server):
    assert run(base(tmp_path, dataset) + ["generate", "--base-url", model_server.url + "/v1", "--model", "m", "--label", "s", "--smoke"]) == 0
    assert (tmp_path / "runs/s-smoke/card/run1/completions.jsonl").is_file()


# ---- judging --------------------------------------------------------------------------------------------------------

def _generated(tmp_path, dataset, server):
    assert run(base(tmp_path, dataset) + ["generate", "--base-url", server.url + "/v1", "--model", "m", "--label", "x"]) == 0


def test_paid_judge_is_plan_only_without_spend(tmp_path, dataset, model_server, judge_server, capsys):
    _generated(tmp_path, dataset, model_server)
    judge_server.bodies.clear()
    url = judge_server.url.replace("127.0.0.1", "judge.example") + "/v1"  # not local: treated as paid, never called
    assert run(base(tmp_path, dataset) + ["judge", "--label", "x", "--judge-url", url]) == 2
    out = capsys.readouterr().out
    assert '"plan_only": true' in out and '"calls": 4' in out and judge_server.bodies == []


def test_unknown_price_refused(tmp_path, dataset, model_server):
    _generated(tmp_path, dataset, model_server)
    assert run(base(tmp_path, dataset) + ["judge", "--label", "x", "--judge-url", "https://judge.example/v1", "--judge-model", "other"]) == 2


def test_local_judge_and_score(tmp_path, dataset, model_server, judge_server, capsys):
    _generated(tmp_path, dataset, model_server)
    assert run(base(tmp_path, dataset) + ["judge", "--label", "x", "--judge-url", judge_server.url + "/v1"]) == 0
    assert all(b["temperature"] == 0 and b["max_tokens"] == 1000 for b in judge_server.bodies)
    sent = [b["messages"][0]["content"] for b in judge_server.bodies]
    assert not any("thought" in s or "t2" in s for s in sent)  # reasoning never reaches the judge
    assert "M - m" in "".join(sent) and '{"literal": 1}' in "".join(sent)
    capsys.readouterr()
    assert run(base(tmp_path, dataset) + ["score", "--label", "x"]) == 0
    result = json.loads((tmp_path / "runs/x/card/report.json").read_text())
    r = result["per_run"][0]
    assert r["graded"] == 4 and r["ungraded_counted_as_0"] == 1 and r["fact_mean"] == 4.0 and r["any_propaganda"] == 0.8
    assert r["by_type"]["named fact"]["n"] == 3 and r["judges"] == ["grok-4.7"]


def test_non_grok_judge_warns(tmp_path, dataset, model_server, judge_server, capsys):
    _generated(tmp_path, dataset, model_server)
    assert run(base(tmp_path, dataset) + ["judge", "--label", "x", "--judge-url", judge_server.url + "/v1", "--judge-model", "llama"]) == 0
    assert "NOT comparable" in capsys.readouterr().err


def test_budget_cap_stops_paid_calls(tmp_path, judge_server):
    price = (2.0, 6.0)
    budget = J.Budget(0.02, price)
    call = J.openai_judge(judge_server.url + "/v1", "grok-4.7", "k", budget)
    made = 0
    with pytest.raises(J.BudgetReached):
        for _ in range(100):
            call("x" * 300)
            made += 1
    # each call bills 1000 in + 200 out = $0.0032; worst case reserve is ~$0.0062, so 3 to 5 calls fit under $0.02
    assert 2 <= made <= 6 and budget.spent <= 0.02 and len(judge_server.bodies) == made


def test_spend_path_uses_budget(tmp_path, dataset, model_server, judge_server, monkeypatch, capsys):
    _generated(tmp_path, dataset, model_server)
    monkeypatch.setenv("XAI_API_KEY", "test-key")
    monkeypatch.setattr(J, "is_local", lambda url: False)
    assert run(base(tmp_path, dataset) + ["judge", "--label", "x", "--judge-url", judge_server.url + "/v1", "--spend", "--max-usd", "1"]) == 0
    out = capsys.readouterr().out
    assert '"spent_usd": 0.0128' in out
    assert (tmp_path / "runs/x/card/judge_spend.jsonl").is_file()


def test_parse_verdict():
    assert J.parse_verdict("x " + json.dumps({**{"score": 3, "propaganda": 0, "refused": False}, "facts_stated": [], "facts_missed": [], "facts_opposed": []}))
    assert J.parse_verdict('{"score": 6, "propaganda": 0}') is None
    assert J.parse_verdict("no json") is None


# ---- compare, published, report, data ---------------------------------------------------------------------------------

def test_compare_and_report(tmp_path, dataset, model_server, judge_server, capsys):
    for label in ("a", "b"):
        assert run(base(tmp_path, dataset) + ["generate", "--base-url", model_server.url + "/v1", "--model", "m", "--label", label]) == 0
        assert run(base(tmp_path, dataset) + ["judge", "--label", label, "--judge-url", judge_server.url + "/v1"]) == 0
    capsys.readouterr()
    assert run(base(tmp_path, dataset) + ["compare", "--base", "a", "--cand", "b"]) == 0
    c = json.loads(capsys.readouterr().out)
    assert c["fact_mean_delta"] == 0 and c["fact_verdict"] == "within noise"
    assert run(base(tmp_path, dataset) + ["report"]) == 0
    folder = tmp_path / "runs/report"
    assert (folder / "index.html").is_file() and (folder / "report.js").is_file()
    assert "answer two" in (folder / "data/model0.js").read_text()


def test_published_conversion(tmp_path):
    rows = [{"item_id": "a1", "prompt": "field?", "reasoning": "r", "response": "x", "status": "answered", "finish_reason": "stop",
             "completion_tokens": 3, "graded": True, "fact_score": 4, "propaganda": 0, "refused": False,
             "facts_stated_count": 2, "facts_missed_count": 1, "facts_opposed_count": 0},
            {"item_id": "a2", "prompt": "tags?", "reasoning": "", "response": "", "status": "cap", "finish_reason": "length", "graded": False}]
    target = pe.run_dir(tmp_path, "ref-x", "card", 1)
    assert published.convert(rows, target, "repo/model", ITEMS) == {"rows": 2, "graded": 1}
    r = pe.score_run(ITEMS, target)
    assert r["graded"] == 1 and r["fact_mean"] == 0.8 and r["score_contradicts_lists"] == 0
    with pytest.raises(D.EvalError):
        published.convert([{**rows[0], "prompt": "changed"}], target, "m", ITEMS)


def test_download_checks_sha(tmp_path, monkeypatch):
    s = Server()
    s.files = {"data/items.jsonl": b"tampered\n", "judge/JUDGE_PROMPT.md": b"x", "LICENSE": b"MIT", "README.md": b"r"}
    try:
        monkeypatch.setenv("HF_ENDPOINT", s.url)
        monkeypatch.setenv("PROPAGANDA_EVAL_CACHE", str(tmp_path / "cache"))
        with pytest.raises(D.EvalError, match="sha256"):
            D.fetch_dataset()
        assert not (tmp_path / "cache/nous" / D.REVISION / "data/items.jsonl").exists()
    finally:
        s.close()


def test_pinned_local_copy_refused_when_different(dataset):
    with pytest.raises(D.EvalError, match="pinned"):
        D.resolve_data(dataset, pinned=True)


def test_example_configs_parse():
    root = Path(__file__).resolve().parents[1] / "examples"
    files = sorted(root.rglob("*.toml"))
    assert files
    for f in files:
        cfg = cli.load_config(f)
        assert "model" in cfg and "run" in cfg, f


# ---- which key goes to which judge host -------------------------------------------------------------------------------

def test_xai_key_default_only_for_api_x_ai():
    assert J.judge_key_env("https://api.x.ai/v1", None) == "XAI_API_KEY"
    assert J.judge_key_env("https://api.x.ai.evil.example/v1", None) == ""
    assert J.judge_key_env("https://openrouter.ai/api/v1", None) == ""
    assert J.judge_key_env("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY") == "OPENROUTER_API_KEY"


def _paid_judge(tmp_path, dataset, model_server, judge_server, monkeypatch, extra):
    _generated(tmp_path, dataset, model_server)
    monkeypatch.setenv("XAI_API_KEY", "xai-secret-should-not-leak")
    monkeypatch.setattr(J, "is_local", lambda url: False)  # treat the fake host as another paid provider
    return run(base(tmp_path, dataset) + ["judge", "--label", "x", "--judge-url", judge_server.url + "/v1",
                                          "--spend", "--max-usd", "1"] + extra)


def test_other_host_without_key_env_sends_no_key(tmp_path, dataset, model_server, judge_server, monkeypatch, capsys):
    assert _paid_judge(tmp_path, dataset, model_server, judge_server, monkeypatch, []) == 0
    assert judge_server.auth and all(h is None for h in judge_server.auth)
    assert "sending no key to 127.0.0.1" in capsys.readouterr().err


def test_other_host_with_explicit_key_env(tmp_path, dataset, model_server, judge_server, monkeypatch, capsys):
    monkeypatch.setenv("OTHER_JUDGE_KEY", "other-key")
    assert _paid_judge(tmp_path, dataset, model_server, judge_server, monkeypatch, ["--judge-key-env", "OTHER_JUDGE_KEY"]) == 0
    assert judge_server.auth and all(h == "Bearer other-key" for h in judge_server.auth)
    assert "the key in OTHER_JUDGE_KEY to 127.0.0.1" in capsys.readouterr().err
