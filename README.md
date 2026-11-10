# propaganda-eval

Run [Nous Research's Censorship and Propaganda Eval](https://huggingface.co/datasets/NousResearch/Propaganda_Eval)
on your own model, judge it the same way the published scores were judged, and read every answer in a local report.

The eval, its 803 prompts and its judge prompt are by **Nous Research** (MIT). This repo downloads them at a pinned
revision and checks their sha256; it copies none of them. See [NOTICE.md](NOTICE.md).

Each prompt is about a topic the Chinese government censors or presents through an official account. The judge
gives two independent grades per answer:

- **Fact score, 0 to 5**: does the answer state the documented facts? Higher is better.
- **Propaganda, 0 to 2**: does the answer adopt the official account as its own? The headline is the share of items
  graded 1 or 2. Lower is better.

Reference results for three small models are at https://propaganda-eval.vercel.app.

## Fast path

Python 3.10 or newer. Your model behind any OpenAI-compatible server on port 8080.

```bash
pip install git+https://github.com/Balllvin/propaganda-eval
propaganda-eval generate --base-url http://127.0.0.1:8080/v1 --model your-model --label mine --smoke
propaganda-eval generate --base-url http://127.0.0.1:8080/v1 --model your-model --label mine --concurrency 16
propaganda-eval judge --label mine                       # prints the cost; add --spend --max-usd 25 to run it
propaganda-eval fetch-published && propaganda-eval report  # open runs/report/index.html
```

`--smoke` runs 5 items under the label `mine-smoke`, so it never mixes with the real run. The judge needs
`XAI_API_KEY` in the environment. Run `propaganda-eval score --label mine` for the numbers as JSON.

## Plug in a model

Any OpenAI-compatible endpoint works: hosted APIs (OpenAI, OpenRouter, Together, Fireworks and similar) and local or
cloud servers (vLLM, SGLang, llama.cpp server, Ollama, LM Studio, `mlx_lm.server`). Give flags, or put them in a TOML
file and pass `--config`. Templates are in [examples/configs](examples/configs):

| File | For |
|---|---|
| `mlx-mac.toml` | `mlx_lm.server` on a Mac |
| `vllm-cloud.toml` | vLLM or SGLang on a cloud GPU |
| `api-provider.toml` | a hosted API |
| `ollama.toml` | Ollama, LM Studio, llama.cpp server |
| `local-judge.toml` | a judge other than grok-4.7 |

Model options (flag, or key under `[model]`):

| Option | Meaning |
|---|---|
| `--base-url`, `base_url` | ends in `/v1` |
| `--model`, `model` | the name the server expects |
| `--api-key-env`, `api_key_env` | the name of the env var holding the key; keys are never read from files |
| `--no-thinking-param`, `no_thinking_param` | do not send `chat_template_kwargs` (hosted APIs reject it) |
| `omit_params` | request fields to drop, such as `["top_p"]` |
| `extra_body` | fields merged into every request, such as a provider's reasoning switch |
| `--content-is-final`, `content_is_final` | the server strips reasoning and returns only the answer |
| `--adapter`, `adapter` | `path/to/file.py:function` instead of an endpoint |

Reasoning is found wherever the server puts it: a `reasoning_content` or `reasoning` field, `<think>` tags in the
content, or a template that already opened `<think>`, so the content reads `reasoning</think>answer`. With thinking on
and no closing tag, the model did not finish thinking and the item has no answer. If your server strips the reasoning
and returns only the answer, set `content_is_final`. Only the final answer goes to the judge.

**Anything else** (a local transformers model, a custom API): write an adapter, one function.

```python
def complete(prompt: str, settings: dict) -> str | dict:
    # settings: temperature, top_p (may be None), max_tokens, thinking, seed (may be None)
    return {"content": "...", "reasoning": "...", "finish_reason": "stop", "usage": {"completion_tokens": 10}}
```

See [examples/adapters/echo_adapter.py](examples/adapters/echo_adapter.py) and
[examples/adapters/transformers_adapter.py](examples/adapters/transformers_adapter.py). Raise an exception to retry the item.

## Profiles

Compare only runs with the same profile.

| Profile | Thinking | Temperature | top_p | Token cap | Use |
|---|---|---|---|---|---|
| `card` | on | 1.0 | 0.95 | 32,768 | a reasoning model as it is meant to run; our reference runs |
| `published` | off | 0.6 | not sent | 32,768 | the Nous setting for most open models on their leaderboard |

Every item is one user turn, no system prompt, no tools.

## How a run works

- **Resume.** Rerun the same command; finished items are skipped, failed ones are retried. Changing the model or
  decoding settings under one label is refused.
- **Concurrency.** `--concurrency N` parallel requests. Total throughput rose with more streams for us: on a 1.7B
  model on MLX, 32 streams gave about 800 tokens per second against 238 at 4.
- **No seed.** No per-request seed is sent by default. A seeded request made our MLX server decode one request at a
  time, so it turned off batching. `--seed` sends one if you need it.
- **First three items one at a time.** A new run sends three items alone first and stops if none has a final answer,
  so a server that strips or truncates thinking does not cost a whole run.
- **Retries.** Network errors, timeouts, 429 and 5xx are retried with backoff, 5 tries. Other errors are recorded and
  redone on the next resume.
- **Several runs.** `--runs 3` writes `run1` to `run3`; `score` gives the mean and spread across them.

Files: `runs/<label>/<profile>/run<N>/` holds `manifest.json`, `completions.jsonl` (answer, reasoning, finish reason,
token use) and `grades.jsonl` (the judge's verdicts).

## Judge and cost

The default judge is **grok-4.7 at temperature 0 through the xAI API**, with the published judge prompt, 1,000 token
cap and verdict check. That is what the Nous leaderboard and our reference runs used, so scores are comparable.

```bash
export XAI_API_KEY=...
propaganda-eval judge --label mine                              # plan: calls, typical and worst case dollars
propaganda-eval judge --label mine --spend --max-usd 25         # spend, never more than $25
```

Without `--spend` and `--max-usd` nothing is sent to a paid judge. The cap counts the worst case cost of calls in
flight, so it is not passed. Spend is logged to `judge_spend.jsonl`. Judging all 803 answers with grok-4.7 cost us
$17.90 to $19.86 per model at $2 per million input tokens and $6 per million output tokens (xAI price list,
2026-11-09). Long answers cost more. Check the current price before a big run.

Any OpenAI-compatible judge also works: `--judge-url`, `--judge-model`, `--judge-key-env`, and
`--judge-price IN OUT` (USD per million tokens) for a paid one. A judge on localhost is free and needs no
`--spend`. **Scores from any judge other than grok-4.7 are not comparable** with the Nous leaderboard, the reference
runs, or runs from a different judge. The tool warns, the report says so, and `compare` refuses to mix judges.

## Read the results

```bash
propaganda-eval score --label mine                   # JSON
propaganda-eval compare --base ref-lfm3-base --cand mine
propaganda-eval report                               # runs/report/index.html
```

`score` gives, per run: `fact_mean` and `any_propaganda` with 95% intervals (`fact_mean_ci95`,
`any_propaganda_ci95`), the counts behind them, refusals, unanswered items, judge failures, finish reasons, and the
means by question type and by topic. Every run is scored over all 803 items: an unanswered or ungraded item counts as
fact 0 and propaganda 0, as in the Nous scorer. Intervals resample the 405 source documents whole, because items from
one source move together.

`compare` pairs the two labels item by item and says `better`, `worse` or `within noise` for each metric, with the
interval of the difference and the smallest difference the comparison could detect (`mde`).

`report` writes a static folder you open in a browser. It never uploads or deploys anything. It shows the headline
numbers with intervals, tables by question type and topic, and every item with its prompt, required facts, reasoning,
answer and verdict, filtered by model, type, topic, fact score, propaganda level, status and text.

`fetch-published` adds our three reference runs as labels `ref-dory-nvfp4`, `ref-lfm3-abliterated` and
`ref-lfm3-base`, downloaded at pinned revisions from their public datasets. They used the `card` profile and
grok-4.7. How we ran each one, with real numbers, is in [examples/worked](examples/worked).

## Python API

```python
from pathlib import Path
import propaganda_eval as pe

data = pe.fetch_dataset()
items = pe.load_items(data)
complete = pe.make_complete(pe.Endpoint(base_url="http://127.0.0.1:8080/v1", model="your-model"))
pe.generate(items, complete, pe.run_dir(Path("runs"), "mine", "card", 1), model="your-model", profile="card",
            items_sha=pe.items_sha(data), concurrency=16)
```

## Develop

```bash
pip install -e '.[test]' && pytest
```

Tests use fake local servers and a fixture dataset; they make no network call and spend nothing.

## Credit and licence

Eval data, judge prompt and reference scorer: Nous Research,
[NousResearch/Propaganda_Eval](https://huggingface.co/datasets/NousResearch/Propaganda_Eval), MIT. Each item's
prompt and facts derive from a cited source under its own terms. This code: MIT, copyright Balllvin.
