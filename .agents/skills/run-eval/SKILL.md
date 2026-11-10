---
name: run-eval
description: Use when asked to run the Propaganda Eval on a model, from server check to judged score.
---

# Run the eval on a model

1. Confirm the server answers: `curl -s BASE_URL/models`. For a local server, note its batch or parallel slot count.
2. Pick a config from `examples/configs/` that matches the server, copy it, fill `base_url`, `model`, `label`.
   Hosted API: set `no_thinking_param = true` and `api_key_env`. Choose the profile: `card` (thinking on) or
   `published` (thinking off, the Nous setting).
3. Smoke test: `propaganda-eval --config my.toml generate --smoke`. Then open
   `runs/<label>-smoke/<profile>/run1/completions.jsonl` and check `status` is `answered` and `response` holds the
   answer, not the thinking. If the run stops with "no final answer": the server may strip reasoning
   (`content_is_final = true`) or the model may not close its thinking (try `published`).
4. Full run: `propaganda-eval --config my.toml generate`. Set `concurrency` to the server's batch size. Do not send
   a seed. If it stops or memory runs short, restart the server smaller and rerun with a lower `--concurrency`:
   finished items are kept.
5. Cost: `propaganda-eval --config my.toml judge`. Show the user `calls`, `typical_usd` and `worst_usd`. Stop here
   until the user approves a cap.
6. With approval: `propaganda-eval --config my.toml judge --spend --max-usd CAP` (needs `XAI_API_KEY`). Rerun the
   same command to retry failed grades; graded items are skipped.
7. `propaganda-eval --config my.toml score`, then follow `read-results`.
