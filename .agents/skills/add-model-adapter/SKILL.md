---
name: add-model-adapter
description: Use when a model or provider does not work through the OpenAI-compatible options, or to add a new provider template.
---

# Add a provider or adapter

First try config options in `[model]`; most providers need only these:

- Request rejected for an unknown field: `no_thinking_param = true`, or `omit_params = ["top_p"]`.
- Provider has its own reasoning switch: `extra_body = { ... }`.
- Reasoning hidden, only the answer returned: `content_is_final = true`.

Otherwise write an adapter: one function in a `.py` file.

```python
def complete(prompt: str, settings: dict) -> str | dict:
    # settings: temperature, top_p (None means do not send), max_tokens, thinking (bool), seed (None means none)
    return {"content": text, "reasoning": thoughts, "finish_reason": "stop" or "length",
            "usage": {"prompt_tokens": n, "completion_tokens": m}}
```

- Return `finish_reason = "length"` when the token cap was hit; it marks capped items.
- Put reasoning in `reasoning`, or leave it in `content` inside `<think>` tags; `extract` in
  `propaganda_eval/models.py` splits it.
- Raise on failure; the item is retried with backoff.
- Must be thread safe if run with `--concurrency` above 1.
- Run: `propaganda-eval generate --adapter path/to/file.py:complete --label NAME --smoke`.
- Copy `examples/adapters/echo_adapter.py` as a start. If you add a template to the repo, add a test with a fake
  in `tests/` and keep it offline.
