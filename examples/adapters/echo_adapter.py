"""The smallest adapter: one function. Use it for a model that has no OpenAI-compatible server.

    propaganda-eval generate --adapter examples/adapters/echo_adapter.py:complete --label echo --smoke

Replace the body with a call to your model. ``settings`` has temperature, top_p (may be None), max_tokens,
thinking (bool) and seed (may be None).
"""


def complete(prompt: str, settings: dict) -> dict:
    # Return a string, or a dict. Reasoning may go in "reasoning", or inside <think>...</think> in "content".
    return {"content": "I do not know. " + prompt[:40], "reasoning": "", "finish_reason": "stop",
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": 8}}
