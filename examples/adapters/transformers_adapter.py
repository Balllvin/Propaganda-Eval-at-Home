"""An adapter for a Hugging Face transformers model loaded in process. Needs ``pip install transformers torch``.

    MODEL_ID=your/model propaganda-eval generate --adapter examples/adapters/transformers_adapter.py:complete \\
        --label my-model --concurrency 1

Run with --concurrency 1: this simple version decodes one prompt at a time.
"""

import os
import threading

_lock = threading.Lock()
_state: dict = {}


def _load():
    if not _state:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = os.environ["MODEL_ID"]
        _state["tok"] = AutoTokenizer.from_pretrained(name)
        _state["model"] = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", device_map="auto")
    return _state["tok"], _state["model"]


def complete(prompt: str, settings: dict) -> dict:
    with _lock:
        tok, model = _load()
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=settings["thinking"])
        inputs = tok(text, return_tensors="pt").to(model.device)
        kwargs = {"max_new_tokens": settings["max_tokens"], "do_sample": settings["temperature"] > 0,
                  "temperature": settings["temperature"]}
        if settings.get("top_p") is not None:
            kwargs["top_p"] = settings["top_p"]
        out = model.generate(**inputs, **kwargs)
        new = out[0][inputs["input_ids"].shape[1]:]
        finish = "length" if len(new) >= settings["max_tokens"] else "stop"
        # Thinking templates leave "reasoning</think>answer"; the eval splits it.
        return {"content": tok.decode(new, skip_special_tokens=True), "finish_reason": finish,
                "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": int(len(new))}}
