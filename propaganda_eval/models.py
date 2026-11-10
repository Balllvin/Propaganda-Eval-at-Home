"""How a prompt reaches a model: any OpenAI-compatible endpoint, or a Python adapter.

An adapter is one function::

    def complete(prompt: str, settings: dict) -> str | dict:
        ...

``settings`` has ``temperature``, ``top_p`` (may be None), ``max_tokens``, ``thinking`` (bool) and ``seed`` (may be
None). Return the reply text, or a dict with any of ``content``, ``reasoning``, ``finish_reason`` and ``usage``
(``{"prompt_tokens": int, "completion_tokens": int}``). Raise an exception to have the item retried.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .data import EvalError

Post = Callable[[str, dict, str, float], dict]
Complete = Callable[[str, dict], "str | dict"]


class HTTPStatusError(RuntimeError):
    def __init__(self, code: int, body: str):
        super().__init__(f"HTTP {code}: {body[:300]}")
        self.code = code


def post_json(url: str, body: dict, api_key: str, timeout: float) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": "propaganda-eval"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise HTTPStatusError(exc.code, exc.read().decode("utf-8", "replace")) from None


def is_local(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    return host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") or host.endswith(".localhost")


def transient(exc: Exception) -> bool:
    """Retry network errors, timeouts, rate limits and server errors; give up on other HTTP errors."""

    return not isinstance(exc, HTTPStatusError) or exc.code in (408, 409, 425, 429, 500, 502, 503, 504)


@dataclass
class Endpoint:
    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    thinking_param: str = "chat_template_kwargs"  # how the thinking switch is sent: chat_template_kwargs or none
    omit_params: list[str] = field(default_factory=list)  # e.g. ["top_p"] for an API that rejects it
    extra_body: dict[str, Any] = field(default_factory=dict)  # merged into every request body
    content_is_final: bool = False  # the server strips reasoning and returns only the answer
    adapter: str = ""  # "path/to/file.py:function" or "package.module:function"
    timeout: float = 3600

    def api_key(self) -> str:
        if not self.api_key_env:
            return ""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise EvalError(f"environment variable {self.api_key_env} is empty")
        return key


def request_body(prompt: str, model: str, settings: Mapping[str, Any], ep: Endpoint) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": settings["max_tokens"], "temperature": settings["temperature"]}
    if settings.get("top_p") is not None:
        body["top_p"] = settings["top_p"]
    if settings.get("seed") is not None:
        body["seed"] = settings["seed"]
    if ep.thinking_param == "chat_template_kwargs":
        body["chat_template_kwargs"] = {"enable_thinking": bool(settings["thinking"])}
    for key in ep.omit_params:
        body.pop(key, None)
    body.update(ep.extra_body)
    return body


def openai_complete(ep: Endpoint, post: Post = post_json) -> Complete:
    if not ep.base_url or not ep.model:
        raise EvalError("an OpenAI-compatible model needs base_url and model")
    url = ep.base_url.rstrip("/") + "/chat/completions"
    key = ep.api_key()

    def complete(prompt: str, settings: dict) -> dict:
        data = post(url, request_body(prompt, ep.model, settings, ep), key, ep.timeout)
        choice = data["choices"][0]
        msg = choice.get("message") or {}
        return {"content": msg.get("content"), "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
                "finish_reason": choice.get("finish_reason"), "usage": data.get("usage") or {}}

    return complete


def load_adapter(spec: str) -> Complete:
    """``file.py:function`` or ``module:function``."""

    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise EvalError(f"adapter {spec!r} must look like path/to/file.py:function or module:function")
    if target.endswith(".py"):
        path = Path(target).resolve()
        if not path.is_file():
            raise EvalError(f"adapter file {path} not found")
        mod_name = "propaganda_eval_adapter_" + path.stem
        loader = importlib.util.spec_from_file_location(mod_name, path)
        assert loader and loader.loader
        module = importlib.util.module_from_spec(loader)
        sys.modules[mod_name] = module
        loader.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    fn = getattr(module, name, None)
    if not callable(fn):
        raise EvalError(f"adapter {spec!r}: {name} is not a function")
    return fn


def make_complete(ep: Endpoint, post: Post = post_json) -> Complete:
    return load_adapter(ep.adapter) if ep.adapter else openai_complete(ep, post)


def extract(reply: "str | Mapping[str, Any]", thinking: bool, content_is_final: bool = False) -> dict[str, Any]:
    """Split a reply into reasoning and final answer.

    Reasoning may come in a separate field (``reasoning_content`` or ``reasoning``), inside ``<think>`` tags, or after a
    chat template that already opened ``<think>``, so the content is ``reasoning</think>answer``. With thinking on and
    no closing tag, the model never finished thinking and there is no answer, unless the server strips reasoning
    (``content_is_final``).
    """

    if isinstance(reply, str):
        reply = {"content": reply}
    content = reply.get("content") or ""
    if isinstance(content, list):
        content = "".join(str(p.get("text") or "") for p in content if isinstance(p, Mapping))
    reasoning = str(reply.get("reasoning") or "")
    finish = reply.get("finish_reason")
    answer = content
    if "</think>" in content:
        head, answer = content.rsplit("</think>", 1)
        reasoning = reasoning or head.replace("<think>", "", 1).strip()
    elif "<think>" in content:
        reasoning, answer = reasoning or content.split("<think>", 1)[1].strip(), ""
    elif thinking and not reasoning and not content_is_final:
        reasoning, answer = content, ""
    answer = answer.strip()
    status = "answered" if answer else "cap" if finish == "length" else "no_final_answer"
    return {"response": answer, "reasoning": reasoning, "status": status, "finish_reason": finish,
            "truncated": bool(answer) and finish == "length", "usage": dict(reply.get("usage") or {})}
