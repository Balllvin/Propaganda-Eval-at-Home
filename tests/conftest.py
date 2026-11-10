"""Fake OpenAI-compatible servers and a tiny fixture dataset. No test touches the network beyond 127.0.0.1."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

TEMPLATE = "Q {prompt}\nF {required_facts}\nM {markers}\nC {censorship_note} P {propaganda_note}\nA {response}\n{{\"literal\": 1}}\n"
ITEMS = [
    {"id": "a1", "prompt": "field?", "required_facts": ["f1", "f2"], "question_description": "named fact", "categories": ["x"], "source_url": "u1"},
    {"id": "a2", "prompt": "tags?", "required_facts": ["f3"], "question_description": "frame testing", "categories": ["x", "y"], "source_url": "u1",
     "official_frame_markers": ["m"]},
    {"id": "a3", "prompt": "opened?", "required_facts": ["f4"], "question_description": "named fact", "categories": ["y"], "source_url": "u2"},
    {"id": "a4", "prompt": "cap?", "required_facts": ["f5"], "question_description": "closed exit", "categories": ["y"], "source_url": "u3"},
    {"id": "a5", "prompt": "plain?", "required_facts": ["f6"], "question_description": "named fact", "categories": ["x"], "source_url": "u3"},
]
VERDICT = {"score": 5, "propaganda": 1, "refused": False, "facts_stated": ["f"], "facts_missed": [], "facts_opposed": [], "note": "n"}


def model_reply(prompt: str) -> dict:
    """Each fixture prompt exercises one way a server returns reasoning."""

    if prompt == "field?":
        return {"message": {"content": "answer one", "reasoning_content": "thought"}, "finish_reason": "stop"}
    if prompt == "tags?":
        return {"message": {"content": "<think>t2</think>answer two"}, "finish_reason": "stop"}
    if prompt == "opened?":
        return {"message": {"content": "t3</think>answer three"}, "finish_reason": "stop"}
    if prompt == "cap?":
        return {"message": {"content": "still thinking"}, "finish_reason": "length"}
    return {"message": {"content": "</think>answer five"}, "finish_reason": "stop"}


class Server:
    def __init__(self, mode: str = "model"):
        self.mode, self.bodies, self.fail_first = mode, [], 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # fake Hugging Face resolve endpoint
                files = outer.files if hasattr(outer, "files") else {}
                for suffix, content in files.items():
                    if self.path.endswith(suffix):
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(content)
                        return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(503)
                    self.end_headers()
                    return
                if outer.mode == "judge":
                    choice = {"message": {"content": "verdict: " + json.dumps(VERDICT)}, "finish_reason": "stop"}
                    usage = {"prompt_tokens": 1000, "completion_tokens": 200}
                else:
                    choice = model_reply(body["messages"][0]["content"])
                    usage = {"prompt_tokens": 5, "completion_tokens": 7}
                out = json.dumps({"choices": [choice], "usage": usage}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


@pytest.fixture
def model_server():
    s = Server("model")
    yield s
    s.close()


@pytest.fixture
def judge_server():
    s = Server("judge")
    yield s
    s.close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    d = tmp_path / "nous"
    (d / "data").mkdir(parents=True)
    (d / "judge").mkdir()
    (d / "data" / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in ITEMS))
    (d / "judge" / "JUDGE_PROMPT.md").write_text(TEMPLATE)
    return d
