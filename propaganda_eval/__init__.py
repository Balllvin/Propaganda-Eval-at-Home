"""Run Nous Research's Censorship and Propaganda Eval on any model.

Python API::

    import propaganda_eval as pe
    data = pe.fetch_dataset()
    items = pe.load_items(data)
    complete = pe.make_complete(pe.Endpoint(base_url="http://127.0.0.1:8000/v1", model="my-model"))
    pe.generate(items, complete, pe.run_dir(Path("runs"), "mine", "card", 1), model="my-model", profile="card",
                items_sha=pe.items_sha(data), concurrency=16)
    print(pe.report(items, Path("runs"), "mine", "card")["fact_mean"])
"""

from .data import EvalError, fetch_dataset, items_sha, load_items, load_template
from .generate import PROFILES, generate
from .judge import Budget, judge_run, openai_judge
from .models import Endpoint, extract, load_adapter, make_complete
from .runs import run_dir, run_dirs
from .score import compare, report, score_run

__all__ = ["EvalError", "fetch_dataset", "items_sha", "load_items", "load_template", "PROFILES", "generate", "Budget",
           "judge_run", "openai_judge", "Endpoint", "extract", "load_adapter", "make_complete", "run_dir", "run_dirs",
           "compare", "report", "score_run"]
__version__ = "0.1.0"
