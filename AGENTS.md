# Agent instructions

This repo runs Nous Research's Censorship and Propaganda Eval on any model. Read README.md first; it is written for you.

## Skills

Skills live in `.agents/skills/<name>/SKILL.md` (`.claude/skills` links to the same folder). Load the matching one first.

- `run-eval`: run the eval on a model, from smoke test to judged score.
- `add-model-adapter`: connect a model or provider that the config options do not cover.
- `read-results`: read scores, compare models and open the report.

## Rules

- Never spend on the judge without the user's explicit OK and a dollar cap: always run `judge` without `--spend` first and show the plan.
- Keys come only from environment variables. Never write a key into a config file, a command you show, or a commit.
- Scores are comparable only with the same profile and the grok-4.7 judge. Say so when either differs.
- Do not change the pinned dataset revision or sha256 in `propaganda_eval/data.py`.
- Run `pytest` before you commit. Tests must stay offline: use the fake servers in `tests/conftest.py`.
- Keep dependencies to the standard library (plus `tomli` on Python 3.10).
