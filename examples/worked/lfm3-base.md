# LFM3 1.7B base on MLX

Label `lfm3-base`. Model `LiquidAI/LFM3-1.7B`, bf16, revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. Results:
[balllvin/Propaganda-Eval-LFM3-1.7B-Base](https://huggingface.co/datasets/balllvin/Propaganda-Eval-LFM3-1.7B-Base).

## 1. Server

Our own MLX server with batching, not public. The public equivalent:

```bash
mlx_lm.server --model LiquidAI/LFM3-1.7B --port 8080 --decode-concurrency 32
```

## 2. Generate

```bash
propaganda-eval generate --base-url http://127.0.0.1:8080/v1 --model LiquidAI/LFM3-1.7B \
  --label lfm3-base --profile card --concurrency 32
```

Profile `card`. Concurrency 32 at the start. No seed on any row.

## 3. Judge and score

```bash
export XAI_API_KEY=...
propaganda-eval judge --label lfm3-base --profile card
propaganda-eval judge --label lfm3-base --profile card --judge-workers 32 --spend --max-usd 25
propaganda-eval score --label lfm3-base --profile card
```

## What happened

- Fallback: we started at 32 streams. Memory passed 40 GB, so we stopped the server, restarted it at 16 streams and
  resumed the same label. Resume skips finished items, so nothing was lost or redone.
  To do the same: stop, restart the server with `--decode-concurrency 16`, and rerun the generate command with
  `--concurrency 16`.
- Generation: 803 of 803 answered, all finished with `stop`. Median completion 3,561 tokens; 3.00 million completion
  tokens in about 1 hour 26 minutes, about 580 tokens per second averaged over the run.
- Judge: 803 calls, $17.90, about 21 minutes at 32 workers. 803 of 803 graded.
- Results: fact score mean 1.473 (95% interval 1.351 to 1.597); any propaganda 2.37% (1.36% to 3.46%). Fact score 4
  or 5 on 225 items, 0 on 492. Propaganda levels 0, 1, 2: 784, 9, 10. Refused 184. One run.
