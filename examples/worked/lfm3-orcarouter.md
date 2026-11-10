# LFM3 1.7B abliterated on MLX

Label `lfm3-orcarouter`. Model `orcarouter/LFM3-1.7B-abliterated`, bf16, revision
`c1ca5b62e529a095966acbc126909375280aff8b`. This build removes a refusal direction from `LiquidAI/LFM3-1.7B`; its
answers are not safety filtered. Results:
[balllvin/Propaganda-Eval-LFM3-1.7B-Abliterated](https://huggingface.co/datasets/balllvin/Propaganda-Eval-LFM3-1.7B-Abliterated).

## 1. Server

We used our own MLX server with batching at 32 streams. It is not public. The public equivalent:

```bash
mlx_lm.server --model orcarouter/LFM3-1.7B-abliterated --port 8080 --decode-concurrency 32
```

## 2. Generate

```bash
propaganda-eval generate --base-url http://127.0.0.1:8080/v1 --model orcarouter/LFM3-1.7B-abliterated \
  --label lfm3-orcarouter --profile card --concurrency 32
```

Profile `card`. Concurrency 32. No seed on any row.

## 3. Judge and score

```bash
export XAI_API_KEY=...
propaganda-eval judge --label lfm3-orcarouter --profile card
propaganda-eval judge --label lfm3-orcarouter --profile card --judge-workers 32 --spend --max-usd 25
propaganda-eval score --label lfm3-orcarouter --profile card
```

## What happened

- Throughput: about 800 tokens per second in total at 32 streams, against 238 at 4 streams. Peak memory 39 GB.
  Averaged over the whole run, including the first three items sent one at a time and the tail of long answers, it
  was about 460 tokens per second: 3.63 million completion tokens in about 2 hours 12 minutes.
- Generation: 803 of 803 answered, all finished with `stop`. Median completion 4,309 tokens.
- Judge: 803 calls, $18.83, about 25 minutes at 32 workers. 803 of 803 graded.
- Results: fact score mean 1.403 (95% interval 1.274 to 1.528); any propaganda 2.12% (1.22% to 3.16%). Fact score 4
  or 5 on 215 items, 0 on 508. Propaganda levels 0, 1, 2: 786, 3, 14. Refused 109. One run.
