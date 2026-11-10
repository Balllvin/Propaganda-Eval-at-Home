# Dory 4B NVFP4 on MLX

Label `dory-nvfp4`. Model `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4` (NVFP4, W4A16), served from a local MLX
conversion. Results: [balllvin/Propaganda-Eval-Dory-4B-NVFP4](https://huggingface.co/datasets/balllvin/Propaganda-Eval-Dory-4B-NVFP4).

## 1. Server

We used our own MLX server for NVFP4 weights, with a batch of 4 streams. It is not public, and we have not checked
whether `mlx_lm.server` loads this checkpoint. With a model it does load, the equivalent is:

```bash
mlx_lm.server --model path/to/your-mlx-model --port 8080 --decode-concurrency 4
```

## 2. Generate

```bash
propaganda-eval generate --base-url http://127.0.0.1:8080/v1 --model path/to/your-mlx-model \
  --label dory-nvfp4 --profile card --concurrency 4
```

Profile `card`. Concurrency 4. No seed (the default here).

## 3. Judge and score

```bash
export XAI_API_KEY=...
propaganda-eval judge --label dory-nvfp4 --profile card                                        # cost plan, spends nothing
propaganda-eval judge --label dory-nvfp4 --profile card --judge-workers 32 --spend --max-usd 25
propaganda-eval score --label dory-nvfp4 --profile card
```

## What happened

- Seeds: the first 14 rows were sent with a per-request seed (1001). A seed made the server decode one request at a
  time, so we dropped it and the other 789 rows were sent with no seed, which let the server batch. The `seeded`
  column in the dataset marks the 14.
- Generation: 803 of 803 answered, all finished with `stop`, none hit the token cap. Median completion 6,433
  tokens; 5.27 million completion tokens in all. The run was stopped and resumed, so there is no single wall time.
- Judge: 803 calls, $19.86, about 30 minutes at 32 workers. 803 of 803 graded.
- Results: fact score mean 1.821 (95% interval 1.688 to 1.949); any propaganda 1.25% (0.52% to 2.02%). Fact score 4
  or 5 on 274 items, 0 on 411. Propaganda levels 0, 1, 2: 793, 1, 9. Refused 91. Intervals resample 405 source
  documents whole. One run, so there is no run to run spread.
