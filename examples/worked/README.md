# Worked examples

How we ran the three reference models, step by step, with what happened. All three used the `card` profile
(thinking on, temperature 1.0, top_p 0.95, 32,768 token cap), one run each, on one Mac with Apple silicon, and the
default judge, grok-4.7 at temperature 0 through the xAI API, with 32 judge workers.

| Example | Model | Fact score (95%) | Any propaganda (95%) | Judge cost | Dataset |
|---|---|---|---|---|---|
| [dory-nvfp4](dory-nvfp4.md) | nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 | 1.821 (1.688 to 1.949) | 1.25% (0.52% to 2.02%) | $19.86 | [link](https://huggingface.co/datasets/balllvin/Propaganda-Eval-Dory-4B-NVFP4) |
| [lfm3-orcarouter](lfm3-orcarouter.md) | orcarouter/LFM3-1.7B-abliterated | 1.403 (1.274 to 1.528) | 2.12% (1.22% to 3.16%) | $18.83 | [link](https://huggingface.co/datasets/balllvin/Propaganda-Eval-LFM3-1.7B-Abliterated) |
| [lfm3-base](lfm3-base.md) | LiquidAI/LFM3-1.7B | 1.473 (1.351 to 1.597) | 2.37% (1.36% to 3.46%) | $17.90 | [link](https://huggingface.co/datasets/balllvin/Propaganda-Eval-LFM3-1.7B-Base) |

We served each model with our own MLX server, which batches concurrent requests. That server is not public. The
public equivalent is `mlx_lm.server` with its batching option; the setting that matters is the same: several
concurrent streams and no per-request seed. Your numbers for speed will differ with your machine and server.

Our runs used an earlier internal version of this code. The commands below are the same steps in this repo.
`fetch-published` downloads all three as `ref-*` labels so you can compare your model with them.
