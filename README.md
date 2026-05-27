# Manifold-Constrained Hyper-Connections for Efficient Finetuning (mHC-PEFT)

This repo contains code for experiments from the paper *"Manifold-Constrained Hyper-Connections for Efficient Finetuning"*. Most parameter-efficient finetuning (PEFT) methods adapt weights or activations, thus leaving one of the key Transformer components unchanged: residual connections. This paper investigates Manifold-Constrained Hyper-Connections (mHC), a generalisation of residual connections, as a novel PEFT approach, wrapping a frozen OLMo-2-1B backbone with learned residual routing modules. With orders of magnitude fewer parameters than competing methods, the current mHC implementation achieves stable training but does not match other PEFT methods in training and downstream task performance. Nonetheless, analysis establishes residual routing as a distinct and underexplored consideration in finetuning, thus highlighting a promising direction forward. 

## Repo overview

| path | what it is | used for |
| --- | --- | --- |
| `main.py` | hydra entry point | runs `mode=train` or `mode=benchmark` |
| `configs/config.yaml` | main hydra config | selects model, method, data, train, benchmark settings |
| `run_finetuning.py` | train/eval orchestration | loads model, injects method, trains, optional perplexity eval |
| `train.py` | hugging face trainer setup | training arguments, saving final trainable params + metadata |
| `models/shc.py` | shc pytorch module | static hyper-connection wrapper with sinkhorn routing |
| `models/olmo_model2.py` | olmo-2 shc wrappers | replaces decoder layers and carries streams across depth |
| `models/injection.py` | method injection | applies lora/ia3/dora/vera/prompt/layer tuning or shc |
| `models/reload.py` | reload trained params | re-injects method and loads `trainable_params.pt` |
| `benchmark.py` | lm-eval harness integration | runs configured tasks and writes a json summary |
| `run_benchmarks.py` | benchmark orchestration | resolves paths and calls `benchmark.evaluate_benchmarks()` |
| `data/` | dataset + caching utilities | preprocessing, tokenization, packing, deterministic splits |

## Setup

the repo is set up around a conda environment defined in `environment.yml`.

1) create and activate the environment

```bash
conda env create -f environment.yml
conda activate FoMo
```

2) (optional) hugging face authentication

if you pull gated models/datasets, make sure you are logged in:

```bash
huggingface-cli login
```

3) (optional) set cache locations

on shared machines it can be useful to redirect hf caches:

```bash
set HF_HOME=path\\to\\hf_cache
```

notes:
- training is gpu-oriented (the default config uses `device_map=auto` and `torch_dtype=bfloat16`).
- outputs are managed by hydra. by default, hydra changes the working directory to a run folder under `outputs/`.

## Shc pytorch module (`models/shc.py`)

`models/shc.py` implements static hyper-connections as a lightweight wrapper around an existing sub-layer (the `branch`). it maintains multiple residual streams and updates them with three components:

- `pre_logits` (shape `[n]`): produces $h_{pre}$ via `sigmoid`, used to mix streams into the branch input (a convex combination after renormalization during init)
- `post_logits` (shape `[n]`): produces $h_{post}$ via `2 * sigmoid`, used to scale the branch output per stream before injecting back
- `res_logits` (shape `[n,n]`): turned into a doubly-stochastic routing matrix via `sinkhorn_logspace()`, used to route residuals across streams

high-level forward pass in `SHC.forward`:

1) accept either `x` with shape `[b, t, d]` or an existing stream tensor `[b, t, n, d]`
2) compute the branch input:
	- if `"pre"` is ablated: average over streams
	- else: weighted sum over streams using `sigmoid(pre_logits)`
3) run the wrapped `branch(branch_in, ...)`
4) compute the residual routing:
	- if `"res"` is ablated: identity routing
	- else: apply optional `dropout_res` masking on `res_logits`, then sinkhorn-normalize to get $H_{res}$
5) compute the post injection:
	- if `"post"` is ablated: broadcast branch output to all streams
	- else: scale per stream using `2 * sigmoid(post_logits)`
6) update streams: `X_new = X_res + X_post`
7) optionally read out to `[b, t, d]` by averaging over streams (`readout=True`)

ablations:
- `method.shc_ablation_mapping` can include any of `pre`, `res`, `post` to disable that component.

## Olmo-2 shc wrapper (`models/olmo_model2.py`)

`models/olmo_model2.py` adapts the shc wrapper to the internals of `transformers` olmo-2:

- `_OlmoAttentionBranch`: calls the base layer attention and applies `post_attention_layernorm`
- `_OlmoMLPBranch`: calls the base layer mlp and applies `post_feedforward_layernorm`
- `SHCOlmoDecoderLayer`: replaces a full decoder layer with two shc-wrapped branches (attention + mlp) so residuals are not applied twice
- `SHCOlmoModel`: carries stream tensors `[b, t, n, d]` through all layers and only reads out once at the end (mean readout by default, or learned softmax readout if enabled)
- `olmo_shc(olmo, ...)`: mutates `olmo.model` in-place by replacing it with `SHCOlmoModel`

this is what enables shc to behave like a peft method: only the new shc parameters are trainable (unless you explicitly enable training the wrapped branch).

## Training

training is driven by hydra. the default config is in `configs/config.yaml` and defaults to `mode: train`.

quick start (uses the current config):

```bash
python main.py
```

common overrides:

```bash
# name your run (shows up in output dir and tensorboard run name)
python main.py run_name=my_run

# choose method
python main.py method.selected_method=shc
python main.py method.selected_method=lora

# change base model
python main.py model.pretrained_model_name_or_path=allenai/OLMo-2-0425-1B

# shorter smoke test
python main.py train.max_train_steps=10 data.max_train_samples=128 data.validation_samples=128
```

where outputs go:
- hydra creates a run directory like `outputs/yyyy-mm-dd/hh-mm-ss_<run_name>_<method>/` and changes cwd into it
- trainer checkpoints are written under `train.checkpoint_output_dir` (default: `checkpoints/` relative to the run directory)
- final trainable parameters + metadata are written to `train.final_trainable_params_dir` (default: `final_model/`)

useful artifacts (under `final_model/`):
- `trainable_params.pt`: only the parameters that required gradients
- `reload_metadata.json`: minimal metadata needed to reconstruct the injected model
- `training_summary.json`: small json summary with train/eval metrics and parameter counts
- `resolved_config.yaml`: fully resolved hydra config saved by `run_finetuning.save_run_config()`

tensorboard:

```bash
tensorboard --logdir runs
```

## Benchmarks (lm-evaluation-harness)

benchmarking is also driven by hydra via `mode=benchmark`.

run the configured tasks on a saved run (example uses `final_model/` in the current directory):

```bash
python main.py mode=benchmark benchmark.checkpoint_path=final_model
```

run the configured tasks on a hugging face model id:

```bash
python main.py mode=benchmark benchmark.checkpoint_path=allenai/OLMo-2-0425-1B
```

common benchmark overrides:

```bash
# smoke test with fewer examples
python main.py mode=benchmark benchmark.limit=10

# change output file name
python main.py mode=benchmark benchmark.harness_output_file=harness.json

# change PEFT model to be benchmarked, for example (ia)^3
python main.py \
	method.selected_method=ia3 \
	mode=benchmark \
	benchmark.checkpoint_path=outputs/ia3-tulu-20k-olmo1b/23037766/final_model \
	benchmark.batch_size=auto \

# change benchmark task
python main.py \
    ~benchmark.tasks \
    ++benchmark.tasks.hendrycks_math.fewshot=4 \
    ++benchmark.tasks.hendrycks_math.metric=exact_match \
```

what happens:
- `run_benchmarks.py` resolves output paths back to the original working directory (so `harness.json` lands where you launched the command)
- `benchmark.py` uses `lm_eval.simple_evaluate()` and writes a summary json (per task: fewshot, metric, rounded value)
- `benchmark.checkpoint_path` can be either a hf model id or a local directory containing `reload_metadata.json` + `trainable_params.pt`
- `benchmark.tasks` can be changed in order to evaluate on different benchmarks with different few-shot settings

# Contact
- Maintainer: F.P.J. de Kam
- E-mail: floris.de.kam@student.uva.nl