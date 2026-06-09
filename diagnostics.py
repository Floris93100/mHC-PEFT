# diagnostics.py
"""
some quick tests for checking mHC method/models behaviour

* MODEL-level tests: check whether OLMo
    - outputs the correct shapes
    - has gradients flowing to the trainable parameters
    - is identity-equivalent to the unwrapped model at init
    - breaks stream symmetry once training starts (streams diverge)

* MODULE-level tests: check whether mHC
    - exposes diagnostics() getter function
    - has a doubly stochastic residual mixing matrix
    - is initialised identity-equivalent

example running command:

python diagnostics.py \
  --config configs/config.yaml \
  --batch_size 1 \
  --seq_len 32 \
  --output diagnostics.json \
  method.selected_method=shc
"""
# code below is LLM-generated, except for the identity-equivalence test

import argparse
import json
import os

import torch
from omegaconf import OmegaConf

from models.injection import inject_method, count_trainable_parameters
from models.loading import load_model_and_tokenizer


def move_batch_to_model(batch, model):
    """moves tokenized batch to the first model device"""
    device = next(model.parameters()).device
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def make_tiny_batch(tokenizer, model, batch_size, seq_len):
    """creates a tiny causal lm batch without loading a dataset"""
    text = "This is a short diagnostic example for testing the model wrapper."
    texts = [text for _ in range(batch_size)]

    batch = tokenizer(
        texts,
        return_tensors = "pt",
        padding = "max_length",
        truncation = True,
        max_length = seq_len,
    )

    return move_batch_to_model(batch, model)


def to_float_tensor(x):
    """converts diagnostic outputs to detached float tensors"""
    if x is None:
        return None

    if not torch.is_tensor(x):
        x = torch.as_tensor(x)

    return x.detach().float()


def to_jsonable(x):
    """converts small values to json-safe objects"""
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x


def get_routing_modules(model):
    """returns all modules that expose the generic diagnostics interface"""
    modules = []

    for name, module in model.named_modules():
        diagnostics_fn = getattr(module, "diagnostics", None)

        if callable(diagnostics_fn):
            modules.append((name, module))

    if not modules:
        raise RuntimeError(
            "no routing modules found. expected modules with a diagnostics() method"
        )

    return modules


def set_module_diagnostics(model, enabled = True):
    """ enable or disable routing-state caching for modules that support it """
    for module in model.modules():
        enable_fn = getattr(module, "enable_diagnostics", None)
        if callable(enable_fn):
            enable_fn(enabled)


def check_shapes(model, tokenizer, batch_size = 1, seq_len = 32):
    """model-level test: checks that the full model preserves causal lm shapes"""
    model.eval()
    batch = make_tiny_batch(tokenizer, model, batch_size, seq_len)

    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits

    expected_batch = batch["input_ids"].shape[0]
    expected_seq_len = batch["input_ids"].shape[1]
    vocab_size = model.config.vocab_size

    assert logits.shape[0] == expected_batch, (
        f"batch mismatch: got {logits.shape[0]}, expected {expected_batch}"
    )
    assert logits.shape[1] == expected_seq_len, (
        f"seq length mismatch: got {logits.shape[1]}, expected {expected_seq_len}"
    )
    assert logits.shape[2] == vocab_size, (
        f"vocab mismatch: got {logits.shape[2]}, expected {vocab_size}"
    )
    assert torch.isfinite(logits).all(), "logits contain nan or inf"

    return {
        "passed": True,
        "logits_shape": list(logits.shape),
    }


def check_vector_state(name, value, num_streams = None, atol = 1e-3):
    """module-level helper: checks h_pre or h_post diagnostics"""
    value = to_float_tensor(value)

    assert value is not None, f"{name} is missing"
    assert torch.isfinite(value).all(), f"{name} contains nan or inf"

    if num_streams is not None:
        assert value.shape[-1] == num_streams, (
            f"{name} last dim should be num_streams={num_streams}, "
            f"got shape {list(value.shape)}"
        )

    return {
        "shape": list(value.shape),
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
    }


def check_double_stochastic_matrix(h_res, num_streams = None, atol = 1e-3):
    """module-level helper: checks one h_res tensor over its last two dims"""
    h_res = to_float_tensor(h_res)

    assert h_res is not None, "h_res is missing"
    assert h_res.dim() >= 2, f"h_res should have at least 2 dims, got {h_res.dim()}"
    assert h_res.shape[-1] == h_res.shape[-2], (
        f"h_res must be square on last two dims, got shape {list(h_res.shape)}"
    )
    assert torch.isfinite(h_res).all(), "h_res contains nan or inf"

    if num_streams is not None:
        assert h_res.shape[-1] == num_streams, (
            f"h_res last dims should be num_streams={num_streams}, "
            f"got shape {list(h_res.shape)}"
        )

    row_error = (h_res.sum(dim = -1) - 1.0).abs().max().item()
    col_error = (h_res.sum(dim = -2) - 1.0).abs().max().item()
    min_entry = h_res.min().item()
    max_entry = h_res.max().item()

    assert row_error <= atol, (
        f"row sums not close to 1: max error {row_error}"
    )
    assert col_error <= atol, (
        f"column sums not close to 1: max error {col_error}"
    )
    assert min_entry >= -atol, (
        f"negative routing entry found: min entry {min_entry}"
    )

    return {
        "shape": list(h_res.shape),
        "max_row_error": row_error,
        "max_col_error": col_error,
        "min_entry": min_entry,
        "max_entry": max_entry,
    }


def check_routing_module(name, module, atol = 1e-3):
    """module-level test: checks one routing module through diagnostics()"""
    state = module.diagnostics()

    required_keys = ["h_res", "h_pre", "h_post", "num_streams"]
    missing_keys = [
        key
        for key in required_keys
        if key not in state
    ]

    assert not missing_keys, (
        f"module {name} diagnostics() is missing keys: {missing_keys}"
    )

    num_streams = int(state["num_streams"])

    h_pre_stats = check_vector_state(
        name = f"{name}.h_pre",
        value = state["h_pre"],
        num_streams = num_streams,
        atol = atol,
    )

    h_post_stats = check_vector_state(
        name = f"{name}.h_post",
        value = state["h_post"],
        num_streams = num_streams,
        atol = atol,
    )

    h_res_stats = check_double_stochastic_matrix(
        h_res = state["h_res"],
        num_streams = num_streams,
        atol = atol,
    )

    return {
        "passed": True,
        "module_name": name,
        "module_type": module.__class__.__name__,
        "num_streams": num_streams,
        "h_pre": h_pre_stats,
        "h_post": h_post_stats,
        "h_res": h_res_stats,
    }


def check_routing_modules(model, atol = 1e-3):
    """module-level test: checks all routing modules in the model"""
    model.eval()
    modules = get_routing_modules(model)

    module_results = []
    max_row_error = 0.0
    max_col_error = 0.0
    min_entry = float("inf")

    for name, module in modules:
        result = check_routing_module(
            name = name,
            module = module,
            atol = atol,
        )

        module_results.append(result)
        max_row_error = max(
            max_row_error,
            result["h_res"]["max_row_error"],
        )
        max_col_error = max(
            max_col_error,
            result["h_res"]["max_col_error"],
        )
        min_entry = min(
            min_entry,
            result["h_res"]["min_entry"],
        )

    return {
        "passed": True,
        "checked_modules": len(module_results),
        "max_row_error": max_row_error,
        "max_col_error": max_col_error,
        "min_entry": min_entry,
        "modules": module_results,
    }


def check_gradients(model, tokenizer, batch_size = 1, seq_len = 32):
    """model-level test: checks gradients on trainable params and frozen params"""
    model.train()
    model.zero_grad(set_to_none = True)

    batch = make_tiny_batch(tokenizer, model, batch_size, seq_len)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    outputs = model(
        input_ids = batch["input_ids"],
        attention_mask = batch["attention_mask"],
        labels = labels,
    )

    loss = outputs.loss
    assert loss is not None, "model did not return a loss"
    assert torch.isfinite(loss), f"loss is not finite: {loss.item()}"

    loss.backward()

    trainable_without_grad = []
    trainable_with_grad = []
    frozen_with_grad = []
    nonfinite_grad = []

    routing_like_grad_names = []

    routing_keywords = [
        "hc",
        "mhc",
        "krom",
        "router",
        "routing",
        "logit",
        "pre",
        "post",
        "res",
    ]

    for name, parameter in model.named_parameters():
        grad = parameter.grad

        if parameter.requires_grad:
            if grad is None:
                trainable_without_grad.append(name)
            else:
                trainable_with_grad.append(name)

                if not torch.isfinite(grad).all():
                    nonfinite_grad.append(name)

                name_lower = name.lower()
                if any(keyword in name_lower for keyword in routing_keywords):
                    routing_like_grad_names.append(name)
        else:
            if grad is not None:
                frozen_with_grad.append(name)

    assert trainable_with_grad, "no trainable parameters received gradients"
    assert not frozen_with_grad, (
        "some frozen parameters received gradients: "
        f"{frozen_with_grad[:10]}"
    )
    assert not nonfinite_grad, (
        "some gradients contain nan or inf: "
        f"{nonfinite_grad[:10]}"
    )

    return {
        "passed": True,
        "loss": float(loss.detach().cpu()),
        "trainable_with_grad": len(trainable_with_grad),
        "trainable_without_grad": len(trainable_without_grad),
        "frozen_with_grad": len(frozen_with_grad),
        "nonfinite_grad": len(nonfinite_grad),
        "trainable_parameter_examples": trainable_with_grad[:20],
        "trainable_without_grad_examples": trainable_without_grad[:20],
        "routing_like_parameter_examples": routing_like_grad_names[:20],
    }


def check_identity_equivalence_model(
    cfg,
    tokenizer,
    batch_size = 1,
    seq_len = 32,
    atol = 1e-4,
    rtol = 1e-4,
):
    """ check whether injected model preserves full model input-output behaviour """
    cfg_dict = {
        k: v for k, v in OmegaConf.to_container(cfg, resolve=False).items()
        if k != "hydra"
    }
    cfg_base = OmegaConf.create(cfg_dict)
    cfg_wrapped = OmegaConf.create(cfg_dict)

    cfg_base.method.selected_method = "frozen"
    cfg_base.model.use_cache = False
    cfg_wrapped.model.use_cache = False

    base_model, base_tokenizer = load_model_and_tokenizer(cfg_base.model)
    wrapped_model, wrapped_tokenizer = load_model_and_tokenizer(cfg_wrapped.model)
    wrapped_model = inject_method(wrapped_model, cfg_wrapped)

    base_model.eval()
    wrapped_model.eval()

    batch = make_tiny_batch(
        tokenizer = base_tokenizer,
        model = base_model,
        batch_size = batch_size,
        seq_len = seq_len,
    )

    wrapped_batch = {
        key: value.to(next(wrapped_model.parameters()).device)
        for key, value in batch.items()
    }

    with torch.no_grad():
        base_outputs = base_model(**batch)
        wrapped_outputs = wrapped_model(**wrapped_batch)

    base_logits = base_outputs.logits.detach().float().cpu()
    wrapped_logits = wrapped_outputs.logits.detach().float().cpu()

    diff = base_logits - wrapped_logits
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()
    relative_l2_diff = (
        diff.norm() / base_logits.norm().clamp_min(1e-12)
    ).item()

    allclose = torch.allclose(
        base_logits,
        wrapped_logits,
        atol = atol,
        rtol = rtol,
    )

    assert allclose, (
        f"complete-model identity equivalence failed: "
        f"max diff {max_abs_diff}, mean diff {mean_abs_diff}, "
        f"relative l2 diff {relative_l2_diff}"
    )

    return {
        "passed": True,
        "atol": atol,
        "rtol": rtol,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "relative_l2_diff": relative_l2_diff,
        "logits_shape": list(base_logits.shape),
    }


def _infer_num_streams(model):
    """best-effort lookup of the residual-stream count on a wrapped model"""
    for module in model.modules():
        for attr in ("num_residual_streams", "num_streams"):
            value = getattr(module, attr, None)
            if isinstance(value, int) and value > 1:
                return value
    return 1


def _get_decoder_layers(model):
    """locates the decoder-layer ModuleList of a (wrapped) causal lm"""
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError(
            "could not locate decoder layers; expected model.model.layers"
        )
    return list(layers)


def _streams_from_layer_output(hidden_states, num_streams):
    """reshapes a captured decoder-layer output to (batch, streams, seq, dim).

    handles both stream layouts used by the wrappers:
      * (batch, seq, streams, dim)   -> SHC-style explicit stream axis
      * (batch * streams, seq, dim)  -> mHC-lite / KromHC stream-in-batch fold
    the batch fold uses einops '(b s)', i.e. streams are the inner/faster index,
    so a plain reshape on the contiguous tensor recovers (b, s, t, d).
    """
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[0]

    hidden_states = hidden_states.detach()

    if hidden_states.dim() == 4:
        # (b, t, s, d) -> (b, s, t, d)
        return hidden_states.permute(0, 2, 1, 3).contiguous()

    if hidden_states.dim() == 3:
        bs, seq, dim = hidden_states.shape
        if num_streams <= 1 or bs % num_streams != 0:
            return None
        batch = bs // num_streams
        return hidden_states.reshape(batch, num_streams, seq, dim)

    return None


def _pairwise_stream_divergence(streams_bstd):
    """mean_{s != s'} ||X[s] - X[s']|| / mean_s ||X[s]||, averaged over batch, seq.

    equals 0 when the streams are identical (the identity-init / symmetric state),
    grows as the streams differentiate.
    """
    if streams_bstd is None:
        return None

    x = streams_bstd.float()
    _, num_streams, _, _ = x.shape
    if num_streams < 2:
        return 0.0

    stream_norms = x.norm(dim = -1)                 # (b, s, t)
    mean_norm = stream_norms.mean().clamp_min(1e-12)

    total = x.new_zeros(())
    pairs = 0
    for i in range(num_streams):
        for j in range(i + 1, num_streams):
            total = total + (x[:, i] - x[:, j]).norm(dim = -1).mean()
            pairs += 1

    return (total / pairs / mean_norm).item()


def _routing_spread(model):
    """spread of per-stream routing coeffs via the diagnostics() interface.

    returns mean over modules of:
      * beta_std_across_streams : std of H_post over the stream axis (0 at init)
      * hres_offdiagonal_mass   : mean |H_res - I|              (0 at init)
    values are None if diagnostics are unavailable for this method/config.
    """
    try:
        modules = get_routing_modules(model)
    except RuntimeError:
        return {"beta_std_across_streams": None, "hres_offdiagonal_mass": None}

    beta_vals = []
    hres_vals = []

    for _, module in modules:
        state = module.diagnostics()

        h_post = to_float_tensor(state.get("h_post"))
        if h_post is not None and h_post.shape[-1] > 1:
            beta_vals.append(h_post.std(dim = -1).mean().item())

        h_res = to_float_tensor(state.get("h_res"))
        if h_res is not None and h_res.dim() >= 2:
            n = h_res.shape[-1]
            eye = torch.eye(n, device = h_res.device, dtype = h_res.dtype)
            hres_vals.append((h_res - eye).abs().mean().item())

    return {
        "beta_std_across_streams": (
            sum(beta_vals) / len(beta_vals) if beta_vals else None
        ),
        "hres_offdiagonal_mass": (
            sum(hres_vals) / len(hres_vals) if hres_vals else None
        ),
    }


def _measure_stream_state(model, batch, num_streams):
    """one eval forward: per-layer stream divergence + routing-coefficient spread"""
    layers = _get_decoder_layers(model)
    captured = {}

    def make_hook(index):
        def hook(_module, _inputs, output):
            captured[index] = output
        return hook

    handles = [
        layer.register_forward_hook(make_hook(i))
        for i, layer in enumerate(layers)
    ]

    was_training = model.training
    model.eval()

    spread = {"beta_std_across_streams": None, "hres_offdiagonal_mass": None}
    try:
        try:                            # diagnostics on -> also get routing spread
            set_module_diagnostics(model, True)
            with torch.no_grad():
                model(**batch)
            spread = _routing_spread(model)
        except Exception:               # method/config without diagnostics: hooks only
            captured.clear()
            set_module_diagnostics(model, False)
            with torch.no_grad():
                model(**batch)
    finally:
        set_module_diagnostics(model, False)
        for handle in handles:
            handle.remove()
        if was_training:
            model.train()

    per_layer = []
    for index in sorted(captured):
        divergence = _pairwise_stream_divergence(
            _streams_from_layer_output(captured[index], num_streams)
        )
        if divergence is not None:
            per_layer.append(divergence)

    result = {
        "mean_stream_divergence": (
            sum(per_layer) / len(per_layer) if per_layer else None
        ),
        "max_stream_divergence": max(per_layer) if per_layer else None,
        "per_layer_stream_divergence": per_layer,
    }
    result.update(spread)
    return result


def check_stream_divergence(
    model,
    tokenizer,
    batch_size = 1,
    seq_len = 32,
    num_steps = 50,
    lr = 1e-3,
    checkpoints = (0, 1, 5, 10, 25, 50),
    seed = 0,
):
    """model-level dynamics test: do the n residual streams diverge during training?

    At an identity-equivalent init every stream holds the same vector, so the
    pairwise stream divergence is ~0. The symmetry break (so n>1 streams can
    specialise) is seeded by the per-layer one-hot H_pre, which gates the backward
    gradient to a different stream in each layer. This runs a short overfitting
    loop on a single tiny batch and records the divergence trajectory, plus the
    spread of the per-stream H_post (beta) and the off-identity mass of H_res.

    It reports symmetry_broke = (divergence grew) but does NOT assert it: symmetry
    breaking on a synthetic batch in a few steps is empirical, not a correctness
    guarantee. This MUTATES the passed model (it trains it), so run it last.
    """
    torch.manual_seed(seed)

    num_streams = _infer_num_streams(model)
    if num_streams < 2:
        return {
            "passed": True,
            "skipped": "num_streams < 2 (single stream, nothing to diverge)",
        }

    batch = make_tiny_batch(tokenizer, model, batch_size, seq_len)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"] == 0] = -100

    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable, "no trainable parameters; cannot probe training dynamics"
    optimizer = torch.optim.AdamW(trainable, lr = lr)

    checkpoints = sorted({c for c in checkpoints if 0 <= c <= num_steps})
    trajectory = {}

    if 0 in checkpoints:
        trajectory[0] = _measure_stream_state(model, batch, num_streams)

    for step in range(1, num_steps + 1):
        model.train()
        model.zero_grad(set_to_none = True)
        outputs = model(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels = labels,
        )
        loss = outputs.loss
        assert torch.isfinite(loss), f"loss became non-finite at step {step}"
        loss.backward()
        optimizer.step()

        if step in checkpoints:
            trajectory[step] = _measure_stream_state(model, batch, num_streams)

    ordered = sorted(trajectory)
    first = trajectory[ordered[0]].get("mean_stream_divergence") or 0.0
    last = trajectory[ordered[-1]].get("mean_stream_divergence") or 0.0

    return {
        "passed": True,
        "num_streams": num_streams,
        "num_steps": num_steps,
        "lr": lr,
        "symmetry_broke": bool(last > first + 1e-4),
        "initial_mean_divergence": first,
        "final_mean_divergence": last,
        "divergence_growth": last - first,
        "trajectory": {str(step): trajectory[step] for step in ordered},
    }


def check_mhc(
    cfg,
    batch_size = 1,
    seq_len = 32,
    atol = 1e-3,
    run_identity = True,
    identity_atol = 1e-4,
    identity_rtol = 1e-4,
    skip_module_tests = False,
    run_divergence = True,
    divergence_steps = 50,
    divergence_lr = 1e-3,
):
    """runs all quick diagnostics"""
    cfg.model.use_cache = False

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model = inject_method(model, cfg)

    counts = count_trainable_parameters(model)

    results = {
        "method": cfg.method.selected_method,
        "model": cfg.model.pretrained_model_name_or_path,
        "parameter_counts": counts,
    }

    set_module_diagnostics(model, True) # turn diagnostics on

    try:                                # run shape test first because
                                        # it performs a tiny forward pass
        results["shapes"] = check_shapes( 
            model = model,                
            tokenizer = tokenizer,
            batch_size = batch_size,
            seq_len = seq_len,
        )
        if not skip_module_tests:
            results["routing_modules"] = check_routing_modules(
                model = model,
                atol = atol,
            )
        results["gradients"] = check_gradients(
            model = model,
            tokenizer = tokenizer,
            batch_size = batch_size,
            seq_len = seq_len,
        )
    finally:                            # turn diagnostics off
        set_module_diagnostics(model, False)

    if run_identity:
        results["identity_equivalence"] = check_identity_equivalence_model(
            cfg = cfg,
            tokenizer = tokenizer,
            batch_size = batch_size,
            seq_len = seq_len,
            atol = identity_atol,
            rtol = identity_rtol,
        )

    if run_divergence:
        results["stream_divergence"] = check_stream_divergence(
            model = model,
            tokenizer = tokenizer,
            batch_size = batch_size,
            seq_len = seq_len,
            num_steps = divergence_steps,
            lr = divergence_lr,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/config.yaml")
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--seq_len", type = int, default = 32)
    parser.add_argument("--atol", type = float, default = 1e-3)
    parser.add_argument("--output", default = None)
                                        # for debugging, identity test can be switched off
    parser.add_argument("--skip_identity", action = "store_true")
    parser.add_argument("--skip_module_tests", action = "store_true")
                                        # stream-divergence dynamics probe (trains
                                        # the model on one tiny batch); off via flag
    parser.add_argument("--skip_divergence", action = "store_true")
    parser.add_argument("--divergence_steps", type = int, default = 50)
    parser.add_argument("--divergence_lr", type = float, default = 1e-3)
    parser.add_argument("--identity_atol", type = float, default = 1e-2)
    parser.add_argument("--identity_rtol", type = float, default = 1e-2)

    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    results = check_mhc(
        cfg = cfg,
        batch_size = args.batch_size,
        seq_len = args.seq_len,
        atol = args.atol,
        run_identity = not args.skip_identity,
        identity_atol = args.identity_atol,
        identity_rtol = args.identity_rtol,
        skip_module_tests = args.skip_module_tests,
        run_divergence = not args.skip_divergence,
        divergence_steps = args.divergence_steps,
        divergence_lr = args.divergence_lr,
    )

    print(json.dumps(results, indent = 2))

    if args.output is not None:
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok = True)

        with open(args.output, "w") as file:
            json.dump(results, file, indent = 2)

        print(f"saved diagnostics to: {args.output}")


if __name__ == "__main__":
    main()