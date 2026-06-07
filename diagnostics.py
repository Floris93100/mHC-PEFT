# diagnostics.py
"""
some quick tests for checking whether the mHC method
* outputs the correct shapes
* is initialised identity-equivalent
* has gradients flowing to the trainable parameters
* has a doubly stochastic residual mixing matrix

example running command:

python diagnostics.py \
  --config configs/config.yaml \
  method.selected_method=shc \
  model.pretrained_model_name_or_path=allenai/OLMo-2-0425-1B
"""
# code below is LLM-generated, except for the identity-equivalence test

import argparse
import json
import math
import os
import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.olmo_model2 import olmo_shc
from models.injection import inject_method, count_trainable_parameters
from models.loading import load_model_and_tokenizer
from models.shc import SHC, sinkhorn_logspace


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


def get_shc_modules(model):
    """returns all shc modules in the model"""
    modules = [
        module
        for module in model.modules()
        if isinstance(module, SHC)
    ]

    if not modules:
        raise RuntimeError("no SHC modules found in model")

    return modules


def check_shapes(model, tokenizer, batch_size = 1, seq_len = 32):
    """checks that the injected model preserves causal lm output shapes"""
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


def check_double_stochastic(model, atol = 1e-3):
    """checks row sums, column sums and non-negativity of shc residual routers"""
    model.eval()
    modules = get_shc_modules(model)

    max_row_error = 0.0
    max_col_error = 0.0
    min_entry = float("inf")
    checked = 0

    for module in modules:
        if "res" in module.ablate_mapping:
            continue

        logits = module.res_logits.detach()
        h_res = sinkhorn_logspace(
            logits.reshape(1, 1, module.num_streams, module.num_streams),
            num_iters = module.sinkhorn_iters,
            eps = module.eps,
        ).squeeze(0).squeeze(0).float()

        row_error = (h_res.sum(dim = -1) - 1.0).abs().max().item()
        col_error = (h_res.sum(dim = -2) - 1.0).abs().max().item()

        max_row_error = max(max_row_error, row_error)
        max_col_error = max(max_col_error, col_error)
        min_entry = min(min_entry, h_res.min().item())
        checked += 1

    if checked == 0:
        raise RuntimeError("no non-ablated SHC residual routers found")

    assert max_row_error <= atol, (
        f"row sums not close to 1: max error {max_row_error}"
    )
    assert max_col_error <= atol, (
        f"column sums not close to 1: max error {max_col_error}"
    )
    assert min_entry >= -atol, (
        f"negative routing entry found: min entry {min_entry}"
    )

    return {
        "passed": True,
        "checked_modules": checked,
        "max_row_error": max_row_error,
        "max_col_error": max_col_error,
        "min_entry": min_entry,
    }


def check_gradients(model, tokenizer, batch_size = 1, seq_len = 32):
    """checks gradients for shc trainable logits and frozen backbone params"""
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

    shc_grad_names = []

    for name, parameter in model.named_parameters():
        grad = parameter.grad

        if parameter.requires_grad:
            if grad is None:
                trainable_without_grad.append(name)
            else:
                trainable_with_grad.append(name)

                if not torch.isfinite(grad).all():
                    nonfinite_grad.append(name)

                if any(x in name for x in ("attn_hc", "mlp_hc", "readout_logits")):
                    shc_grad_names.append(name)
        else:
            if grad is not None:
                frozen_with_grad.append(name)

    assert trainable_with_grad, "no trainable parameters received gradients"
    assert shc_grad_names, "no shc routing logits received gradients"
    assert not trainable_without_grad, (
        "some trainable parameters did not receive gradients: "
        f"{trainable_without_grad[:10]}"
    )
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
        "shc_trainable_with_grad": len(shc_grad_names),
        "frozen_with_grad": len(frozen_with_grad),
    }


def check_identity_equivalence(model_name, atol = 1e-4, layer_idx = 0):
    """checks that fully ablated shc matches the original residual model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    olmo_normal = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code = True,
    ).to(device)
    olmo_normal.eval()

    olmo_wrapped = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code = True,
    ).to(device)

    olmo_wrapped = olmo_shc(
        olmo_wrapped,
        num_streams = 4,
        train_branch = False,
        ablate_mapping = ["pre", "res", "post"],
    )
    olmo_wrapped.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        "Hello, my name is",
        return_tensors = "pt",
    )
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        out_normal = olmo_normal(**inputs)
        out_shc = olmo_wrapped(**inputs)

    logits_normal = out_normal.logits
    logits_shc = out_shc.logits

    full_max_abs_diff = (logits_normal - logits_shc).abs().max().item()
    full_allclose = torch.allclose(logits_normal, logits_shc, atol = atol)

    normal_layer = olmo_normal.model.layers[layer_idx]
    shc_layer = olmo_wrapped.model.layers[layer_idx]

    with torch.no_grad():
        inputs_embeds = olmo_normal.model.embed_tokens(inputs["input_ids"])

        position_ids = torch.arange(
            inputs_embeds.shape[1],
            device = inputs_embeds.device,
        ).unsqueeze(0)

        position_embeddings = olmo_normal.model.rotary_emb(
            inputs_embeds,
            position_ids = position_ids,
        )

        x = inputs_embeds.clone()

        out_normal_layer = normal_layer(
            x.clone(),
            attention_mask = None,
            position_ids = position_ids,
            position_embeddings = position_embeddings,
        )
        if isinstance(out_normal_layer, tuple):
            out_normal_layer = out_normal_layer[0]

        out_shc_layer = shc_layer(
            x.clone(),
            attention_mask = None,
            position_ids = position_ids,
            position_embeddings = position_embeddings,
        )
        if isinstance(out_shc_layer, tuple):
            out_shc_layer = out_shc_layer[0]

        out_shc_readout = (
            out_shc_layer.mean(dim = 2)
            if out_shc_layer.dim() == 4
            else out_shc_layer
        )

        layer_max_abs_diff = (out_normal_layer - out_shc_readout).abs().max().item()
        layer_allclose = torch.allclose(
            out_normal_layer,
            out_shc_readout,
            atol = 1e-3,
        )

        attn_hc = shc_layer.attn_hc

        branch_out = attn_hc.branch(
            x.clone(),
            position_ids = position_ids,
            position_embeddings = position_embeddings,
        )
        if isinstance(branch_out, tuple):
            branch_out = branch_out[0]

        shc_out = attn_hc(
            x.clone(),
            position_ids = position_ids,
            position_embeddings = position_embeddings,
            readout = True,
        )
        if isinstance(shc_out, tuple):
            shc_out = shc_out[0]

        expected = x + branch_out

        direct_max_abs_diff = (shc_out - expected).abs().max().item()
        direct_allclose = torch.allclose(
            shc_out,
            expected,
            atol = atol,
        )

        n = attn_hc.num_streams
        h_res = sinkhorn_logspace(
            attn_hc.res_logits.view(1, 1, n, n).expand(
                x.shape[0],
                x.shape[1],
                n,
                n,
            ),
            num_iters = attn_hc.sinkhorn_iters,
            eps = attn_hc.eps,
        )

        h_res_row_error = (h_res[0, 0].sum(dim = -1) - 1.0).abs().max().item()
        h_res_col_error = (h_res[0, 0].sum(dim = -2) - 1.0).abs().max().item()

    assert full_allclose, (
        f"full model identity equivalence failed: max diff {full_max_abs_diff}"
    )
    assert layer_allclose, (
        f"layer identity equivalence failed: max diff {layer_max_abs_diff}"
    )
    assert direct_allclose, (
        f"direct shc identity equivalence failed: max diff {direct_max_abs_diff}"
    )

    return {
        "passed": True,
        "model_name": model_name,
        "atol": atol,
        "layer_idx": layer_idx,
        "full_logits_shape": list(logits_normal.shape),
        "full_max_abs_diff": full_max_abs_diff,
        "full_allclose": full_allclose,
        "layer_max_abs_diff": layer_max_abs_diff,
        "layer_allclose_atol_1e_3": layer_allclose,
        "direct_max_abs_diff": direct_max_abs_diff,
        "direct_allclose": direct_allclose,
        "h_pre": torch.sigmoid(attn_hc.pre_logits).detach().cpu().tolist(),
        "h_post": (2 * torch.sigmoid(attn_hc.post_logits)).detach().cpu().tolist(),
        "h_res_max_row_error": h_res_row_error,
        "h_res_max_col_error": h_res_col_error,
    }


def check_mhc(
    cfg,
    batch_size = 1,
    seq_len = 32,
    atol = 1e-3,
    run_identity = True,
    identity_atol = 1e-4,
    identity_layer_idx = 0,
    skip_double_stochastic = False,  

):
    """runs all quick shc diagnostics"""
    cfg.model.use_cache = False

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model = inject_method(model, cfg)

    counts = count_trainable_parameters(model)

    results = {
        "method": cfg.method.selected_method,
        "model": cfg.model.pretrained_model_name_or_path,
        "parameter_counts": counts,
    }

    results["shapes"] = check_shapes(
        model = model,
        tokenizer = tokenizer,
        batch_size = batch_size,
        seq_len = seq_len,
    )

    if not skip_double_stochastic:
        results["double_stochastic"] = check_double_stochastic(
        model = model,
        atol = atol,
    )

    results["gradients"] = check_gradients(
        model = model,
        tokenizer = tokenizer,
        batch_size = batch_size,
        seq_len = seq_len,
    )

    if run_identity:
        results["identity_equivalence"] = check_identity_equivalence(
            model_name = cfg.model.pretrained_model_name_or_path,
            atol = identity_atol,
            layer_idx = identity_layer_idx,
        )

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/config.yaml")
    parser.add_argument("--batch_size", type = int, default = 1)
    parser.add_argument("--seq_len", type = int, default = 32)
    parser.add_argument("--atol", type = float, default = 1e-3)
    parser.add_argument("--output", default = None)
                                        # optionally skip the identity test for speed
    parser.add_argument("--skip_identity", action = "store_true")
    parser.add_argument("--skip_double_stochastic", action="store_true")
    parser.add_argument("--identity_atol", type = float, default = 1e-4)
    parser.add_argument("--identity_layer_idx", type = int, default = 0)
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_dotlist(overrides)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.method.selected_method.lower() not in ["shc", "shc_lora", "shc_vera", "mhc_lite"]:
        raise ValueError(
            "diagnostics.py is intended for methods containing SHC. "
            f"got method.selected_method={cfg.method.selected_method}"
        )

    results = check_mhc(
        cfg = cfg,
        batch_size = args.batch_size,
        seq_len = args.seq_len,
        atol = args.atol,
        run_identity = not args.skip_identity,
        identity_atol = args.identity_atol,
        identity_layer_idx = args.identity_layer_idx,
        skip_double_stochastic = args.skip_double_stochastic,,
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