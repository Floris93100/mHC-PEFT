# models/reload.py
""" reloads saved trainable parameters into an injected model """

import json
import os
import torch
from omegaconf import OmegaConf

from models.loading import load_model_and_tokenizer
from models.injection import inject_method


def load_trainable_parameters(model, params_path):
    """ loads saved trainable parameters into the injected model """
    state = torch.load(params_path, map_location = "cpu")
    model_params = dict(model.named_parameters())
    missing = []

    for name, value in state.items():
        if name not in model_params:
            missing.append(name)
            continue

        target = model_params[name]
                                        #! check of dit de shape bug pakt?
        if tuple(value.shape) != tuple(target.shape):
            raise ValueError(
                f"shape mismatch for {name}: "
                f"saved {tuple(value.shape)}, model {tuple(target.shape)}"
            )

        target.data.copy_(
            value.to(
                device = target.device,
                dtype = target.dtype,
            )
        )

    if missing:
        raise ValueError(f"saved params not found in model: {missing[:10]}")

    return model

def build_reload_cfg(metadata):
    """ builds a (partly incomplete) config from reload metadata """
    cfg = {
        "model" : {
            "pretrained_model_name_or_path" : metadata["model"],
            "torch_dtype" : metadata["torch_dtype"],
            "device_map" : metadata["device_map"],
            "trust_remote_code" : metadata["trust_remote_code"],
            "use_cache" : False,
        },
        "method" : {
            "selected_method" : metadata["method"],

            "shc_num_streams" : metadata.get("shc_num_streams", 1),
            "shc_sinkhorn_iterations" : metadata.get("shc_sinkhorn_iterations", 20),
            "shc_sinkhorn_epsilon" : metadata.get("shc_sinkhorn_epsilon", 1e-1),
            "shc_train_wrapped_branch" : metadata.get("shc_train_wrapped_branch", True),
            "shc_dropout_stream" : metadata.get("shc_dropout_stream", 0.0),
            "shc_dropout_res" : metadata.get("shc_dropout_res", 0.1),
            "shc_noise_std" : metadata.get("shc_noise_std", 1e-2),
            "shc_ablation_mapping" : metadata.get("shc_ablation_mapping", []),
            "shc_softmax_readout" : metadata.get("shc_softmax_readout", False),

            "peft_lora_rank" : metadata.get("peft_lora_rank", 8),
            "peft_lora_alpha" : metadata.get("peft_lora_alpha", 16),
            "peft_lora_dropout" : metadata.get("peft_lora_dropout", 0.05),
            "peft_vera_rank" : metadata.get("peft_vera_rank", 8),
            "peft_vera_dropout" : metadata.get("peft_vera_dropout", 0.05),
            "peft_vera_projection_prng_key" : metadata.get("peft_vera_projection_prng_key", 343434),
            "peft_bias" : metadata.get("peft_bias", "none"),
            "peft_target_modules" : metadata.get("peft_target_modules", ["all-linear"]),
            
            # FIX 1: Ensure it defaults gracefully but respects the saved metadata list
            "peft_target_modules" : metadata.get("peft_target_modules", ["q_proj", "v_proj"]),
            
            # FIX 2: Add the missing IA3 feedforward modules key
            "peft_feedforward_modules" : metadata.get("peft_feedforward_modules", ["up_proj", "down_proj", "gate_proj"]),

            "prompt_tuning_num_virtual_tokens" : metadata.get("prompt_tuning_num_virtual_tokens", 20),
            "prompt_tuning_init_text" : metadata.get("prompt_tuning_init_text", None),

            "layer_tuning_train_last_n_layers" : metadata.get("layer_tuning_train_last_n_layers", 1),
            "layer_tuning_train_lm_head" : metadata.get("layer_tuning_train_lm_head", False),
            "layer_tuning_train_norm" : metadata.get("layer_tuning_train_norm", False),
        },
    }

    return OmegaConf.create(cfg)


def load_finetuned_model(model_dir):
    """ loads base model, injects method, and restores trained params """
    metadata_path = os.path.join(model_dir, "reload_metadata.json")
    params_path = os.path.join(model_dir, "trainable_params.pt")

    with open(metadata_path) as file:
        metadata = json.load(file)

    cfg = build_reload_cfg(metadata)

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model = inject_method(model, cfg)
    model = load_trainable_parameters(model, params_path)
    # enable HF generation cache for reloaded (injected) models to avoid
    # recomputing full context on every generation step (greatly speeds up inference)
    try:
        model.config.use_cache = True
    except Exception:
        pass
    model.eval()

    return model, tokenizer, cfg