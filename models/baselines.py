# models/baselines.py
""" baseline finetuning methods for (already loaded) LMs """

# https://huggingface.co/docs/peft/package_reference/lora
# https://huggingface.co/docs/peft/package_reference/ia3
# https://huggingface.co/docs/peft/package_reference/prompt_tuning
# https://huggingface.co/docs/peft/package_reference/peft_model
from peft import (
    LoraConfig,
    IA3Config,
    VeraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)


def apply_lora(model, cfg):
    """ adds LoRA adapters to the given model """
    peft_cfg = LoraConfig(              # PEFT freezes base model internally
        task_type = TaskType.CAUSAL_LM,
        r = cfg.peft_lora_rank,
        lora_alpha = cfg.peft_lora_alpha,
        lora_dropout = cfg.peft_lora_dropout,
        target_modules = list(cfg.peft_target_modules),
        bias = cfg.peft_bias,
    )
    
    return get_peft_model(
        model,
        peft_cfg,
    )


def apply_ia3(model, cfg):
    """ adds IA3 adapters to the given model """
    peft_cfg = IA3Config(               # PEFT freezes base model internally
        task_type = TaskType.CAUSAL_LM,
        target_modules = list(cfg.peft_target_modules),
        feedforward_modules=list(cfg.peft_feedforward_modules),
    )

    return get_peft_model(
        model,
        peft_cfg,
    )


def apply_dora(model, cfg):
    """ adds DoRA adapters to the given model """
    peft_cfg = LoraConfig(              # PEFT freezes base model internally
        task_type = TaskType.CAUSAL_LM,
        r = cfg.peft_lora_rank,
        lora_alpha = cfg.peft_lora_alpha,
        lora_dropout = cfg.peft_lora_dropout,
        target_modules = list(cfg.peft_target_modules),
        bias = cfg.peft_bias,
        use_dora = True,                # DoRA is enabled via LoRA config
    )

    return get_peft_model(
        model,
        peft_cfg,
    )

def apply_vera(model, cfg): 
    """ adds VeRA adapters to given model """
    peft_cfg = VeraConfig(              # PEFT freezes base model internally
        task_type = TaskType.CAUSAL_LM,
        r = cfg.peft_vera_rank,
        target_modules = list(cfg.peft_target_modules),
        projection_prng_key = cfg.peft_vera_projection_prng_key,
        vera_dropout = cfg.peft_vera_dropout,
        bias = cfg.peft_bias,
    )

    return get_peft_model(
        model,
        peft_cfg,
    )



def apply_prompt_tuning(model, method_cfg):
    """ adds trainable soft prompt tokens using PEFT """
    if method_cfg.prompt_tuning_init_text is None:
        prompt_init = PromptTuningInit.RANDOM
        prompt_init_text = None
    else:
        prompt_init = PromptTuningInit.TEXT
        prompt_init_text = method_cfg.prompt_tuning_init_text

    peft_cfg = PromptTuningConfig(      # PEFT freezes base model internally
        task_type = TaskType.CAUSAL_LM,
        num_virtual_tokens = method_cfg.prompt_tuning_num_virtual_tokens,
        prompt_tuning_init = prompt_init,
        prompt_tuning_init_text = prompt_init_text,
        tokenizer_name_or_path = None,
    )

    return get_peft_model(
        model,
        peft_cfg,
    )


def apply_layer_tuning(selected_layers, lm_head = None, final_norm = None):
    """ unfreezes given models, i.e. does layer tuning """
    for layer in selected_layers:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    if lm_head is not None:
        for parameter in lm_head.parameters():
            parameter.requires_grad = True

    if final_norm is not None:
        for parameter in final_norm.parameters():
            parameter.requires_grad = True