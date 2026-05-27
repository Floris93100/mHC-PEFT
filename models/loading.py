# models/loading.py
""" loads pretrained model and tokenizer from Hugging Face """

import torch
# https://huggingface.co/docs/transformers/model_doc/auto
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_torch_dtype(dtype_name):
    """ uniformize dtypes """
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "auto":
        return "auto"
    raise ValueError(f"unknown dtype: {dtype_name}")


def load_tokenizer(cfg):
    """ loads only the tokenizer from Hugging Face (not the model) """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        trust_remote_code = cfg.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(cfg):
    """ load base LM and tokenizer """
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        trust_remote_code = cfg.trust_remote_code,
    )

    if tokenizer.pad_token is None:
                                        # use EOS also as padding
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.pretrained_model_name_or_path,
        torch_dtype = get_torch_dtype(cfg.torch_dtype),
        device_map = cfg.device_map,
        trust_remote_code = cfg.trust_remote_code,
    )
                                        # disable cache during training
    model.config.use_cache = cfg.use_cache

    return model, tokenizer