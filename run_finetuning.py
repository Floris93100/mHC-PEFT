# run_finetuning.py
""" runs one full training/evaluation experiment """

import os
import random
import numpy as np
import torch
# https://omegaconf.readthedocs.io/en/latest/
from omegaconf import OmegaConf

from models.loading import load_model_and_tokenizer
from models.injection import inject_method, print_trainable_parameters
from data.data_loader import prepare_lm_dataset, prepare_train_eval_test_datasets
from train import train_model
from evaluate import evaluate_perplexity


def set_seed(seed):
    """ sets random seeds for reproducible runs """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_config(cfg, output_dir):
    """ saves the resolved config next to the trained parameters """
    os.makedirs(output_dir, exist_ok = True)

    with open(os.path.join(output_dir, "resolved_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve = True))


def run_experiment(cfg):
    """ runs model loading, method injection, training, and evaluation """
    set_seed(cfg.seed)

    print(f"working directory: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))

    model, tokenizer = load_model_and_tokenizer(cfg.model)
    model = inject_method(model, cfg)
    print_trainable_parameters(model)

    train_dataset, validation_dataset, _ = prepare_train_eval_test_datasets(
        cfg.data,
        tokenizer,
        include_test = False,
    )

    if cfg.train.enabled:
        train_model(
            model,
            tokenizer,
            train_dataset,
            cfg.train,
            cfg,
            eval_dataset = validation_dataset,
        )
        save_run_config(cfg, cfg.train.final_trainable_params_dir)
        
                                        # optional extra perplexity evaluation after training,
                                        # separate from the validation set used during training;.
                                        # enable with eval.enabled in config
        
        print_trainable_parameters(model)
        
    if cfg.eval.enabled:
        if cfg.data.eval_split is None:
            raise ValueError("eval.enabled=true, but data.eval_split is null")

        eval_cfg = cfg.data.copy()
        eval_cfg.train_split = cfg.data.eval_split
        eval_dataset = prepare_lm_dataset(eval_cfg, tokenizer)

        metrics = evaluate_perplexity(model, tokenizer, eval_dataset, cfg.eval)
        print(metrics)