# evaluate_base_val_test.py
"""
evaluates a base or finetuned causal LM on tokenized validation/test splits

examples:

base OLMo-2 on validation and test:
python evaluate_lm_splits.py \
  --model_name allenai/OLMo-2-0425-1B \
  --tokenized_dataset_dir /path/to/packed_dataset \
  --splits validation test \
  --output_dir results/base_olmo2 \
  --batch_size 4

finetuned model on test only:
python evaluate_lm_splits.py \
  --checkpoint_dir outputs/my_run/final_model \
  --tokenized_dataset_dir /path/to/packed_dataset \
  --splits test \
  --output_dir results/my_run_test \
  --batch_size 4

the tokenized dataset dir should contain split folders, e.g.:
  /path/to/packed_dataset/validation
  /path/to/packed_dataset/test
"""
# LLM-generated

import argparse
import json
import math
import os

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from models.reload import load_finetuned_model


def load_base_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
        trust_remote_code = True,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    return model, tokenizer


def load_eval_model(model_name, checkpoint_dir):
    if checkpoint_dir is not None:
        model, tokenizer, reload_cfg = load_finetuned_model(checkpoint_dir)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        return model, tokenizer, {
            "model_type": "finetuned",
            "checkpoint_dir": checkpoint_dir,
            "reload_method": reload_cfg.method.selected_method,
        }

    model, tokenizer = load_base_model_and_tokenizer(model_name)

    return model, tokenizer, {
        "model_type": "base",
        "model_name": model_name,
    }


def safe_perplexity(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def evaluate_split(model, tokenizer, dataset, output_dir, split_name, batch_size):
    collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )

    args = TrainingArguments(
        output_dir = os.path.join(output_dir, f"trainer_{split_name}"),
        per_device_eval_batch_size = batch_size,
        report_to = "none",
        remove_unused_columns = False,
        bf16 = True,
    )

    trainer = Trainer(
        model = model,
        args = args,
        eval_dataset = dataset,
        data_collator = collator,
        processing_class = tokenizer,
    )

    metrics = trainer.evaluate(metric_key_prefix = split_name)

    loss_key = f"{split_name}_loss"
    if loss_key in metrics:
        metrics[f"{split_name}_perplexity"] = safe_perplexity(metrics[loss_key])

    metrics[f"{split_name}_examples"] = len(dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "allenai/OLMo-2-0425-1B")
    parser.add_argument("--checkpoint_dir", default = None)
    parser.add_argument("--tokenized_dataset_dir", required = True)
    parser.add_argument("--splits", nargs = "+", default = ["validation", "test"])
    parser.add_argument("--output_dir", required = True)
    parser.add_argument("--output_file", default = "eval_metrics.json")
    parser.add_argument("--batch_size", type = int, default = 1)
    args = parser.parse_args()

    valid_splits = {"train", "validation", "test"}
    unknown_splits = [split for split in args.splits if split not in valid_splits]
    if unknown_splits:
        raise ValueError(f"unknown split(s): {unknown_splits}. valid splits: {sorted(valid_splits)}")

    if args.checkpoint_dir is not None and not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"checkpoint_dir does not exist: {args.checkpoint_dir}")

    os.makedirs(args.output_dir, exist_ok = True)

    model, tokenizer, model_info = load_eval_model(
        model_name = args.model_name,
        checkpoint_dir = args.checkpoint_dir,
    )

    all_metrics = {
        **model_info,
        "tokenized_dataset_dir": args.tokenized_dataset_dir,
        "splits": args.splits,
        "batch_size": args.batch_size,
    }

    for split_name in args.splits:
        split_path = os.path.join(args.tokenized_dataset_dir, split_name)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"split path does not exist: {split_path}")

        dataset = load_from_disk(split_path)

        metrics = evaluate_split(
            model = model,
            tokenizer = tokenizer,
            dataset = dataset,
            output_dir = args.output_dir,
            split_name = split_name,
            batch_size = args.batch_size,
        )

        all_metrics.update(metrics)

    output_path = os.path.join(args.output_dir, args.output_file)
    with open(output_path, "w") as file:
        json.dump(all_metrics, file, indent = 2)

    print(json.dumps(all_metrics, indent = 2))
    print(f"saved metrics to: {output_path}")


if __name__ == "__main__":
    main()