# evaluate_base_val_test.py
""" evaluates base OLMo-2 1B on validation and test set """
# LLM-generated

import argparse
import json
import math
import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def load_model_and_tokenizer(model_name):
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


def evaluate_split(model, tokenizer, dataset, output_dir, split_name, batch_size):
    collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,
    )

    args = TrainingArguments(
        output_dir = os.path.join(output_dir, split_name),
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
        metrics[f"{split_name}_perplexity"] = math.exp(metrics[loss_key])

    metrics[f"{split_name}_examples"] = len(dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "allenai/OLMo-2-0425-1B")
    parser.add_argument("--tokenized_dataset_dir", required = True)
    parser.add_argument("--output_dir", required = True)
    parser.add_argument("--batch_size", type = int, default = 1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok = True)

    model, tokenizer = load_model_and_tokenizer(args.model_name)

    all_metrics = {
        "model_name": args.model_name,
        "tokenized_dataset_dir": args.tokenized_dataset_dir,
    }

    for split_name in ["validation", "test"]:
        split_path = os.path.join(args.tokenized_dataset_dir, split_name)
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

    output_file = os.path.join(args.output_dir, "base_val_test_metrics.json")
    with open(output_file, "w") as file:
        json.dump(all_metrics, file, indent = 2)

    print(json.dumps(all_metrics, indent = 2))
    print(f"saved metrics to: {output_file}")


if __name__ == "__main__":
    main()