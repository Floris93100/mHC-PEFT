# evaluate.py
""" evaluates LM """

import json
import math
# https://huggingface.co/docs/transformers/main_classes/trainer
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


def evaluate_perplexity(model, tokenizer, eval_dataset, cfg):
    """ computes perplexity on an evaluation dataset """
    collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = False,                     # causal LM, not masked LM
    )

    args = TrainingArguments(
        output_dir = cfg.output_dir,
        per_device_eval_batch_size = cfg.per_device_eval_batch_size,
        report_to = "none",
        remove_unused_columns = False,
    )

    trainer = Trainer(
        model = model,
        args = args,
        eval_dataset = eval_dataset,
        data_collator = collator,
        processing_class = tokenizer,
    )

    metrics = trainer.evaluate()

    if "eval_loss" in metrics:
        metrics["perplexity"] = math.exp(metrics["eval_loss"])

    with open(cfg.metrics_output_file, "w") as file:
        json.dump(metrics, file, indent = 2)

    return metrics