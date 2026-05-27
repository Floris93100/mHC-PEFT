# benchmark.py
""" runs external benchmarks with lm-evaluation-harness """

import json
import os
import lm_eval
from lm_eval.models.huggingface import HFLM

from models.reload import load_finetuned_model

from lm_eval.api.task import ConfigurableTask

_original_build_qa_turn = ConfigurableTask.build_qa_turn

def _patched_build_qa_turn(self, *args, **kwargs):
    """Bulletproof patch to sanitize ALL arguments going into build_qa_turn,
    ensuring no raw lists, tuples, or integers cause an AssertionError."""
    
    # 1. Clean up positional arguments (*args)
    sanitized_args = []
    for arg in args:
        if isinstance(arg, (list, tuple, int, float)):
            sanitized_args.append(str(arg))
        elif isinstance(arg, dict) and "target" in arg:
            # Deep copy or mutate the dictionary target if present
            arg_copy = arg.copy()
            arg_copy["target"] = str(arg_copy["target"])
            sanitized_args.append(arg_copy)
        else:
            sanitized_args.append(arg)
    sanitized_args = tuple(sanitized_args)

    # 2. Clean up keyword arguments (**kwargs)
    sanitized_kwargs = {}
    for key, value in kwargs.items():
        if isinstance(value, (list, tuple, int, float)):
            sanitized_kwargs[key] = str(value)
        elif isinstance(value, dict) and "target" in value:
            value_copy = value.copy()
            value_copy["target"] = str(value_copy["target"])
            sanitized_kwargs[key] = value_copy
        else:
            sanitized_kwargs[key] = value

    # 3. Hand over execution to the original method completely sanitized
    return _original_build_qa_turn(self, *sanitized_args, **sanitized_kwargs)
# Apply the signature-safe patch globally
ConfigurableTask.build_qa_turn = _patched_build_qa_turn


                                        # lm-evaluation-harness task settings:
# HARNESS_TASKS = {                       # task -> (few-shot count, metric key)
#     # "bbh":            (3,  "exact_match"),
#     # "drop":           (3,  "f1"),
#     # "gsm8k":          (8,  "exact_match"),
#     "hellaswag":      (10, "acc"),
#     # "hendrycks_math": (4,  "exact_match"),
#     # "mmlu":           (5,  "acc"),
#     # "piqa":           (0,  "acc"),
#     # "triviaqa":       (5,  "exact_match"),
# }                                       # just as in the mHC paper


def get_metric_value(task_results, metric):
    """Extract a metric value from lm-eval task results."""
    if metric in task_results:
        return task_results[metric]
        
    # Substring match handles "acc,none", "acc_norm,none", and "exact_match,get-answer"
    for key in task_results.keys():
        if metric in key:
            return task_results[key]
            
    raise KeyError(f"could not find metric {metric} in {task_results.keys()}")


def summarize_harness_results(all_results, cfg):
    """ returns short summary of the results using configuration dictionary """
    summary = {}
    # Read directly from the new hydra configuration block
    for task, task_cfg in cfg.tasks.items():
        num_fewshot = task_cfg.fewshot
        metric = task_cfg.metric
        
        summary[task] = {
            "num_fewshot": num_fewshot,
            "metric": metric,
            "value": round(
                get_metric_value(all_results[task], metric),
                4
            ),
        }
    return summary


def save_benchmark_results(summary, output_file):
    """ saves benchmark results as .json """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok = True)
    with open(output_file, "w") as file:
        json.dump(summary, file, indent = 2)


def build_harness_model_from_hf_id(checkpoint_path, cfg):
    """ constructs lm-eval model from a normal HF model id/path """
    return HFLM(                        # constructor
        pretrained = checkpoint_path,
        device = cfg.device,
        batch_size = cfg.batch_size,
    )


def build_harness_model_from_saved_params(checkpoint_path, cfg):
    """ constructs lm-eval model from saved (only) trainable params """
    model, tokenizer, _ = load_finetuned_model(checkpoint_path)
    return HFLM(                        # constructor
        pretrained = model,
        tokenizer = tokenizer,
        device = cfg.device,
        batch_size = cfg.batch_size,
        model_kwargs = {"use_cache": True},
    )


def build_harness_model(checkpoint_path, cfg):
    """ constructs lm-eval model wrapper (from HF or saved trainable params) """
    metadata_path = os.path.join(checkpoint_path, "reload_metadata.json")

    if os.path.exists(metadata_path):
        return build_harness_model_from_saved_params(checkpoint_path, cfg)
    return build_harness_model_from_hf_id(checkpoint_path, cfg)


def evaluate_benchmarks(checkpoint_path, cfg):
    """ runs lm-eval benchmarks on a saved or pretrained model """
    all_results = {}
    harness_model = build_harness_model(checkpoint_path, cfg)

    # Loop over hydra configured tasks instead of the hardcoded global dict
    for task, task_cfg in cfg.tasks.items():
        output = lm_eval.simple_evaluate(
            model = harness_model,
            tasks = [task],
            num_fewshot = task_cfg.fewshot,
            limit = cfg.limit,
        )
        all_results[task] = output["results"][task]

    # Pass cfg down to the summary helper
    summary = summarize_harness_results(all_results, cfg)
    save_benchmark_results(summary, cfg.harness_output_file)
    return summary