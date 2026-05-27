# data/data_loader.py
""" loads text dataset and turns it into tokenized LM examples """

import os
import json
# https://huggingface.co/docs/datasets/package_reference/loading_methods
from datasets import Dataset, load_dataset, load_from_disk

from data.prepare_data import tokenize_dataset
from data.stratify import proportional_sample


def load_tiny_dataset():
    """ LLM-generated tiny dataset for quick smoke tests (kleine toeter) """
    examples = [
        {
            "messages": [
                {"role": "user", "content": "Explain what a residual connection is."},
                {"role": "assistant", "content": "A residual connection adds the input of a layer back to its output."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is parameter-efficient finetuning?"},
                {"role": "assistant", "content": "It adapts a model by training only a small number of extra parameters."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What does LoRA do?"},
                {"role": "assistant", "content": "LoRA adds low-rank trainable matrices to selected model layers."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "What is a hyper-connection?"},
                {"role": "assistant", "content": "It generalizes residual connections by routing information through multiple streams."},
            ]
        },
    ]

    return Dataset.from_list(examples)


def load_raw_dataset(cfg):
    """ loads the tiny dataset or a Hugging Face dataset """
    if cfg.source_type == "tiny":
        dataset = load_tiny_dataset()
    elif cfg.source_type == "hf":
        dataset = load_dataset(
            cfg.hf_dataset_name,
            cfg.hf_dataset_subset if cfg.hf_dataset_subset else None,
            split = cfg.train_split,
        )
    elif cfg.source_type == "mixture":                              
        dataset = load_dataset(
            cfg.hf_dataset_name,
            cfg.hf_dataset_subset if cfg.hf_dataset_subset else None,
            split=cfg.train_split,
        )
        dataset = proportional_sample(
            dataset,
            n = getattr(cfg, "stratify_max_samples", 20000),
            source_column = getattr(cfg, "stratify_source_column", "source"),
            seed = getattr(cfg, "validation_seed", 34343),
        )
    else:
        raise ValueError(f"unknown data source: {cfg.source_type}")

    return dataset


def load_tokenized_train_eval_test_datasets(cfg, include_test = False):
    """ loads already-tokenized train/validation/test splits """
    train_dataset = load_from_disk(
        os.path.join(cfg.tokenized_dataset_dir, "train")
    )
    eval_path = os.path.join(cfg.tokenized_dataset_dir, "validation")
    eval_dataset = load_from_disk(eval_path) if os.path.exists(eval_path) else None
    test_path = os.path.join(cfg.tokenized_dataset_dir, "test")
    test_dataset = None

    if include_test and os.path.exists(test_path):
        test_dataset = load_from_disk(test_path)
    return train_dataset, eval_dataset, test_dataset


def load_cached_train_eval_test_datasets(cfg):
    """ loads fixed train/validation/test splits from disk """
    train_dataset = load_from_disk(os.path.join(cfg.cached_dataset_dir, "train"))
    eval_path = os.path.join(cfg.cached_dataset_dir, "validation")
    eval_dataset = load_from_disk(eval_path) if os.path.exists(eval_path) else None
    test_path = os.path.join(cfg.cached_dataset_dir, "test")
    test_dataset = load_from_disk(test_path) if os.path.exists(test_path) else None
    return train_dataset, eval_dataset, test_dataset


def add_row_ids(dataset):
    """ adds stable row ids before shuffling/splitting """
    if "_row_id" in dataset.column_names:
        return dataset
    return dataset.map(
        lambda example, index: {"_row_id": index},
        with_indices = True,
    )


def select_by_row_ids(dataset, row_ids):
    """ selects dataset rows using saved row ids """
    row_id_to_position = {
        int(row_id): position
        for position, row_id in enumerate(dataset["_row_id"])
    }
    positions = [
        row_id_to_position[int(row_id)]
        for row_id in row_ids
    ]

    return dataset.select(positions)


def save_split_ids(eval_dataset, test_dataset, cfg):
    """ saves exact validation/test row ids for later reuse """
    output_file = getattr(cfg, "split_indices_output_file", None)
    if output_file is None:
        return
    
    payload = {
        "dataset_name" : cfg.hf_dataset_name,
        "dataset_subset" : cfg.hf_dataset_subset,
        "train_split": cfg.train_split,
        "validation_seed" : cfg.validation_seed,
        "validation_split_strategy" : cfg.validation_split_strategy,
        "validation_row_ids" : [] if eval_dataset is None else list(eval_dataset["_row_id"]),
        "test_row_ids" : [] if test_dataset is None else list(test_dataset["_row_id"]),
    }

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok = True)
    with open(output_file, "w") as file:
        json.dump(payload, file, indent = 2)


def split_from_saved_ids(dataset, cfg):
    """ recreates validation/test sets from saved row ids """
    with open(cfg.split_indices_input_file) as file:
        payload = json.load(file)

    validation_row_ids = payload.get("validation_row_ids", [])
    test_row_ids = payload.get("test_row_ids", [])

    eval_dataset = None
    if validation_row_ids:
        eval_dataset = select_by_row_ids(dataset, validation_row_ids)
    test_dataset = None
    if test_row_ids:
        test_dataset = select_by_row_ids(dataset, test_row_ids)

    reserved_row_ids = set(validation_row_ids + test_row_ids)
    train_row_ids = [
        row_id
        for row_id in dataset["_row_id"]
        if row_id not in reserved_row_ids
    ]
    train_dataset = select_by_row_ids(dataset, train_row_ids)

    return train_dataset, eval_dataset, test_dataset


def split_train_eval_test_dataset(dataset, cfg):
    """ creates deterministic train/validation/test splits """
    dataset = add_row_ids(dataset)

    if getattr(cfg, "split_indices_input_file", None):
        return split_from_saved_ids(dataset, cfg)

    dataset_size = len(dataset)
    validation_samples = getattr(cfg, "validation_samples", 0) or 0
    test_samples = getattr(cfg, "test_samples", 0) or 0

    if validation_samples + test_samples >= dataset_size:
        raise ValueError("validation_samples + test_samples must be smaller than dataset size")

    split_strategy = getattr(cfg, "validation_split_strategy", "random")
    validation_seed = getattr(cfg, "validation_seed", 343434)

    if split_strategy == "random":
        dataset = dataset.shuffle(seed = validation_seed)
        test_start = dataset_size - test_samples
        eval_start = test_start - validation_samples
        train_dataset = dataset.select(range(0, eval_start))

        eval_dataset = None
        if validation_samples > 0:
            eval_dataset = dataset.select(range(eval_start, test_start))
        test_dataset = None
        if test_samples > 0:
            test_dataset = dataset.select(range(test_start, dataset_size))

    elif split_strategy == "tail":
        train_end = dataset_size - validation_samples - test_samples
        eval_end = train_end + validation_samples

        train_dataset = dataset.select(range(0, train_end))

        eval_dataset = None
        if validation_samples > 0:
            eval_dataset = dataset.select(range(train_end, eval_end))
        test_dataset = None
        if test_samples > 0:
            test_dataset = dataset.select(range(eval_end, dataset_size))

    else:
        raise ValueError(f"unknown validation_split_strategy: {split_strategy}")

    save_split_ids(eval_dataset, test_dataset, cfg)

    max_train_samples = getattr(cfg, "max_train_samples", None)
    if max_train_samples is not None:
        train_dataset = train_dataset.select(
            range(min(max_train_samples, len(train_dataset)))
        )
    max_eval_samples = getattr(cfg, "max_eval_samples", None)
    if eval_dataset is not None and max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(min(max_eval_samples, len(eval_dataset)))
        )

    return train_dataset, eval_dataset, test_dataset


def prepare_train_eval_test_datasets(cfg, tokenizer, include_test = False):
    """ loads, splits, and tokenizes train/eval/test datasets """
    if getattr(cfg, "tokenized_dataset_dir", None):
        return load_tokenized_train_eval_test_datasets(
            cfg,
            include_test = include_test,
        )
    
    if getattr(cfg, "cached_dataset_dir", None):
        train_dataset, eval_dataset, test_dataset = load_cached_train_eval_test_datasets(cfg)
    else:
        dataset = load_raw_dataset(cfg)
        train_dataset, eval_dataset, test_dataset = split_train_eval_test_dataset(dataset, cfg)

    train_dataset = tokenize_dataset(train_dataset, tokenizer, cfg)

    if eval_dataset is not None:
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, cfg)

    if include_test and test_dataset is not None:
        test_dataset = tokenize_dataset(test_dataset, tokenizer, cfg)
    else:
        test_dataset = None

    return train_dataset, eval_dataset, test_dataset


def prepare_lm_dataset(cfg, tokenizer):
    """ helper function to load and tokenize dataset """
    dataset = load_raw_dataset(cfg)
    return tokenize_dataset(dataset, tokenizer, cfg)