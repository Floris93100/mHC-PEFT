# data/build_320k_train_2500_eval_cache.py
"""builds raw cache with reproduced 320k train and larger validation/test splits"""
# LLM-generated

import argparse
import json
import os
import random
from collections import Counter, defaultdict

from datasets import concatenate_datasets, load_dataset
from omegaconf import OmegaConf

from data.stratify import proportional_sample


def add_original_row_id(dataset):
    if "_original_row_id" in dataset.column_names:
        return dataset
    return dataset.map(
        lambda example, index: {"_original_row_id": index},
        with_indices = True,
    )


def remove_internal_columns(dataset):
    columns = [
        column
        for column in dataset.column_names
        if column.startswith("_")
    ]
    if columns:
        return dataset.remove_columns(columns)
    return dataset


def select_extra_stratified(dataset, n, source_column, seed):
    rng = random.Random(seed)

    source_to_indices = defaultdict(list)
    for index, source in enumerate(dataset[source_column]):
        source_to_indices[source].append(index)

    total = len(dataset)
    counts = Counter(dataset[source_column])

    exact_counts = {
        source: n * count / total
        for source, count in counts.items()
    }
    sampled_counts = {
        source: int(value)
        for source, value in exact_counts.items()
    }

    remainder = n - sum(sampled_counts.values())
    sorted_sources = sorted(
        exact_counts,
        key = lambda source: exact_counts[source] - sampled_counts[source],
        reverse = True,
    )

    for source in sorted_sources[:remainder]:
        sampled_counts[source] += 1

    selected = []
    for source, sample_count in sampled_counts.items():
        available = source_to_indices[source]
        selected.extend(rng.sample(available, min(sample_count, len(available))))

    rng.shuffle(selected)
    return dataset.select(selected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/config.yaml")
    parser.add_argument("--output_dir", required = True)
    parser.add_argument("--old_total_samples", type = int, default = 322000)
    parser.add_argument("--train_samples", type = int, default = 320000)
    parser.add_argument("--old_validation_samples", type = int, default = 1000)
    parser.add_argument("--old_test_samples", type = int, default = 1000)
    parser.add_argument("--new_validation_samples", type = int, default = 2500)
    parser.add_argument("--new_test_samples", type = int, default = 2500)
    parser.add_argument("--seed", type = int, default = 343434)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_cfg = cfg.data
    source_column = getattr(data_cfg, "stratify_source_column", "source")

    full_dataset = load_dataset(
        data_cfg.hf_dataset_name,
        data_cfg.hf_dataset_subset if data_cfg.hf_dataset_subset else None,
        split = data_cfg.train_split,
    )
    full_dataset = add_original_row_id(full_dataset)

    old_sample = proportional_sample(
        full_dataset,
        n = args.old_total_samples,
        source_column = source_column,
        seed = args.seed,
    )

    old_sample = old_sample.shuffle(seed = args.seed)

    test_start = args.old_total_samples - args.old_test_samples
    eval_start = test_start - args.old_validation_samples

    old_train = old_sample.select(range(0, eval_start))
    old_validation = old_sample.select(range(eval_start, test_start))
    old_test = old_sample.select(range(test_start, args.old_total_samples))

    if len(old_train) != args.train_samples:
        raise ValueError(f"expected {args.train_samples} train examples, got {len(old_train)}")

    used_ids = set(old_sample["_original_row_id"])

    def is_unused(example):
        return example["_original_row_id"] not in used_ids

    remaining = full_dataset.filter(
        is_unused,
        num_proc = getattr(data_cfg, "preprocessing_num_proc", 1),
    )

    extra_validation_n = args.new_validation_samples - len(old_validation)
    extra_test_n = args.new_test_samples - len(old_test)

    if extra_validation_n < 0 or extra_test_n < 0:
        raise ValueError("new validation/test sizes must be at least old validation/test sizes")

    extra_validation = select_extra_stratified(
        remaining,
        n = extra_validation_n,
        source_column = source_column,
        seed = args.seed + 1,
    )

    validation_extra_ids = set(extra_validation["_original_row_id"])

    def not_in_extra_validation(example):
        return example["_original_row_id"] not in validation_extra_ids

    remaining_for_test = remaining.filter(
        not_in_extra_validation,
        num_proc = getattr(data_cfg, "preprocessing_num_proc", 1),
    )

    extra_test = select_extra_stratified(
        remaining_for_test,
        n = extra_test_n,
        source_column = source_column,
        seed = args.seed + 2,
    )

    new_train = remove_internal_columns(old_train)
    new_validation = remove_internal_columns(
        concatenate_datasets([old_validation, extra_validation])
    )
    new_test = remove_internal_columns(
        concatenate_datasets([old_test, extra_test])
    )

    os.makedirs(args.output_dir, exist_ok = True)

    new_train.save_to_disk(os.path.join(args.output_dir, "train"))
    new_validation.save_to_disk(os.path.join(args.output_dir, "validation"))
    new_test.save_to_disk(os.path.join(args.output_dir, "test"))

    metadata = {
        "seed": args.seed,
        "old_total_samples": args.old_total_samples,
        "train_examples": len(new_train),
        "validation_examples": len(new_validation),
        "test_examples": len(new_test),
        "extra_validation_examples": len(extra_validation),
        "extra_test_examples": len(extra_test),
        "hf_dataset_name": data_cfg.hf_dataset_name,
        "hf_dataset_subset": data_cfg.hf_dataset_subset,
        "train_split": data_cfg.train_split,
        "source_column": source_column,
    }

    with open(os.path.join(args.output_dir, "expanded_eval_metadata.json"), "w") as file:
        json.dump(metadata, file, indent = 2)

    with open(os.path.join(args.output_dir, "dataset_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(data_cfg, resolve = True))

    print(json.dumps(metadata, indent = 2))
    print(f"saved raw cache to: {args.output_dir}")


if __name__ == "__main__":
    main()