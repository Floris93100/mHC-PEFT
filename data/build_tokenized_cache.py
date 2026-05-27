# data/build_tokenized_cache.py
""" builds and saves tokenized train/validation/test datasets """

import argparse
import os
import shutil

from omegaconf import OmegaConf

from models.loading import load_tokenizer
from data.data_loader import load_cached_train_eval_test_datasets
from data.prepare_data import tokenize_dataset


def copy_if_exists(source_path, target_path):
    """ copies a metadata file if it exists """
    if os.path.exists(source_path):
        shutil.copyfile(source_path, target_path)


def main():
    """ tokenizes cached raw splits and saves them to disk """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/config.yaml")
    parser.add_argument("--raw_cache_dir", required = True)
    parser.add_argument("--output_dir", required = True)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.data.cached_dataset_dir = args.raw_cache_dir

    os.makedirs(args.output_dir, exist_ok = True)

    tokenizer = load_tokenizer(cfg.model)
    train_dataset, eval_dataset, test_dataset = load_cached_train_eval_test_datasets(
        cfg.data,
    )

    train_dataset = tokenize_dataset(train_dataset, tokenizer, cfg.data)
    if eval_dataset is not None:
        eval_dataset = tokenize_dataset(eval_dataset, tokenizer, cfg.data)
    if test_dataset is not None:
        test_dataset = tokenize_dataset(test_dataset, tokenizer, cfg.data)

    train_dataset.save_to_disk(os.path.join(args.output_dir, "train"))

    if eval_dataset is not None:
        eval_dataset.save_to_disk(os.path.join(args.output_dir, "validation"))
    if test_dataset is not None:
        test_dataset.save_to_disk(os.path.join(args.output_dir, "test"))

    copy_if_exists(
        os.path.join(args.raw_cache_dir, "split_ids.json"),
        os.path.join(args.output_dir, "split_ids.json"),
    )
    copy_if_exists(
        os.path.join(args.raw_cache_dir, "dataset_config.yaml"),
        os.path.join(args.output_dir, "raw_dataset_config.yaml"),
    )

    with open(os.path.join(args.output_dir, "tokenized_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(
            OmegaConf.create({"model": cfg.model, "data": cfg.data}),
            resolve = False
        ))

    print(f"saved tokenized cache to: {args.output_dir}")
    print(f"train examples: {len(train_dataset)}")
    print(f"validation examples: {0 if eval_dataset is None else len(eval_dataset)}")
    print(f"test examples: {0 if test_dataset is None else len(test_dataset)}")


if __name__ == "__main__":
    main()