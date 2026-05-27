# data/build_dataset_cache.py
""" builds and saves one fixed stratified dataset split """

import os
import argparse

from omegaconf import OmegaConf

from data.data_loader import load_raw_dataset, split_train_eval_test_dataset


def main():
    """ saves fixed train/validation/test splits to disk """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default = "configs/config.yaml")
    parser.add_argument("--output_dir", required = True)
    parser.add_argument("--source_type", default = "mixture")
    parser.add_argument("--stratify_max_samples", type = int, required = True)
    parser.add_argument("--validation_samples", type = int, default = 1000)
    parser.add_argument("--test_samples", type = int, default = 1000)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    cfg.data.source_type = args.source_type
    cfg.data.stratify_max_samples = args.stratify_max_samples
    cfg.data.validation_samples = args.validation_samples
    cfg.data.test_samples = args.test_samples
    cfg.data.max_train_samples = None
    cfg.data.split_indices_output_file = os.path.join(args.output_dir, "split_ids.json")
    cfg.data.split_indices_input_file = None

    os.makedirs(args.output_dir, exist_ok = True)

    dataset = load_raw_dataset(cfg.data)
    train_dataset, eval_dataset, test_dataset = split_train_eval_test_dataset(
        dataset,
        cfg.data,
    )

    train_dataset.save_to_disk(os.path.join(args.output_dir, "train"))
    if eval_dataset is not None:
        eval_dataset.save_to_disk(os.path.join(args.output_dir, "validation"))
    if test_dataset is not None:
        test_dataset.save_to_disk(os.path.join(args.output_dir, "test"))

    with open(os.path.join(args.output_dir, "dataset_config.yaml"), "w") as file:
        file.write(OmegaConf.to_yaml(cfg.data, resolve = True))

    print(f"saved dataset cache to: {args.output_dir}")
    print(f"train examples: {len(train_dataset)}")
    print(f"validation examples: {0 if eval_dataset is None else len(eval_dataset)}")
    print(f"test examples: {0 if test_dataset is None else len(test_dataset)}")


if __name__ == "__main__":
    main()