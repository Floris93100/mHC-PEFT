# data/stratify.py
""" stratifies datasets based on inner sources/subdatasets """

import math
import random
from collections import Counter, defaultdict

from datasets import Dataset


def proportional_sample(
    dataset: Dataset,
    n: int,
    source_column: str = "source",
    seed: int = 42,
) -> Dataset:
    """ draws n samples while preserving source_column proportions """
    sources = dataset[source_column]
    total = len(sources)

    if n >= total:
        return dataset

    actual_counts = Counter(sources)

    exact_counts = {
        source: n * count / total
        for source, count in actual_counts.items()
    }
    sampled_counts = {
        source: math.floor(value)
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

    source_to_indices = defaultdict(list)

    for index, source in enumerate(sources):
        source_to_indices[source].append(index)

    rng = random.Random(seed)
    selected_indices = []

    for source, sample_count in sampled_counts.items():
        available_indices = source_to_indices[source]
        selected_indices.extend(
            rng.sample(
                available_indices,
                sample_count,
            )
        )

    rng.shuffle(selected_indices)

    return dataset.select(selected_indices)