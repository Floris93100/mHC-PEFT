# run_benchmarks.py
""" runs benchmark mode from the Hydra config """
from hydra.utils import to_absolute_path, get_original_cwd
import os

from benchmark import evaluate_benchmarks


# LLM-generated: path check
def resolve_checkpoint_path(checkpoint_path):
    """ resolves local paths while leaving HF model ids unchanged """
    if checkpoint_path.startswith(".") or checkpoint_path.startswith("/"):
        return to_absolute_path(checkpoint_path)
    
    if checkpoint_path.startswith("outputs/") or checkpoint_path.startswith("final_model"):
        return to_absolute_path(checkpoint_path)

    return checkpoint_path


def run_benchmarks(cfg):
    """ loads a checkpoint or HF model id and runs benchmarks """
    raw_output_path = cfg.benchmark.harness_output_file

    if not os.path.isabs(raw_output_path):
        resolved_output_path = os.path.join(get_original_cwd(), raw_output_path)
    else:
        resolved_output_path = raw_output_path

    cfg.benchmark.harness_output_file = resolved_output_path
    
    summary = evaluate_benchmarks(
        resolve_checkpoint_path(cfg.benchmark.checkpoint_path),
        cfg.benchmark,
    )                                   # outputs are stored in evaluate_benchmarks()
    print(summary)