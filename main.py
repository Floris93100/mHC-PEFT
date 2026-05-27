# main.py
""" main() for the project """

import os
import hydra
# https://omegaconf.readthedocs.io/en/latest/
from omegaconf import DictConfig

from run_finetuning import run_experiment
from run_benchmarks import run_benchmarks 


# https://hydra.cc/docs/intro/
@hydra.main(version_base = None, config_path = "configs", config_name = "config")
def main(cfg: DictConfig):
    """ starts one Hydra experiment run """
    if cfg.carbon_tracking.enabled:
        # https://mlco2.github.io/codecarbon/
        from codecarbon import EmissionsTracker

        os.makedirs(cfg.carbon_tracking.output_dir, exist_ok = True)
        tracker = EmissionsTracker(
            project_name = cfg.carbon_tracking.project_name,
            output_dir = cfg.carbon_tracking.output_dir,
            output_file = cfg.carbon_tracking.output_file,
            experiment_id = cfg.carbon_tracking.experiment_id,
            log_level = "error",
        )

        tracker.start()
        try:
            if cfg.mode == "train":
                run_experiment(cfg)
            elif cfg.mode == "benchmark":
                run_benchmarks(cfg)
            else:
                raise ValueError(f"unknown mode: {cfg.mode}")
        finally:
            emissions = tracker.stop()
            print(f"carbon emissions: {emissions} kg co2eq")
    else:
        if cfg.mode == "train":
            run_experiment(cfg)
        elif cfg.mode == "benchmark":
            run_benchmarks(cfg)
        else:
            raise ValueError(f"unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()