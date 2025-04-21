"""Fine-tuning script."""

import fire

from deepforest_finetuning.config import TrainingConfig
from deepforest_finetuning.training import finetuning
from deepforest_finetuning.utils import load_config


def fine_tuning(config_path: str):
    """Fine-tuning script."""

    config = load_config(config_path, TrainingConfig)
    finetuning(config)


if __name__ == "__main__":
    fire.Fire(fine_tuning)
