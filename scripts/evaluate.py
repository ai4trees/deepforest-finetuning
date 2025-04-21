"""Evaluation script."""

import fire

from deepforest import utilities

from deepforest_finetuning.config import EvaluationConfig
from deepforest_finetuning.evaluation import evaluate
from deepforest_finetuning.utils import load_config


def evaluation_script(config_path: str):
    """Evaluation script."""

    config = load_config(config_path, EvaluationConfig)

    prediction = utilities.read_file(config.prediction_file)
    target = utilities.read_file(config.label_file)

    evaluate(prediction, target, config.iou_threshold, config.output_file)


if __name__ == "__main__":
    fire.Fire(evaluation_script)
