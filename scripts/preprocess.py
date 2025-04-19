"""Preprocessing scripts."""

from functools import partial
from typing import Callable, Type

import fire

from deepforest_finetuning.config import PointCloudLabelProjectionConfig, ImageRescalingConfig
from deepforest_finetuning.preprocessing import project_point_cloud_labels, rescale_images
from deepforest_finetuning.utils import load_config


def preprocessing_step(config_path: str, config_type: Type, script_fn: Callable):
    """
    Loads the specified config file, parses it based on the given configuration type and then calls the script function
    with the given configuration.

    Args:
        config_path: Path of the configuration file to be loaded.
        config_type: Configuration type.
        script_fn: Callable implementing the preprocessing script to be executed with the given configuration.
    """

    config = load_config(config_path, config_type)
    script_fn(config)


if __name__ == "__main__":
    preprocessing_steps = [
        ("project_point_cloud_labels", project_point_cloud_labels, PointCloudLabelProjectionConfig),
        ("rescale_images", rescale_images, ImageRescalingConfig),
    ]
    fire_dict = {}
    for step in preprocessing_steps:
        script_name, script_fn, config_type = step
        fire_dict[script_name] = partial(preprocessing_step, config_type=config_type, script_fn=script_fn)

    fire.Fire(fire_dict)
