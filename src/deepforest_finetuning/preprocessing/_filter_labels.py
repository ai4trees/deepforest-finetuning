"""Filtering of labels using non-maximum suppression."""

__all__ = ["filter_bounding_boxed_with_size_based_nms", "filter_labels"]

from copy import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torchvision.ops import nms

from deepforest_finetuning.config import LabelFilteringConfig


def filter_bounding_boxed_with_size_based_nms(coco_json: Dict[str, Any], iou_threshold: float) -> Dict[str, Any]:
    """
    Applies non-maximum suppression to the bounding boxes, using the box sizes as scores.

    Args:
        coco_json: A dictionary containing annotations in COCO format.
        iou_threshold: IoU threshold for non-maximum suppression.

    Returns:
        A new dictionary containing annotations that are filtered by non-maximum suppression.
    """

    assert len(coco_json["images"]) == 1
    coco_json = copy(coco_json)

    bounding_boxes = []
    bounding_box_sizes = []

    if len(coco_json["annotations"]) == 0:
        return coco_json

    for annotation in coco_json["annotations"]:
        bounding_box = annotation["bbox"]
        bounding_boxes.append(
            [bounding_box[0], bounding_box[1], bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]]
        )
        bounding_box_sizes.append(bounding_box[2] * bounding_box[3])

    selected_indices = nms(
        torch.tensor(bounding_boxes).float(),
        scores=torch.tensor(bounding_box_sizes).float(),
        iou_threshold=iou_threshold,
    )
    coco_json["annotations"] = [coco_json["annotations"][i] for i in selected_indices]

    return coco_json


def filter_labels(config: LabelFilteringConfig):
    """Filters labels using non-maximum suppression."""

    base_dir = Path(config.base_dir)
    label_folder = base_dir / config.input_label_folder

    subfolders = [file for file in os.listdir(label_folder) if os.path.isdir(os.path.join(label_folder, file))]

    for subfolder in subfolders:
        for file in os.listdir(label_folder / subfolder):
            file_path = label_folder / subfolder / file
            if file.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as f:
                    coco_json = json.load(f)

                assert len(coco_json["images"]) == 1
                coco_json = filter_bounding_boxed_with_size_based_nms(coco_json, iou_threshold=config.iou_threshold)

                output_file_path = base_dir / config.output_label_folder / subfolder / file
                output_file_path.parent.mkdir(exist_ok=True, parents=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(coco_json, f, indent=4)
