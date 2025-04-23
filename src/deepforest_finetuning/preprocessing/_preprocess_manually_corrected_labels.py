"""Preprocessing of manually corrected labels."""

__all__ = ["preprocess_manually_corrected_labels"]

from copy import deepcopy
import json
import os
from pathlib import Path

from deepforest import utilities
import numpy as np
import rasterio

from deepforest_finetuning.config import ManuallyCorrectedLabelPreprocessingConfig
from deepforest_finetuning.utils import annotations_to_coco, get_image_size_from_pascal_voc, rescale_coco_json


def preprocess_manually_corrected_labels(  # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-statements
    config: ManuallyCorrectedLabelPreprocessingConfig,
):
    """Preprocessing of manually corrected labels. This script was used to convert the labels obtained from manually
    correcting DeepForest labels to COCO JSON format. It is not needed for the data provided with the paper since they
    were already preprocessed using this script.
    """

    image_name_mapping = {
        "20230720_Sauen_3512a1_2x3_tile.tif": "s1_p1_ext_mc.tif",
        "20230720_Sauen_3512a1_tile.tif": "s1_p2_mc.tif",
        "20230809_Sauen_3510b3_tile.tif": "s2_p1_mc.tif",
    }

    base_dir = Path(config.base_dir)
    input_label_folder = base_dir / config.input_label_folder
    output_label_folder = base_dir / config.output_label_folder
    output_label_folder.parent.mkdir(exist_ok=True, parents=True)

    for file in os.listdir(input_label_folder):
        file_path = input_label_folder / file
        if file.endswith(".xml"):
            annotations = utilities.read_pascal_voc(file_path)

            image_width, image_height = get_image_size_from_pascal_voc(file_path)

            assert len(annotations["image_path"].unique()) == 1

            image_path = annotations["image_path"].iloc[0]
            date_prefix = Path(image_path).stem[:8]
            capture_date = f"{date_prefix[:4]}-{date_prefix[4:6]}-{date_prefix[6:]}"

            image_path = image_name_mapping[image_path]
            annotations["image_path"] = image_path
            annotations["label"] = "Tree"

            target_image_path = base_dir / config.input_image_folder / image_path

            coco_json = annotations_to_coco(annotations, image_width, image_height, capture_date=capture_date)
            coco_json = rescale_coco_json(
                coco_json, target_image_path, source_image_shape=np.array([image_height, image_width])
            )

            output_file_path = (output_label_folder / f"{Path(image_path).stem}_coco").with_suffix(".json")
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(coco_json, f, indent=4)

    # s1_p1_ext_mc contains labels for a larger tile
    # we crop the part for which we also have fully manually and automatically created labels
    if "s1_p1_ext_mc_coco.json" in os.listdir(output_label_folder):
        with open(output_label_folder / "s1_p1_ext_mc_coco.json", "r", encoding="utf-8") as f:
            coco_json = json.load(f)

        source_image_path = base_dir / config.input_image_folder / "s1_p1_ext_mc.tif"
        target_image_path = base_dir / config.input_image_folder / "s1_p1_small.tif"

        with rasterio.open(source_image_path) as image:
            transform = image.transform

            src_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

            src_top_left = np.array([transform.c, transform.f], dtype=np.float64)
            src_bottom_left = src_top_left.copy()
            src_bottom_left[1] -= image.height * src_pixel_size[1]
            src_top_right = src_top_left.copy()
            src_top_right[0] += image.width * src_pixel_size[0]

        with rasterio.open(target_image_path) as image:
            transform = image.transform
            target_width = image.width
            target_height = image.height

            target_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

            target_top_left = np.array([transform.c, transform.f], dtype=np.float64)
            target_bottom_left = target_top_left.copy()
            target_bottom_left[1] -= image.height * target_pixel_size[1]
            target_top_right = target_top_left.copy()
            target_top_right[0] += image.width * target_pixel_size[0]

        clipped_annotations = []
        for annotation in coco_json["annotations"]:
            x_min_meter = src_top_left[0] + annotation["bbox"][0] * src_pixel_size[0]
            x_max_meter = src_top_left[0] + (annotation["bbox"][0] + annotation["bbox"][2]) * src_pixel_size[0]
            y_min_meter = src_top_left[1] - annotation["bbox"][1] * src_pixel_size[1]
            y_max_meter = src_top_left[1] - (annotation["bbox"][1] + annotation["bbox"][3]) * src_pixel_size[1]

            if (
                x_max_meter < target_top_left[0]
                or x_min_meter > target_top_right[0]
                or y_max_meter > target_top_left[1]
                or y_min_meter < target_bottom_left[1]
            ):
                continue

            clipped_x_min = max(int((x_min_meter - target_top_left[0]) / target_pixel_size[0]), 0)
            clipped_x_max = min(int((x_max_meter - target_top_left[0]) / target_pixel_size[0]), target_width)
            clipped_y_min = max(int((target_top_left[1] - y_min_meter) / target_pixel_size[1]), 0)
            clipped_y_max = min(int((target_top_left[1] - y_max_meter) / target_pixel_size[1]), target_height)

            clipped_annotation = deepcopy(annotation)
            clipped_annotation["bbox"] = [
                clipped_x_min,
                clipped_y_min,
                clipped_x_max - clipped_x_min,
                clipped_y_max - clipped_y_min,
            ]

            clipped_annotations.append(clipped_annotation)

        coco_json["images"] = [
            {
                "id": 0,
                "width": target_width,
                "height": target_height,
                "file_name": "s1_p1_small.tif",
                "date_captured": "2023-07-20",
            }
        ]
        coco_json["annotations"] = clipped_annotations

        output_file_path = output_label_folder / "s1_p1_small_coco.json"
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(coco_json, f, indent=4)
