"""Preprocessing of manually corrected labels."""

__all__ = ["preprocess_manually_corrected_labels"]

import json
import os
from pathlib import Path

from deepforest import utilities
import numpy as np
import pandas as pd
import rasterio

from deepforest_finetuning.config import ManuallyCorrectedLabelPreprocessingConfig
from deepforest_finetuning.utils import annotations_to_coco


def preprocess_manually_corrected_labels(config: ManuallyCorrectedLabelPreprocessingConfig):
    """Preprocessing of manually corrected labels. This script was used to convert the labels obtained from manually
    correcting DeepForest labels to COCO JSON format. It is not needed for the data provided with the paper since they
    were already preprocessed using this script.
    """

    label_folder = Path(config.input_label_folder)

    for file in os.listdir(label_folder):
        file_path = label_folder / file
        if file.endswith(".xml"):
            annotations = utilities.read_pascal_voc(file_path)

            assert len(annotations["image_path"].unique()) == 1

            image_name = annotations["image_path"].iloc[0]

            raw_image_path = Path(config.input_image_folder) / image_name

            # 20230720_Sauen_3512a1_2x3-tile.xml contains labels for a larger tile
            # we crop the part for which we also have fully manually created labels and labels from SegmentAnyTree
            if file == "20230720_Sauen_3512a1_2x3-tile.xml":

                with rasterio.open(raw_image_path) as image:
                    transform = image.transform

                    source_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

                    src_top_left = np.array([transform.c, transform.f], dtype=np.float64)
                    src_bottom_left = src_top_left.copy()
                    src_bottom_left[1] -= image.height * source_pixel_size[1]

                target_image_paths = [
                    (
                        Path(config.input_image_folder) / "20230720_Sauen_3512a1_8901_115852.tif",
                        "20230720_Sauen_3512a1_8901_115852",
                    ),
                    (
                        Path(config.input_image_folder) / "20230720_Sauen_3512a1_2x3_tile_clipped.tif",
                        "20230720_Sauen_3512a1_2x3_tile_clipped",
                    ),
                ]

                coco_jsons = []
                for target_image_path, file in target_image_paths:
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
                    for _, annotation in annotations.iterrows():
                        x_min_meter = src_top_left[0] + annotation["xmin"] * source_pixel_size[0]
                        x_max_meter = src_top_left[0] + annotation["xmax"] * source_pixel_size[0]
                        y_min_meter = src_top_left[1] - annotation["ymin"] * source_pixel_size[1]
                        y_max_meter = src_top_left[1] - annotation["ymax"] * source_pixel_size[1]

                        if (
                            x_max_meter < target_top_left[0]
                            or x_min_meter > target_top_right[0]
                            or y_max_meter > target_top_left[1]
                            or y_min_meter < target_bottom_left[1]
                        ):
                            continue

                        annotation["xmin"] = int(max(x_min_meter - target_top_left[0], 0) / target_pixel_size[0])
                        annotation["xmax"] = int(
                            min(x_max_meter - target_top_left[0], target_width) / target_pixel_size[0]
                        )
                        annotation["ymin"] = int(max(target_top_left[1] - y_min_meter, 0) / target_pixel_size[1])
                        annotation["ymax"] = int(
                            min(target_top_left[1] - y_max_meter, target_height) / target_pixel_size[1]
                        )
                        clipped_annotations.append(annotation)

                    annotations = pd.DataFrame(clipped_annotations)
                    coco_jsons.append(annotations_to_coco(annotations, target_image_path), file)
            else:
                coco_jsons = [(annotations_to_coco(annotations, raw_image_path), file)]

            for coco_json, file in coco_jsons:

                output_file_path = (Path(config.output_label_folder) / f"{Path(file).stem}_coco").with_suffix(".json")
                output_file_path.parent.mkdir(exist_ok=True, parents=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(coco_json, f, indent=4)
