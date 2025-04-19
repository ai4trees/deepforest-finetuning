"""Image rescaling."""

__all__ = ["rescale_coco_json_labels", "rescale_images"]

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Dict
import json

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from deepforest_finetuning.config import ImageRescalingConfig
from deepforest_finetuning.utils import coco_bbox_to_polygon


def rescale_coco_json_labels(
    coco_json: Dict[str, Any], source_image_path: Path, target_image_path: Path
) -> Dict[str, Any]:
    """
    Rescale COCO annotations to match a new image size. This function adjusts the bounding boxes and image metadata in a
    COCO-style annotation dictionary to correspond to a new target image size.

    Args:
        coco_json: A dictionary containing annotations in COCO format.
        source_image_path: Path to the original image file used for the annotations.
        target_image_path: Path to the resized image file.

    Returns:
        A new dictionary containing annotations that are rescaled to match the target image size.
    """

    assert len(coco_json["images"]) == 1
    coco_json = deepcopy(coco_json)

    with rasterio.open(source_image_path) as image:
        transform = image.transform
        source_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

    with rasterio.open(target_image_path) as image:
        transform = image.transform
        target_width = image.width
        target_height = image.height

        target_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

    annotations = []
    for annotation in coco_json["annotations"]:
        bounding_box = np.array(annotation["bbox"])
        bounding_box[[0, 2]] = bounding_box[[0, 2]] * source_pixel_size[0] / target_pixel_size[0]
        bounding_box[[1, 3]] = bounding_box[[1, 3]] * source_pixel_size[1] / target_pixel_size[1]
        annotation["bbox"] = bounding_box.astype(int).tolist()
        annotation["segmentation"] = coco_bbox_to_polygon(annotation["bbox"])
        annotations.append(annotation)
    if Path(coco_json["images"][0]["file_name"]).resolve() == Path(source_image_path).resolve():
        coco_json["images"][0]["file_name"] = str(target_image_path)
    coco_json["annotations"] = annotations
    coco_json["images"][0]["width"] = target_width
    coco_json["images"][0]["height"] = target_height

    return coco_json


def rescale_images(config: ImageRescalingConfig):
    """
    Rescale a set of images and labels to given target resolutions.
    """

    if len(config.output_folders) != len(config.target_resolutions):
        raise ValueError("The number of output folders and target resolutions must be the same.")

    for output_folder, target_resolution in zip(config.output_folders, config.target_resolutions):
        if isinstance(config.input_images, str):
            input_folder = Path(config.input_images)
            image_files = [input_folder / file for file in os.listdir(input_folder) if file.endswith(".tif")]
        else:
            image_files = [Path(file) for file in config.input_images]

        image_output_folder = Path(output_folder)
        image_output_folder.mkdir(exist_ok=True, parents=True)
        label_output_folder = Path(output_folder.replace("images", "labels"))
        label_output_folder.mkdir(exist_ok=True, parents=True)

        for original_image_path in image_files:
            target_image_path = image_output_folder / original_image_path.name

            with rasterio.open(original_image_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, src.crs, src.width, src.height, *src.bounds, resolution=target_resolution
                )

                input_pixel_size = np.abs(np.array([src.transform[0], src.transform[4]], dtype=np.float64))
                target_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

                if width > src.width or height > src.height:
                    raise ValueError(
                        f"Target resolution is higher than input resolution for {original_image_path.name} "
                        + f"({np.round(input_pixel_size, 3)} m vs. {np.round(target_pixel_size, 3)} m)."
                    )

                kwargs = src.meta.copy()
                kwargs.update({"transform": transform, "width": width, "height": height})

                with rasterio.open(target_image_path, "w", **kwargs) as dst:
                    for i in range(1, src.count + 1):  # reproject each channel
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=src.crs,
                            resampling=Resampling.bilinear,
                        )

            for label_subfolder in config.label_subfolders:
                label_file_name = f"{original_image_path.stem}_coco.json"
                label_file = Path(config.input_label_folder) / label_subfolder / label_file_name

                if not label_file.exists():
                    continue

                target_label_path = label_output_folder / label_subfolder / label_file_name
                target_label_path.parent.mkdir(exist_ok=True, parents=True)

                with open(label_file, "r", encoding="utf-8") as f:
                    coco_json = json.load(f)

                coco_json = rescale_coco_json_labels(coco_json, original_image_path, target_image_path)

                with open(target_label_path, "w", encoding="utf-8") as f:
                    json.dump(coco_json, f, indent=4)
