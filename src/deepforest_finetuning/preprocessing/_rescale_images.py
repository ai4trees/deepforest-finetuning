"""Image rescaling."""

__all__ = ["rescale_images"]

import os
from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from deepforest_finetuning.config import ImageRescalingConfig
from deepforest_finetuning.utils import rescale_coco_json


def rescale_images(config: ImageRescalingConfig):  # pylint: disable=too-many-locals
    """
    Rescale a set of images and labels to given target resolutions.
    """

    if len(config.output_folders) != len(config.target_resolutions):
        raise ValueError("The number of output folders and target resolutions must be the same.")

    base_dir = Path(config.base_dir)

    for output_folder, target_resolution in zip(config.output_folders, config.target_resolutions):
        if isinstance(config.input_images, str):
            input_folder = base_dir / config.input_images
            image_files = [input_folder / file for file in os.listdir(input_folder) if file.endswith(".tif")]
        else:
            image_files = [base_dir / file for file in config.input_images]

        image_output_folder = base_dir / output_folder
        image_output_folder.mkdir(exist_ok=True, parents=True)

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

            for folder in config.input_label_folders:
                input_label_folder = base_dir / folder

                label_output_folder = base_dir / output_folder.replace("images", Path(folder).stem)
                label_output_folder.mkdir(exist_ok=True, parents=True)

                label_subfolders = [x for x in os.listdir(input_label_folder) if os.path.isdir(input_label_folder / x)]
                for label_subfolder in label_subfolders:
                    label_file_name = f"{original_image_path.stem}_coco.json"
                    label_file = input_label_folder / label_subfolder / label_file_name

                    if not label_file.exists():
                        continue

                    target_label_path = label_output_folder / label_subfolder / label_file_name
                    target_label_path.parent.mkdir(exist_ok=True, parents=True)

                    with open(label_file, "r", encoding="utf-8") as f:
                        coco_json = json.load(f)

                    coco_json = rescale_coco_json(coco_json, target_image_path, source_image_path=original_image_path)

                    with open(target_label_path, "w", encoding="utf-8") as f:
                        json.dump(coco_json, f, indent=4)
