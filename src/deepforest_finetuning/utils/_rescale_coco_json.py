"""Rescaling of bounding box labels in COCO JSON format."""

__all__ = ["rescale_coco_json"]

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt
import rasterio

from ._coco_bbox_to_polygon import coco_bbox_to_polygon


def rescale_coco_json(
    coco_json: Dict[str, Any],
    target_image_path: Path,
    source_image_path: Optional[Path] = None,
    source_image_shape: Optional[npt.NDArray] = None,
) -> Dict[str, Any]:
    """
    Rescale COCO annotations to match a new image size. This function adjusts the bounding boxes and image metadata in a
    COCO-style annotation dictionary to correspond to a new target image size.

    Args:
        coco_json: A dictionary containing annotations in COCO format.
        target_image_path: Path to the resized image file.
        source_image_path: Path to the original image file used for the annotations. Defaults to :code:`None`. Either
            :code:`source_image_path` or :code:`source_image_shape` must not be :code:`None`.
        source_image_shape: Shape of the original image file (image height, image width). Defaults to :code:`None`.
            Either :code:`source_image_path` or :code:`source_image_shape` must not be :code:`None`.

    Returns:
        A new dictionary containing annotations that are rescaled to match the target image size.

    Raises:
        ValueError: If :code:`source_image_path` and :code:`source_image_shape` are :code:`None`.
    """

    if source_image_path is None and source_image_shape is None:
        raise ValueError("Either source_image_path or source_image_shape must not be None.")

    assert len(coco_json["images"]) == 1
    coco_json = deepcopy(coco_json)

    with rasterio.open(target_image_path) as image:
        transform = image.transform
        target_width = image.width
        target_height = image.height
        target_image_shape = np.array([target_height, target_width], dtype=np.int32)

        target_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))

    if source_image_path is not None:
        with rasterio.open(source_image_path) as image:
            transform = image.transform
            source_pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))
    else:
        source_pixel_size = (target_image_shape / source_image_shape) * target_pixel_size
        source_pixel_size = np.flip(source_pixel_size)

    annotations = []
    for annotation in coco_json["annotations"]:
        bounding_box = np.array(annotation["bbox"])
        bounding_box[[0, 2]] = bounding_box[[0, 2]] * source_pixel_size[0] / target_pixel_size[0]
        bounding_box[[1, 3]] = bounding_box[[1, 3]] * source_pixel_size[1] / target_pixel_size[1]
        annotation["bbox"] = bounding_box.astype(int).tolist()
        annotation["segmentation"] = coco_bbox_to_polygon(annotation["bbox"])
        annotations.append(annotation)

    coco_json["annotations"] = annotations
    coco_json["images"][0]["width"] = target_width
    coco_json["images"][0]["height"] = target_height

    return coco_json
