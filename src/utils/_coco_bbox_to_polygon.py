__all__ = ["coco_bbox_to_polygon"]

from typing import List

import numpy as np


def coco_bbox_to_polygon(bounding_box: List[int]) -> List[int]:
    """
    Converts a COCO-format bounding box to a COCO polygon.

    The COCO bounding box format is [x, y, width, height], where (x, y)  represents the top-left corner of the box. This
    function converts it into a polygon format represented by a list of the corner points  in the following order:
    top-left, top-right, bottom-right, bottom-left, top-left.

    Args:
        bounding_box: A list containing the bounding box in COCO format.

    Returns:
        A list representing the polygon in COCO format.
    """

    # fmt: off
    return np.array(
        [
            [
                bounding_box[0], bounding_box[1],
                bounding_box[0] + bounding_box[2], bounding_box[1],
                bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3],
                bounding_box[0], bounding_box[1] + bounding_box[3],
                bounding_box[0], bounding_box[1],
            ]
        ]
    ).tolist()
    # fmt: on
