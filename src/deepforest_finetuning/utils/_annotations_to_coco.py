"""Conversion of DeepForest annotations into COCO format."""

__all__ = ["annotations_to_coco"]

from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from ._coco_bbox_to_polygon import coco_bbox_to_polygon


def annotations_to_coco(
    annotations: pd.DataFrame, image_width: int, image_height: int, capture_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Converts DeepForest annotations into COCO format.

    Args:
        annotations: A DataFrame containing bounding box annotations.
            Expected columns include:
                - 'xmin': The x-coordinate of the top-left corner of the bounding box.
                - 'ymin': The y-coordinate of the top-left corner of the bounding box.
                - 'xmax': The x-coordinate of the bottom-right corner of the bounding box.
                - 'ymax': The y-coordinate of the bottom-right corner of the bounding box.
                - 'label' or 'class': (Optional) Class label for each annotation.
        image_width: Width of the input image in pixels.
        image_height: Height of the input image in pixels.
        capture_date: Date when the image was captured in :code:`YYYY-MM-DD` format. Defaults to :code:`None`.

    Returns:
        A dictionary containing the labels in COCO format.
    """

    if capture_date is None:
        capture_date = ""

    if "label" in annotations:
        category_to_id = {}
        for idx, category in enumerate(annotations["label"].unique()):
            category_to_id[category] = idx
    else:
        category_to_id = {"Tree": 0}

    assert len(annotations["image_path"].unique()) == 1
    image_path = annotations["image_path"].iloc[0]

    coco_annotations = []
    next_id = 0
    for _, annotation in annotations.iterrows():
        bounding_box = [
            annotation["xmin"],
            annotation["ymin"],
            annotation["xmax"] - annotation["xmin"],
            annotation["ymax"] - annotation["ymin"],
        ]
        if "label" in annotation:
            category = annotation["label"]
        else:
            category = "Tree"

        coco_annotations.append(
            {
                "id": next_id,
                "image_id": 0,
                "category_id": category_to_id[category],
                "segmentation": coco_bbox_to_polygon(bounding_box),
                "bbox": bounding_box,
                "iscrowd": 0,
            }
        )
        next_id += 1

    coco_json = {
        "info": {"year": "2024", "version": "1.0.0", "date_created": datetime.today().strftime("%Y-%m-%d")},
        "licenses": [{"id": 0, "name": "Attribution License", "url": "https://creativecommons.org/licenses/by/4.0/"}],
        "images": [
            {
                "id": 0,
                "width": image_width,
                "height": image_height,
                "file_name": image_path,
                "date_captured": capture_date,
            }
        ],
        "annotations": coco_annotations,
        "categories": [
            {"id": idx, "name": category, "supercategory": category} for category, idx in category_to_id.items()
        ],
    }

    return coco_json
