__all__ = ["annotations_to_coco"]

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
import rasterio

from ._coco_bbox_to_polygon import coco_bbox_to_polygon


def annotations_to_coco(annotations: pd.DataFrame, image_path: Union[str, Path]) -> Dict[str, Any]:
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
        image_path: Path to the image file associated with the annotations. Used to extract image metadata such as width
            and height.

    Returns:
        A dictionary containing the labels in COCO format.
    """

    if not isinstance(image_path, Path):
        image_path = Path(image_path)

    # in the image files used in our dataset, the capture date is encoded in the file name
    if len(image_path.stem) >= 8 and (image_path.stem[:8]).isnumeric():
        date_prefix = image_path.stem[:8]
        image_capture_date = f"{date_prefix[:4]}-{date_prefix[4:6]}-{date_prefix[6:]}"
    else:
        image_capture_date = ""

    with rasterio.open(image_path) as image:
        image_width = image.width
        image_height = image.height

    if "label" in annotations:
        categories = []
        category_to_id = {}
        for idx, category in enumerate(annotations.unique()):
            categories.append({"id": idx, "name": category, "supercategory": category})
            category_to_id[category] = idx
    else:
        categories = [{"id": 0, "name": "Tree", "supercategory": "Tree"}]
        category_to_id = {"Tree": 0}

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
                "file_name": str(image_path),
                "date_captured": image_capture_date,
            }
        ],
        "annotations": coco_annotations,
        "categories": categories,
    }

    return coco_json
