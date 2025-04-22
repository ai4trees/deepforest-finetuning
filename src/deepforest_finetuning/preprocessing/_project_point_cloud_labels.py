"""Projects point cloud tree instance segmentation labels to an orthophoto."""

__all__ = ["project_point_cloud_labels"]

from datetime import datetime
import json
from pathlib import Path

import numpy as np
from pointtorch import read
from pointtorch.operations.numpy import make_labels_consecutive
from pointtree.operations import cloth_simulation_filtering, create_digital_terrain_model, distance_to_dtm
import rasterio
from rasterio.transform import from_origin
from skimage.filters.rank import modal
import torch
from torch_scatter import scatter_max

from deepforest_finetuning.utils import coco_bbox_to_polygon
from deepforest_finetuning.config import PointCloudLabelProjectionConfig


def project_point_cloud_labels(  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
    config: PointCloudLabelProjectionConfig,
):
    """Projects point cloud tree instance segmentation labels to an orthophoto."""

    if len(config.point_cloud_paths) != len(config.image_paths):
        raise ValueError("point_cloud_paths and image_paths must have the same length.")
    if len(config.point_cloud_paths) != len(config.label_json_output_paths):
        raise ValueError("point_cloud_paths and label_json_output_paths must have the same length.")
    if config.label_image_output_paths is not None and len(config.label_json_output_paths) != len(
        config.label_image_output_paths
    ):
        raise ValueError("label_json_output_paths and label_image_output_paths must have the same length.")

    base_dir = Path(config.base_dir)

    for idx, point_cloud_path in enumerate(config.point_cloud_paths):
        image_path = base_dir / config.image_paths[idx]

        if not image_path.suffix == ".tif":
            raise ValueError("Image must be a geotif file.")

        with rasterio.open(image_path) as image:
            transform = image.transform
            image_width = image.width
            image_height = image.height
            crs = image.crs

            pixel_size = np.abs(np.array([transform[0], transform[4]], dtype=np.float64))
            image_width_meter = image_width * pixel_size[1]
            image_height_meter = image_height * pixel_size[0]

            tile_upper_left_corner = np.array([transform.c, transform.f], dtype=np.float64)
            tile_lower_left_corner = tile_upper_left_corner.copy()
            tile_lower_left_corner[1] -= image.height * pixel_size[1]

        print(
            f"Loaded geotiff {image_path.name} with CRS={crs}, width={image_width}, height={image_height}, "
            + f"pixel_size={pixel_size} m"
        )

        point_cloud = read(base_dir / point_cloud_path)

        print(f"Loaded point cloud {Path(point_cloud_path).name} with {len(point_cloud)} points.")

        point_cloud["instance_id_prediction"] = make_labels_consecutive(
            point_cloud["instance_id_prediction"].to_numpy(), ignore_id=0
        )

        if (
            point_cloud["instance_id_prediction"].min() != 0
            or point_cloud["instance_id_prediction"].max() != len(point_cloud["instance_id_prediction"].unique()) - 1
        ):
            raise ValueError("The predicted instance IDs must be continuous, starting from zero.")

        pixel_indices = np.floor((point_cloud.xyz()[:, :2] - tile_lower_left_corner) / config.grid_resolution).astype(
            np.int64
        )

        label_image_shape = np.ceil(np.array([image_height_meter, image_width_meter]) / config.grid_resolution).astype(
            np.int64
        )
        label_image = np.zeros(label_image_shape, dtype=np.int64)

        valid_mask = np.logical_and(
            (pixel_indices >= 0).all(axis=-1), (pixel_indices < np.flip(label_image_shape)).all(axis=-1)
        )

        pixel_indices = pixel_indices[valid_mask]
        valid_point_cloud = point_cloud[valid_mask]

        unique_pixel_indices, inverse_indices = np.unique(pixel_indices, axis=0, return_inverse=True)

        terrain_classification = cloth_simulation_filtering(
            valid_point_cloud.xyz(),
            classification_threshold=config.dtm_classification_threshold,
            resolution=config.dtm_resolution,
            rigidness=config.dtm_rigidness,
        )

        dtm, dtm_offset = create_digital_terrain_model(
            valid_point_cloud.xyz()[terrain_classification == 0],
            grid_resolution=config.dtm_resolution,
            k=100,
            p=2,
            voxel_size=0.1,
        )

        dist_to_dtm = distance_to_dtm(
            valid_point_cloud.xyz(),
            dtm,
            dtm_offset,
            dtm_resolution=config.dtm_resolution,
        )

        max_height, max_indices = scatter_max(
            torch.from_numpy(dist_to_dtm), torch.from_numpy(inverse_indices).long(), dim=-1
        )

        instance_ids = valid_point_cloud["instance_id_prediction"].to_numpy()[max_indices.cpu().numpy()]

        instance_ids[max_height.cpu().numpy() < config.min_tree_height] = 0

        label_image[unique_pixel_indices[:, 1], unique_pixel_indices[:, 0]] = instance_ids
        # image coordinate system starts in upper left corner and not in lower left
        label_image = np.flip(label_image, axis=0)

        label_image = modal(label_image.astype(np.uint16), footprint=np.ones((3, 3), dtype=np.uint16))

        transform = from_origin(
            west=tile_upper_left_corner[0],
            north=tile_upper_left_corner[1],
            xsize=config.grid_resolution,
            ysize=config.grid_resolution,
        )

        if config.label_image_output_paths is not None:
            label_image_output_path = base_dir / config.label_image_output_paths[idx]
            label_image_output_path.parent.mkdir(exist_ok=True, parents=True)

            with rasterio.open(
                label_image_output_path,
                "w",
                driver="GTiff",
                height=label_image.shape[0],
                width=label_image.shape[1],
                count=1,  # number of bands
                dtype="uint16",
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(label_image, 1)

        annotations = []

        next_id = 0
        for instance_id in np.unique(label_image):
            if instance_id == 0:
                continue

            instance_mask = label_image == instance_id

            non_zero_indices_x = np.flatnonzero(instance_mask.sum(axis=0))
            non_zero_indices_y = np.flatnonzero(instance_mask.sum(axis=1))

            x_min = int(non_zero_indices_x.min() * image_width / label_image.shape[1])
            y_min = int(non_zero_indices_y.min() * image_height / label_image.shape[0])
            x_max = int(non_zero_indices_x.max() * image_width / label_image.shape[1])
            y_max = int(non_zero_indices_y.max() * image_height / label_image.shape[0])
            width = x_max - x_min
            height = y_max - y_min
            width_meter = width * pixel_size[0]
            height_meter = height * pixel_size[1]

            if width_meter < config.min_bounding_box_width:
                print(f"Skipping bounding box {instance_id} with width of {np.round(width_meter, 2)} m.")
                continue
            if height_meter < config.min_bounding_box_width:
                print(f"Skipping bounding box {instance_id} with width of {np.round(height_meter, 2)} m.")
                continue

            bounding_box = [x_min, y_min, width, height]
            annotation = {"id": next_id, "image_id": 0, "category_id": 0, "iscrowd": 0, "bbox": bounding_box}
            annotation["segmentation"] = coco_bbox_to_polygon(bounding_box)
            annotations.append(annotation)
            next_id += 1

        # in the image files used in our dataset, the capture date is encoded in the file name
        if len(image_path.stem) >= 8 and (image_path.stem[:8]).isnumeric():
            date_prefix = Path(image_path).stem[:8]
            image_capture_date = f"{date_prefix[:4]}-{date_prefix[4:6]}-{date_prefix[6:]}"
        else:
            image_capture_date = ""

        coco_json = {
            "info": {"year": "2024", "version": "1.0.0", "date_created": datetime.today().strftime("%Y-%m-%d")},
            "licenses": [
                {"id": 0, "name": "Attribution License", "url": "https://creativecommons.org/licenses/by/4.0/"}
            ],
            "images": [
                {
                    "id": 0,
                    "width": image_width,
                    "height": image_height,
                    "file_name": image_path.name,
                    "date_captured": image_capture_date,
                }
            ],
            "annotations": annotations,
            "categories": [{"id": 0, "name": "Tree", "supercategory": "Tree"}],
        }

        label_json_output_path = base_dir / config.label_json_output_paths[idx]
        with open(label_json_output_path, "w", encoding="utf-8") as f:
            json.dump(coco_json, f, indent=4)
