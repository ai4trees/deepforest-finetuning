"""Script to project point cloud tree instance segmentation labels to an orthophoto."""

from deepforest_finetuning.config import PointCloudLabelProjectionConfig
from deepforest_finetuning.preprocessing import project_point_cloud_labels
from deepforest_finetuning.utils import load_config


if __name__ == "__main__":
    config = load_config(PointCloudLabelProjectionConfig)
    project_point_cloud_labels(config)
