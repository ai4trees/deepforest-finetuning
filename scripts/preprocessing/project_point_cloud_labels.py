"""Projects point cloud tree instance segmentation labels to an orthophoto."""

from deepforest_finetuning.config import PointCloudLabelProjectionConfig
from deepforest_finetuning.utils import load_config


if __name__ == "__main__":
    config = load_config(PointCloudLabelProjectionConfig)
