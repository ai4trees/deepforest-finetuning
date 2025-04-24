"""Config definitions."""

__all__ = [
    "EvaluationConfig",
    "ExportConfig",
    "ImageRescalingConfig",
    "LabelFilteringConfig",
    "ManuallyCorrectedLabelPreprocessingConfig",
    "PointCloudLabelProjectionConfig",
    "PredictionConfig",
    "TrainingConfig",
]

import dataclasses
from typing import List, Optional, Union


@dataclasses.dataclass
class PointCloudLabelProjectionConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for projection of point cloud labels to orthophotos."""

    base_dir: str
    point_cloud_paths: List[str]
    image_paths: List[str]
    label_json_output_paths: List[str]
    label_image_output_paths: Optional[List[str]] = None
    grid_resolution: float = 0.2
    min_tree_height: float = 10
    min_bounding_box_width: float = 2
    dtm_classification_threshold: float = 0.5
    dtm_resolution: float = 1.0
    dtm_rigidness: int = 3


@dataclasses.dataclass
class ImageRescalingConfig:
    """Configuration for image rescaling."""

    base_dir: str
    input_images: Union[List[str], str]
    target_resolutions: List[float]
    input_label_folders: List[str]
    output_folders: List[str]


@dataclasses.dataclass
class ManuallyCorrectedLabelPreprocessingConfig:
    """Configuration for preprocessing of manually corrected labels."""

    base_dir: str
    input_label_folder: str
    output_label_folder: str
    input_image_folder: str


@dataclasses.dataclass
class LabelFilteringConfig:
    """Configuration for label preprocessing."""

    base_dir: str
    input_label_folder: str
    output_label_folder: str
    iou_threshold: float = 0.5


@dataclasses.dataclass
class ExportConfig:
    """Configuration for label export."""

    output_folder: str
    sort_by: Optional[str] = None
    column_order: Optional[List[str]] = None
    index_as_label_suffix: bool = False
    output_file_name: str = "predictions.csv"


@dataclasses.dataclass
class TrainingConfig:  # pylint: disable=too-many-instance-attributes
    """Training configuration."""

    base_dir: str
    image_folder: str
    patch_size: int
    patch_overlap: float
    learning_rate: float
    tmp_dir: str
    train_annotation_files: List[str]
    test_annotation_files: List[str]
    prediction_export: ExportConfig
    checkpoint_dir: Optional[str] = None
    iou_threshold: float = 0.5
    pretrain_annotation_files: Optional[List[str]] = None
    pretrain_learning_rate: Optional[float] = None
    epochs: int = 20
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4])
    precision: str = "16-mixed"
    float32_matmul_precision: str = "medium"
    log_dir: str = "./logs"


@dataclasses.dataclass
class PredictionConfig:
    """
    Prediction configuration.
    """

    image_files: List[str]
    predict_tile: bool
    prediction_export: ExportConfig
    checkpoint_path: Optional[str] = None
    patch_size: Optional[int] = None
    patch_overlap: Optional[float] = None


@dataclasses.dataclass
class EvaluationConfig:
    """
    Evaluation configuration.
    """

    prediction_file: str
    label_file: str
    iou_threshold: float
    output_file: str
