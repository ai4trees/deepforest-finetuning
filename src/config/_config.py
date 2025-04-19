"""Config definitions."""

__all__ = [
    "PointCloudLabelProjectionConfig",
    "ImageRescalingConfig",
    "LabelPreprocessingConfig",
    "ExportConfig",
    "TrainingConfig",
    "PredictionConfig",
    "EvaluationConfig",
]

import dataclasses
from typing import List, Optional, Union


@dataclasses.dataclass
class PointCloudLabelProjectionConfig:
    point_cloud_path: str
    image_path: str
    label_json_output_path: str
    label_image_output_path: Optional[str] = None
    grid_resolution: float = 0.2
    min_tree_height: float = 10
    min_bounding_box_width: float = 2
    dtm_classification_threshold: float = 0.5
    dtm_resolution: float = 1.0
    dtm_rigidness: int = 3


@dataclasses.dataclass
class ImageRescalingConfig:
    input_images: Union[List[str], str]
    target_resolutions: List[float]
    input_label_folder: str
    label_subfolders: List[str]
    output_folders: List[str]


@dataclasses.dataclass
class LabelPreprocessingConfig:
    input_label_folder: str
    output_label_folder: str
    input_image_folder: str
    iou_threshold: float = 0.4


@dataclasses.dataclass
class ExportConfig:
    output_folder: str
    annotation_format: str = "csv"
    sort_by: Optional[str] = None
    column_order: Optional[List[str]] = None
    index_as_label_suffix: bool = False
    output_file_name: str = "predictions.csv"


@dataclasses.dataclass
class TrainingConfig:
    image_folder: str
    patch_size: int
    patch_overlap: float
    learning_rate: float
    tmp_dir: str
    train_annotation_files: List[str]
    test_annotation_files: List[str]
    prediction_export: ExportConfig
    checkpoint_dir: Optional[str] = None
    iou_threshold: float = 0.4
    pretrain_annotation_files: Optional[List[str]] = None
    pretrain_learning_rate: Optional[float] = None
    epochs: List[int] = dataclasses.field(default_factory=lambda: [10])
    seeds: list[int] = dataclasses.field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclasses.dataclass
class PredictionConfig:
    image_files: List[str]
    predict_tile: bool
    prediction_export: ExportConfig
    checkpoint_path: Optional[str] = None
    patch_size: Optional[int] = None
    patch_overlap: Optional[float] = None


@dataclasses.dataclass
class EvaluationConfig:
    prediction_file: str
    label_file: str
    iou_threshold: float
    output_file: str
