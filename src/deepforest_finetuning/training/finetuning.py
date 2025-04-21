"""Fine-tuning of the DeepForest model."""

__all__ = ["split_images_into_patches", "finetuning"]

import copy
import os
from pathlib import Path
import shutil
import uuid

from deepforest import utilities, preprocess
from deepforest import main as deepforest_main
import pandas as pd
import numpy as np
import torch

from deepforest_finetuning.config import TrainingConfig
from deepforest_finetuning.evaluation import evaluate
from deepforest_finetuning.prediction import start_prediction


def split_images_into_patches(
    annotations: pd.DataFrame, image_folder: str, output_dir: str, patch_size: int, patch_overlap: float = 0.05
) -> str:
    """
    Splits large images into smaller overlapping patches and updates the annotations accordingly.

    Args:
        annotations: A DataFrame containing bounding box annotations for the original images.
        image_folder: Path to the folder containing the original images.
        output_dir: Directory where the image patches and updated annotations will be saved.
        patch_size: The width and height (in pixels) the image patches to create.
        patch_overlap: Fraction of overlap between adjacent patches. Defaults to :code:`0.05`.

    Returns:
        Path to the CSV file containing the updated annotations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    processed_annotations = []

    for image_file in annotations["image_path"].unique():
        image_annotations = annotations.loc[annotations["image_path"] == image_file].copy()
        image_annotations["label"] = "Tree"

        _ = preprocess.split_raster(
            path_to_raster=Path(image_folder) / image_file,
            annotations_file=image_annotations,
            root_dir=image_folder,
            save_dir=output_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
        label_file_path = str((Path(output_dir) / image_file).with_suffix(".csv"))
        processed_annotations.append(utilities.read_file(label_file_path, label="Tree"))

    annotations_path = Path(output_dir) / "labels.csv"
    processed_annotations_df = pd.concat(processed_annotations)
    processed_annotations_df.to_csv(annotations_path)

    return annotations_path


def finetuning(config: TrainingConfig):
    """Fine-tunes the DeepForest model."""

    tmp_dir = Path(config.tmp_dir) / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    preprocessed_image_folders = {}
    preprocessed_annotation_files = {}

    splitting_configs = [("train", config.train_annotation_files)]
    if config.pretrain_annotation_files is not None and len(config.pretrain_annotation_files) > 0:
        splitting_configs.append("pretraining", config.pretrain_annotation_files)

    for prefix, annotation_files in splitting_configs:
        annotations = []
        for file_path in annotation_files:
            annotations.append(utilities.read_coco(file_path))
        csv_path = tmp_dir / f"{prefix}_labels.csv"
        annotations_df = pd.concat(annotations)
        annotations_df["label"] = "Tree"
        annotations_df.to_csv(csv_path, index=False)

        preprocessed_image_folders[prefix] = tmp_dir / "pretraining"

        preprocessed_annotation_files[prefix] = split_images_into_patches(
            annotations_df,
            config.image_folder,
            preprocessed_image_folders[prefix],
            patch_size=config.patch_size,
            patch_overlap=config.patch_overlap,
        )

    # load model
    model = deepforest_main.deepforest()
    model.use_release()

    print("\nStarting training ...")

    for num_epochs in config.epochs:
        for seed in config.seeds:
            print(f"INFO: Training for {num_epochs} epochs with seed {seed}...")

            # copy config to avoid overwriting
            current_config = copy.deepcopy(config)

            # set seeds for reproducibility
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))

            # configure model
            current_model = copy.deepcopy(model)
            if current_config.pretrain_learning_rate is None:
                current_model.config["train"]["lr"] = current_config.learning_rate
            else:
                current_model.config["train"]["lr"] = current_config.pretrain_learning_rate

            current_model.config["train"]["epochs"] = num_epochs
            current_model.config["save-snapshot"] = False

            if "pretrain" in annotation_files:
                current_model.config["train"]["csv_file"] = preprocessed_annotation_files["pretrain"]
                current_model.config["train"]["root_dir"] = preprocessed_image_folders["pretrain"]
                current_model.create_trainer(precision=16 if torch.cuda.is_available() else 32, log_every_n_steps=1)
                current_model.trainer.fit(current_model)

            current_model.config["train"]["lr"] = current_config.learning_rate
            current_model.config["train"]["csv_file"] = preprocessed_annotation_files["train"]
            current_model.config["train"]["root_dir"] = preprocessed_image_folders["train"]
            current_model.create_trainer(precision=16 if torch.cuda.is_available() else 32, log_every_n_steps=1)
            current_model.trainer.fit(current_model)

            if config.checkpoint_dir is not None:
                current_model.trainer.save_checkpoint(
                    Path(config.checkpoint_dir) / f"{current_config.num_epochs}_epochs_seed_{seed}.pl"
                )

            # evaluate on training and test set
            for prefix, annotation_files in [
                ("train", config.train_annotation_files),
                ("test", config.test_annotation_files),
            ]:
                image_files = []
                annotations = []
                for file_path in annotation_files:
                    current_annotations = utilities.read_file(file_path, label="Tree")
                    annotations.append(current_annotations)

                    image_files.extend(
                        [
                            Path(config.image_folder) / img_file
                            for img_file in current_annotations["image_path"].unique()
                        ]
                    )
                image_files = np.unique(image_files)

                export_config = copy.deepcopy(config.prediction_export)
                export_config.output_folder = Path(export_config.output_folder) / f"{num_epochs}_epochs"
                export_config.output_file_name = f"{prefix}_predictions_seed_{seed}.csv"

                start_prediction(
                    current_model,
                    image_files=image_files,
                    predict_tile=True,
                    patch_size=config.patch_size,
                    patch_overlap=config.patch_overlap,
                    export_config=export_config,
                )

                prediction = utilities.read_file(
                    str(Path(export_config.output_folder) / f"{prefix}_predictions_seed_{seed}.csv"), label="Tree"
                )

                metrics_file = (
                    Path(config.prediction_export.output_folder)
                    / f"{num_epochs}_epochs"
                    / f"{prefix}_metrics_seed_{seed}.csv"
                )
                evaluate(
                    prediction,
                    pd.concat(annotations),
                    config.iou_threshold,
                    metrics_file,
                )

    shutil.rmtree(tmp_dir)
