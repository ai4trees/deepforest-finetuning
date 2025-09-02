"""Fine-tuning of the DeepForest model."""

__all__ = ["split_images_into_patches", "finetuning"]

import copy
from functools import partial
import os
from pathlib import Path
import shutil
from typing import List, Optional, Union
import uuid

import albumentations as A
from albumentations.pytorch import ToTensorV2
from deepforest import utilities, preprocess
from deepforest import main as deepforest_main
from pytorch_lightning import seed_everything, Trainer, LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.seed import isolate_rng
import numpy as np
import pandas as pd
import torch

from deepforest_finetuning.config import TrainingConfig
from deepforest_finetuning.evaluation import evaluate
from deepforest_finetuning.prediction import prediction as run_prediction


def get_transform(seed: Optional[int] = None):
    """
    Albumentations transformation of bounding boxes.

    Returns:
        Transforms.
    """

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5), ToTensorV2()],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        seed=seed,
    )

    return transform


def split_images_into_patches(
    annotations: pd.DataFrame,
    image_folder: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int,
    patch_overlap: float = 0.05,
) -> Path:
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

    if isinstance(image_folder, str):
        image_folder = Path(image_folder)

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    for image_file in annotations["image_path"].unique():
        image_annotations = annotations.loc[annotations["image_path"] == image_file].copy()
        image_annotations["label"] = "Tree"

        _ = preprocess.split_raster(
            path_to_raster=image_folder / image_file,
            annotations_file=image_annotations,
            root_dir=image_folder,
            save_dir=output_dir,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
        )
        label_file_path = str((output_dir / image_file).with_suffix(".csv"))
        processed_annotations.append(utilities.read_file(label_file_path, label="Tree"))

    annotations_path = output_dir / "labels.csv"
    processed_annotations_df = pd.concat(processed_annotations, ignore_index=True)
    processed_annotations_df.to_csv(annotations_path)

    return annotations_path


def _collect_annotation_paths(base_dir: Path, annotation_files: Union[List[str], str]) -> List[str]:
    """
    Process annotation file paths, handling both individual files and directories.

    Args:
        base_dir: Base directory for relative paths.
        annotation_files: List of annotation file paths or directories containing annotation files.

    Returns:
        List of processed annotation file paths.
    """
    annotation_paths = []

    if isinstance(annotation_files, str):
        annotation_files = [annotation_files]

    for file_path in annotation_files:
        path = base_dir / file_path
        if path.is_dir():
            json_files = list(path.glob("*.json"))
            relative_paths = [str(json_file.relative_to(base_dir)) for json_file in json_files]
            annotation_paths.extend(relative_paths)
            print(f"INFO: Found {len(json_files)} JSON files in directory {path}.")
        else:
            annotation_paths.append(file_path)
            print(f"INFO: Using annotation file {file_path}.")

    return annotation_paths


class EvaluationCallBack(Callback):
    """
    Callback that evaluates the model after each training epoch.

    Args:
        config: Training config.
        seed: Random seed.
    """

    def __init__(self, config: TrainingConfig, seed: int):
        super().__init__()
        self._config = config
        self._base_dir = Path(config.base_dir)
        self._seed = seed

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):  # pylint: disable=too-many-locals
        """
        Hook that evaluates the model after each training epoch.

        Args:
            trainer: PyTorch lightning trainer.
            pl_module: Model to be evaluated.
        """

        # Create a copy of the model for evaluation to avoid affecting the random state of the training
        eval_model = copy.deepcopy(pl_module)
        eval_model.trainer = copy.deepcopy(trainer)

        # evaluate on training and test set
        train_annotation_files = _collect_annotation_paths(self._base_dir, self._config.train_annotation_files)
        test_annotation_files = _collect_annotation_paths(self._base_dir, self._config.test_annotation_files)

        for prefix, annotation_files in [
            ("train", train_annotation_files),
            ("test", test_annotation_files),
        ]:
            image_files = []
            annotations = []
            for file_path in annotation_files:
                current_annotations = utilities.read_file(str(self._base_dir / file_path), label="Tree")
                annotations.append(current_annotations)

                image_files.extend(
                    [
                        self._base_dir / self._config.image_folder / img_file
                        for img_file in current_annotations["image_path"].unique()
                    ]
                )
            image_files = np.unique(image_files)

            export_config = copy.deepcopy(self._config.prediction_export)
            export_config.output_folder = str(
                self._base_dir / export_config.output_folder / f"{trainer.current_epoch + 1}_epochs"
            )
            export_config.output_file_name = f"{prefix}_predictions_seed_{self._seed}.csv"

            run_prediction(
                eval_model,
                image_files=image_files,
                predict_tile=True,
                patch_size=self._config.patch_size,
                patch_overlap=self._config.patch_overlap,
                export_config=export_config,
            )

            prediction = utilities.read_file(
                str(self._base_dir / export_config.output_folder / f"{prefix}_predictions_seed_{self._seed}.csv"),
                label="Tree",
            )

            metrics_file = (
                self._base_dir
                / self._config.prediction_export.output_folder
                / f"{trainer.current_epoch + 1}_epochs"
                / f"{prefix}_metrics_seed_{self._seed}.csv"
            )
            metrics = evaluate(
                prediction,
                pd.concat(annotations),
                self._config.iou_threshold,
                metrics_file,
            )

            # Log metrics to Lightning logger for each prefix (train/test)
            # This makes them available for callbacks like EarlyStopping
            for metric_name, metric_value in metrics.items():
                if trainer.logger is not None:
                    trainer.logger.log_metrics(
                        {f"{prefix}_{metric_name}": metric_value},
                        step=trainer.current_epoch,
                    )

                if metric_name == "f1":
                    # Access callback_metrics directly on the original trainer
                    trainer.callback_metrics[f"{prefix}_{metric_name}"] = torch.tensor(metric_value)


def finetuning(
    config: TrainingConfig,
):  # pylint: disable=too-many-locals, too-many-statements
    """Fine-tunes the DeepForest model."""

    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    tmp_dir = Path(config.tmp_dir) / str(uuid.uuid4())
    tmp_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(config.base_dir)

    preprocessed_image_folders = {}
    preprocessed_annotation_files = {}

    train_annotation_files = _collect_annotation_paths(base_dir, config.train_annotation_files)

    splitting_configs = [("train", train_annotation_files)]
    if config.pretrain_annotation_files is not None and len(config.pretrain_annotation_files) > 0:
        pretrain_annotation_files = _collect_annotation_paths(base_dir, config.pretrain_annotation_files)
        splitting_configs.append(("pretraining", pretrain_annotation_files))

    for prefix, annotation_files in splitting_configs:
        annotations = []
        for file_path in annotation_files:
            annotations.append(utilities.read_coco(base_dir / file_path))
        csv_path = tmp_dir / f"{prefix}_labels.csv"
        annotations_df = pd.concat(annotations)
        annotations_df["label"] = "Tree"
        annotations_df.to_csv(csv_path, index=False)

        preprocessed_image_folders[prefix] = tmp_dir / prefix

        preprocessed_annotation_files[prefix] = split_images_into_patches(
            annotations_df,
            base_dir / config.image_folder,
            preprocessed_image_folders[prefix],
            patch_size=config.patch_size,
            patch_overlap=config.patch_overlap,
        )

    print("\nStarting training ...")

    if config.early_stopping_patience is not None:
        print(f"Early stopping enabled with patience of {config.early_stopping_patience} epochs")

    for seed in config.seeds:
        # set seeds for reproducibility
        seed_everything(seed, workers=True, verbose=True)

        with isolate_rng(include_cuda=True):
            print(f"INFO: Training for {config.epochs} epochs with seed {seed}...")

            # load model
            model = deepforest_main.deepforest(transforms=get_transform(seed=seed))
            model.use_release()

            # copy config to avoid overwriting
            current_config = copy.deepcopy(config)

            # configure model
            if current_config.pretrain_learning_rate is None:
                model.config.train.lr = current_config.learning_rate
            else:
                model.config.train.lr = current_config.pretrain_learning_rate

            model.config.train.epochs = config.epochs

            if "pretraining" in preprocessed_annotation_files:
                model.config["train"]["csv_file"] = preprocessed_annotation_files["pretraining"]
                model.config["train"]["root_dir"] = preprocessed_image_folders["pretraining"]
                logger = CSVLogger(
                    config.log_dir,
                    name=f"{config.epochs}_epochs_seed_{seed}_pretraining",
                )

                pretraining_callbacks = []
                if config.early_stopping_patience is not None:
                    pretraining_callbacks.append(
                        EarlyStopping(
                            monitor=config.target_metric,
                            min_delta=0.0,
                            patience=config.early_stopping_patience,
                            verbose=True,
                            mode="min" if "loss" in config.target_metric else "max",
                        )
                    )

                model.create_trainer(
                    precision=config.precision if torch.cuda.is_available() else 32,
                    log_every_n_steps=1,
                    benchmark=False,
                    deterministic=True,
                    logger=logger,
                    callbacks=pretraining_callbacks if pretraining_callbacks else None,
                )
                model.trainer.fit(model)

            model.config["train"]["lr"] = current_config.learning_rate
            model.config["train"]["csv_file"] = preprocessed_annotation_files["train"]
            model.config["train"]["root_dir"] = preprocessed_image_folders["train"]
            logger = CSVLogger(config.log_dir, name=f"{config.epochs}_epochs_seed_{seed}")

            callbacks: List[Callback] = [EvaluationCallBack(config, seed)]
            if config.checkpoint_dir is not None:
                callbacks.append(
                    ModelCheckpoint(
                        dirpath=base_dir / config.checkpoint_dir,
                        filename="{epoch}_" + f"seed={seed}",
                        monitor=config.target_metric,
                        save_top_k=config.save_top_k,
                        every_n_epochs=1,
                        enable_version_counter=False,
                        mode="min" if "loss" in config.target_metric else "max",
                    )
                )

            if config.early_stopping_patience is not None:
                callbacks.append(
                    EarlyStopping(
                        monitor=config.target_metric,
                        min_delta=0.0,
                        patience=config.early_stopping_patience,
                        verbose=True,
                        mode="min" if "loss" in config.target_metric else "max",
                    )
                )

            model.create_trainer(
                precision=config.precision if torch.cuda.is_available() else 32,
                log_every_n_steps=1,
                benchmark=False,
                deterministic=True,
                logger=logger,
                callbacks=callbacks,
            )
            model.trainer.fit(model)

    shutil.rmtree(tmp_dir)
