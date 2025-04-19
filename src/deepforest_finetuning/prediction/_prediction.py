"""Prediction with DeepForest model."""

__all__ = ["prediction"]

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from deepforest_finetuning.config import ExportConfig
from deepforest_finetuning.utils import export_labels
from ._prediction_dataset import PredictionDataset


def prediction(
    model,
    image_files: List[str],
    predict_tile: bool,
    export_config: ExportConfig,
    patch_size: Optional[int] = None,
    patch_overlap: Optional[float] = None,
) -> None:
    """
    Run object detection predictions on a list of images using the provided model.

    Args:
        model: The trained model used for making predictions.
        image_files: A list of paths to image files for prediction.
        predict_tile: Whether the input images need to be split into patches for tiled prediction. Can be set to
            :code:`False` if the input images already were split into tiles during preprocessing.
        export_config: Configuration object specifying how to export the results.
        patch_size: Size of each patch used during tiled prediction. Required if :code:`predict_tile` is True.
        patch_overlap: Fractional overlap between patches during tiled prediction. Required if :code:`predict_tile` is
            True.
    """

    if predict_tile:
        if patch_size is None:
            raise ValueError("Patch size must be specified when predict_tile is set to True.")
        if patch_overlap is None:
            raise ValueError("Patch overlap must be specified when predict_tile is set to True.")

    print("\nLoading dataset and model ...")

    Path(export_config.output_folder).mkdir(exist_ok=True, parents=True)

    tree_dataset = PredictionDataset(image_files)

    all_predictions = []

    # predict images
    print(f"\nRunning predictions for {len(tree_dataset)} image(s)...")
    for img_idx in tqdm(range(len(tree_dataset))):
        if predict_tile:
            prediction = model.predict_tile(
                image=tree_dataset[img_idx].astype(np.float32),
                return_plot=False,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
            )
        else:
            prediction = model.predict_image(
                image=tree_dataset[img_idx].astype(np.float32),
                return_plot=False,
            )
        image_name = tree_dataset.__getname__(img_idx)
        prediction["image_path"] = image_name
        all_predictions.append(prediction)

        export_labels(
            prediction,
            export_path=(Path(export_config.output_folder) / image_name).with_suffix(".csv"),
            column_order=export_config.column_order,
            index_as_label_suffix=export_config.index_as_label_suffix,
            sort_by=export_config.sort_by,
        )

    export_labels(
        pd.concat(all_predictions),
        export_path=Path(export_config.output_folder) / export_config.output_file_name,
        column_order=export_config.column_order,
        index_as_label_suffix=export_config.index_as_label_suffix,
        sort_by=export_config.sort_by,
    )

    print(f"\nPredictions exported to: {export_config.output_folder}.")
