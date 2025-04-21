"""Evaluation of model predictions."""

__all__ = ["evaluate"]

from pathlib import Path
import warnings
from typing import Union

from deepforest.evaluate import evaluate_boxes
import pandas as pd


def evaluate(
    predictions: pd.DataFrame, annotations: pd.DataFrame, iou_threshold: float, output_file: Union[str, Path]
) -> None:
    """
    Evaluates a model's predictions and stores the evaluation metrics as CSV file.

    Args:
        predictions: A DataFrame containing the predictions to be evaluated.
        annotations: A DataFrame containing the target labels.
        iou_threshold: Threshold for the IoU between predicted and target bounding boxes at which predicted bounding
            boxes are counted as true positives.
        output_file: Path of the CSV file in which to store the evaluation metrics.
    """

    # ignore deprecated warnings from pandas raised by deepforest.IoU (line 113: iou_df = pd.concat(iou_df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = evaluate_boxes(
            predictions=predictions,
            ground_df=annotations,
            iou_threshold=iou_threshold,
        )

    results["precision"] = results.pop("box_precision")
    results["recall"] = results.pop("box_recall")
    results["f1"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])

    print(f"Precision:\t{results['precision']}\nRecall:\t\t{results['recall']}\nF1:\t\t{results['f1']}")

    metrics = []
    metrics.append({"metric": "precision", "score": results["precision"]})
    metrics.append({"metric": "recall", "score": results["recall"]})
    metrics.append({"metric": "f1", "score": results["f1"]})
    df = pd.DataFrame(metrics)

    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    df.to_csv(output_file, index=False)
