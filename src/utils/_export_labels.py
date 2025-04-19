"""Label export."""

__all__ = ["export_labels"]

from typing import List, Optional

from pathlib import Path
import pandas as pd


def export_labels(
    pred_df: pd.DataFrame,
    export_path: str,
    column_order: Optional[List[str]] = None,
    index_as_label_suffix: bool = False,
    sort_by: Optional[str] = None,
) -> None:
    """
    Exports prediction results to a CSV file.

    This function saves the given DataFrame to a CSV file, with optional column reordering, label suffixing using the
    index, and row sorting.

    Args:
        pred_df: The DataFrame containing prediction results to export.
        export_path: The file path where the CSV will be saved.
        column_order: Optional list of column names to define the column order in the output CSV.
        index_as_label_suffix: If :code:`True`, appends the DataFrame index to the :code:`"label"` column as a suffix.
        sort_by: Name of a column by which to sort the DataFrame before exporting.
    """

    if sort_by is not None:
        pred_df.sort_values(by=sort_by, inplace=True)
        pred_df = pred_df.reset_index(drop=True)

    if index_as_label_suffix:
        pred_df["label"] = pred_df["label"] + pred_df.index.astype(str)

    if column_order is not None:
        pred_df = pred_df[column_order]

    Path(export_path).parent.mkdir(exist_ok=True, parents=False)
    pred_df.to_csv(export_path, index=False)
    print(f">>> Exported predictions to {export_path}.")
