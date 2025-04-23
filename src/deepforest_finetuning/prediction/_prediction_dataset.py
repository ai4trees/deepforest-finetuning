"""Prediction dataset."""

__all__ = ["PredictionDataset"]

from pathlib import Path
from typing import List, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image
from tifffile import imread
from torch.utils.data import Dataset


class PredictionDataset(Dataset):
    """
    Prediction dataset.

    Args:
        image_files: Paths of the image files to be predicted.
        resize_images_to: Target image width (in pixel) to which the images should be rescaled. Defaults to
            :code:`None`, which means that the images are not rescaled.
    """

    def __init__(self, image_files: List[str], resize_images_to: Optional[int] = None):
        self.image_files = image_files
        self.resize_images_to = resize_images_to

    def __len__(self) -> int:
        """
        Return: Length of the dataset.
        """
        return len(self.image_files)

    def __getitem__(self, idx: int) -> npt.NDArray:
        """
        Loads the image at the given index.

        Args:
            idx: Index of the image file.

        Returns: Image data.
        """
        img_path = self.image_files[idx]
        image_array = np.array(imread(img_path))[:, :, :3].astype(np.uint8)

        if self.resize_images_to is not None:
            image = Image.fromarray(image_array)
            image = image.resize((self.resize_images_to, self.resize_images_to))
            image_array = np.array(image)

        return image_array

    def __getname__(self, idx: int) -> str:
        """
        Retrieves the name of the image file at the given index.

        Args:
            idx: Index of the image file.

        Returns: Image name.
        """
        return Path(self.image_files[idx]).name
