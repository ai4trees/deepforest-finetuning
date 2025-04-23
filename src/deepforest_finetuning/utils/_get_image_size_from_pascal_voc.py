"""Extracts the image width and height from a PASCAL VOC XML annotation file."""

__all__ = ["get_image_size_from_pascal_voc"]

from pathlib import Path
from typing import Union, Tuple
import xml.etree.ElementTree as ET


def get_image_size_from_pascal_voc(xml_file_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Extracts the image width and height from a PASCAL VOC XML annotation file.

    Args:
        xml_file_path: Path to the PASCAL VOC XML annotation file.

    Returns:
        A tuple containing:
            - width (int): The width of the image.
            - height (int): The height of the image.

    Raises:
        ValueError: If the <size> tag is not found in the XML file.
    """

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    size = root.find("size")
    if size is not None:
        width = int(size.find("width").text)
        height = int(size.find("height").text)
        return width, height
    else:
        raise ValueError("No <size> tag found in the XML.")
