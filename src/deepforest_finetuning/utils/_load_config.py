"""Loading of configuration files."""

__all__ = ["load_config"]

from typing import Any, Type

import tomllib
from dacite import from_dict


def load_config(config_path: str, config_type: Type) -> Any:
    """
    Parses the specified configuration file based on the given configuration type and returns the parsed configuration.

    Args:
        config_path: Path of the configuration file.
        config_type: The class representing the configuration to load.

    Returns:
        An instance of the specified configuration type populated with default or loaded values.
    """

    with open(config_path, "rb") as f:
        config = from_dict(data_class=config_type, data=tomllib.load(f))

    return config
