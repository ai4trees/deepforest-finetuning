"""Loading of configuration files."""

__all__ = ["load_config"]

from argparse import ArgumentParser
from typing import Any, Type

import tomllib
from dacite import from_dict


def load_config(config_type: Type) -> Any:
    """
    Creates a command-line argument parser that consumes the path to a configuration file, parses the specified
    configuration file based on the given configuration type and returns the parsed configuration.

    Args:
        config_type: The class representing the configuration to load.

    Returns:
        An instance of the specified configuration type populated with default or loaded values.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-path",
        dest="config",
        action="store",
        help="Path to config file",
    )
    args = parser.parse_args()

    if not args.config:
        raise ValueError("No config file provided. Please provide a relative path to a config file.")

    with open(args.config, "rb") as f:
        config = from_dict(data_class=config_type, data=tomllib.load(f))

    return config
