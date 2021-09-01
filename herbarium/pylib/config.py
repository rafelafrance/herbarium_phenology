"""Utilities for working with configurations."""

import sys
from pathlib import Path
from typing import Optional

import yaml

from herbarium.pylib import const


class Configs:
    """Handle configurations."""

    default_path = const.ROOT_DIR / 'herbarium_phenology.yaml'

    def __init__(self, path: Optional[Path] = None):
        path = path if path else self.default_path
        self.configs = self.read_file(path)
        self.script = Path(sys.argv[0]).stem

    @staticmethod
    def read_file(path):
        """Read data from config file."""
        with open(path) as cfg_file:
            configs = yaml.safe_load(cfg_file)
        return configs

    def script_defaults(self):
        """Get argument defaults and constants for the current script."""
        return self.configs[self.script]
