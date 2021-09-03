"""Utilities for working with configurations."""

import sys
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path
from typing import Optional

from . import const, util


class Config:
    """Handle configurations."""

    default_path = const.ROOT_DIR / 'herbarium_phenology.cfg'

    def __init__(self, path: Optional[Path] = None, module: str = "") -> None:
        path = path if path else self.default_path
        self.configs = self.read_file(path)
        self.module = module if module else Path(sys.argv[0]).stem

    @staticmethod
    def read_file(path):
        """Read data from config file."""
        configs = ConfigParser(interpolation=ExtendedInterpolation())

        with open(path) as cfg_file:
            configs.read_file(cfg_file)

        return configs

    def module_defaults(self):
        """Get argument module_defaults."""
        return util.Struct(**self.configs[self.module])

    def default_list(self, key, section=''):
        """Make a list from a multi-line configuration."""
        section = section if section else self.module
        return self.configs[section][key].splitlines()

    @staticmethod
    def reroot(path):
        """Adjust the root directory for a path when using in a notebook."""
        return const.ROOT_DIR / path
