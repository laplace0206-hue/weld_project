from __future__ import annotations

import importlib
import pkgutil
import sys

from .config import ExperimentConfig, load_experiment_config


def register_legacy_import_aliases() -> None:
    package = sys.modules[__name__]
    sys.modules.setdefault("weld_project", package)

    for _, module_name, _ in pkgutil.walk_packages(package.__path__, prefix=f"{__name__}."):
        module = importlib.import_module(module_name)
        legacy_name = module_name.removeprefix("src.")
        sys.modules.setdefault(legacy_name, module)


__all__ = ["ExperimentConfig", "load_experiment_config", "register_legacy_import_aliases"]
