import os
from typing import List

from transformer.AbstractTransformer import AbstractTransformer
from utils.Registries import TRANSFORMER_REGISTRY


def get_consistent_transformers(transformer_labels: List[str]) -> List[AbstractTransformer]:
    return [TRANSFORMER_REGISTRY.get(name) for name in sorted(transformer_labels)]


def fix_psutil_disk_usage_on_windows():
    """
    Monkey-patch psutil.disk_usage to suppress SystemError (bad format char) on Windows.
    This prevents crashes in MLflow system metrics collection.
    """
    if os.name == 'nt':
        try:
            import psutil
            # Monkey-patch psutil.disk_usage to suppress SystemError (bad format char)
            _original_disk_usage = psutil.disk_usage

            class _DummyUsage:
                total = 0
                used = 0
                free = 0
                percent = 0

            def _robust_disk_usage(path):
                try:
                    return _original_disk_usage(path)
                except Exception:
                    return _DummyUsage()

            psutil.disk_usage = _robust_disk_usage
        except ImportError:
            pass