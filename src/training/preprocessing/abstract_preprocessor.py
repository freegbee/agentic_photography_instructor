import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TypeVar, Generic

from training.mlflow_utils import mlflow_logging

logger = logging.getLogger(__name__)

RESULT = TypeVar('RESULT')


class AbstractPreprocessor(Generic[RESULT], ABC):
    """Basisklasse für Preprocessing-Schritte.

        Generic[RESULT] definiert den Rückgabetyp von `get_preprocessing_result`.
        Subklassen müssen `_preprocess_impl` implementieren, das die eigentliche
        Vorverarbeitung ausführt, sowie `get_preprocessing_result`, das das Ergebnis
        zurückliefert. Die Methode `preprocess` ist mit dem `mlflow_logging`
        Dekorator versehen, um die Dauer des Preprocessing-Schritts zu messen.
    """

    def __init__(self, preprocessing_step: Optional[int]) -> None:
        self.step_arg = preprocessing_step

    @abstractmethod
    def get_preprocessing_result(self) -> RESULT:
        pass

    @abstractmethod
    def _preprocess_impl(self):
        pass

    @mlflow_logging("preprocessing_duration_seconds")
    def preprocess(self) -> RESULT:
        self._preprocess_impl()
        return self.get_preprocessing_result()

    @staticmethod
    def get_image_root_path() -> Path:
        return Path(os.environ["IMAGE_VOLUME_PATH"])
