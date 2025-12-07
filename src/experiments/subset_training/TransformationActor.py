import logging
import os

from numpy import ndarray

from juror_client import JurorClient
from juror_shared.models_v1 import ScoringResponsePayloadV1
from transformer.AbstractTransformer import AbstractTransformer
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)


class TransformationActor:
    def __init__(self):
        self.transformer_registry = TRANSFORMER_REGISTRY
        juror_service_url = os.environ.get("JUROR_SERVICE_URL")
        if juror_service_url is None:
            raise RuntimeError(
                "Environment variable JUROR_SERVICE_URL is not set. Please set it to the Juror service endpoint URL.")
        self.juror = JurorClient(juror_service_url)

    def _get_transformer(self, transformer_label: str) -> AbstractTransformer:
        return self.transformer_registry.get(transformer_label)

    def apply_transformations(self, image_data: ndarray, transformer_label: str) -> ndarray:
        return self._get_transformer(transformer_label).transform(image_data)

    def get_score(self, image_data: ndarray) -> float:
        response: ScoringResponsePayloadV1 = self.juror.score_ndarray_bgr(image_data)
        logger.debug("Scored %s", response)
        return response.score

    def transform_and_score(self, image_data: ndarray, transformer_label: str) -> tuple[ndarray, float]:
        # Wende die Transformation an
        logger.debug("Transform and score with transformer label %s", transformer_label)
        transformed_image = self.apply_transformations(image_data, transformer_label)
        logger.debug("Transformed %s", transformer_label)
        # Berechne den Score für das transformierte Bild
        score = self.get_score(transformed_image)
        logger.debug("Scored %f", score)
        return transformed_image, score
