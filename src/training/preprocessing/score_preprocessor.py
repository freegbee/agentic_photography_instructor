import logging
from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from dataset.COCODataset import COCODataset
from dataset.Utils import Utils
from dataset.agentic_coco_image import CocoImageData
from experiments.image_scoring.BatchImageScoringMetricAccumulator import BatchImageScoringMetricAccumulator
from juror_client import JurorClient
from juror_shared.models_v1 import ScoringResponsePayloadV1
from training import mlflow_helper
from training.hyperparameter_registry import HyperparameterRegistry
from training.mlflow_utils import mlflow_logging
from training.preprocessing.abstract_preprocessor import AbstractPreprocessor, RESULT
from training.rl_training.training_params import GeneralPreprocessingParams, TrainingExecutionParams
from utils.CocoBuilder import CocoBuilder

logger = logging.getLogger(__name__)


class ScorePreprocessorResult:
    def __init__(self, coco_builder: CocoBuilder, annotations_file_path: Path):
        self.coco_builder = coco_builder
        self.annotations_file_path = annotations_file_path


class ScorePreprocessor(AbstractPreprocessor[ScorePreprocessorResult]):
    """Preprocessor for scoring datasets and writing corresponding annotations."""

    def __init__(self, preprocessing_step: int, juror_client: Optional[JurorClient] = None):
        super().__init__(preprocessing_step)
        general_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams).get()
        training_params: TrainingExecutionParams = HyperparameterRegistry.get_store(TrainingExecutionParams).get()
        self.source_path: Optional[Path] = None
        self.annotations_file_path: Optional[Path] = None
        self.image_path: Optional[Path] = None
        self.coco_builder: Optional[CocoBuilder] = None
        self.batch_size = general_params["batch_size"]
        self.juror_client = juror_client if juror_client is not None else JurorClient(
            use_local=training_params["use_local_juror"])

    def with_source_path(self, source_path: Path) -> 'ScorePreprocessor':
        self.source_path = source_path
        return self

    def with_annotations_file_path(self, annotations_file_path: Path) -> 'ScorePreprocessor':
        self.annotations_file_path = annotations_file_path
        return self

    def with_image_path(self, image_path: Path) -> 'ScorePreprocessor':
        self.image_path = image_path
        return self

    @mlflow_logging("perf/score_preprocessor_duration_seconds")
    def _preprocess_impl(self):
        self.coco_builder = CocoBuilder(source_path=self.source_path)
        self._score_images()

        # Nachverarbeitung
        mlflow_helper.log_metric("total_number_of_images_scored", len(self.coco_builder.images))

        self.coco_builder.save(str(self.annotations_file_path))
        mlflow_helper.log_artifact(local_path=str(self.annotations_file_path),
                                   step=self.step_arg,
                                   artifact_path="scored_annotation")

    def _score_images(self):
        logger.info("Start scoring for image in path %s based on cocofile %s", self.image_path,
                    self.annotations_file_path)
        # Don't include transformations yet - they haven't been applied at this stage
        dataset: COCODataset = COCODataset(self.image_path, self.annotations_file_path, include_transformations=False)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=Utils.collate_keep_size)
        metrics_accumulator = BatchImageScoringMetricAccumulator()
        for batch_index, batch in enumerate(dataloader):
            metrics_accumulator.reset()
            self._process_batch(batch_index, batch, dataset.coco, metrics_accumulator)
            mlflow_helper.log_batch_metrics(metrics_accumulator.compute_metrics(), batch_index)

        self.coco_builder.images = [dict(item) for item in dataset.coco.imgs.values()]
        self.coco_builder.annotations = [dict(item) for item in dataset.coco.anns.values()]
        self.coco_builder.categories = [dict(item) for item in dataset.coco.cats.values()]

    def _process_batch(self, batch_index, batch, coco, metrics_accumulator: BatchImageScoringMetricAccumulator):
        metrics_accumulator.start(batch_index)
        image_data: CocoImageData
        for image_data in batch:
            scoring_response: ScoringResponsePayloadV1 = self.juror_client.score_ndarray_bgr(
                image_data.get_image_data('BGR'))
            coco.add_scoring_annotation(image_id=image_data.id,
                                        score=scoring_response.score,
                                        initial_score=scoring_response.score)
            metrics_accumulator.add_score(scoring_response.score)
        metrics_accumulator.stop()

    def get_preprocessing_result(self) -> RESULT:
        return ScorePreprocessorResult(self.coco_builder, self.annotations_file_path)
