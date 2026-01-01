import logging
from pathlib import Path
from typing import Dict, Optional

from dataset.enhanced_coco import AnnotationFileAndImagePath
from juror_client import JurorClient
from training.abstract_trainer import AbstractTrainer
from training.data_loading.dataset_load_data import DatasetLoadData
from training.hyperparameter_registry import HyperparameterRegistry
from training.preprocessing.annotationfile_creator import AnnotationFileCreator
from training.preprocessing.copy_and_resize_preprocessor import CopyAndResizePreprocessor
from training.preprocessing.score_preprocessor import ScorePreprocessor
from training.preprocessing.split_preprocessor import SplitPreprocessor
from training.rl_training.training_params import ImagePreprocessingParams, GeneralPreprocessingParams
from training.training_dataset_creator.training_params import TrainingExecutionParams, DataParams, \
    TransformPreprocessingParams

logger = logging.getLogger(__name__)


class TrainingDatasetCreator(AbstractTrainer):
    def __init__(self):
        training_params: TrainingExecutionParams = HyperparameterRegistry.get_store(TrainingExecutionParams).get()
        data_params: DataParams = HyperparameterRegistry.get_store(DataParams).get()
        super().__init__(experiment_name=training_params["experiment_name"],
                         source_dataset_id=data_params["dataset_id"])

        general_preprocessing_params: GeneralPreprocessingParams = HyperparameterRegistry.get_store(
            GeneralPreprocessingParams).get()
        transform_preprocessing_params: TransformPreprocessingParams = HyperparameterRegistry.get_store(
            TransformPreprocessingParams).get()

        self.splits: Optional[Dict[str, AnnotationFileAndImagePath]] = None

        (self.ml_flow_param_builder
         .with_param_class(TrainingExecutionParams)
         .with_param_class(DataParams)
         .with_param_class(GeneralPreprocessingParams)
         .with_param_class(ImagePreprocessingParams)
         .with_param_class(TransformPreprocessingParams))

        preprocessing_step_counter = 1
        self.copied_image_directory = None
        self.data_loader = DatasetLoadData(data_params["dataset_id"])
        self.copy_and_resize_preprocessor = CopyAndResizePreprocessor(preprocessing_step_counter,
                                                                      data_params["dataset_id"],
                                                                      None, Path(""))
        preprocessing_step_counter += 1
        self.annotation_file_creator = AnnotationFileCreator(preprocessing_step=preprocessing_step_counter)

        self.juror_client = JurorClient(use_local=training_params["use_local_juror"])
        preprocessing_step_counter += 1
        self.score_preprocessor = ScorePreprocessor(preprocessing_step_counter, self.juror_client)
        preprocessing_step_counter += 1
        self.split_preprocessor = (SplitPreprocessor(preprocessing_step_counter)
                                   .with_random_seed(training_params["random_seed"])
                                   .with_split_ratios(transform_preprocessing_params["split"])
                                   .with_batch_size(general_preprocessing_params["batch_size"])
                                   )

    def _load_data_impl(self):
        self.data_loader.load_data()

    def _preprocess_impl(self):
        resizing_result = self.copy_and_resize_preprocessor.preprocess()
        (self.annotation_file_creator
         .with_source_path(resizing_result.effective_destination_path)
         .with_image_path(resizing_result.effective_images_path))
        annotations_creator_result = self.annotation_file_creator.preprocess()
        (self.score_preprocessor
         .with_annotations_file_path(annotations_creator_result.annotations_file_path)
         .with_source_path(resizing_result.effective_destination_path)
         .with_image_path(resizing_result.effective_images_path))
        scoring_result = self.score_preprocessor.preprocess()
        (self.split_preprocessor
         .with_source_annotation_file(scoring_result.annotations_file_path)
         .with_source_path(resizing_result.effective_images_path)
         .with_destination_root_path(Path("")))

        split_response = self.split_preprocessor.preprocess()
        self.splits = split_response

    def _train_impl(self):
        pass

    def _evaluate_impl(self):
        pass
