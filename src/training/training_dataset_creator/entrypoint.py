from training.hyperparameter_registry import HyperparameterRegistry
from training.rl_training.training_params import ImagePreprocessingParams, GeneralPreprocessingParams
from training.split_ratios import SplitRatios
from training.training_dataset_creator.training_dataset_creator import TrainingDatasetCreator
from training.training_dataset_creator.training_params import TrainingExecutionParams, DataParams, \
    TransformPreprocessingParams
from utils.LoggingUtils import configure_logging

configure_logging()


def main():
    training_params = HyperparameterRegistry.get_store(TrainingExecutionParams)
    training_params.set(
        {"experiment_name": "Training Dataset Creator 0.1", "use_local_juror": True, "random_seed": 42})

    data_params = HyperparameterRegistry.get_store(DataParams)
    # Mögliche alternative Datensätze:
    data_params.set({"dataset_id": "lhq_landscapes"})
    # data_params.set({"dataset_id": "twenty_images"})
    # data_params.set({"dataset_id": "flickr8k"})
    # data_params.set({"dataset_id": "places365_val_large"})

    general_processing_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams)
    general_processing_params.set({"batch_size": 64, "random_seed": 67})

    image_preprocessing_params = HyperparameterRegistry.get_store(ImagePreprocessingParams)
    image_preprocessing_params.set({"batch_size": 64, "resize_max_size": 1000})

    transform_preprocessing_params = HyperparameterRegistry.get_store(TransformPreprocessingParams)
    transform_preprocessing_params.set({"batch_size": 64, "split": SplitRatios.create_default_split_ratios()})

    trainer = TrainingDatasetCreator()
    trainer.run_training()


if __name__ == '__main__':
    main()
