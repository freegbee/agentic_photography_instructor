from training.hyperparameter_registry import HyperparameterRegistry
from training.rl_training.rl_trainer import RlTrainer
from training.rl_training.training_params import ImagePreprocessingParams, TransformPreprocessingParams, DataParams, \
    TrainingExecutionParams, GeneralPreprocessingParams
from training.split_ratios import SplitRatios
from transformer import POC_MULTI_TWO_STEP_TRANSFORMERS
from utils.LoggingUtils import configure_logging

configure_logging()


def main():
    training_params = HyperparameterRegistry.get_store(TrainingExecutionParams)
    training_params.set(
        {"experiment_name": "Multy Transformation Degrading PoC 0.2", "use_local_juror": True, "random_seed": 42})

    data_params = HyperparameterRegistry.get_store(DataParams)
    # Mögliche alternative Datensätze:
    data_params.set({"dataset_id": "lhq_landscapes"})
    # data_params.set({"dataset_id": "twenty_images"})
    # data_params.set({"dataset_id": "flickr8k"})
    # data_params.set({"dataset_id": "places365_val_large"})

    general_processing_params = HyperparameterRegistry.get_store(GeneralPreprocessingParams)
    general_processing_params.set({"batch_size": 64, "random_seed": 67})

    image_preprocessing_params = HyperparameterRegistry.get_store(ImagePreprocessingParams)
    image_preprocessing_params.set({"batch_size": 64, "resize_max_size": 384})

    transform_preprocessing_params = HyperparameterRegistry.get_store(TransformPreprocessingParams)
    h = {"batch_size": 64,
         "transformer_names": POC_MULTI_TWO_STEP_TRANSFORMERS,
         "use_random_transformer": True,
         "num_transformations": 2,  # Apply 2 transformations per image
         "split": SplitRatios.create_default_split_ratios()}
    transform_preprocessing_params.set(h)

    trainer = RlTrainer()
    trainer.run_training()


if __name__ == '__main__':
    main()
