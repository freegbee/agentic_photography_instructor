import time

from training.dataset_analyser.analysis_trainer import AnalysisTrainer
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams


def main():
    # fixiere den data loader als dummy data loader
    # fixiere einen output path, wo die fiftyone app läuft
    # instanziiere den TrainingDataSetCreator
    experiment_name = "Dataset Analyser"
    run_description = "POC 0.1"
    run_name = f"{time.strftime('%Y%m%d-%H%M%S')}_{run_description}"

    # Daten & Umgebung
    dataset_id = "twenty_original_split_amd-win"
    dataset_id = "flickr2k_big_original_HQ_split_amd-win"
    dataset_id = "div2k_valid_hr"
    image_size = (384, 384)

    data_params = HyperparameterRegistry.get_store(DataParams)
    data_params.set({"dataset_id": dataset_id, "image_max_size":image_size})

    trainer = AnalysisTrainer(experiment_name=experiment_name, source_dataset_id=dataset_id, fiftyone_analysis_name='div2k_valid_hr_Dataset')
    trainer.run_training(run_name)


if __name__ == '__main__':
    main()
