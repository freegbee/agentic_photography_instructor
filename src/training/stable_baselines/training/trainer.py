import logging
import os
from pathlib import Path
from typing import Optional, Dict

from stable_baselines3.common.env_util import make_vec_env

from dataset.COCODataset import COCODataset
from dataset.enhanced_coco import AnnotationFileAndImagePath
from juror_client import JurorClient
from training import mlflow_helper
from training.abstract_trainer import AbstractTrainer
from training.data_loading.dataset_load_data import DatasetLoadData
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.callbacks.rollout_success_callback import RolloutSuccessCallback
from training.stable_baselines.environment.image_render_wrapper import ImageRenderWrapper
from training.stable_baselines.environment.image_transform_env import ImageTransformEnv
from training.stable_baselines.environment.success_counting_wrapper import SuccessCountingWrapper
from training.stable_baselines.models.models import create_ppo_with_resnet_model
from training.stable_baselines.training.hyper_params import TrainingParams, DataParams, GeneralParams
from training.stable_baselines.utils.utils import get_consistent_transformers

logger = logging.getLogger(__name__)


class StableBaselineTrainer(AbstractTrainer):
    def __init__(self):
        self.training_params: TrainingParams = HyperparameterRegistry.get_store(TrainingParams).get()
        self.data_params: DataParams = HyperparameterRegistry.get_store(DataParams).get()
        super().__init__(experiment_name=self.training_params["experiment_name"],
                         source_dataset_id=self.data_params["dataset_id"])
        general_params: GeneralParams = HyperparameterRegistry.get_store(GeneralParams).get()
        self.training_seed = self.training_params["random_seed"]

        self.data_loader = DatasetLoadData(self.data_params["dataset_id"])
        self.transformers = get_consistent_transformers(general_params["transformer_labels"])
        self.success_bonus = general_params["success_bonus"]
        self.image_max_size = general_params["image_max_size"]
        self.training_source_path: Optional[Path] = None
        self.dataset_info: Dict[str, AnnotationFileAndImagePath] = {}
        self.juror_client = JurorClient(use_local=self.training_params["use_local_juror"])

        self.render_mode = self.training_params["render_mode"]
        self.render_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.training_params["render_save_dir"]

        self._register_mlflow_params()

    def _register_mlflow_params(self):
        """
        Register hyperparameter classes for MLflow tracking.
        """
        (self.ml_flow_param_builder
         .with_param_class(TrainingParams)
         .with_param_class(DataParams)
         .with_param_class(GeneralParams))

    def _load_data_impl(self):
        result = self.data_loader.load_data()
        self.training_source_path = result.destination_dir
        logger.info(
            f"Data loaded to {self.training_source_path} (is type {type(self.training_source_path)}, preparing dataset info.")
        train_ann = AnnotationFileAndImagePath(self.training_source_path / "train" / "annotations.json",
                                               self.training_source_path / "train" / "images")
        test_ann = AnnotationFileAndImagePath(self.training_source_path / "test" / "annotations.json",
                                              self.training_source_path / "test" / "images")
        valid_ann = AnnotationFileAndImagePath(self.training_source_path / "validation" / "annotations.json",
                                               self.training_source_path / "validation" / "images")
        self.dataset_info = {
            "train": train_ann,
            "test": test_ann,
            "validation": valid_ann
        }
        logger.info(f"Loaded data to destination {self.training_source_path}")

    def _preprocess_impl(self):
        os.makedirs(str(self.render_save_dir), exist_ok=True)

    def _train_impl(self):
        logger.info(f"Training started for {self.experiment_name} with images at: {self.training_source_path}")
        ann = self.dataset_info["train"]
        coco_dataset = COCODataset(ann.images_path, ann.annotation_file)
        mlflow_helper.log_param("training_annotation_file", str(ann.annotation_file))
        mlflow_helper.log_param("training_annotation_images", str(ann.images_path))
        env_fn = lambda: SuccessCountingWrapper(
            ImageRenderWrapper(
                ImageTransformEnv(
                    transformers=self.transformers,
                    coco_dataset=coco_dataset,
                    juror_client=self.juror_client,
                    success_bonus=self.success_bonus,
                    image_max_size=self.image_max_size,
                    max_transformations=self.training_params["max_transformations"],
                    seed=self.training_seed
                ),
                render_mode=self.render_mode,
                render_save_dir=self.render_save_dir,
            ),
            stats_key="episode_stats"
        )

        vec_env = make_vec_env(env_fn,
                               n_envs=self.training_params["num_vector_envs"],
                               seed=self.training_seed)
        model = create_ppo_with_resnet_model(vec_env=vec_env,
                                             n_steps=self.training_params["n_steps"],
                                             batch_size=self.training_params["mini_batch_size"],
                                             n_epochs=self.training_params["n_epochs"],
                                             feature_dim=512)

        rollout_callback = RolloutSuccessCallback(stats_key="episode_success", episode_stats_key="episode_stats")

        logger.info(f"total_training_steps param: {self.training_params['total_training_steps']}")
        logger.info(
            f"ppo n_steps: {model.n_steps}, num_envs: {vec_env.num_envs}, rollout_size: {model.n_steps * vec_env.num_envs}")

        model.learn(total_timesteps=self.training_params["total_training_steps"],
                    callback=rollout_callback)
        # TODO: Modell speichern/serialisieren, z.B.model.save("ppo_image_transform")
        logger.info(f"Training ended for {self.experiment_name}")

    def _evaluate_impl(self):
        logger.info(f"Evaluation started for {self.experiment_name} with images at: {self.training_source_path}")
        # noop
        logger.info(f"Evaluation ended for {self.experiment_name}")
