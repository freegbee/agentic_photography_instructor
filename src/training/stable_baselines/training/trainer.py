import logging
import os
from pathlib import Path
from typing import Optional, Dict

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from dataset.COCODataset import COCODataset
from dataset.enhanced_coco import AnnotationFileAndImagePath
from juror_client.juror_worker_pool import JurorWorkerPool
from training import mlflow_helper
from training.abstract_trainer import AbstractTrainer
from training.data_loading.dataset_load_data import DatasetLoadData
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.callbacks.image_transform_evaluation_callback import ImageTransformEvaluationCallback
from training.stable_baselines.callbacks.performance_callback import MlflowPerformanceCallback
from training.stable_baselines.callbacks.rollout_success_callback import RolloutSuccessCallback
from training.stable_baselines.environment.environment_factory import ImageTransformEnvFactory
from training.stable_baselines.environment.samplers import SequentialCocoDatasetSampler, RandomCocoDatasetSampler
from training.stable_baselines.environment.welldefined_environments import WellDefinedEnvironment
from training.stable_baselines.models.learning_rate_schedules import linear_schedule
from training.stable_baselines.models.model_factory import PpoModelFactory
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
        self.learning_rate = general_params["learning_rate"]
        self.success_bonus = general_params["success_bonus"]
        self.image_max_size = general_params["image_max_size"]
        self.vec_env_cls = general_params["vec_env_cls"]
        self.use_worker_pool = general_params["use_worker_pool"]
        self.num_juror_workers = general_params["num_juror_workers"]
        self.training_source_path: Optional[Path] = None
        self.dataset_info: Dict[str, AnnotationFileAndImagePath] = {}

        self.render_mode = self.training_params["render_mode"]
        self.render_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.training_params["render_save_dir"]

        # Evaluation parameters
        self.evaluation_seed = self.training_params["evaluation_seed"]
        self.evaluation_interval = self.training_params["evaluation_interval"]
        self.evaluation_deterministic = self.training_params["evaluation_deterministic"]
        self.evaluation_visual_history = self.training_params["evaluation_visual_history"]
        self.evaluation_visual_history_max_images = self.training_params["evaluation_visual_history_max_images"]
        self.evaluation_visual_history_max_size = self.training_params["evaluation_visual_history_max_size"]
        self.evaluation_render_mode = self.training_params["evaluation_render_mode"]
        self.evaluation_render_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.training_params[
            "evaluation_render_save_dir"]
        self.evaluation_model_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.training_params[
            "evaluation_model_save_dir"]
        self.evaluation_log_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.training_params["evaluation_log_path"]

        # Multy step degredation parameters
        self.use_multi_step = self.training_params["use_multi_step_wrapper"]
        self.steps_per_episode = self.training_params["steps_per_episode"]
        self.intermediate_reward = self.training_params["multi_step_intermediate_reward"]
        self.reward_shaping = self.training_params["multi_step_reward_shaping"]

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
        os.makedirs(str(self.evaluation_render_save_dir), exist_ok=True)

    def _train_impl(self):
        logger.info(f"Training started for {self.experiment_name} with images at: {self.training_source_path}")

        # Create training environment
        training_annotations = self.dataset_info["train"]
        training_coco_dataset = COCODataset(images_root_path=training_annotations.images_path,
                                            annotation_file=training_annotations.annotation_file)

        # --- JUROR WORKER POOL SETUP ---
        # Wenn wir Multiprocessing nutzen, starten wir einen Pool von GPU-Workern.
        # Das entkoppelt die Environments (CPU) von der Bewertung (GPU).
        juror_worker_pool = None
        if self.use_worker_pool:
            num_workers = self.num_juror_workers
            juror_worker_pool = JurorWorkerPool(num_workers=num_workers)
            juror_worker_pool.start()
        # -------------------------------

        # Bestimmen der Environment-Klasse basierend auf den Parametern (String -> Klasse)
        vec_env_cls = SubprocVecEnv if self.vec_env_cls == "SubprocVecEnv" else DummyVecEnv

        # Bestimmen der Core-Environment Klasse
        # Standard ist ImageTransformEnv, kann via GeneralParams überschrieben werden
        core_env_cls = self.training_params.get("core_env", WellDefinedEnvironment.IMAGE_DEDEGRATION).env_class

        # Factory initialisieren
        # Hier wird definiert, WELCHES Environment wir nutzen.
        env_factory = ImageTransformEnvFactory(
            transformers=self.transformers,
            image_max_size=self.image_max_size,
            max_transformations=self.training_params["max_transformations"],
            success_bonus=self.success_bonus,
            juror_use_local=self.training_params["use_local_juror"],
            vec_env_cls=vec_env_cls,
            core_env_cls=core_env_cls,
            use_multi_step=self.use_multi_step,
            steps_per_episode=self.steps_per_episode,
            intermediate_reward=self.intermediate_reward,
            reward_shaping=self.reward_shaping
        )

        # WICHTIG: Wir erstellen eine Liste von Funktionen, damit jedes Environment einen EIGENEN Seed bekommt.
        # Sonst nutzen alle 12 Envs den gleichen Seed und liefern exakt die gleichen Bilder -> Verschwendung.
        training_env_fns = []
        for i in range(self.training_params["num_vector_envs"]):
            # Seed inkrementieren pro Environment
            env_seed = self.training_seed + i
            # Lambda muss den aktuellen Wert von env_seed binden (s=env_seed)
            sampler_factory = lambda s=env_seed: RandomCocoDatasetSampler(training_coco_dataset, s)

            # Reply Queue für dieses spezifische Environment erstellen (via Manager)
            env_reply_queue = None
            if juror_worker_pool:
                # WICHTIG: Muss via Manager erstellt werden für SubprocVecEnv auf Windows
                env_reply_queue = juror_worker_pool.manager.Queue()

            training_env_fns.append(env_factory.create_env_fn(
                coco_dataset_sampler_factory=sampler_factory,
                seed=env_seed,
                render_mode=self.render_mode,
                render_save_dir=self.render_save_dir,
                stats_key="episode_success",
                keep_image_history=False,
                pool_request_queue=juror_worker_pool.request_queue if juror_worker_pool else None,
                pool_reply_queue=env_reply_queue
            ))

        training_vec_env = vec_env_cls(training_env_fns)

        model_learning_schedule = linear_schedule(self.learning_rate)

        try:
            model = (PpoModelFactory(self.training_params["ppo_model_variant"])
                     .create_model(vec_env=training_vec_env,
                                   learning_rate=model_learning_schedule,
                                   n_steps=self.training_params["n_steps"],
                                   batch_size=self.training_params["mini_batch_size"],
                                   n_epochs=self.training_params["n_epochs"]))

            # model = create_dqn_with_resnet_model(vec_env=training_vec_env,
            #                                      learning_rate=model_learning_schedule,
            #                                      buffer_size=2_000,
            #                                      batch_size=self.training_params["mini_batch_size"],
            #                                      learning_starts=1_000,
            #                                      train_freq=4,
            #                                      feature_dim=512)

            rollout_callback = RolloutSuccessCallback(training_episode_stats_key="episode_success",
                                                      evaluation_episode_stats_key="evaluation_episode_success")
            performance_callback = MlflowPerformanceCallback()

            mlflow_helper.log_param("training_annotation_file", str(training_annotations.annotation_file))
            mlflow_helper.log_param("training_annotation_images", str(training_annotations.images_path))

            # Create evaluation environment
            evaluation_annotations = self.dataset_info["validation"]
            evaluation_coco_dataset = COCODataset(images_root_path=evaluation_annotations.images_path,
                                                  annotation_file=evaluation_annotations.annotation_file,
                                                  include_transformations=False)
            evaluation_sampler_factory = lambda: SequentialCocoDatasetSampler(evaluation_coco_dataset)

            # Für Evaluation nutzen wir n_envs=1.
            # Grund: Der SequentialSampler startet immer bei 0. Mit 12 Envs würden wir 12x das gleiche Bild 0, dann 12x Bild 1 bewerten.
            # Das ist redundant. Die Warnung von SB3 bzgl. unterschiedlicher Env-Anzahl können wir ignorieren.
            eval_reply_queue = None
            if juror_worker_pool:
                # WICHTIG: Muss via Manager erstellt werden
                eval_reply_queue = juror_worker_pool.manager.Queue()

            evaluation_env_fn = env_factory.create_env_fn(
                coco_dataset_sampler_factory=evaluation_sampler_factory,
                seed=self.evaluation_seed,
                render_mode=self.evaluation_render_mode,
                render_save_dir=self.evaluation_render_save_dir,
                keep_image_history=self.evaluation_visual_history,
                history_image_max_size=self.evaluation_visual_history_max_size,
                stats_key="evaluation_episode_success",
                pool_request_queue=juror_worker_pool.request_queue if juror_worker_pool else None,
                pool_reply_queue=eval_reply_queue)

            evaluation_vec_env = make_vec_env(env_id=evaluation_env_fn,
                                              n_envs=1,
                                              seed=self.evaluation_seed,
                                              vec_env_cls=DummyVecEnv)
            # eval_callback = EvaluationCallback(
            #     eval_env=evaluation_vec_env,
            #     best_model_save_path=str(self.evaluation_model_save_dir),
            #     log_path=str(self.evaluation_log_path),
            #     eval_freq=self.evaluation_interval,
            #     n_eval_episodes=len(evaluation_coco_dataset),
            #     deterministic=self.evaluation_deterministic,
            #     # NICE: Render callback implementieren
            #     render=False
            # )

            eval_callback = ImageTransformEvaluationCallback(
                stats_key="evaluation_episode_success",
                num_images_to_log=self.evaluation_visual_history_max_images,
                tile_max_size=self.evaluation_visual_history_max_size,
                eval_env=evaluation_vec_env,
                best_model_save_path=str(self.evaluation_model_save_dir),
                log_path=str(self.evaluation_log_path),
                eval_freq=self.evaluation_interval,
                n_eval_episodes=len(evaluation_coco_dataset),
                deterministic=self.evaluation_deterministic,
                # NICE: Render callback implementieren
                render=False
            )

            logger.info("total_training_steps param: %d", self.training_params["total_training_steps"])
            logger.info("ppo n_steps: %d, num_envs: %d, rollout_size: %d", model.n_steps, training_vec_env.num_envs,
                        model.n_steps * training_vec_env.num_envs)

            callbacks = [eval_callback, rollout_callback, performance_callback]

            # Start training
            model.learn(total_timesteps=self.training_params["total_training_steps"],
                        callback=callbacks,
                        progress_bar=False)

            logger.info(f"Training ended for {self.experiment_name}")

        finally:
            # Aufräumen
            if juror_worker_pool:
                juror_worker_pool.stop()

    def _evaluate_impl(self):
        logger.info(f"Evaluation started for {self.experiment_name} with images at: {self.training_source_path}")
        # noop
        logger.info(f"Evaluation ended for {self.experiment_name}")
