import logging
import os
from pathlib import Path
from typing import Optional, Dict, Type

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
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams
from training.stable_baselines.hyperparameter.runtime_hyperparams import RuntimeParams
from training.stable_baselines.hyperparameter.task_hyperparams import TaskParams
from training.stable_baselines.models.model_factory import AbstractModelFactory
from training.stable_baselines.utils.utils import get_consistent_transformers

logger = logging.getLogger(__name__)


class StableBaselineTrainer(AbstractTrainer):

    def __init__(self,
                 model_factory: AbstractModelFactory,
                 model_params_class: Type):
        self.model_factory = model_factory
        self.model_params_class = model_params_class

        self.runtime_params: RuntimeParams = HyperparameterRegistry.get_store(RuntimeParams).get()
        self.task_params: TaskParams = HyperparameterRegistry.get_store(TaskParams).get()
        self.data_params: DataParams = HyperparameterRegistry.get_store(DataParams).get()

        super().__init__(experiment_name=self.runtime_params["experiment_name"],
                         source_dataset_id=self.data_params["dataset_id"])

        self.model_params = HyperparameterRegistry.get_store(self.model_params_class).get()
        self.training_seed = self.runtime_params["random_seed"]
        self.data_loader = DatasetLoadData(self.data_params["dataset_id"])
        self.transformers = get_consistent_transformers(self.task_params["transformer_labels"])
        self.success_bonus = self.task_params["success_bonus"]
        self.image_max_size = self.data_params["image_max_size"]
        self.vec_env_cls = self.runtime_params["vec_env_cls"]
        self.use_worker_pool = self.runtime_params["use_worker_pool"]
        self.num_juror_workers = self.runtime_params["num_juror_workers"]
        self.training_source_path: Optional[Path] = None
        self.dataset_info: Dict[str, AnnotationFileAndImagePath] = {}

        # Pfade für das Speichern von Artefakten isolieren (pro Run), um Überschreiben zu verhindern
        # Wir nutzen den run_name, bereinigen ihn aber für das Dateisystem
        safe_run_name = "".join([c if c.isalnum() or c in "._- " else "_" for c in self.runtime_params["run_name"]]).strip()

        self.render_mode = self.runtime_params["render_mode"]
        self.render_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.runtime_params["render_save_dir"] / safe_run_name

        # Evaluation parameters
        self.evaluation_seed = self.runtime_params["evaluation_seed"]
        self.evaluation_interval = self.runtime_params["evaluation_interval"]
        self.evaluation_deterministic = self.runtime_params["evaluation_deterministic"]
        self.evaluation_visual_history = self.runtime_params["evaluation_visual_history"]
        self.evaluation_visual_history_max_images = self.runtime_params["evaluation_visual_history_max_images"]
        self.evaluation_visual_history_max_size = self.runtime_params["evaluation_visual_history_max_size"]
        self.evaluation_render_mode = self.runtime_params["evaluation_render_mode"]
        self.evaluation_render_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.runtime_params["evaluation_render_save_dir"] / safe_run_name
        self.evaluation_model_save_dir = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.runtime_params["evaluation_model_save_dir"] / safe_run_name
        self.evaluation_log_path = Path(os.environ["IMAGE_VOLUME_PATH"]) / self.runtime_params["evaluation_log_path"] / safe_run_name

        # Multy step degredation parameters
        self.use_multi_step = self.task_params["use_multi_step_wrapper"]
        self.steps_per_episode = self.task_params["steps_per_episode"]
        self.intermediate_reward = self.task_params["multi_step_intermediate_reward"]
        self.reward_shaping = self.task_params["multi_step_reward_shaping"]

        self._register_mlflow_params()

    def _register_mlflow_params(self):
        """
        Register hyperparameter classes for MLflow tracking.
        """
        (self.ml_flow_param_builder
         .with_param_class(RuntimeParams)
         .with_param_class(TaskParams)
         .with_param_class(DataParams)
         .with_param_class(self.model_params_class))

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
        os.makedirs(str(self.evaluation_model_save_dir), exist_ok=True)

    def _train_impl(self):
        logger.info(f"Training started for {self.experiment_name} with images at: {self.training_source_path}")

        # Initialisierung der Ressourcen für sauberes Aufräumen im finally-Block
        training_vec_env = None
        evaluation_vec_env = None
        eval_callback = None

        # Log datasets to MLflow (Datasets tab)
        for context, info in self.dataset_info.items():
            mlflow_helper.log_dataset(dataset_id=self.data_params["dataset_id"],
                                      annotations_file=str(info.annotation_file),
                                      images_source_path=str(info.images_path),
                                      context=context)

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
        core_env_cls = self.task_params["core_env"].env_class

        # Factory initialisieren
        # Hier wird definiert, WELCHES Environment wir nutzen.
        env_factory = ImageTransformEnvFactory(
            transformers=self.transformers,
            image_max_size=self.image_max_size,
            max_transformations=self.task_params["max_transformations"],
            success_bonus=self.success_bonus,
            juror_use_local=self.runtime_params["use_local_juror"],
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
        for i in range(self.runtime_params["num_vector_envs"]):
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

        try:
            model = self.model_factory.create_model(vec_env=training_vec_env,
                                                    params=self.model_params)

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

            # Create evaluation callback
            eval_callback = ImageTransformEvaluationCallback(
                stats_key="evaluation_episode_success",
                num_images_to_log=self.evaluation_visual_history_max_images,
                tile_max_size=self.evaluation_visual_history_max_size,
                eval_env=evaluation_vec_env,
                best_model_save_path=str(self.evaluation_model_save_dir) if self.runtime_params["store_best_model"] else None,
                log_path=str(self.evaluation_log_path),
                eval_freq=self.evaluation_interval,
                n_eval_episodes=len(evaluation_coco_dataset),
                deterministic=self.evaluation_deterministic,
                # NICE: Render callback implementieren
                render=False
            )

            logger.info("total_training_steps param: %d", self.runtime_params["total_training_steps"])

            # Logging spezifisch für PPO (n_steps) oder allgemein
            if hasattr(model, "n_steps"):
                logger.info("Model n_steps: %d, num_envs: %d, rollout_size: %d", model.n_steps,
                            training_vec_env.num_envs,
                            model.n_steps * training_vec_env.num_envs)
            elif hasattr(model, "train_freq"):
                logger.info("Model train_freq: %s, num_envs: %d", str(model.train_freq), training_vec_env.num_envs)

            callbacks = [eval_callback, rollout_callback, performance_callback]

            # Start training
            model.learn(total_timesteps=self.runtime_params["total_training_steps"],
                        callback=callbacks,
                        progress_bar=False)

            # Save final model locally (upload happens in _postprocess_impl)
            if self.runtime_params["store_final_model"]:
                final_model_path = self.evaluation_model_save_dir / "final_model"
                model.save(final_model_path)

            logger.info(f"Training ended for {self.experiment_name}")

        finally:
            # Aufräumen
            if juror_worker_pool:
                logger.info("Stopping JurorWorkerPool...")
                juror_worker_pool.stop()
            
            if training_vec_env:
                logger.info("Closing training environment...")
                training_vec_env.close()

            if evaluation_vec_env:
                logger.info("Closing evaluation environment...")
                evaluation_vec_env.close()

            # Falls das Training crasht, wird _on_training_end nicht aufgerufen.
            # Wir rufen es hier manuell auf (oder eine Cleanup-Methode), um temporäre Ordner zu löschen.
            if eval_callback and hasattr(eval_callback, "_on_training_end"):
                # Dies löscht den temporären Ordner für die Video-Frames
                eval_callback._on_training_end()

    def _evaluate_impl(self):
        logger.info(f"Evaluation started for {self.experiment_name} with images at: {self.training_source_path}")
        # noop
        logger.info(f"Evaluation ended for {self.experiment_name}")
        
    def _postprocess_impl(self):
        logger.info("Post-processing started: Uploading models to MLflow if configured.")

        # Upload final model
        if self.runtime_params["store_final_model"]:
            final_model_path = self.evaluation_model_save_dir / "final_model.zip"
            if final_model_path.exists():
                mlflow_helper.log_artifact(str(final_model_path), artifact_path="models")
                try:
                    os.remove(final_model_path)
                    logger.info(f"Deleted local final model: {final_model_path}")
                except OSError as e:
                    logger.warning(f"Could not delete local final model: {e}")

        # Upload best model
        if self.runtime_params["store_best_model"]:
            best_model_path = self.evaluation_model_save_dir / "best_model.zip"
            if best_model_path.exists():
                mlflow_helper.log_artifact(str(best_model_path), artifact_path="models")
                try:
                    os.remove(best_model_path)
                    logger.info(f"Deleted local best model: {best_model_path}")
                except OSError as e:
                    logger.warning(f"Could not delete local best model: {e}")
