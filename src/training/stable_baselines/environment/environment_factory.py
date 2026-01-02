import abc
import logging
import uuid
from pathlib import Path
from queue import Queue
from typing import Optional, Callable, List, Tuple

import cv2
from stable_baselines3.common.vec_env import SubprocVecEnv

from juror_client import JurorClient
from juror_client.juror_worker_pool import JurorQueueService
from training.stable_baselines.environment.image_observation_wrapper import ImageObservationWrapper
from training.stable_baselines.environment.image_render_wrapper import ImageRenderWrapper
from training.stable_baselines.environment.image_transform_env import ImageTransformEnv
from training.stable_baselines.environment.samplers import CocoDatasetSampler
from training.stable_baselines.environment.success_counting_wrapper import SuccessCountingWrapper
from transformer.AbstractTransformer import AbstractTransformer

logger = logging.getLogger(__name__)


class AbstractEnvFactory(abc.ABC):
    """
    Abstrakte Basis-Factory für RL-Environments.
    Definiert den Lebenszyklus der Environment-Erstellung im Worker-Prozess (Template Method Pattern).
    """

    def __init__(self, vec_env_cls=None):
        self.vec_env_cls = vec_env_cls

    def create_env_fn(self,
                      seed: int,
                      pool_request_queue: Optional[Queue] = None,
                      pool_reply_queue: Optional[Queue] = None,
                      **kwargs
                      ) -> Callable[[], SuccessCountingWrapper]:
        """
        Erstellt das Callable für SubprocVecEnv.
        Nimmt dynamische Runtime-Parameter (Seed, Queues) entgegen.
        """
        # Wir nutzen hier ein Lambda, das auf die Instanz-Methode _construct_environment verweist.
        # Da diese Factory-Instanz picklable ist, wird sie mit in den Worker-Prozess kopiert.
        return lambda: self._construct_environment(seed, pool_request_queue, pool_reply_queue, **kwargs)

    def _construct_environment(self, seed: int, pool_request_queue: Optional[Queue],
                               pool_reply_queue: Optional[Queue], **kwargs):
        """
        Die interne Methode, die TATSÄCHLICH im Worker-Prozess ausgeführt wird.
        Führt das Setup (Threading) durch und baut den Environment-Stack.
        """
        # 1. Prozess-Setup (Threading limitieren)
        cv2.setNumThreads(0)

        # 2. Core Environment erstellen (abstrakt)
        env = self._create_core_env(seed, pool_request_queue, pool_reply_queue, **kwargs)

        # 3. Wrapper anwenden (überschreibbar)
        env = self._apply_wrappers(env, **kwargs)

        return env

    @abc.abstractmethod
    def _create_core_env(self, seed: int, pool_request_queue: Optional[Queue],
                         pool_reply_queue: Optional[Queue], **kwargs):
        """Erstellt das innerste Gym-Environment."""
        pass

    def _apply_wrappers(self, env, **kwargs):
        """Kann überschrieben werden, um Wrapper hinzuzufügen."""
        return env


class ImageTransformEnvFactory(AbstractEnvFactory):
    """
    Konkrete Factory für das ImageTransformEnv (Photography Instructor).
    """

    def __init__(self,
                 transformers: List[AbstractTransformer],
                 image_max_size: Tuple[int, int],
                 max_transformations: int,
                 success_bonus: float,
                 juror_use_local: bool,
                 vec_env_cls=None,
                 core_env_cls=ImageTransformEnv):
        """
        Initialisiert die Factory mit den statischen Parametern, die für alle Environments gleich sind.
        """
        super().__init__(vec_env_cls)
        self.transformers = transformers
        self.image_max_size = image_max_size
        self.max_transformations = max_transformations
        self.success_bonus = success_bonus
        self.juror_use_local = juror_use_local
        self.core_env_cls = core_env_cls

    # Überschreiben der Signatur für Type-Hinting und spezifische Argumente
    # noinspection PyMethodOverriding
    def create_env_fn(self,
                      coco_dataset_sampler_factory: Callable[[], CocoDatasetSampler],
                      seed: int,
                      render_mode: str,
                      render_save_dir: Path,
                      stats_key: str,
                      keep_image_history: bool = False,
                      history_image_max_size: int = 150,
                      pool_request_queue: Optional[Queue] = None,
                      pool_reply_queue: Optional[Queue] = None
                      ) -> Callable[[], SuccessCountingWrapper]:

        return super().create_env_fn(
            seed=seed,
            pool_request_queue=pool_request_queue,
            pool_reply_queue=pool_reply_queue,
            # Kwargs für _create_core_env und _apply_wrappers:
            coco_dataset_sampler_factory=coco_dataset_sampler_factory,
            render_mode=render_mode,
            render_save_dir=render_save_dir,
            stats_key=stats_key,
            keep_image_history=keep_image_history,
            history_image_max_size=history_image_max_size
        )

    def _create_core_env(self, seed: int, pool_request_queue: Optional[Queue],
                         pool_reply_queue: Optional[Queue], **kwargs):
        """
        Erstellt das ImageTransformEnv inkl. Juror-Client Setup.
        """
        # 1. Juror Service Setup
        register_name = "default_juror_client"
        if self.vec_env_cls == SubprocVecEnv:
            register_name = f"juror_client_{uuid.uuid4()}"

        service_instance = None
        if pool_request_queue is not None and pool_reply_queue is not None:
            service_instance = JurorQueueService(pool_request_queue, pool_reply_queue)

        # 2. Sampler erstellen
        # Die Factory wird hier aufgerufen, da der Sampler im Worker-Prozess erstellt werden muss
        sampler = kwargs["coco_dataset_sampler_factory"]()

        # 3. Environment instanziieren
        return self.core_env_cls(
            transformers=self.transformers,
            coco_dataset_sampler=sampler,
            juror_client=JurorClient(use_local=self.juror_use_local, register_name=register_name,
                                     service=service_instance),
            success_bonus=self.success_bonus,
            image_max_size=self.image_max_size,
            max_transformations=self.max_transformations,
            seed=seed
        )

    def _apply_wrappers(self, env, **kwargs):
        """
        Wendet die Standard-Wrapper (Observation, Render, Success) an.
        """
        env = ImageObservationWrapper(env, image_max_size=self.image_max_size)

        env = ImageRenderWrapper(
            env,
            render_mode=kwargs["render_mode"],
            render_save_dir=kwargs["render_save_dir"],
            keep_image_history=kwargs["keep_image_history"],
            history_image_max_size=kwargs["history_image_max_size"]
        )

        return SuccessCountingWrapper(env, stats_key=kwargs["stats_key"])