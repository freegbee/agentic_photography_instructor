import logging
from pathlib import Path
from typing import List

from mlflow.entities import Experiment, Run as MlflowRun

from dataset.Utils import Utils
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.subset_training.TransformationActor import TransformationActor

logger = logging.getLogger(__name__)


class SubsetTraining(PhotographyExperiment):

    def __init__(self, experiment_name: str, run_name: str, dataset_root: Path, topk: int, batch_size: int):
        super(SubsetTraining, self).__init__(experiment_name)
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.dataset_root = dataset_root
        self.topk = topk
        self.batch_size = batch_size
        logger.info(
            "Experiment initialized with experiment_name=%s, run_name=%s, dataset_root=%s, topk=%s, batch_size=%d",
            experiment_name, run_name, dataset_root, topk, batch_size)

    def configure(self, config: dict):
        pass

    def _run_impl(self, experiment: Experiment, active_run: MlflowRun):
        dataloader = Utils.create_topk_coco_dataloader(self.dataset_root, batch_size=self.batch_size, k=int(self.topk))
        transformation_actor: TransformationActor = TransformationActor()
        result_csv: List[str] = ["image_id,image_relative_path,score_before,score_after,score_change,transformation"]
        for batch in dataloader:
            for img_data in batch:
                logger.debug(" ImageData: id=%s, path=%s, score=%.4f", img_data.id, img_data.image_relative_path,
                             img_data.score)
                transformed, score = transformation_actor.transform_and_score(img_data.get_image_data(), "CA_INV_B")
                logger.info("Transformation for image %s Score before=%.4f , score after=%.4f  --> change=%.4f",
                            img_data.image_relative_path, img_data.score, score, score - img_data.score)
                result_csv.append(
                    f"{img_data.id},{img_data.image_relative_path},{img_data.score:.4f},{score:.4f},{score - img_data.score:.4f},CA_INV_B")

        for line in result_csv:
            logger.info(line)
