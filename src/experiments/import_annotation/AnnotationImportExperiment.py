import logging
import os
from typing import Optional

from experiments.shared.PhotographyExperiment import PhotographyExperiment

logger = logging.getLogger(__name__)


class AnnotationImportExperiment(PhotographyExperiment):
    """
    Experiment that imports and uploads an annotations.json file to MLflow.

    This experiment simply logs a manually created annotations.json file
    as an artifact to an MLflow run.
    """

    def __init__(self, experiment_name: str, annotation_file_path: str, run_name: Optional[str] = None):
        """
        Initialize the annotation import experiment.

        Args:
            experiment_name: Name of the MLflow experiment
            annotation_file_path: Path to the annotations.json file (relative to src/ or absolute)
            run_name: Optional name for the MLflow run
        """
        super().__init__(experiment_name)
        self.annotation_file_path = annotation_file_path
        self.run_name = run_name

    def configure(self, config: dict):
        """Configure the experiment (not used for this simple experiment)."""
        pass

    def _get_run_name(self) -> Optional[str]:
        """Return the run name if provided."""
        return self.run_name

    def _get_tags_for_run(self):
        """Return tags for the MLflow run."""
        return {
            "annotation_file": os.path.basename(self.annotation_file_path),
            "experiment_type": "annotation_import"
        }

    def _run_impl(self, experiment_created, active_run):
        """
        Main experiment logic: validate and upload the annotations file.

        Args:
            experiment_created: The MLflow experiment object
            active_run: The active MLflow run
        """
        logger.info("Starting annotation import experiment")
        logger.info("Experiment: %s", experiment_created.name)
        logger.info("Run ID: %s", active_run.info.run_id)
        logger.info("Annotation file: %s", self.annotation_file_path)

        # Validate that the file exists
        if not os.path.isabs(self.annotation_file_path):
            # If relative path, try to resolve it relative to src directory
            src_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            absolute_path = os.path.join(src_dir, self.annotation_file_path)
        else:
            absolute_path = self.annotation_file_path

        if not os.path.exists(absolute_path):
            error_msg = f"Annotation file not found: {absolute_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not os.path.isfile(absolute_path):
            error_msg = f"Path is not a file: {absolute_path}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log file information
        file_size = os.path.getsize(absolute_path)
        logger.info("Annotation file size: %d bytes", file_size)

        # Log parameters
        self.log_param("annotation_file_path", self.annotation_file_path)
        self.log_param("annotation_file_size_bytes", file_size)
        self.log_param("annotation_file_name", os.path.basename(absolute_path))

        # Upload the annotation file as artifact
        logger.info("Uploading annotation file to MLflow...")
        try:
            self.log_artifact(absolute_path)
            logger.info("Successfully uploaded annotation file to MLflow")
            self.log_metric("upload_success", 1.0)
        except Exception as e:
            logger.error("Failed to upload annotation file: %s", e)
            self.log_metric("upload_success", 0.0)
            raise

        logger.info("Annotation import experiment completed successfully")
