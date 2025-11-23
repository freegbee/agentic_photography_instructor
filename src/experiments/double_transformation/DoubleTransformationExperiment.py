import logging
import os
import random
import uuid
from pathlib import Path
from typing import Optional, Dict, Any

from torch.utils.data import DataLoader

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from dataset.COCODataset import COCODataset
from dataset.TopKSampler import TopKSampler
from dataset.Utils import Utils as DatasetUtils
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.shared.Utils import Utils as SharedUtils
# lokale Imports für Verarbeitung
from juror_client import JurorClient
from utils.CocoBuilder import CocoBuilder
from utils.ConfigLoader import ConfigLoader
from utils.ImageUtils import ImageUtils

logger = logging.getLogger(__name__)


class DoubleTransformationExperiment(PhotographyExperiment):
    def __init__(self, experiment_name: str, target_directory_root: str = "double_transformed",
                 run_name: Optional[str] = None, source_dataset_id: str = "single_image",
                 max_images: Optional[int] = None, seed: int = 42):
        super().__init__(experiment_name)
        self.run_name = run_name
        self.source_dataset_id = source_dataset_id
        self.max_images = max_images
        self.target_directory_root = Path(os.environ.get("IMAGE_VOLUME_PATH", ".")) / target_directory_root
        self.seed = seed
        self.random = random.Random(seed)
        self.coco_builder = None
        logger.info("Initialized DoubleTransformationExperiment: %s", experiment_name)

    def configure(self, config: dict):
        pass

    def _get_tags_for_run(self) -> Dict[str, Any]:
        return {"dataset_id": self.source_dataset_id}

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment_created, active_run):
        logger.info(
            "Running DoubleTransformationExperiment with source_dataset_id=%s, target_directory_root=%s, max_images=%s, seed=%d",
            self.source_dataset_id, self.target_directory_root, self.max_images, self.seed)

        try:
            config_dict = ConfigLoader.load_dataset_config(self.source_dataset_id)
        except Exception as e:
            logger.exception("Exception loading config for dataset %s: %s", self.source_dataset_id, e)
            raise

        # Dataset-Konfiguration laden und Bildpfad ermitteln
        self.dataset_config = ImageDatasetConfiguration.from_dict(self.source_dataset_id, config_dict)

        # Sicherstellen, dass Images vorhanden sind
        images_root_path = self.dataset_config.calculate_images_root_path()
        image_dataset_hash = SharedUtils.ensure_image_dataset(self.dataset_config.dataset_id)
        self.log_metric("ensure_image_dataset_duration_seconds", 0.0)
        self.log_param("dataset_hash", image_dataset_hash)

        # COCO-Input Dataset
        annotation_path = self.dataset_config.calculate_annotations_file_path()
        source_dataset = COCODataset(images_root_path, annotation_path)

        # Wenn max_images gesetzt, nutze TopKSampler -> vorhandene Utils-Funktion verwenden
        if self.max_images is not None:
            # Berechne Scores für das Dataset (nutzt dataset.Utils.calculate_dataset_scores intern)
            DatasetUtils.calculate_dataset_scores(source_dataset)
            logger.info("Calculated %d scores for dataset", len(source_dataset.scores))
            sampler = TopKSampler(source_dataset, k=int(self.max_images))
            # Note: 'sampler' is assigned here for future use in image selection, but is not yet used in this skeleton implementation.
        else:
            sampler = None

        # Output COCO-Builder initialisieren
        self.coco_builder = CocoBuilder(self.source_dataset_id)
        self.coco_builder.set_description(f"Double transformation dataset derived from {self.source_dataset_id}")

        # Ensure output directories exist
        try:
            self.target_directory_root.mkdir(parents=True, exist_ok=True)
            (self.target_directory_root / "images").mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create target directories: %s", e)

        # Speichere initiale leere COCO Datei (wird später befüllt)
        coco_file_path = self.target_directory_root / "annotations.json"
        self.coco_builder.save(str(coco_file_path))
        # Logge das erzeugte Artefakt
        try:
            self.log_artifact(local_path=str(coco_file_path))
        except Exception:
            # logging only, nicht fatal
            logger.debug("log_artifact failed or mlflow not reachable")

        logger.info("DoubleTransformationExperiment preparation complete. Ready to process images.")

        # JurorClient initialisieren
        try:
            self.jurorClient = JurorClient(os.environ.get("JUROR_SERVICE_URL", "http://localhost:5010"))
        except Exception:
            logger.exception("Unable to initialize JurorClient; proceeding without juror (scores will be None)")
            self.jurorClient = None

        # DataLoader aufbauen (TopKSampler falls vorhanden)
        if sampler is not None:
            dataloader = DataLoader(source_dataset, batch_size=1, sampler=sampler,
                                    collate_fn=DatasetUtils.collate_keep_size)
        else:
            dataloader = DataLoader(source_dataset, batch_size=1, collate_fn=DatasetUtils.collate_keep_size)

        # Haupt-Loop: Verarbeite Bilder
        for batch_index, batch in enumerate(dataloader):
            for image_data in batch:
                try:
                    # Aktuell werden noch keine echten Transformer-Paare übergeben. Verwende None als Identity-Transformer.
                    self.process_image(image_data, None, None)
                except Exception as e:
                    logger.exception("Error processing image %s: %s", getattr(image_data, 'image_path', None), e)

        # Nachverarbeitung: speichere das finale COCO-File
        coco_file_path = self.target_directory_root / "annotations.json"
        self.coco_builder.save(str(coco_file_path))
        try:
            self.log_artifact(local_path=str(coco_file_path))
        except Exception:
            logger.debug("log_artifact failed or mlflow not reachable during final save")

        logger.info("Finished processing images in DoubleTransformationExperiment")

    def process_image(self, image_data, transformer1, transformer2):
        """
        Process a single image with two transformers, save the resulting image and register it in the CocoBuilder.
        If `transformer1` or `transformer2` is None, treat them as identity transformers.

        NOTE: This function currently only saves the transformed image and adds the image entry to the CocoBuilder.
        The annotations and categories are left as a TODO placeholder to be implemented in a later step.
        """
        # Lade Originalbild als ndarray
        original = image_data.get_image_data("BGR")

        # Hilfsfunktion: score via jurorClient if available
        def _score(arr):
            if self.jurorClient is None:
                return None
            try:
                resp = self.jurorClient.score_ndarray(arr, filename=str(image_data.image_relative_path.name))
                # support different return types
                score = None
                if hasattr(resp, 'score'):
                    score = getattr(resp, 'score')
                elif isinstance(resp, dict):
                    score = resp.get('score')
                return score
            except Exception:
                logger.exception("Juror scoring failed")
                return None

        # Verwende zuerst den bereits vom DataLoader gelieferten score (falls vorhanden).
        initial_score = getattr(image_data, 'score', None)
        if initial_score is None:
            initial_score = _score(original)

        # Transformer 1 (identity if None)
        if transformer1 is None:
            img_t1 = original
            t1_label = "IDENTITY"
        else:
            img_t1 = transformer1.transform(original)
            t1_label = getattr(transformer1, 'label', transformer1.__class__.__name__)

        score_after_t1 = _score(img_t1)

        # Transformer 2 (identity if None)
        if transformer2 is None:
            img_t2 = img_t1
            t2_label = "IDENTITY"
        else:
            img_t2 = transformer2.transform(img_t1)
            t2_label = getattr(transformer2, 'label', transformer2.__class__.__name__)

        score_after_t2 = _score(img_t2)

        # Generiere eindeutigen Dateinamen
        orig_name = image_data.image_relative_path.stem if image_data.image_relative_path is not None else "img"
        unique_suffix = uuid.uuid4().hex
        out_filename = f"{orig_name}__{t1_label}__{t2_label}__{unique_suffix}.png"
        out_path = self.target_directory_root / "images" / out_filename

        # Speichern
        try:
            ImageUtils.save_image(img_t2, str(out_path))
        except Exception:
            logger.exception("Failed to save transformed image to %s", out_path)

        # Bildgröße bestimmen
        if hasattr(img_t2, 'shape'):
            h = int(img_t2.shape[0])
            w = int(img_t2.shape[1])
        else:
            h = 0
            w = 0

        # Füge Bild ins COCO-Builder ein
        image_id = self.coco_builder.add_image(out_filename, w, h)

        # Schreibe Annotationen für Transformation 1 (falls vorhanden)
        try:
            # Transformation-Label hinzufügen
            self.coco_builder.add_image_transformation_annotation(image_id, t1_label)
            # Score-Annotation mit initial_score und score_after_t1
            if score_after_t1 is not None or initial_score is not None:
                # Wenn score_after_t1 None, wird trotzdem eingetragen (value kann None sein)
                # CocoBuilder erwartet float; guard: only pass if score_after_t1 is not None
                if score_after_t1 is not None:
                    self.coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t1), float(initial_score) if initial_score is not None else None, transformer_name=t1_label)
                else:
                    # Wenn kein After-Score, speichern wir nur initial als score (fallback)
                    if initial_score is not None:
                        self.coco_builder.add_image_transformation_score_annotation(image_id, float(initial_score), float(initial_score), transformer_name=t1_label)
        except Exception:
            logger.exception("Failed to add transformation 1 annotations for image_id %s", image_id)

        # Schreibe Annotationen für Transformation 2 (falls vorhanden)
        try:
            self.coco_builder.add_image_transformation_annotation(image_id, t2_label)
            if score_after_t2 is not None or score_after_t1 is not None:
                if score_after_t2 is not None:
                    self.coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t2), float(score_after_t1) if score_after_t1 is not None else None, transformer_name=t2_label)
                else:
                    if score_after_t1 is not None:
                        self.coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t1), float(score_after_t1), transformer_name=t2_label)

        except Exception:
            logger.exception("Failed to add transformation 1 annotations for image_id %s", image_id)
