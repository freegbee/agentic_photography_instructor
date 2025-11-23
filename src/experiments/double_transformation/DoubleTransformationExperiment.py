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
from transformer import generate_transformer_pairs
from utils.CocoBuilder import CocoBuilder
from utils.ConfigLoader import ConfigLoader
from utils.ImageUtils import ImageUtils
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)


class DoubleTransformationExperiment(PhotographyExperiment):
    def __init__(self, experiment_name: str, target_directory_root: str = "double_transformed",
                 run_name: Optional[str] = None, source_dataset_id: str = "single_image",
                 max_images: Optional[int] = None, seed: int = 42,
                 transformer_sample_size: Optional[int] = None, transformer_sample_seed: Optional[int] = None,
                 batch_size: int = 4):
        super().__init__(experiment_name)
        self.run_name = run_name
        self.source_dataset_id = source_dataset_id
        self.max_images = max_images
        self.target_directory_root = Path(os.environ.get("IMAGE_VOLUME_PATH", ".")) / target_directory_root
        self.seed = seed
        self.random = random.Random(seed)
        # Performance tuning: batch_size for DataLoader
        self.batch_size = int(batch_size)
        # Einstellungen für Transformer-Paare
        self.transformer_sample_size = transformer_sample_size
        self.transformer_sample_seed = transformer_sample_seed
        self.coco_builder = None
        # ensure attributes exist so other methods can reference them directly
        self.jurorClient = None
        logger.info("Initialized DoubleTransformationExperiment: %s", experiment_name)

    def configure(self, config: dict):
        pass

    def _get_tags_for_run(self) -> Dict[str, Any]:
        return {"dataset_id": self.source_dataset_id, "performance": "batchsize"}

    def _get_run_name(self) -> Optional[str]:
        return self.run_name

    def _run_impl(self, experiment_created, active_run):
        logger.info(
            "Running DoubleTransformationExperiment with source_dataset_id=%s, target_directory_root=%s, max_images=%s, seed=%d, batch_size=%s",
            self.source_dataset_id, self.target_directory_root, self.max_images, self.seed, self.batch_size)

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

        # Log transformer sampling params to MLflow (safe stringify)
        try:
            self.log_param("transformer_sample_size", str(self.transformer_sample_size))
            self.log_param("transformer_sample_seed", str(self.transformer_sample_seed))
            # Log batch_size as MLflow parameter
            try:
                self.log_param("batch_size", str(self.batch_size))
            except Exception:
                logger.debug("Failed to log batch_size to MLflow")
        except Exception:
            logger.debug("Failed to log transformer sampling params to MLflow")

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
            logger.debug("Juror client ready to process images. Using service %s", self.jurorClient._service)
        except Exception:
            logger.exception("Unable to initialize JurorClient; proceeding without juror (scores will be None)")
            self.jurorClient = None

        # DataLoader aufbauen (TopKSampler falls vorhanden)
        if sampler is not None:
            dataloader = DataLoader(source_dataset, batch_size=self.batch_size, sampler=sampler,
                                    collate_fn=DatasetUtils.collate_keep_size)
        else:
            dataloader = DataLoader(source_dataset, batch_size=self.batch_size, collate_fn=DatasetUtils.collate_keep_size)

        # Erzeuge Transformer-Paare (Kreuzprodukt mit Filter). Verwende sample_size/seed falls gesetzt.
        pair_labels = generate_transformer_pairs(sample_size=self.transformer_sample_size,
                                                 seed=self.transformer_sample_seed)  # default: exclude_identity=True

        # Transformer-Instanz-Cache (Label -> transformer instance)
        transformer_cache = {}

        # Zähler für MLflow-Logging
        attempted_pairs = 0
        successful_pairs = 0
        # Persistent counters stored on self so process_image can increment when save succeeds
        self._images_created_count = 0
        self._juror_call_count = 0

        # Haupt-Loop: Verarbeite Bilder. Für jedes Bild wende alle Paare systematisch an.
        for batch_index, batch in enumerate(dataloader):
            for image_data in batch:
                for t1_label, t2_label in pair_labels:
                    try:
                        attempted_pairs += 1
                        # Hole Transformer-Instanzen aus Cache oder Registry
                        if t1_label not in transformer_cache:
                            try:
                                transformer_cache[t1_label] = TRANSFORMER_REGISTRY.get(t1_label)
                            except Exception:
                                transformer_cache[t1_label] = None
                        if t2_label not in transformer_cache:
                            try:
                                transformer_cache[t2_label] = TRANSFORMER_REGISTRY.get(t2_label)
                            except Exception:
                                transformer_cache[t2_label] = None

                        t1 = transformer_cache.get(t1_label)
                        t2 = transformer_cache.get(t2_label)

                        # Process image with transformer instances (can be None -> identity)
                        self.process_image(image_data, t1, t2)
                        successful_pairs += 1
                    except Exception as e:
                        logger.exception("Error processing image %s with transformers %s,%s: %s",
                                         getattr(image_data, 'image_path', None), t1_label, t2_label, e)

        # Nachverarbeitung: Logge Kennzahlen und speichere das finale COCO-File
        try:
            # Anzahl erzeugter Bilder via CocoBuilder
            self.log_metric("total_number_of_images", float(len(self.coco_builder.images)))
            # Anzahl tatsächlich erzeugter Bilddateien während dieses Runs (tatsächliche Dateischreibvorgänge)
            self.log_metric("images_created_count", float(self._images_created_count))
            # Anzahl aufgerufener Juror-Anfragen
            self.log_metric("juror_call_count", float(self._juror_call_count))

            # Juror cache metrics via public JurorClient API (graceful fallback if unsupported)
            cache_hits = cache_misses = cache_size = 0
            if self.jurorClient is not None:
                cm = self.jurorClient.get_cache_metrics() or {}
                cache_hits = int(cm.get('hits', 0))
                cache_misses = int(cm.get('misses', 0))
                cache_size = int(cm.get('size', 0))

            # log regardless (0 if not available)
            self.log_metric("juror_cache_hits", float(cache_hits))
            self.log_metric("juror_cache_misses", float(cache_misses))
            self.log_metric("juror_cache_size", float(cache_size))

            # Anzahl angewendeter Paare
            self.log_metric("applied_pairs_attempted", float(attempted_pairs))
            self.log_metric("applied_pairs_successful", float(successful_pairs))
        except Exception:
            logger.debug("Failed to log pair/image metrics to MLflow")

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
        """
        # Lade Originalbild als ndarray
        original = image_data.get_image_data("BGR")

        # Hilfsfunktion: score via jurorClient if available
        def _score(arr):
            if self.jurorClient is None:
                return None
            try:
                # Track juror invocations
                try:
                    self._juror_call_count += 1
                except Exception:
                    # defensive: if counter missing, set it
                    self._juror_call_count = getattr(self, '_juror_call_count', 0) + 1

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
            logger.info("No score found for image %s", image_data.image_relative_path)
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
        saved_successfully = False
        try:
            ImageUtils.save_image(img_t2, str(out_path))
            saved_successfully = True
            # Inkrementiere echten Schreibzähler
            try:
                self._images_created_count += 1
            except Exception:
                self._images_created_count = getattr(self, '_images_created_count', 0) + 1
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
            # Schreibe genau eine Annotation für Transformation 1: category (t1_label), sequence (vom Builder), score und initial_score
            if score_after_t1 is not None or initial_score is not None:
                if score_after_t1 is not None:
                    self.coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t1), float(
                        initial_score) if initial_score is not None else None, transformer_name=t1_label)
                else:
                    # Fallback: kein After-Score, nutze initial als score
                    if initial_score is not None:
                        self.coco_builder.add_image_transformation_score_annotation(image_id, float(initial_score),
                                                                                    float(initial_score),
                                                                                    transformer_name=t1_label)
        except Exception:
            logger.exception("Failed to add transformation 1 annotation for image_id %s", image_id)

        # Schreibe Annotationen für Transformation 2 (falls vorhanden)
        try:
            # Schreibe genau eine Annotation für Transformation 2: category (t2_label), sequence, score und initial_score = score_after_t1
            if score_after_t2 is not None or score_after_t1 is not None:
                if score_after_t2 is not None:
                    self.coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t2), float(
                        score_after_t1) if score_after_t1 is not None else None, transformer_name=t2_label)
                else:
                    if score_after_t1 is not None:
                        self.coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t1),
                                                                                    float(score_after_t1),
                                                                                    transformer_name=t2_label)
        except Exception:
            logger.exception("Failed to add transformation 2 annotation for image_id %s", image_id)

        # Füge abschliessende Bild-Level-Annotation mit initial_score und finalem Score hinzu (ohne sequence, category_id=0)
        try:
            # use score_after_t2 as final score if available, otherwise fall back to score_after_t1 or initial
            final_score = score_after_t2 if score_after_t2 is not None else (
                score_after_t1 if score_after_t1 is not None else initial_score)
            if final_score is not None:
                # CocoBuilder hat eine Helfermethode für image-level final score annotation
                try:
                    self.coco_builder.add_image_final_score_annotation(image_id, float(final_score), float(
                        initial_score) if initial_score is not None else None)
                except Exception:
                    # fallback: use generic add_image_score_annotation
                    self.coco_builder.add_image_score_annotation(image_id, float(final_score))
        except Exception:
            logger.exception("Failed to add final image-level score annotation for image_id %s", image_id)

        # Option: Logge die Scores als Metriken
        if initial_score is not None:
            try:
                self.log_metric("image_initial_score", float(initial_score))
            except Exception:
                pass
        if score_after_t2 is not None:
            try:
                self.log_metric("image_score_after_two_transformations", float(score_after_t2))
            except Exception:
                pass
