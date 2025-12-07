import logging
import os
import random
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from pyinstrument import Profiler
from torch.utils.data import DataLoader

from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from dataset.COCODataset import COCODataset
from dataset.TopKSampler import TopKSampler
from dataset.Utils import Utils as DatasetUtils
from experiments.shared.BatchImageMetricAccumulator import BatchImageMetricAccumulator
from experiments.shared.PhotographyExperiment import PhotographyExperiment
from experiments.shared.Utils import Utils as SharedUtils
# lokale Imports für Verarbeitung
from juror_client import JurorClient
from transformer import generate_transformer_pairs
from utils.AsyncImageSaver import AsyncImageSaver
from utils.CocoBuilder import CocoBuilder
from utils.ConfigLoader import ConfigLoader
from utils.ImageUtils import ImageUtils
from utils.Registries import TRANSFORMER_REGISTRY

logger = logging.getLogger(__name__)


class DoubleTransformationExperiment(PhotographyExperiment):
    def __init__(self, experiment_name: str,
                 target_directory_root: str = "double_transformed",
                 run_name: Optional[str] = None,
                 source_dataset_id: str = "single_image",
                 max_images: Optional[int] = None,
                 seed: int = 42,
                 transformer_sample_size: Optional[int] = None,
                 transformer_sample_seed: Optional[int] = None,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 io_workers: Optional[int] = None,
                 max_queue_size: Optional[int] = None,
                 save_executor_type: str = "process",
                 instrumentation: bool = False,
                 split_ratios: Optional[Tuple[float, float, float]] = None):
        super().__init__(experiment_name)
        self.run_name = run_name
        self.source_dataset_id = source_dataset_id
        self.max_images = max_images
        self.target_directory_root = Path(os.environ.get("IMAGE_VOLUME_PATH", ".")) / target_directory_root
        self.seed = seed
        self.random = random.Random(seed)
        # Split-Konfiguration (train/val/test) optional
        self.split_ratios: Optional[Tuple[float, float, float]] = split_ratios
        # separater Zufallsgenerator für Split-Shuffling (stabil durch seed)
        self.split_random = random.Random(seed)
        # Performance tuning: batch_size for DataLoader
        self.batch_size = int(batch_size)
        # Number of DataLoader workers
        self.num_workers = int(num_workers)
        # Einstellungen für Transformer-Paare
        self.transformer_sample_size = transformer_sample_size
        self.transformer_sample_seed = transformer_sample_seed
        self.coco_builder = None
        # ensure attributes exist so other methods can reference them directly
        self.jurorClient = None
        logger.info("Initialized DoubleTransformationExperiment: %s", experiment_name)

        # Async saver configuration (may be None -> compute defaults in _run_impl if absent)
        self.io_workers = int(io_workers) if io_workers is not None else None
        self.max_queue_size = int(max_queue_size) if max_queue_size is not None else None
        self.save_executor_type = save_executor_type

        # Accumulator für globale Aggregation
        self._global_initial_scores = []
        self._global_final_scores = []
        self._global_changes = []
        self._global_images_created = 0

        if instrumentation:
            self.profiler = Profiler()
        else:
            self.profiler = None

    def configure(self, config: dict):
        pass

    def _get_tags_for_run(self) -> Dict[str, Any]:
        # Tag run with dataset_id and performance indicator (we use num_workers tuning)
        return {"dataset_id": self.source_dataset_id, "performance": "async_saver"}

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
            self.log_param("batch_size", str(self.batch_size))
            self.log_param("num_workers", str(self.num_workers))
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
            # Note: 'sampler' is assigned hier für zukünftige Verwendung in der Bildauswahl.
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
            self.jurorClient = JurorClient(use_local=True)
            logger.debug("Juror client ready to process images. Using service %s", self.jurorClient._service)
        except Exception:
            logger.exception("Unable to initialize JurorClient; proceeding without juror (scores will be None)")
            self.jurorClient = None

        # DataLoader aufbauen (TopKSampler falls vorhanden)
        # Configure DataLoader using configured batch_size and num_workers
        dataloader_kwargs = dict(batch_size=self.batch_size, collate_fn=DatasetUtils.collate_keep_size)
        if self.num_workers > 0:
            dataloader_kwargs.update({"num_workers": self.num_workers, "pin_memory": True, "persistent_workers": True})
        if sampler is not None:
            dataloader = DataLoader(source_dataset, sampler=sampler, **dataloader_kwargs)
        else:
            dataloader = DataLoader(source_dataset, **dataloader_kwargs)

        # Asynchronen Image-Saver initialisieren. Nutze Werte aus Konstruktor falls vorhanden,
        # sonst berechne sinnvolle Defaults wie zuvor.
        try:
            if self.io_workers is not None:
                io_workers = max(1, int(self.io_workers))
            else:
                io_workers = max(2, min(16, int(self.num_workers or 4)))

            if self.max_queue_size is not None:
                max_pending = max(1, int(self.max_queue_size))
            else:
                max_pending = max(128, int(self.batch_size * max(1, self.num_workers) * 8))

            saver = AsyncImageSaver(max_workers=io_workers, max_queue_size=max_pending,
                                    executor_type=self.save_executor_type)
            logger.debug("Started AsyncImageSaver max_workers=%s max_pending=%s", io_workers, max_pending)
            # Log saver config to MLflow as params
            try:
                self.log_param("async_io_workers", str(io_workers))
                self.log_param("async_max_queue_size", str(max_pending))
                # log executor type choice
                self.log_param("save_executor_type", str(self.save_executor_type))
            except Exception:
                logger.debug("Failed to log async saver params to MLflow")
        except Exception:
            logger.exception("Failed to start AsyncImageSaver; falling back to synchronous saves")
            saver = None

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

        # Accumulator für Batch-Metriken
        batch_accumulator = BatchImageMetricAccumulator()

        # Wenn Splits konfiguriert sind, generiere und verarbeite sie separat
        if self.split_ratios is not None:
            logger.info("split_ratios configured, generating splits instead of processing full dataset in single run")
            attempted_pairs, successful_pairs = self._generate_and_process_splits(source_dataset, pair_labels, transformer_cache, saver, attempted_pairs, successful_pairs)
            # After split processing, perform final logging and exit run
            try:
                self.log_metric("total_number_of_images", float(len(self.coco_builder.images)))
                self.log_metric("images_created_count", float(self._images_created_count))
                self.log_metric("juror_call_count", float(self._juror_call_count))
            except Exception:
                logger.debug("Failed to log post-split metrics")

            # Save top-level coco (may be empty or aggregate)
            coco_file_path = self.target_directory_root / "annotations.json"
            try:
                self.coco_builder.save(str(coco_file_path))
                self.log_artifact(local_path=str(coco_file_path))
            except Exception:
                logger.debug("Failed to save or log root annotations after split processing")

            # Shutdown saver
            try:
                if saver is not None:
                    saver.shutdown(wait=True)
            except Exception:
                logger.exception("Error shutting down AsyncImageSaver after split processing")

            # Return after split processing
            logger.info("Finished split processing in DoubleTransformationExperiment")
            return

        if self.profiler is not None:
            self.profiler.start()

        # Haupt-Loop: Verarbeite Bilder. Für jedes Bild wende alle Paare systematisch an.
        for batch_index, batch in enumerate(dataloader):
            # Starte Batch-Accumulator
            batch_accumulator.reset()
            batch_accumulator.start(batch_index)

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
                        # Pass in saver so write is executed asynchronously with backpressure
                        # process_image will update coco_builder and counters; it returns tuple of (initial, final)
                        init_score, final_score = self.process_image(image_data, self.coco_builder, t1, t2, destination_root_dir=self.target_directory_root, saver=saver)

                        # add to batch accumulator
                        batch_accumulator.add_image(init_score, final_score)

                        successful_pairs += 1
                    except Exception as e:
                        logger.exception("Error processing image %s with transformers %s,%s: %s",
                                         getattr(image_data, 'image_path', None), t1_label, t2_label, e)

            # Batch abgeschlossen
            batch_accumulator.stop()
            # Logge Batch-Metriken
            try:
                self.log_batch_metrics(batch_accumulator.compute_metrics(), batch_index)
            except Exception:
                logger.exception("Failed to log batch metrics for batch %s", batch_index)

            # Update globale Aggregation
            try:
                self._global_images_created += batch_accumulator.number_of_images
                # extend lists while ignoring None
                for v in batch_accumulator.initial_scores:
                    self._global_initial_scores.append(v)
                for v in batch_accumulator.final_scores:
                    self._global_final_scores.append(v)
                for v in batch_accumulator.changes:
                    self._global_changes.append(v)
            except Exception:
                logger.exception("Failed to update global metrics from batch %s", batch_index)

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
        finally:
            if self.profiler is not None:
                self.profiler.stop()
                profile_output_path = self.target_directory_root / "profiling.html"
                try:
                    with open(profile_output_path, "w") as f:
                        f.write(self.profiler.output_html())
                    # log profiling output as artifact
                    self.log_artifact(local_path=str(profile_output_path))
                except Exception:
                    logger.exception("Failed to write or log profiling output to %s", profile_output_path)

        # Nachverarbeitung: speichere das finale COCO-File
        coco_file_path = self.target_directory_root / "annotations.json"
        self.coco_builder.save(str(coco_file_path))
        try:
            self.log_artifact(local_path=str(coco_file_path))
        except Exception:
            logger.debug("log_artifact failed or mlflow not reachable during final save")

        # Warte auf alle async IO-Tasks bevor finale COCO-Datei gespeichert wird
        try:
            if saver is not None:
                saver.shutdown(wait=True)
        except Exception:
            logger.exception("Error shutting down AsyncImageSaver")

        # Berechne globale Metriken (average/min/max for initial and final scores and change)
        try:
            def safe_stats(vals):
                if not vals:
                    return None, None, None
                avg = sum(vals) / len(vals)
                return avg, min(vals), max(vals)

            avg_init, min_init, max_init = safe_stats(self._global_initial_scores)
            avg_final, min_final, max_final = safe_stats(self._global_final_scores)
            avg_change, min_change, max_change = safe_stats(self._global_changes)

            # log global metrics
            if avg_init is not None:
                self.log_metric("global_average_initial_score", float(avg_init))
                self.log_metric("global_min_initial_score", float(min_init))
                self.log_metric("global_max_initial_score", float(max_init))
            if avg_final is not None:
                self.log_metric("global_average_final_score", float(avg_final))
                self.log_metric("global_min_final_score", float(min_final))
                self.log_metric("global_max_final_score", float(max_final))
            if avg_change is not None:
                self.log_metric("global_average_change_score", float(avg_change))
                self.log_metric("global_min_change_score", float(min_change))
                self.log_metric("global_max_change_score", float(max_change))

            # log aggregate counts
            self.log_metric("global_images_created_count", float(self._global_images_created))
        except Exception:
            logger.exception("Failed to compute or log global aggregated metrics")

        logger.info("Finished processing images in DoubleTransformationExperiment")

    def process_image(self, image_data, coco_builder: CocoBuilder, transformer1, transformer2, destination_root_dir: Path, saver: Optional["AsyncImageSaver"] = None):
        """
        Process a single image with two transformers, save the resulting image and register it in the CocoBuilder.
        If `transformer1` or `transformer2` is None, treat them as identity transformers.

        Returns (initial_score, final_score) so callers (batch aggregator) can record metrics.
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
            logger.info("No score found für image %s", image_data.image_relative_path)
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
        out_path = destination_root_dir / "images" / out_filename

        # Speichern: entweder asynchron enqueuen (saver) oder synchron schreiben
        saved_successfully = False
        try:
            if saver is not None:
                fut = saver.save_async(img_t2, str(out_path))

                # optional: attach a done callback that increments counter on success
                def _on_done(f):
                    try:
                        if f.result():
                            try:
                                self._images_created_count += 1
                            except Exception:
                                self._images_created_count = getattr(self, '_images_created_count', 0) + 1
                    except Exception:
                        logger.exception("Error in async save done callback für %s", out_path)

                try:
                    fut.add_done_callback(_on_done)
                except Exception:
                    # If adding callback fails, fall back to not tracking
                    pass
            else:
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
        image_id = coco_builder.add_image(out_filename, w, h)

        # Schreibe Annotationen für Transformation 1 (falls vorhanden)
        try:
            # Schreibe genau eine Annotation für Transformation 1: category (t1_label), sequence (vom Builder), score und initial_score
            if score_after_t1 is not None or initial_score is not None:
                if score_after_t1 is not None:
                    coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t1), float(
                        initial_score) if initial_score is not None else None, transformer_name=t1_label)
                else:
                    # Fallback: kein After-Score, nutze initial als score
                    if initial_score is not None:
                        coco_builder.add_image_transformation_score_annotation(image_id, float(initial_score),
                                                                                    float(initial_score),
                                                                                    transformer_name=t1_label)
        except Exception:
            logger.exception("Failed to add transformation 1 annotation for image_id %s", image_id)

        # Schreibe Annotationen für Transformation 2 (falls vorhanden)
        try:
            # Schreibe genau eine Annotation für Transformation 2: category (t2_label), sequence, score und initial_score = score_after_t1
            if score_after_t2 is not None or score_after_t1 is not None:
                if score_after_t2 is not None:
                    coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t2), float(
                        score_after_t1) if score_after_t1 is not None else None, transformer_name=t2_label)
                else:
                    if score_after_t1 is not None:
                        coco_builder.add_image_transformation_score_annotation(image_id, float(score_after_t1),
                                                                                    float(score_after_t1),
                                                                                    transformer_name=t2_label)
        except Exception:
            logger.exception("Failed to add transformation 2 annotation for image_id %s", image_id)

        # Füge abschliessende Bild-Level-Annotation mit initial_score und finalem Score hinzu (ohne sequence, category_id=0)
        try:
            # use score_after_t2 as final score if available, otherwise fall back to score_after_t1 oder initial
            final_score = score_after_t2 if score_after_t2 is not None else (
                score_after_t1 if score_after_t1 is not None else initial_score)
            if final_score is not None:
                # CocoBuilder hat eine Helfermethode für image-level final score annotation
                try:
                    coco_builder.add_image_final_score_annotation(image_id, float(final_score), float(
                        initial_score) if initial_score is not None else None)
                except Exception:
                    # fallback: use generic add_image_score_annotation
                    coco_builder.add_image_score_annotation(image_id, float(final_score))
        except Exception:
            logger.exception("Failed to add final image-level score annotation for image_id %s", image_id)

        # return scores for batch aggregator
        return initial_score, (
            score_after_t2 if score_after_t2 is not None else (score_after_t1 if score_after_t1 is not None else None))

    def _create_subset_dataset(self, source_dataset: COCODataset, indices: List[int]) -> list:
        """Create a list-based subset of the COCODataset for given indices."""
        return [source_dataset[i] for i in indices]

    def _generate_and_process_splits(self, source_dataset: COCODataset, pair_labels, transformer_cache, saver, attempted_pairs_start=0, successful_pairs_start=0):
        """Erzeuge Splits (train/val/test) und verarbeite jeden Split getrennt.
        Gibt (attempted_pairs, successful_pairs) zurück (eingehende Startwerte addiert).
        """
        # compute sizes
        train_ratio, val_ratio, test_ratio = self.split_ratios
        total_images = len(source_dataset)

        indices = list(range(total_images))
        self.split_random.shuffle(indices)

        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        splits = [("train", train_indices), ("val", val_indices), ("test", test_indices)]

        # log split info
        try:
            self.log_param("split_ratios", f"{train_ratio}/{val_ratio}/{test_ratio}")
            self.log_param("train_size", len(train_indices))
            self.log_param("val_size", len(val_indices))
            self.log_param("test_size", len(test_indices))
        except Exception:
            logger.debug("Failed to log split params to MLflow")

        attempted_pairs = attempted_pairs_start
        successful_pairs = successful_pairs_start

        for split_name, split_indices in splits:
            logger.info("Generating %s split with %d images", split_name, len(split_indices))

            split_dir = self.target_directory_root / split_name
            try:
                split_dir.mkdir(parents=True, exist_ok=True)
                (split_dir / "images").mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.exception("Failed to create split directories for %s", split_dir)

            split_coco_builder = CocoBuilder(f"{self.source_dataset_id}_{split_name}")
            split_coco_builder.set_description(f"Double transformation split {split_name} derived from {self.source_dataset_id}")

            subset = self._create_subset_dataset(source_dataset, split_indices)

            for batch_start in range(0, len(subset), self.batch_size):
                batch = subset[batch_start:batch_start + self.batch_size]
                for image_data in batch:
                    for t1_label, t2_label in pair_labels:
                        try:
                            # ensure cached transformer instances
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

                            # swap builders
                            orig_builder = getattr(self, 'coco_builder', None)
                            try:
                                self.coco_builder = split_coco_builder
                                init_score, final_score = self.process_image(image_data, split_coco_builder, t1, t2, destination_root_dir=split_dir, saver=saver)
                            finally:
                                self.coco_builder = orig_builder

                            successful_pairs += 1
                        except Exception as e:
                            logger.exception("Error processing image %s in split %s with transformers %s,%s: %s",
                                             getattr(image_data, 'image_path', None), split_name, t1_label, t2_label, e)

            # Save split annotations
            split_coco_file = split_dir / "annotations.json"
            split_coco_builder.save(str(split_coco_file))
            try:
                self.log_artifact(local_path=str(split_coco_file))
            except Exception:
                logger.debug("Failed to log split coco artifact for %s", split_coco_file)

            self.log_metric(f"{split_name}_num_images", float(len(split_coco_builder.images)))

        return attempted_pairs, successful_pairs
