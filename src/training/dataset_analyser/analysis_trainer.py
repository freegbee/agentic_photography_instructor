import logging

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from dataset.enhanced_coco import AnnotationFileAndImagePath
from dataset.COCODataset import COCODataset
from image_acquisition.acquisition_client.dummy_acquisition_client import DummyAcquisitionClient
from training.abstract_trainer import AbstractTrainer
from training.data_loading.dataset_load_data import DatasetLoadData
from training.hyperparameter_registry import HyperparameterRegistry
from training.stable_baselines.hyperparameter.data_hyperparams import DataParams

logger = logging.getLogger(__name__)

class AnalysisTrainer(AbstractTrainer):
    def __init__(self, experiment_name: str, source_dataset_id: str, fiftyone_analysis_name: str, acquisition_client=None):
        super().__init__(experiment_name, source_dataset_id)
        self.fiftyone_analysis_name = fiftyone_analysis_name
        self.data_params: DataParams = HyperparameterRegistry.get_store(DataParams).get()
        self.data_loader = DatasetLoadData(self.data_params["dataset_id"], acquisition_client=acquisition_client)

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
        pass

    def _train_impl(self):
        logger.info(f"Starting FiftyOne analysis for experiment: {self.experiment_name}")

        # 1. FiftyOne Dataset initialisieren
        # Namen bereinigen, da FiftyOne keine Leerzeichen/Sonderzeichen mag
        dataset_name = f"analysis_{self.fiftyone_analysis_name}".replace(" ", "_")
        
        if dataset_name in fo.list_datasets():
            logger.info(f"Dataset '{dataset_name}' already exists. Deleting old version.")
            fo.delete_dataset(dataset_name)

        dataset = fo.Dataset(dataset_name)
        dataset.persistent = True  # Dataset in der lokalen MongoDB von FiftyOne speichern

        samples = []

        # 2. Daten manuell laden (Train, Test, Validation)
        # Wir nutzen COCODataset Wrapper, um Pfade aufzulösen, laden aber manuell für FiftyOne,
        # da der Standard-Importer bei Custom-COCO-Files Probleme machen kann.
        for split_name, info in self.dataset_info.items():
            logger.info(f"Loading {split_name} data manually from filesystem...")
            
            # Instanzieren des Wrappers, um Zugriff auf die geparste Struktur zu erhalten
            coco_ds = COCODataset(images_root_path=info.images_path, 
                                  annotation_file=info.annotation_file,
                                  include_transformations=False)
            
            coco_api = coco_ds.coco
            
            # Iteration über Image IDs (vermeidet das Laden der Bilddaten via cv2 in __getitem__)
            for img_id in coco_ds.image_ids:
                img_meta = coco_api.loadImgs(img_id)[0]
                file_name = img_meta['file_name']
                image_path = info.images_path / file_name
                
                # Sample erstellen
                sample = fo.Sample(filepath=str(image_path))
                sample.tags.append(split_name)
                sample["coco_id"] = img_id
                
                # Metadaten
                w = img_meta.get('width')
                h = img_meta.get('height')
                if w and h:
                    sample["width"] = w
                    sample["height"] = h

                # Annotations verarbeiten (manuell mappen)
                ann_ids = coco_api.getAnnIds(imgIds=img_id)
                anns = coco_api.loadAnns(ann_ids)
                
                detections = []
                for ann in anns:
                    # Spezifische Felder wie 'score' (Qualitätsscore) auf Sample-Ebene heben
                    # Annahme: category_id 0 ist der Score (gemäß COCODataset default)
                    if ann.get("category_id") == 0 and "score" in ann:
                        sample["juror_score"] = ann["score"]

                if detections:
                    sample["ground_truth"] = fo.Detections(detections=detections)
                
                samples.append(sample)

        # Alle Samples hinzufügen
        if samples:
            dataset.add_samples(samples)
            logger.info(f"Loaded {len(dataset)} samples total.")
        else:
            logger.warning("No samples found!")

        # 3. Analyse: Uniqueness (Qualität/Redundanz)
        logger.info("Computing uniqueness (this may take a while)...")
        fob.compute_uniqueness(dataset)

        # 4. Analyse: Similarity & Clustering (Gruppen finden)
        logger.info("Computing embeddings for similarity and clustering...")
        # Wir berechnen die Embeddings explizit, um sie für das Clustering nutzen zu können.
        # 'clip-vit-base32-torch' ist sehr gut für semantische Ähnlichkeit.
        model = foz.load_zoo_model("clip-vit-base32-torch")
        embeddings = dataset.compute_embeddings(model)

        try:
            from sklearn.cluster import KMeans
            # Anzahl Cluster: Heuristik (z.B. 10 oder 20) oder abhängig von der Datenmenge.
            n_clusters = 15
            logger.info(f"Computing {n_clusters} clusters using KMeans...")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Cluster-ID als String speichern (besser für Filterung in der UI)
            dataset.set_values("cluster_id", [str(l) for l in cluster_labels])
            logger.info("Clustering complete. Field 'cluster_id' added.")
        except ImportError:
            logger.warning("scikit-learn not installed. Skipping clustering. (pip install scikit-learn)")

        # 5. Visualisierung (Scatterplot) basierend auf den Embeddings
        logger.info("Computing visualization (UMAP)...")
        fob.compute_visualization(dataset, embeddings=embeddings, brain_key="image_embeddings", method="umap")

        logger.info(f"Analysis complete. Dataset '{dataset_name}' is saved and ready for inspection via 'fiftyone app launch'.")
        logger.info(dataset.summary())

    def _evaluate_impl(self):
        pass