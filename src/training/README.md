# Training-Modul — Überblick

Zweck
- Kurze Dokumentation der zentralen (abstrakten) Klassen im Trainings-Subsystem und Beschreibung des typischen Trainingsablaufs. Zielgruppe: Entwickler*innen, die neue Trainingsvarianten implementieren oder bestehende erweitern wollen.

Kurzübersicht zentraler abstrakter Klassen
- `AbstractTrainer` (in `abstract_trainer.py`)
  - Verantwortlich für die Orchestrierung eines Trainingslaufs.
  - Wichtige Methoden: `run_training(run_name=None)`, `_run_training_pipeline()`, `load_data()`, `preprocess()`, `train()`, `evaluate()`.
  - Implementierende Klassen müssen die abstrakten Implementierungen bereitstellen: `_load_data_impl()`, `_preprocess_impl()`, `_train_impl()`, `_evaluate_impl()`.
  - Integriert MLflow-Tracking (Start/End run, automatische Zeitmessung via Dekoratoren) und Hilfsmethoden zum Loggen (`log_metric`, `log_batch_metrics`, `log_artifact`).

- `AbstractLoadData[RESULT]` (in `data_loading/abstract_load_data.py`)
  - Basisklasse für Datenlade-Operationen (z. B. Herunterladen oder Sicherstellen von Datensätzen).
  - Erwartete Methoden: `_load_data_impl()` und `get_result()`; eine `load_data()`-Methode ist vorhanden und mlflow-annotiert.
  - Beispiel-Implementierung: `DatasetLoadData` (in `data_loading/dataset_load_data.py`).

- `AbstractPreprocessor[RESULT]` (in `preprocessing/abstract_preprocessor.py`)
  - Repräsentiert einen einzelnen Vorverarbeitungsschritt (z. B. Kopieren/Resize, Scoring, Erzeugen von Annotationen).
  - Erwartete Methoden: `_preprocess_impl()` und `get_preprocessing_result()`; `preprocess()` ist mit Laufzeitmessung versehen.
  - Beispiele: `CopyAndResizePreprocessor`, `AnnotationFileCreator`, `ScorePreprocessor`.

- `HyperparameterStore` / `HyperparameterRegistry` (in `hyperparameter_registry.py`)
  - Zentraler Ort zur Speicherung und einfachen Validierung von Hyperparametern (verwendet `TypedDict` als Schema).
  - Nutzbar, um parameterisierte Stores (z. B. `TrainingExecutionParams`, `DataParams`) global verfügbar und typgesichert zu machen.

- `MlflowHelper` (in `mlflow_helper.py`)
  - Singleton-Wrapper um `mlflow`-Funktionen: Run-Management, Metrics/Params/Artifacts logging.
  - Unterstützt lokalen Log-Modus für Debugging.

Hinweis: Weitere konkrete Komponenten und RL-spezifische Trainer finden sich unter `rl_training/` (z. B. `RlTrainer`, `training_params.py`, `entrypoint.py`).

Typischer Trainingsablauf (konzeptionell)
1. Konfiguration / Hyperparameter
   - Hyperparameter über `HyperparameterRegistry` setzen oder laden (z. B. `TrainingExecutionParams`, `DataParams`).
2. Experiment initialisieren
   - `AbstractTrainer.__init__` initialisiert MLflow-Tracking, run/experiment Namen und Hilfsobjekte.
3. Daten laden
   - `trainer.load_data()` → delegiert auf `_load_data_impl()` einer konkreten `AbstractLoadData`-Implementierung.
   - Beispiel: `DatasetLoadData` lädt Dataset-Konfiguration und sorgt dafür, dass Bilder vorhanden sind.
4. Preprocessing
   - `trainer.preprocess()` → führt mehrere `AbstractPreprocessor`-Schritte aus (Kopieren/Resize, Erzeugen von Annotationen, Scoring, Degradierung & Splitten).
   - Jeder Schritt ist verantwortlich für eigene Artefakte (z. B. annotations.json) und loggt Metriken/Artefakte via MLflow.
5. Training
   - `trainer.train()` → enthält die eigentliche Trainingsschleife (Episoden/Batch-Loop/Optimierung). In RL-Varianten wäre hier die Policy-/Agent-Optimierung implementiert.
6. Evaluation
   - `trainer.evaluate()` → Auswertung des trainierten Modells (Metriken, ggf. Visualisierungen).
7. Checkpoints & Artefakte
   - Modell-Checkpoints, Metriken, Plots und Annotationen werden als Artefakte ins Tracking-System hochgeladen.
8. Abschluss
   - `MlflowHelper.end_run()` wird aufgerufen; Laufzeit und Summaries sind in MLflow verfügbar.

Erweiterungspunkte / Implementierungs-Hooks
- Trainer erweitern: Neue `Trainer`-Klasse von `AbstractTrainer` ableiten und die vier `_..._impl`-Methoden implementieren.
- Neue Datenlade-Schritte: `AbstractLoadData` implementieren und Ergebnisse über `get_result()` bereitstellen.
- Zusätzliche Preprocessors: `AbstractPreprocessor` implementieren für weitere Vorverarbeitungen (z. B. Augmentations-Pipelines).
- Hyperparameter: Neue `TypedDict`-Definitionen anlegen und über `HyperparameterRegistry.get_store(...)` verfügbar machen.
- MLflow: Nutzen der vorhandenen `mlflow_logging` Dekoratoren und `MlflowHelper` für konsistentes Tracking.
- Checkpoint-Mechanismus: (falls nicht vorhanden) eigene Checkpointer-Klasse entwerfen, die beim Training periodisch Modelle speichert und als Artefakt loggt.

Best Practices / Hinweise
- Trenne konzeptionell: Datenakquisition ↔ Preprocessing ↔ Training ↔ Evaluation.
- Kleine, testbare Preprocessing-Schritte: jede Klasse gibt ein genau beschriebenes Ergebnisobjekt zurück.
- Verwende `HyperparameterRegistry` für reproduzierbare Konfigurationen und um magic strings zu vermeiden.
- Logge wichtige Artefakte und Metriken frühzeitig (z. B. generierte annotation files, Anzahl Bilder, Scoring-Statistiken).
- Achte auf Umgebungsvariablen (z. B. `IMAGE_VOLUME_PATH`, `IMAGE_ACQUISITION_SERVICE_URL`), die von Preprocessors und Loadern verwendet werden.

Konkretes Beispiel (wo im Repo)
- Einstiegspunkt der RL-Variante: `src/training/rl_training/entrypoint.py` — hier werden Hyperparameter gesetzt und `RlTrainer().run_training()` aufgerufen.
- Trainer-Implementierung: `src/training/rl_training/rl_trainer.py` zeigt, wie `AbstractTrainer` verwendet wird und wie Preprocessors zusammengesetzt werden.

Weiteres / ToDo
- Ergänzen von Checkpointer-Abstracts und einer Dokumentation zur Hardware-/Device-Strategie (CPU/GPU) falls Training-Implementierungen dies benötigen.

----
(Automatisch erstellt: kurze, technikerfreundliche Übersicht. Für API-Details oder Signaturen siehe die referenzierten Module im `src/training`-Ordner.)

