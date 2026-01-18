# Projektdokumentation: Agentic Photography Instructor

Diese Dokumentation bietet einen technischen Überblick über die Codebasis des Projekts "Agentic Photography Instructor". Das System ist darauf ausgelegt, mittels Reinforcement Learning (RL) einen Agenten zu trainieren, der autonom Bildbearbeitungswerkzeuge (Transformers) anwendet, um die ästhetische Qualität von Fotos zu verbessern.

Die Implementierung zeichnet sich durch eine tiefe Integration von modernen ML-Frameworks (PyTorch, Stable Baselines 3), umfangreichem Experiment-Tracking (MLflow) und spezialisierten Tools zur Datenanalyse und Visualisierung aus.

---

## 1. Modulübersicht

### 1.1 Reinforcement Learning Core (`src/training/stable_baselines`)
Dies ist das Herzstück der Anwendung, in dem die Trainingslogik implementiert ist. Es baut auf `Stable Baselines 3` (SB3) und `Gymnasium` auf.

*   **Trainer & Entrypoints:** Implementierung verschiedener Trainings-Szenarien (PPO, DQN) über zentrale Einstiegspunkte (`entrypoint_ppo.py`, `entrypoint_dqn.py`).
*   **Custom Environments:** Definition von Gym-Umgebungen, in denen der Agent Bilder manipuliert. Die Umgebung verwaltet den State (aktuelles Bild) und berechnet Rewards basierend auf der ästhetischen Verbesserung.
*   **Modell-Architektur:**
    *   Eigene **Feature Extractors** (`models/base_feature_extractor.py`) basierend auf ResNet18 und ResNet50, um visuelle Features aus den Bildern für den RL-Agenten zu extrahieren.
    *   Unterstützung für verschiedene Backbone-Konfigurationen (Frozen/Unfrozen Layers).
*   **Reward Engineering:** Ein komplexes System aus Reward-Strategien (`rewards/reward_strategies.py`), das verschiedene Ansätze unterstützt:
    *   *Score Difference*: Belohnung für die Differenz zum vorherigen Score.
    *   *Step Penalty*: Bestrafung für lange Bearbeitungssequenzen.
    *   *Success Bonus*: Zusätzliche Belohnung bei Erreichen eines Zielwerts.
*   **Hyperparameter-Optimierung:** Integration von **Optuna** (`training/optimize_ppo.py`) zur automatisierten Suche nach optimalen Hyperparametern (Learning Rate, Batch Size, Gamma, etc.) mit Pruning-Strategien.
*   **Multiprocessing Architektur:** Um die CPU-lastige Bildverarbeitung von der GPU-lastigen Bewertung zu entkoppeln, wurde ein **Worker-Pool-System** implementiert (`JurorWorkerPool`), das über Shared Memory kommuniziert.

### 1.2 The Juror - Aesthetic Assessment (`src/juror`)
Dieses Modul fungiert als "Kritiker" im System. Es bewertet die Qualität von Bildern.

*   **Modell:** Verwendung eines Transformer-basierten Modells (SigLIP / `siglib_v2_5.py`) zur Vorhersage ästhetischer Scores.
*   **Inference:** Kapselung der PyTorch-Inferenzlogik in einer `Juror`-Klasse, die Bilder (Numpy Arrays) entgegennimmt und Scores (0-10) zurückgibt.
*   **Hardware-Support:** Automatische Erkennung und Nutzung von CUDA (Nvidia) oder MPS (Apple Silicon) zur Beschleunigung.

### 1.3 Data Pipeline & Preprocessing (`src/training/preprocessing`, `src/dataset`)
Ein umfangreiches Framework zur Vorbereitung und Verwaltung der Trainingsdaten.

*   **COCO Integration:** Eigene Implementierung eines `CocoBuilder` und `COCODataset` Wrappers, um Bilddaten und Metadaten (Scores, Historie) im standardisierten COCO-Format zu verwalten.
*   **Preprocessing Pipeline:** Eine Kette von `AbstractPreprocessor`-Implementierungen:
    1.  **Copy & Resize:** Standardisierung der Bildgrößen.
    2.  **Annotation Creation:** Initialisierung der Metadaten.
    3.  **Scoring:** Initiale Bewertung aller Bilder durch den Juror.
    4.  **Degradation & Split:** Künstliche Verschlechterung der Bilder (z.B. Rauschen, Unschärfe) mittels `DegradingFunctionFactory`, um Trainingsfälle zu generieren, und Aufteilung in Train/Test/Val-Sets.
*   **Dataset Analysis:** Integration von **FiftyOne** (`src/training/dataset_analyser`) zur visuellen Inspektion, Cluster-Analyse und Qualitätsprüfung der Datensätze.

### 1.4 Web Applications (`src/webapps`)
Tools zur Interaktion und Analyse durch den Benutzer.

*   **Annotations Browser:** Eine **FastAPI**-Anwendung mit Frontend (HTML/JS), die es ermöglicht, die generierten COCO-Datensätze zu durchsuchen.
    *   Features: Filterung nach Kategorien, Sortierung nach Scores/Verbesserung, Anzeige von Thumbnails und Metadaten.
    *   Asynchrones Laden von großen JSON-Dateien im Hintergrund.

### 1.5 Infrastructure & Utilities (`src/utils`, `src/training`)
Allgemeine Hilfsfunktionen und Infrastruktur-Code.

*   **MLflow Integration:** Ein zentraler `MlflowHelper` (`src/training/mlflow_helper.py`) abstrahiert das Logging von Metriken, Parametern und Artefakten.
*   **Visual Logging:** Generierung von Videos (`artifact_video_gen`) und Mosaik-Bildern (`VisualSnapshotLogger`), um den Lernfortschritt des Agenten über Episoden hinweg sichtbar zu machen.
*   **Konfiguration:** Typ-sicheres Konfigurationsmanagement über eine `HyperparameterRegistry`.
*   **Async I/O:** `AsyncImageSaver` zur performanten, nicht-blockierenden Speicherung von Bildern während der Generierung.

---

## 2. Feature Highlights

*   **Zero-Copy Data Transfer:** Implementierung von Shared Memory Mechanismen für den Datenaustausch zwischen Environment-Prozessen und Juror-Prozessen, um den Overhead bei der Übertragung großer Bild-Arrays zu minimieren.
*   **Dynamische Reward-Shaping:** Die Reward-Funktionen können dynamisch konfiguriert werden (z.B. Sigmoid-basierte Boni), um das Lernverhalten des Agenten fein zu steuern.
*   **Wiederherstellbarkeit:** Das System speichert den kompletten Zustand (inkl. Seeds und Konfigurationen) in MLflow, um Experimente reproduzierbar zu machen.
*   **Multi-Step Reasoning:** Unterstützung für Szenarien, in denen der Agent mehrere Schritte planen muss, bevor er eine Belohnung erhält (`entrypoint_multi_step.py`).

---

## 3. Architektur-Visualisierung

### 3.1 Trainings-Loop & Komponenten
Dieses Diagramm zeigt das Zusammenspiel der Hauptkomponenten während einer Trainings-Iteration.

### 3.2 Preprocessing Pipeline
Der Ablauf der Datenaufbereitung vor dem eigentlichen Training.


### 3.3 System Context & Web Tools
Überblick über die Einbettung der Web-Tools zur Analyse.
