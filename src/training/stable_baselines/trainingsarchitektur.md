# Trainingsarchitektur & Performance-Optimierung
 
 Dieses Dokument beschreibt die Architektur der Trainingspipeline für den "Agentic Photography Instructor". Der Fokus liegt auf der **Parallelisierung** der Umgebungen (Environments) und der **Entkopplung** der GPU-intensiven Juror-Bewertung mittels eines Worker-Pools.
 
 ## Übersicht
 
 Das System nutzt `Stable Baselines 3` (SB3) für das Reinforcement Learning. Da die Simulation (Bildbearbeitung) CPU-lastig und die Bewertung (Juror) GPU-lastig ist, wurde eine Architektur gewählt, die beide Ressourcen unabhängig voneinander skalieren lässt.
 
 ### Kernkomponenten
 
 1.  **Trainer (Main Process):** Orchestriert das Training, hält das PPO-Modell und sammelt Daten.
 2.  **Vector Environments (CPU):** Mehrere parallele Prozesse (`SubprocVecEnv`), die jeweils eine Instanz der Simulation (`ImageTransformEnv`) ausführen.
 3.  **Juror Worker Pool (GPU):** Eine feste Anzahl von Prozessen, die das Juror-Modell im VRAM halten und Bewertungsanfragen abarbeiten.
 
 ---
 
 ## Technische Details
 
 ### 1. Parallelisierung & Threading
 
 Um "Thread Oversubscription" zu vermeiden (wenn zu viele Prozesse jeweils zu viele Threads starten und das OS nur noch Context-Switching betreibt), werden folgende Maßnahmen getroffen:
 
 *   **Environment Variablen:** `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` etc. werden *vor* dem Import von Numpy/Torch gesetzt.
 *   **OpenCV:** `cv2.setNumThreads(0)` wird explizit in jedem Worker-Prozess aufgerufen.
 *   **Ziel:** Jeder Environment-Prozess nutzt exakt **einen** CPU-Kern. Dies erlaubt eine lineare Skalierung über die Anzahl der Environments (z.B. 8 Envs auf 16 Cores).
 
 ### 2. Juror Worker Pool (Inference Server Pattern)
 
 Da das Juror-Modell viel VRAM benötigt, können wir nicht in jedem Environment eine eigene Instanz laden. Stattdessen nutzen wir ein Client-Server-Modell über `multiprocessing`:
 
 *   **Worker:** Eine begrenzte Anzahl (z.B. 5) von Prozessen lädt das Modell auf die GPU.
 *   **Queue:** Eine zentrale `request_queue` verteilt Arbeit an freie Worker.
 *   **Reply:** Jedes Environment besitzt eine eigene `reply_queue`, um sicherzustellen, dass die Antwort den richtigen Empfänger erreicht.
 
 ### 3. Shared Memory (Zero-Copy Data Transfer)
 
 Die Übertragung von Bildern (384x384x3) über Standard-Queues (Pickling) ist langsam. Wir nutzen `multiprocessing.shared_memory`:
 
 1.  **Environment:** Erstellt einen Shared Memory Block, schreibt das Bild-Array hinein.
2.  **Queue:** Sendet nur den **Namen** des Speicherblocks und Metadaten (Shape, Dtype) an den Worker.
 3.  **Worker:** Öffnet den Shared Memory Block, liest das Bild (Zero-Copy View), berechnet den Score.
 4.  **Cleanup:** Environment löscht den Shared Memory Block nach Erhalt der Antwort.
 
 Dies reduziert den Overhead der Datenübertragung massiv.
 
 ### 4. Serialisierung & Cloudpickle
 
 `SubprocVecEnv` nutzt `cloudpickle`, um Environments an neue Prozesse zu senden. Dies führt zu Problemen, wenn Objekte nicht serialisierbar sind (z.B. `weakref`, offene Sockets, MLflow Clients).
 
 **Lösungen:**
 *   **Static Method Factory:** Die Erstellung des Environments erfolgt über eine statische Methode `_init_env_static`. Dadurch wird verhindert, dass `self` (die Trainer-Instanz mit unpicklable Objekten) versehentlich in die Closure gepackt wird.
 *   **Manager Queues:** Auf Windows/macOS (Spawn-Methode) können native `multiprocessing.Queue` Objekte nicht einfach gepickled werden. Wir nutzen `multiprocessing.Manager().Queue()`, welche Proxy-Objekte sind und prozessübergreifend funktionieren.
 *   **Parameter-Extraktion:** Nicht-serialisierbare Objekte (wie der Pool selbst) werden nicht übergeben; stattdessen werden nur die benötigten Queues extrahiert und übergeben.
 
 ### 5. Random Seeds
 
 Standardmäßig nutzen Factory-Funktionen oft denselben Seed.
 *   **Implementierung:** Wir iterieren im Trainer über die Anzahl der Environments und inkrementieren den Seed (`base_seed + i`).
 *   **Lambda Binding:** `lambda s=env_seed: ...` bindet den aktuellen Wert des Seeds fest an die Factory-Funktion für das jeweilige Environment.
 
 ---
 
 ## Konfiguration
 
 Die Steuerung erfolgt zentral in `entrypoint.py`:
 
 *   `OPTIMIZE_FOR_MULTIPROCESSING = True`: Aktiviert `SubprocVecEnv`, Worker Pool und Thread-Limitierung.
 *   `NUM_VECTOR_ENVS`: Anzahl der CPU-Prozesse (sollte < Anzahl CPU Cores sein).
 *   `num_juror_workers`: Anzahl der GPU-Prozesse (limitiert durch VRAM).
 
 Die Batch-Größe (`n_steps`) wird dynamisch berechnet, um unabhängig von der Anzahl der Environments eine konstante `total_rollout_size` (z.B. 4000 Steps pro Update) zu gewährleisten