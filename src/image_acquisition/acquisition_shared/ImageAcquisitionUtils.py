import logging

logger = logging.getLogger(__name__)

class ImageAcquisitionUtils:
    @staticmethod
    def download_file(remote_url: str, local_path: str) -> str:
        """Lädt eine Datei von einer URL herunter und speichert sie lokal.

        Args:
            remote_url: Die URL der Datei, die heruntergeladen werden soll.
            local_path: Der lokale Pfad, unter dem die Datei gespeichert werden soll.

        Raises:
            requests.HTTPError: Wenn der Download fehlschlägt.
        """
        # Imports
        import os
        import httpx

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Zielverzeichnis erstellen, falls es nicht existiert
        if os.path.isdir(local_path):
            filename = os.path.basename(remote_url.split("?", 1)[0]) or "download"
            dest = os.path.join(local_path, filename)
        else:
            dest = local_path
            dir_name = os.path.dirname(dest)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        # Downloadparameter konfigurieren
        chunk_size = 8192
        downloaded = 0

        # http download starten
        with httpx.stream("GET", remote_url, timeout=30.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)

            pbar = None
            if tqdm:
                pbar = tqdm(total=(total if total > 0 else None), unit="B", unit_scale=True, desc="Downloading")

            # Chunkweise herunterladen und speichern
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if pbar:
                        pbar.update(len(chunk))

            if pbar:
                pbar.close()

        return dest

    @staticmethod
    def extract_tar(tar_file: str, extract_dir: str) -> None:
        """Entpackt eine TAR(/gz/xz) Datei sicher in extract_dir.
        - Verwendet `tarfile.open(..., "r:*")` zur automatischen Kompressions-Erkennung.
        - Verhindert Path‑Traversal (keine Dateien außerhalb von extract_dir).
        - Überspringt absolute Pfade, Symlinks und Hardlinks.
        - Erstellt Zielverzeichnis falls nötig und loggt Warnungen bei Auffälligkeiten.

        Args:
            tar_file: Pfad zur TAR-Datei.
            extract_dir: Verzeichnis, in das die Dateien entpackt werden sollen.
        """
        import os
        import tarfile
        import logging

        # Verzeichnis erstellen, falls nicht existent und Pfad absolutieren
        os.makedirs(extract_dir, exist_ok=True)
        abs_dest = os.path.abspath(extract_dir)

        try:
            with tarfile.open(tar_file, "r:*") as tar:
                for member in tar.getmembers():
                    name = member.name

                    # Keine absoluten Pfade
                    if os.path.isabs(name):
                        logging.warning("Skipping absolute path in tar: %s", name)
                        continue

                    # Zielpfad berechnen und path traversal verhindern
                    dest_path = os.path.join(abs_dest, name)
                    if os.path.commonpath([abs_dest, os.path.abspath(dest_path)]) != abs_dest:
                        logging.warning("Skipping path traversal entry in tar: %s", name)
                        continue

                    # symlinks und hardlinks aus sicherheitsgründen überspringen
                    if member.issym() or member.islnk():
                        logging.warning("Skipping link in tar (symlink/hardlink): %s", name)
                        continue

                    # Parent-Link für den Member sicherstellen, da wir manuell extrahieren
                    parent_dir = os.path.dirname(dest_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)

                    # Den member extrahieren
                    tar.extract(member, path=abs_dest)
                    logging.debug(f"Extracted member {member} from {tar_file} into {abs_dest}")

            logging.info(f"Extracted tar file {tar_file} into {abs_dest}")

        except tarfile.ReadError as e:
            raise RuntimeError(f"Unable to read tar file {tar_file}: {e}") from e


    @staticmethod
    def extract_zip(zip_file: str, extract_dir: str) -> None:
        """Entpackt eine ZIP-Datei sicher in extract_dir.
        - Verhindert Path‑Traversal (keine Dateien außerhalb von extract_dir).
        - Erstellt Zielverzeichnis, falls nötig und loggt Warnungen bei Auffälligkeiten.

        Args:
            zip_file: Pfad zur ZIP-Datei.
            extract_dir: Verzeichnis, in das die Dateien entpackt werden sollen.
        """
        import os
        import zipfile
        import logging

        # Verzeichnis erstellen, falls nicht existent und Pfad absolutieren
        os.makedirs(extract_dir, exist_ok=True)
        abs_dest = os.path.abspath(extract_dir)

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    # Keine absoluten Pfade
                    if os.path.isabs(member):
                        logging.warning("Skipping absolute path in zip: %s", member)
                        continue

                    # Zielpfad berechnen und path traversal verhindern
                    dest_path = os.path.join(abs_dest, member)
                    if os.path.commonpath([abs_dest, os.path.abspath(dest_path)]) != abs_dest:
                        logging.warning("Skipping path traversal entry in zip: %s", member)
                        continue

                    # Parent-Link für den Member sicherstellen, da wir manuell extrahieren
                    parent_dir = os.path.dirname(dest_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)

                    # Den member extrahieren
                    zip_ref.extract(member, path=abs_dest)
                    logging.debug(f"Extracted member {member} from {zip_file} into {abs_dest}")

            logging.info(f"Extracted zip file {zip_file} into {abs_dest}")

        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Unable to read zip file {zip_file}: {e}") from e

    @staticmethod
    def cleanup_temp_file(file_path: str) -> None:
        """Löscht eine temporäre Datei, falls sie existiert.

        Args:
            file_path: Pfad zur temporären Datei.
        """
        import os

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Fehler beim Löschen der Datei {file_path}: {e}")

    @staticmethod
    def compute_dir_hash(dir_path: str, algorithm: str = "sha256", chunk_size: int = 8192, include_hidden: bool = False) -> str | None:
        """Berechnet einen deterministischen Hash eines Verzeichnisses.

        - Geht rekursiv alle Dateien durch (alphabetisch sortiert).
        - Ignoriert Symlinks; optional können versteckte Dateien übersprungen werden.
        - Hash basiert auf relativem Pfad (mit '/' als Trenner) + Inhalt.
        Returns: hex Digest des Hashes.
        """
        import os
        import hashlib
        import logging

        if not os.path.exists(dir_path):
            logger.warning("Directory %s doesn't exist", dir_path)
            return None

        abs_base = os.path.abspath(dir_path)
        hasher = hashlib.new(algorithm)

        for root, dirs, files in os.walk(abs_base):
            dirs.sort()
            files.sort()
            for fname in files:
                if not include_hidden and fname.startswith("."):
                    continue

                fpath = os.path.join(root, fname)
                # Skip symlinks for safety
                if os.path.islink(fpath):
                    logging.debug("Skipping symlink in dir_hash: %s", fpath)
                    continue

                # Relative path with normalized separator for determinism
                rel_path = os.path.relpath(fpath, abs_base).replace(os.sep, "/")
                hasher.update(rel_path.encode("utf-8"))
                hasher.update(b"\0")  # Separator

                # Read file in chunks
                with open(fpath, "rb") as fh:
                    while True:
                        chunk = fh.read(chunk_size)
                        if not chunk:
                            break
                        hasher.update(chunk)

                hasher.update(b"\0")  # Separator between files

        return hasher.hexdigest()
