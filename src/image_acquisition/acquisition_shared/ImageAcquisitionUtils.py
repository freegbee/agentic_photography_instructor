import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageAcquisitionUtils:
    @staticmethod
    def download_file(remote_url: str, local_path: Path) -> Path:
        """Lädt eine Datei von einer URL herunter und speichert sie lokal.

        Args:
            remote_url: Die URL der Datei, die heruntergeladen werden soll.
            local_path: Der lokale Pfad, unter dem die Datei gespeichert werden soll.

        Raises:
            requests.HTTPError: Wenn der Download fehlschlägt.
        """
        # Imports
        import httpx

        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Zielverzeichnis erstellen, falls es nicht existiert
        if local_path.is_dir():
            filename = os.path.basename(remote_url.split("?", 1)[0]) or "download"
            dest: Path = local_path / filename
        else:
            dest: Path = local_path
            dir_name = dest.parent.resolve()
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
    def copy_resource_file(src_path: Path, dest_dir: Path, dest_name: str | None = None,
                           overwrite: bool = False) -> Path:
        """
        Kopiert eine Datei von `src_path` nach `dest_dir`.
        Args:
            src_path: Pfad zur Quelldatei.
            dest_dir: Zielverzeichnis (wird erstellt, falls nicht existent).
            dest_name: Optionaler neuer Dateiname im Ziel. Standard: Dateiname von `src_path`.
            overwrite: Falls True, vorhandene Datei im Ziel wird überschrieben.

        Returns:
            Absoluter Pfad zur kopierten Datei.

        Raises:
            FileNotFoundError: Wenn die Quelldatei nicht existiert.
            FileExistsError: Wenn Zieldatei existiert und overwrite=False.
            OSError: Bei anderen Dateisystemfehlern.
        """

        logger.info("Copying resource file from %s to %s with name %s (overwrite=%s)", src_path, dest_dir, dest_name,
                    overwrite)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")

        # Bestimme Zielname und -pfad
        name = dest_name or os.path.basename(src_path)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = dest_dir / name

        # Existenz prüfen
        if os.path.exists(dest_path) and not overwrite:
            raise FileExistsError(f"Destination file already exists: {dest_path}")

        try:
            # copy2 kopiert inkl. Metadaten (Timestamp, Berechtigungen wenn möglich)
            shutil.copy2(src_path, dest_path)
            logger.info("Copied file %s -> %s", src_path, dest_path)
            return dest_path
        except Exception as e:
            logger.exception("Fehler beim Kopieren der Datei %s nach %s: %s", src_path, dest_path, e)
            raise

    @staticmethod
    def extract_tar(tar_file: Path, extract_dir: Path, member_root: Path | None) -> None:
        """Entpackt eine TAR(/gz/xz) Datei sicher in extract_dir.
        - Verwendet `tarfile.open(..., "r:*")` zur automatischen Kompressions-Erkennung.
        - Verhindert Path‑Traversal (keine Dateien außerhalb von extract_dir).
        - Überspringt absolute Pfade, Symlinks und Hardlinks.
        - Erstellt Zielverzeichnis falls nötig und loggt Warnungen bei Auffälligkeiten.

        Args:
            tar_file: Pfad zur TAR-Datei.
            extract_dir: Verzeichnis, in das die Dateien entpackt werden sollen.
            member_root: verzeichnis innerhalb des tar-archives, das extrahiert werden soll (oder None für alle).
        """
        import tarfile

        # Verzeichnis erstellen, falls nicht existent und Pfad absolutieren
        logger.info(f"Extracting {tar_file}")
        os.makedirs(extract_dir, exist_ok=True)
        abs_dest = os.path.abspath(extract_dir)
        normalized_member_root = None
        if member_root:
            # Normalisiere den gewünschten Root (keine führenden/trailenden '/')
            normalized_member_root = member_root.as_posix().lstrip("./").lstrip("/").rstrip("/")

        logger.info(f"Extracting {tar_file} to {normalized_member_root}")

        try:
            with tarfile.open(tar_file, "r:*") as tar:
                for member in tar.getmembers():
                    member_name = member.name

                    # Keine absoluten Pfade
                    if os.path.isabs(member_name):
                        logger.warning("Skipping absolute path in tar: %s", member_name)
                        continue

                    # Filter auf member_root, falls gesetzt
                    if normalized_member_root:
                        if member_name == normalized_member_root:
                            rel_path = ""  # das Verzeichnis selbst
                        elif member_name.startswith(normalized_member_root + "/"):
                            rel_path = member_name[len(normalized_member_root) + 1:]
                        else:
                            continue
                    else:
                        rel_path = member_name

                    # Zielpfad berechnen und path traversal verhindern
                    dest_path = os.path.join(abs_dest, rel_path) if rel_path else abs_dest
                    if os.path.commonpath([abs_dest, os.path.abspath(dest_path)]) != abs_dest:
                        logger.warning("Skipping path traversal entry in tar: %s", member_name)
                        continue

                    # symlinks und hardlinks aus sicherheitsgründen überspringen
                    if member.issym() or member.islnk():
                        logger.warning("Skipping link in tar (symlink/hardlink): %s", member_name)
                        continue

                    # Verzeichnisse anlegen
                    if member.isdir():
                        os.makedirs(dest_path, exist_ok=True)
                        continue

                    # Parent-Link für den Member sicherstellen, da wir manuell extrahieren
                    parent_dir = os.path.dirname(dest_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)

                    ## Dateiinhalt sicher extrahieren
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        logger.warning("Could not extract member (no fileobj): %s", member_name)
                        continue

                    with open(dest_path, "wb") as out_f:
                        shutil.copyfileobj(fileobj, out_f)

                    # Berechtigungen setzen, falls vorhanden
                    try:
                        os.chmod(dest_path, member.mode)
                    except Exception:
                        logger.debug("Could not set mode for %s", dest_path)
                logger.debug(f"Extracted member {member} from {tar_file} into {abs_dest}")

            logger.info(f"Extracted tar file {tar_file} into {abs_dest}")

        except tarfile.ReadError as e:
            logger.exception(e)
            raise RuntimeError(f"Unable to read tar file {tar_file}: {e}") from e

    @staticmethod
    def extract_zip(zip_file: Path, extract_dir: Path, member_root: Path | None) -> None:
        """Entpackt eine ZIP-Datei sicher in extract_dir.
        - Verhindert Path‑Traversal (keine Dateien außerhalb von extract_dir).
        - Erstellt Zielverzeichnis, falls nötig und loggt Warnungen bei Auffälligkeiten.

        Args:
            zip_file: Pfad zur ZIP-Datei.
            extract_dir: Verzeichnis, in das die Dateien entpackt werden sollen.
            member_root: verzeichnis innerhalb des zip-archives, das extrahiert werden soll (oder None für alle).
        """
        import zipfile

        # Verzeichnis erstellen, falls nicht existent und Pfad absolutieren
        os.makedirs(extract_dir, exist_ok=True)
        abs_dest = os.path.abspath(extract_dir)

        normalized_member_root = None
        if member_root:
            normalized_member_root = member_root.as_posix().lstrip("./").lstrip("/").rstrip("/")

        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                for info in zip_ref.infolist():
                    member = info.filename

                    # Skip absolute paths
                    if os.path.isabs(member):
                        logger.warning("Skipping absolute path in zip: %s", member)
                        continue

                    # Normalize member names (zip may use trailing '/')
                    member_norm = member.rstrip("/")

                    # Filter auf member_root, falls gesetzt
                    if normalized_member_root:
                        if member_norm == normalized_member_root:
                            rel_path = ""  # das Verzeichnis selbst
                        elif member_norm.startswith(normalized_member_root + "/"):
                            rel_path = member_norm[len(normalized_member_root) + 1:]
                        else:
                            continue
                    else:
                        rel_path = member_norm

                    # Zielpfad berechnen und path traversal verhindern
                    dest_path = os.path.join(abs_dest, rel_path) if rel_path else abs_dest
                    if os.path.commonpath([abs_dest, os.path.abspath(dest_path)]) != abs_dest:
                        logger.warning("Skipping path traversal entry in zip: %s", member)
                        continue

                    # Directory entries
                    if info.is_dir() or member.endswith("/"):
                        os.makedirs(dest_path, exist_ok=True)
                        continue

                    # Detect symlink in zip (Unix external_attr high bits)
                    try:
                        is_symlink = (info.external_attr >> 16) & 0o170000 == 0o120000
                    except Exception:
                        is_symlink = False
                    if is_symlink:
                        logger.warning("Skipping link in zip (symlink): %s", member)
                        continue

                    # Ensure parent exists
                    parent_dir = os.path.dirname(dest_path)
                    if parent_dir and not os.path.exists(parent_dir):
                        os.makedirs(parent_dir, exist_ok=True)

                    # Extract file content safely
                    try:
                        with zip_ref.open(info, "r") as src, open(dest_path, "wb") as out_f:
                            shutil.copyfileobj(src, out_f)
                    except RuntimeError as e:
                        logger.warning("Could not extract member (runtime error): %s -> %s: %s", member, dest_path, e)
                        continue

                    # Try to set permissions if present in external_attr
                    try:
                        mode = (info.external_attr >> 16) & 0xFFFF
                        if mode:
                            os.chmod(dest_path, mode)
                    except Exception:
                        logger.debug("Could not set mode for %s", dest_path)

                    logger.debug("Extracted member %s from %s into %s", member, zip_file, abs_dest)

        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Unable to read zip file {zip_file}: {e}") from e

    @staticmethod
    def cleanup_temp_file(file_path: Path) -> None:
        """Löscht eine temporäre Datei, falls sie existiert.

        Args:
            file_path: Pfad zur temporären Datei.
        """
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Fehler beim Löschen der Datei {file_path}: {e}")

    @staticmethod
    def compute_dir_hash(dir_path: Path, algorithm: str = "sha256", chunk_size: int = 8192,
                         include_hidden: bool = False) -> str | None:
        """Berechnet einen deterministischen Hash eines Verzeichnisses.

        - Geht rekursiv alle Dateien durch (alphabetisch sortiert).
        - Ignoriert Symlinks; optional können versteckte Dateien übersprungen werden.
        - Hash basiert auf relativem Pfad (mit '/' als Trenner) + Inhalt.
        Returns: hex Digest des Hashes.
        """
        import hashlib

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
                    logger.debug("Skipping symlink in dir_hash: %s", fpath)
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
