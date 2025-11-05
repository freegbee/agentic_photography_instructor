import os
import sys
import tarfile
from abc import ABC
from pathlib import Path
from typing import Dict

import requests

from utils.ConfigLoader import ConfigLoader


class TarDownloader(ABC):
    def __init__(self, config_section: str):
        config: Dict = ConfigLoader().load(env=os.environ["ENV_NAME"])
        self.url = config['data']['tar'][config_section]['url']
        self.download_dir = Path(config['data']['tar'][config_section]['download_dir'])
        self.tar_temp_dir = Path(self.download_dir, "./temp")

    def _calculate_temp_tar_file(self):
        return os.path.join(self.tar_temp_dir, 'download.tar')

    def download_tar(self):
        print(f"Lade {self.url} herunter...")
        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        if not os.path.exists(self.tar_temp_dir):
            os.makedirs(self.tar_temp_dir)

        self.download_dir.mkdir(parents=True, exist_ok=True)
        with open(self._calculate_temp_tar_file(), "wb") as tar_download_file:
            total_length = int(response.headers.get('content-length'))
            chunk_size = 8192
            dl_length = 0
            for chunk in response.iter_content(chunk_size=chunk_size):
                tar_download_file.write(chunk)

                # Display progress bar, see https://stackoverflow.com/a/15645088
                dl_length += len(chunk)
                done = int(50 * dl_length / total_length)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {(100 * dl_length / total_length):.2f} %")
                sys.stdout.flush()
        print(f"Download abgeschlossen: {self.tar_temp_dir}")

    def extract_tar(self):
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)

        print(f"Entpacke {self.tar_temp_dir} nach {self.download_dir}...")
        with tarfile.open(self._calculate_temp_tar_file(), "r") as tar:
            tar.extractall(path=self.download_dir)
        print("Entpacken abgeschlossen.")

    def cleanup(self):
        if Path(self._calculate_temp_tar_file()).exists():
            try:
                os.remove(self._calculate_temp_tar_file())
                print(f"Cleanup: {self._calculate_temp_tar_file()} gelöscht.")
            except Exception as e:
                print(f"Fehler beim Löschen von {self._calculate_temp_tar_file()}: {e}")

        if Path(self.tar_temp_dir).exists():
            try:
                os.removedirs(self.tar_temp_dir)
                print(f"Cleanup: {self.tar_temp_dir} gelöscht.")
            except Exception as e:
                print(f"Fehler beim Löschen von {self.tar_temp_dir}: {e}")

    def download_and_extract(self):
        tar_path = self.download_tar()
        self.extract_tar()
        self.cleanup()
