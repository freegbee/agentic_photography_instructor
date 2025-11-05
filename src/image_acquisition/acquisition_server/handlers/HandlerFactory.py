from image_acquisition.TarDownloader import TarDownloader
from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_server.handlers.ZipDownloader import ZipDownloader
from image_acquisition.acquisition_shared.ImageDatasetConfiguration import ImageDatasetConfiguration


class HandlerFactory:
    @staticmethod
    def create(dataset_config: ImageDatasetConfiguration) -> AbstractHandler:
        if dataset_config.type == "tar":
            return TarDownloader(dataset_config.source_url, dataset_config.destination_dir)
        elif dataset_config.type == "zip":
            return ZipDownloader(dataset_config.source_url, dataset_config.destination_dir)
        else:
            raise ValueError(f"Unknown handler type: {dataset_config.type}")