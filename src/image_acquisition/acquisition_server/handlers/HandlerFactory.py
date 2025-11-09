from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_server.handlers.LocalResourceDownloader import LocalResourceDownloader
from image_acquisition.acquisition_server.handlers.TarDownloader import TarDownloader
from image_acquisition.acquisition_server.handlers.ZipDownloader import ZipDownloader
from image_acquisition.acquisition_shared.ImageDatasetConfiguration import ImageDatasetConfiguration


class HandlerFactory:
    @staticmethod
    def create(dataset_config: ImageDatasetConfiguration) -> AbstractHandler:
        if dataset_config.handler_type == "tar":
            return TarDownloader(dataset_config.source_url, dataset_config.destination_dir, dataset_config.target_hash)
        elif dataset_config.handler_type == "zip":
            zdl = ZipDownloader(dataset_config.source_url, dataset_config.destination_dir, dataset_config.target_hash)
            return zdl
        elif dataset_config.handler_type == "local_resource":
            lrd = LocalResourceDownloader(dataset_config.resource_file_path, dataset_config.destination_dir, dataset_config.target_hash)
            return lrd
        else:
            raise ValueError(f"Unknown handler type: {dataset_config.handler_type}")