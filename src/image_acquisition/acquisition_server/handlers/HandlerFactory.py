from data_types.ImageDatasetConfiguration import ImageDatasetConfiguration
from image_acquisition.acquisition_server.handlers.AbstractHandler import AbstractHandler
from image_acquisition.acquisition_server.handlers.LocalResourceDownloader import LocalResourceDownloader
from image_acquisition.acquisition_server.handlers.TarDownloader import TarDownloader
from image_acquisition.acquisition_server.handlers.ZipDownloader import ZipDownloader
from image_acquisition.acquisition_server.handlers.KaggleDownloader import KaggleDownloader
from image_acquisition.acquisition_server.handlers.KaggleLhqDownloader import KaggleLhqDownloader


class HandlerFactory:
    @staticmethod
    def create(dataset_config: ImageDatasetConfiguration) -> AbstractHandler:
        if dataset_config.handler_type == "tar":
            return TarDownloader(dataset_config.dataset_id,
                                 dataset_config.source_url,
                                 dataset_config.calculate_images_root_path(),
                                 dataset_config.archive_root,
                                 dataset_config.target_hash)
        elif dataset_config.handler_type == "zip":
            zdl = ZipDownloader(dataset_config.dataset_id,
                                dataset_config.source_url,
                                dataset_config.calculate_images_root_path(),
                                dataset_config.archive_root,
                                dataset_config.target_hash)
            return zdl
        elif dataset_config.handler_type == "local_resource":
            lrd = LocalResourceDownloader(dataset_config.dataset_id,
                                          dataset_config.resource_file_path,
                                          dataset_config.calculate_images_root_path(),
                                          dataset_config.archive_root,
                                          dataset_config.target_hash)
            return lrd
        elif dataset_config.handler_type == "kaggle":
            kd = KaggleDownloader(dataset_config.dataset_id,
                                 dataset_config.kaggle_dataset,
                                 dataset_config.calculate_images_root_path(),
                                 dataset_config.archive_root,
                                 dataset_config.target_hash)
            return kd
        elif dataset_config.handler_type == "lhq_mountains":
            kmd = KaggleLhqDownloader(dataset_config.dataset_id,
                                      dataset_config.kaggle_dataset,
                                      dataset_config.calculate_images_root_path(),
                                      dataset_config.category,
                                      dataset_config.target_hash)
            return kmd
        else:
            raise ValueError(f"Unknown handler type: {dataset_config.handler_type}")
