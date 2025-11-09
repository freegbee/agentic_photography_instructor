class ImageDatasetConfiguration:
    def __init__(self, dataset_id: str, handler_type: str, source_url: str, resource_file_path: str, destination_dir: str, extracted_root: str,
                 target_hash: str = None):
        self.dataset_id = dataset_id
        self.handler_type = handler_type
        self.source_url = source_url
        self.resource_file_path = resource_file_path
        self.destination_dir = destination_dir
        self.extracted_root = extracted_root
        self.target_hash = target_hash

    @staticmethod
    def from_dict(dataset_id: str, config_dict: dict):
        return ImageDatasetConfiguration(
            dataset_id = dataset_id,
            handler_type=config_dict.get("type"),
            source_url=config_dict.get("source_url"),
            resource_file_path=config_dict.get("resource_file_path"),
            destination_dir=config_dict.get("destination_dir"),
            extracted_root=config_dict.get("extracted_root"),
            target_hash=config_dict.get("target_hash"),
        )

    def calculate_image_root_path(self, image_volume_path) -> str:
        return f"{image_volume_path}/{self.destination_dir}/{self.extracted_root}"