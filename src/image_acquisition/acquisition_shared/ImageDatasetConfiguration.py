class ImageDatasetConfiguration:
    def __init__(self, type: str, source_url: str, destination_dir: str, extracted_root: str):
        self.type = type
        self.source_url = source_url
        self.destination_dir = destination_dir
        self.extracted_root = extracted_root

    def from_dict(config_dict: dict):
        return ImageDatasetConfiguration(
            type=config_dict.get("type"),
            source_url=config_dict.get("source_url"),
            destination_dir=config_dict.get("destination_dir"),
            extracted_root=config_dict.get("extracted_root"),
        )