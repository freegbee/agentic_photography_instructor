from src.image_aquisition.TarDownloader import TarDownloader

class Places365TarDownloader(TarDownloader):
    def __init__(self):
        super().__init__('places365_validation')

