from .TarDownloader import TarDownloader

class Places365TarDownloader(TarDownloader):
    def __init__(self):
        super().__init__('https://data.csail.mit.edu/places/places365/val_large.tar', '/app/volumes/resources/places365')

