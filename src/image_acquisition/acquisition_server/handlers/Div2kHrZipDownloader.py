from image_acquisition.acquisition_server.handlers.ZipDownloader import ZipDownloader


class Div2kHrZipDownloader(ZipDownloader):
    def __init__(self):
        super().__init__('https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip', '/app/volumes/resources/div2k/hr/validation')