from image_acquisition.acquisition_server.handlers.Div2kHrZipDownloader import Div2kHrZipDownloader
from image_acquisition.acquisition_server.handlers.Places365TarDownloader import Places365TarDownloader


def run():
    # downloader = Places365TarDownloader()
    # downloader.download_and_extract()
    downloader = Div2kHrZipDownloader()
    downloader.download_and_extract()

if __name__ == "__main__":
    run()