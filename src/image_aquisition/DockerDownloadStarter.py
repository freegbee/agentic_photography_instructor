from image_aquisition.FiftyonePlacesImageDownloader import FiftyonePlacesImageDownloader


def main():
    FiftyonePlacesImageDownloader().download_images('validation')

if __name__ == "__main__":
    main()