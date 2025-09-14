from image_aquisition.Places365TarDownloader import Places365TarDownloader


def main():
    Places365TarDownloader().download_and_extract()
    # Places365TarDownloader().extract_tar()
    # Places365TarDownloader().cleanup()


if __name__ == "__main__":
    main()
