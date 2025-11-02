from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz

from image_aquisition.FiftyoneImageDownloader import FiftyoneImageDownloader


class FiftyonePlacesImageDownloader(FiftyoneImageDownloader):
    def __init__(self):
        super().__init__('places')
