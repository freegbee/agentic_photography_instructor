from pathlib import Path

from compress_pickle import dump, load

from data_types.AgenticImage import AgenticImage


class AgenticImageDataStorage:

    COMPRESSION = "lzma"

    @staticmethod
    def save(folder: str | Path, agentic_image: AgenticImage) -> Path:
        if not isinstance(folder, Path):
            folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if agentic_image.filename is None:
            raise ValueError("AgenticImage must have a filename to be saved")

        filename = folder / f"{agentic_image.filename}.cpkl"
        with open(filename, "wb") as f:
            dump(agentic_image, f, compression=AgenticImageDataStorage.COMPRESSION, set_default_extension=False)

        return filename

    @staticmethod
    def load_agentic_image(file_path: str | Path) -> AgenticImage:
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        with open(file_path, "rb") as f:
            agentic_image: AgenticImage = load(f, compression=AgenticImageDataStorage.COMPRESSION, set_default_extension=False)
        return agentic_image
