from typing import TypedDict, Tuple


class DataParams(TypedDict):
    # ID des Datasets, das geladen werden soll (definiert in DatasetRegistry).
    dataset_id: str

    # Maximale Größe der Bilder (Breite, Höhe). Bilder werden auf diese Größe herunterskaliert.
    image_max_size: Tuple[int, int]
