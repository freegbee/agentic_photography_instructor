from pathlib import Path
from typing import List, Union, Optional

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, field_validator


class TransformerApplication(BaseModel):
    """ Class to keep track of applied transformers and their effect on the original image score. """
    applied_transformers: List[str] | None = None
    score: float | None = None
    score_change: float | None = None

class ImageData(BaseModel):
    image_path: Optional[Path] = None
    image_relative_path: Optional[Path] = None
    image_data: Optional[np.ndarray] = None
    image_color_order: Optional[str] = None  # 'RGB' or 'BGR'
    score: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator('image_data')
    def validate_image_data(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            raise TypeError("image_data muss ein numpy.ndarray sein")
        return v

    def clone(self):
        return ImageData(
            image_path=self.image_path,
            image_data=self.image_data.copy() if self.image_data is not None else None,
            image_color_order=self.image_color_order,
            score=self.score
        )

    def get_image_data(self, image_channel_order=None) -> np.ndarray:
        """ Return the stored image data as a numpy ndarray. Change color order if specified. """
        if self.image_data is None:
            raise ValueError("Image data is not set")
        if self.image_data.ndim == 2:
            # grayscale image with only one channel
            return cv2.merge((self.image_data, self.image_data, self.image_data))
        if image_channel_order is not None and image_channel_order != self.image_color_order:
            if image_channel_order == 'RGB' and self.image_color_order == 'BGR':
                return cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB)
            elif image_channel_order == 'BGR' and self.image_color_order == 'RGB':
                return cv2.cvtColor(self.image_data, cv2.COLOR_RGB2BGR)
            else:
                raise ValueError(
                    f"Unsupported color order conversion from {self.image_color_order} to {image_channel_order}")
        return self.image_data

    def get_image(self) -> Image.Image:
        """ Convert the stored image data to a PIL Image and return it. """
        if self.image_data is None:
            raise ValueError("Image data is not set")
        if self.image_color_order == 'BGR':
            return Image.fromarray(cv2.cvtColor(self.image_data, cv2.COLOR_BGR2RGB))

        return Image.fromarray(self.image_data)


class AgenticImage(BaseModel):
    filename: str | None = None
    source_image: ImageData | None = None
    transformed_image: ImageData | None = None
    applied_transformers: List[str] | None = None

    transformer_protocol: List[TransformerApplication] = []

    class Config:
        arbitrary_types_allowed = True

    def update_source_image(self, image_data: np.ndarray, image_color_order: str, filename: str, score: float | None = None):
        """ Add image data and related metadata to the AgenticImage instance. """
        self.source_image = ImageData(image_data=image_data,
                                      image_color_order=image_color_order)
        self.filename = filename
        self.update_source_score(score)

    def update_source_score(self, score: float | None):
        if self.source_image is None:
            raise ValueError("Source image must be set before setting the score")
        self.source_image.score = score

    def update_transformed_image(self, image_data: np.ndarray, image_color_order: str,
                                 applied_transformers: str | List[str], score: float | None = None):
        """ Add transformed image data and related metadata to the AgenticImage instance. """
        self.transformed_image = ImageData(image_data=image_data,
                                           image_color_order=image_color_order)
        self.update_applied_transformers(applied_transformers)
        self.update_transformed_score(score)

    @staticmethod
    def __standardize_applied_transformer(applied_transformer: str | List[str]) -> List[str]:
        if isinstance(applied_transformer, str):
            return applied_transformer.split('&')
        elif isinstance(applied_transformer, list):
            return applied_transformer
        else:
            raise TypeError("applied_transformer must be a string or a list of strings")

    def update_applied_transformers(self, applied_transformers: str | List[str]):
        self.applied_transformers = self.__standardize_applied_transformer(applied_transformers)

    def update_transformed_score(self, score: float | None):
        if self.transformed_image is None:
            raise ValueError("Transformed image must be set before setting the score")
        self.transformed_image.score = score

    def append_transformer_protocol(self, applied_transformer: str | List[str], score: float):
        self.transformer_protocol.append(TransformerApplication(
            applied_transformers=self.__standardize_applied_transformer(applied_transformer),
            score=score,
            score_change=score - self.source_image.score if self.source_image is not None and self.source_image.score is not None else None
        ))

    def clone(self):
        """ Create a deep copy of the AgenticImage instance. """
        return AgenticImage(
            filename=self.filename,
            source_image=self.source_image.clone() if self.source_image is not None else None,
            transformed_image=self.transformed_image.clone() if self.transformed_image is not None else None,
            applied_transformers=self.applied_transformers.copy() if self.applied_transformers is not None else None
        )

    def __ensure_scores(self):
        if self.source_image is None or self.source_image.score is None or self.transformed_image is None or self.transformed_image.score is None:
            raise ValueError("Both source and transformed scores must be set to compare")

    def calculate_score_change(self):
        self.__ensure_scores()
        return self.transformed_image.score - self.source_image.score

    def has_transformations(self) -> bool:
        """ Check if any transformations have been applied to the image. """
        return self.applied_transformers is not None and len(self.applied_transformers) > 0

    def transform_is_better(self) -> bool:
        """ Check if the transformed score is better than the original score. """
        self.__ensure_scores()
        return self.transformed_image.score > self.source_image.score

    def get_comparable_score(self):
        return self.transformed_image.score if self.transformed_image is not None and self.transformed_image.score is not None else self.source_image.score if self.source_image is not None else None

    def is_better_than(self, other_image) -> bool:
        """ Determine if the current image is better than another image based on their scores. """
        if self.get_comparable_score() is None or other_image.get_comparable_score() is None:
            raise ValueError("Both images must have a  score to compare")
        return self.get_comparable_score() > other_image.get_comparable_score()
