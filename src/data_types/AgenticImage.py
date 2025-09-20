from typing import List

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, field_validator


class AgenticImage(BaseModel):
    filename: str | None = None
    is_source: bool | None = None
    image_data: np.ndarray | None = None
    image_color_order: str | None = None
    score: float | None = None
    transformed_score: float | None = None
    applied_transformers: List[str] | None = None

    class Config:
        arbitrary_types_allowed = True

    def add_image(self, image_data: np.ndarray, image_channel_order: str, is_source: bool, filename: str):
        """ Add image data and related metadata to the AgenticImage instance. """
        self.image_data = image_data
        self.image_color_order = image_channel_order
        self.is_source = is_source
        self.filename = filename

    @field_validator('image_data')
    def validate_image_data(cls, v):
        if v is not None and not isinstance(v, np.ndarray):
            raise TypeError("image_data muss ein numpy.ndarray sein")
        return v

    def set_applied_transformers(self, applied_transformers: str | List[str]):
        if isinstance(applied_transformers, str):
            self.applied_transformers = applied_transformers.split('&')
        else:
            self.applied_transformers = applied_transformers.copy()

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

    def clone(self):
        """ Create a deep copy of the AgenticImage instance. """
        return AgenticImage(
            filename=self.filename,
            is_source=self.is_source,
            image_data=self.image_data.copy() if self.image_data is not None else None,
            image_color_order=self.image_color_order,
            score=self.score,
            applied_transformers=self.applied_transformers.copy() if self.applied_transformers is not None else None
        )

    def calculate_score_change(self):
        if self.score is not None and self.transformed_score is not None:
            return self.transformed_score - self.score
        return None

    def is_better(self):
        score_change = self.calculate_score_change()
        if score_change is not None:
            return score_change > 0
        return None
