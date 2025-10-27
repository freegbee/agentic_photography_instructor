import numpy as np
import torch


class Juror:
    def __init__(self):
        from juror.siglib_v2_5 import convert_v2_5_from_siglip
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16).cuda()

    def inference(self, image_rgb: np.ndarray) -> float:
        """
        Calculate score for the image. The image is represented by a numpy array.
        Important: the array must be in the shape (height, width, channels) and the values must be in the range [0, 255].
        The channels must be in RGB order.
        :param image_rgb: ndarray in shape (height, width, channels) with values in range [0, 255] and channels in RGB order
        :return: score: float
        0 = worst, 10 = best
        """
        # preprocess image
        pixel_values = self.preprocessor(
            images=image_rgb,
            return_tensors="pt"
        ).pixel_values

        if torch.cuda.is_available():
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

        # predict aesthetic score
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().numpy()

        return score
