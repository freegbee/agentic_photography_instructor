import numpy as np
import torch
import logging


class Juror:
    def __init__(self):
        from juror.siglib_v2_5 import convert_v2_5_from_siglip
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self.model = self.model.to(torch.bfloat16).cuda()

        if torch.mps.is_available():
            # Mac MPS Support: bfloat16 ist erst ab macOS 14+ stabil.
            # Fallback auf float16, falls bfloat16 nicht unterstützt wird.
            device = torch.device("mps")
            try:
                self.model = self.model.to(device).to(torch.bfloat16)
            except (RuntimeError, TypeError):
                logging.getLogger(__name__).warning(
                    "MPS does not support bfloat16 on this system. Falling back to float16.")
                self.model = self.model.to(device).to(torch.float16)

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

        if torch.mps.is_available():
            # Nutze den Datentyp, den das Modell tatsächlich hat (bfloat16 oder float16)
            # self.model.dtype ist zuverlässiger als hardcoded bfloat16
            target_dtype = self.model.dtype if hasattr(self.model, "dtype") else torch.float16
            pixel_values = pixel_values.to('mps').to(target_dtype)

        # predict aesthetic score
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().numpy()

        return score
