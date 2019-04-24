import attr
import cv2
import numpy as np


@attr.s(auto_attribs=True)
class Image:
    image: np.array
    _hsv: np.array = attr.ib(init=False, default=None)
    _greyscale: np.array = attr.ib(init=False, default=None)

    def as_HSV(self) -> np.array:
        if self._hsv is None:
            self._hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        return self._hsv

    def as_greyscale(self) -> np.array:
        if self._greyscale is None:
            self._greyscale = np.mean(self.image, axis=2).astype(np.uint8)
        return self._greyscale
