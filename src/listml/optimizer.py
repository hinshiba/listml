import numpy as np
import numpy.typing as npt

from typing import Protocol


class Optimizer(Protocol):
    def calc(self, w: npt.NDArray, d: npt.NDArray):
        """重みを変更する

        Parameters
        ----------
        w: npt.NDArray
            重み
        d: npt.NDArray
            その重みの勾配
        """
        ...


class StochasticGradientDescent:
    """確率的勾配降下法"""

    def __init__(self, lr: np.float32):
        self.lr = lr

    def calc(self, w: npt.NDArray, d: npt.NDArray):
        w -= self.lr * d
