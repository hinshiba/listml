import numpy as np
import numpy.typing as npt

from .consts import AXIS_DATA
from .optimizer import Optimizer


class Relu:
    """Reluの実装"""

    def __init__(self) -> None:
        pass

    def prepare(self, X: npt.NDArray) -> npt.NDArray:
        return X

    def forward(self, X: npt.NDArray) -> npt.NDArray:
        # 0にする場所がTrueのmask
        self.mask = np.copy(X) <= 0
        X[self.mask] = 0
        return X

    def backward(self, e: npt.NDArray) -> npt.NDArray:
        # 順伝播で通さなかったのなら，その部分の変化量は0にする
        e[self.mask] = 0
        return e

    def learn(self, optimizer: Optimizer):
        pass


class Softmax:
    """ソフトマックスの実装"""

    def __init__(self) -> None:
        pass

    def prepare(self, X: npt.NDArray) -> npt.NDArray:
        return X

    def forward(self, X: npt.NDArray) -> npt.NDArray:
        X = np.exp(X - np.max(X, axis=AXIS_DATA, keepdims=True))
        self.Y = X / np.sum(X, axis=AXIS_DATA, keepdims=True)
        return self.Y

    def backward(self, e: npt.NDArray) -> npt.NDArray:
        return self.Y * (e - np.sum(e * self.Y, axis=1, keepdims=True))

    def learn(self, optimizer: Optimizer):
        pass
