from typing import Protocol
import numpy as np
import numpy.typing as npt

from .consts import AXIS_BATCH, DELTA


class LossFunction(Protocol):
    def loss(self, X: npt.NDArray, teach: npt.NDArray) -> np.float32: ...
    def backward(self, e: np.float32 = np.float32(1)) -> npt.NDArray: ...


def cross_entropy(output: npt.NDArray[np.float32], teach: npt.NDArray[np.float32]):
    """交差エントロピーの計算

    Parameters
    ----------
    output: npt.NDArray[np.float32]
        出力 (batch_size, )
    teach: npt.NDArray[np.float32]
        教師信号

    Returns
    -------
    e: batch中の交差エントロピーの平均

    """
    return -np.sum(teach * np.log(output)) / output.shape[AXIS_BATCH]


class CrossEntropy:
    def __init__(self) -> None:
        pass

    def loss(self, X: npt.NDArray, teach: npt.NDArray) -> np.float32:
        """

        Parameters
        ----------
        X: npt.NDArray
            modelの出力
        teach: npt.NDArray
            教師信号 (0, 0, ..., 1, 0, ..., 0)
            one hot形式
        Returns
        -------
        e: batch中の交差エントロピーの平均
        """
        self.X = X
        self.teach = teach
        return cross_entropy(X, teach)

    def backward(self, e: np.float32 = np.float32(1)) -> npt.NDArray:
        dy = -self.teach / (self.X + DELTA) / self.teach.shape[AXIS_BATCH]
        return e * dy
