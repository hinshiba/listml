import numpy as np
import numpy.typing as npt

from typing import Final, Protocol

from .consts import AXIS_BATCH, AXIS_DATA
from .optimizer import Optimizer

HE_INIT_PARAM: Final[int] = 2


# 抽象クラスの予定
class Layer(Protocol):
    def prepare(self, X: npt.NDArray) -> npt.NDArray:
        """適当なデータを流して次元数を確定させつつ初期化する"""
        ...

    def forward(self, X: npt.NDArray) -> npt.NDArray:
        """順伝播"""
        ...

    def backward(self, e: npt.NDArray) -> npt.NDArray:
        """逆伝播"""
        ...

    def learn(self, optimizer: Optimizer):
        """学習"""
        ...


def pred(network: list[Layer], X: npt.NDArray) -> npt.NDArray:
    """ネットワークの順伝播を計算する"""
    for n in network:
        # print(X)
        X = n.forward(X)
    return X


class FullConn:
    def __init__(self, unit: int) -> None:
        self.unit = unit

    def prepare(self, X: npt.NDArray) -> npt.NDArray:
        shape = X.shape
        data_dim = shape[AXIS_DATA]
        # print(f"{data_dim=}")
        # ばらつくようにガウス分布のsqrt{2/入力}倍で初期化
        scale = np.sqrt(HE_INIT_PARAM / data_dim)
        # 内積で計算できるような形で初期化
        self.weight = scale * np.random.randn(data_dim, self.unit)
        self.bias = scale * np.random.randn(self.unit)
        return self.forward(X)

    def forward(self, X: npt.NDArray) -> npt.NDArray:
        # @で新しいオブジェクトになるのでcopyしない
        self.X = X
        return X @ self.weight + self.bias

    def backward(self, e: npt.NDArray) -> npt.NDArray:
        # オプティマイザとかの実装の分離のために勾配を保存
        self.d_bias = np.sum(e, axis=AXIS_BATCH)
        self.d_weight = self.X.T @ e

        return e @ self.weight.T

    def learn(self, optimizer: Optimizer):
        optimizer.calc(self.weight, self.d_weight)
        optimizer.calc(self.bias, self.d_bias)
