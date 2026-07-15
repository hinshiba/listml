# listml

NumPyだけで実装したミニマルなニューラルネットのライブラリ．
全結合層と基本的な活性化 / 損失 / 最適化を提供する．

## インストール

```
uv add "git+https://github.com/hinshiba/listml.git"
```

## 使い方

```python
import numpy as np
from listml.layer import FullConn, pred
from listml.activation import Relu, Softmax
from listml.loss import CrossEntropy
from listml.optimizer import StochasticGradientDescent

network = [
    FullConn(64),
    Relu(),
    FullConn(10),
    Softmax(),
]

# 適当なデータを流して各層を初期化する
batch_size, in_dim = 32, 784
dummy = np.empty((batch_size, in_dim))
for layer in network:
    dummy = layer.prepare(dummy)

loss_func = CrossEntropy()
optimizer = StochasticGradientDescent(np.float32(2e-3))

# X: (batch_size, in_dim), labels: one-hot (batch_size, 10)
def step(X, labels):
    out = pred(network, X)
    loss = loss_func.loss(out, labels)

    e = loss_func.backward(loss)
    for layer in reversed(network):
        e = layer.backward(e)
    for layer in network:
        layer.learn(optimizer)
    return loss
```

## 構成

- `layer` : 全結合層 `FullConn`，順伝播ヘルパ `pred`，`Layer` プロトコル
- `activation` : `Relu`，`Softmax`
- `loss` : `CrossEntropy`
- `optimizer` : `StochasticGradientDescent`
