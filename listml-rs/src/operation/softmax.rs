use crate::backend::Backend;

/// 行ごとのsoftmax
pub trait Softmax: Backend {
    fn softmax(&self, x: &Self::Tensor) -> Self::Tensor;
    /// 入力 `x` に関する勾配．`y` は `softmax(x)`，`e` は `y` に関する勾配
    fn softmax_grad(&self, y: &Self::Tensor, e: &Self::Tensor) -> Self::Tensor;
}
