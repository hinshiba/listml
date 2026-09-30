use crate::backend::Backend;

/// 交差エントロピー．`y` は予測，`t` は真の値
pub trait CrossEntropy: Backend {
    /// 1 x 1 のテンソルを返す
    fn cross_entropy(&self, y: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor;
    /// `y` に関する勾配
    fn cross_entropy_grad(&self, y: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor;
}
