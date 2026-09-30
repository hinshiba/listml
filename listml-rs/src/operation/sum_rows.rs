use crate::backend::Backend;

/// `x` を行方向に合計し，1 x cols のテンソルを返す
pub trait SumRows: Backend {
    fn sum_rows(&self, x: &Self::Tensor) -> Self::Tensor;
}
