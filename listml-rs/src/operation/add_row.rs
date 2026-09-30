use crate::backend::Backend;

/// `x` の各行に 1 x cols の `row` を加える
pub trait AddRow: Backend {
    fn add_row(&self, x: &Self::Tensor, row: &Self::Tensor) -> Self::Tensor;
}
