use crate::backend::Backend;

/// `0 < sig` の要素だけ `x` を通す
pub trait Gate: Backend {
    fn gate(&self, sig: &Self::Tensor, x: &Self::Tensor) -> Self::Tensor;
}
