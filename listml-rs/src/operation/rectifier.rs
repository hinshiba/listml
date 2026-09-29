use crate::backend::Backend;

/// Rectified linear function
pub trait Rectifier: Backend {
    fn rectifier(&self, x: &Self::Tensor) -> Self::Tensor;
    fn gate(&self, sig: &Self::Tensor, x: &Self::Tensor) -> Self::Tensor;
}
