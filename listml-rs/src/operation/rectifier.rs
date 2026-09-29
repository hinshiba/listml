use crate::backend::Backend;

/// Rectified linear function
pub trait Rectifier: Backend {
    fn rectifier(&self, x: &Self::Tensor) -> Self::Tensor;
}
