use crate::backend::Backend;

pub mod relu;

/// NNの単一の層を表現する
pub trait Layer<B: Backend> {
    fn forward(&mut self, x: &B::Tensor) -> B::Tensor;
    fn backward(&mut self, e: &B::Tensor) -> B::Tensor;
}
