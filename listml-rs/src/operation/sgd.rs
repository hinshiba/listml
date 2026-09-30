use crate::backend::Backend;

/// 確率的勾配降下法．`w - lr * g` を返す
pub trait Sgd: Backend {
    fn sgd(&self, w: &Self::Tensor, g: &Self::Tensor, lr: f32) -> Self::Tensor;
}
