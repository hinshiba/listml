use crate::{backend::Backend, layer::Layer, operation::rectifier::Rectifier};
use std::sync::Arc;

pub struct Relu<B: Backend> {
    /// バックエンド
    ///
    /// キャッシュと同時に持っておかないとキャッシュがそのバックエンド上にないということが生じうる
    b: Arc<B>,

    /// `forward` での出力値
    ///
    /// boolの値より実装がシンプル
    y: Option<B::Tensor>,
}

impl<B: Rectifier> Relu<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self {
            b: backend,
            y: None,
        }
    }
}

impl<B: Rectifier> Layer<B> for Relu<B> {
    fn forward(&mut self, x: &B::Tensor) -> B::Tensor {
        let y = self.b.rectifier(x);
        self.y = Some(y.clone());
        y
    }

    fn backward(&mut self, e: &B::Tensor) -> B::Tensor {
        self.b
            .gate(self.y.as_ref().expect("y is None. Do first forward."), e)
    }
}
