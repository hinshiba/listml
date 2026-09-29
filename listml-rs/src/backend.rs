pub mod cpu;

/// 計算バックエンド
pub trait Backend {
    /// この計算機上でのデータ表現
    ///
    /// `clone` はO(1)のハンドル複製でありデータはコピーしない
    type Tensor: Clone;
    fn write(&self, data: &[f32], rows: usize, cols: usize) -> Self::Tensor;
    fn read(&self, t: &Self::Tensor) -> impl Future<Output = Result<Vec<f32>, ()>>;
    fn flush(&self);
}
