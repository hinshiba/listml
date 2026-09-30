use crate::backend::Backend;

/// 行列積 `op(a) * op(b)`．`ta` , `tb` が真なら該当の引数を転置して扱う
pub trait MatMul: Backend {
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor, ta: bool, tb: bool) -> Self::Tensor;
}
