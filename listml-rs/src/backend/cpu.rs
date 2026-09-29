use std::convert::Infallible;

use crate::{
    backend::Backend,
    operation::{self, rectifier::Rectifier},
};

pub struct CpuBackend {}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }
}

pub struct CpuTensor {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;
    fn write(&self, data: &[f32], rows: usize, cols: usize) -> Self::Tensor {
        assert_eq!(data.len(), rows * cols);
        Self::Tensor {
            data: data.to_vec(),
            rows,
            cols,
        }
    }

    async fn read(&self, t: &Self::Tensor) -> Result<Vec<f32>, ()> {
        Ok(t.data.clone())
    }

    fn flush(&self) {}
}

impl Rectifier for CpuBackend {
    fn rectifier(&self, x: &Self::Tensor) -> Self::Tensor {
        Self::Tensor {
            data: x.data.iter().map(|&v| v.max(0.)).collect(),
            rows: x.rows,
            cols: x.cols,
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{Backend, cpu::CpuBackend},
        operation::rectifier::Rectifier,
    };

    #[pollster::test]
    async fn op_relu() {
        let cpu = CpuBackend::new();
        let data: Vec<f32> = vec![-1., -0.001, 0., 0.001, 1.];
        let expect: Vec<f32> = vec![0., 0., 0., 0.001, 1.];

        let t = cpu.write(data.as_slice(), data.len(), 1);
        let r = cpu.rectifier(&t);
        cpu.flush();
        assert_eq!(expect, cpu.read(&r).await.unwrap());
    }

    #[pollster::test]
    async fn layer_relu() {
        let cpu = CpuBackend::new();
        let data: Vec<f32> = vec![-1., -0.001, 0., 0.001, 1.];
        let expect: Vec<f32> = vec![0., 0., 0., 0.001, 1.];

        let t = cpu.write(data.as_slice(), data.len(), 1);
        let r = cpu.rectifier(&t);
        cpu.flush();
        assert_eq!(expect, cpu.read(&r).await.unwrap());
    }
}
