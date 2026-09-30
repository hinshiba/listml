use crate::backend::Backend;
use std::sync::Arc;

pub mod operation;

pub struct CpuBackend {}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Clone)]
pub struct CpuTensor(Arc<CpuTensorData>);

struct CpuTensorData {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl CpuTensor {
    fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self(Arc::new(CpuTensorData { data, rows, cols }))
    }
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;
    fn write(&self, data: &[f32], rows: usize, cols: usize) -> Self::Tensor {
        CpuTensor::new(data.to_vec(), rows, cols)
    }

    async fn read(&self, t: &Self::Tensor) -> Result<Vec<f32>, ()> {
        Ok(t.0.data.clone())
    }

    fn flush(&self) {}
}

#[cfg(test)]
mod test {
    use crate::{
        backend::{Backend, cpu::CpuBackend},
        layer::{Layer, relu::Relu},
        operation::rectifier::Rectifier,
    };
    use std::sync::Arc;

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
        let cpu = Arc::new(CpuBackend::new());
        let data: Vec<f32> = vec![-1., -0.001, 0., 0.001, 1.];
        let expect: Vec<f32> = vec![0., 0., 0., 0.001, 1.];

        let mut relu = Relu::new(cpu.clone());
        let x = cpu.write(data.as_slice(), data.len(), 1);

        let r = relu.forward(&x);

        cpu.flush();
        assert_eq!(expect, cpu.read(&r).await.unwrap());
    }
}
