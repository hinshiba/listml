use wgpu::util::DeviceExt;

use crate::backend::Backend;

pub struct WebGpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[derive(Clone)]
struct WebGpuBuffer {
    data: wgpu::Buffer,
    rows: usize,
    cols: usize,
}

impl Backend for WebGpuBackend {
    type Tensor = WebGpuBuffer;

    fn write(&self, data: &[f32], rows: usize, cols: usize) -> Self::Tensor {
        WebGpuBuffer {
            data: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            rows,
            cols,
        }
    }

    async fn read(&self, t: &Self::Tensor) -> Result<Vec<f32>, ()> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (t.rows * t.cols) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let slice = staging.slice(..);

        let (tx, rx) = futures_channel::oneshot::channel();

        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        // wasmでは自動
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        rx.await.unwrap().unwrap();
        let data = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
        staging.unmap();
        Ok(data)
    }

    fn flush(&self) {
        self.queue.submit(command_buffers)
    }
}
