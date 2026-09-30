use crate::common::{init, read_f32};
use wgpu::util::DeviceExt;

#[macro_use]
mod common;

gpu_test!(relu_works, async {
    let (device, queue) = init().await;

    let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let n = input.len() as u32;
    let size = (input.len() * std::mem::size_of::<f32>()) as u64;

    let x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"),
        contents: bytemuck::cast_slice(&input),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let y = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n: u32,
    }

    let p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&Params { n }),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let module = device.create_shader_module(wgpu::include_wgsl!("relu.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("relu"),
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relu"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: y.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: p.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&y, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);

    let got = read_f32(&device, &staging).await;

    assert_eq!(got, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
});
