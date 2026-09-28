use wgpu::util::DeviceExt;

macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[cfg(not(target_arch = "wasm32"))]
        #[test]
        fn $name() {
            pollster::block_on($body)
        }

        #[cfg(target_arch = "wasm32")]
        #[wasm_bindgen_test::wasm_bindgen_test]
        async fn $name() {
            $body.await
        }
    };
}

// ブラウザで実行させる
#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

async fn init() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("missing WebGPU adapter");

    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("hello"),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        })
        .await
        .expect("cannot create device")
}

async fn read_f32(device: &wgpu::Device, staging: &wgpu::Buffer) -> Vec<f32> {
    let slice = staging.slice(..); // ..ってなんだこれ 
    let (tx, rx) = futures_channel::oneshot::channel();
    // mapとは 多分知っているmapとは違う
    // そもそも何をしようとしているのか
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap(); // poll typeとは

    rx.await.unwrap().unwrap();
    let data = bytemuck::cast_slice(&slice.get_mapped_range().unwrap()).to_vec();
    staging.unmap();
    data
}

gpu_test!(relu_works, async {
    let (device, queue) = init().await;

    let input: Vec<f32> = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
    let n = input.len() as u32;
    let size = (input.len() * std::mem::size_of::<f32>()) as u64;

    // 入力で初期値があるので BufferInit
    let x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("x"), // これは多分意味ないということだよね
        // sizeは初期値から自動推測?
        contents: bytemuck::cast_slice(&input), // これは as &[u8]とは違うのか
        usage: wgpu::BufferUsages::STORAGE,     // usageとは?
    });

    let y = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("y"),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, // 同上
        mapped_at_creation: false,                                         // ???
    });

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        n: u32,
    }
    // これは何?
    let p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&Params { n }), // これは as &[u8]とは違うのか cast_sliceとも
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // map_readでcopy_dstの読み込み先
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ---

    // shaderモジュールとは?
    let module = device.create_shader_module(wgpu::include_wgsl!("relu.wgsl"));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("relu"),
        layout: None, // ?
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // バインドとは??
    // wgslの
    // @group(0) @binding(0) var<storage, read> x: array<f32>;
    // と対応させている??
    // 関数の引数として渡す感じではなくグローバル変数みたいな感じか?
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

    // ---

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]); // 最後の引数??
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1); // ??? @workgroup_size(64)と関係がある??
    } // 可変参照を用いるpassをdropさせる
    encoder.copy_buffer_to_buffer(&y, 0, &staging, 0, size);
    queue.submit([encoder.finish()]); // これは何 encoder.finishとは
    // てかpass先にdropしたけどsubmitしなくてもいいのか??

    // ---

    let got = read_f32(&device, &staging).await;

    assert_eq!(got, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
});
