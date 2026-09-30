#![allow(dead_code, unused_macros)]

use std::iter::zip;

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

pub async fn init() -> (wgpu::Device, wgpu::Queue) {
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

pub async fn read_f32(device: &wgpu::Device, staging: &wgpu::Buffer) -> Vec<f32> {
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

/// 基準値に近い値であることを表明する
///
/// `val` と `base` の差が
/// 絶対誤差 `atol` (Absolute Tolerance) と `rtol` (Relative Tolerance) の
/// 和以下であることを表明する
#[track_caller]
pub fn assert_close(val: &[f32], base: &[f32], atol: f32, rtol: f32) {
    assert_eq!(
        val.len(),
        base.len(),
        "not eq len. val.len()={}, but base.len()={}",
        val.len(),
        base.len()
    );

    for (i, (v, b)) in zip(val, base).enumerate() {
        assert!(
            (v - b).abs() <= atol + rtol * b.abs(),
            "not close. error: {}, but tolerance: {} in index {}.",
            (v - b).abs(),
            atol + rtol * b.abs(),
            i
        )
    }
}
