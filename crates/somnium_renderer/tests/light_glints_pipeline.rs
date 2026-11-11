//! The lamp-point pass on a real device (sunscar-photoreal, 12 Nov 2026). Its pipeline is built when the renderer
//! starts, even with the points off, so a layout or shader the device refuses would stop every launch. A far lamp
//! lands as its intensity over the distance squared over one pixel's solid angle, shared among the four pixels round
//! where it falls, the same light wherever inside a pixel it sits; behind the scene's depth it adds nothing.
use somnium_renderer::{
    cluster::GpuLocalLight,
    pass::{light_glints::LightGlintsPass, postprocess::HDR_FORMAT},
    shaders::Shaders,
};
#[path = "support/gpu.rs"]
mod gpu;

const SIZE: u32 = 32;

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => f32::NAN,
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// One frame: the scene's reversed depth cleared to `depth` everywhere, the lamps drawn, the HDR's red read back.
fn draw(device: &wgpu::Device, queue: &wgpu::Queue, pass: &LightGlintsPass, lamps: &[GpuLocalLight], depth: f32) -> Vec<f32> {
    let hdr = gpu::texture(
        device,
        SIZE,
        HDR_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let depth_tex = gpu::texture(
        device,
        SIZE,
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    );
    let hdr_view = hdr.create_view(&Default::default());
    let depth_view = depth_tex.create_view(&Default::default());
    let lights = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (std::mem::size_of::<GpuLocalLight>() * lamps.len().max(1)) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&lights, 0, bytemuck::cast_slice(lamps));
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            multiview_mask: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &hdr_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(depth), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    let eye = glam::Vec3::ZERO;
    let view = glam::Mat4::look_at_rh(eye, glam::Vec3::NEG_Z, glam::Vec3::Y);
    let proj = glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 10000.0);
    pass.record(device, queue, &mut encoder, &hdr_view, &depth_view, &lights, lamps.len() as u32, view, proj, eye, (SIZE, SIZE), 1.0);
    // Rgba16Float: 8 bytes a pixel.
    let stride = (SIZE * 8).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(stride * SIZE),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        hdr.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(SIZE) },
        },
        hdr.size(),
    );
    queue.submit([encoder.finish()]);
    let (send, recv) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| send.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    recv.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range().unwrap();
    mapped
        .chunks(stride as usize)
        .flat_map(|row| {
            row[..SIZE as usize * 8]
                .chunks(8)
                .map(|px| f16_to_f32(u16::from_le_bytes([px[0], px[1]])))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn lamp(at: glam::Vec3, intensity: f32, radius: f32) -> GpuLocalLight {
    let mut l: GpuLocalLight = bytemuck::Zeroable::zeroed();
    l.position_ws = at.to_array();
    l.range = 8.0;
    l.color = [intensity; 3];
    l.light_type = 0;
    l.radius = radius;
    l
}

#[test]
#[ignore = "Requires a graphics adapter; run for lamp-point (light glints) changes"]
fn far_lamps_land_as_steady_points_and_hide_behind_the_scene() {
    let (device, queue) = gpu::device_with(wgpu::Features::empty());
    let pass = LightGlintsPass::new(&device, &Shaders::new());
    // A lamp 50 m off, a flame's 3 cm: far under a pixel (a pixel is about 3.4 cm across at 1 m here, 1.7 m at 50 m).
    let d = 50.0_f32;
    let intensity = 1000.0;
    let pixel = 2.0 / ((1.0_f32 / 0.5_f32.tan()) * SIZE as f32);
    let expected = intensity * (-d / 30000.0).exp() / (d * d) / (pixel * pixel);
    let mut sums = Vec::new();
    for shift in [0.0_f32, 0.25, 0.5, 0.8] {
        // Slide the lamp across a pixel: the light it lands with must not change.
        let at = glam::Vec3::new(shift * pixel * d, 0.37 * pixel * d, -d);
        let red = draw(&device, &queue, &pass, &[lamp(at, intensity, 0.03)], 0.0);
        let sum: f32 = red.iter().sum();
        let lit = red.iter().filter(|v| **v > 1.0e-3 * expected).count();
        assert!(lit <= 4, "shift {shift}: the lamp spread over {lit} pixels, not four at most");
        assert!((sum - expected).abs() < 0.02 * expected, "shift {shift}: landed {sum}, expected {expected}");
        sums.push(sum);
    }
    let (lo, hi) = sums.iter().fold((f32::MAX, f32::MIN), |(lo, hi), s| (lo.min(*s), hi.max(*s)));
    assert!(hi - lo < 0.01 * expected, "the lamp's light changed as it slid across a pixel: {sums:?}");
    // A wall 5 m off covers the lamp (reversed infinite depth: near / view depth).
    let hidden = draw(&device, &queue, &pass, &[lamp(glam::Vec3::new(0.0, 0.0, -d), intensity, 0.03)], 0.1 / 5.0);
    assert!(hidden.iter().all(|v| *v == 0.0), "a lamp behind the scene's depth showed");
    // A lamp near enough to draw its own source adds nothing here.
    let near = draw(&device, &queue, &pass, &[lamp(glam::Vec3::new(0.0, 0.0, -2.0), intensity, 0.2)], 0.0);
    assert!(near.iter().all(|v| *v == 0.0), "a lamp wider than a pixel was drawn again as a point");
    // No lamps, or strength 0, records nothing (the renderer's default).
    let none = draw(&device, &queue, &pass, &[], 0.0);
    assert!(none.iter().all(|v| *v == 0.0));
}
