//! Small lights that hold up under a pixel (`light_glints.wgsl`; sunscar-photoreal, 12 Nov 2026).
//!
//! A far lamp's flame or glass is smaller than a pixel, so the jittered, upscaled picture finds it in some frames and
//! not in others: it vanishes or flickers. This pass draws every visible point and spot lamp as its own light spread
//! bilinearly over the pixels round where it falls, into the render-resolution HDR before bloom, the lens's glare and
//! the upscale, so the far lamps of a night hold steady and carry their halo. Off unless a scene asks
//! (PostProcess `light_glints`): TSF's lamps keep their picture.

use crate::pass::postprocess::HDR_FORMAT;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlintParams {
    view_proj: [[f32; 4]; 4],
    camera: [f32; 4],
    frame: [f32; 4],
    reach: [f32; 4],
}

/// The farthest lamp drawn, metres: past it the haze has the light and the halo would be a speck.
const FARTHEST_M: f32 = 4000.0;
/// The haze's extinction per metre for a lamp's light on its way to the eye: about 30 km of visibility.
const EXTINCTION_PER_M: f32 = 1.0 / 30000.0;

pub struct LightGlintsPass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

impl LightGlintsPass {
    #[must_use]
    pub fn new(device: &wgpu::Device, shaders: &crate::shaders::Shaders) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Glints"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("light_glints.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Light Glints"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Glints"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Light Glints"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState { color: add, alpha: add }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..wgpu::PrimitiveState::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Glints Params"),
            size: std::mem::size_of::<GlintParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { pipeline, layout, params }
    }

    /// Draw the frame's `count` local lights (`lights`, the cluster grid's `GpuLocalLight` buffer) onto `hdr`, tested
    /// against `depth` (the visibility pass's reversed-Z depth, at `size`). `strength` 0 draws nothing.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hdr: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        lights: &wgpu::Buffer,
        count: u32,
        view: glam::Mat4,
        proj: glam::Mat4,
        camera: glam::Vec3,
        size: (u32, u32),
        strength: f32,
    ) {
        if count == 0 || strength <= 0.0 || size.0 == 0 || size.1 == 0 {
            return;
        }
        // A pixel's angle from the vertical field of view (proj[1][1] = 1 / tan(fov_y / 2)), and its solid angle.
        let pixel = 2.0 / (proj.y_axis.y.abs().max(1.0e-6) * size.1 as f32);
        let params = GlintParams {
            view_proj: (proj * view).to_cols_array_2d(),
            camera: [camera.x, camera.y, camera.z, crate::view::near_plane(proj)],
            frame: [size.0 as f32, size.1 as f32, 1.0 / (pixel * pixel), strength],
            reach: [EXTINCTION_PER_M, FARTHEST_M, pixel, 0.0],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Light Glints"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Light Glints"),
            multiview_mask: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: hdr,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.draw(0..4, 0..count);
    }
}
