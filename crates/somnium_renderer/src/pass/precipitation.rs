//! Rain, snow and hail near the eye (Sunscar weather, 11 Nov 2026): `precipitation.wgsl`'s field of drops.
//!
//! A game sets [`PrecipitationPass::settings`]; the default draws nothing and costs nothing. Far rain is not this
//! pass's: beyond a few metres a drop is thinner than a pixel and almost transparent, and what the eye sees there is
//! the air's haze (the fog) and, at a distance, the shaft under the cloud (a storm body in the cloud march).

use glam::Vec3;

/// The most drops a frame draws.
pub const MAX_DROPS: u32 = 200_000;

/// What falls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Falling {
    /// A lens: a pale streak that shows the sky behind it and glitters toward the sun.
    #[default]
    Rain,
    /// A slow white flake.
    Snow,
    /// A fast white stone.
    Hail,
}

/// The field of drops round the eye.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrecipitationSettings {
    /// Drops in the box. 0 draws nothing.
    pub count: u32,
    pub falling: Falling,
    /// The box's half-size, metres. Drops further off are thinner than a pixel: 5 m is enough for rain.
    pub extent: Vec3,
    /// How far ahead of the eye the box's centre sits, along the way it looks on the ground, as a share of the
    /// box's half-size (the drops behind the eye are wasted).
    pub lead: f32,
    /// The mean drop's velocity, metres a second: the fall and the wind's share.
    pub velocity: Vec3,
    /// How far a drop wanders from its fall, metres a second (snow drifts; rain does not).
    pub wander: f32,
    /// The mean drop's diameter, metres (rain 0.002, snow 0.006, hail 0.012).
    pub size: f32,
    /// The shutter, seconds: a drop draws the streak it falls in this time (a film camera's 1/48).
    pub exposure: f32,
    pub tint: Vec3,
    /// A scale on how much the drops cover (1 as physics gives it).
    pub opacity: f32,
}

impl Default for PrecipitationSettings {
    fn default() -> Self {
        Self {
            count: 0,
            falling: Falling::Rain,
            extent: Vec3::new(5.0, 4.0, 5.0),
            lead: 0.6,
            velocity: Vec3::new(0.0, -7.5, 0.0),
            wander: 0.0,
            size: 0.002,
            exposure: 1.0 / 48.0,
            tint: Vec3::ONE,
            opacity: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    pixel_angle: f32,
    centre: [f32; 3],
    time: f32,
    extent: [f32; 3],
    count: f32,
    velocity: [f32; 3],
    exposure: f32,
    camera_velocity: [f32; 3],
    wander: f32,
    sun_direction: [f32; 3],
    kind: f32,
    sun_light: [f32; 3],
    size: f32,
    tint: [f32; 3],
    opacity: f32,
    flash_pos: [[f32; 4]; 4],
    flash_color: [[f32; 4]; 4],
}

pub struct PrecipitationPass {
    pub settings: PrecipitationSettings,
    /// Lightning and lamps that light the drops this frame: where, and candela by colour (the first four count).
    pub flashes: Vec<(Vec3, Vec3)>,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    /// The eye and the renderer's clock a frame ago, for the eye's velocity and the frame's length.
    last_camera: Option<Vec3>,
    last_time: f32,
    /// The clock the drops fall by, kept under an hour so their places stay exact.
    clock: f32,
}

impl PrecipitationPass {
    pub fn new(device: &wgpu::Device, shaders: &crate::shaders::Shaders) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("precipitation.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("precipitation.wgsl").into()),
        });
        let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture = |binding, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Precipitation BGL"),
            entries: &[
                uniform(0, wgpu::ShaderStages::VERTEX),
                texture(1, wgpu::TextureViewDimension::Cube),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture(3, wgpu::TextureViewDimension::D2),
                uniform(4, wgpu::ShaderStages::VERTEX),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Precipitation PL"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Precipitation"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                // As the sprites: over the scene, and into FSR's reactive mask.
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: crate::pass::postprocess::HDR_FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Max },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::pass::visibility::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                // Reversed depth: nearer is larger.
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            settings: PrecipitationSettings::default(),
            flashes: Vec::new(),
            pipeline,
            layout,
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Precipitation Params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            bind_group: None,
            last_camera: None,
            last_time: 0.0,
            clock: 0.0,
        }
    }

    /// Bind the sky light's cube and the clouds' ground shadow. Built once.
    pub fn ensure_bind_group(
        &mut self,
        device: &wgpu::Device,
        env_cube: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        cloud_shadow: &wgpu::TextureView,
        cloud_shadow_params: &wgpu::Buffer,
    ) {
        if self.bind_group.is_some() {
            return;
        }
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Precipitation BG"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_cube) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(cloud_shadow) },
                wgpu::BindGroupEntry { binding: 4, resource: cloud_shadow_params.as_entire_binding() },
            ],
        }));
    }

    /// Draw the drops over the scene. `view_proj` is camera-relative and unjittered; `reactive` the sprites' mask
    /// for this view (already cleared by them this frame); `time` the renderer's clock, seconds.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hdr: &wgpu::TextureView,
        reactive: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        size: (u32, u32),
        view_proj: glam::Mat4,
        camera_pos: Vec3,
        look: Vec3,
        sun_direction: Vec3,
        sun_light: Vec3,
        time: f32,
    ) {
        let dt = (time - self.last_time).clamp(0.0, 0.25);
        self.last_time = time;
        // The clock runs whether or not anything falls, so rain that starts again is not where it stopped.
        self.clock = (self.clock + dt).rem_euclid(3600.0);
        let camera_velocity = match self.last_camera {
            // A cut is not a velocity.
            Some(last) if dt > 1e-4 && last.distance(camera_pos) < 30.0 => (camera_pos - last) / dt,
            _ => Vec3::ZERO,
        };
        self.last_camera = Some(camera_pos);
        let s = self.settings;
        let (Some(bind_group), true) = (self.bind_group.as_ref(), s.count > 0) else {
            return;
        };
        let count = s.count.min(MAX_DROPS);
        let ahead = Vec3::new(look.x, 0.0, look.z).normalize_or_zero();
        let mut flash_pos = [[0.0; 4]; 4];
        let mut flash_color = [[0.0; 4]; 4];
        for (i, (at, candela)) in self.flashes.iter().take(4).enumerate() {
            flash_pos[i] = [at.x, at.y, at.z, 1.0];
            flash_color[i] = [candela.x, candela.y, candela.z, 0.0];
        }
        #[allow(clippy::cast_precision_loss)]
        let height = size.1.max(1) as f32;
        // A perspective's second row is its 1 / tan(half the vertical field of view), turned by the view's rotation.
        let tan_half_fov_y = 1.0 / Vec3::new(view_proj.x_axis.y, view_proj.y_axis.y, view_proj.z_axis.y).length().max(1e-4);
        #[allow(clippy::cast_precision_loss)]
        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                view_proj: view_proj.to_cols_array_2d(),
                camera_pos: camera_pos.to_array(),
                pixel_angle: 2.0 * tan_half_fov_y / height,
                centre: (camera_pos + ahead * (s.extent.x.min(s.extent.z) * s.lead.clamp(0.0, 1.0))).to_array(),
                time: self.clock,
                extent: s.extent.max(Vec3::splat(0.5)).to_array(),
                count: count as f32,
                velocity: s.velocity.to_array(),
                exposure: s.exposure.clamp(1e-4, 0.1),
                camera_velocity: camera_velocity.to_array(),
                wander: s.wander.max(0.0),
                sun_direction: sun_direction.normalize_or_zero().to_array(),
                kind: match s.falling {
                    Falling::Rain => 0.0,
                    Falling::Snow => 1.0,
                    Falling::Hail => 2.0,
                },
                sun_light: sun_light.to_array(),
                size: s.size.max(1e-4),
                tint: s.tint.to_array(),
                opacity: s.opacity.max(0.0),
                flash_pos,
                flash_color,
            }),
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Precipitation"),
            multiview_mask: None,
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: hdr,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: reactive,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: depth, depth_ops: None, stencil_ops: None }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.draw(0..6, 0..count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_matches_the_shaders_layout() {
        // A mat4, eight vec3 + f32 rows, two arrays of four vec4.
        assert_eq!(std::mem::size_of::<Params>(), 64 + 8 * 16 + 2 * 64);
    }

    #[test]
    fn nothing_falls_by_default() {
        assert_eq!(PrecipitationSettings::default().count, 0);
    }
}
