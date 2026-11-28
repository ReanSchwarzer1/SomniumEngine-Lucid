//! Depth-tested textured particles in HDR. Each viewport owns its upload buffers.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 3],
    /// Full sprite width in metres.
    pub size: f32,
    pub color: [f32; 4],
    pub tip_tint: [f32; 3],
    pub aspect: f32,
    pub uv_rect: [f32; 4],
    pub rotation: f32,
    pub texture_index: i32,
    /// Bit 0 additive; bit 1 world-up billboard; bit 2 mirrored sprite UV; bit 3 lit by the sun and sky
    /// (`colour` is then an albedo); bit 4 an untextured sprite drawn as a flame's tongue.
    pub flags: u32,
    /// Tip displacement in sprite-width units; its base stays pinned.
    pub flutter: f32,
}
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ParticleView {
    view_proj: [f32; 16],
    camera_right: [f32; 3],
    _pad0: f32,
    camera_up: [f32; 3],
    _pad1: f32,
    camera_pos: [f32; 3],
    _pad2: f32,
}
const MAX_PARTICLES: usize = 10_000;
struct ViewBuffers {
    view: wgpu::Buffer,
    instances: wgpu::Buffer,
    group: wgpu::BindGroup,
    reactive: wgpu::Texture,
    reactive_view: wgpu::TextureView,
}
pub struct ParticlePass {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    views: Vec<ViewBuffers>,
    /// Group 2: the light round a lit sprite (sun shadow, sky cube, cloud shadow). Stand-ins until `bind_sky`.
    sky_layout: wgpu::BindGroupLayout,
    sky_group: wgpu::BindGroup,
    sky_on: wgpu::Buffer,
}
impl ParticlePass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &crate::shaders::Shaders,
        global: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Textured particles"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("particle.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle view and instances"),
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
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sky_layout = sky_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Particle pipeline layout"),
            bind_group_layouts: &[Some(&layout), Some(global), Some(&sky_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Textured HDR particles"),
            layout: Some(&pipeline_layout),
            multiview_mask: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: crate::pass::postprocess::HDR_FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Max,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::RED,
                    }),
                ],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: crate::pass::visibility::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                // Reversed-Z: nearer is larger (renderer.rs `set_view`).
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Particle sprite sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let (sky_group, sky_on) = stand_in_sky(device, &sky_layout);
        Self {
            sky_layout,
            sky_group,
            sky_on,
            pipeline,
            layout,
            sampler,
            views: vec![],
        }
    }
    fn ensure_view(&mut self, device: &wgpu::Device, index: usize, size: [u32; 2]) {
        while self.views.len() <= index {
            let view = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle view"),
                size: std::mem::size_of::<ParticleView>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let instances = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle instances"),
                size: (MAX_PARTICLES * std::mem::size_of::<GpuParticle>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Particle view resources"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: view.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            let reactive = reactive_target(device, size);
            let reactive_view = reactive.create_view(&Default::default());
            self.views.push(ViewBuffers {
                view,
                instances,
                group,
                reactive,
                reactive_view,
            });
        }
        let buffers = &mut self.views[index];
        if [buffers.reactive.width(), buffers.reactive.height()] != size {
            buffers.reactive = reactive_target(device, size);
            buffers.reactive_view = buffers.reactive.create_view(&Default::default());
        }
    }
    /// Light the lit sprites (flag 8) as the scene is lit: the sun through its shadow cascades and the clouds'
    /// ground shadow, the sky's light cube as the fill. Rebuilt every frame (one small bind group), as the views
    /// may be remade when a setting changes.
    #[allow(clippy::too_many_arguments)]
    pub fn bind_sky(
        &mut self,
        device: &wgpu::Device,
        shadow_atlas: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
        env_cube: &wgpu::TextureView,
        env_sampler: &wgpu::Sampler,
        cloud_shadow: &wgpu::TextureView,
        cloud_shadow_params: &wgpu::Buffer,
    ) {
        self.sky_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle sky light"),
            layout: &self.sky_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(shadow_atlas) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(shadow_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(env_cube) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(env_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(cloud_shadow) },
                wgpu::BindGroupEntry { binding: 5, resource: cloud_shadow_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.sky_on.as_entire_binding() },
            ],
        });
    }
    /// This frame's depth-tested sprite coverage; cleared even when emission stops.
    pub fn reactive_view(&self, slot: usize) -> Option<&wgpu::TextureView> {
        self.views.get(slot).map(|view| &view.reactive_view)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        hdr: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        global: &wgpu::BindGroup,
        view_proj: glam::Mat4,
        view: glam::Mat4,
        slot: usize,
        size: [u32; 2],
        particles: &[GpuParticle],
    ) {
        if slot >= 16 {
            return;
        }
        self.ensure_view(device, slot, size);
        let buffers = &self.views[slot];
        let right = view.row(0).truncate();
        let up = view.row(1).truncate();
        queue.write_buffer(
            &buffers.view,
            0,
            bytemuck::bytes_of(&ParticleView {
                view_proj: view_proj.to_cols_array(),
                camera_right: right.to_array(),
                _pad0: 0.0,
                camera_up: up.to_array(),
                _pad1: 0.0,
                camera_pos: view.inverse().w_axis.truncate().to_array(),
                _pad2: 0.0,
            }),
        );
        let mut sorted: Vec<_> = particles
            .iter()
            .copied()
            .filter(|p| {
                p.position.iter().all(|x| x.is_finite())
                    && p.size.is_finite()
                    && p.color.iter().all(|x| x.is_finite())
            })
            .take(MAX_PARTICLES)
            .collect();
        sorted.sort_by(|a, b| {
            view.transform_point3(glam::Vec3::from(a.position))
                .z
                .total_cmp(&view.transform_point3(glam::Vec3::from(b.position)).z)
        });
        if !sorted.is_empty() {
            queue.write_buffer(&buffers.instances, 0, bytemuck::cast_slice(&sorted));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Textured HDR particles"),
            multiview_mask: None,
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: hdr,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &buffers.reactive_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &buffers.group, &[]);
        pass.set_bind_group(1, global, &[]);
        pass.set_bind_group(2, &self.sky_group, &[]);
        pass.draw(0..6, 0..sorted.len() as u32);
    }
}

fn sky_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let fragment = wgpu::ShaderStages::FRAGMENT;
    let texture = |binding, sample_type, view_dimension| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension, multisampled: false },
        count: None,
    };
    let sampler = |binding, kind| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Sampler(kind),
        count: None,
    };
    let uniform = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: fragment,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    let float = wgpu::TextureSampleType::Float { filterable: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Particle sky light BGL"),
        entries: &[
            texture(0, wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2),
            sampler(1, wgpu::SamplerBindingType::Comparison),
            texture(2, float, wgpu::TextureViewDimension::Cube),
            sampler(3, wgpu::SamplerBindingType::Filtering),
            texture(4, float, wgpu::TextureViewDimension::D2),
            uniform(5),
            uniform(6),
        ],
    })
}

/// One-texel stand-ins for the sky group, with `sky_on` 0 (the old fill), and the buffer `bind_sky` binds with 1.
fn stand_in_sky(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> (wgpu::BindGroup, wgpu::Buffer) {
    use wgpu::util::DeviceExt;
    let texel = |label, format, layers| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
    };
    let depth = texel("Particle sky stand-in depth", wgpu::TextureFormat::Depth32Float, 1).create_view(&Default::default());
    let cube = texel("Particle sky stand-in cube", wgpu::TextureFormat::Rgba16Float, 6).create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    let flat = texel("Particle sky stand-in shadow", wgpu::TextureFormat::Rgba16Float, 1).create_view(&Default::default());
    let compare = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Particle sky stand-in compare"),
        compare: Some(wgpu::CompareFunction::GreaterEqual),
        ..Default::default()
    });
    let filter = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("Particle sky stand-in filter"), ..Default::default() });
    let zeros = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particle sky off"),
        contents: bytemuck::cast_slice(&[0.0f32; 4]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let on = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Particle sky on"),
        contents: bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0]),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Particle sky stand-ins"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&depth) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&compare) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&cube) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&filter) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&flat) },
            wgpu::BindGroupEntry { binding: 5, resource: zeros.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: zeros.as_entire_binding() },
        ],
    });
    (group, on)
}

fn reactive_target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Particle visible coverage"),
        size: wgpu::Extent3d {
            width: size[0].max(1),
            height: size[1].max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}
