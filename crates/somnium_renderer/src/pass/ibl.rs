//! Phase 19A: image-based lighting — environment cubemap generation.
//!
//! ## Reference Architecture
//!
//! Brian Karis, *Real Shading in Unreal Engine 4* (SIGGRAPH 2013) — the
//! split-sum approximation. This file builds its first half: a prefiltered
//! environment map where mip `i` holds the radiance convolved for roughness
//! `i / (mips - 1)`.
//!
//! The environment is captured from the engine's own procedural sky rather
//! than an HDRI asset, so reflections always agree with the sky the camera
//! sees, and they stay correct when the sun moves.
//!
//! Regeneration is cheap but not free, so it runs at startup and only again
//! when the sun direction or colour actually changes.

/// Cubemap face resolution.
const CUBE_SIZE: u32 = 256;
/// Mip levels; mip `i` is prefiltered for roughness `i / (MIP_COUNT - 1)`.
const MIP_COUNT: u32 = 6;

/// While a day clock runs (`generate_if_needed`'s `moving`): faces and mips of the second cube built a frame (36 in
/// all, mip 0's six first), then the frames over which the bound cube crossfades to it. About a second a cycle at 60
/// fps, in which a clock that runs a day an hour turns the sun 0.08 degrees: the sky light follows it in small
/// straight steps and never jumps. A clock run fast (the sun half a degree or more ahead of the cube when a cycle
/// starts) takes the quick cycle, twelve faces a frame and six to blend, so the light keeps up within a degree.
const MOVING_JOBS_PER_FRAME: u32 = 2;
const MOVING_BLEND_FRAMES: u32 = 30;
const MOVING_QUICK_JOBS_PER_FRAME: u32 = 12;
const MOVING_QUICK_BLEND_FRAMES: u32 = 6;
/// cos 0.5 degrees: a sun this far ahead of the bound cube's takes the quick cycle.
const MOVING_QUICK_COS: f32 = 0.999_962;
/// A sun that moved further than this since the bound cube was built (a key that moves the clock an hour) rebuilds
/// it whole at once, as a still scene's does: cos 5 degrees.
const MOVING_JUMP_COS: f32 = 0.996_194_7;

/// What a cube was built for: the sun's direction, its colour and the overcast's bits; the night; the clouds' capture.
type SkyKey = (([f32; 3], [f32; 3], u32), NightSky, Option<u64>);

/// Where a running clock's rebuild stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MovingPhase {
    Idle,
    /// Building the second cube: the next of its 36 faces and mips, and how many a frame.
    Building { job: u32, per_frame: u32, blend: u32 },
    /// Crossfading the bound cube to it: frames done, of how many.
    Blending { done: u32, of: u32 },
}

/// The second cube a running day clock rebuilds into, a few faces a frame, while the bound cube is left alone; then
/// the bound cube crossfades to it. Made the first time a clock runs, so a scene whose sun stands still never has one.
struct MovingSky {
    _texture: wgpu::Texture,
    /// Render targets, indexed `[mip * 6 + face]`, as the bound cube's.
    face_mip_views: Vec<wgpu::TextureView>,
    /// The prefilter, reading this cube's own mip 0.
    prefilter_bind_group: wgpu::BindGroup,
    mip0_view: wgpu::TextureView,
    /// Each face and mip as the crossfade's source, in `face_mip_views`' order.
    blend_bind_groups: Vec<wgpu::BindGroup>,
    /// What it is being built for, or was last built for.
    key: Option<SkyKey>,
    phase: MovingPhase,
}

/// Photometric luminance of a linear-RGB light colour (Rec. 709 weights).
fn sun_luminance(color: glam::Vec3) -> f32 {
    color.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722))
}

/// Uniform matching `GenParams` in `ibl_gen.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GenParams {
    face: u32,
    roughness: f32,
    _src_size: f32,
    /// Cloud-deck coverage, 0..1; the sky is blended towards overcast by it.
    overcast: f32,
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    /// `NightSky`: glow (cd/m²), aurora, Halo Road, Thread.
    night: [f32; 4],
    /// `NightSky::tint`, then padding.
    night_tint: [f32; 4],
    /// x: 1 lays the clouds' capture over the sky (`set_cloud_capture`).
    clouds: [f32; 4],
}

/// A night sky with no moon (Rimeborn's Undersun under the Lid), drawn into the environment cubemap so the background,
/// the ambient light and the reflections all carry it: a glow over the whole dome, the Halo Road (a ring of arcs
/// through the hidden sun), the Thread (a thin line from the horizon below the sun toward it) and aurora low on the
/// side away from the sun. All zero (the default) draws nothing, which is every scene that does not ask for it.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NightSky {
    /// The dome's glow at the zenith, cd/m² (a dark Earth night is about 0.0002; 0.03 reads, under auto-exposure).
    pub glow: f32,
    /// Aurora, 0..1 of a strong display.
    pub aurora: f32,
    /// The Halo Road's arcs, 0..1.
    pub halo_road: f32,
    /// The Thread, 0..1.
    pub thread: f32,
    /// The glow's colour (linear RGB).
    pub tint: [f32; 3],
}

/// Owns the environment cubemap and the passes that fill it.
pub struct IblPass {
    #[allow(dead_code)]
    cubemap: wgpu::Texture,
    /// Full cube view (all mips) — bound into the shading pass.
    pub cube_view: wgpu::TextureView,
    /// The night sky drawn into the cubemap (`NightSky`); a change rebuilds it.
    pub night: NightSky,
    /// Trilinear sampler for the environment map.
    pub sampler: wgpu::Sampler,

    /// Render targets, indexed `[mip * 6 + face]`.
    face_mip_views: Vec<wgpu::TextureView>,

    sky_pipeline: wgpu::RenderPipeline,
    prefilter_pipeline: wgpu::RenderPipeline,
    sky_bind_group: wgpu::BindGroup,
    /// Kept to rebuild the prefilter's bind group when the clouds' capture arrives (`set_cloud_capture`).
    prefilter_bgl: wgpu::BindGroupLayout,
    mip0_view: wgpu::TextureView,
    last_clouds: Option<u64>,
    prefilter_bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,

    /// Last sun the cubemap was built for, so we only regenerate on change.
    last_sun: Option<([f32; 3], [f32; 3], u32)>,
    /// The night the cube was last built with.
    last_night: Option<NightSky>,

    /// What the prefilter lays over the sky until the clouds' capture is set, and the capture once it is: the second
    /// cube's prefilter is built from whichever is current.
    no_clouds: wgpu::TextureView,
    clouds_view: Option<wgpu::TextureView>,
    /// The crossfade (`fs_blend`: the bound cube takes a share of the second each frame).
    blend_bgl: wgpu::BindGroupLayout,
    blend_pipeline: wgpu::RenderPipeline,
    /// A running clock's second cube (`MovingSky`), once one has run.
    moving: Option<MovingSky>,
}

/// Environment map format — HDR, since the sun disk is far brighter than 1.0.
const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

impl IblPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &crate::shaders::Shaders,
        atmosphere: &super::atmosphere::AtmospherePass,
    ) -> Self {
        let cubemap = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment Cubemap"),
            size: wgpu::Extent3d {
                width: CUBE_SIZE,
                height: CUBE_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: MIP_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let cube_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Env Cube View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        // Prefilter source: mip 0 only. Reading mip 0 while writing mips 1+
        // needs distinct views or it is a usage conflict. The bind group keeps
        // this view alive, so it is not stored on the struct.
        let mip0_view = cubemap.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Env Cube Mip0"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let mut face_mip_views = Vec::with_capacity((MIP_COUNT * 6) as usize);
        for mip in 0..MIP_COUNT {
            for face in 0..6 {
                face_mip_views.push(cubemap.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Env Face Target"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    base_array_layer: face,
                    array_layer_count: Some(1),
                    ..Default::default()
                }));
            }
        }

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Env Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            // Trilinear: the shading pass picks a mip from surface roughness.
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("IBL Gen Params"),
            size: std::mem::size_of::<GenParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sky pass: uniform plus the Phase 24C atmosphere LUTs.
        let lut_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Sky BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                lut_entry(3),
                lut_entry(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Prefilter pass: uniform + source cube + sampler.
        let prefilter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Prefilter BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The clouds' capture (clouds.wgsl `cloud_env`), laid over every sample the prefilter takes.
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        // Until the clouds' capture is set: a 1x1 cube the prefilter never reads (`clouds.x` 0).
        let no_clouds = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("IBL No Cloud Capture"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: ENV_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            });
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let sky_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL Sky BG"),
            layout: &sky_bgl,
            entries: &[
                entry(0, params_buffer.as_entire_binding()),
                entry(3, wgpu::BindingResource::TextureView(atmosphere.transmittance_view())),
                entry(4, wgpu::BindingResource::TextureView(atmosphere.multiscatter_view())),
                entry(5, wgpu::BindingResource::Sampler(atmosphere.sampler())),
            ],
        });

        let prefilter_bind_group =
            Self::prefilter_bind_group(device, &prefilter_bgl, &params_buffer, &mip0_view, &sampler, &no_clouds);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IBL Gen Shader"),
            // MORROWIND-C: composition is declared in `ibl_gen.wgsl` and
            // resolved by `somnium_shader`; this site no longer knows the order.
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("ibl_gen.wgsl").into()),
        });

        let make_pipeline = |label: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                multiview_mask: None,
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ENV_FORMAT,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                cache: None,
            })
        };

        let sky_pipeline = make_pipeline("IBL Sky Pipeline", &sky_bgl, "fs_sky");
        let prefilter_pipeline =
            make_pipeline("IBL Prefilter Pipeline", &prefilter_bgl, "fs_prefilter");

        // The crossfade: one face and mip of the second cube, texel for texel, blended over the same face and mip of
        // the bound one by a constant share (`blend_front`).
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IBL Blend BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 7,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let share = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        };
        let blend_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("IBL Blend Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IBL Blend Pipeline"),
                bind_group_layouts: &[Some(&blend_bgl)],
                immediate_size: 0,
            })),
            multiview_mask: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_blend"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: ENV_FORMAT,
                    blend: Some(wgpu::BlendState { color: share, alpha: share }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        });

        Self {
            no_clouds,
            clouds_view: None,
            blend_bgl,
            blend_pipeline,
            moving: None,
            cubemap,
            cube_view,
            sampler,
            face_mip_views,
            sky_pipeline,
            prefilter_pipeline,
            sky_bind_group,
            prefilter_bgl,
            mip0_view,
            last_clouds: None,
            prefilter_bind_group,
            params_buffer,
            last_sun: None,
            night: NightSky::default(),
            last_night: None,
        }
    }

    /// Rebuild the cubemap if the sun changed (or it has never been built).
    ///
    /// Returns `true` when work was done.
    fn prefilter_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        params: &wgpu::Buffer,
        mip0: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        clouds: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let entry = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("IBL Prefilter BG"),
            layout,
            entries: &[
                entry(0, params.as_entire_binding()),
                entry(1, wgpu::BindingResource::TextureView(mip0)),
                entry(2, wgpu::BindingResource::Sampler(sampler)),
                entry(6, wgpu::BindingResource::TextureView(clouds)),
            ],
        })
    }

    /// The clouds' capture of the sky as a cube (`CloudPass::sky_light_view`), laid over every prefiltered mip
    /// whenever `generate_if_needed` is given a capture. Mip 0 stays the bare sky: the background and mirrors draw
    /// it under the full-resolution clouds, which the 64^2 capture would show through as blocks.
    pub fn set_cloud_capture(&mut self, device: &wgpu::Device, clouds: &wgpu::TextureView) {
        self.prefilter_bind_group = Self::prefilter_bind_group(
            device,
            &self.prefilter_bgl,
            &self.params_buffer,
            &self.mip0_view,
            &self.sampler,
            clouds,
        );
        self.clouds_view = Some(clouds.clone());
        if let Some(m) = self.moving.as_mut() {
            m.prefilter_bind_group = Self::prefilter_bind_group(
                device,
                &self.prefilter_bgl,
                &self.params_buffer,
                &m.mip0_view,
                &self.sampler,
                clouds,
            );
            m.phase = MovingPhase::Idle;
        }
        self.last_sun = None;
    }

    /// `clouds`: which capture of the clouds to lay over the sky (`CloudPass::sky_light_clouds`); each new one
    /// regenerates the cube, and with one the overcast dome stands down.
    ///
    /// `moving`: a day clock is running, so the sun changes a little every frame. Then the cube is not rebuilt whole
    /// on each change (36 passes a frame): a second cube is built two faces a frame (twelve for a fast clock), and the
    /// bound cube crossfades to it, then the next (`step_moving`). A sun further than 5 degrees from the one the cube
    /// holds is rebuilt at once, as always.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_if_needed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sun_direction: glam::Vec3,
        sun_color: glam::Vec3,
        overcast: f32,
        clouds: Option<u64>,
        moving: bool,
    ) -> bool {
        let overcast = if clouds.is_some() { 0.0 } else { overcast.clamp(0.0, 1.0) };
        let key = (sun_direction.to_array(), sun_color.to_array(), overcast.to_bits());
        let toward = sun_direction.normalize_or(glam::Vec3::Y);
        if moving
            && self
                .last_sun
                .is_some_and(|(sun, _, _)| glam::Vec3::from(sun).normalize_or(glam::Vec3::Y).dot(toward) >= MOVING_JUMP_COS)
        {
            return self.step_moving(device, queue, (key, self.night, clouds));
        }
        // A crossfade cut short leaves the bound cube between two skies: it is rebuilt whole.
        let cut = self
            .moving
            .as_mut()
            .is_some_and(|m| std::mem::replace(&mut m.phase, MovingPhase::Idle) != MovingPhase::Idle);
        if !cut && self.last_sun == Some(key) && self.last_night == Some(self.night) && self.last_clouds == clouds {
            return false;
        }
        self.last_sun = Some(key);
        self.last_night = Some(self.night);
        self.last_clouds = clouds;
        let clouds_on = f32::from(u8::from(clouds.is_some()));

        // One submission per face/mip. `queue.write_buffer` only lands once
        // before a submission executes, so a single command buffer would see
        // the last parameters for every pass — 36 tiny submissions is the
        // simple correct alternative, and this runs once, not per frame.
        let dir = sun_direction.normalize_or(glam::Vec3::Y);

        // Mip 0: capture the sky.
        for face in 0..6u32 {
            self.write_params(queue, face, 0.0, dir, sun_color, overcast, clouds_on);
            self.run_face(device, queue, &self.face_mip_views[face as usize], true, &self.prefilter_bind_group);
        }

        // Mips 1..N: GGX prefilter of mip 0, one roughness per level.
        for mip in 1..MIP_COUNT {
            let roughness = mip as f32 / (MIP_COUNT - 1) as f32;
            for face in 0..6u32 {
                self.write_params(queue, face, roughness, dir, sun_color, overcast, clouds_on);
                let view = &self.face_mip_views[(mip * 6 + face) as usize];
                self.run_face(device, queue, view, false, &self.prefilter_bind_group);
            }
        }
        true
    }

    /// One frame of a running clock's rebuild (`generate_if_needed`'s `moving`). Idle: a sky other than the one the
    /// bound cube holds starts the second cube's build. Building: the cycle's next faces and mips. Blending: the bound
    /// cube takes `1 / (frames left)` of the second, so at the cycle's last frame it is the new sky exactly and has
    /// moved to it along a straight line, never by a step.
    fn step_moving(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, want: SkyKey) -> bool {
        let mut m = match self.moving.take() {
            Some(m) => m,
            None => self.make_moving(device),
        };
        let worked = match m.phase {
            MovingPhase::Idle => {
                let held = (self.last_sun.unwrap_or_default(), self.last_night.unwrap_or_default(), self.last_clouds);
                let changed = self.last_night.is_none() || held != want;
                if changed {
                    // How far the sun has run ahead of the bound cube's picks the cycle: the quick one for a fast clock.
                    let ahead = glam::Vec3::from(held.0.0)
                        .normalize_or(glam::Vec3::Y)
                        .dot(glam::Vec3::from(want.0.0).normalize_or(glam::Vec3::Y));
                    let (per_frame, blend) = if ahead < MOVING_QUICK_COS {
                        (MOVING_QUICK_JOBS_PER_FRAME, MOVING_QUICK_BLEND_FRAMES)
                    } else {
                        (MOVING_JOBS_PER_FRAME, MOVING_BLEND_FRAMES)
                    };
                    m.key = Some(want);
                    m.phase = MovingPhase::Building { job: 0, per_frame, blend };
                    self.build_jobs(device, queue, &mut m);
                }
                changed
            }
            MovingPhase::Building { .. } => {
                self.build_jobs(device, queue, &mut m);
                true
            }
            MovingPhase::Blending { done, of } => {
                self.blend_front(device, queue, &m, 1.0 / f64::from(of - done));
                m.phase = if done + 1 < of {
                    MovingPhase::Blending { done: done + 1, of }
                } else {
                    if let Some((sun, night, clouds)) = m.key {
                        (self.last_sun, self.last_night, self.last_clouds) = (Some(sun), Some(night), clouds);
                    }
                    MovingPhase::Idle
                };
                true
            }
        };
        self.moving = Some(m);
        worked
    }

    /// The cycle's next faces and mips of the second cube, for the sky it is being built for; the last of them starts
    /// the crossfade. Mip 0's faces come first: the prefilter reads them.
    fn build_jobs(&self, device: &wgpu::Device, queue: &wgpu::Queue, m: &mut MovingSky) {
        let (MovingPhase::Building { mut job, per_frame, blend }, Some(((sun, color, overcast), _, clouds))) =
            (m.phase, m.key)
        else {
            return;
        };
        let dir = glam::Vec3::from(sun).normalize_or(glam::Vec3::Y);
        let clouds_on = f32::from(u8::from(clouds.is_some()));
        for _ in 0..per_frame {
            let (mip, face) = (job / 6, job % 6);
            let roughness = mip as f32 / (MIP_COUNT - 1) as f32;
            self.write_params(queue, face, roughness, dir, color.into(), f32::from_bits(overcast), clouds_on);
            self.run_face(device, queue, &m.face_mip_views[job as usize], mip == 0, &m.prefilter_bind_group);
            job += 1;
            if job == MIP_COUNT * 6 {
                m.phase = MovingPhase::Blending { done: 0, of: blend };
                return;
            }
        }
        m.phase = MovingPhase::Building { job, per_frame, blend };
    }

    /// Every face and mip of the bound cube moves `share` of the way to the second cube's (`fs_blend`).
    fn blend_front(&self, device: &wgpu::Device, queue: &wgpu::Queue, m: &MovingSky, share: f64) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("IBL Blend Encoder"),
        });
        for (view, source) in self.face_mip_views.iter().zip(&m.blend_bind_groups) {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("IBL Blend Pass"),
                multiview_mask: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.blend_pipeline);
            rpass.set_bind_group(0, source, &[]);
            rpass.set_blend_constant(wgpu::Color { r: share, g: share, b: share, a: share });
            rpass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// The second cube, laid out as the bound one, its prefilter reading its own mip 0 under the current clouds.
    fn make_moving(&self, device: &wgpu::Device) -> MovingSky {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Environment Cubemap (moving sun)"),
            size: wgpu::Extent3d { width: CUBE_SIZE, height: CUBE_SIZE, depth_or_array_layers: 6 },
            mip_level_count: MIP_COUNT,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mip0_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Env Cube Mip0 (moving sun)"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let face_mip_views: Vec<_> = (0..MIP_COUNT * 6)
            .map(|job| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Env Face Target (moving sun)"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_mip_level: job / 6,
                    mip_level_count: Some(1),
                    base_array_layer: job % 6,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let blend_bind_groups = face_mip_views
            .iter()
            .map(|view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("IBL Blend BG"),
                    layout: &self.blend_bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(view) }],
                })
            })
            .collect();
        let prefilter_bind_group = Self::prefilter_bind_group(
            device,
            &self.prefilter_bgl,
            &self.params_buffer,
            &mip0_view,
            &self.sampler,
            self.clouds_view.as_ref().unwrap_or(&self.no_clouds),
        );
        MovingSky {
            _texture: texture,
            face_mip_views,
            prefilter_bind_group,
            mip0_view,
            blend_bind_groups,
            key: None,
            phase: MovingPhase::Idle,
        }
    }

    fn write_params(
        &self,
        queue: &wgpu::Queue,
        face: u32,
        roughness: f32,
        sun_direction: glam::Vec3,
        sun_color: glam::Vec3,
        overcast: f32,
        clouds: f32,
    ) {
        let p = GenParams {
            face,
            roughness,
            _src_size: CUBE_SIZE as f32,
            overcast,
            sun_direction: [sun_direction.x, sun_direction.y, sun_direction.z, 0.0],
            // Phase 24A: `.w` carries the sky-dome luminance scale. The sky
            // gradient is authored as a unit-ish colour, but with the sun now
            // in lux it has to be a luminance too, or ambient is five orders of
            // magnitude too dark and every shadow reads as pure black.
            //
            // A clear day sky is ~8 000 cd/m² under ~100 000 lux of sun, hence
            // 0.08 cd/m² per lux. Scaling the dome by the sun's own output is
            // also what finally lets night happen: lower the sun and the sky
            // darkens with it, instead of holding daylight ambient forever.
            //
            // Interim — Phase 24C computes sky radiance from real atmospheric
            // scattering and this scale factor disappears.
            sun_color: [
                sun_color.x,
                sun_color.y,
                sun_color.z,
                sun_luminance(sun_color),
            ],
            night: [self.night.glow.max(0.0), self.night.aurora, self.night.halo_road, self.night.thread],
            night_tint: [self.night.tint[0], self.night.tint[1], self.night.tint[2], 0.0],
            clouds: [clouds, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&p));
    }

    /// Draw one face and mip, `view`: the sky (mip 0), or the prefilter of the cube `prefilter` reads.
    fn run_face(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        is_sky: bool,
        prefilter: &wgpu::BindGroup,
    ) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("IBL Gen Encoder"),
        });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("IBL Gen Pass"),
                multiview_mask: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if is_sky {
                rpass.set_pipeline(&self.sky_pipeline);
                rpass.set_bind_group(0, &self.sky_bind_group, &[]);
            } else {
                rpass.set_pipeline(&self.prefilter_pipeline);
                rpass.set_bind_group(0, prefilter, &[]);
            }
            rpass.draw(0..3, 0..1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Highest mip index, for mapping roughness to a mip in the shading pass.
    pub const fn max_mip() -> f32 {
        (MIP_COUNT - 1) as f32
    }
}
