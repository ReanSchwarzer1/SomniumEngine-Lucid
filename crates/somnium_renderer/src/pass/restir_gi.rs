//! ReSTIR GI — ray-traced indirect diffuse (Phase 24L).
//!
//! Two compute dispatches per frame over `restir_gi.wgsl`: the first draws one
//! bounce per pixel and merges it with that pixel's history, the second borrows
//! from spatial neighbours, traces the survivor's visibility ray and writes the
//! bounce. A third (`gi_accumulate.wgsl`) averages the bounce and the sky's share
//! over neighbours and time, into the two layers the shading pass consumes.
//!
//! Like 24J and 24K it does nothing at all when the device did not grant ray
//! query — the constant ambient term remains the fallback, and an output alpha
//! of zero is the signal that says so. wgpu zero-fills a new texture, so the
//! unsupported path needs no feature check of its own.
//!
//! # Bind groups
//!
//! Group 0 is the **shared global pool** — the same bind group the shading pass
//! uses. That is the point of Phase 24L's `global_pool.wgsl` extraction: a ray
//! hit resolves to geometry and material through the same `instances` array a
//! visibility-buffer hit does, so the traced and rasterised views of the scene
//! cannot drift apart. Group 1 is this pass's own.

/// Indirect radiance output. Alpha is the "a traced result exists" flag, on the
/// same convention 24K established for its visibility target.
const GI_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// GI grid step in pixels: traced at half resolution and upsampled, depth-aware,
/// by the shading pass (`gi_upsample`). Indirect diffuse is low frequency; this
/// is a quarter of the rays for an image that cannot be told apart at 2K.
/// `SOMNIUM_GI_SCALE=1` restores full resolution for an A/B.
fn gi_scale() -> u32 {
    std::env::var("SOMNIUM_GI_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|s| (1..=4).contains(s))
        .unwrap_or(2)
}

/// Matches `GiReservoir` in `restir_gi.wgsl`: 12 floats, 48 bytes.
const RESERVOIR_BYTES: u64 = 48;
const SURFACE_BYTES: u64 = 32;

/// Most frames the sky's share averages (`gi_accumulate.wgsl`): it depends on the scene's shape alone, so a long
/// memory costs nothing when the lights change. About two seconds at 30 frames a second.
const SKY_FRAMES: f32 = 64.0;
/// Most frames the bounce averages while the lights hold still, and while they change (a moving sun, a flickering
/// lamp): the reservoirs' own shorter confidence, so the bounce follows the light within a few frames.
const BOUNCE_FRAMES: f32 = 32.0;

/// `AccumulateParams` in `gi_accumulate.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AccumulateParams {
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    camera_step: [f32; 3],
    history_valid: f32,
    inv_resolution: [f32; 2],
    scale: f32,
    frame: u32,
    sky_frames: f32,
    bounce_frames: f32,
    _pad: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GiParams {
    /// Camera-relative: inverse(jittered projection x the view's rotation alone); add `camera_pos` for world
    /// space. The absolute inverse snaps to 2^-7 in ndc 13 km from the origin (Sunscar, 5 Nov 2026).
    inv_view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    frame: u32,
    inv_resolution: [f32; 2],
    history_valid: f32,
    intensity: f32,
    max_distance: f32,
    /// Pads to 112, which is what the WGSL mirror rounds to. See the shader's
    /// note on why the three scalars there are not one `vec3`.
    /// GI grid step in pixels (`gi_scale`).
    scale: f32,
    /// Most confidence a reservoir may carry into this frame (`GI_M_CAP` when
    /// the lights held still; less while they flicker or move).
    history_m_cap: f32,
    _pad: f32,
}

#[cfg(test)]
mod tests {
    use super::GiParams;

    /// The mismatch that stopped this pass dispatching the first time it ran.
    ///
    /// wgpu rejected the dispatch outright — "bound with size 112 where the
    /// shader expects 128" — because a trailing `vec3<f32>` aligns to 16 in
    /// WGSL and to 4 in Rust. `tests/shaders_validate.rs` proves the WGSL side
    /// of every struct it checks; this pins the Rust side of this one.
    #[test]
    fn the_params_struct_is_the_112_byte_uniform_layout() {
        assert_eq!(std::mem::size_of::<GiParams>(), 112);
        assert_eq!(std::mem::size_of::<GiParams>() % 16, 0);
        // gi_accumulate.wgsl's: two matrices, then a vec3 and scalars that pack to 176 in both languages.
        assert_eq!(std::mem::size_of::<super::AccumulateParams>(), 176);
    }

    #[test]
    fn flicker_and_carried_lights_shorten_history_instead_of_discarding_it() {
        use super::classify_light_change;
        let lamp = |pos: [f32; 3], lum: f32| crate::cluster::GpuLocalLight {
            position_ws: pos,
            range: 6.0,
            color: [lum, lum * 0.8, lum * 0.5],
            light_type: 0,
            direction_ws: [0.0, -1.0, 0.0],
            spot_cos_outer: 0.7,
            spot_cos_inner: 0.9,
            radius: 0.05,
            _pad: [0.0; 2],
        };
        let base = [lamp([0.0, 3.0, 0.0], 100.0)];
        // Buzz: 19% dimmer. Changed, gentle.
        assert_eq!(classify_light_change(&[lamp([0.0, 3.0, 0.0], 81.0)], &base), (false, true, false));
        // A carried lighter, one running frame on.
        assert_eq!(classify_light_change(&[lamp([0.12, 3.0, 0.0], 100.0)], &base), (false, true, false));
        // A stutter toward dark.
        assert_eq!(classify_light_change(&[lamp([0.0, 3.0, 0.0], 4.0)], &base), (false, true, true));
        // A cut, and a light switched out of the list.
        assert!(classify_light_change(&[lamp([5.0, 3.0, 0.0], 100.0)], &base).0);
        assert!(classify_light_change(&[], &base).0);
        assert_eq!(classify_light_change(&base, &base), (false, false, false));
    }
}

/// Confidence a reservoir may carry while the lights hold still (the shader's
/// `GI_M_CAP`), while they drift (buzzing, carried), and while one stutters
/// toward dark.
const HISTORY_M_CAP: f32 = 24.0;
const HISTORY_M_CAP_DRIFT: f32 = 12.0;
const HISTORY_M_CAP_STUTTER: f32 = 3.0;

/// `(structural, changed, strong)` between this frame's lights and the last
/// recorded set. Structural: a light added, removed or retyped, or one that
/// moved more than a metre (a cut, not motion). Changed: the old 2% / 5 cm
/// test. Strong: output more than halved or doubled, or moved over 25 cm.
fn classify_light_change(
    now: &[crate::cluster::GpuLocalLight],
    before: &[crate::cluster::GpuLocalLight],
) -> (bool, bool, bool) {
    if now.len() != before.len() {
        return (true, true, true);
    }
    let (mut structural, mut changed, mut strong) = (false, false, false);
    for (a, b) in now.iter().zip(before) {
        let color_a = glam::Vec3::from_array(a.color);
        let color_b = glam::Vec3::from_array(b.color);
        let scale = color_a.abs().max(color_b.abs()).max_element().max(0.001);
        let moved = glam::Vec3::from_array(a.position_ws).distance(glam::Vec3::from_array(b.position_ws));
        let (lum_a, lum_b) = (color_a.max_element(), color_b.max_element());
        structural |= a.light_type != b.light_type || moved > 1.0;
        strong |= moved > 0.25 || lum_a > 2.0 * lum_b + 1e-3 || lum_b > 2.0 * lum_a + 1e-3;
        changed |= a.light_type != b.light_type
            || (color_a - color_b).abs().max_element() > 0.02 * scale
            || moved > 0.05
            || glam::Vec3::from_array(a.direction_ws).distance(glam::Vec3::from_array(b.direction_ws)) > 0.01
            || (a.range - b.range).abs() > 0.01
            || (a.radius - b.radius).abs() > 0.001
            || a._pad != b._pad
            || a.spot_cos_inner != b.spot_cos_inner
            || a.spot_cos_outer != b.spot_cos_outer;
    }
    (structural, changed, strong)
}

pub struct RestirGiPass {
    /// The environment cube and sampler escaped rays read (`set_environment`).
    environment: Option<(wgpu::TextureView, wgpu::Sampler)>,
    initial: Option<wgpu::ComputePipeline>,
    spatial: Option<wgpu::ComputePipeline>,
    layout: Option<wgpu::BindGroupLayout>,
    params: Option<wgpu::Buffer>,
    sampler: wgpu::Sampler,
    /// `gi_a` carries the previous frame's finished reservoirs; `gi_b` is the
    /// handoff between the two dispatches. Fixed roles, not a ping-pong — see
    /// the shader's comment on why reading and writing one buffer in the
    /// spatial pass would both race and double-count.
    gi_a: Option<wgpu::Buffer>,
    gi_b: Option<wgpu::Buffer>,
    /// Current-frame primary geometry, shared with spatial neighbours.
    surfaces: Option<wgpu::Buffer>,
    cache_surfaces: bool,
    bind: Option<wgpu::BindGroup>,
    grain_view: wgpu::TextureView,
    grain_enabled: bool,
    out_view: Option<wgpu::TextureView>,
    out_tex: Option<wgpu::Texture>,
    /// This frame's sums for the sky's share, per GI texel (restir_gi.wgsl `sky_out`).
    sky_view: Option<wgpu::TextureView>,
    /// What the shading pass reads (`gi_accumulate.wgsl`): layer 0 the bounce and the surface's distance, layer 1 the
    /// sky's sums and the frames behind them. `history` is its copy from the frame before.
    result_tex: Option<wgpu::Texture>,
    result_view: Option<wgpu::TextureView>,
    result_layers: Option<[wgpu::TextureView; 2]>,
    history_tex: Option<wgpu::Texture>,
    history_view: Option<wgpu::TextureView>,
    accumulate: Option<(wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer)>,
    accumulate_bind: Option<wgpu::BindGroup>,
    /// Whether `history` holds last frame's result for this resolution, and the camera it was made from
    /// (projection x view rotation, unjittered, and the position).
    accumulated: bool,
    prev_view_proj: glam::Mat4,
    prev_camera: glam::Vec3,
    /// The clouds' shadow field and its params, for the sun a bounce receives (`set_cloud_shadow`).
    cloud_shadow: Option<(wgpu::TextureView, wgpu::Buffer)>,
    /// Set once the pass has written the target, so a disabled pass knows it
    /// still has stale contents to clear. Same trap 24K hit: switching the
    /// feature off left every surface pinned to the last traced result.
    dirty: bool,
    frame: u32,
    history_valid: bool,
    /// Lighting represented by the temporal reservoirs. A materially changed
    /// sun must discard history so daytime bounce cannot bleed into night.
    last_light: Option<([f32; 3], [f32; 3])>,
    last_local_lights: Vec<crate::cluster::GpuLocalLight>,
    /// See `GiParams::history_m_cap`.
    history_m_cap: f32,
    supported: bool,
    pub enabled: bool,
    /// Scales the indirect contribution. Exposed so the A/B can vary the
    /// strength without disabling the pass.
    pub intensity: f32,
    /// Metres beyond which a bounce ray is not traced.
    pub max_distance: f32,
}

impl RestirGiPass {
    pub fn new(
        device: &wgpu::Device,
        shaders: &crate::shaders::Shaders,
        global_layout: &wgpu::BindGroupLayout,
        supported: bool,
        grain_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ReSTIR GI sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let mut pass = Self {
            initial: None,
            spatial: None,
            layout: None,
            params: None,
            sampler,
            gi_a: None,
            gi_b: None,
            surfaces: None,
            cache_surfaces: std::env::var("SOMNIUM_GI_SURFACE_CACHE").as_deref() == Ok("1"),
            bind: None,
            environment: None,
            grain_view: grain_view.clone(),
            grain_enabled: supported
                && crate::pass::grain::enabled_by_default("SOMNIUM_DREAMS_GRAIN"),
            out_view: None,
            out_tex: None,
            sky_view: None,
            result_tex: None,
            result_view: None,
            result_layers: None,
            history_tex: None,
            history_view: None,
            accumulate: None,
            accumulate_bind: None,
            accumulated: false,
            prev_view_proj: glam::Mat4::IDENTITY,
            prev_camera: glam::Vec3::ZERO,
            cloud_shadow: None,
            dirty: false,
            frame: 0,
            history_valid: false,
            last_light: None,
            last_local_lights: Vec::new(),
            history_m_cap: HISTORY_M_CAP,
            supported,
            // On wherever the hardware allows it. `SOMNIUM_RESTIR_GI=0` is the
            // A/B against the environment map's constant diffuse, and a device
            // without ray query never reaches this branch at all.
            enabled: supported && std::env::var("SOMNIUM_RESTIR_GI").as_deref() != Ok("0"),
            intensity: 1.0,
            max_distance: 200.0,
        };

        if !supported {
            pass.allocate_targets(device, width, height);
            return pass;
        }

        // GI first: `enable wgpu_ray_query;` is a directive, and directives must
        // precede every declaration in the module. Declarations themselves are
        // order-independent, which is what lets the pool it depends on be
        // concatenated after it. `tests/shaders_validate.rs` pins this exact
        // concatenation.
        // MORROWIND-C: composition is declared in `restir_gi.wgsl` and
        // resolved by `somnium_shader`; this site no longer knows the order.
        let source = shaders.source_or_panic("restir_gi.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("restir_gi.wgsl"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ReSTIR GI BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // The visibility buffer, for the primary surface's geometric
                // normal. A compute shader has no quad to take derivatives
                // across, and a normal reconstructed from neighbouring depths is
                // wrong exactly at the silhouettes indirect light lives on.
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: GI_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(5),
                storage(6),
                storage(9),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                // The environment cube an escaped ray sees, and its sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // This frame's sums for the sky's share (`sky_out`).
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: GI_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // The clouds' shadow field and its params: the sun a bounce receives comes through the deck.
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ReSTIR GI PL"),
            bind_group_layouts: &[Some(global_layout), Some(&layout)],
            immediate_size: 0,
        });

        // Alpha-test leaf/grass cards as traversal candidates instead of
        // restarting the ray from the root at every cutout hole (up to four
        // hops). Same-session native A/B/A 2026-10-04: Gardens GI initial +
        // temporal 15.5 / 21.8 / 15.6 ms; Facility I 3.2 / 2.8 / 3.2 ms (off /
        // on / off order there). `SOMNIUM_GI_CANDIDATE_ALPHA=0` restores hops.
        let candidate_alpha = std::env::var("SOMNIUM_GI_CANDIDATE_ALPHA").as_deref() != Ok("0");
        // Native Gardens A/B/A retained identical terrain arithmetic while
        // avoiding local-array spills. Keep an explicit reference opt-out.
        let terrain_storage =
            std::env::var("SOMNIUM_GI_TERRAIN_STORAGE_READS").as_deref() != Ok("0");
        // Terrain albedo for a GI hit is looked up where it is used, not at the
        // hit: the same lookup and arithmetic, and the one consumer. Gardens
        // A/B/A/B 2026-10-04: GI initial + temporal 31.5 / 31.4 -> 21.9 ms.
        // `SOMNIUM_GI_DEFER_TERRAIN_ALBEDO=0` restores the eager lookup.
        let deferred_terrain_albedo =
            std::env::var("SOMNIUM_GI_DEFER_TERRAIN_ALBEDO").as_deref() != Ok("0");
        let constants = [
            ("rt_candidate_alpha", f64::from(u32::from(candidate_alpha))),
            (
                "rt_terrain_direct_storage",
                f64::from(u32::from(terrain_storage)),
            ),
            (
                "rt_gi_deferred_terrain_albedo",
                f64::from(u32::from(deferred_terrain_albedo)),
            ),
            (
                "gi_cache_surfaces",
                f64::from(u32::from(pass.cache_surfaces)),
            ),
        ];
        let make = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..Default::default()
                },
                cache: None,
            })
        };
        pass.initial = Some(make("initial_and_temporal", "ReSTIR GI initial+temporal"));
        pass.accumulate = Some(Self::accumulate_pipeline(device, shaders));
        pass.spatial = Some(make("spatial_and_shade", "ReSTIR GI spatial+shade"));
        pass.layout = Some(layout);
        pass.params = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR GI params"),
            size: std::mem::size_of::<GiParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        pass.resize(device, width, height);
        pass
    }

    /// Select the shared DREAMS sampling sequence when this adapter supports the pass.
    pub fn set_grain_enabled(&mut self, enabled: bool) {
        let enabled = self.supported && enabled;
        if self.grain_enabled != enabled {
            self.history_valid = false;
        }
        self.grain_enabled = enabled;
    }

    pub fn supported(&self) -> bool {
        self.supported
    }

    /// Reservoir radiance includes practical lights, so history has to follow
    /// them. Only a *structural* change discards it: a light added, removed or
    /// retyped, or one that jumped. Anything gentler shortens history instead.
    ///
    /// It used to discard on any 2% change. A flickering lamp's buzz moves its
    /// output ~19% at 11 Hz and a carried lighter moves 5+ cm a frame, so in
    /// Town (17 flickering lamps) and wherever the player carried a flame, GI
    /// never reused a single frame: one bounce sample per texel, every frame,
    /// which is the gold speckle on the hands and streets while running.
    pub fn update_local_lights(&mut self, lights: &[crate::cluster::GpuLocalLight]) {
        if !self.active() { return; }
        let (structural, changed, strong) = classify_light_change(lights, &self.last_local_lights);
        if structural {
            self.history_valid = false;
        }
        self.history_m_cap = if !changed {
            HISTORY_M_CAP
        } else if strong {
            HISTORY_M_CAP_STUTTER
        } else {
            HISTORY_M_CAP_DRIFT
        };
        if changed {
            self.last_local_lights.clear();
            self.last_local_lights.extend_from_slice(lights);
        }
    }

    pub fn active(&self) -> bool {
        self.supported && self.enabled
    }

    /// What the traced GI gives the shading pass, as a two-layer array: the bounce and the surface's distance, then
    /// the sky's sums and the frames behind them (`gi_accumulate.wgsl`). All zero where nothing was traced, and on a
    /// device without ray query: the shading pass then lights by its fallback.
    pub fn radiance_view(&self) -> Option<&wgpu::TextureView> {
        self.result_view.as_ref()
    }

    /// The bounce alone (layer 0: rgb, with the surface's distance in alpha), for the passes that read it as a plain
    /// texture.
    pub fn bounce_view(&self) -> Option<&wgpu::TextureView> {
        self.result_layers.as_ref().map(|layers| &layers[0])
    }

    /// The clouds' shadow on the ground and its params (`CloudPass::shadow_view`, `shadow_params`): the sun a bounce
    /// ray's hit receives comes through the deck. Until it is set the pass binds nothing.
    pub fn set_cloud_shadow(&mut self, field: &wgpu::TextureView, params: &wgpu::Buffer) {
        self.cloud_shadow = Some((field.clone(), params.clone()));
        self.bind = None;
    }

    /// The accumulate pass (`gi_accumulate.wgsl`): this frame's bounce and sky sums and last frame's result in, the
    /// result out, on the GI grid.
    fn accumulate_pipeline(
        device: &wgpu::Device,
        shaders: &crate::shaders::Shaders,
    ) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout, wgpu::Buffer) {
        let loaded = |binding: u32, view_dimension: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GI accumulate BGL"),
            entries: &[
                loaded(0, wgpu::TextureViewDimension::D2),
                loaded(1, wgpu::TextureViewDimension::D2),
                loaded(2, wgpu::TextureViewDimension::D2Array),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: GI_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GI accumulate PL"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gi_accumulate.wgsl"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("gi_accumulate.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GI accumulate"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("accumulate"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GI accumulate params"),
            size: std::mem::size_of::<AccumulateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (pipeline, layout, params)
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.allocate_targets(device, width, height);
    }

    fn allocate_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let s = gi_scale();
        let w = width.max(1).div_ceil(s);
        let h = height.max(1).div_ceil(s);

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ReSTIR GI radiance"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GI_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.out_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.out_tex = Some(tex);
        let sky = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ReSTIR GI sky sums"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: GI_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.sky_view = Some(sky.create_view(&wgpu::TextureViewDescriptor::default()));
        // Two layers each: the result the shading pass reads, and its copy from the frame before.
        let layered = |label: &str, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 2 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: GI_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let result = layered(
            "ReSTIR GI result",
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let history =
            layered("ReSTIR GI history", wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let layer = |index: u32| {
            result.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: index,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        self.result_layers = Some([layer(0), layer(1)]);
        self.result_view = Some(array_view(&result));
        self.history_view = Some(array_view(&history));
        self.result_tex = Some(result);
        self.history_tex = Some(history);
        self.accumulate_bind = None;
        self.accumulated = false;
        self.dirty = false;

        if !self.supported {
            return;
        }

        let size = u64::from(w) * u64::from(h) * RESERVOIR_BYTES;
        let make = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        self.gi_a = Some(make("ReSTIR GI reservoirs A"));
        self.gi_b = Some(make("ReSTIR GI reservoirs B"));
        self.surfaces = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR GI current surfaces"),
            size: if self.cache_surfaces {
                u64::from(w) * u64::from(h) * SURFACE_BYTES
            } else {
                SURFACE_BYTES
            },
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));
        self.bind = None;
        // Reservoirs from a different resolution index differently, so history
        // has to be discarded rather than reinterpreted.
        self.history_valid = false;
    }

    fn ensure_bind(
        &mut self,
        device: &wgpu::Device,
        tlas: &wgpu::Tlas,
        depth: &wgpu::TextureView,
        vis: &wgpu::TextureView,
    ) {
        if self.bind.is_some() {
            return;
        }
        let (
            Some(layout),
            Some(params),
            Some(out),
            Some(a),
            Some(b),
            Some(surfaces),
            Some((env, env_sampler)),
            Some(sky),
            Some((cloud_field, cloud_params)),
        ) = (
            self.layout.as_ref(),
            self.params.as_ref(),
            self.out_view.as_ref(),
            self.gi_a.as_ref(),
            self.gi_b.as_ref(),
            self.surfaces.as_ref(),
            self.environment.as_ref(),
            self.sky_view.as_ref(),
            self.cloud_shadow.as_ref(),
        ) else {
            return;
        };
        self.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReSTIR GI BG"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::AccelerationStructure(tlas),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(vis),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(out),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.grain_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: surfaces.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(env),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(env_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(sky),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: wgpu::BindingResource::TextureView(cloud_field),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: cloud_params.as_entire_binding(),
                },
            ],
        }));
    }

    /// The environment cube an escaped ray sees (`IblPass::cube_view`). Until it is set the pass binds nothing.
    pub fn set_environment(&mut self, cube: &wgpu::TextureView, sampler: &wgpu::Sampler) {
        self.environment = Some((cube.clone(), sampler.clone()));
        self.bind = None;
    }

    /// Zero the radiance target when the pass is switched off.
    ///
    /// The same failure 24K documented: `record` early-returns when inactive but
    /// the texture keeps whatever it last wrote, so shading would go on adding a
    /// frozen bounce for ever. Alpha 0 is the signal an unsupported device
    /// produces, so clearing restores the ambient fallback exactly.
    pub fn clear_if_inactive(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.active() || !self.dirty {
            return;
        }
        for view in self.out_view.iter().chain(self.result_layers.iter().flatten()) {
            // A load-op clear, not `clear_texture`: that needs the optional
            // CLEAR_TEXTURE feature this device never requests.
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("ReSTIR GI clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            self.dirty = false;
            self.history_valid = false;
            self.accumulated = false;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        profiler: &mut crate::profiler::GpuProfiler,
        global_bind: &wgpu::BindGroup,
        tlas: &wgpu::Tlas,
        depth_view: &wgpu::TextureView,
        vis_view: &wgpu::TextureView,
        view_proj: glam::Mat4,
        // The same without the frame's jitter (camera-relative too), kept to find this frame's surfaces next frame.
        view_proj_unjittered: glam::Mat4,
        camera_pos: glam::Vec3,
        light_direction: glam::Vec3,
        light_color: glam::Vec3,
        width: u32,
        height: u32,
    ) {
        if !self.active() {
            return;
        }
        let light_key = (light_direction.to_array(), light_color.to_array());
        let light_changed = self.last_light.is_none_or(|(old_direction, old_color)| {
            const MAX_ANGLE_RADIANS: f32 = 0.25_f32.to_radians();
            const MAX_RELATIVE_COLOR_CHANGE: f32 = 0.02;

            let old_direction = glam::Vec3::from_array(old_direction).normalize_or_zero();
            let direction = light_direction.normalize_or_zero();
            let direction_changed = old_direction.dot(direction) < MAX_ANGLE_RADIANS.cos();

            let old_color = glam::Vec3::from_array(old_color);
            let color_scale = old_color
                .abs()
                .max(light_color.abs())
                .max_element()
                .max(1.0);
            let color_changed = (old_color - light_color).abs().max_element()
                > color_scale * MAX_RELATIVE_COLOR_CHANGE;

            direction_changed || color_changed
        });
        if light_changed {
            self.history_valid = false;
            self.last_light = Some(light_key);
        }
        self.ensure_bind(device, tlas, depth_view, vis_view);
        let (Some(initial), Some(spatial), Some(params), Some(bind)) = (
            self.initial.as_ref(),
            self.spatial.as_ref(),
            self.params.as_ref(),
            self.bind.as_ref(),
        ) else {
            return;
        };

        queue.write_buffer(
            params,
            0,
            bytemuck::bytes_of(&GiParams {
                inv_view_proj: view_proj.inverse().to_cols_array_2d(),
                camera_pos: camera_pos.to_array(),
                frame: self.frame,
                inv_resolution: [1.0 / width as f32, 1.0 / height as f32],
                history_valid: f32::from(u8::from(self.history_valid))
                    + 2.0 * f32::from(u8::from(self.grain_enabled)),
                intensity: self.intensity,
                max_distance: self.max_distance,
                scale: gi_scale() as f32,
                history_m_cap: self.history_m_cap,
                _pad: 0.0,
            }),
        );

        let s = gi_scale();
        let groups = (
            width.div_ceil(s).div_ceil(8),
            height.div_ceil(s).div_ceil(8),
        );

        // Two dispatches, not one pass with a barrier between: the spatial pass
        // reads neighbouring pixels' reservoirs, so every pixel of the first
        // dispatch has to have landed before any pixel of the second reads it.
        // Separate dispatches in the same encoder give exactly that ordering.
        profiler.begin(encoder, "GI initial + temporal");
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReSTIR GI initial+temporal"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(initial);
            cpass.set_bind_group(0, global_bind, &[]);
            cpass.set_bind_group(1, bind, &[]);
            cpass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        profiler.end(encoder);
        profiler.begin(encoder, "GI spatial + shade");
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReSTIR GI spatial+shade"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(spatial);
            cpass.set_bind_group(0, global_bind, &[]);
            cpass.set_bind_group(1, bind, &[]);
            cpass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        profiler.end(encoder);
        if let (
            Some((pipeline, layout, accumulate_params)),
            Some(out),
            Some(sky),
            Some(result),
            Some(history),
            Some(result_tex),
            Some(history_tex),
        ) = (
            self.accumulate.as_ref(),
            self.out_view.as_ref(),
            self.sky_view.as_ref(),
            self.result_view.as_ref(),
            self.history_view.as_ref(),
            self.result_tex.as_ref(),
            self.history_tex.as_ref(),
        ) {
            queue.write_buffer(
                accumulate_params,
                0,
                bytemuck::bytes_of(&AccumulateParams {
                    inv_view_proj: view_proj_unjittered.inverse().to_cols_array_2d(),
                    prev_view_proj: self.prev_view_proj.to_cols_array_2d(),
                    camera_step: (camera_pos - self.prev_camera).to_array(),
                    history_valid: f32::from(u8::from(self.accumulated)),
                    inv_resolution: [1.0 / width as f32, 1.0 / height as f32],
                    scale: s as f32,
                    frame: self.frame,
                    sky_frames: SKY_FRAMES,
                    // The bounce follows the lights: the reservoirs' confidence when they change, longer when not.
                    bounce_frames: if light_changed || self.history_m_cap < HISTORY_M_CAP {
                        self.history_m_cap.min(HISTORY_M_CAP_DRIFT)
                    } else {
                        BOUNCE_FRAMES
                    },
                    _pad: [0.0; 2],
                }),
            );
            let bind = self.accumulate_bind.get_or_insert_with(|| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("GI accumulate BG"),
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(out) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(sky) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(history) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(result) },
                        wgpu::BindGroupEntry { binding: 4, resource: accumulate_params.as_entire_binding() },
                    ],
                })
            });
            profiler.begin(encoder, "GI accumulate");
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("ReSTIR GI accumulate"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, &*bind, &[]);
                cpass.dispatch_workgroups(groups.0, groups.1, 1);
            }
            // Kept for the next frame as a copy (a few megabytes), so the view the shading pass binds never changes.
            encoder.copy_texture_to_texture(
                result_tex.as_image_copy(),
                history_tex.as_image_copy(),
                wgpu::Extent3d {
                    width: width.div_ceil(s),
                    height: height.div_ceil(s),
                    depth_or_array_layers: 2,
                },
            );
            profiler.end(encoder);
            self.accumulated = true;
            self.prev_view_proj = view_proj_unjittered;
            self.prev_camera = camera_pos;
        }

        self.dirty = true;
        self.frame = self.frame.wrapping_add(1);
        self.history_valid = true;
    }
}
