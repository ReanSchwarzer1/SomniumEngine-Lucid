//! Phase 11.5K: Post-processing pass — ACES tone mapping + vignette.
//!
//! Owns the Rgba16Float HDR render target that the shading and grid passes
//! write into. On each frame, reads the HDR texture and writes tone-mapped
//! LDR output to the swapchain surface view.
use wgpu;

/// HDR render target format shared by all passes that write to the HDR buffer.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Colour grading and lens settings (Phases 24Y and 24Z).
///
/// Grouped rather than passed positionally: `set_params` had already grown past
/// the point where a caller could get the order right by reading the call site.
#[derive(Debug, Clone, Copy)]
pub struct Grading {
    pub temperature: f32,
    pub tint: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub gain: f32,
    pub lift: f32,
    pub gamma: f32,
    pub grain: f32,
    /// Mode, strength, speed and reserved padding for the optional dream lens.
    pub dream: [f32; 4],
    /// Seconds, so grain animates instead of sitting still.
    pub time: f32,
    /// CONTROL-K. 32 uniform samples of the authored tone-response curve over
    /// `0..=1`, or `None` for the unauthored case. Sampled by the caller
    /// rather than passed as a `Curve` so this crate stays free of the
    /// reflection vocabulary — a renderer should not know what a keyframe is.
    pub response: Option<[f32; 32]>,
    /// Painterly amount and ink-line amount (`PostProcessComponent::painterly`, `ink_lines`), then padding. Zero is
    /// off and the shader skips the stage.
    pub stylise: [f32; 4],
    /// How much of the grading LUT (`PostProcessPass::set_lut`) to take, 0..1; 0 leaves it unread.
    pub lut_strength: f32,
    /// Low-light vision (`postprocess.wgsl` `low_light`): 0 off, 1 the eye's own Hunt and Purkinje shift.
    pub low_light: f32,
    /// The photographic camera (`PostProcessComponent::photographic`, sunscar-photoreal 6 Nov 2026): the grade on
    /// the exposed light before the curve, the lens's glare, fall-off and spectral fringe, and sensor noise. Off is
    /// the chain exactly as before.
    pub photographic: bool,
    /// The lens's glare share under the photographic camera (the bloom chain mixed in with energy kept).
    pub lens_glare: f32,
    /// Sensor noise under the photographic camera: 1 a real sensor's at the exposure's ISO.
    pub sensor_noise: f32,
    /// The camera's white balance, a linear-RGB matrix by rows ([`white_balance_matrix`]); the identity when off.
    pub white_balance: [[f32; 3]; 3],
    /// tan(half the vertical field of view) and the aspect (width / height), for the lens's fall-off. The renderer
    /// fills these from its projection each frame.
    pub lens: [f32; 2],
}

/// The tone-map index of the debug passthrough (`postprocess.wgsl` `tonemap`): no curve, so a debug view reaches the
/// screen as the numbers it holds.
pub const TONEMAP_PASSTHROUGH: u32 = 15;

/// The camera's white balance as a linear-RGB (Rec. 709, D65) matrix, by rows: a Bradford chromatic adaptation from
/// the white of `kelvin` to D65. The picture is rendered balanced for D65, so a camera set for a lower temperature
/// (expecting warmer light than it got) renders it cooler, and a higher one warmer, as a real camera does. `kelvin`
/// 0 (or under 1000) is the identity. The white is the CIE daylight locus from 4,000 K and the Planckian locus below
/// it (Kim et al. 2002), so 6,504 K is D65 itself.
#[must_use]
pub fn white_balance_matrix(kelvin: f32) -> [[f32; 3]; 3] {
    const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    if !(kelvin >= 1000.0) {
        return IDENTITY;
    }
    let t = f64::from(kelvin).clamp(1667.0, 25000.0);
    let (x, y) = if t >= 4000.0 {
        let x = if t <= 7000.0 {
            -4.6070e9 / t.powi(3) + 2.9678e6 / t.powi(2) + 0.09911e3 / t + 0.244063
        } else {
            -2.0064e9 / t.powi(3) + 1.9018e6 / t.powi(2) + 0.24748e3 / t + 0.237040
        };
        (x, -3.000 * x * x + 2.870 * x - 0.275)
    } else {
        let x = -0.2661239e9 / t.powi(3) - 0.2343589e6 / t.powi(2) + 0.8776956e3 / t + 0.179910;
        let y = if t <= 2222.0 {
            -1.1063814 * x.powi(3) - 1.34811020 * x * x + 2.18555832 * x - 0.20219683
        } else {
            -0.9549476 * x.powi(3) - 1.37418593 * x * x + 2.09137015 * x - 0.16748867
        };
        (x, y)
    };
    let src = glam::DVec3::new(x / y, 1.0, (1.0 - x - y) / y);
    let d65 = glam::DVec3::new(0.95047, 1.0, 1.08883);
    // Bradford, rows as written, so the column constructor takes its transpose.
    let bradford = glam::DMat3::from_cols_array(&[0.8951, -0.7502, 0.0389, 0.2664, 1.7135, -0.0685, -0.1614, 0.0367, 1.0296]);
    let scale = (bradford * d65) / (bradford * src);
    let cat = bradford.inverse() * glam::DMat3::from_diagonal(scale) * bradford;
    let rgb_to_xyz = glam::DMat3::from_cols_array(&[
        0.4124564, 0.2126729, 0.0193339, 0.3575761, 0.7151522, 0.1191920, 0.1804375, 0.0721750, 0.9503041,
    ]);
    let m = rgb_to_xyz.inverse() * cat * rgb_to_xyz;
    // White balance moves colour, not exposure: scaled so the rendered white keeps its luminance, as a camera's
    // channel gains are normalised.
    let white = m * glam::DVec3::ONE;
    let m = m * (1.0 / glam::DVec3::new(0.2126, 0.7152, 0.0722).dot(white).max(1e-6));
    let row = |i: usize| [m.col(0)[i] as f32, m.col(1)[i] as f32, m.col(2)[i] as f32];
    [row(0), row(1), row(2)]
}

impl Default for Grading {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            tint: 0.0,
            contrast: 1.0,
            saturation: 1.0,
            gain: 1.0,
            lift: 0.0,
            gamma: 1.0,
            grain: 0.0,
            dream: [0.0; 4],
            time: 0.0,
            response: None,
            stylise: [0.0; 4],
            lut_strength: 0.0,
            low_light: 0.0,
            photographic: false,
            lens_glare: 0.0,
            sensor_noise: 0.0,
            white_balance: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            lens: [0.5, 16.0 / 9.0],
        }
    }
}

/// A `.cube` 3-D LUT (Resolve, Adobe): its edge and `size`^3 RGBA texels, red fastest, as the file lists them.
/// Only the 0..1 domain; a 1-D LUT or another domain is an error rather than a wrong picture.
pub fn parse_cube(text: &str) -> Result<(u32, Vec<[f32; 4]>), String> {
    let mut size = 0_u32;
    let mut texels = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("TITLE") {
            continue;
        }
        let mut words = line.split_whitespace();
        let first = words.next().unwrap_or_default();
        match first {
            "LUT_3D_SIZE" => {
                size = words.next().and_then(|w| w.parse().ok()).filter(|s| (2..=256).contains(s))
                    .ok_or_else(|| format!("line {}: LUT_3D_SIZE must be 2..256", n + 1))?;
                texels.reserve((size as usize).pow(3));
            }
            "LUT_1D_SIZE" => return Err("a 1-D LUT; grading takes a 3-D one".into()),
            "DOMAIN_MIN" | "DOMAIN_MAX" => {
                let want = if first == "DOMAIN_MIN" { 0.0 } else { 1.0 };
                if !words.all(|w| w.parse::<f32>().is_ok_and(|v| v == want)) {
                    return Err(format!("line {}: only the 0..1 domain", n + 1));
                }
            }
            _ => {
                let rgb: Vec<f32> = line.split_whitespace().filter_map(|w| w.parse().ok()).collect();
                if rgb.len() != 3 {
                    return Err(format!("line {}: expected three numbers", n + 1));
                }
                texels.push([rgb[0], rgb[1], rgb[2], 1.0]);
            }
        }
    }
    if size == 0 || texels.len() != (size as usize).pow(3) {
        return Err(format!("{} entries for LUT_3D_SIZE {size}", texels.len()));
    }
    Ok((size, texels))
}

/// A transition drawn over the whole picture, set by the game each frame it runs (The Somnus Fracture's ways into
/// and out of its PS2 chapters, `dream_shift.rs`). Kind 0 is none; 1 is pushing into a television, its phosphor
/// stripes and scanlines growing until one stripe fills the view; 2 is the eyes closing, `progress` how far the lids
/// are shut; 3 is a wave of the retro picture spreading through the world from `at`, its front `progress * reach`
/// metres out, read from the depth buffer.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PictureShift {
    pub kind: u32,
    pub progress: f32,
    pub at: [f32; 3],
    pub reach: f32,
    /// Nothing in it flickers (a player's reduced flashing).
    pub steady: bool,
}

/// 16 scalars, 32 response samples, the dream-lens vector, the picture shift (two vectors) and the inverse
/// view-projection the wave reads the world through, the stylise vector and the LUT vector; then the photographic
/// camera's vector, its lens and its white balance (three rows).
const POST_PARAM_FLOATS: usize = 104;

/// Post-processing pass: HDR texture → tone-mapped swapchain output.
pub struct PostProcessPass {
    pub pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    pub hdr_texture: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    /// Pre-water snapshot of `hdr_texture`, sampled by the water pass for
    /// refraction (Phase 22).
    pub scene_copy_texture: wgpu::Texture,
    pub scene_copy_view: wgpu::TextureView,
    params_buffer: wgpu::Buffer,
    /// Handle to the auto-exposure result, kept so the bind group can be
    /// rebuilt on resize without threading it back through the caller.
    exposure_buffer: wgpu::Buffer,
    /// Bloom chain, kept so the bind group survives a resize.
    bloom_view: wgpu::TextureView,
    /// The grading LUT (binding 5): an unread 2^3 placeholder until `set_lut`.
    lut_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    /// Group 1: the visibility buffer and the paint words, from which a pixel is known to belong to a draw kept out
    /// of the retro picture (`vertex_paint::OUTSIDE_RETRO`). Placeholders until `bind_mask`.
    mask_layout: wgpu::BindGroupLayout,
    mask_group: wgpu::BindGroup,
}

impl PostProcessPass {
    /// Create the post-process pass.
    ///
    /// - `surface_format`: swapchain format (pipeline output target).
    /// - `width` / `height`: initial render dimensions in physical pixels.
    pub fn new(
        device: &wgpu::Device,
        shaders: &crate::shaders::Shaders,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        // Result buffer of `crate::pass::auto_exposure::AutoExposurePass`.
        exposure_buffer: &wgpu::Buffer,
        bloom_view: &wgpu::TextureView,
    ) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Phase 24T: the blurred bloom chain.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Phase 24A-3: metered exposure, written by the
                // auto-exposure compute pass earlier in the frame.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The grading LUT (Sunscar, 5 Nov 2026), read through the clamped linear sampler.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Default: exposure=1.0, vignette_strength=1.0.
        let params_buffer = {
            // Must match `PostParams` in postprocess.wgsl exactly: three floats
            // then four u32s carried as raw bits. `copy_from_slice` below
            // requires the lengths to agree.
            let mut data = [0.0_f32; POST_PARAM_FLOATS];
            data[..16].copy_from_slice(&[
                1.0 / (1.2 * 32768.0), // exposure at EV100 15
                1.0,                   // vignette strength
                0.0,                   // chromatic aberration
                f32::from_bits(0),     // tonemapper: AgX
                f32::from_bits(1),     // auto exposure on
                0.0,                   // bloom intensity
                0.0,
                0.0, // temperature, tint
                1.0,
                1.0, // contrast, saturation
                1.0,
                0.0,
                1.0, // gain, lift, gamma
                0.0, // grain
                0.0, // time
                0.0, // response curve disabled
            ]);
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PostProcess Params"),
                size: (POST_PARAM_FLOATS * 4) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            buf.slice(..)
                .get_mapped_range_mut()
                .expect("mapped_at_creation")
                .copy_from_slice(bytemuck::bytes_of(&data));
            buf.unmap();
            buf
        };

        let (hdr_texture, hdr_view) = Self::make_hdr_texture(device, width, height);
        let (scene_copy_texture, scene_copy_view) =
            Self::make_scene_copy_texture(device, width, height);
        let lut_view = Self::make_lut(device, 2).create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = Self::make_bind_group(
            device,
            &bind_group_layout,
            &hdr_view,
            &sampler,
            &params_buffer,
            exposure_buffer,
            bloom_view,
            &lut_view,
        );
        let exposure_buffer = exposure_buffer.clone();
        let bloom_view = bloom_view.clone();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("postprocess.wgsl").into()),
        });

        let mask_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Retro Mask Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // The scene's depth, for the picture shift's wave (`PictureShift` kind 3).
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        // Until the real ones are bound: nothing visible, no paint table.
        let empty_vis = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("PostProcess Empty Visibility"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rg32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let empty_words = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Empty Paint Words"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let empty_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("PostProcess Empty Depth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mask_group = Self::make_mask_group(device, &mask_layout, &empty_vis, &empty_words, &empty_depth);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Pipeline Layout"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&mask_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Pipeline"),
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
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
        });

        Self {
            exposure_buffer,
            bloom_view,
            pipeline,
            bind_group_layout,
            bind_group,
            hdr_texture,
            hdr_view,
            scene_copy_texture,
            scene_copy_view,
            params_buffer,
            sampler,
            lut_view,
            mask_layout,
            mask_group,
        }
    }

    fn make_mask_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        vis: &wgpu::TextureView,
        words: &wgpu::Buffer,
        depth: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Retro Mask"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(vis) },
                wgpu::BindGroupEntry { binding: 1, resource: words.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth) },
            ],
        })
    }

    /// The visibility buffer, the paint words and the scene's depth this frame (each can be remade: a resize, a grown
    /// paint pool). ponytail: rebuilt each frame; one small bind group, cache by identity if it ever shows in a profile.
    pub fn bind_mask(&mut self, device: &wgpu::Device, vis: &wgpu::TextureView, words: &wgpu::Buffer, depth: &wgpu::TextureView) {
        self.mask_group = Self::make_mask_group(device, &self.mask_layout, vis, words, depth);
    }

    fn make_hdr_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HDR Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                // Phase 22: the water pass refracts what is behind the surface,
                // which means sampling the scene colour. A pass cannot sample
                // the target it renders into, so the HDR texture is copied into
                // `scene_copy_texture` first.
                | wgpu::TextureUsages::COPY_SRC
                // Phase 24F: TAA resolves into its own history buffer and
                // copies the result back here, so every later pass keeps
                // reading one well-known target instead of a view that
                // alternates between two ping-pong textures each frame.
                | wgpu::TextureUsages::COPY_DST
                // Phase 24J: the ray-traced shadow debug view writes straight
                // into the HDR target, so the existing post chain displays it
                // without a separate blit.
                | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Snapshot of the HDR target taken before the water pass, so water can
    /// sample the scene behind it for refraction. Same size and format as the
    /// HDR texture, so a straight texture-to-texture copy is enough.
    fn make_scene_copy_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Scene Colour Copy"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR_FORMAT,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        hdr_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        params_buffer: &wgpu::Buffer,
        exposure_buffer: &wgpu::Buffer,
        bloom_view: &wgpu::TextureView,
        lut_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: exposure_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(bloom_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(lut_view),
                },
            ],
        })
    }

    /// Recreate the HDR texture at the new dimensions (called on window resize).
    pub fn resize(
        &mut self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        bloom_view: &wgpu::TextureView,
    ) {
        // Bloom owns and recreates its mip views. Keeping the clone captured at
        // construction leaves this bind group sampling an orphaned, never
        // rendered texture after the first resize.
        self.bloom_view = bloom_view.clone();
        let (hdr_texture, hdr_view) = Self::make_hdr_texture(device, width, height);
        self.hdr_texture = hdr_texture;
        self.hdr_view = hdr_view;
        let (scene_copy_texture, scene_copy_view) =
            Self::make_scene_copy_texture(device, width, height);
        self.scene_copy_texture = scene_copy_texture;
        self.scene_copy_view = scene_copy_view;
        self.bind_group = Self::make_bind_group(
            device,
            &self.bind_group_layout,
            &self.hdr_view,
            &self.sampler,
            &self.params_buffer,
            &self.exposure_buffer,
            &self.bloom_view,
            &self.lut_view,
        );
    }

    /// Upload exposure and vignette strength to the GPU params buffer.
    /// Upload the post-process parameters.
    ///
    /// `ca_strength` is the chromatic-aberration offset in UV units at the
    /// screen edge; `0.0` disables it (all three channels sample the same texel).
    pub fn set_params(
        &self,
        queue: &wgpu::Queue,
        exposure: f32,
        vignette_strength: f32,
        ca_strength: f32,
        tonemapper: u32,
        auto_exposure: bool,
        bloom_intensity: f32,
        grading: Grading,
        shift: PictureShift,
        inv_view_proj: [f32; 16],
    ) {
        // The uniform is three floats plus a u32; bytemuck cannot cast a mixed
        // array, so the index is bit-cast into the fourth float slot and read
        // back as a u32 in WGSL.
        // Three floats then four u32s. bytemuck cannot cast a mixed array, so
        // the integers ride in float slots as raw bits and WGSL reads them back
        // as u32 — same four bytes either way.
        let mut data = [0.0_f32; POST_PARAM_FLOATS];
        data[..16].copy_from_slice(&[
            exposure,
            vignette_strength,
            ca_strength,
            f32::from_bits(tonemapper),
            f32::from_bits(u32::from(auto_exposure)),
            bloom_intensity,
            grading.temperature,
            grading.tint,
            grading.contrast,
            grading.saturation,
            grading.gain,
            grading.lift,
            grading.gamma,
            grading.grain,
            grading.time,
            f32::from(u8::from(grading.response.is_some())),
        ]);
        if let Some(response) = grading.response {
            data[16..48].copy_from_slice(&response);
        }
        data[48..52].copy_from_slice(&grading.dream);
        data[52..56].copy_from_slice(&[shift.kind as f32, shift.progress, shift.reach, f32::from(u8::from(shift.steady))]);
        data[56..60].copy_from_slice(&[shift.at[0], shift.at[1], shift.at[2], 0.0]);
        data[60..76].copy_from_slice(&inv_view_proj);
        data[76..80].copy_from_slice(&grading.stylise);
        data[80..84].copy_from_slice(&[grading.lut_strength.clamp(0.0, 1.0), grading.low_light.clamp(0.0, 1.0), 0.0, 0.0]);
        // The photographic camera (sunscar-photoreal). All zero but the identity white balance when off, which the
        // shader never reads then.
        data[84..88].copy_from_slice(&[
            f32::from(u8::from(grading.photographic)),
            if grading.photographic { grading.lens_glare.clamp(0.0, 0.5) } else { 0.0 },
            if grading.photographic { grading.sensor_noise.max(0.0) } else { 0.0 },
            0.0,
        ]);
        data[88..92].copy_from_slice(&[grading.lens[0], grading.lens[1], 0.0, 0.0]);
        for (i, row) in grading.white_balance.iter().enumerate() {
            data[92 + i * 4..95 + i * 4].copy_from_slice(row);
        }
        queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&data));
    }

    /// A `size`^3 texture for the grading LUT, Rgba16Float.
    fn make_lut(device: &wgpu::Device, size: u32) -> wgpu::Texture {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Grading LUT"),
                size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
    }

    /// Upload a grading LUT (`parse_cube`): `size`^3 texels, red fastest. Bound from the next `bind_scene` or
    /// `bind_color`; read where `Grading::lut_strength` is above zero.
    pub fn set_lut(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, size: u32, texels: &[[f32; 4]]) {
        assert_eq!(texels.len(), (size as usize).pow(3), "a LUT is size^3 texels");
        let texture = Self::make_lut(device, size);
        let bits: Vec<u16> =
            texels.iter().flatten().map(|&v| crate::pass::lighting_extra::f32_to_f16_bits(v)).collect();
        queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(&bits),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
        );
        self.lut_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    }

    /// Point tone-map at a different HDR colour target (FSR output vs scene).
    pub fn bind_color(&mut self, device: &wgpu::Device, color: &wgpu::TextureView) {
        self.bind_group = Self::make_bind_group(
            device,
            &self.bind_group_layout,
            color,
            &self.sampler,
            &self.params_buffer,
            &self.exposure_buffer,
            &self.bloom_view,
            &self.lut_view,
        );
    }

    /// Restore tone-map sampling of the scene-resolution HDR target.
    pub fn bind_scene(&mut self, device: &wgpu::Device) {
        self.bind_group = Self::make_bind_group(
            device,
            &self.bind_group_layout,
            &self.hdr_view,
            &self.sampler,
            &self.params_buffer,
            &self.exposure_buffer,
            &self.bloom_view,
            &self.lut_view,
        );
    }

    /// Record the post-process pass into `encoder`, writing to `surface_view`.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder, surface_view: &wgpu::TextureView) {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("PostProcess Pass"),
            multiview_mask: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: surface_view,
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
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.set_bind_group(1, &self.mask_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}

#[cfg(test)]
mod white_balance_tests {
    use super::white_balance_matrix;

    fn apply(m: [[f32; 3]; 3], c: [f32; 3]) -> [f32; 3] {
        [0, 1, 2].map(|i| m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2])
    }

    #[test]
    fn off_and_d65_leave_the_light_alone() {
        assert_eq!(white_balance_matrix(0.0), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let d65 = white_balance_matrix(6504.0);
        for (i, row) in d65.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 0.01, "6504 K [{i}][{j}] = {v}");
            }
        }
    }

    #[test]
    fn a_camera_set_warmer_than_the_light_renders_it_cooler_and_keeps_its_brightness() {
        let luma = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        let cool = apply(white_balance_matrix(5000.0), [1.0, 1.0, 1.0]);
        assert!(cool[2] > cool[0] * 1.1, "5000 K renders daylight white bluer: {cool:?}");
        let warm = apply(white_balance_matrix(9000.0), [1.0, 1.0, 1.0]);
        assert!(warm[0] > warm[2] * 1.05, "9000 K renders it warmer: {warm:?}");
        let tungsten = apply(white_balance_matrix(3000.0), [1.0, 1.0, 1.0]);
        assert!(tungsten[2] > tungsten[0] * 1.6, "3000 K (tungsten) turns daylight blue: {tungsten:?}");
        for c in [cool, warm, tungsten] {
            assert!((luma(c) - 1.0).abs() < 0.05, "white keeps its luminance: {c:?}");
        }
    }
}

#[cfg(test)]
mod lut_tests {
    use super::parse_cube;

    #[test]
    fn a_cube_file_reads_red_fastest_and_refuses_what_it_cannot_hold() {
        let mut text = String::from("TITLE \"identity\"
# made by hand
LUT_3D_SIZE 2
DOMAIN_MIN 0 0 0
DOMAIN_MAX 1 1 1
");
        for b in 0..2 {
            for g in 0..2 {
                for r in 0..2 {
                    text += &format!("{r} {g} {b}
");
                }
            }
        }
        let (size, texels) = parse_cube(&text).expect("identity");
        assert_eq!(size, 2);
        assert_eq!(texels[1], [1.0, 0.0, 0.0, 1.0], "red fastest");
        assert_eq!(texels[2], [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(texels[4], [0.0, 0.0, 1.0, 1.0]);
        assert!(parse_cube("LUT_3D_SIZE 2
0 0 0
").is_err(), "too few entries");
        assert!(parse_cube("LUT_1D_SIZE 16
").is_err(), "1-D");
        assert!(parse_cube("LUT_3D_SIZE 2
DOMAIN_MAX 2 2 2
").is_err(), "another domain");
    }
}
