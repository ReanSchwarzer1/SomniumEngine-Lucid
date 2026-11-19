//! GPU grass (Sunscar, 19 Nov 2026): blades laid out every frame on the GPU around the eye and drawn into the
//! visibility buffer, as Ghost of Tsushima draws its grass (Wohllaib, GDC 2021).
//!
//! The developer, 19 Nov: "grass in most areas that have grass is still tooooo sparse, i need DENSE dense". The CPU
//! ground cover grows 100-250 blades per m² near the eye; a real sward is 1,500-6,000 (groundplants' table). This draws
//! the sward itself; groundplants' authored plants (forbs, flowers, tussocks) stay instanced ground cover over it.
//!
//! - **Where blades live:** a span of the shared vertex pool, one `Vertex`-sized record per blade
//!   (`grass_blade.wgsl`), rewritten each frame by `grass_gen.wgsl`. One grass instance (identity model, a material
//!   with [`MATERIAL_FLAG_GRASS`]) names the span, so every pass that rebuilds a surface from the visibility buffer
//!   reads a blade through the pool it already binds (`grass_pool.wgsl`).
//! - **What grows where:** groundplants' habitat table (`data/foliage/grass_habitats.json`), per layer set and splat
//!   slot. Each terrain near the eye gets a habitat map (two habitats and their weights per splat texel) and its
//!   heights, uploaded once as it comes in reach.
//! - **Each frame:** the CPU lists the 8 m tiles in view with grass and how many candidates each of their 0.5 m cells
//!   visits; the GPU keeps, culls (frustum, this frame's Hi-Z) and writes the blades; three indirect draws (3, 2 and 1
//!   segments) put them in the visibility buffer after the frame's final Hi-Z.
//! - **Shadows and GI:** as ground cover. No grass casts a shadow (cover has `shadow_distance` 0) or is traced (small
//!   instances stay out of the TLAS); it takes sun shadows, sky and GI on its own normal, and an occlusion ramp down
//!   each blade in place of self-shadowing.
//!
//! `SOMNIUM_GPU_GRASS=0` turns it off (the A/B). `SOMNIUM_GRASS_CAP` (blades per m², 3000), `SOMNIUM_GRASS_LOD` (full
//! density within, 5 m), `SOMNIUM_GRASS_RANGE` (300 m) and `SOMNIUM_GRASS_MIN_PX` (1.0) tune it;
//! `SOMNIUM_GRASS_TABLE` names another table. `SOMNIUM_GRASS_LOG=1` logs the tile list every 240 frames.

use crate::instance::GpuInstanceData;
use crate::material::pool::{
    GpuMaterial, MaterialPool, MATERIAL_FLAG_DOUBLE_SIDED, MATERIAL_FLAG_FOLIAGE, MATERIAL_FLAG_GRASS,
};

/// Side of a tile the CPU lists (m). Terrain origins sit on this grid, so a tile lies in one terrain.
const TILE: f32 = 8.0;
/// Side of a candidate cell (m): `grass_gen.wgsl` `CELL`; a tile is 16 x 16 of them, one workgroup's threads.
const CELL: f32 = 0.5;
/// Terrains with a ground slot at once: a 300 m reach touches at most four 1 km tiles; two spare for the walk.
const SLOTS: usize = 6;
/// A slot's room: Sunscar's 1 km tiles (513 x 513 heights, 512 x 512 splat texels). A larger terrain gets no grass.
const SLOT_HEIGHTS: usize = 513 * 513;
const SLOT_TEXELS: usize = 512 * 512;
/// `grass_gen.wgsl` `NO_HABITAT`: a habitat index is 5 bits.
const NO_HABITAT: u8 = 31;
/// Blades each class may hold: 3 segments (within `NEAR_M`), 2 (within `MID_M`), 1 beyond. At the default cap and
/// `d_lod` an 80 degree view holds about 100,000, 115,000 and 265,000 (the design in SUNSCAR_FOLIAGE_RENDER.md).
const CAPACITY: [u32; 3] = [240_000, 280_000, 520_000];
const NEAR_M: f32 = 8.0;
const MID_M: f32 = 24.0;
/// The tallest blade a tile's box allows for above its highest ground (m), for frustum culling.
const BLADE_REACH: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    view_proj: [[f32; 4]; 4],
    eye_clip: [f32; 4],
    eye: [f32; 4],
    planes: [[f32; 4]; 6],
    lod: [f32; 4],
    seg: [f32; 4],
    hiz: [f32; 4],
    base: [u32; 4],
    misc: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTile {
    origin: [f32; 2],
    terrain: u32,
    candidates: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuTerrain {
    origin: [f32; 2],
    cell: f32,
    verts: u32,
    height_off: u32,
    hab_off: u32,
    texels: u32,
    texel_size: f32,
}

/// One habitat as `grass_gen.wgsl` reads it. Angles in radians.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuHabitat {
    pub density: f32,
    pub h_min: f32,
    pub h_max: f32,
    pub w_min: f32,
    pub w_max: f32,
    pub lean_min: f32,
    pub lean_max: f32,
    pub lodged: f32,
    pub clump_m: f32,
    pub contrast: f32,
    pub seed_h_min: f32,
    pub seed_h_max: f32,
    pub share_green: f32,
    pub share_dead: f32,
    pub bend_min: f32,
    pub bend_max: f32,
    pub stiffness: f32,
    pub root_ao: f32,
    pub bare: f32,
    pub w_mean: f32,
    pub green: [f32; 4],
    pub green_alt: [f32; 4],
    pub dead: [f32; 4],
    pub dead_alt: [f32; 4],
    pub seed: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuDraw {
    slot_base: u32,
    capacity: u32,
    instance: u32,
    _pad: u32,
}

/// groundplants' habitat table (`data/foliage/grass_habitats.json`), ready for the GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct HabitatTable {
    pub habitats: Vec<GpuHabitat>,
    pub names: Vec<String>,
    /// Per layer set ("" the project's own): each splat slot's habitat and its weight in it.
    pub sets: std::collections::HashMap<String, [(u8, f32); 32]>,
    pub season: String,
    /// Palette kinds the blades replace, by their source file's name (`"replaces"`: prefixes, as "gp_sward_"):
    /// not drawn as ground cover while the blades draw. `SOMNIUM_GRASS_REPLACES` (comma-separated) overrides it.
    pub replaces: Vec<String>,
}

fn pair(v: &serde_json::Value, key: &str, default: [f32; 2]) -> [f32; 2] {
    match v.get(key).and_then(|x| x.as_array()) {
        Some(a) if a.len() >= 2 => [
            a[0].as_f64().map_or(default[0], |x| x as f32),
            a[1].as_f64().map_or(default[1], |x| x as f32),
        ],
        _ => default,
    }
}

fn number(v: &serde_json::Value, key: &str, default: f32) -> f32 {
    v.get(key).and_then(serde_json::Value::as_f64).map_or(default, |x| x as f32)
}

fn colour(v: &serde_json::Value, key: &str, default: [f32; 3]) -> [f32; 4] {
    let c = pair3(v.get("colour").and_then(|c| c.get(key)), default);
    [c[0], c[1], c[2], 1.0]
}

fn pair3(v: Option<&serde_json::Value>, default: [f32; 3]) -> [f32; 3] {
    match v.and_then(|x| x.as_array()) {
        Some(a) if a.len() >= 3 => std::array::from_fn(|i| a[i].as_f64().map_or(default[i], |x| x as f32)),
        _ => default,
    }
}

impl HabitatTable {
    /// Parse the table. A (set, slot) two habitats claim is refused: which one grows there would depend on file order.
    pub fn parse(text: &str) -> Result<Self, String> {
        let doc: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("grass habitats: {e}"))?;
        let season = doc.get("season_now").and_then(|s| s.as_str()).unwrap_or("winter").to_owned();
        let rows = doc.get("habitats").and_then(|h| h.as_array()).ok_or("grass habitats: no \"habitats\" list")?;
        let replaces = match std::env::var("SOMNIUM_GRASS_REPLACES") {
            Ok(list) => list.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_owned).collect(),
            Err(_) => doc
                .get("replaces")
                .and_then(|r| r.as_array())
                .map(|a| a.iter().filter_map(|s| s.as_str()).map(str::to_owned).collect())
                .unwrap_or_default(),
        };
        let mut table =
            Self { habitats: Vec::new(), names: Vec::new(), sets: Default::default(), season, replaces };
        for row in rows {
            if table.habitats.len() >= usize::from(NO_HABITAT) {
                tracing::warn!("grass habitats: more than {NO_HABITAT}; the rest are left out");
                break;
            }
            let id = row.get("id").and_then(|s| s.as_str()).unwrap_or("?").to_owned();
            let index = u8::try_from(table.habitats.len()).expect("under 31");
            let h = pair(row, "height_m", [0.1, 0.4]);
            let w = pair(row, "width_m", [0.002, 0.006]);
            let lean = pair(row, "lean_deg", [5.0, 55.0]);
            let seed_h = pair(row, "seed_height_m", [h[1], h[1] * 2.0]);
            let bend = pair(row, "bend", [0.2, 0.6]);
            let shares = row.get("shares").and_then(|s| s.get(&table.season).or_else(|| s.get("winter")));
            let share = |k: &str, d: f32| shares.and_then(|s| s.get(k)).and_then(serde_json::Value::as_f64).map_or(d, |x| x as f32);
            let (green, dead) = (share("green", 0.6), share("dead", 0.35));
            table.habitats.push(GpuHabitat {
                density: number(row, "blades_per_m2", 1500.0).max(0.0),
                h_min: h[0],
                h_max: h[1],
                w_min: w[0],
                w_max: w[1],
                lean_min: lean[0].to_radians(),
                lean_max: lean[1].to_radians(),
                lodged: number(row, "lodged_share", 0.1),
                clump_m: number(row, "clump_m", 2.0),
                contrast: number(row, "clump_contrast", 0.3),
                seed_h_min: seed_h[0],
                seed_h_max: seed_h[1],
                share_green: green,
                share_dead: dead,
                bend_min: bend[0],
                bend_max: bend[1],
                stiffness: number(row, "stiffness", 0.5),
                root_ao: number(row, "root_ao", 0.45),
                bare: number(row, "bare_share", 0.0),
                w_mean: 0.5 * (w[0] + w[1]),
                green: colour(row, "green", [0.048, 0.071, 0.02]),
                green_alt: colour(row, "green_alt", [0.072, 0.087, 0.027]),
                dead: colour(row, "dead", [0.118, 0.107, 0.088]),
                dead_alt: colour(row, "dead_alt", [0.189, 0.152, 0.094]),
                seed: colour(row, "seed", [0.34, 0.313, 0.239]),
            });
            let sets: Vec<String> = row
                .get("sets")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().filter_map(|s| s.as_str()).map(|s| s.trim().to_owned()).collect())
                .unwrap_or_else(|| vec![String::new()]);
            let slots = row.get("slots").and_then(|s| s.as_object());
            for set in sets {
                let map = table.sets.entry(set.clone()).or_insert([(NO_HABITAT, 0.0); 32]);
                for (slot, weight) in slots.into_iter().flatten() {
                    let Some(slot) = slot.parse::<usize>().ok().filter(|s| *s < 32) else { continue };
                    if map[slot].0 != NO_HABITAT {
                        return Err(format!(
                            "grass habitats: slot {slot} of set \"{set}\" is claimed by {} and {id}",
                            table.names[usize::from(map[slot].0)]
                        ));
                    }
                    map[slot] = (index, weight.as_f64().unwrap_or(1.0) as f32);
                }
            }
            table.names.push(id);
        }
        Ok(table)
    }

    /// Two habitats per splat texel (the strongest), packed as `grass_gen.wgsl` reads them:
    /// habitat a | weight a << 5 | habitat b << 13 | weight b << 18. Texels in no habitat hold 31 with weight 0.
    pub fn habitat_word(map: &[(u8, f32); 32], texel: &[u8; 32]) -> u32 {
        let mut found: [(u8, f32); 4] = [(NO_HABITAT, 0.0); 4];
        let mut n = 0;
        for (slot, &w) in texel.iter().enumerate() {
            let (h, f) = map[slot];
            if w == 0 || h == NO_HABITAT {
                continue;
            }
            let add = f32::from(w) / 255.0 * f;
            if let Some(e) = found[..n].iter_mut().find(|e| e.0 == h) {
                e.1 += add;
            } else if n < found.len() {
                found[n] = (h, add);
                n += 1;
            }
        }
        found[..n].sort_by(|a, b| b.1.total_cmp(&a.1));
        let q = |w: f32| (w.clamp(0.0, 1.0) * 255.0).round() as u32;
        let (a, b) = (found[0], if n > 1 { found[1] } else { (NO_HABITAT, 0.0) });
        u32::from(a.0) | (q(a.1) << 5) | (u32::from(b.0) << 13) | (q(b.1) << 18)
    }
}

/// Tuning, read once from the environment.
#[derive(Clone, Copy, Debug)]
struct Knobs {
    cap: f32,
    d_lod: f32,
    range: f32,
    min_px: f32,
    log: bool,
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<f32>().ok()).filter(|v| v.is_finite() && *v > 0.0).unwrap_or(default)
}

impl Knobs {
    fn from_env() -> Self {
        Self {
            cap: env_f32("SOMNIUM_GRASS_CAP", 3000.0),
            d_lod: env_f32("SOMNIUM_GRASS_LOD", 5.0),
            range: env_f32("SOMNIUM_GRASS_RANGE", 300.0),
            min_px: env_f32("SOMNIUM_GRASS_MIN_PX", 1.0),
            log: std::env::var("SOMNIUM_GRASS_LOG").as_deref() == Ok("1"),
        }
    }

    /// Candidates a 0.5 m cell holds at the cap: `grass_gen.wgsl` ranks a candidate by k / this.
    fn per_cell(&self) -> u32 {
        (self.cap * CELL * CELL).ceil().max(1.0) as u32
    }
}

/// One terrain's ground on the GPU: which terrain, and per 8 m tile its ground's height range and whether grass grows.
struct GroundSlot {
    terrain_id: u32,
    origin: glam::Vec3,
    size: glam::Vec2,
    tiles: [u32; 2],
    /// (lowest, highest) ground per tile, NaN where no grass grows.
    tile_range: Vec<[f32; 2]>,
}

/// The pass. Built when a habitat table loads; a game without one never builds it.
pub struct GrassPass {
    table: HabitatTable,
    knobs: Knobs,
    /// Each terrain's habitat set by terrain id (`set_terrain_set`): `None` until the game says, and grows nothing.
    terrain_sets: Vec<Option<String>>,
    slots: [Option<GroundSlot>; SLOTS],
    /// The blades' span in the vertex pool, and the grass instance's material.
    span: u32,
    material_id: u32,

    gen_pipeline: wgpu::ComputePipeline,
    gen_layout: wgpu::BindGroupLayout,
    draw_pipeline: wgpu::RenderPipeline,
    draw_groups: [wgpu::BindGroup; 3],
    draw_uniforms: [wgpu::Buffer; 3],
    params: wgpu::Buffer,
    tiles: wgpu::Buffer,
    chunks: wgpu::Buffer,
    terrains: wgpu::Buffer,
    heights: wgpu::Buffer,
    habitat_map: wgpu::Buffer,
    habitats: wgpu::Buffer,
    draws: wgpu::Buffer,

    tile_scratch: Vec<GpuTile>,
    chunk_scratch: Vec<[u32; 2]>,
    chunk_count: u32,
    frame: u32,
    /// The grass instance's index this frame, set by the renderer after it appends it (`instance`).
    instance_index: Option<u32>,
    /// Blades were laid out last frame (not held off by the A/B): the kinds they replace stay hidden.
    drawing: bool,
}

fn storage(device: &wgpu::Device, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
        mapped_at_creation: false,
    })
}

fn entry(binding: u32, ty: wgpu::BindingType, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility, ty, count: None }
}

fn buffer_ty(kind: wgpu::BufferBindingType) -> wgpu::BindingType {
    wgpu::BindingType::Buffer { ty: kind, has_dynamic_offset: false, min_binding_size: None }
}

/// The table's path: `SOMNIUM_GRASS_TABLE`, else `<project>/data/foliage/grass_habitats.json`.
pub fn table_path(project_root: Option<&std::path::Path>) -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("SOMNIUM_GRASS_TABLE") {
        return Some(std::path::PathBuf::from(path));
    }
    project_root.map(|root| root.join("data").join("foliage").join("grass_habitats.json"))
}

/// `SOMNIUM_GPU_GRASS_AB=captures` (proof stills): no blades until the run's first capture is written, then blades,
/// none, ... by each one written, so a walk naming every view twice shoots before and after in one launch, as
/// `SOMNIUM_SURFACES_AB` does. `=1` (cost): none in the frame=120, 360, ... blocks of a run with
/// SOMNIUM_PROFILE_EVERY=120. While there are none the kinds the blades replace draw again.
pub fn ab_off(frame: u64) -> bool {
    static AB: std::sync::LazyLock<u8> =
        std::sync::LazyLock::new(|| match std::env::var("SOMNIUM_GPU_GRASS_AB").as_deref() {
            Ok("1") => 1,
            Ok("captures") => 2,
            _ => 0,
        });
    match *AB {
        1 => (frame + 60) / 120 % 2 == 1,
        2 => crate::capture::captures_written() % 2 == 0,
        _ => false,
    }
}

/// Whether GPU grass may run at all: `SOMNIUM_GPU_GRASS=0` turns it off.
pub fn enabled() -> bool {
    std::env::var("SOMNIUM_GPU_GRASS").as_deref() != Ok("0")
}

impl GrassPass {
    /// Build the pass for `table`: reserves the blades' span in the vertex pool and adds the grass material. Call it
    /// between frames (the renderer rebinds a grown material buffer at the top of the next). `None` when the pool has
    /// no room.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shaders: &crate::shaders::Shaders,
        global_bind_group_layout: &wgpu::BindGroupLayout,
        geometry: &mut crate::geometry::GeometryPool,
        materials: &mut MaterialPool,
        table: HabitatTable,
    ) -> Option<Self> {
        let total: u32 = CAPACITY.iter().sum();
        let Some(span) = geometry.reserve_vertices(total) else {
            tracing::warn!("GPU grass: the geometry pool has no room for {total} blades; no grass");
            return None;
        };
        let material_id = materials.add_material(queue, grass_material());
        let knobs = Knobs::from_env();
        let compute = wgpu::ShaderStages::COMPUTE;
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let gen_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grass Gen BGL"),
            entries: &[
                entry(0, buffer_ty(wgpu::BufferBindingType::Storage { read_only: false }), compute),
                entry(1, buffer_ty(wgpu::BufferBindingType::Uniform), compute),
                entry(2, buffer_ty(read), compute),
                entry(3, buffer_ty(read), compute),
                entry(4, buffer_ty(read), compute),
                entry(5, buffer_ty(read), compute),
                entry(6, buffer_ty(read), compute),
                entry(7, buffer_ty(read), compute),
                entry(
                    8,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    compute,
                ),
                entry(9, buffer_ty(wgpu::BufferBindingType::Storage { read_only: false }), compute),
            ],
        });
        let gen_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grass Gen Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("grass_gen.wgsl").into()),
        });
        let gen_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grass Gen Pipeline Layout"),
            bind_group_layouts: &[Some(&gen_layout)],
            immediate_size: 0,
        });
        let gen_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Grass Gen Pipeline"),
            layout: Some(&gen_pipeline_layout),
            module: &gen_shader,
            entry_point: Some("grass_gen"),
            compilation_options: Default::default(),
            cache: None,
        });

        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grass Draw BGL"),
            entries: &[entry(
                0,
                buffer_ty(wgpu::BufferBindingType::Uniform),
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            )],
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grass Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(shaders.source_or_panic("grass_vis.wgsl").into()),
        });
        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Grass Draw Pipeline Layout"),
            bind_group_layouts: &[Some(global_bind_group_layout), Some(&draw_layout)],
            immediate_size: 0,
        });
        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grass Draw Pipeline"),
            layout: Some(&draw_pipeline_layout),
            multiview_mask: None,
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: Some("vs_grass"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: Some("fs_grass"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rg32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Blades are two-sided, and the strip's winding alternates.
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                // Reversed-Z: nearer is larger (renderer.rs `set_view`).
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
            cache: None,
        });
        // One pipeline serves the three classes: a class differs only in its draw's vertex count and its uniform.
        let draw_uniforms: [wgpu::Buffer; 3] = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(["Grass Draw 3", "Grass Draw 2", "Grass Draw 1"][i]),
                size: std::mem::size_of::<GpuDraw>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let draw_groups: [wgpu::BindGroup; 3] = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Grass Draw BG"),
                layout: &draw_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: draw_uniforms[i].as_entire_binding() }],
            })
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grass Params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let none = wgpu::BufferUsages::empty();
        let habitats = storage(device, "Grass Habitats", (std::mem::size_of::<GpuHabitat>() * table.habitats.len()) as u64, none);
        let terrains = storage(device, "Grass Terrains", (std::mem::size_of::<GpuTerrain>() * SLOTS) as u64, none);
        let heights = storage(device, "Grass Heights", (SLOT_HEIGHTS * SLOTS * 4) as u64, none);
        let habitat_map = storage(device, "Grass Habitat Map", (SLOT_TEXELS * SLOTS * 4) as u64, none);
        let tiles = storage(device, "Grass Tiles", 4096 * std::mem::size_of::<GpuTile>() as u64, none);
        let chunks = storage(device, "Grass Chunks", 16384 * 8, none);
        let draws = storage(device, "Grass Draw Args", 48, wgpu::BufferUsages::INDIRECT);
        queue.write_buffer(&habitats, 0, bytemuck::cast_slice(&table.habitats));
        tracing::info!(
            habitats = table.habitats.len(),
            sets = table.sets.len(),
            season = table.season.as_str(),
            blades = total,
            "GPU grass: on"
        );
        Some(Self {
            table,
            knobs,
            terrain_sets: Vec::new(),
            slots: Default::default(),
            span,
            material_id,
            gen_pipeline,
            gen_layout,
            draw_pipeline,
            draw_groups,
            draw_uniforms,
            params,
            tiles,
            chunks,
            terrains,
            heights,
            habitat_map,
            habitats,
            draws,
            tile_scratch: Vec::new(),
            chunk_scratch: Vec::new(),
            chunk_count: 0,
            frame: 0,
            instance_index: None,
            drawing: false,
        })
    }

    /// Whether blades are laid out this frame (on, and not held off by the A/B).
    pub fn drawing(&self) -> bool {
        self.drawing
    }

    /// The palette kinds (source name prefixes) not to draw as ground cover: the blades' while they draw, else none.
    pub fn replaces(&self) -> &[String] {
        if self.drawing { self.table.replaces.as_slice() } else { &[] }
    }

    /// Which habitat set terrain `terrain_id` grows (its ground layer set; "" the project's own). Cheap when unchanged.
    pub fn set_terrain_set(&mut self, terrain_id: u32, set: &str) {
        let i = terrain_id as usize;
        if self.terrain_sets.len() <= i {
            self.terrain_sets.resize(i + 1, None);
        }
        if self.terrain_sets[i].as_deref() != Some(set) {
            self.terrain_sets[i] = Some(set.to_owned());
            // A slot built under another set is rebuilt.
            for slot in &mut self.slots {
                if slot.as_ref().is_some_and(|s| s.terrain_id == terrain_id) {
                    *slot = None;
                }
            }
        }
    }

    /// The grass instance for this frame's instance buffer, or `None` when nothing is in view.
    pub fn instance(&self) -> Option<GpuInstanceData> {
        (self.chunk_count > 0).then(|| GpuInstanceData {
            model_matrix: glam::Mat4::IDENTITY.to_cols_array_2d(),
            material_id: self.material_id,
            mesh_vertex_offset: self.span,
            mesh_index_offset: 0,
            _padding: 0,
        })
    }

    /// The index the renderer gave the grass instance this frame (after `instance`), for the draws' uniforms.
    pub fn set_instance_index(&mut self, queue: &wgpu::Queue, index: Option<u32>) {
        self.instance_index = index;
        let Some(instance) = index else { return };
        let mut base = 0;
        for (class, uniform) in self.draw_uniforms.iter().enumerate() {
            let draw = GpuDraw { slot_base: base, capacity: CAPACITY[class], instance, _pad: 0 };
            queue.write_buffer(uniform, 0, bytemuck::bytes_of(&draw));
            base += CAPACITY[class];
        }
    }

    /// Give each terrain in reach its ground slot (heights and habitat map), one terrain a frame: building one reads
    /// its 262,144 splat texels (a few ms) and uploads 2 MB. Slots out of reach are freed.
    fn update_slots(&mut self, queue: &wgpu::Queue, terrains: &[crate::terrain::TerrainData], eye: glam::Vec3) {
        let reach = self.knobs.range + 64.0;
        let hold = self.knobs.range + 256.0;
        for s in &mut self.slots {
            let drop = s.as_ref().is_some_and(|slot| {
                terrains.get(slot.terrain_id as usize).is_none_or(|t| t.heightmap.is_empty())
                    || rect_distance(slot.origin, slot.size, eye) > hold
            });
            if drop {
                *s = None;
            }
        }
        for (id, t) in terrains.iter().enumerate() {
            let id = id as u32;
            if t.heightmap.is_empty() || self.slots.iter().flatten().any(|s| s.terrain_id == id) {
                continue;
            }
            let Some(Some(set)) = self.terrain_sets.get(id as usize) else { continue };
            let Some(map) = self.table.sets.get(set) else { continue };
            let origin = t.model.w_axis.truncate();
            let size = glam::Vec2::from(t.desc.world_size());
            if rect_distance(origin, size, eye) > reach {
                continue;
            }
            let (nx, nz) = (t.desc.total_vertices_x() as usize, t.desc.total_vertices_z() as usize);
            let (sw, sh) = (t.splatmap.width as usize, t.splatmap.height as usize);
            if nx != nz
                || sw != sh
                || sw == 0
                || nx * nz > SLOT_HEIGHTS
                || sw * sh > SLOT_TEXELS
                || t.heightmap.len() < nx * nz
                || t.splatmap.data.len() < sw * sh
            {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(nx, nz, sw, sh, "GPU grass: a terrain does not fit a ground slot; no grass on it");
                });
                continue;
            }
            let Some(free) = self.slots.iter().position(Option::is_none) else { break };
            let scale = t.desc.height_scale;
            let heights: Vec<f32> = t.heightmap[..nx * nz].iter().map(|h| h * scale + origin.y).collect();
            let words: Vec<u32> =
                t.splatmap.data[..sw * sh].iter().map(|texel| HabitatTable::habitat_word(map, texel)).collect();
            let cell = t.desc.cell_size;
            let texel_size = size.x / sw as f32;
            let tiles = [(size.x / TILE).floor() as u32, (size.y / TILE).floor() as u32];
            let tile_range = tile_ranges(&heights, nx, &words, sw, cell, texel_size, tiles);
            queue.write_buffer(&self.heights, (free * SLOT_HEIGHTS * 4) as u64, bytemuck::cast_slice(&heights));
            queue.write_buffer(&self.habitat_map, (free * SLOT_TEXELS * 4) as u64, bytemuck::cast_slice(&words));
            let record = GpuTerrain {
                origin: [origin.x, origin.z],
                cell,
                verts: nx as u32,
                height_off: (free * SLOT_HEIGHTS) as u32,
                hab_off: (free * SLOT_TEXELS) as u32,
                texels: sw as u32,
                texel_size,
            };
            queue.write_buffer(
                &self.terrains,
                (free * std::mem::size_of::<GpuTerrain>()) as u64,
                bytemuck::bytes_of(&record),
            );
            if self.knobs.log {
                let grassy = tile_range.iter().filter(|r| !r[0].is_nan()).count();
                tracing::info!(terrain = id, set = set.as_str(), slot = free, grassy_tiles = grassy, "GPU grass: ground slot built");
            }
            self.slots[free] = Some(GroundSlot { terrain_id: id, origin, size, tiles, tile_range });
            break;
        }
    }

    /// Lay out this frame's tiles and upload what the GPU needs. `view_proj_rel` is the jittered, reversed-Z
    /// view-projection of camera-relative points (`Renderer::view_proj_rel`); `pixel_angle` the radians one pixel
    /// spans; `hiz` the pyramid's size and levels when it holds this frame's depth.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        terrains: &[crate::terrain::TerrainData],
        eye: glam::Vec3,
        view_proj_rel: glam::Mat4,
        pixel_angle: f32,
        hiz: Option<((u32, u32), u32)>,
        on: bool,
    ) {
        self.chunk_count = 0;
        self.instance_index = None;
        self.frame = self.frame.wrapping_add(1);
        self.drawing = on;
        if !on {
            return;
        }
        self.update_slots(queue, terrains, eye);

        let k = self.knobs;
        let planes = frustum_planes(&view_proj_rel);
        let per_cell = k.per_cell();
        self.tile_scratch.clear();
        self.chunk_scratch.clear();
        let eye_xz = glam::Vec2::new(eye.x, eye.z);
        for (slot_index, slot) in self.slots.iter().enumerate() {
            let Some(slot) = slot else { continue };
            let origin = glam::Vec2::new(slot.origin.x, slot.origin.z);
            let max = glam::Vec2::new(slot.tiles[0] as f32, slot.tiles[1] as f32);
            let t_lo = ((eye_xz - k.range - origin) / TILE).floor().clamp(glam::Vec2::ZERO, max);
            let t_hi = ((eye_xz + k.range - origin) / TILE).ceil().clamp(glam::Vec2::ZERO, max);
            for tz in t_lo.y as u32..t_hi.y as u32 {
                for tx in t_lo.x as u32..t_hi.x as u32 {
                    let [y0, y1] = slot.tile_range[(tz * slot.tiles[0] + tx) as usize];
                    if y0.is_nan() {
                        continue;
                    }
                    let o = origin + glam::Vec2::new(tx as f32, tz as f32) * TILE;
                    let dx = (o.x - eye.x).max(eye.x - (o.x + TILE)).max(0.0);
                    let dz = (o.y - eye.z).max(eye.z - (o.y + TILE)).max(0.0);
                    let d = dx.hypot(dz);
                    if d >= k.range {
                        continue;
                    }
                    let lo = glam::Vec3::new(o.x, y0, o.y) - eye;
                    let hi = glam::Vec3::new(o.x + TILE, y1 + BLADE_REACH, o.y + TILE) - eye;
                    if !aabb_in_frustum(&planes, lo, hi) {
                        continue;
                    }
                    let n = tile_candidates(per_cell, k.d_lod, d);
                    if n == 0 {
                        continue;
                    }
                    let tile = self.tile_scratch.len() as u32;
                    self.tile_scratch.push(GpuTile { origin: o.to_array(), terrain: slot_index as u32, candidates: n });
                    self.chunk_scratch.extend((0..n).map(|c| [tile, c]));
                }
            }
        }
        if self.chunk_scratch.is_empty() {
            return;
        }
        self.chunk_count = self.chunk_scratch.len() as u32;
        if k.log && self.frame % 240 == 0 {
            tracing::info!(tiles = self.tile_scratch.len(), chunks = self.chunk_count, "GPU grass");
        }

        let tile_bytes: &[u8] = bytemuck::cast_slice(&self.tile_scratch);
        if tile_bytes.len() as u64 > self.tiles.size() {
            self.tiles = storage(device, "Grass Tiles", (tile_bytes.len() as u64).next_power_of_two(), wgpu::BufferUsages::empty());
        }
        queue.write_buffer(&self.tiles, 0, tile_bytes);
        let chunk_bytes: &[u8] = bytemuck::cast_slice(&self.chunk_scratch);
        if chunk_bytes.len() as u64 > self.chunks.size() {
            self.chunks = storage(device, "Grass Chunks", (chunk_bytes.len() as u64).next_power_of_two(), wgpu::BufferUsages::empty());
        }
        queue.write_buffer(&self.chunks, 0, chunk_bytes);

        let ((hiz_w, hiz_h), hiz_mips) = hiz.unwrap_or(((1, 1), 1));
        let params = GpuParams {
            view_proj: view_proj_rel.to_cols_array_2d(),
            eye_clip: view_proj_rel.w_axis.to_array(),
            eye: [eye.x, eye.y, eye.z, pixel_angle],
            planes,
            lod: [k.d_lod, k.range, k.range * 0.82, k.cap],
            seg: [NEAR_M, MID_M, k.min_px, per_cell as f32],
            hiz: [hiz_w as f32, hiz_h as f32, hiz_mips as f32, if hiz.is_some() { 1.0 } else { 0.0 }],
            base: [self.span, CAPACITY[0], CAPACITY[1], CAPACITY[2]],
            misc: [self.tile_scratch.len() as u32, self.chunk_count, self.table.habitats.len() as u32, self.frame],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
        // Each class's draw: 3 vertices a triangle, 2S - 1 triangles a blade; the counts start at none.
        let args: [u32; 12] = [15, 0, 0, 0, 9, 0, 0, 0, 3, 0, 0, 0];
        queue.write_buffer(&self.draws, 0, bytemuck::cast_slice(&args));
    }

    /// Lay the blades out (compute) and draw them into the visibility buffer, after the frame's final Hi-Z.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        vertex_buffer: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
        global_group: &wgpu::BindGroup,
        vis_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        if self.chunk_count == 0 || self.instance_index.is_none() {
            return;
        }
        // Built each frame: microseconds, and never stale after a resize or a grown buffer.
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Grass Gen BG"),
            layout: &self.gen_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: vertex_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tiles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.chunks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.terrains.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.heights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: self.habitat_map.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.habitats.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(hiz_view) },
                wgpu::BindGroupEntry { binding: 9, resource: self.draws.as_entire_binding() },
            ],
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Grass Gen"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gen_pipeline);
            cpass.set_bind_group(0, &group, &[]);
            cpass.dispatch_workgroups(self.chunk_count.min(65_535), self.chunk_count.div_ceil(65_535), 1);
        }
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Grass Draw"),
            multiview_mask: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: vis_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_bind_group(0, global_group, &[]);
        rpass.set_pipeline(&self.draw_pipeline);
        for class in 0..3 {
            rpass.set_bind_group(1, &self.draw_groups[class], &[]);
            rpass.draw_indirect(&self.draws, class as u64 * 16);
        }
    }
}

/// Candidates each cell of a tile visits when its nearest point is `d` from the eye: the most the tile keeps anywhere,
/// full density within `d_lod`, then a quarter per doubling of distance (`grass_gen.wgsl` cuts each by its own).
fn tile_candidates(per_cell: u32, d_lod: f32, d: f32) -> u32 {
    let keep = (d_lod / d.max(1e-3)).powi(2).min(1.0);
    (per_cell as f32 * keep).ceil() as u32
}

/// Per 8 m tile of a terrain: its ground's (lowest, highest) heights, or NaN where no grass can grow (no texel of it
/// reaches the 0.35 grassland weight presence starts at).
fn tile_ranges(heights: &[f32], nx: usize, words: &[u32], texels: usize, cell: f32, texel_size: f32, tiles: [u32; 2]) -> Vec<[f32; 2]> {
    let nz = heights.len() / nx;
    let vpt = (TILE / cell).round().max(1.0) as usize;
    let tpt = (TILE / texel_size).round().max(1.0) as usize;
    let mut out = vec![[f32::NAN; 2]; (tiles[0] * tiles[1]) as usize];
    for tz in 0..tiles[1] as usize {
        for tx in 0..tiles[0] as usize {
            let grows = (tz * tpt..((tz + 1) * tpt).min(texels)).any(|j| {
                (tx * tpt..((tx + 1) * tpt).min(texels)).any(|i| {
                    let w = words[j * texels + i];
                    ((w >> 5) & 255) + ((w >> 18) & 255) >= 89
                })
            });
            if !grows {
                continue;
            }
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for j in tz * vpt..=((tz + 1) * vpt).min(nz - 1) {
                for i in tx * vpt..=((tx + 1) * vpt).min(nx - 1) {
                    let h = heights[j * nx + i];
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            out[tz * tiles[0] as usize + tx] = [lo, hi];
        }
    }
    out
}

/// Horizontal distance from `eye` to a terrain's rectangle (0 inside it).
fn rect_distance(origin: glam::Vec3, size: glam::Vec2, eye: glam::Vec3) -> f32 {
    let dx = (origin.x - eye.x).max(eye.x - (origin.x + size.x)).max(0.0);
    let dz = (origin.z - eye.z).max(eye.z - (origin.z + size.y)).max(0.0);
    dx.hypot(dz)
}

/// The grass instance's material: two-sided foliage (its pixels shade and count as foliage), wind on, no maps.
fn grass_material() -> GpuMaterial {
    GpuMaterial {
        base_color: [1.0, 1.0, 1.0, 1.0],
        roughness: 0.55,
        metallic: 0.0,
        albedo_map: -1,
        normal_map: -1,
        metallic_roughness_map: -1,
        // Above 0 so every pass takes it for a leaf (`thin_leaf`, the pixel class); the grass draw has no alpha test.
        alpha_cutoff: 0.5,
        flags: MATERIAL_FLAG_DOUBLE_SIDED | MATERIAL_FLAG_FOLIAGE | MATERIAL_FLAG_GRASS,
        occlusion_map: -1,
        transmission: 0.3,
        emissive: [0.0; 3],
        emissive_map: -1,
        terrain_index: -1,
        porosity: 0.3,
        normal_scale: 1.0,
        height_map: -1,
        height_depth: 0.0,
        weathering: 0.0,
        detile: 0.0,
        // `grass_blade.wgsl` `grass_wind` scales these by each blade's stiffness: a 0.3 m blade's tip moves about
        // 0.1 m in a 4 m/s wind.
        wind: [2.0, 0.03],
        subsurface: 0.0,
        reserved: 0.0,
    }
}

/// Camera-relative frustum planes of a reversed-Z view-projection (Gribb and Hartmann), inside where
/// dot(n, p) + d >= 0, normalised. The far plane of an infinite reversed-Z projection is no plane; its row is kept.
fn frustum_planes(m: &glam::Mat4) -> [[f32; 4]; 6] {
    let row = |i: usize| glam::Vec4::new(m.x_axis[i], m.y_axis[i], m.z_axis[i], m.w_axis[i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2, r2];
    planes.map(|p| {
        let len = p.truncate().length();
        if len > 1e-12 { (p / len).to_array() } else { [0.0, 0.0, 0.0, 1.0] }
    })
}

/// Whether a box (camera-relative) is at least partly inside every plane.
fn aabb_in_frustum(planes: &[[f32; 4]; 6], min: glam::Vec3, max: glam::Vec3) -> bool {
    planes.iter().all(|p| {
        let n = glam::Vec3::new(p[0], p[1], p[2]);
        let far = glam::Vec3::select(n.cmpge(glam::Vec3::ZERO), max, min);
        n.dot(far) + p[3] >= 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const TABLE: &str = r#"{"season_now": "winter", "habitats": [
        {"id": "meadow", "sets": ["", "spine"], "slots": {"0": 1.0}, "blades_per_m2": 3000,
         "height_m": [0.12, 0.45], "width_m": [0.002, 0.008], "lean_deg": [5, 60],
         "shares": {"winter": {"green": 0.58, "dead": 0.38, "seed": 0.04}}},
        {"id": "turf", "sets": [""], "slots": {"12": 1.0, "18": 1.0}, "blades_per_m2": 4000}]}"#;

    #[test]
    fn the_table_reads_and_maps_slots_per_set() {
        let t = HabitatTable::parse(TABLE).unwrap();
        assert_eq!(t.names, ["meadow", "turf"]);
        assert_eq!(t.habitats[0].density, 3000.0);
        assert!((t.habitats[0].lean_max - 60f32.to_radians()).abs() < 1e-6);
        assert_eq!(t.habitats[0].share_green, 0.58);
        assert_eq!(t.sets[""][0], (0, 1.0));
        assert_eq!(t.sets[""][18], (1, 1.0));
        assert_eq!(t.sets["spine"][0], (0, 1.0));
        assert_eq!(t.sets["spine"][12].0, NO_HABITAT, "turf is not in spine");
    }

    #[test]
    fn the_table_names_what_the_blades_replace() {
        let t = HabitatTable::parse(&TABLE.replacen('{', r#"{"replaces": ["gp_sward_", "gp_thatch"], "#, 1)).unwrap();
        assert_eq!(t.replaces, ["gp_sward_", "gp_thatch"]);
        assert!(HabitatTable::parse(TABLE).unwrap().replaces.is_empty());
    }

    #[test]
    fn a_slot_two_habitats_claim_is_refused() {
        let twice = TABLE.replace(r#""slots": {"12": 1.0, "18": 1.0}"#, r#""slots": {"0": 1.0}"#);
        assert!(HabitatTable::parse(&twice).unwrap_err().contains("claimed by meadow and turf"));
    }

    #[test]
    fn a_texel_keeps_its_two_strongest_habitats_summed_over_their_slots() {
        let t = HabitatTable::parse(TABLE).unwrap();
        let map = &t.sets[""];
        let mut texel = [0u8; 32];
        texel[12] = 100;
        texel[18] = 100;
        texel[0] = 55;
        let w = HabitatTable::habitat_word(map, &texel);
        assert_eq!((w & 31, (w >> 5) & 255), (1, 200), "turf's two slots add up");
        assert_eq!(((w >> 13) & 31, (w >> 18) & 255), (0, 55));
        let bare = HabitatTable::habitat_word(map, &{
            let mut x = [0u8; 32];
            x[13] = 255;
            x
        });
        assert_eq!((bare & 31, (bare >> 5) & 255, (bare >> 13) & 31), (31, 0, 31), "a path grows nothing");
    }

    #[test]
    fn a_tile_visits_fewer_candidates_by_the_square_of_its_distance() {
        assert_eq!(tile_candidates(750, 5.0, 0.0), 750);
        assert_eq!(tile_candidates(750, 5.0, 5.0), 750);
        assert_eq!(tile_candidates(750, 5.0, 10.0), 188);
        assert_eq!(tile_candidates(750, 5.0, 300.0), 1);
    }

    #[test]
    fn the_frustum_keeps_what_is_ahead() {
        let proj = crate::view::reversed_z(glam::Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 1000.0));
        let planes = frustum_planes(&proj);
        assert!(aabb_in_frustum(&planes, glam::Vec3::new(-1.0, -1.0, -11.0), glam::Vec3::new(1.0, 1.0, -9.0)));
        assert!(!aabb_in_frustum(&planes, glam::Vec3::new(-1.0, -1.0, 9.0), glam::Vec3::new(1.0, 1.0, 11.0)));
        assert!(!aabb_in_frustum(&planes, glam::Vec3::new(50.0, -1.0, -11.0), glam::Vec3::new(52.0, 1.0, -9.0)));
    }

    #[test]
    fn the_gpu_structs_match_their_wgsl() {
        assert_eq!(std::mem::size_of::<GpuParams>(), 272);
        assert_eq!(std::mem::size_of::<GpuHabitat>(), 160);
        assert_eq!(std::mem::size_of::<GpuTerrain>(), 32);
        assert_eq!(std::mem::size_of::<GpuTile>(), 16);
        assert_eq!(std::mem::size_of::<GpuDraw>(), 16);
        // A blade is one pool vertex.
        assert_eq!(std::mem::size_of::<somnium_asset::Vertex>(), 32);
    }
}
