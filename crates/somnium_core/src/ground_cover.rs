//! Procedural near-camera ground cover (Rimeborn's WS1-13): grass, flowers and the like generated per 16 m cell
//! around where the view looks (WS1-19: an isometric camera stands 60 m behind its focus) from a project foliage kind's `ground_cover` rules and the terrain's splat layer, never stored.
//!
//! A painted sidecar holds at most a million instances per terrain, which over a 16 km² tile is one per 16 m². Ground
//! cover only matters near the camera, so it is made there: each cell's instances come from a hash of the terrain's
//! seed key, the kind and the cell, so they are the same every time the cell comes back. The seed key is the
//! terrain's id, except for a streamed tile, whose renderer slot changes from visit to visit: its key is its place in
//! the world ([`streamed_seed_key`]), so a tile grows the same grass whichever slot it comes back in. A kind gets a candidate per
//! `1 / density` m², kept with the probability of its layer's weight (none below `min_layer_weight`) and only on ground
//! no steeper than `max_slope_deg` (the kind's brush rules). Cells beyond the radius are dropped; a terrain edit
//! regenerates. A project without `ground_cover` kinds generates nothing. With `align_to_ground` each instance is
//! tilted to the slope under it, so a patch neither floats nor sinks on a hillside.
//!
//! First person (Sunscar, 30 Oct 2026: "grass needs to be much much more denser"): a cell is generated once, on
//! the calling thread under a per-frame candidate and time budget, and drawn as one instance batch per terrain, kind and cell
//! whose copies are one instanced draw per part ([`CoverBatches`]). Each copy carries a rank; past [`BAND_NEAR`] a
//! copy at distance d draws while its rank is below `BAND_NEAR / d` (half the copies per doubling), widened to hold
//! the ground they cover (AMD's mesh-shader grass drops blades and widens the rest the same way). Measured per copy,
//! not per cell: per-cell steps striped a meadow along the cell columns. A cell is rebuilt each half-octave of its
//! distance, at most [`SLOTS_SET_PER_FRAME`] copies a frame, nearest first. `SOMNIUM_COVER_DENSITY` scales every
//! kind's density.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use somnium_renderer::instance_batch::{BatchPart, InstanceBatch};
use somnium_renderer::terrain::TerrainData;
use somnium_renderer::terrain::foliage_paint::PaintedFoliage;

use crate::foliage_palette::ProjectEntry;

/// Cell edge, metres.
pub const CELL: f32 = 16.0;
/// Most candidates generated in one frame, over every cell made then (the rest come next frame); at least one cell.
const CANDIDATES_PER_FRAME: f32 = 16_000.0;
/// Most time one terrain's generation takes in a frame, on the calling thread; at least one cell. Worker threads were
/// starved by the engine's own while tiles installed, and the frame waited on them: 20-500 ms for 4-17 cells (5 Nov).
const GENERATE_MS_PER_FRAME: f32 = 2.0;
/// Most copies of parts (slots) set in batches in one frame, nearest cells first (the rest wait a frame): a tile
/// streaming in beside the player once set 100,000+ in one frame (40 MB of slots, a 350 ms frame).
const SLOTS_SET_PER_FRAME: i64 = 48_000;
/// Inside this distance every copy draws; past it each doubling keeps half ([`keep_at`]). 24 m (was 12): the developer,
/// 6 Nov, "even grass is too aggressive ... in immediate view they should appear real"; at 12 m a meadow 30 m off kept
/// 40% of its blades. Copies cost the visibility pass (1.5 ms of a 25 ms frame), not shading.
pub const BAND_NEAR: f32 = 24.0;
/// Metres over which a kind with an [`GroundCoverRules::inner`] radius rises out of the ground past it, overlapping
/// the band before it as that one sinks at its own edge ([`edge_height`]): a cross-fade, not a line.
pub const INNER_FADE: f32 = 12.0;
/// The widest a thinned copy is drawn, against its own width.
pub(crate) const MOST_WIDEN: f32 = 2.5;

/// What ground cover reads of a terrain: its edit revision, size, heights and splat. A renderer terrain is one; the
/// tests' plain ground is another, so cover can be checked without a GPU.
pub(crate) trait Ground {
    /// Bumped by every edit (and every adopted tile): cover made before it is made again.
    fn edit_revision(&self) -> u64;
    /// Metres along X and Z.
    fn world_size(&self) -> [f32; 2];
    /// Metres between neighbouring heights.
    fn cell_size(&self) -> f32;
    /// Height at a terrain-local point, metres.
    fn world_height_at(&self, x: f32, z: f32) -> f32;
    /// `layer`'s splat weight at a terrain-local point, `0..=1`.
    fn layer_weight(&self, x: f32, z: f32, layer: u8) -> f32;
    /// The layers with any weight in the square `size` metres across from (`x0`, `z0`), terrain-local: bit `l` for
    /// layer `l`. Sampled on a metre grid here (as [`generate`]'s presence test); a terrain with its splat at hand reads
    /// the texels instead.
    fn layer_mask(&self, x0: f32, z0: f32, size: f32) -> u32 {
        let [wx, wz] = self.world_size();
        let steps = size.ceil().max(1.0) as i32;
        let mut mask = 0u32;
        for j in 0..=steps {
            for i in 0..=steps {
                let x = (x0 + size * i as f32 / steps as f32).clamp(0.0, wx);
                let z = (z0 + size * j as f32 / steps as f32).clamp(0.0, wz);
                for layer in 0..32u8 {
                    if mask & (1 << layer) == 0 && self.layer_weight(x, z, layer) > 0.0 {
                        mask |= 1 << layer;
                    }
                }
            }
        }
        mask
    }
}

impl Ground for TerrainData {
    fn edit_revision(&self) -> u64 {
        self.edit_revision
    }

    fn world_size(&self) -> [f32; 2] {
        self.desc.world_size()
    }

    fn cell_size(&self) -> f32 {
        self.desc.cell_size
    }

    fn world_height_at(&self, x: f32, z: f32) -> f32 {
        TerrainData::world_height_at(self, x, z)
    }

    fn layer_weight(&self, x: f32, z: f32, layer: u8) -> f32 {
        TerrainData::layer_weight_at(self, x, z, layer)
    }

    fn layer_mask(&self, x0: f32, z0: f32, size: f32) -> u32 {
        // The texels under the square, as layer_weight_at reads them (nearest texel), each layer's bit where it has any
        // weight: exact, and a few thousand byte reads a cell.
        let sm = &self.splatmap;
        let [wx, wz] = self.desc.world_size();
        if wx <= 0.0 || wz <= 0.0 || sm.width == 0 || sm.height == 0 {
            return 0;
        }
        let texel = |v: f32, world: f32, n: u32| (((v / world).clamp(0.0, 1.0) * n as f32) as u32).min(n - 1);
        let (tx0, tx1) = (texel(x0, wx, sm.width), texel(x0 + size, wx, sm.width));
        let (tz0, tz1) = (texel(z0, wz, sm.height), texel(z0 + size, wz, sm.height));
        let mut mask = 0u32;
        for tz in tz0..=tz1 {
            for tx in tx0..=tx1 {
                if let Some(t) = sm.data.get((tz * sm.width + tx) as usize) {
                    for (layer, &w) in t.iter().enumerate().take(32) {
                        if w > 0 {
                            mask |= 1 << layer;
                        }
                    }
                }
            }
        }
        mask
    }
}

/// A streamed tile's seed key: its place in the world (`translation`'s x and z, to the metre), hashed. The tile's
/// renderer slot changes from visit to visit, its place does not.
#[must_use]
pub(crate) fn streamed_seed_key(translation: Vec3) -> u64 {
    let (x, z) = (translation.x.round() as i64 as u64, translation.z.round() as i64 as u64);
    mix(mix(x) ^ z.rotate_left(32))
}

#[derive(Default)]
struct TerrainCover {
    /// Unique per remake of this terrain's cover: its batches are rebuilt when it changes.
    epoch: u64,
    revision: u64,
    /// The seed key the cells were made with.
    seed: u64,
    /// The area cover grows in (`TerrainStreaming::own_area`), if not the whole terrain.
    area: Option<glam::Vec4>,
    /// Where the cover last grew round (terrain-local): its copies thin with their distance from here.
    focus: Vec3,
    cells: HashMap<(u16, i32, i32), Vec<Cover>>,
    /// Every live cell's instances, rebuilt when asked after the set of cells changed (the per-instance path).
    flat: Vec<Cover>,
    flat_stale: bool,
    /// The ground layers each cell holds ([`Ground::layer_mask`]), read once: a kind is planned only where its layer
    /// lies, so a tile of meadow, wood and rock plans a fraction of every kind x cell (6 Nov 2026: with 179-600 kinds
    /// the plan alone took 20-27 ms a frame and the frame made one cell).
    layers: HashMap<(i32, i32), u32>,
    /// The (kind, cell) jobs still to make, nearest first, kept while the view stays in one cell and the kinds are the
    /// same; `plan_at` is how far the frames have worked through it.
    plan: Vec<(f32, usize, i32, i32)>,
    plan_at: usize,
    plan_key: Option<(i32, i32, u64)>,
}

/// One generated instance: where it stands (its yaw and tilt as a painted one's) and its whole rotation.
#[derive(Clone, Copy)]
pub struct Cover {
    /// The instance, terrain-local.
    pub instance: PaintedFoliage,
    /// `Ry(yaw) * Rx(tilt)`, turned onto the ground's normal for `align_to_ground` kinds.
    pub rotation: Quat,
    /// Uniform in `0..1`, hashed: a copy draws in distance band `b` while `rank < 2^-b`.
    pub rank: f32,
}

/// Generated ground cover, per terrain.
#[derive(Default)]
pub struct GroundCover {
    terrains: HashMap<u32, TerrainCover>,
}

impl GroundCover {
    /// Bring `terrain`'s cover up to date round `camera` (terrain-local: where the view looks, on the ground), its
    /// instances hashed from the terrain's id ([`Self::update_seeded`] with the id as the seed key).
    pub(crate) fn update(
        &mut self,
        terrain_id: u32,
        terrain: &impl Ground,
        palette: &std::collections::BTreeMap<u16, ProjectEntry>,
        camera: Vec3,
        area: Option<glam::Vec4>,
    ) {
        self.update_seeded(terrain_id, u64::from(terrain_id), terrain, palette, camera, area, glam::Vec2::ZERO, "");
    }

    /// As [`Self::update`], with the instances hashed from `seed_key` rather than the terrain's id: a streamed tile's
    /// [`streamed_seed_key`], so it grows the same cover in whichever renderer slot it comes back. The cover is still
    /// kept by `terrain_id`, and made again when the key changes.
    pub(crate) fn update_seeded(
        &mut self,
        terrain_id: u32,
        seed_key: u64,
        terrain: &impl Ground,
        palette: &std::collections::BTreeMap<u16, ProjectEntry>,
        camera: Vec3,
        area: Option<glam::Vec4>,
        origin: glam::Vec2,
        layer_set: &str,
    ) {
        let scale = density_scale();
        // Only the kinds this terrain's layer set grows (`GroundCoverRules::layer_sets`; unset grows everywhere).
        let kinds: Vec<(u16, &ProjectEntry, f32, f32, f32)> = palette
            .iter()
            .filter_map(|(kind, entry)| {
                entry
                    .ground_cover
                    .as_ref()
                    .filter(|g| g.grows_on(layer_set))
                    .map(|g| (*kind, entry, g.density * scale, g.radius, g.inner))
            })
            .collect();
        if kinds.is_empty() {
            self.terrains.remove(&terrain_id);
            return;
        }
        let started = std::time::Instant::now();
        let cover = self.terrains.entry(terrain_id).or_default();
        let revision = terrain.edit_revision();
        let reset = cover.revision != revision || cover.area != area || cover.seed != seed_key || cover.epoch == 0;
        let why = (cover.revision != revision, cover.area != area, cover.seed != seed_key);
        if reset {
            static EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            let epoch = EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *cover = TerrainCover { epoch, revision, seed: seed_key, area, ..TerrainCover::default() };
        }
        cover.focus = camera;
        // One pass drops every kind's cells past its radius (a pass per kind was kinds x cells a frame, per terrain).
        let band_of: HashMap<u16, (f32, f32)> =
            kinds.iter().map(|&(kind, _, _, radius, inner)| (kind, (inner, radius))).collect();
        let before = cover.cells.len();
        cover.cells.retain(|&(k, cx, cz), _| {
            band_of.get(&k).is_some_and(|&(inner, r)| {
                let d = cell_distance(cx, cz, camera);
                d <= r + CELL && !inside_inner(d, inner)
            })
        });
        let mut changed = cover.cells.len() != before;
        // Only cells on this terrain's own ground: a tile the player is not on grows nothing round him. Every loaded
        // tile once kept thousands of empty cells round the eye (Sunscar, 30 Oct 2026: 40-100 ms a frame).
        let [wx, wz] = terrain.world_size();
        let own = area.unwrap_or(glam::Vec4::new(0.0, 0.0, wx, wz));
        let (lo_x, lo_z) = ((own.x / CELL).floor() as i32, (own.y / CELL).floor() as i32);
        let (hi_x, hi_z) = (((own.z / CELL).ceil() as i32) - 1, ((own.w / CELL).ceil() as i32) - 1);
        // Every kind's wanted cells where its layer lies, nearest first: planned when the view enters another cell (or the
        // kinds or the cover change), then worked through over the frames.
        let (ox, oz) = ((camera.x / CELL).floor() as i32, (camera.z / CELL).floor() as i32);
        let kinds_key = kinds.iter().fold(kinds.len() as u64, |h, k| mix(h ^ u64::from(k.0)));
        let key = Some((ox, oz, kinds_key));
        let replan = reset || changed || cover.plan_key != key;
        if replan {
            let reach = kinds.iter().map(|k| (k.3 / CELL).ceil() as i32 + 1).max().unwrap_or(0);
            for cz in (oz - reach).max(lo_z)..=(oz + reach).min(hi_z) {
                for cx in (ox - reach).max(lo_x)..=(ox + reach).min(hi_x) {
                    cover.layers.entry((cx, cz)).or_insert_with(|| terrain.layer_mask(cx as f32 * CELL, cz as f32 * CELL, CELL));
                }
            }
            let mut plan: Vec<(f32, usize, i32, i32)> = Vec::new();
            for (index, &(kind, entry, _, radius, inner)) in kinds.iter().enumerate() {
                let bit = 1u32 << u32::from(entry.brush.layer.min(31));
                let reach = (radius / CELL).ceil() as i32 + 1;
                let (cells, layers) = (&cover.cells, &cover.layers);
                plan.extend(
                    ((oz - reach).max(lo_z)..=(oz + reach).min(hi_z))
                        .flat_map(|cz| ((ox - reach).max(lo_x)..=(ox + reach).min(hi_x)).map(move |cx| (cx, cz)))
                        .filter(|cell| layers.get(cell).is_none_or(|m| m & bit != 0))
                        .map(|(cx, cz)| (cell_distance(cx, cz, camera), index, cx, cz))
                        .filter(|&(d, _, cx, cz)| {
                            d <= radius && !inside_inner(d, inner) && !cells.contains_key(&(kind, cx, cz))
                        }),
                );
            }
            plan.sort_by(|a, b| a.0.total_cmp(&b.0));
            cover.plan = plan;
            cover.plan_at = 0;
            cover.plan_key = key;
        }
        let mut budget = CANDIDATES_PER_FRAME;
        let mut take = 0;
        for &(_, index, _, _) in &cover.plan[cover.plan_at..] {
            if take > 0 && budget <= 0.0 {
                break;
            }
            budget -= kinds[index].2 * CELL * CELL;
            take += 1;
        }
        let wanted_len = cover.plan.len() - cover.plan_at;
        let planned = started.elapsed();
        // Nearest first, on this thread, until the frame's generation time is spent (the rest come next frame). The
        // clock starts after planning: counted from before it, a slow plan left time for one cell a frame.
        let generating = std::time::Instant::now();
        let mut made: Vec<((u16, i32, i32), Vec<Cover>)> = Vec::new();
        let mut done = 0;
        for &(_, i, cx, cz) in &cover.plan[cover.plan_at..cover.plan_at + take] {
            if !made.is_empty() && generating.elapsed().as_secs_f32() * 1000.0 > GENERATE_MS_PER_FRAME {
                break;
            }
            done += 1;
            let (kind, entry, density, _, _) = kinds[i];
            if cover.cells.contains_key(&(kind, cx, cz)) {
                continue;
            }
            made.push(((kind, cx, cz), generate(seed_key, terrain, kind, entry, density, cx, cz, area, origin)));
        }
        cover.plan_at += done;
        changed |= !made.is_empty();
        let copies: usize = made.iter().map(|(_, c)| c.len()).sum();
        let made_cells = made.len();
        cover.cells.extend(made);
        if changed {
            cover.flat_stale = true;
        }
        // A slow frame says where it went (the tile-arrival stalls, 31 Oct - 5 Nov).
        let total = started.elapsed();
        if total.as_secs_f32() > 0.02 {
            tracing::warn!(
                terrain_id,
                reset,
                revision_changed = why.0,
                area_changed = why.1,
                seed_changed = why.2,
                kinds = kinds.len(),
                wanted = wanted_len,
                replan,
                jobs = take,
                made_cells,
                copies,
                cells = cover.cells.len(),
                plan_ms = planned.as_secs_f32() * 1000.0,
                total_ms = total.as_secs_f32() * 1000.0,
                "slow cover generation"
            );
        }
    }

    /// `terrain`'s generated instances (terrain-local), as painted ones are.
    #[must_use]
    pub fn instances(&mut self, terrain_id: u32) -> &[Cover] {
        let Some(cover) = self.terrains.get_mut(&terrain_id) else { return &[] };
        if cover.flat_stale {
            cover.flat.clear();
            for cell in cover.cells.values() {
                cover.flat.extend_from_slice(cell);
            }
            cover.flat_stale = false;
        }
        cover.flat.as_slice()
    }

    /// `terrain_id`'s cells, each with its kind and place: what the editor's click looks through (`app_editing.rs`).
    pub(crate) fn cells(&self, terrain_id: u32) -> impl Iterator<Item = ((u16, i32, i32), &[Cover])> {
        self.terrains.get(&terrain_id).into_iter().flat_map(|t| t.cells.iter().map(|(&key, cell)| (key, cell.as_slice())))
    }

    /// Where `terrain_id`'s cover last grew round (terrain-local): its copies thin with their distance from here.
    #[must_use]
    pub(crate) fn focus(&self, terrain_id: u32) -> Option<Vec3> {
        self.terrains.get(&terrain_id).map(|t| t.focus)
    }

    /// Forget a terrain (or every terrain, on a scene change).
    pub fn clear(&mut self) {
        self.terrains.clear();
    }
}

/// Horizontal distance from `camera` to the nearest point of a cell.
fn cell_distance(cx: i32, cz: i32, camera: Vec3) -> f32 {
    let (x0, z0) = (cx as f32 * CELL, cz as f32 * CELL);
    let dx = (x0 - camera.x).max(camera.x - (x0 + CELL)).max(0.0);
    let dz = (z0 - camera.z).max(camera.z - (z0 + CELL)).max(0.0);
    dx.hypot(dz)
}

/// A small counter hash (splitmix64), so a cell's instances never depend on what was generated before it.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// A cell's seed, from the terrain's seed key, the kind and the cell. A key below 2^24 (every renderer terrain id)
/// with a kind under 256 hashes exactly as the id and byte kind always did. A wider kind's high bits would overlap
/// the key's at bit 40 (key t with kind k + 256 would grow key t ^ 1's kind k), so it is mixed in once more; and the
/// bits of a larger key (a streamed tile's place) that the shift drops are mixed in after, so no two places share a
/// seed for want of them.
fn cell_seed(seed_key: u64, kind: u16, cx: i32, cz: i32) -> u64 {
    let seed = mix(seed_key << 40 ^ u64::from(kind) << 32 ^ (cx as u32 as u64) << 16 ^ (cz as u32 as u64));
    let seed = if kind > 255 { mix(seed ^ u64::from(kind)) } else { seed };
    match seed_key >> 24 {
        0 => seed,
        high => mix(seed ^ high),
    }
}

#[allow(clippy::too_many_arguments)]
fn generate(
    seed_key: u64,
    terrain: &impl Ground,
    kind: u16,
    entry: &ProjectEntry,
    density: f32,
    cx: i32,
    cz: i32,
    area: Option<glam::Vec4>,
    origin: glam::Vec2,
) -> Vec<Cover> {
    let b = &entry.brush;
    let [wx, wz] = terrain.world_size();
    let seed = cell_seed(seed_key, kind, cx, cz);
    let expected = density * CELL * CELL;
    let count = expected.floor() as u64 + u64::from(unit(mix(seed)) < expected.fract());
    let min_slope_cos = b.max_slope_deg.to_radians().cos();
    let least = b.min_layer_weight.max(1e-3);
    let clump = entry.ground_cover.as_ref().and_then(|g| g.clump);
    let mut out = Vec::new();
    // A cell with none of the kind's layer on a metre grid grows nothing: skip its candidates.
    let present = (0..=16).any(|j| {
        (0..=16).any(|i| {
            let (x, z) = ((cx as f32 + i as f32 / 16.0) * CELL, (cz as f32 + j as f32 / 16.0) * CELL);
            terrain.layer_weight(x.clamp(0.0, wx), z.clamp(0.0, wz), b.layer) >= least
        })
    });
    if !present {
        return out;
    }
    for i in 0..count {
        let h = mix(seed ^ (i + 1).wrapping_mul(0xA24B_AED4_963E_E407));
        let (x, z) = ((cx as f32 + unit(h)) * CELL, (cz as f32 + unit(mix(h))) * CELL);
        if !(0.0..wx).contains(&x) || !(0.0..wz).contains(&z) || !area.is_none_or(|a| inside(a, x, z)) {
            continue;
        }
        // The cheap test first: most candidates fall on another layer.
        let weight = terrain.layer_weight(x, z, b.layer);
        if weight < least || unit(mix(h ^ 1)) > weight {
            continue;
        }
        // Drifts, from the world's position so they carry across tile seams.
        let patch = clump.map(|m| patch_noise((origin.x + x) / m, (origin.y + z) / m, u64::from(kind)));
        if patch.is_some_and(|n| unit(mix(h ^ 6)) > drift(n)) {
            continue;
        }
        let normal = ground_normal(terrain, x, z);
        if normal.y < min_slope_cos {
            continue;
        }
        // Taller inside a drift, shorter at its thin edge: tone that changes over metres, not per tuft.
        let t = patch.map_or(unit(mix(h ^ 2)), |n| 0.4 * unit(mix(h ^ 2)) + 0.6 * n);
        let instance = PaintedFoliage {
            kind,
            position: Vec3::new(x, terrain.world_height_at(x, z), z),
            yaw: unit(mix(h ^ 3)) * std::f32::consts::TAU,
            tilt: b.max_tilt_deg.to_radians() * unit(mix(h ^ 4)),
            scale: b.scale_min + (b.scale_max - b.scale_min) * t,
        };
        let mut rotation = Quat::from_rotation_y(instance.yaw) * Quat::from_rotation_x(instance.tilt);
        if entry.ground_cover.as_ref().is_some_and(|g| g.align_to_ground) {
            rotation = Quat::from_rotation_arc(Vec3::Y, normal) * rotation;
        }
        out.push(Cover { instance, rotation, rank: unit(mix(h ^ 5)) });
    }
    out
}

/// How many copies of `kind` a square metre at a terrain-local point grows on average, by its rules: its density
/// (times `SOMNIUM_COVER_DENSITY`) kept with its layer's weight there and inside its drifts, on ground no steeper than
/// its slope. Before distance thinning. `origin` is a streamed tile's world XZ (its drifts follow the world) and zero
/// for any other terrain, as [`GroundCover::update_seeded`] has it.
#[must_use]
pub(crate) fn density_at(terrain: &impl Ground, kind: u16, entry: &ProjectEntry, x: f32, z: f32, origin: glam::Vec2) -> f32 {
    let Some(rules) = entry.ground_cover.as_ref() else { return 0.0 };
    let b = &entry.brush;
    let weight = terrain.layer_weight(x, z, b.layer);
    if weight < b.min_layer_weight.max(1e-3) || ground_normal(terrain, x, z).y < b.max_slope_deg.to_radians().cos() {
        return 0.0;
    }
    let drifts = rules.clump.map_or(1.0, |m| drift(patch_noise((origin.x + x) / m, (origin.y + z) / m, u64::from(kind))));
    rules.density * density_scale() * weight * drifts
}

/// Smooth value noise in `0..1` at a point measured in drift widths, two octaves, its own field per `seed` (kind).
fn patch_noise(x: f32, z: f32, seed: u64) -> f32 {
    let octave = |x: f32, z: f32, seed: u64| {
        let (ix, iz) = (x.floor(), z.floor());
        let (fx, fz) = (x - ix, z - iz);
        let corner = |dx: i64, dz: i64| {
            unit(mix(seed.wrapping_mul(0x9E37_79B9) ^ ((ix as i64 + dx) as u64) << 32 ^ ((iz as i64 + dz) as u64 & 0xFFFF_FFFF)))
        };
        let (sx, sz) = (fx * fx * (3.0 - 2.0 * fx), fz * fz * (3.0 - 2.0 * fz));
        let top = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * sx;
        let bottom = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * sx;
        top + (bottom - top) * sz
    };
    0.65 * octave(x, z, seed) + 0.35 * octave(x * 2.3 + 17.1, z * 2.3 - 5.3, seed ^ 0x5bd1_e995)
}

/// The keep chance inside a drift at noise `n`: none in the gaps, all at the heart, a soft edge between.
fn drift(n: f32) -> f32 {
    let t = ((n - 0.32) / 0.4).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Whether a terrain-local point is in an own area (x0, z0, x1, z1): a window owns its lower edges, not its upper.
fn inside(area: glam::Vec4, x: f32, z: f32) -> bool {
    (area.x..area.z).contains(&x) && (area.y..area.w).contains(&z)
}

/// The ground's normal at a terrain-local point, from the heights a cell either side.
fn ground_normal(terrain: &impl Ground, x: f32, z: f32) -> Vec3 {
    let d = terrain.cell_size();
    let hx = terrain.world_height_at(x + d, z) - terrain.world_height_at(x - d, z);
    let hz = terrain.world_height_at(x, z + d) - terrain.world_height_at(x, z - d);
    Vec3::new(-hx, 2.0 * d, -hz).try_normalize().unwrap_or(Vec3::Y)
}

/// Every ground cover kind's density times `SOMNIUM_COVER_DENSITY` (1 by default): a whole-world dial for density
/// sweeps, while the palette's densities are the look's.
#[must_use]
pub fn density_scale() -> f32 {
    static SCALE: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        std::env::var("SOMNIUM_COVER_DENSITY")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v > 0.0)
            .map_or(1.0, |v| v.min(64.0))
    });
    *SCALE
}

/// A cell's distance step, `distance` metres from the cover's centre: 0 inside [`BAND_NEAR`], then one more per half
/// doubling. A cell's batch is rebuilt (its copies thinned again) when its step changes.
#[must_use]
pub fn band(distance: f32) -> u32 {
    if distance <= BAND_NEAR { 0 } else { (((distance / BAND_NEAR).log2() * 2.0).ceil() as u32).min(16) }
}

/// The share of copies drawn `distance` metres from the cover's centre: all inside [`BAND_NEAR`], then half per
/// doubling.
fn keep_at(distance: f32) -> f32 {
    if distance <= BAND_NEAR { 1.0 } else { BAND_NEAR / distance }
}

/// Whether a whole cell, `d` metres to its nearest point, lies inside a kind's `inner` radius (a cell's far corner is
/// at most about 1.5 cells beyond its nearest point).
fn inside_inner(d: f32, inner: f32) -> bool {
    inner > 0.0 && d + 1.5 * CELL < inner
}

/// A copy's height share `distance` metres out past a kind's `inner` radius: nothing inside it, rising smoothly over
/// [`INNER_FADE`], whole beyond. A kind without one is whole from the eye.
fn inner_height(distance: f32, inner: f32) -> f32 {
    if inner <= 0.0 {
        return 1.0;
    }
    let t = ((distance - inner) / INNER_FADE).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A copy's height share `distance` metres out of a kind's `radius`: whole until the last fifth, then sinking smoothly to
/// nothing at the radius, so the cover ends in the ground rather than in a line (the lead, 31 Oct: "a hard band").
fn edge_height(distance: f32, radius: f32) -> f32 {
    let t = ((radius - distance) / (0.2 * radius)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// How much wider a copy draws where only `keep` of them do, to hold the ground they cover (at most [`MOST_WIDEN`]).
fn widen(keep: f32) -> f32 {
    keep.sqrt().recip().min(MOST_WIDEN)
}

/// How a copy draws round `focus` (terrain-local) between its kind's `inner` radius and its `radius`: its scale on each
/// axis (a thinned copy widened, one near either edge sunk into the ground), or `None` while it is thinned out. The one
/// rule for [`CoverBatches`] and the editor's click, so a click finds what is drawn.
#[must_use]
pub(crate) fn drawn_scale(cover: &Cover, focus: Vec3, radius: f32, inner: f32) -> Option<Vec3> {
    let at = cover.instance.position;
    let distance = (at.x - focus.x).hypot(at.z - focus.z);
    // A far band thins from where it starts, not from the eye's near band.
    let keep = keep_at(distance - inner.max(0.0));
    let sink = edge_height(distance, radius) * inner_height(distance, inner);
    if cover.rank >= keep || cover.instance.scale <= 0.0 || sink <= 0.05 {
        return None;
    }
    let (s, w) = (cover.instance.scale, widen(keep));
    Some(Vec3::new(s * w, s * sink, s * w))
}

/// Ground cover as instance batches: one per terrain, kind and cell, every part's copies one instanced draw
/// (`SomniumRenderer::set_instanced_batch`). A cell's batch is set when it comes within its kind's radius, set again
/// when its band changes or its terrain's cover is remade, and removed when it leaves.
#[derive(Default)]
pub(crate) struct CoverBatches {
    /// Batch id -> (signature, frame last needed, copies).
    live: HashMap<u64, (u64, u64, u32)>,
    frame: u64,
    /// Slots this frame may still set.
    budget: i64,
    pub set: Vec<(u64, InstanceBatch)>,
    pub remove: Vec<u64>,
}

/// Namespace byte 0xF1, then 12 bits of terrain, 13 of kind (every project id) and 15 each of the cell's x and z.
fn cover_batch_id(terrain: u32, kind: u16, cx: i32, cz: i32) -> u64 {
    (0xF1u64 << 56)
        | (u64::from(terrain & 0xFFF) << 44)
        | (u64::from(kind & 0x1FFF) << 31)
        | ((cx as u32 as u64 & 0x7FFF) << 15)
        | (cz as u32 as u64 & 0x7FFF)
}

fn fold(h: &mut u64, word: u32) {
    *h = (*h ^ u64::from(word)).wrapping_mul(0x0100_0000_01b3);
}

impl CoverBatches {
    pub(crate) fn begin_frame(&mut self) {
        self.frame += 1;
        self.budget = SLOTS_SET_PER_FRAME;
        self.set.clear();
        self.remove.clear();
    }

    /// One terrain's cells, round `focus` (terrain-local). `parts[kind]` is `None` while a kind's mesh is not loaded.
    /// Returns the copies its live batches draw.
    pub(crate) fn terrain(
        &mut self,
        terrain_id: u32,
        cover: &GroundCover,
        palette: &std::collections::BTreeMap<u16, ProjectEntry>,
        parts: &[Option<Vec<BatchPart>>],
        model: Mat4,
        focus: Vec3,
    ) -> u32 {
        let Some(t) = cover.terrains.get(&terrain_id) else { return 0 };
        let mut cells: Vec<(f32, (u16, i32, i32), &Vec<Cover>)> = t
            .cells
            .iter()
            .filter(|(_, cell)| !cell.is_empty())
            .map(|(&key, cell)| (cell_distance(key.1, key.2, focus), key, cell))
            .collect();
        cells.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut copies = 0;
        for (d, (kind, cx, cz), cell) in cells {
            let Some(Some(list)) = parts.get(usize::from(kind)) else { continue };
            let Some((radius, inner)) =
                palette.get(&kind).and_then(|e| e.ground_cover.as_ref()).map(|g| (g.radius, g.inner))
            else {
                continue;
            };
            if d > radius || inside_inner(d, inner) {
                continue;
            }
            // A cell past its kind's shadow distance casts nothing: a casting batch set or removed redraws every
            // cascade, and the far cells come and go all the time.
            let casts = list.iter().any(|p| p.casts_shadow && (p.shadow_far <= 0.0 || d <= p.shadow_far));
            let mut sig = 0xcbf2_9ce4_8422_2325u64 ^ t.epoch.rotate_left(17);
            fold(&mut sig, band(d));
            fold(&mut sig, u32::from(casts));
            for v in model.to_cols_array() {
                fold(&mut sig, v.to_bits());
            }
            for p in list {
                for word in [p.vertex_offset, p.index_offset, p.index_count, p.material_id, u32::from(p.casts_shadow)] {
                    fold(&mut sig, word);
                }
                fold(&mut sig, p.shadow_far.to_bits());
            }
            let id = cover_batch_id(terrain_id, kind, cx, cz);
            let live = self.live.get(&id).copied();
            let n = match live {
                Some((old, _, n)) if old == sig => n,
                // Over this frame's budget: an old batch draws on until next frame, a new one waits for it.
                Some((old, _, n)) if self.budget <= 0 => {
                    self.live.insert(id, (old, self.frame, n));
                    copies += n;
                    continue;
                }
                None if self.budget <= 0 => continue,
                _ => {
                    let placements: Vec<Mat4> = cell
                        .iter()
                        .filter_map(|c| {
                            drawn_scale(c, focus, radius, inner).map(|scale| {
                                model * Mat4::from_scale_rotation_translation(scale, c.rotation, c.instance.position)
                            })
                        })
                        .collect();
                    let n = placements.len() as u32;
                    if n == 0 {
                        continue;
                    }
                    self.budget -= i64::from(n) * list.len() as i64;
                    let mut parts = list.clone();
                    if !casts {
                        parts.iter_mut().for_each(|p| p.casts_shadow = false);
                    }
                    self.set.push((id, InstanceBatch { parts, placements, fade: 0.0 }));
                    n
                }
            };
            self.live.insert(id, (sig, self.frame, n));
            copies += n;
        }
        copies
    }

    /// Every batch not needed this frame is removed.
    pub(crate) fn end_frame(&mut self) {
        let (frame, remove) = (self.frame, &mut self.remove);
        self.live.retain(|&id, &mut (_, last, _)| {
            if last != frame {
                remove.push(id);
            }
            last == frame
        });
    }
}

/// How far along the view ray the cover's centre may move ahead of the camera, metres: `SOMNIUM_COVER_FOCUS_REACH`,
/// 400 by default (an isometric camera grows its cover where it looks on the ground). A first-person game sets a few
/// metres (Sunscar, 30 Oct 2026: 4), so the ring round the feet stays full when the eye looks down.
#[must_use]
pub fn focus_reach() -> f32 {
    static REACH: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        std::env::var("SOMNIUM_COVER_FOCUS_REACH")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && *v >= 0.0)
            .unwrap_or(400.0)
    });
    *REACH
}

/// Where the view looks, on the ground: the view's forward ray (from `camera`, terrain-local) met with the ground
/// height under the camera, at most `reach` metres out. A view looking level or up keeps the camera's own place.
#[must_use]
pub fn view_focus(camera: Vec3, forward: Vec3, ground: f32, reach: f32) -> Vec3 {
    if forward.y > -0.05 || camera.y <= ground {
        return camera;
    }
    let t = ((camera.y - ground) / -forward.y).min(reach);
    camera + forward * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_are_measured_to_their_nearest_edge() {
        let camera = Vec3::new(8.0, 0.0, 8.0);
        assert_eq!(cell_distance(0, 0, camera), 0.0);
        assert!((cell_distance(2, 0, camera) - 24.0).abs() < 1e-4);
        assert!((cell_distance(-1, -1, camera) - 8.0f32.hypot(8.0)).abs() < 1e-4);
    }

    #[test]
    fn an_isometric_view_grows_cover_where_it_looks() {
        // 90 m from the focus, 45 degrees down: the focus is ~64 m ahead of the camera along the ground.
        let forward = Vec3::new(0.0, -1.0, -1.0).normalize();
        let camera = Vec3::new(0.0, 10.0 + 63.64, 0.0);
        let focus = view_focus(camera, forward, 10.0, 400.0);
        assert!((focus - Vec3::new(0.0, 10.0, -63.64)).length() < 0.05, "{focus}");
        assert_eq!(view_focus(camera, Vec3::NEG_Z, 10.0, 400.0), camera, "a level view keeps the camera's place");
        assert!((view_focus(camera, forward, 10.0, 20.0) - camera).length() <= 20.0 + 1e-3, "never past reach");
    }

    #[test]
    fn neighbouring_windows_never_both_own_a_point() {
        // Two 4,096 m windows on a 4,000 m grid, each owning up to the middle of the overlap (WS1-20).
        let west = glam::Vec4::new(0.0, 0.0, 4048.0, 4096.0);
        let east = glam::Vec4::new(48.0, 0.0, 4096.0, 4096.0);
        for x in [3990.0, 4047.9, 4048.0, 4050.0] {
            assert!(inside(west, x + 0.0, 10.0) != inside(east, x - 4000.0, 10.0), "{x}");
        }
    }

    /// Flat ground 2 m up, 128 m square, its splat all layer 0.
    struct Plain {
        revision: u64,
    }

    impl Ground for Plain {
        fn edit_revision(&self) -> u64 {
            self.revision
        }

        fn world_size(&self) -> [f32; 2] {
            [128.0, 128.0]
        }

        fn cell_size(&self) -> f32 {
            2.0
        }

        fn world_height_at(&self, _x: f32, _z: f32) -> f32 {
            2.0
        }

        fn layer_weight(&self, _x: f32, _z: f32, layer: u8) -> f32 {
            if layer == 0 { 1.0 } else { 0.0 }
        }
    }

    /// Flat ground, layer 0 in its west half and layer 3 in its east half.
    struct Halves;

    impl Ground for Halves {
        fn edit_revision(&self) -> u64 {
            1
        }

        fn world_size(&self) -> [f32; 2] {
            [128.0, 128.0]
        }

        fn cell_size(&self) -> f32 {
            2.0
        }

        fn world_height_at(&self, _x: f32, _z: f32) -> f32 {
            0.0
        }

        fn layer_weight(&self, x: f32, _z: f32, layer: u8) -> f32 {
            f32::from(u8::from((layer == 0 && x < 64.0) || (layer == 3 && x >= 64.0)))
        }
    }

    #[test]
    fn a_cells_layer_mask_holds_the_layers_under_it() {
        assert_eq!(Halves.layer_mask(0.0, 0.0, CELL), 1, "west: layer 0 only");
        assert_eq!(Halves.layer_mask(96.0, 0.0, CELL), 1 << 3, "east: layer 3 only");
        assert_eq!(Halves.layer_mask(32.0, 0.0, CELL), 1, "west of the seam");
        assert_eq!(Halves.layer_mask(48.0, 0.0, CELL), 1 | 1 << 3, "its far edge on the seam, as generate's presence test");
        assert_eq!(Halves.layer_mask(56.0, 0.0, CELL), 1 | 1 << 3, "across the seam");
    }

    #[test]
    fn a_kind_is_planned_and_grown_only_where_its_layer_lies() {
        let mut palette = grass();
        let mut east: ProjectEntry = serde_json::from_value(serde_json::json!({
            "kind": 133, "name": "east", "source": "g.gltf", "primitives": [0],
            "brush": {"layer": 3},
            "ground_cover": {"density": 0.2, "radius": 40.0}}))
        .unwrap();
        east.kind = 133;
        palette.insert(133, east);
        let mut cover = GroundCover::default();
        let camera = Vec3::new(64.0, 0.0, 64.0);
        for _ in 0..16 {
            cover.update_seeded(7, 7, &Halves, &palette, camera, None, glam::Vec2::ZERO, "");
        }
        let t = &cover.terrains[&7];
        assert!(t.cells.keys().all(|&(k, cx, _)| if k == 132 { cx < 4 } else { cx >= 3 }), "each kind only on its own half");
        let made = |kind: u16| t.cells.keys().filter(|&&(k, _, _)| k == kind).count();
        assert!(made(132) > 0 && made(133) > 0);
        assert!(cover.instances(7).iter().all(|c| (c.instance.kind == 132) == (c.instance.position.x < 64.0)));
    }

    #[test]
    fn the_plan_is_kept_while_the_view_stays_in_its_cell() {
        let palette = grass();
        let mut cover = GroundCover::default();
        let ground = Plain { revision: 1 };
        cover.update_seeded(9, 9, &ground, &palette, Vec3::new(64.0, 2.0, 64.0), None, glam::Vec2::ZERO, "");
        let first = cover.terrains[&9].plan_key;
        cover.update_seeded(9, 9, &ground, &palette, Vec3::new(66.0, 2.0, 70.0), None, glam::Vec2::ZERO, "");
        assert_eq!(cover.terrains[&9].plan_key, first, "same cell: same plan");
        cover.update_seeded(9, 9, &ground, &palette, Vec3::new(90.0, 2.0, 64.0), None, glam::Vec2::ZERO, "");
        assert_ne!(cover.terrains[&9].plan_key, first, "another cell: planned again");
    }

    fn grass() -> std::collections::BTreeMap<u16, ProjectEntry> {
        let entry: ProjectEntry = serde_json::from_value(serde_json::json!({
            "kind": 132, "name": "grass", "source": "g.gltf", "primitives": [0],
            "brush": {"max_tilt_deg": 10.0},
            "ground_cover": {"density": 0.2, "radius": 40.0, "align_to_ground": true}}))
        .unwrap();
        std::collections::BTreeMap::from([(132, entry)])
    }

    /// Terrain `id`'s cover once every cell within reach is made, in a fixed order (cells are kept in a hash map).
    fn grown(cover: &mut GroundCover, id: u32, seed_key: Option<u64>, ground: &Plain) -> Vec<(PaintedFoliage, Quat)> {
        let palette = grass();
        for _ in 0..8 {
            let camera = Vec3::new(64.0, 2.0, 64.0);
            match seed_key {
                Some(key) => cover.update_seeded(id, key, ground, &palette, camera, None, glam::Vec2::ZERO, ""),
                None => cover.update(id, ground, &palette, camera, None),
            }
        }
        let mut out: Vec<(PaintedFoliage, Quat)> = cover.instances(id).iter().map(|c| (c.instance, c.rotation)).collect();
        out.sort_by(|a, b| {
            let key = |f: &PaintedFoliage| (f.position.x.to_bits(), f.position.z.to_bits(), f.yaw.to_bits());
            key(&a.0).cmp(&key(&b.0))
        });
        out
    }

    #[test]
    fn a_terrain_seeded_with_its_own_id_grows_the_cover_it_always_did() {
        // The hash a scene's terrain has always had (a renderer id is far below 2^24), written out as it was.
        for id in [0u32, 1, 3, 15, 4095] {
            for (kind, cx, cz) in [(132u8, 0, 0), (200, -3, 7), (255, 1000, -1000)] {
                let old = mix(u64::from(id) << 40 ^ u64::from(kind) << 32 ^ (cx as u32 as u64) << 16 ^ (cz as u32 as u64));
                assert_eq!(cell_seed(u64::from(id), u16::from(kind), cx, cz), old, "terrain {id}, cell {cx} {cz}");
            }
        }
        let ground = Plain { revision: 4 };
        let by_id = grown(&mut GroundCover::default(), 3, None, &ground);
        assert!(by_id.iter().all(|(f, _)| (f.position.y - 2.0).abs() < 1e-6), "on the ground");
        let seeded = grown(&mut GroundCover::default(), 3, Some(3), &ground);
        assert!(by_id.len() > 100, "{} instances", by_id.len());
        assert_eq!(by_id, seeded, "update is update_seeded with the id as the key");
    }

    #[test]
    fn a_tile_grows_the_same_cover_in_whichever_slot_it_comes_back() {
        let ground = Plain { revision: 1 };
        let place = streamed_seed_key(Vec3::new(1400.3, 5.0, 2100.0));
        assert_eq!(place, streamed_seed_key(Vec3::new(1399.8, -40.0, 2099.6)), "to the metre; height is no part of it");
        assert_ne!(place, streamed_seed_key(Vec3::new(2400.0, 5.0, 2100.0)));
        assert_ne!(place, streamed_seed_key(Vec3::new(2100.0, 5.0, 1400.0)), "x and z are not interchangeable");
        let mut cover = GroundCover::default();
        let in_slot_1 = grown(&mut cover, 1, Some(place), &ground);
        let in_slot_7 = grown(&mut cover, 7, Some(place), &ground);
        assert!(!in_slot_1.is_empty());
        assert_eq!(in_slot_1, in_slot_7, "the same key in two slots");
        assert_ne!(grown(&mut cover, 2, None, &ground), grown(&mut cover, 5, None, &ground), "by id, slots differ");
        // Slot 1 reused by another tile (same revision, say): its cover is made again from the new key.
        let elsewhere = streamed_seed_key(Vec3::new(5400.0, 0.0, 2100.0));
        let reused = grown(&mut cover, 1, Some(elsewhere), &ground);
        assert_eq!(reused, grown(&mut GroundCover::default(), 9, Some(elsewhere), &ground));
        assert_ne!(reused, in_slot_1);
    }

    #[test]
    fn the_engine_seeds_a_streamed_tile_by_its_place_and_any_other_terrain_by_its_id() {
        // The engine's side, read from its source: submitting foliage needs a window and a GPU.
        let app = include_str!("app.rs");
        let submit = app.split_once("fn submit_foliage(").expect("submit_foliage exists").1;
        let submit = &submit[..submit.find("\n    }\n").unwrap_or(submit.len())];
        let seed = submit.split_once("let seed_key").expect("a seed key per terrain").1;
        let seed = &seed[..seed.find(';').unwrap_or(seed.len())];
        assert!(seed.contains("Streamed>") && seed.contains("streamed_seed_key("), "a tile's slot changes per visit");
        assert!(submit.contains("update_seeded(") && submit.contains(".update(terrain_id"), "the rest as always");
    }

    #[test]
    fn the_hash_spreads_and_repeats() {
        let a: Vec<f32> = (0..1000).map(|i| unit(mix(i))).collect();
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "{mean}");
        assert!(a.iter().all(|v| (0.0..1.0).contains(v)));
        assert_eq!(unit(mix(42)), unit(mix(42)));
    }

    #[test]
    fn byte_kinds_seed_as_before_and_wide_kinds_their_own_cells() {
        // The seed byte kinds had, so every existing cover kind grows the same instances.
        let old = |t: u32, k: u8, cx: i32, cz: i32| {
            mix(u64::from(t) << 40 ^ u64::from(k) << 32 ^ (cx as u32 as u64) << 16 ^ (cz as u32 as u64))
        };
        for (t, k, cx, cz) in [(0, 25, 0, 0), (3, 128, -4, 9), (17, 255, 250, -1)] {
            assert_eq!(cell_seed(u64::from(t), u16::from(k), cx, cz), old(t, k, cx, cz));
        }
        // A wide kind is not its low byte, nor a neighbouring terrain's kind.
        assert_ne!(cell_seed(4, 300, 1, 2), cell_seed(4, 44, 1, 2));
        assert_ne!(cell_seed(4, 44 + 256, 1, 2), cell_seed(5, 44, 1, 2));
        assert_ne!(cell_seed(4, 300, 1, 2), cell_seed(4, 301, 1, 2));
        // A terrain id's wide kind seeds as it did before tiles had places.
        assert_eq!(cell_seed(4, 300, 1, 2), mix(mix(4 << 40 ^ 300 << 32 ^ 1 << 16 ^ 2) ^ 300));
        // A tile's place (a key past 2^24) keeps the wide-kind mix too: without it, the kind's high byte and the key's
        // low bit meet at bit 40, and the high bits mixed in after are the same for both.
        let place = streamed_seed_key(Vec3::new(1400.0, 0.0, 2100.0));
        assert_ne!(place >> 24, 0);
        assert_ne!(cell_seed(place, 44 + 256, 1, 2), cell_seed(place ^ 1, 44, 1, 2));
        assert_ne!(cell_seed(place, 300, 1, 2), cell_seed(place, 44, 1, 2));
    }

    #[test]
    fn copies_halve_per_doubling_and_cells_rebuild_each_half_doubling() {
        assert_eq!([band(0.0), band(BAND_NEAR), band(BAND_NEAR + 0.1), band(2.0 * BAND_NEAR), band(4.0 * BAND_NEAR)], [0, 0, 1, 2, 4]);
        assert_eq!(band(1.0e6), 16, "capped");
        assert_eq!((keep_at(1.0), keep_at(BAND_NEAR), keep_at(4.0 * BAND_NEAR)), (1.0, 1.0, 0.25));
        assert!((widen(0.25) - 2.0).abs() < 1e-6 && widen(0.01) == MOST_WIDEN && widen(1.0) == 1.0);
        assert_eq!((edge_height(10.0, 40.0), edge_height(40.0, 40.0), edge_height(50.0, 40.0)), (1.0, 0.0, 0.0));
        assert!((edge_height(36.0, 40.0) - 0.5).abs() < 1e-6, "half height midway through the last fifth");
    }

    #[test]
    fn a_far_band_starts_at_its_inner_radius_and_rises_out_of_the_ground() {
        let ground = Plain { revision: 1 };
        let entry: ProjectEntry = serde_json::from_value(serde_json::json!({
            "kind": 132, "name": "grass far", "source": "g.gltf", "primitives": [0],
            "brush": {"max_tilt_deg": 10.0},
            "ground_cover": {"density": 0.5, "radius": 90.0, "inner": 30.0, "align_to_ground": true}}))
        .unwrap();
        let palette = std::collections::BTreeMap::from([(132, entry)]);
        let eye = Vec3::new(64.0, 2.0, 64.0);
        let mut cover = GroundCover::default();
        for _ in 0..8 {
            cover.update(1, &ground, &palette, eye, None);
        }
        let part = BatchPart {
            vertex_offset: 1,
            index_offset: 0,
            index_count: 30,
            material_id: 0,
            local: Mat4::IDENTITY,
            near: 0.0,
            far: 0.0,
            casts_shadow: false,
            shadow_far: 0.0,
        };
        let mut parts = vec![None; 133];
        parts[132] = Some(vec![part]);
        let mut batches = CoverBatches::default();
        batches.begin_frame();
        let copies = batches.terrain(1, &cover, &palette, &parts, Mat4::IDENTITY, eye);
        batches.end_frame();
        assert!(copies > 0, "the far band draws beyond its inner radius");
        let nearest = batches
            .set
            .iter()
            .flat_map(|(_, b)| b.placements.iter())
            .map(|m| (m.w_axis.x - eye.x).hypot(m.w_axis.z - eye.z))
            .fold(f32::MAX, f32::min);
        assert!(nearest >= 30.0, "nothing inside the inner radius: nearest copy {nearest} m");
        assert!(inner_height(30.0, 30.0) == 0.0 && inner_height(30.0 + INNER_FADE, 30.0) == 1.0);
        assert!(inner_height(5.0, 0.0) == 1.0, "a kind without an inner radius is whole from the eye");
    }

    #[test]
    fn cover_cells_become_batches_thinned_by_band_and_removed_when_they_leave() {
        let ground = Plain { revision: 1 };
        let palette = grass();
        let mut cover = GroundCover::default();
        for _ in 0..8 {
            cover.update(1, &ground, &palette, Vec3::new(64.0, 2.0, 64.0), None);
        }
        let part = BatchPart {
            vertex_offset: 1,
            index_offset: 0,
            index_count: 30,
            material_id: 0,
            local: Mat4::IDENTITY,
            near: 0.0,
            far: 0.0,
            casts_shadow: false,
            shadow_far: 0.0,
        };
        let mut parts = vec![None; 133];
        parts[132] = Some(vec![part]);
        let mut batches = CoverBatches::default();
        batches.begin_frame();
        let copies = batches.terrain(1, &cover, &palette, &parts, Mat4::IDENTITY, Vec3::new(64.0, 2.0, 64.0));
        batches.end_frame();
        let all: usize = cover.instances(1).len();
        assert!(copies > 0 && (copies as usize) < all, "{copies} of {all}: far cells thinned");
        assert!(batches.set.iter().all(|(id, b)| id >> 56 == 0xF1 && !b.placements.is_empty()));
        let first = batches.set.len();
        let slots: usize = batches.set.iter().map(|(_, b)| b.placements.len()).sum();
        assert!(slots as i64 <= SLOTS_SET_PER_FRAME + 10_000, "a frame's sets stay near its budget");
        // The same frame again: nothing to send. Far away: everything removed.
        batches.begin_frame();
        batches.terrain(1, &cover, &palette, &parts, Mat4::IDENTITY, Vec3::new(64.0, 2.0, 64.0));
        batches.end_frame();
        assert!(batches.set.is_empty() && batches.remove.is_empty());
        batches.begin_frame();
        batches.end_frame();
        assert_eq!(batches.remove.len(), first);
        // Cell ids never collide across kind, terrain or cell.
        assert_ne!(cover_batch_id(1, 200, 3, 4), cover_batch_id(1, 200, 4, 3));
        assert_ne!(cover_batch_id(1, 200, 3, 4), cover_batch_id(2, 200, 3, 4));
        assert_ne!(cover_batch_id(1, 200, 3, 4), cover_batch_id(1, 201, 3, 4));
    }

    #[test]
    fn the_density_a_click_reads_is_the_density_grown() {
        let ground = Plain { revision: 1 };
        let flat = grass();
        assert_eq!(density_at(&ground, 132, &flat[&132], 10.0, 10.0, glam::Vec2::ZERO), 0.2 * density_scale());
        assert_eq!(density_at(&ground, 132, &flat[&132], 10.0, 10.0, glam::Vec2::ONE), 0.2 * density_scale(), "no drifts");
        let mut palette = grass();
        palette.get_mut(&132).unwrap().ground_cover.as_mut().unwrap().clump = Some(12.0);
        let mut cover = GroundCover::default();
        let centre = Vec3::new(64.0, 2.0, 64.0);
        for _ in 0..8 {
            cover.update(3, &ground, &palette, centre, None);
        }
        assert_eq!(cover.focus(3), Some(centre));
        // Every cell within 24 m of the centre (whole), against the rule integrated over the same ground.
        let (mut grown, mut expected) = (0.0f32, 0.0f32);
        let step = CELL / 16.0;
        for ((kind, cx, cz), cell) in cover.cells(3) {
            if cell_distance(cx, cz, centre) > 24.0 {
                continue;
            }
            grown += cell.len() as f32;
            for j in 0..16 {
                for i in 0..16 {
                    let (x, z) = ((cx as f32 + (i as f32 + 0.5) / 16.0) * CELL, (cz as f32 + (j as f32 + 0.5) / 16.0) * CELL);
                    expected += density_at(&ground, kind, &palette[&kind], x, z, glam::Vec2::ZERO) * step * step;
                }
            }
        }
        assert!(grown > 100.0 && (grown - expected).abs() < 0.15 * expected, "{grown} grown, {expected} by the rule");
    }

    #[test]
    fn a_copy_draws_thinned_widened_and_sunk_as_the_batches_draw_it() {
        let copy = |x: f32, rank: f32| Cover {
            instance: PaintedFoliage { kind: 132, position: Vec3::new(x, 0.0, 0.0), yaw: 0.0, tilt: 0.0, scale: 2.0 },
            rotation: Quat::IDENTITY,
            rank,
        };
        assert_eq!(drawn_scale(&copy(1.0, 0.99), Vec3::ZERO, 400.0, 0.0), Some(Vec3::splat(2.0)), "every copy near the centre");
        let far = 4.0 * BAND_NEAR;
        assert_eq!(drawn_scale(&copy(far, 0.3), Vec3::ZERO, 400.0, 0.0), None, "a quarter kept that far");
        assert_eq!(drawn_scale(&copy(far, 0.2), Vec3::ZERO, 400.0, 0.0), Some(Vec3::new(4.0, 2.0, 4.0)), "widened twice");
        assert_eq!(drawn_scale(&copy(far, 0.0), Vec3::ZERO, far, 0.0), None, "sunk at the radius");
        // A far band: nothing inside its inner radius, whole once past its fade, thinned from where it starts.
        assert_eq!(drawn_scale(&copy(10.0, 0.0), Vec3::ZERO, 400.0, 30.0), None, "inside the inner radius");
        let past = 30.0 + INNER_FADE;
        assert_eq!(drawn_scale(&copy(past, 0.0), Vec3::ZERO, 400.0, 30.0).map(|s| s.y), Some(2.0), "risen out past it");
    }

    #[test]
    fn a_terrain_grows_cover_only_on_its_own_ground() {
        let ground = Plain { revision: 1 };
        let palette = grass();
        let mut cover = GroundCover::default();
        // 200 m east of the 128 m square: past the 40 m radius of every cell it owns.
        for _ in 0..4 {
            cover.update(1, &ground, &palette, Vec3::new(328.0, 2.0, 64.0), None);
        }
        assert_eq!(cover.terrains[&1].cells.len(), 0, "no cells kept round an eye off this terrain");
        // At its east edge: only its own cells, none past x = 128.
        cover.update(1, &ground, &palette, Vec3::new(130.0, 2.0, 64.0), None);
        assert!(!cover.terrains[&1].cells.is_empty());
        assert!(cover.terrains[&1].cells.keys().all(|&(_, cx, cz)| (0..8).contains(&cx) && (0..8).contains(&cz)));
    }

    #[test]
    fn clumped_kinds_grow_in_drifts_that_follow_the_world() {
        // Smooth: neighbours a metre apart differ little; spread over 0..1; the same at the same world point.
        let samples: Vec<f32> = (0..400).map(|i| patch_noise(i as f32 * 0.37, 5.0, 9)).collect();
        assert!(samples.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(samples.iter().cloned().fold(0.0, f32::max) > 0.7 && samples.iter().cloned().fold(1.0, f32::min) < 0.3);
        assert!((patch_noise(3.0, 4.0, 9) - patch_noise(3.05, 4.0, 9)).abs() < 0.05);
        assert_ne!(patch_noise(3.0, 4.0, 9), patch_noise(3.0, 4.0, 10), "each kind its own drifts");
        // A clumped kind keeps about half its candidates, in patches; unclumped grows as before.
        let ground = Plain { revision: 1 };
        let mut palette = grass();
        let plain = grown(&mut GroundCover::default(), 3, None, &ground).len();
        let entry = palette.get_mut(&132).unwrap();
        entry.ground_cover.as_mut().unwrap().clump = Some(12.0);
        let mut cover = GroundCover::default();
        for _ in 0..8 {
            cover.update(3, &ground, &palette, Vec3::new(64.0, 2.0, 64.0), None);
        }
        let clumped = cover.instances(3).len();
        assert!(clumped * 4 > plain && clumped * 4 < plain * 3, "{clumped} of {plain}");
        assert_eq!((drift(0.0), drift(1.0)), (0.0, 1.0));
    }
}
