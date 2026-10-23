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

use std::collections::HashMap;

use glam::{Quat, Vec3};
use somnium_renderer::terrain::TerrainData;
use somnium_renderer::terrain::foliage::SurfaceSample;
use somnium_renderer::terrain::foliage_paint::PaintedFoliage;

use crate::foliage_palette::ProjectEntry;

/// Cell edge, metres.
pub const CELL: f32 = 16.0;
/// Most cells generated in one frame (the rest come next frame).
const CELLS_PER_FRAME: usize = 24;

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
    /// Height, slope and `layer`'s weight at a terrain-local point.
    fn surface_sample(&self, x: f32, z: f32, layer: u8) -> SurfaceSample;
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

    fn surface_sample(&self, x: f32, z: f32, layer: u8) -> SurfaceSample {
        TerrainData::surface_sample(self, x, z, layer)
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
    revision: u64,
    /// The seed key the cells were made with.
    seed: u64,
    /// The area cover grows in (`TerrainStreaming::own_area`), if not the whole terrain.
    area: Option<glam::Vec4>,
    cells: HashMap<(u16, i32, i32), Vec<Cover>>,
    /// Every live cell's instances, rebuilt when the set of cells changes.
    flat: Vec<Cover>,
}

/// One generated instance: where it stands (its yaw and tilt as a painted one's) and its whole rotation.
#[derive(Clone, Copy)]
pub struct Cover {
    /// The instance, terrain-local.
    pub instance: PaintedFoliage,
    /// `Ry(yaw) * Rx(tilt)`, turned onto the ground's normal for `align_to_ground` kinds.
    pub rotation: Quat,
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
        self.update_seeded(terrain_id, u64::from(terrain_id), terrain, palette, camera, area);
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
    ) {
        let kinds: Vec<(u16, &ProjectEntry, f32, f32)> = palette
            .iter()
            .filter_map(|(kind, entry)| entry.ground_cover.as_ref().map(|g| (*kind, entry, g.density, g.radius)))
            .collect();
        if kinds.is_empty() {
            self.terrains.remove(&terrain_id);
            return;
        }
        let cover = self.terrains.entry(terrain_id).or_default();
        let revision = terrain.edit_revision();
        if cover.revision != revision || cover.area != area || cover.seed != seed_key {
            *cover = TerrainCover { revision, seed: seed_key, area, ..TerrainCover::default() };
        }
        let mut changed = false;
        let mut made = 0;
        for &(kind, entry, density, radius) in &kinds {
            let before = cover.cells.len();
            cover.cells.retain(|&(k, cx, cz), _| k != kind || cell_distance(cx, cz, camera) <= radius + CELL);
            changed |= cover.cells.len() != before;
            let reach = (radius / CELL).ceil() as i32 + 1;
            let (ox, oz) = ((camera.x / CELL).floor() as i32, (camera.z / CELL).floor() as i32);
            let mut wanted: Vec<(f32, i32, i32)> = (-reach..=reach)
                .flat_map(|dz| (-reach..=reach).map(move |dx| (ox + dx, oz + dz)))
                .map(|(cx, cz)| (cell_distance(cx, cz, camera), cx, cz))
                .filter(|&(d, cx, cz)| d <= radius && !cover.cells.contains_key(&(kind, cx, cz)))
                .collect();
            wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (_, cx, cz) in wanted {
                if made >= CELLS_PER_FRAME {
                    break;
                }
                cover.cells.insert((kind, cx, cz), generate(seed_key, terrain, kind, entry, density, cx, cz, area));
                made += 1;
                changed = true;
            }
        }
        if changed {
            cover.flat.clear();
            for cell in cover.cells.values() {
                cover.flat.extend_from_slice(cell);
            }
        }
    }

    /// `terrain`'s generated instances (terrain-local), as painted ones are.
    #[must_use]
    pub fn instances(&self, terrain_id: u32) -> &[Cover] {
        self.terrains.get(&terrain_id).map_or(&[], |c| c.flat.as_slice())
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
) -> Vec<Cover> {
    let b = &entry.brush;
    let [wx, wz] = terrain.world_size();
    let seed = cell_seed(seed_key, kind, cx, cz);
    let expected = density * CELL * CELL;
    let count = expected.floor() as u64 + u64::from(unit(mix(seed)) < expected.fract());
    let min_slope_cos = b.max_slope_deg.to_radians().cos();
    let mut out = Vec::new();
    for i in 0..count {
        let h = mix(seed ^ (i + 1).wrapping_mul(0xA24B_AED4_963E_E407));
        let (x, z) = ((cx as f32 + unit(h)) * CELL, (cz as f32 + unit(mix(h))) * CELL);
        if !(0.0..wx).contains(&x) || !(0.0..wz).contains(&z) || !area.is_none_or(|a| inside(a, x, z)) {
            continue;
        }
        let s = terrain.surface_sample(x, z, b.layer);
        let keep = unit(mix(h ^ 1));
        if s.layer_weight < b.min_layer_weight.max(1e-3) || keep > s.layer_weight || s.slope_cos < min_slope_cos {
            continue;
        }
        let t = unit(mix(h ^ 2));
        let instance = PaintedFoliage {
            kind,
            position: Vec3::new(x, s.height, z),
            yaw: unit(mix(h ^ 3)) * std::f32::consts::TAU,
            tilt: b.max_tilt_deg.to_radians() * unit(mix(h ^ 4)),
            scale: b.scale_min + (b.scale_max - b.scale_min) * t,
        };
        let mut rotation = Quat::from_rotation_y(instance.yaw) * Quat::from_rotation_x(instance.tilt);
        if entry.ground_cover.as_ref().is_some_and(|g| g.align_to_ground) {
            rotation = Quat::from_rotation_arc(Vec3::Y, ground_normal(terrain, x, z)) * rotation;
        }
        out.push(Cover { instance, rotation });
    }
    out
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

        fn surface_sample(&self, _x: f32, _z: f32, layer: u8) -> SurfaceSample {
            SurfaceSample { height: 2.0, slope_cos: 1.0, layer_weight: if layer == 0 { 1.0 } else { 0.0 } }
        }
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
                Some(key) => cover.update_seeded(id, key, ground, &palette, camera, None),
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
}
