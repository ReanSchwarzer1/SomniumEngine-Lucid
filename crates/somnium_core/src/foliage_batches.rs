//! Painted foliage as instance batches (`somnium_renderer::instance_batch`, open-world render scale, 19 Oct 2026).
//!
//! The per-instance path (`App::submit_foliage`) visits every instance within reach and submits a draw per part per
//! instance per frame: at Rimeborn's budget of a million instances in a 600 m disc that is a hundred thousand draws or
//! more, about 0.25 µs each on the CPU between the visit and the renderer. Here each 32 m ground cell's instances of
//! one kind are one batch, set when the cell comes within the kind's reach and removed when it falls 32 m beyond it.
//! A frame then walks the cells near the camera, and the GPU picks each copy's parts by distance, as the per-instance
//! path would have: the kind's draw distance, a project entry's LOD mesh, and the legacy leaf and heavy-part drops
//! at `lod_distance` and `impostor_distance`.
//!
//! Each cell's copies of a part are one instanced draw (`SomniumRenderer::set_instanced_batch`, Sunscar, 5 Nov 2026):
//! a forest of 400-600 stems a hectare is hundreds of thousands of copies, and a draw each would cost the GPU's front
//! end more than the trees. A run is culled and banded as one, so a cell's copies are split into one batch per
//! distance band (a near mesh, a LOD mesh) on the CPU, by each copy's distance from the camera snapped to
//! [`BAND_SNAP`]: a cell straddling a band edge is split again only when the snapped camera moves, at most
//! [`BUILDS_PER_FRAME`] cells a frame, nearest first.
//!
//! Opt-in with `SOMNIUM_FOLIAGE_BATCHES=1`. A terrain with an LOD falloff curve keeps the per-instance path (the
//! curve scales every copy every frame). What batches do not have is listed in `instance_batch.rs`: no ray-traced
//! foliage, no virtual shadow pages, two-sided raster.

use glam::{Mat4, Vec2};
use somnium_renderer::instance_batch::{BatchPart, InstanceBatch};
use somnium_renderer::terrain::foliage_paint::PaintedFoliage;

/// Ground metres per cell, as `foliage_grid`'s.
const CELL: f32 = 32.0;
/// A cell is kept this far past its kind's reach, so walking along the edge does not set and remove it every frame.
const KEEP: f32 = 32.0;
/// Batch ids in this namespace are foliage's: the top byte.
const ID_SPACE: u64 = 0xF0 << 56;
/// Copies are put in their distance band from the camera snapped to this grid, metres: a band edge moves in steps, so a
/// cell across one is rebuilt every few seconds of walking rather than every frame (and its shadows with it).
const BAND_SNAP: f32 = 16.0;
/// Most (cell, kind, band) batches built in one frame, nearest first; the rest keep their old batch a frame longer.
const BUILDS_PER_FRAME: usize = 96;
/// Copies this near a band edge with another band beyond it are in both batches, which hand them over there
/// (`InstanceBatch::fade`; visibility.wgsl dithers 8 m across the edge, as the eye sees it): half that dither band plus
/// the most the snapped camera ([`BAND_SNAP`]) can be from the eye, 11.3 m.
const HANDOVER: f32 = 16.0;

/// Whether `SOMNIUM_FOLIAGE_BATCHES=1` is set.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("SOMNIUM_FOLIAGE_BATCHES").as_deref() == Ok("1"));
    *ON
}

/// One part of a palette mesh, as `App`'s foliage parts carry it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Piece {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub material_id: u32,
    pub local: Mat4,
    pub is_leaf: bool,
}

/// A kind's distances, as the per-instance path reads them: squared draw and shadow distances (`f32::MAX` for no
/// limit), the squared distances past which a project entry's LOD mesh and impostor draw (`f32::INFINITY` for none),
/// and the terrain component's legacy `lod_distance` and `impostor_distance` (0 for none).
#[derive(Clone, Copy, Debug)]
pub(crate) struct KindDistances {
    pub cull_sq: f32,
    pub shadow_sq: f32,
    pub lod_sq: f32,
    pub impostor_sq: f32,
    pub legacy_lod: f32,
    pub legacy_impostor: f32,
}

/// `d` as a limit: 0 (none) for a squared `f32::MAX` or infinity, else its root.
fn limit(sq: f32) -> f32 {
    if sq >= f32::MAX { 0.0 } else { sq.max(0.0).sqrt() }
}

/// The parts one copy of `kind` draws and between which horizontal distances, the per-instance path's choice
/// written as distance bands: `near` (a LOD mesh from its swap) and `far` (`0` for no limit).
pub(crate) fn kind_parts(
    kind: u16,
    near_mesh: &[Piece],
    lod_mesh: Option<&[Piece]>,
    impostor_mesh: Option<&[Piece]>,
    d: KindDistances,
) -> Vec<BatchPart> {
    let reach = limit(d.cull_sq);
    let shadow_far = limit(d.shadow_sq);
    let casts = d.shadow_sq > 0.0;
    // A project entry's LOD mesh and impostor past their distances: a band each, the near mesh until the first.
    let mut tiers: Vec<(f32, &[Piece])> = vec![(0.0, near_mesh)];
    for (mesh, sq) in [(lod_mesh, d.lod_sq), (impostor_mesh, d.impostor_sq)] {
        if let Some(mesh) = mesh.filter(|_| sq.is_finite()) {
            tiers.push((sq.max(0.0).sqrt(), mesh));
        }
    }
    tiers.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let mut band = |mesh: &[Piece], near: f32, far: f32| {
        for (piece, legacy_far) in mesh.iter().zip(legacy_limits(kind, mesh, d)) {
            let far = match (far, legacy_far) {
                (0.0, l) => l,
                (f, 0.0) => f,
                (f, l) => f.min(l),
            };
            if far > 0.0 && far <= near {
                continue;
            }
            out.push(BatchPart {
                vertex_offset: piece.vertex_offset,
                index_offset: piece.index_offset,
                index_count: piece.index_count,
                material_id: piece.material_id,
                local: piece.local,
                near,
                far,
                casts_shadow: casts,
                shadow_far,
            });
        }
    };
    for (i, &(start, mesh)) in tiers.iter().enumerate() {
        if reach > 0.0 && start >= reach {
            break;
        }
        let end = tiers.get(i + 1).map_or(reach, |&(next, _)| if reach > 0.0 { next.min(reach) } else { next });
        band(mesh, start, end);
    }
    out
}

/// Per part of `mesh`, the distance past which the legacy LOD (a built-in kind) drops it, 0 if never: past
/// `lod_distance` the leaves go (the heaviest part, without leaves), past `impostor_distance` everything but the
/// solid parts (the cheapest, without leaves). Read off `App::submit_foliage`'s `keep_only`/`drop_heavy`.
fn legacy_limits(kind: u16, mesh: &[Piece], d: KindDistances) -> Vec<f32> {
    if crate::foliage_palette::is_project_kind(kind) || mesh.len() < 2 {
        return vec![0.0; mesh.len()];
    }
    let has_leaf = mesh.iter().any(|p| p.is_leaf);
    let cheapest = mesh.iter().enumerate().min_by_key(|(_, p)| p.index_count).map(|(i, _)| i);
    let heaviest = mesh.iter().enumerate().max_by_key(|(_, p)| p.index_count).map(|(i, _)| i);
    let first = |a: f32, b: f32| match (a > 0.0, b > 0.0) {
        (true, true) => a.min(b),
        (true, false) => a,
        (false, true) => b,
        (false, false) => 0.0,
    };
    mesh.iter()
        .enumerate()
        .map(|(i, p)| {
            let by_impostor = if has_leaf { p.is_leaf } else { cheapest != Some(i) };
            let by_lod = if has_leaf { p.is_leaf } else { heaviest == Some(i) };
            match (by_lod, by_impostor) {
                (true, true) => first(d.legacy_lod, d.legacy_impostor),
                (true, false) => d.legacy_lod,
                (false, true) => d.legacy_impostor,
                (false, false) => 0.0,
            }
        })
        .collect()
}

/// A terrain's instances by cell and kind.
#[derive(Default)]
struct Cells {
    key: Option<(u64, usize)>,
    min: [f32; 2],
    dims: [i32; 2],
    /// Per cell, `(kind, start, end, members)` into `order`; `members` hashes the copies, so a band holding them all
    /// is signed without walking them.
    cells: Vec<Vec<(u16, u32, u32, u64)>>,
    order: Vec<u32>,
    /// The snapped camera, the kinds' signatures folded, and the copies held, of the last frame that built every
    /// batch it wanted: a frame that finds the same has nothing to change, and keeps the batches without walking them.
    steady: Option<(Vec2, u64, u32)>,
}

impl Cells {
    fn refresh(&mut self, revision: u64, instances: &[PaintedFoliage]) {
        let key = (revision, instances.len());
        if self.key == Some(key) {
            return;
        }
        self.key = Some(key);
        let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
        for p in instances {
            lo = [lo[0].min(p.position.x), lo[1].min(p.position.z)];
            hi = [hi[0].max(p.position.x), hi[1].max(p.position.z)];
        }
        if instances.is_empty() {
            (lo, hi) = ([0.0; 2], [0.0; 2]);
        }
        self.min = lo;
        self.dims = [((hi[0] - lo[0]) / CELL) as i32 + 1, ((hi[1] - lo[1]) / CELL) as i32 + 1];
        debug_assert!(i64::from(self.dims[0]) * i64::from(self.dims[1]) <= 1 << 24, "more cells than a batch id holds");
        let cell_of = |p: &PaintedFoliage| {
            let cx = ((p.position.x - lo[0]) / CELL) as i32;
            let cz = ((p.position.z - lo[1]) / CELL) as i32;
            (cz * self.dims[0] + cx) as usize
        };
        let mut order: Vec<u32> = (0..instances.len() as u32).collect();
        order.sort_by_key(|&i| {
            let p = &instances[i as usize];
            (cell_of(p), p.kind)
        });
        let mut cells = vec![Vec::new(); (self.dims[0] * self.dims[1]) as usize];
        let mut start = 0;
        while start < order.len() {
            let first = &instances[order[start] as usize];
            let (cell, kind) = (cell_of(first), first.kind);
            let mut end = start + 1;
            while end < order.len() {
                let p = &instances[order[end] as usize];
                if cell_of(p) != cell || p.kind != kind {
                    break;
                }
                end += 1;
            }
            let mut members = 0xcbf2_9ce4_8422_2325u64;
            for i in &order[start..end] {
                mix(&mut members, *i);
            }
            cells[cell].push((kind, start as u32, end as u32, members));
            start = end;
        }
        (self.cells, self.order) = (cells, order);
    }
}

/// What one frame asks of the renderer.
#[derive(Default)]
pub(crate) struct Changes {
    pub set: Vec<(u64, InstanceBatch)>,
    pub remove: Vec<u64>,
}

/// The foliage batches set on the renderer, by terrain.
#[derive(Default)]
pub(crate) struct FoliageBatches {
    cells: std::collections::HashMap<u32, Cells>,
    /// `(terrain, cell, kind and band)` -> (signature, frame last needed, copies).
    live: std::collections::HashMap<(u32, u32, u16), (u64, u64, u32)>,
    frame: u64,
    seen: Vec<u32>,
    /// Terrains whose frame changed nothing ([`Cells::steady`]): their batches are kept as they are.
    steady: Vec<u32>,
    pub changes: Changes,
}

/// The namespace byte, 16 bits of terrain, 24 of cell (a 131 km square of 32 m cells) and 16 of kind, so every
/// kind a palette can hold has its own batch in a cell. The kind field carries the band in its top three bits
/// ([`band_kind`]): kinds stop at 13 bits. Runtime only: never saved.
fn batch_id(terrain: u32, cell: u32, kind: u16) -> u64 {
    ID_SPACE | (u64::from(terrain & 0xFFFF) << 40) | (u64::from(cell & 0xFF_FFFF) << 16) | u64::from(kind)
}

/// `kind` with distance band `band` (0..8) in its top three bits.
fn band_kind(kind: u16, band: usize) -> u16 {
    debug_assert!(kind < 1 << 13 && band < 8);
    kind | (band as u16) << 13
}

/// A kind's parts grouped by their distance band `(near, far)` (0 far: no limit), each group's parts drawn without a
/// band of their own: the copies in a band are chosen on the CPU.
fn bands(list: &[BatchPart]) -> Vec<(f32, f32, Vec<BatchPart>)> {
    let mut out: Vec<(f32, f32, Vec<BatchPart>)> = Vec::new();
    for p in list {
        let part = BatchPart { near: 0.0, far: 0.0, ..*p };
        match out.iter_mut().find(|(n, f, _)| *n == p.near && *f == p.far) {
            Some((_, _, parts)) => parts.push(part),
            None => out.push((p.near, p.far, vec![part])),
        }
    }
    out
}

fn mix(h: &mut u64, word: u32) {
    *h = (*h ^ u64::from(word)).wrapping_mul(0x0100_0000_01b3);
}

impl FoliageBatches {
    /// Start a frame: what the last frame asked is taken already.
    pub(crate) fn begin_frame(&mut self) {
        self.frame += 1;
        self.seen.clear();
        self.steady.clear();
        self.changes = Changes::default();
    }

    /// One terrain's frame: set the batches of cells newly within reach (or whose foliage, parts or distances
    /// changed), and keep the rest live. `parts[kind]` is `None` while a kind's mesh is not loaded, and a kind past
    /// its end is not drawn. Returns the copies its live batches hold.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn terrain(
        &mut self,
        terrain: u32,
        revision: u64,
        instances: &[PaintedFoliage],
        model: Mat4,
        camera_local: glam::Vec3,
        parts: &[Option<Vec<BatchPart>>],
    ) -> u32 {
        self.seen.push(terrain);
        let cells = self.cells.entry(terrain).or_default();
        cells.refresh(revision, instances);
        // Per kind: its reach and the signature its batches are built to (a new one rebuilds them).
        let mut reach = vec![f32::NAN; parts.len()];
        let mut signature = vec![0u64; parts.len()];
        let mut farthest = 0.0f32;
        for (kind, list) in parts.iter().enumerate() {
            let Some(list) = list else { continue };
            let r = if list.iter().any(|p| p.far == 0.0) {
                f32::INFINITY
            } else {
                list.iter().map(|p| p.far).fold(0.0, f32::max)
            };
            reach[kind] = r;
            farthest = farthest.max(r);
            let mut h = 0xcbf2_9ce4_8422_2325u64 ^ revision.rotate_left(13) ^ instances.len() as u64;
            for value in model.to_cols_array() {
                mix(&mut h, value.to_bits());
            }
            for p in list {
                for word in [p.vertex_offset, p.index_offset, p.index_count, p.material_id, u32::from(p.casts_shadow)] {
                    mix(&mut h, word);
                }
                for value in [p.near, p.far, p.shadow_far].into_iter().chain(p.local.to_cols_array()) {
                    mix(&mut h, value.to_bits());
                }
            }
            signature[kind] = h;
        }
        if cells.order.is_empty() {
            return 0;
        }
        let centre = (Vec2::new(camera_local.x, camera_local.z) / BAND_SNAP).round() * BAND_SNAP;
        // At 600,000 trees, walking the copies in reach to find nothing changed cost 3-4 ms a frame.
        let mut all = 0xcbf2_9ce4_8422_2325u64;
        for (h, r) in signature.iter().zip(&reach) {
            mix(&mut all, *h as u32);
            mix(&mut all, (*h >> 32) as u32);
            mix(&mut all, r.to_bits());
        }
        if let Some((c, a, held)) = cells.steady {
            if c == centre && a == all {
                self.steady.push(terrain);
                return held;
            }
        }
        let span = farthest + KEEP;
        let cell_range = |v: f32, lo: f32, n: i32| (((v - lo) / CELL).floor() as i32).clamp(0, n - 1);
        let (x0, x1, z0, z1) = if span.is_finite() {
            (
                cell_range(centre.x - span, cells.min[0], cells.dims[0]),
                cell_range(centre.x + span, cells.min[0], cells.dims[0]),
                cell_range(centre.y - span, cells.min[1], cells.dims[1]),
                cell_range(centre.y + span, cells.min[1], cells.dims[1]),
            )
        } else {
            (0, cells.dims[0] - 1, 0, cells.dims[1] - 1)
        };
        let grouped: Vec<Option<Vec<(f32, f32, Vec<BatchPart>)>>> =
            parts.iter().map(|list| list.as_deref().map(bands)).collect();
        // Every (cell, kind) in reach, nearest first, so a frame's builds go to the nearest.
        let mut wanted: Vec<(f32, f32, u32, u16, u32, u32, u64)> = Vec::new();
        for cz in z0..=z1 {
            for cx in x0..=x1 {
                let cell = (cz * cells.dims[0] + cx) as u32;
                let lo = Vec2::new(cells.min[0] + cx as f32 * CELL, cells.min[1] + cz as f32 * CELL);
                let gap = (lo - centre).max(centre - (lo + Vec2::splat(CELL))).max(Vec2::ZERO).length();
                let far_corner = (lo + Vec2::splat(CELL * 0.5) - centre).abs() + Vec2::splat(CELL * 0.5);
                for &(kind, start, end, members) in &cells.cells[cell as usize] {
                    let r = reach.get(usize::from(kind)).copied().unwrap_or(f32::NAN);
                    if r.is_nan() || gap > r {
                        continue;
                    }
                    wanted.push((gap, far_corner.length(), cell, kind, start, end, members));
                }
            }
        }
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut builds = 0;
        let mut copies = 0u32;
        let mut deferred = false;
        for (gap, far_corner, cell, kind, start, end, member_hash) in wanted {
            let Some(Some(groups)) = grouped.get(usize::from(kind)) else { continue };
            let members = &cells.order[start as usize..end as usize];
            for (band, (near, far, band_parts)) in groups.iter().enumerate().take(8) {
                // An edge with another band beyond it is a hand-over: the copies near it go in both bands, this one
                // fading them out past its far edge, or in past its near edge (never both: a middle band fades in).
                let fade_out = *far > 0.0 && groups.iter().any(|(n, _, _)| n == far);
                let fade_in = *near > 0.0 && groups.iter().any(|(_, f, _)| f == near);
                let lo = if fade_in { *near - HANDOVER } else { *near };
                let hi = if fade_out { *far + HANDOVER } else { *far };
                let fade = if fade_in { -*near } else if fade_out { *far } else { 0.0 };
                let whole = lo <= gap && (*far == 0.0 || far_corner < hi);
                let none = (*far > 0.0 && gap >= hi) || far_corner < lo;
                if none {
                    continue;
                }
                let inside = |i: &u32| {
                    let p = instances[*i as usize].position;
                    let d = Vec2::new(p.x, p.z).distance(centre);
                    d >= lo && (*far == 0.0 || d < hi)
                };
                // Which copies the band holds, folded into the signature: a copy crossing the edge rebuilds it.
                let mut sig = signature[kind as usize] ^ (band as u64).rotate_left(29);
                let mut count = 0u32;
                if whole {
                    // Every copy: the cell's member hash signs them (a forest to 900 m is too many to walk).
                    mix(&mut sig, member_hash as u32);
                    mix(&mut sig, (member_hash >> 32) as u32);
                    count = end - start;
                } else {
                    for i in members.iter().filter(|i| inside(i)) {
                        mix(&mut sig, *i);
                        count += 1;
                    }
                }
                if count == 0 {
                    continue;
                }
                let key = (terrain, cell, band_kind(kind, band));
                let live = self.live.get(&key).copied();
                let held = match live {
                    Some((old, _, n)) if old == sig => n,
                    // Over this frame's builds: an old batch draws on a frame longer, a new one waits for it.
                    Some((old, _, n)) if builds >= BUILDS_PER_FRAME => {
                        self.live.insert(key, (old, self.frame, n));
                        copies += n;
                        deferred = true;
                        continue;
                    }
                    None if builds >= BUILDS_PER_FRAME => {
                        deferred = true;
                        continue;
                    }
                    _ => {
                        builds += 1;
                        let placements: Vec<Mat4> = members
                            .iter()
                            .filter(|i| whole || inside(i))
                            .map(|&i| &instances[i as usize])
                            .filter(|inst| inst.scale > 0.0)
                            .map(|inst| {
                                model
                                    * Mat4::from_scale_rotation_translation(
                                        glam::Vec3::splat(inst.scale),
                                        glam::Quat::from_rotation_y(inst.yaw) * glam::Quat::from_rotation_x(inst.tilt),
                                        inst.position,
                                    )
                            })
                            .collect();
                        let n = placements.len() as u32;
                        self.changes.set.push((
                            batch_id(terrain, cell, key.2),
                            InstanceBatch { parts: band_parts.clone(), placements, fade },
                        ));
                        n
                    }
                };
                self.live.insert(key, (sig, self.frame, held));
                copies += held;
            }
        }
        if let Some(cells) = self.cells.get_mut(&terrain) {
            cells.steady = (!deferred).then_some((centre, all, copies));
        }
        copies
    }

    /// End the frame: every live batch not needed this frame (out of reach, a terrain gone or back on the
    /// per-instance path) is removed.
    pub(crate) fn end_frame(&mut self) {
        let frame = self.frame;
        let remove = &mut self.changes.remove;
        let steady = &self.steady;
        self.live.retain(|&(terrain, cell, kind), &mut (_, last, _)| {
            let keep = last == frame || steady.contains(&terrain);
            if !keep {
                remove.push(batch_id(terrain, cell, kind));
            }
            keep
        });
        let seen = &self.seen;
        self.cells.retain(|terrain, _| seen.contains(terrain));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(index_count: u32, is_leaf: bool) -> Piece {
        Piece { vertex_offset: index_count, index_offset: 0, index_count, material_id: 0, local: Mat4::IDENTITY, is_leaf }
    }

    fn distances(cull: f32, shadow: f32) -> KindDistances {
        KindDistances {
            impostor_sq: f32::INFINITY,
            cull_sq: cull * cull,
            shadow_sq: shadow * shadow,
            lod_sq: f32::INFINITY,
            legacy_lod: 0.0,
            legacy_impostor: 0.0,
        }
    }

    #[test]
    fn a_kind_draws_to_its_reach_and_casts_to_its_shadow_distance() {
        let parts = kind_parts(130, &[piece(300, false)], None, None, distances(60.0, 20.0));
        assert_eq!((parts[0].near, parts[0].far, parts[0].shadow_far, parts[0].casts_shadow), (0.0, 60.0, 20.0, true));
        let unlimited = KindDistances { cull_sq: f32::MAX, shadow_sq: f32::MAX, ..distances(0.0, 0.0) };
        let parts = kind_parts(130, &[piece(300, false)], None, None, unlimited);
        assert_eq!((parts[0].far, parts[0].shadow_far, parts[0].casts_shadow), (0.0, 0.0, true));
        let parts = kind_parts(130, &[piece(300, false)], None, None, distances(60.0, 0.0));
        assert!(!parts[0].casts_shadow, "a shadow distance of 0 casts nothing");
    }

    #[test]
    fn a_project_lod_mesh_takes_over_at_its_distance() {
        let d = KindDistances { lod_sq: 150.0 * 150.0, ..distances(400.0, 200.0) };
        let parts = kind_parts(140, &[piece(9000, false)], Some(&[piece(800, false)]), None, d);
        let bands: Vec<_> = parts.iter().map(|p| (p.index_count, p.near, p.far)).collect();
        assert_eq!(bands, [(9000, 0.0, 150.0), (800, 150.0, 400.0)]);
    }

    #[test]
    fn an_impostor_takes_over_from_the_lod_mesh() {
        let d = KindDistances { lod_sq: 50.0 * 50.0, impostor_sq: 200.0 * 200.0, ..distances(800.0, 100.0) };
        let (near, lod, imp) = ([piece(9000, false)], [piece(800, false)], [piece(4, false)]);
        let bands: Vec<(u32, f32, f32)> =
            kind_parts(140, &near, Some(&lod), Some(&imp), d).iter().map(|p| (p.index_count, p.near, p.far)).collect();
        assert_eq!(bands, vec![(9000, 0.0, 50.0), (800, 50.0, 200.0), (4, 200.0, 800.0)]);
        // An impostor past the draw distance never draws; the LOD mesh keeps the rest.
        let d = KindDistances { impostor_sq: 900.0 * 900.0, ..d };
        let bands: Vec<(u32, f32, f32)> =
            kind_parts(140, &near, Some(&lod), Some(&imp), d).iter().map(|p| (p.index_count, p.near, p.far)).collect();
        assert_eq!(bands, vec![(9000, 0.0, 50.0), (800, 50.0, 800.0)]);
    }

    #[test]
    fn legacy_trees_drop_leaves_then_keep_only_bark() {
        let d = KindDistances { legacy_lod: 45.0, legacy_impostor: 90.0, ..distances(300.0, 100.0) };
        let tree = [piece(5000, false), piece(20000, true), piece(2000, false)];
        let fars: Vec<f32> = kind_parts(5, &tree, None, None, d).iter().map(|p| p.far).collect();
        // Bark to the reach; leaves gone past lod_distance.
        assert_eq!(fars, [300.0, 45.0, 300.0]);
        // No leaf marked: the heaviest goes past lod_distance, all but the cheapest past impostor_distance.
        let plain = [piece(5000, false), piece(20000, false), piece(2000, false)];
        let fars: Vec<f32> = kind_parts(5, &plain, None, None, d).iter().map(|p| p.far).collect();
        assert_eq!(fars, [90.0, 45.0, 300.0]);
        // Project kinds (128 and up) keep every part.
        let fars: Vec<f32> = kind_parts(200, &plain, None, None, d).iter().map(|p| p.far).collect();
        assert_eq!(fars, [300.0; 3]);
    }

    #[test]
    fn a_batch_id_keeps_terrain_cell_and_kind_apart() {
        let fields = |id: u64| (id >> 56, (id >> 40) & 0xFFFF, (id >> 16) & 0xFF_FFFF, id & 0xFFFF);
        assert_eq!(fields(batch_id(7, 0xAB_CDEF, 200)), (0xF0, 7, 0xAB_CDEF, 200));
        assert_eq!(fields(batch_id(0x1_0007, 0, 0)), (0xF0, 7, 0, 0), "terrain bits never reach the namespace");
        assert_ne!(batch_id(1, 1, 0), batch_id(1, 0, 0));
        let top = (crate::foliage_palette::FOLIAGE_KINDS - 1) as u16;
        assert_eq!(fields(batch_id(7, 9, top)), (0xF0, 7, 9, u64::from(top)));
        assert_ne!(batch_id(1, 5, 44), batch_id(1, 5, 300), "a kind past a byte is not its low byte");
        assert_ne!(batch_id(1, 5, 44 + 256), batch_id(1, 6, 44), "nor does it spill into the cell");
    }

    /// Two kinds that share a low byte, painted on the same ground, are two batches, each with its own parts.
    #[test]
    fn kinds_past_a_byte_in_one_cell_are_their_own_batches() {
        let instances: Vec<PaintedFoliage> = (0..8)
            .map(|i| PaintedFoliage {
                kind: [44, 300][i % 2],
                position: glam::Vec3::new(i as f32, 0.0, 1.0),
                yaw: 0.0,
                tilt: 0.0,
                scale: 1.0,
            })
            .collect();
        let mut parts = vec![None; 301];
        parts[44] = Some(kind_parts(44, &[piece(30, false)], None, None, distances(100.0, 0.0)));
        parts[300] = Some(kind_parts(300, &[piece(60, false)], None, None, distances(100.0, 0.0)));
        let mut batches = FoliageBatches::default();
        batches.begin_frame();
        let copies = batches.terrain(3, 1, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts);
        batches.end_frame();
        assert_eq!(copies, 8);
        let mut set: Vec<(u64, u32, usize)> = batches
            .changes
            .set
            .iter()
            .map(|(id, b)| (id & 0xFFFF, b.parts[0].index_count, b.placements.len()))
            .collect();
        set.sort_unstable();
        assert_eq!(set, [(44, 30, 4), (300, 60, 4)]);
        // A kind past the parts it was given is not drawn, rather than read out of bounds.
        batches.begin_frame();
        batches.terrain(3, 2, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts[..45]);
        batches.end_frame();
        assert_eq!(batches.changes.set.iter().map(|(id, _)| id & 0xFFFF).collect::<Vec<_>>(), [44]);
    }

    fn field() -> Vec<PaintedFoliage> {
        (0..200)
            .map(|i| PaintedFoliage {
                kind: [130, 131][i % 2],
                position: glam::Vec3::new((i % 20) as f32 * 20.0, 0.0, (i / 20) as f32 * 20.0),
                yaw: 0.0,
                tilt: 0.0,
                scale: 1.0,
            })
            .collect()
    }

    fn parts_for(reach_a: f32, reach_b: f32) -> Vec<Option<Vec<BatchPart>>> {
        let mut parts = vec![None; 256];
        parts[130] = Some(kind_parts(130, &[piece(30, false)], None, None, distances(reach_a, 0.0)));
        parts[131] = Some(kind_parts(131, &[piece(60, false)], None, None, distances(reach_b, 0.0)));
        parts
    }

    #[test]
    fn cells_are_set_as_they_come_into_reach_and_removed_past_it() {
        let instances = field();
        let mut batches = FoliageBatches::default();
        let parts = parts_for(50.0, 150.0);
        batches.begin_frame();
        let copies = batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::new(0.0, 0.0, 0.0), &parts);
        batches.end_frame();
        let first = batches.changes.set.len();
        assert!(first > 0 && copies > 0);
        let wide = batches.changes.set.iter().filter(|(id, _)| id & 0xFFFF == 131).count();
        let short = batches.changes.set.iter().filter(|(id, _)| id & 0xFFFF == 130).count();
        assert!(wide > short, "the longer-reach kind is set over more cells");
        // The same frame again: nothing to send.
        batches.begin_frame();
        batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts);
        batches.end_frame();
        assert!(batches.changes.set.is_empty() && batches.changes.remove.is_empty());
        // Far away: everything removed.
        batches.begin_frame();
        batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::new(5000.0, 0.0, 5000.0), &parts);
        batches.end_frame();
        assert_eq!(batches.changes.remove.len(), first);
        // A repaint (new revision) rebuilds what is in reach; a terrain no longer seen is removed.
        batches.begin_frame();
        batches.terrain(1, 2, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts);
        batches.end_frame();
        assert_eq!(batches.changes.set.len(), first);
        batches.begin_frame();
        batches.end_frame();
        assert_eq!(batches.changes.remove.len(), first);
    }

    #[test]
    fn a_frame_that_changes_nothing_keeps_every_batch_without_walking_them() {
        let instances = field();
        let parts = parts_for(50.0, 50.0);
        let mut batches = FoliageBatches::default();
        batches.begin_frame();
        let first = batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts);
        batches.end_frame();
        let live = batches.live.len();
        assert!(first > 0 && live > 0);
        // The same snapped camera: the steady path, the same copies, every batch kept, nothing set or removed.
        batches.begin_frame();
        let again = batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::new(3.0, 0.0, -2.0), &parts);
        assert_eq!(batches.steady, vec![1]);
        batches.end_frame();
        assert_eq!(again, first);
        assert_eq!(batches.live.len(), live);
        assert!(batches.changes.set.is_empty() && batches.changes.remove.is_empty());
        // A step past the snap: walked again.
        batches.begin_frame();
        batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::new(40.0, 0.0, 0.0), &parts);
        assert!(batches.steady.is_empty());
        batches.end_frame();
    }

    #[test]
    fn a_small_step_changes_nothing_and_copies_near_a_lod_edge_draw_in_both_bands() {
        let instances = field();
        let mut batches = FoliageBatches::default();
        let parts = parts_for(50.0, 50.0);
        batches.begin_frame();
        batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts);
        batches.end_frame();
        // 6 m on: the camera snaps to the same place, so no batch is set or removed.
        batches.begin_frame();
        batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::new(6.0, 0.0, 0.0), &parts);
        batches.end_frame();
        assert!(batches.changes.set.is_empty() && batches.changes.remove.is_empty());
        // A near mesh to 150 m and a LOD mesh from it to 400: every copy in reach is in one band's batch, and in both
        // within HANDOVER of 150 m, where the near batch fades out and the far one in.
        let d = KindDistances { lod_sq: 150.0 * 150.0, ..distances(400.0, 0.0) };
        let mut parts = vec![None; 256];
        parts[130] = Some(kind_parts(130, &[piece(9000, false)], Some(&[piece(800, false)]), None, d));
        let mut batches = FoliageBatches::default();
        batches.begin_frame();
        let copies = batches.terrain(1, 1, &instances, Mat4::IDENTITY, glam::Vec3::ZERO, &parts);
        batches.end_frame();
        let in_reach = instances.iter().filter(|i| i.kind == 130 && Vec2::new(i.position.x, i.position.z).length() < 400.0).count();
        let near: usize = batches.changes.set.iter().filter(|(id, _)| (id & 0xFFFF) >> 13 == 0).map(|(_, b)| b.placements.len()).sum();
        let far: usize = batches.changes.set.iter().filter(|(id, _)| (id & 0xFFFF) >> 13 == 1).map(|(_, b)| b.placements.len()).sum();
        let both = instances
            .iter()
            .filter(|i| i.kind == 130 && (Vec2::new(i.position.x, i.position.z).length() - 150.0).abs() < HANDOVER)
            .count();
        assert_eq!(copies as usize, near + far);
        assert!(near + far >= in_reach && near + far <= in_reach + both, "{near} + {far} vs {in_reach} (+{both})");
        assert!(near > 0 && far > 0, "{near} near, {far} far");
        assert!(batches.changes.set.iter().all(|(id, b)| {
            let fade = if (id & 0xFFFF) >> 13 == 0 { 150.0 } else { -150.0 };
            b.fade == fade
        }));
        assert!(batches.changes.set.iter().all(|(id, b)| {
            let mesh = if (id & 0xFFFF) >> 13 == 0 { 9000 } else { 800 };
            b.parts.iter().all(|p| p.index_count == mesh && p.near == 0.0 && p.far == 0.0)
        }));
    }
}
