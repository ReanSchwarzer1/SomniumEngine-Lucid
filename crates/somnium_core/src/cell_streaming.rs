//! Read-only world cells streamed by distance, for open worlds built offline (Rimeborn's Tyffir).
//!
//! A `somnium.CellStreaming` names a manifest of cells. Each cell is a plain schema scene (a composer's output, never
//! edited in place) with the XZ bounds of what it holds. Cells within reach of the view are read and parsed on worker
//! threads, installed a few per frame into the live world with their meshes restored from the same imported-mesh
//! cache a scene open uses, and despawned once the view is past reach plus a margin. Streamed entities carry
//! [`Streamed`], which is not a registered component: saves, Play checkpoints and undo snapshots leave them out, so
//! a saved scene never fills up with its cells.
//!
//! Not the actor World Partition (`world_partition.rs`): that one keeps one editable file per actor and writes cells
//! back on unload. This one only reads, streams whole hierarchies, and costs nothing for a scene without the
//! component.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use somnium_ecs::component_schema;
use somnium_ecs::reflect::TypeRegistry;
use somnium_ecs::{Component, Entity};

/// Streams the cells of a manifest around the view.
#[derive(Debug, Clone, PartialEq)]
pub struct CellStreamingComponent {
    /// Off: every streamed cell unloads.
    pub enabled: bool,
    /// The cell manifest, relative to the project.
    pub manifest: String,
    /// Cells whose bounds come within this distance of the view load (a cell's own `reach` wins).
    pub load_radius: f32,
    /// Loaded cells unload once the view is this much further than their reach.
    pub unload_margin: f32,
    /// Most cells installed in one frame (each read and parsed off the main thread first).
    pub cells_per_frame: u32,
    /// Live: cells in the world now.
    pub loaded_cells: u32,
    /// Live: cells being read.
    pub pending_cells: u32,
    /// Live: entities the cells hold now.
    pub streamed_entities: u32,
    /// Live: what the streamer is doing.
    pub status: String,
}

impl Component for CellStreamingComponent {}

impl Default for CellStreamingComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            manifest: String::new(),
            load_radius: 600.0,
            unload_margin: 100.0,
            cells_per_frame: 2,
            loaded_cells: 0,
            pending_cells: 0,
            streamed_entities: 0,
            status: String::new(),
        }
    }
}

pub(crate) fn register(registry: &mut TypeRegistry) {
    registry.register(component_schema! {
        TerrainStreamingComponent as "somnium.TerrainStreaming", display "Terrain Streaming", version 1,
        fields {
            geometry_distance { min: 0.0, soft_max: 8192.0, unit: "m", group: "Streaming", doc: "An open world's terrain window: its geometry is resident only while the camera is within this distance of the terrain's edge. 0 keeps it resident. Read when the terrain is created." },
            own_area { group: "Streaming", doc: "The part of the window that is its own where neighbouring windows overlap: x0, z0, x1, z1 in terrain-local metres. Generated ground cover grows only inside it, so an overlap is not covered twice. All zero: the whole terrain." },
            layer_set { group: "Streaming", doc: "The region's own ground layers: the folder `terrain/sets/<name>` under the content root, in place of `terrain`. Empty: the project's layers. Read when the terrain is created." },
        }
    });
    registry.register(component_schema! {
        CellStreamingComponent as "somnium.CellStreaming", display "Cell Streaming", version 1,
        fields {
            enabled { group: "Cells", doc: "Off: every streamed cell unloads." },
            manifest { group: "Cells", doc: "The cell manifest (JSON), relative to the project: {\"version\": 1, \"cells\": [{\"file\", \"min\": [x, z], \"max\": [x, z], \"reach\"?}]}, files relative to the manifest." },
            load_radius { min: 0.0, soft_max: 4096.0, unit: "m", group: "Cells", doc: "Cells whose bounds come this near the view load, unless a cell names its own reach." },
            unload_margin { min: 0.0, soft_max: 1024.0, unit: "m", group: "Cells", doc: "A loaded cell unloads once the view is this much past its reach." },
            cells_per_frame { min: 1, soft_max: 16, group: "Cells", doc: "Most cells installed in one frame." },
            loaded_cells { read_only: true, group: "Diagnostics", flags: FieldFlags::EDIT.union(FieldFlags::SCRIPT_READ) },
            pending_cells { read_only: true, group: "Diagnostics", flags: FieldFlags::EDIT.union(FieldFlags::SCRIPT_READ) },
            streamed_entities { read_only: true, group: "Diagnostics", flags: FieldFlags::EDIT.union(FieldFlags::SCRIPT_READ) },
            status { read_only: true, group: "Diagnostics", flags: FieldFlags::EDIT.union(FieldFlags::SCRIPT_READ) },
        }
    });
}

/// On a terrain: an open world's window, whose chunk geometry is resident only while the camera is within
/// `geometry_distance` metres of it (`TerrainData::stream_geometry`). Read when the terrain is created.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainStreamingComponent {
    /// Metres from the terrain's edge within which its geometry is resident; 0 keeps it resident always.
    pub geometry_distance: f32,
    /// Terrain-local x0, z0, x1, z1 of the window's own ground; zero is the whole terrain.
    pub own_area: glam::Vec4,
    /// A region's own ground layers, `terrain/sets/<name>` under the content root; empty is the project's `terrain`
    /// (WS1-24: Verglas wears its own ground, not Tyffir's).
    pub layer_set: String,
}

impl TerrainStreamingComponent {
    /// The folder this terrain's ground layers load from, under `terrain_root` (the content root's `terrain`): its
    /// [`layer_set_dir`]. A set name that would reach outside the sets folder is logged and the project's own layers
    /// are used.
    #[must_use]
    pub fn layer_dir(&self, terrain_root: &std::path::Path) -> std::path::PathBuf {
        layer_set_dir(terrain_root, &self.layer_set).unwrap_or_else(|error| {
            tracing::warn!(%error, "terrain layer set refused; the project's ground layers are used");
            terrain_root.to_path_buf()
        })
    }
}

/// The folder of ground layer set `layer_set` under `terrain_root`: `terrain_root/sets/<layer_set>`, or `terrain_root`
/// itself for an empty name. A set is named inside the sets folder, so an absolute path, a drive, or a `..` (each of
/// which would reach outside it) is refused with the reason.
pub fn layer_set_dir(terrain_root: &Path, layer_set: &str) -> Result<PathBuf, String> {
    use std::path::Component;
    let set = layer_set.trim();
    if set.is_empty() {
        return Ok(terrain_root.to_path_buf());
    }
    let path = Path::new(set);
    let outside = path.is_absolute()
        || path.has_root()
        || set.starts_with(['/', '\\'])
        || set.contains(':')
        || set.split(['/', '\\']).any(|part| part == "..")
        || path.components().any(|part| !matches!(part, Component::Normal(_) | Component::CurDir));
    if outside {
        return Err(format!("layer set {set:?} is not a folder name inside terrain/sets"));
    }
    Ok(terrain_root.join("sets").join(set))
}

impl Component for TerrainStreamingComponent {}

impl Default for TerrainStreamingComponent {
    fn default() -> Self {
        Self { geometry_distance: 800.0, own_area: glam::Vec4::ZERO, layer_set: String::new() }
    }
}

/// Marks an entity a cell installed. Not registered, so never saved or checkpointed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Streamed {
    /// The cell's index in the manifest.
    pub cell: u32,
}

impl Component for Streamed {}

/// One cell of a manifest.
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct CellEntry {
    /// The cell's scene, relative to the manifest.
    pub file: PathBuf,
    /// Lowest XZ corner of what the cell holds, world metres.
    pub min: [f32; 2],
    /// Highest XZ corner.
    pub max: [f32; 2],
    /// Its own load distance, for cells of big things seen from afar.
    #[serde(default)]
    pub reach: Option<f32>,
}

#[derive(serde::Deserialize)]
struct ManifestFile {
    version: u32,
    cells: Vec<CellEntry>,
}

/// Read a manifest; cell files come back resolved against its folder.
pub fn read_manifest(path: &Path) -> Result<Vec<CellEntry>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let file: ManifestFile = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if file.version != 1 {
        return Err(format!("{}: manifest version {} is not 1", path.display(), file.version));
    }
    let folder = path.parent().unwrap_or(Path::new("."));
    Ok(file.cells.into_iter().map(|cell| CellEntry { file: folder.join(&cell.file), ..cell }).collect())
}

/// Distance in XZ from `point` to a cell's bounds (0 inside).
#[must_use]
pub fn distance_to(cell: &CellEntry, point: [f32; 2]) -> f32 {
    let dx = (cell.min[0] - point[0]).max(point[0] - cell.max[0]).max(0.0);
    let dz = (cell.min[1] - point[1]).max(point[1] - cell.max[1]).max(0.0);
    dx.hypot(dz)
}

/// Which cells to start loading (nearest first) and which to unload, for a view at `focus`.
#[must_use]
pub fn plan(
    cells: &[CellEntry],
    focus: [f32; 2],
    radius: f32,
    margin: f32,
    is_resident: impl Fn(usize) -> bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut load = Vec::new();
    let mut unload = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        let distance = distance_to(cell, focus);
        let reach = cell.reach.unwrap_or(radius);
        if is_resident(index) {
            if distance > reach + margin {
                unload.push(index);
            }
        } else if distance <= reach {
            load.push((distance, index));
        }
    }
    load.sort_by(|a, b| a.0.total_cmp(&b.0));
    (load.into_iter().map(|(_, index)| index).collect(), unload)
}

/// [`plan`] for the view and the points held ([`hold`]), each held point with its own radius (`holds`: point and
/// radius). A cell loads when it is within reach of the view (its own `reach`, else `radius`) or within a hold's
/// radius, nearest first; it unloads only once it is past reach and `margin` from the view and every hold.
#[must_use]
pub fn plan_many(
    cells: &[CellEntry],
    view: [f32; 2],
    radius: f32,
    margin: f32,
    holds: &[([f32; 2], f32)],
    is_resident: impl Fn(usize) -> bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut load = Vec::new();
    let mut unload = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        // How far inside (negative) or outside its reach the cell is, from the view and from each hold.
        let view_reach = cell.reach.unwrap_or(radius);
        let view_distance = distance_to(cell, view);
        let mut over = view_distance - view_reach;
        let mut nearest = view_distance;
        for &(at, hold_radius) in holds {
            let distance = distance_to(cell, at);
            over = over.min(distance - hold_radius);
            nearest = nearest.min(distance);
        }
        if is_resident(index) {
            if over > margin {
                unload.push(index);
            }
        } else if over <= 0.0 {
            load.push((nearest, index));
        }
    }
    load.sort_by(|a, b| a.0.total_cmp(&b.0));
    (load.into_iter().map(|(_, index)| index).collect(), unload)
}

/// The most places held at once (DUAL_PROTAGONISTS needs two: a joint mission's two men). Each adds a disc of cells,
/// and a region's first fills already reach the GPU's memory.
pub const MAX_HOLDS: usize = 2;
/// A hold's radius when none is given, metres: enough to land a camera on (the view's own is the component's
/// `load_radius`, 600 m by default).
pub const HOLD_RADIUS: f32 = 250.0;

/// Points other than the view whose cells stay loaded, by key, and a pinned view, kept with the world
/// (DUAL_PROTAGONISTS DP-2: the switch's destination streams while the camera lifts; both men's cells stay in during
/// a joint mission).
#[derive(Default)]
pub(crate) struct Holds {
    held: BTreeMap<String, Hold>,
    /// Where the streamer takes the view from instead of the camera ([`pin_view`]).
    pinned: Option<glam::Vec3>,
    refused: bool,
}

#[derive(Clone, Copy)]
struct Hold {
    at: glam::Vec3,
    radius: f32,
    ready: bool,
}

impl Holds {
    /// The held points and their radii, for [`plan_many`].
    pub(crate) fn discs(&self) -> Vec<([f32; 2], f32)> {
        self.held.values().map(|hold| ([hold.at.x, hold.at.z], hold.radius)).collect()
    }

    pub(crate) fn pinned(&self) -> Option<glam::Vec3> {
        self.pinned
    }

    /// After a frame's planning: a hold is ready when every cell within its radius is loaded.
    pub(crate) fn mark_ready(&mut self, cells: &[CellEntry], loaded: impl Fn(usize) -> bool) {
        for hold in self.held.values_mut() {
            let at = [hold.at.x, hold.at.z];
            hold.ready =
                cells.iter().enumerate().filter(|(_, cell)| distance_to(cell, at) <= hold.radius).all(|(index, _)| loaded(index));
        }
    }
}

fn with_holds<T>(world: &mut somnium_ecs::World, f: impl FnOnce(&mut Holds) -> T) -> T {
    let mut holds: Holds = world.take_cache();
    let result = f(&mut holds);
    world.put_cache(holds);
    result
}

/// Keep the cells within [`HOLD_RADIUS`] of `at` loaded under `key`, as if the view were there too, until [`release`].
/// See [`hold_within`].
pub fn hold(world: &mut somnium_ecs::World, key: &str, at: glam::Vec3) -> bool {
    hold_within(world, key, at, HOLD_RADIUS)
}

/// Keep the cells within `radius` of `at` loaded under `key` (the switch's destination, the other protagonist) until
/// [`release`]. Holding a key again moves it; a hold moved along the ground is not ready until its new cells are.
/// Refused (false, logged once) when [`MAX_HOLDS`] other keys are held.
pub fn hold_within(world: &mut somnium_ecs::World, key: &str, at: glam::Vec3, radius: f32) -> bool {
    with_holds(world, |holds| {
        if !holds.held.contains_key(key) && holds.held.len() >= MAX_HOLDS {
            if !holds.refused {
                tracing::warn!(key, held = holds.held.len(), "cell streaming: a hold past the cap is ignored");
                holds.refused = true;
            }
            return false;
        }
        let entry = holds.held.entry(key.to_owned()).or_insert(Hold { at, radius, ready: false });
        let moved = (entry.at.x, entry.at.z) != (at.x, at.z) || entry.radius != radius;
        *entry = Hold { at, radius, ready: entry.ready && !moved };
        true
    })
}

/// Stop holding `key`: its cells go once the view and the other holds are far from them.
pub fn release(world: &mut somnium_ecs::World, key: &str) {
    with_holds(world, |holds| {
        holds.held.remove(key);
    });
}

/// Whether every cell within `key`'s radius is loaded (as of the streamer's last frame). False for a key not held.
pub fn hold_ready(world: &mut somnium_ecs::World, key: &str) -> bool {
    with_holds(world, |holds| holds.held.get(key).is_some_and(|hold| hold.ready))
}

/// The points held, for the systems that think by distance (combat's brain LOD, the life sim's focus): in a joint
/// mission the far man's half of the world must think too, not only stream.
pub fn held_points(world: &mut somnium_ecs::World) -> Vec<glam::Vec3> {
    with_holds(world, |holds| holds.held.values().map(|hold| hold.at).collect())
}

/// Stream from `view` instead of the camera (`None`: the camera again). The switch pins the view at the place it
/// leaves while the camera is up, so the crossing starts no readers for the land beneath it.
pub fn pin_view(world: &mut somnium_ecs::World, view: Option<glam::Vec3>) {
    with_holds(world, |holds| holds.pinned = view);
}

/// What a cell's reader sends, in order: every source it needs that was not uploaded when it started, decoded and
/// prepared off the main thread, then the cell itself. The channel holds one piece, so a reader is never more than
/// one decoded source ahead of the uploads.
pub(crate) enum Piece {
    Source {
        /// As `ImportedMesh.source` resolves in the project (not canonical).
        path: PathBuf,
        parts: HashSet<usize>,
        scene: somnium_asset::LoadedScene,
        prepared: somnium_renderer::PreparedScene,
    },
    /// The parsed cell, then each of its terrains' sidecars by saved id, read and baked (none when it is not drawn).
    Cell(Result<serde_json::Value, String>, Vec<(u32, Result<somnium_renderer::terrain::TerrainSidecar, String>)>),
}

/// A renderer terrain a streamed tile holds or held, with what another tile must match to reuse it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TerrainSlot {
    /// The renderer's terrain id.
    pub(crate) id: u32,
    pub(crate) desc: somnium_renderer::terrain::TerrainDescriptor,
    /// The ground layer set it wears now: the folder it was made with, or the last one
    /// `SomniumRenderer::set_terrain_layers` dressed it in.
    pub(crate) asset_dir: PathBuf,
}

/// Take a retired slot made for `desc`, the most recently retired first: one already wearing `asset_dir`'s ground
/// layers when there is one, else one wearing another set's, whose layers the caller swaps (it then compares the
/// slot's `asset_dir`). A virtual-textured slot keeps the set it was made with, so it goes only to its own set.
pub(crate) fn take_free_slot(
    free: &mut Vec<TerrainSlot>,
    desc: &somnium_renderer::terrain::TerrainDescriptor,
    asset_dir: &Path,
) -> Option<TerrainSlot> {
    let at = free
        .iter()
        .rposition(|slot| slot.desc == *desc && slot.asset_dir == asset_dir)
        .or_else(|| free.iter().rposition(|slot| slot.desc == *desc && !desc.virtual_texturing))?;
    Some(free.remove(at))
}

/// Where a tile's terrain goes ([`choose_slot`]).
#[derive(Debug, PartialEq)]
pub(crate) enum SlotChoice {
    /// A retired slot of its shape and ground.
    Reuse(TerrainSlot),
    /// A retired slot of its shape wearing another set's ground layers: they are swapped for the tile's
    /// (`SomniumRenderer::set_terrain_layers`) before it adopts the tile.
    Swap(TerrainSlot),
    /// A new renderer terrain.
    New,
    /// Nowhere: no retired slot fits and the renderer's terrain material pool is full. A new terrain would share
    /// material entry 0 with another terrain and shade with that terrain's ground, so the tile is not drawn instead.
    Full,
}

/// A retired slot of `desc`'s shape ([`take_free_slot`]), wearing `asset_dir`'s ground or, failing that, another
/// set's; else a new terrain while the terrain material pool has `material_room` entries left for it; else none.
/// Since any retired slot of the shape will do, the slots made only ever number the most tiles live at once, however
/// many sets the world wears.
pub(crate) fn choose_slot(
    free: &mut Vec<TerrainSlot>,
    desc: &somnium_renderer::terrain::TerrainDescriptor,
    asset_dir: &Path,
    material_room: u32,
) -> SlotChoice {
    match take_free_slot(free, desc, asset_dir) {
        Some(slot) if slot.asset_dir == asset_dir => SlotChoice::Reuse(slot),
        Some(slot) => SlotChoice::Swap(slot),
        None if material_room > 0 => SlotChoice::New,
        None => SlotChoice::Full,
    }
}

/// The next retired slot to dress in a live ground set, and that set. A retired slot wearing a set no live slot
/// wears would keep that set's layer arrays (about 213 MiB of BC7 a region) in memory while it waits, and a region
/// left behind may never take it back, so it takes the set the most live slots wear (the region the view is in;
/// the first folder of those that tie), ready for that region's next tile. `None` while every retired slot wears a
/// live set, or while no slot is live to say which (a teleport's first frames: its first tile takes a retired slot
/// of any set, and the rest follow). A virtual-textured slot cannot change its ground and is left.
pub(crate) fn next_rewear(live: &[&TerrainSlot], free: &[TerrainSlot]) -> Option<(usize, PathBuf)> {
    let mut worn: BTreeMap<&Path, usize> = BTreeMap::new();
    for slot in live {
        *worn.entry(slot.asset_dir.as_path()).or_default() += 1;
    }
    let target = worn.iter().max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))?.0;
    let index =
        free.iter().rposition(|slot| !slot.desc.virtual_texturing && !worn.contains_key(slot.asset_dir.as_path()))?;
    Some((index, target.to_path_buf()))
}

/// Why `entity` may not be sculpted, painted or foliage-painted in place, or `None`. A streamed cell's entity is
/// read-only: Save never writes its edits, and once its cell unloads its terrain slot goes to another tile, where an
/// undo recorded against the slot would land.
pub(crate) fn read_only_reason(world: &somnium_ecs::World, entity: Entity) -> Option<&'static str> {
    world.get::<Streamed>(entity).is_some().then_some("A streamed tile is read-only: open its cell scene to edit it.")
}

/// A cell whose ground (a terrain sidecar) did not read: a sidecar being rewritten, or held by a sync, reads a moment
/// later.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GroundRetry {
    /// Why the last read failed.
    pub(crate) error: String,
    /// Reads that failed in a row.
    pub(crate) failures: u32,
    /// When the next read may start.
    pub(crate) next: std::time::Instant,
}

/// Failed ground reads in a row after which a hold stops waiting for the cell ([`CellStreamer::settled`]), so a
/// missing sidecar never strands a hold. Reads go on meanwhile.
pub(crate) const GROUND_TRIES: u32 = 4;
/// The longest pause between ground reads, seconds (they start at one second and double).
pub(crate) const GROUND_RETRY_MAX_SECONDS: u64 = 30;

/// The streamer's state, owned by the app.
#[derive(Default)]
pub struct CellStreamer {
    /// The component's entity and the manifest it named: a change of either starts over.
    pub(crate) key: Option<(Entity, PathBuf)>,
    pub(crate) cells: Vec<CellEntry>,
    /// Cell index -> the entities it installed.
    pub(crate) loaded: HashMap<usize, Vec<Entity>>,
    pub(crate) reading: HashMap<usize, Receiver<Piece>>,
    /// Cells that failed to read or install: not retried until the manifest changes.
    pub(crate) failed: HashMap<usize, String>,
    /// The first fill (the view's cells taken in whole) has happened for this manifest.
    pub(crate) filled: bool,
    /// Canonical sources each loaded cell draws, and how many loaded cells draw each source.
    pub(crate) cell_sources: HashMap<usize, Vec<PathBuf>>,
    pub(crate) source_users: HashMap<PathBuf, u32>,
    /// Sources no loaded cell draws, since when: released once they have been cold for `COLD_SECONDS`.
    pub(crate) cold: HashMap<PathBuf, std::time::Instant>,
    /// When cold sources were last released (batched: each release waits for the GPU).
    pub(crate) last_eviction: Option<std::time::Instant>,
    /// Cell index -> the renderer terrains its tiles hold.
    pub(crate) cell_terrains: HashMap<usize, Vec<TerrainSlot>>,
    /// Terrains retired with their cells, kept for the next tile of the same shape.
    pub(crate) free_terrains: Vec<TerrainSlot>,
    /// The renderer's `scene_resets` the two above belong to: a scene reset drops every terrain by itself.
    pub(crate) terrain_resets: u64,
    /// Cells whose ground did not read, read again after a pause ([`Self::ground_failed`]). Not in `failed`.
    pub(crate) ground_retries: HashMap<usize, GroundRetry>,
}

/// How long a streamed mesh no loaded cell uses stays uploaded (coming back soon costs nothing).
pub(crate) const COLD_SECONDS: u64 = 30;

/// Most cells read at once.
pub(crate) const READERS: usize = 4;

/// Each source an entity list draws (`somnium.ImportedMesh` source -> node ordinals), from a scene document.
#[must_use]
pub fn sources_of(document: &serde_json::Value) -> BTreeMap<String, HashSet<usize>> {
    let mut sources: BTreeMap<String, HashSet<usize>> = BTreeMap::new();
    for entity in document.get("entities").and_then(serde_json::Value::as_array).into_iter().flatten() {
        let Some(fields) = entity.pointer("/components/somnium.ImportedMesh/fields") else {
            continue;
        };
        if let (Some(source), Some(node)) =
            (fields.get("source").and_then(serde_json::Value::as_str), fields.get("node").and_then(serde_json::Value::as_u64))
        {
            sources.entry(source.to_owned()).or_default().insert(node as usize);
        }
    }
    sources
}

/// A terrain a cell holds, as its document saves it: what its reader needs to read and bake its sidecar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainCell {
    /// The id it was saved under, which names its sidecar (`{cell}.terrain{id}.bin`).
    pub saved_id: u32,
    /// What `read_sidecar` expects: vertices along X and Z, then splat texels along X and Z.
    pub dims: [u32; 4],
    /// Metres between neighbouring heights.
    pub cell_size: f32,
    /// Multiplier on the raw heights.
    pub height_scale: f32,
}

/// Each terrain (`somnium.Terrain`) a scene document holds, once per saved id. Fields it leaves out take the
/// component's defaults, as a scene load gives them. A terrain the renderer could not make is left out; its install
/// reports it.
#[must_use]
pub fn terrains_of(document: &serde_json::Value) -> Vec<TerrainCell> {
    let mut terrains: Vec<TerrainCell> = Vec::new();
    for entity in document.get("entities").and_then(serde_json::Value::as_array).into_iter().flatten() {
        let Some(fields) = entity.pointer("/components/somnium.Terrain/fields") else {
            continue;
        };
        let whole = |name: &str, default: u32| {
            fields.get(name).and_then(serde_json::Value::as_u64).and_then(|v| u32::try_from(v).ok()).unwrap_or(default)
        };
        let real =
            |name: &str, default: f32| fields.get(name).and_then(serde_json::Value::as_f64).map_or(default, |v| v as f32);
        let defaults = crate::TerrainComponent::default();
        let component = crate::TerrainComponent {
            terrain_id: whole("terrain_id", defaults.terrain_id),
            chunk_cells: whole("chunk_cells", defaults.chunk_cells),
            grid_x: whole("grid_x", defaults.grid_x),
            grid_z: whole("grid_z", defaults.grid_z),
            cell_size: real("cell_size", defaults.cell_size),
            height_scale: real("height_scale", defaults.height_scale),
            ..defaults
        };
        let Some(desc) = terrain_descriptor(&component) else {
            continue;
        };
        if terrains.iter().any(|terrain| terrain.saved_id == component.terrain_id) {
            continue;
        }
        let [grid_x, grid_z] = desc.grid_size;
        terrains.push(TerrainCell {
            saved_id: component.terrain_id,
            dims: [desc.total_vertices_x(), desc.total_vertices_z(), grid_x * desc.chunk_cells, grid_z * desc.chunk_cells],
            cell_size: desc.cell_size,
            height_scale: desc.height_scale,
        });
    }
    terrains
}

/// The renderer descriptor a saved terrain asks for, or `None` when the renderer could not make it: a chunk size the
/// LOD scheme cannot split, a zero or overflowing grid, or a cell size or height scale that is not a positive number.
pub(crate) fn terrain_descriptor(
    component: &crate::TerrainComponent,
) -> Option<somnium_renderer::terrain::TerrainDescriptor> {
    let desc = somnium_renderer::terrain::TerrainDescriptor {
        chunk_cells: component.chunk_cells,
        grid_size: [component.grid_x, component.grid_z],
        cell_size: component.cell_size,
        height_scale: component.height_scale,
        virtual_texturing: component.virtual_texturing,
        ..Default::default()
    };
    let valid_axis = |chunks: u32| {
        chunks > 0 && chunks.checked_mul(desc.chunk_cells).and_then(|cells| cells.checked_add(1)).is_some()
    };
    let valid = desc.chunk_cells.is_power_of_two()
        && desc.chunk_cells >= (1 << somnium_renderer::terrain::mesh::MAX_TERRAIN_LOD)
        && valid_axis(component.grid_x)
        && valid_axis(component.grid_z)
        && desc.cell_size.is_finite()
        && desc.cell_size > 0.0
        && desc.height_scale.is_finite()
        && desc.height_scale > 0.0
        && desc.total_vertices_x().checked_mul(desc.total_vertices_z()).is_some();
    valid.then_some(desc)
}

impl CellStreamer {
    /// Entities installed by every loaded cell.
    #[must_use]
    pub fn streamed_entities(&self) -> usize {
        self.loaded.values().map(Vec::len).sum()
    }

    /// Cell `index` read without its ground (`error`) at `now`: it is read again after a pause that starts at one
    /// second and doubles with each failure in a row, up to [`GROUND_RETRY_MAX_SECONDS`].
    pub(crate) fn ground_failed(&mut self, index: usize, error: String, now: std::time::Instant) {
        let failures = self.ground_retries.get(&index).map_or(0, |retry| retry.failures).saturating_add(1);
        let pause = (1u64 << (failures - 1).min(16)).min(GROUND_RETRY_MAX_SECONDS);
        let next = now + std::time::Duration::from_secs(pause);
        self.ground_retries.insert(index, GroundRetry { error, failures, next });
    }

    /// Whether cell `index` may start a read at `now`: not failed for good, and not pausing before its ground is read
    /// again.
    pub(crate) fn may_read(&self, index: usize, now: std::time::Instant) -> bool {
        !self.failed.contains_key(&index) && self.ground_retries.get(&index).is_none_or(|retry| now >= retry.next)
    }

    /// Whether a hold counts cell `index` as in: loaded, or failed for good, so a broken file never strands it. A cell
    /// whose ground is to be read again is not in until [`GROUND_TRIES`] reads of it have failed in a row.
    pub(crate) fn settled(&self, index: usize) -> bool {
        self.loaded.contains_key(&index)
            || self.failed.contains_key(&index)
            || self.ground_retries.get(&index).is_some_and(|retry| retry.failures >= GROUND_TRIES)
    }

    /// Whether renderer terrain `id` is a streamed tile's slot, live or retired, as of the renderer's `resets`
    /// (`SomniumRenderer::scene_resets`). A scene's own terrain must never take one over.
    pub(crate) fn owns_terrain(&self, id: u32, resets: u64) -> bool {
        resets == self.terrain_resets
            && self.cell_terrains.values().flatten().chain(&self.free_terrains).any(|slot| slot.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_names_its_regions_ground_layers() {
        let root = std::path::Path::new("assets/terrain");
        let mut window = TerrainStreamingComponent::default();
        assert_eq!(window.layer_dir(root), root, "the project's layers by default");
        window.layer_set = "verglas".into();
        assert_eq!(window.layer_dir(root), root.join("sets").join("verglas"));
        window.layer_set = "  ".into();
        assert_eq!(window.layer_dir(root), root);
    }

    fn cell(min: [f32; 2], max: [f32; 2], reach: Option<f32>) -> CellEntry {
        CellEntry { file: PathBuf::from("c.json"), min, max, reach }
    }

    #[test]
    fn near_cells_load_nearest_first_and_far_ones_unload_past_the_margin() {
        let cells = vec![
            cell([0.0, 0.0], [256.0, 256.0], None),       // the view is inside
            cell([800.0, 0.0], [1056.0, 256.0], None),    // 672 m off: beyond 600
            cell([400.0, 0.0], [656.0, 256.0], None),     // 272 m off
            cell([3000.0, 0.0], [4024.0, 1024.0], Some(3000.0)), // a big thing with its own reach
        ];
        let (load, unload) = plan(&cells, [128.0, 128.0], 600.0, 100.0, |_| false);
        assert_eq!(load, vec![0, 2, 3]);
        assert!(unload.is_empty());
        // Resident cells unload only past reach + margin: 672 m stays, the 3 km reach keeps the big cell.
        let (load, unload) = plan(&cells, [128.0, 128.0], 600.0, 100.0, |_| true);
        assert!(load.is_empty());
        assert!(unload.is_empty());
        // 650 m from the first cell keeps it (inside 700); the rest are past their reach + margin.
        let (_, unload) = plan(&cells, [-650.0, 128.0], 600.0, 100.0, |_| true);
        assert_eq!(unload, vec![1, 2, 3]);
    }

    #[test]
    fn a_held_point_streams_its_own_small_disc_beside_the_view() {
        // Two towns 5 km apart: the view in one, a hold in the other.
        let cells = vec![
            cell([0.0, 0.0], [256.0, 256.0], None),
            cell([5000.0, 0.0], [5256.0, 256.0], None),           // the hold is inside
            cell([2500.0, 0.0], [2756.0, 256.0], None),           // between: near neither
            cell([5656.0, 0.0], [5912.0, 256.0], None),           // 528 m from the hold: inside a view's 600, not a hold's 250
            cell([6000.0, 0.0], [6256.0, 256.0], Some(3000.0)),   // a landmark: its long reach is the view's, not a hold's
        ];
        let (view, held) = ([128.0, 128.0], ([5128.0, 128.0], HOLD_RADIUS));
        assert_eq!(
            plan_many(&cells, view, 600.0, 100.0, &[], |_| false),
            plan(&cells, view, 600.0, 100.0, |_| false),
            "no holds: the plan it always was"
        );
        let (load, _) = plan_many(&cells, view, 600.0, 100.0, &[held], |_| false);
        assert_eq!(load, vec![0, 1], "the other town's own cell, not its far edge, the landmark or the land between");
        let (_, unload) = plan_many(&cells, view, 600.0, 100.0, &[held], |_| true);
        assert_eq!(unload, vec![2, 3, 4], "only what is near the view or inside the hold stays");
    }

    #[test]
    fn holds_are_capped_ready_once_in_moved_afresh_and_the_view_can_be_pinned() {
        let mut world = somnium_ecs::World::new();
        let cells = vec![cell([5000.0, 0.0], [5256.0, 256.0], None), cell([5300.0, 0.0], [5556.0, 256.0], None)];
        let rhun = glam::Vec3::new(5280.0, 4.0, 128.0);
        assert!(!hold_ready(&mut world, "rhun"), "not held: not ready");
        assert!(hold(&mut world, "rhun", rhun));
        assert!(hold(&mut world, "caelan", glam::Vec3::ZERO));
        assert!(!hold(&mut world, "switch", glam::Vec3::X), "a third is refused");
        assert!(hold(&mut world, "rhun", rhun), "a held key may still move");
        // The streamer's frame: the two cells within 250 m of him, one in and then both.
        let mark = |world: &mut somnium_ecs::World, loaded: &[usize]| {
            let mut holds: Holds = world.take_cache();
            holds.mark_ready(&cells, |i| loaded.contains(&i));
            world.put_cache(holds);
        };
        mark(&mut world, &[0]);
        assert!(!hold_ready(&mut world, "rhun"), "half in");
        mark(&mut world, &[0, 1]);
        assert!(hold_ready(&mut world, "rhun"));
        assert!(hold(&mut world, "rhun", rhun + glam::Vec3::Y * 5.0) && hold_ready(&mut world, "rhun"), "height does not move a hold");
        assert!(hold(&mut world, "rhun", glam::Vec3::new(9000.0, 0.0, 0.0)) && !hold_ready(&mut world, "rhun"), "moved: not yet");
        // The held points, for the brains and the crowds; the pinned view.
        assert_eq!(held_points(&mut world).len(), 2);
        pin_view(&mut world, Some(glam::Vec3::new(1.0, 2.0, 3.0)));
        let holds: Holds = world.take_cache();
        assert_eq!(holds.pinned(), Some(glam::Vec3::new(1.0, 2.0, 3.0)));
        world.put_cache(holds);
        release(&mut world, "rhun");
        release(&mut world, "caelan");
        assert!(held_points(&mut world).is_empty());
        assert!(hold(&mut world, "switch", glam::Vec3::X), "room again");
    }

    #[test]
    fn a_cell_names_each_source_once_with_the_parts_it_draws() {
        let document = serde_json::json!({"version": 4, "entities": [
            {"components": {"somnium.ImportedMesh": {"fields": {"source": "a.gltf", "node": 2}}}},
            {"components": {"somnium.ImportedMesh": {"fields": {"source": "a.gltf", "node": 5}}}},
            {"components": {"somnium.ImportedMesh": {"fields": {"source": "b.gltf", "node": 0}}}},
            {"components": {"somnium.Name": {"fields": {"value": "no mesh"}}}},
        ]});
        let sources = sources_of(&document);
        assert_eq!(sources.len(), 2);
        assert_eq!(sources["a.gltf"], HashSet::from([2, 5]));
        assert_eq!(sources["b.gltf"], HashSet::from([0]));
    }

    #[test]
    fn a_cell_names_each_terrain_once_with_the_sidecar_size_it_reads() {
        let document = serde_json::json!({"version": 4, "entities": [
            {"components": {"somnium.Terrain": {"version": 2, "fields": {
                "terrain_id": 0, "chunk_cells": 64, "grid_x": 8, "grid_z": 8, "cell_size": 2.0, "height_scale": 1.0}}}},
            // The same saved id again: one terrain, as a scene open makes it.
            {"components": {"somnium.Terrain": {"version": 2, "fields": {"terrain_id": 0, "grid_x": 8, "grid_z": 8}}}},
            // Missing fields take the component's defaults: 64-cell chunks, one chunk along Z, 1 m cells.
            {"components": {"somnium.Terrain": {"version": 1, "fields": {"terrain_id": 3, "grid_x": 2}}}},
            // Twelve cells per chunk is no power of two: the renderer could not make it.
            {"components": {"somnium.Terrain": {"version": 2, "fields": {"terrain_id": 5, "chunk_cells": 12}}}},
            {"components": {"somnium.Name": {"fields": {"value": "no terrain"}}}},
        ]});
        assert_eq!(
            terrains_of(&document),
            vec![
                TerrainCell { saved_id: 0, dims: [513, 513, 512, 512], cell_size: 2.0, height_scale: 1.0 },
                TerrainCell { saved_id: 3, dims: [129, 65, 128, 64], cell_size: 1.0, height_scale: 1.0 },
            ]
        );
        assert!(terrains_of(&serde_json::json!({"version": 4, "entities": []})).is_empty());
    }

    #[test]
    fn a_layer_set_is_a_folder_name_inside_the_sets_folder() {
        let root = Path::new("assets/terrain");
        assert_eq!(layer_set_dir(root, ""), Ok(root.to_path_buf()), "the project's own layers");
        assert_eq!(layer_set_dir(root, " \t"), Ok(root.to_path_buf()));
        assert_eq!(layer_set_dir(root, "verglas_south"), Ok(root.join("sets").join("verglas_south")));
        assert_eq!(layer_set_dir(root, " verglas "), Ok(root.join("sets").join("verglas")), "trimmed");
        assert_eq!(layer_set_dir(root, "south/mirage"), Ok(root.join("sets").join("south/mirage")), "a folder within");
        let outside = [
            "..",
            "../terrain_verglas",
            "south/../../..",
            "south\\..\\..\\secrets",
            "/etc",
            "\\Windows",
            "C:/Windows",
            "C:relative",
            "\\\\server\\share",
        ];
        for set in outside {
            let refused = layer_set_dir(root, set);
            assert!(refused.as_ref().is_err_and(|e| e.contains("terrain/sets")), "{set:?}: {refused:?}");
        }
        // A window naming one wears the project's layers instead, rather than reading from anywhere.
        let window = TerrainStreamingComponent { layer_set: "../../../elsewhere".into(), ..Default::default() };
        assert_eq!(window.layer_dir(root), root);
    }

    #[test]
    fn a_retired_slot_goes_to_its_own_set_first_then_to_any_set_of_its_shape() {
        use somnium_renderer::terrain::TerrainDescriptor;
        let tile = TerrainDescriptor { chunk_cells: 64, grid_size: [8, 8], cell_size: 2.0, ..Default::default() };
        let slot = |id, desc, dir: &str| TerrainSlot { id, desc, asset_dir: PathBuf::from(dir) };
        let (ground, verglas) = ("assets/terrain", "assets/terrain/sets/verglas");
        let mut free = vec![
            slot(1, tile, ground),
            slot(2, TerrainDescriptor { height_scale: 2.0, ..tile }, ground),
            slot(3, tile, verglas),
            slot(4, tile, ground),
            slot(5, tile, verglas),
        ];
        let take = |free: &mut Vec<TerrainSlot>, desc: &TerrainDescriptor, dir: &str| {
            take_free_slot(free, desc, Path::new(dir)).map(|s| (s.id, s.asset_dir.to_string_lossy().into_owned()))
        };
        assert_eq!(take(&mut free, &tile, ground), Some((4, ground.into())), "its own set, the last retired first");
        assert_eq!(take(&mut free, &tile, ground), Some((1, ground.into())));
        assert_eq!(take(&mut free, &tile, ground), Some((5, verglas.into())), "then another set's, to be re-dressed");
        assert_eq!(take(&mut free, &tile, verglas), Some((3, verglas.into())));
        assert_eq!(take(&mut free, &tile, ground), None, "another height scale is no match");
        assert_eq!(take(&mut free, &TerrainDescriptor { grid_size: [16, 8], ..tile }, ground), None);
        assert_eq!(free.iter().map(|s| s.id).collect::<Vec<_>>(), vec![2], "what did not match stays");

        // A virtual-textured slot's page atlas is its own: it goes to its own set only.
        let paged = TerrainDescriptor { virtual_texturing: true, ..tile };
        let mut free = vec![slot(6, paged, verglas)];
        assert_eq!(take(&mut free, &paged, ground), None);
        assert_eq!(take(&mut free, &paged, verglas), Some((6, verglas.into())));
    }

    #[test]
    fn a_tile_gets_a_new_terrain_only_while_no_retired_slot_fits_and_the_material_pool_has_room() {
        use somnium_renderer::terrain::TerrainDescriptor;
        let tile = TerrainDescriptor { chunk_cells: 64, grid_size: [8, 8], cell_size: 2.0, ..Default::default() };
        let ground = Path::new("assets/terrain");
        let verglas = Path::new("assets/terrain/sets/verglas");
        let mut free = vec![TerrainSlot { id: 7, desc: tile, asset_dir: ground.to_path_buf() }];
        let reused = choose_slot(&mut free, &tile, ground, 0);
        assert_eq!(reused, SlotChoice::Reuse(free_slot(7, tile)), "a retired slot needs no entry");
        assert_eq!(choose_slot(&mut free, &tile, ground, 1), SlotChoice::New);
        // The seventeenth live slot with the default sixteen entries: a new terrain would shade with entry 0's ground.
        assert_eq!(choose_slot(&mut free, &tile, ground, 0), SlotChoice::Full);
        // A slot of another region's set is taken before a new terrain, with or without room, and swapped.
        for room in [0, 5] {
            let mut free = vec![free_slot(8, tile)];
            assert_eq!(choose_slot(&mut free, &tile, verglas, room), SlotChoice::Swap(free_slot(8, tile)));
            assert!(free.is_empty());
        }
        let mut free = vec![free_slot(8, tile), TerrainSlot { id: 9, desc: tile, asset_dir: verglas.to_path_buf() }];
        assert_eq!(choose_slot(&mut free, &tile, ground, 0), SlotChoice::Reuse(free_slot(8, tile)), "own set first");
    }

    #[test]
    fn a_retired_slot_of_a_region_left_behind_takes_the_live_regions_ground() {
        use somnium_renderer::terrain::TerrainDescriptor;
        let tile = TerrainDescriptor { chunk_cells: 64, grid_size: [8, 8], cell_size: 2.0, ..Default::default() };
        let slot = |id, dir: &str| TerrainSlot { id, desc: tile, asset_dir: PathBuf::from(dir) };
        let (tyffir, verglas, south) = ("assets/terrain", "assets/terrain/sets/verglas", "assets/terrain/sets/south");
        let live = [slot(1, verglas), slot(2, verglas), slot(3, tyffir)];
        let live: Vec<&TerrainSlot> = live.iter().collect();
        // Every retired slot wears a live set: nothing to do.
        assert_eq!(next_rewear(&live, &[slot(4, tyffir), slot(5, verglas)]), None);
        // The south was left behind: its retired slots take Verglas's ground (worn by the most live tiles), the last
        // retired first.
        let free = [slot(4, south), slot(5, tyffir), slot(6, south)];
        assert_eq!(next_rewear(&live, &free), Some((2, PathBuf::from(verglas))));
        assert_eq!(next_rewear(&live, &free[..2]), Some((0, PathBuf::from(verglas))));
        // No live tile (a teleport's first frames): no set to take yet.
        assert_eq!(next_rewear(&[], &free), None);
        // Sets worn by as many live tiles: the first folder of them.
        let even = [slot(1, verglas), slot(3, tyffir)];
        let even: Vec<&TerrainSlot> = even.iter().collect();
        assert_eq!(next_rewear(&even, &free), Some((2, PathBuf::from(tyffir))));
        // A virtual-textured slot keeps its own.
        let paged = TerrainSlot { desc: TerrainDescriptor { virtual_texturing: true, ..tile }, ..slot(7, south) };
        assert_eq!(next_rewear(&live, &[paged]), None);
    }

    fn free_slot(id: u32, desc: somnium_renderer::terrain::TerrainDescriptor) -> TerrainSlot {
        TerrainSlot { id, desc, asset_dir: PathBuf::from("assets/terrain") }
    }

    #[test]
    fn a_cell_whose_ground_did_not_read_is_read_again_and_a_hold_waits_for_it() {
        let mut streamer = CellStreamer::default();
        let start = std::time::Instant::now();
        let at = |seconds: u64| start + std::time::Duration::from_secs(seconds);
        // A cell that is broken for good (it did not parse) is in for a hold and never read again, as before.
        streamer.failed.insert(0, "c.json: expected value".into());
        assert!(streamer.settled(0) && !streamer.may_read(0, at(3600)));

        // A sidecar being rewritten: the cell is not in, and reads again a second later, then two, four...
        streamer.ground_failed(1, "terrain 0: truncated terrain sidecar".into(), start);
        assert!(!streamer.settled(1), "a hold does not put the player over a hole");
        assert!(!streamer.may_read(1, start) && streamer.may_read(1, at(1)));
        streamer.ground_failed(1, "terrain 0: read: sharing violation".into(), at(1));
        assert!(!streamer.may_read(1, at(2)) && streamer.may_read(1, at(3)), "the pause doubles");
        assert_eq!(streamer.ground_retries[&1].error, "terrain 0: read: sharing violation");
        streamer.ground_failed(1, "again".into(), at(3));
        assert!(!streamer.settled(1));
        // Four in a row: a missing sidecar must not strand a hold, but the reads go on, at most 30 s apart.
        streamer.ground_failed(1, "again".into(), at(7));
        assert!(streamer.settled(1) && streamer.may_read(1, at(15)));
        for failed_at in [15, 31, 61] {
            streamer.ground_failed(1, "again".into(), at(failed_at));
        }
        assert!(!streamer.may_read(1, at(90)) && streamer.may_read(1, at(91)), "thirty seconds at most");
        assert!(!streamer.failed.contains_key(&1), "a ground retry is not a failure for good");
    }

    #[test]
    fn a_streamed_entity_is_read_only_and_its_terrain_slots_are_the_streamers() {
        let mut world = somnium_ecs::World::new();
        let scene = world.spawn((TerrainStreamingComponent::default(),));
        let tile = world.spawn((TerrainStreamingComponent::default(), Streamed { cell: 3 }));
        assert_eq!(read_only_reason(&world, scene), None);
        assert!(read_only_reason(&world, tile).is_some(), "a tile's edits would never be saved");

        use somnium_renderer::terrain::TerrainDescriptor;
        let tile_desc = TerrainDescriptor::default();
        let mut streamer = CellStreamer { terrain_resets: 2, ..CellStreamer::default() };
        streamer.cell_terrains.insert(3, vec![free_slot(0, tile_desc)]);
        streamer.free_terrains.push(free_slot(4, tile_desc));
        assert!(streamer.owns_terrain(0, 2), "a live tile's slot");
        assert!(streamer.owns_terrain(4, 2), "a retired slot");
        assert!(!streamer.owns_terrain(1, 2), "a scene's own terrain");
        assert!(!streamer.owns_terrain(0, 3), "after a scene reset the recorded slots are gone");
    }

    // The engine's side of the four checks above, read from its source: the methods need a window and a GPU.

    /// `fn name`'s text in `source`, from its name to its closing brace (the first line that is four spaces and `}`).
    fn method<'a>(source: &'a str, name: &str) -> &'a str {
        let rest = source.split_once(&format!("fn {name}(")).unwrap_or_else(|| panic!("fn {name} exists")).1;
        &rest[..rest.find("\n    }").unwrap_or(rest.len())]
    }

    #[test]
    fn a_tile_slot_is_chosen_knowing_the_material_pool_s_room() {
        let adopt = method(include_str!("app_cell_streaming.rs"), "adopt_terrain");
        assert!(
            ["terrain_material_room()", "choose_slot(", "SlotChoice::Full"].iter().all(|code| adopt.contains(code)),
            "a tile past the pool's last entry would be given entry 0 and shade with another tile's ground"
        );
    }

    #[test]
    fn a_slot_of_another_set_is_re_dressed_and_a_region_left_behind_lets_its_ground_go() {
        let source = include_str!("app_cell_streaming.rs");
        let adopt = method(source, "adopt_terrain");
        let swap = adopt.split_once("SlotChoice::Swap(").expect("a retired slot of another set is taken").1;
        let swap = &swap[..swap.find("SlotChoice::New").unwrap_or(swap.len())];
        assert!(swap.contains("set_terrain_layers("), "a tile would shade with the ground its slot wore last");
        assert!(
            method(source, "update_cell_streaming").contains("self.rewear_retired_slot()"),
            "retired slots would keep a region's layer arrays after its last tile went"
        );
        let rewear = method(source, "rewear_retired_slot");
        assert!(rewear.contains("next_rewear(") && rewear.contains("set_terrain_layers("));
    }

    #[test]
    fn a_ground_read_failure_is_retried_and_holds_wait_through_it() {
        let source = include_str!("app_cell_streaming.rs");
        let pump = method(source, "pump_cell_reads");
        assert!(pump.contains("ground_failed("), "a sidecar read error fails the cell for good");
        let schedule = method(source, "schedule_cells");
        assert!(schedule.contains("may_read(index"), "a cell waiting out a ground retry is read again every frame");
        assert!(schedule.contains("streamer.settled(index)"), "a hold is ready over a tile whose ground is retried");
    }

    #[test]
    fn streamed_tiles_are_not_sculpted_painted_or_foliage_painted() {
        assert!(method(include_str!("app.rs"), "persona_target_reason").contains("read_only_reason("));
        let mcp = include_str!("app_authoring_specialized.rs");
        let arms = [("\"terrain_stroke\" => {", "terrain_stroke = Some("), ("\"foliage_stroke\" => {", "push_silent(")];
        for (arm, before) in arms {
            let arm = mcp.split_once(arm).unwrap_or_else(|| panic!("{arm} exists")).1;
            let arm = &arm[..arm.find(before).unwrap_or_else(|| panic!("{before} follows"))];
            assert!(arm.contains("read_only_reason("), "an undo recorded against a tile's slot lands in the next tile");
        }
    }

    #[test]
    fn a_scene_terrain_never_keeps_a_tile_s_slot() {
        let restore = method(include_str!("app.rs"), "restore_terrain_gpu");
        let existed = restore.split_once("let existed").expect("restore_terrain_gpu keeps existing terrains").1;
        let existed = &existed[..existed.find(';').unwrap_or(existed.len())];
        assert!(existed.contains("owns_terrain("), "a prefab terrain saved with a tile's id would bind to the tile");
    }

    #[test]
    fn a_manifest_resolves_its_cells_beside_it() {
        let folder = std::env::temp_dir().join(format!("cell_streaming_{}", std::process::id()));
        std::fs::create_dir_all(&folder).unwrap();
        let path = folder.join("world.cells.json");
        std::fs::write(
            &path,
            r#"{"version": 1, "cells": [{"file": "cells/0_0.scene.json", "min": [0, 0], "max": [256, 256]}]}"#,
        )
        .unwrap();
        let cells = read_manifest(&path).unwrap();
        assert_eq!(cells[0].file, folder.join("cells/0_0.scene.json"));
        assert_eq!(cells[0].reach, None);
        std::fs::write(&path, r#"{"version": 2, "cells": []}"#).unwrap();
        assert!(read_manifest(&path).is_err());
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&folder);
    }
}
