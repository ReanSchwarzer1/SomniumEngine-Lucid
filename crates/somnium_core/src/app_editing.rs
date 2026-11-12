//! Level editing on generated worlds: what a clicked object is and where it came from (the selection card), an edit
//! layer that outlives the generator scripts, object-first and triangle-accurate clicks, Drop to Ground, Align to
//! Surface, Find Floating Objects, hide and isolate, Find in Outliner and Save Scene As.
//!
//! The pure halves live in `scene_edits.rs` (the layer: apply and record) and `editor_ground.rs` (BVHs and the
//! ground rules); this file wires them to the live world, the renderer and the panels.

use super::*;
use crate::editor_ground::{self, Ground, Hit, MeshShape, MeshSurface, TriangleBvh};
use crate::scene_edits::{self, EditOp, Origin, SceneEdits, Scope};
use serde_json::Value;
use somnium_ecs::{Entity, PersistentId};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A click picks the part, not the object, when the object has more parts than this.
const CONTAINER_PARTS: usize = 400;
/// Find Floating Objects looks this far round the camera.
const FLOAT_RADIUS: f32 = 80.0;
/// Meshes with more triangles than this are clicked and stood on by their box.
const BVH_TRIANGLE_LIMIT: u32 = 3_000_000;
/// Painted copies (foliage kept with its ground, not entities) are looked for this far along a click.
const PAINTED_REACH: f32 = 600.0;
/// The click ray is walked in steps this long, each visiting the foliage cells within this distance of it.
const PAINTED_STEP: f32 = 24.0;
/// Copies whose box the ray enters are tested by their triangles, nearest first, at most this many.
const PAINTED_REFINE: usize = 32;
/// A painted kind smaller than this (box diagonal, metres, at scale 1) never hides an object behind it: grass and
/// litter cover the ground everywhere, and a click meant for a rock should find the rock.
const PAINTED_SMALL: f32 = 1.5;
/// A copy is still found this far past where a click meets the ground (its foot sits in the slope); past that it is
/// behind the hill.
const GROUND_MARGIN: f32 = 0.5;
/// The ground-cover card counts every copy this close to the clicked one, by kind.
const COVER_NEARBY: f32 = 2.0;
/// The card checks the surface of a selection up to this many triangles; Check Meshes does any size.
const CARD_SHAPE_LIMIT: u64 = 500_000;
/// Words that make a model file rock-like: a rock or cliff must be closed on every side it shows.
const ROCK_WORDS: [&str; 20] = [
    "rock", "cliff", "boulder", "stone", "scree", "tor", "crag", "outcrop", "slab", "talus", "ledge", "granite",
    "limestone", "basalt", "chalk", "flint", "pebble", "rubble", "bluff", "monolith",
];
/// Words that make a model file tree-like, for the census's list of trees from outside the painted palette.
const TREE_WORDS: [&str; 22] = [
    "tree", "oak", "pine", "cypress", "fig", "olive", "chestnut", "beech", "maple", "ash", "alder", "willow",
    "poplar", "plane", "holm", "juniper", "carob", "hornbeam", "shrub", "bush", "palm", "laurel",
];

/// Take out of `copies` every one a [`scene_edits::PaintedEdit`] on `ground` names (same kind, within
/// [`scene_edits::PAINTED_MATCH`]); how many went.
fn hide_painted(
    edits: &[scene_edits::PaintedEdit],
    ground: &str,
    copies: &mut Vec<somnium_renderer::terrain::foliage_paint::PaintedFoliage>,
) -> usize {
    let here: Vec<&scene_edits::PaintedEdit> = edits.iter().filter(|edit| edit.ground == ground).collect();
    if here.is_empty() {
        return 0;
    }
    let before = copies.len();
    copies.retain(|copy| {
        !here.iter().any(|edit| {
            edit.kind == copy.kind && glam::Vec3::from_array(edit.at).distance(copy.position) <= scene_edits::PAINTED_MATCH
        })
    });
    before - copies.len()
}

/// A painted copy as a new object, ready to paste: a root where the copy stood (`placed`) and a part for each piece
/// of its kind's model, each an `ImportedMesh` of `source` so a reopened scene draws it again. `pieces`: the mesh, its
/// material, its node in `source` and its transform within the model.
fn painted_object(
    name: &str,
    source: &str,
    placed: glam::Mat4,
    pieces: &[(MeshComponent, u32, u32, glam::Mat4)],
) -> crate::clipboard::EntityClipboard {
    let transform = |matrix: glam::Mat4| {
        let (scale, rotation, translation) = matrix.to_scale_rotation_translation();
        Transform { translation, rotation, scale }
    };
    let mut scratch = somnium_ecs::World::new();
    let root = scratch.spawn((Name::new(name), transform(placed), WorldTransform::identity()));
    for (n, (mesh, material, node, local)) in pieces.iter().enumerate() {
        let part = scratch.spawn((
            Name::new(&format!("{name} part {}", n + 1)),
            transform(*local),
            WorldTransform::identity(),
            *mesh,
            MaterialComponent { asset: somnium_asset::database::AssetId::NONE, runtime_id: *material },
            crate::ImportedMesh { source: source.to_owned(), node: *node },
            Parent { entity: root },
        ));
        debug_assert!(scratch.is_alive(part));
    }
    crate::clipboard::EntityClipboard::copy(&scratch, &[root])
}

/// The warning for a scene the world is composed from: Sunscar's `places_*` scenes are read raw by its composer, so
/// edits saved with one open never reach the world (design/editor/EDIT_ANYTHING.md, D). `None` for the world itself
/// (the startup scene) and for any other scene.
fn composed_scene_warning(opened: &Path, startup: Option<&Path>) -> Option<String> {
    let name = opened.file_name()?.to_str()?;
    let world = startup?.file_name()?.to_str()?;
    (name.starts_with("places_") && name != world).then(|| {
        format!("Edits saved in {name} do not reach the world: the world is composed from it. Open {world} to edit the world")
    })
}

/// Whether a model path names a tree: a whole word of it (folders included, plurals too) is a tree word, so
/// `trees/grown/holm_oak_b.gltf` and `wall_fig.gltf` are and `silver_street_01.gltf` is not.
fn tree_like(path: &str) -> bool {
    named_by(path, &TREE_WORDS)
}

/// Whether a model path names a rock or a cliff, the same way: `rocks/granite_tor_03.gltf` is, `stonemason_hut.gltf`
/// is not (a whole word, plurals too), and neither is a plant (`foliage/.../rock_rose.gltf`, `stone_pine.gltf`).
fn rock_like(path: &str) -> bool {
    let plant = tree_like(path) || path.to_ascii_lowercase().replace('\\', "/").contains("/foliage/");
    !plant && named_by(path, &ROCK_WORDS)
}

fn named_by(path: &str, words: &[&str]) -> bool {
    path.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphabetic()).any(|word| {
        words.contains(&word) || word.strip_suffix('s').is_some_and(|single| words.contains(&single))
    })
}

/// A count with thousands marked: 12,400.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// What the card and the report say of a surface.
fn shape_text(shape: &MeshShape) -> String {
    let triangles = thousands(u64::from(shape.triangles));
    let mut text = if shape.open_above_base > 0 {
        format!(
            "OPEN on a side or top: {} edges with one face, {} above its base (hollow from behind them); {triangles} triangles",
            thousands(u64::from(shape.open_edges)),
            thousands(u64::from(shape.open_above_base))
        )
    } else if shape.open_edges > 0 {
        format!(
            "open at its base only ({} edges): fine where it sits in the ground; {triangles} triangles",
            thousands(u64::from(shape.open_edges))
        )
    } else {
        format!("closed on every side; {triangles} triangles")
    };
    for (n, what) in [(shape.non_manifold_edges, "edges with 3+ faces"), (shape.degenerate, "slivers"), (shape.broken, "broken triangles")] {
        if n > 0 {
            text.push_str(&format!(", {} {what}", thousands(u64::from(n))));
        }
    }
    text
}

/// A painted copy a click found: the ground it lives with, its place in that ground's list, its kind.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PaintedPick {
    pub terrain: u32,
    pub index: usize,
    pub kind: u16,
    pub distance: f32,
}

/// A ground-cover copy a click found (grass, a flower, a pebble). Cover is grown round the view and dropped behind
/// it, never stored, so the pick keeps the copy itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoverPick {
    pub terrain: u32,
    /// Its 16 m cell (`ground_cover::CELL`).
    pub cell: (i32, i32),
    /// The copy, terrain-local.
    pub copy: somnium_renderer::terrain::foliage_paint::PaintedFoliage,
    pub distance: f32,
}

/// An eye that has gone to a place and is not settled there yet: in the air until the ground under it streams in,
/// then on the ground until the place's own cells are in and it can stand on what is built there.
#[derive(Clone, Copy)]
struct PlaceDrop {
    x: f32,
    z: f32,
    yaw: f32,
    pitch: f32,
    /// The height it stands on now, once it is down.
    stood: Option<f32>,
    /// When it last moved: the look for something built there waits a moment for the streamer to start on the place.
    since: std::time::Instant,
    until: std::time::Instant,
}

/// How far above the ground a place's eye may stand on something built there (a bridge's deck over its river). A
/// roof higher than this over a place is not stood on.
const STRUCTURE_REACH: f32 = 12.0;

/// What the editor knows about the open scene's edits and the level tools' state.
#[derive(Default)]
pub(crate) struct EditingState {
    /// The opened scene file; Save writes the edit layer beside it.
    scene: Option<PathBuf>,
    /// The layer: as loaded, re-keyed where a generator renamed ids, with this session folded in at each save.
    layer: Option<SceneEdits>,
    /// The layer file exists but could not be read: Save refuses rather than overwrite it.
    layer_error: Option<String>,
    /// First-touch snapshots of every entity selected this session, and of everything under it.
    baselines: HashMap<PersistentId, Baseline>,
    /// Ids the opened scene and its streamed cells loaded with.
    base: HashSet<PersistentId>,
    /// Ids with saved edits, plus touched ones changed since their snapshot.
    edited: HashSet<PersistentId>,
    /// Ids the last floating scan reported.
    floating: HashSet<PersistentId>,
    /// Who made a streamed object: a cell's `sources` (unit root id -> label).
    sources: HashMap<PersistentId, String>,
    /// Each loaded file's own `made_by`, by project-relative path.
    made_by: HashMap<String, String>,
    /// glTF parts by source path (`None`: unreadable).
    parts: HashMap<String, Option<Arc<Vec<somnium_asset::GltfPart>>>>,
    /// Triangle BVHs by mesh range.
    bvhs: HashMap<(u32, u32, u32), Option<Arc<TriangleBvh>>>,
    /// Tools that write a name prefix, by prefix.
    prefix_tools: HashMap<String, Vec<String>>,
    /// A unit cube's BVH, for static box colliders.
    cube: Option<Arc<TriangleBvh>>,
    /// Entities Isolate hid, with the flags they had.
    isolated: Vec<(Entity, Option<EditorFlags>)>,
    /// The opened scene's places (`<scene>.places.json`), in the order the palette's "Go to" rows name them.
    places: Vec<crate::scene_places::Place>,
    /// The palette has been given `places`.
    places_shown: bool,
    /// A place the camera went to before its ground was loaded: the eye drops onto the ground when it comes.
    place_drop: Option<PlaceDrop>,
    /// The editor script pressed the gizmo last frame: it lets go this frame.
    script_release: bool,
    /// What the last scene load did with its layer, for one toast.
    summary: Option<String>,
    /// Terrain was sculpted or painted, which the layer does not hold.
    terrain_touched: bool,
    /// The card and the state it describes.
    card: Option<(CardKey, String)>,
    /// The undo generation the edited set was last checked at.
    checked_generation: u64,
    /// The selection last touched.
    touched_selection: Vec<Entity>,
    /// Frames counted for `SOMNIUM_EDITOR_SCRIPT`.
    script_frame: u64,
    /// The painted copy the last click found; the card describes it while nothing is selected.
    painted: Option<PaintedPick>,
    /// Its card, built once per pick.
    painted_card: Option<(PaintedPick, String)>,
    /// Each painted kind's box in its own units (the union of its parts); `None` while it has no mesh.
    kind_boxes: HashMap<u16, Option<(glam::Vec3, glam::Vec3)>>,
    /// The ground-cover copy the last click found, when no object or painted copy was nearer.
    cover: Option<CoverPick>,
    /// Its card, built once per pick.
    cover_card: Option<(CoverPick, String)>,
    /// The project's tools that name the foliage palette file, with their text, read once: the cards say which of
    /// them writes a kind's rules.
    palette_tools: Option<Vec<(String, String)>>,
    /// Each ground material set's names for its slots, by the set's folder (`SOURCE.json`), read once.
    set_names: HashMap<PathBuf, Arc<BTreeMap<u8, String>>>,
    /// Each mesh's surface, by mesh range (`None`: too big or unreadable).
    shapes: HashMap<(u32, u32, u32), Option<MeshShape>>,
}

/// What a ground material set calls its slots: `layers[]` with `slot` and `layer` in the set's `SOURCE.json` (Sunscar's
/// look and biome tools write it). Empty where the file or a name is missing.
fn set_slot_names(dir: &Path) -> BTreeMap<u8, String> {
    let Ok(text) = std::fs::read_to_string(dir.join("SOURCE.json")) else {
        return BTreeMap::new();
    };
    let Ok(document) = serde_json::from_str::<Value>(&text) else {
        return BTreeMap::new();
    };
    document["layers"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|layer| {
            let slot = u8::try_from(layer["slot"].as_u64()?).ok()?;
            let name = layer["layer"].as_str()?.trim();
            (!name.is_empty()).then(|| (slot, name.to_owned()))
        })
        .collect()
}

#[derive(Clone, PartialEq)]
struct CardKey {
    entity: Entity,
    generation: u64,
    count: usize,
    edited: bool,
    dirty: bool,
}

struct Baseline {
    json: Value,
    origin: Origin,
}

// ─── Small helpers ──────────────────────────────────────────────────────────

/// A path relative to the project, with forward slashes and `..` resolved, so one file always reads the same.
pub(crate) fn project_relative(root: Option<&Path>, path: &Path) -> String {
    // The project root may be held in its canonical form (`\\?\F:\...`) while a scene path was joined from the
    // plain one, or the other way round: try both before giving up on a relative path.
    let canonical = |p: &Path| std::fs::canonicalize(p).ok();
    let relative = root.and_then(|root| {
        path.strip_prefix(root).ok().map(Path::to_path_buf).or_else(|| {
            let (root, path) = (canonical(root)?, canonical(path)?);
            path.strip_prefix(&root).ok().map(Path::to_path_buf)
        })
    });
    let path = relative.as_deref().unwrap_or(path);
    let mut parts: Vec<String> = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                if parts.last().is_some_and(|last| last != "..") {
                    parts.pop();
                } else {
                    parts.push("..".into());
                }
            }
            std::path::Component::CurDir | std::path::Component::RootDir => {}
            std::path::Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            std::path::Component::Prefix(prefix) => parts.push(prefix.as_os_str().to_string_lossy().into_owned()),
        }
    }
    parts.join("/")
}

/// The entity at the top of `entity`'s hierarchy.
fn top_ancestor(world: &World, entity: Entity) -> Entity {
    let mut current = entity;
    for _ in 0..64 {
        match world.get::<Parent>(current).map(|parent| parent.entity) {
            Some(parent) if parent != Entity::DANGLING && world.is_alive(parent) => current = parent,
            _ => break,
        }
    }
    current
}

/// Parent -> children from every `Parent` (`Children` keeps only sixteen).
fn children_map(world: &World) -> HashMap<Entity, Vec<Entity>> {
    let mut map: HashMap<Entity, Vec<Entity>> = HashMap::new();
    for (entity, parent) in world.iter_with::<Parent>() {
        if parent.entity != Entity::DANGLING && world.is_alive(parent.entity) {
            map.entry(parent.entity).or_default().push(entity);
        }
    }
    map
}

/// `root` and everything under it, parents first.
fn subtree(children: &HashMap<Entity, Vec<Entity>>, root: Entity) -> Vec<Entity> {
    let mut found = vec![root];
    let mut next = 0;
    while next < found.len() {
        if let Some(kids) = children.get(&found[next]) {
            found.extend(kids.iter().copied());
        }
        next += 1;
    }
    found
}

fn model_of(world: &World, entity: Entity) -> Option<glam::Mat4> {
    world
        .get::<WorldTransform>(entity)
        .map(|world| world.0)
        .or_else(|| world.get::<Transform>(entity).map(Transform::to_matrix))
}

fn transformed_box(model: glam::Mat4, min: glam::Vec3, max: glam::Vec3) -> (glam::Vec3, glam::Vec3) {
    let mut lo = glam::Vec3::splat(f32::INFINITY);
    let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
    for i in 0..8 {
        let corner = glam::Vec3::new(
            if i & 1 == 0 { min.x } else { max.x },
            if i & 2 == 0 { min.y } else { max.y },
            if i & 4 == 0 { min.z } else { max.z },
        );
        let point = model.transform_point3(corner);
        lo = lo.min(point);
        hi = hi.max(point);
    }
    (lo, hi)
}

/// The world box round every mesh in `members`.
fn world_bounds(world: &World, renderer: &SomniumRenderer, members: &[Entity]) -> Option<(glam::Vec3, glam::Vec3)> {
    let mut lo = glam::Vec3::splat(f32::INFINITY);
    let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for &entity in members {
        let Some(mesh) = world.get::<MeshComponent>(entity) else {
            continue;
        };
        let (Some((min, max)), Some(model)) = (renderer.geometry.mesh_aabb(mesh.vertex_offset), model_of(world, entity))
        else {
            continue;
        };
        let (a, b) = transformed_box(model, glam::Vec3::from_array(min), glam::Vec3::from_array(max));
        lo = lo.min(a);
        hi = hi.max(b);
        any = true;
    }
    any.then_some((lo, hi))
}

fn name_of(world: &World, entity: Entity) -> String {
    world
        .get::<Name>(entity)
        .map(|name| name.as_str().to_owned())
        .unwrap_or_else(|| format!("Entity {}", entity.index()))
}

/// One mesh drawn somewhere, for the ground and click queries.
struct MeshItem {
    entity: Entity,
    mesh: MeshComponent,
    model: glam::Mat4,
    min: glam::Vec3,
    max: glam::Vec3,
}

/// Every visible mesh's world box, binned on an 8 m grid in XZ.
struct MeshIndex {
    items: Vec<MeshItem>,
    grid: HashMap<(i32, i32), Vec<usize>>,
}

const GRID: f32 = 8.0;

impl MeshIndex {
    fn build(world: &World, renderer: &SomniumRenderer) -> Self {
        let mut items = drawn_items(world, renderer);
        for (entity, mesh) in world.iter_with::<MeshComponent>() {
            if crate::is_hidden(world, entity) {
                continue;
            }
            let (Some((min, max)), Some(model)) = (renderer.geometry.mesh_aabb(mesh.vertex_offset), model_of(world, entity))
            else {
                continue;
            };
            let (lo, hi) = transformed_box(model, glam::Vec3::from_array(min), glam::Vec3::from_array(max));
            if !lo.is_finite() || !hi.is_finite() {
                continue;
            }
            items.push(MeshItem { entity, mesh: *mesh, model, min: lo, max: hi });
        }
        let mut grid: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
        for (i, item) in items.iter().enumerate() {
            let (x0, z0) = ((item.min.x / GRID).floor() as i32, (item.min.z / GRID).floor() as i32);
            let (x1, z1) = ((item.max.x / GRID).floor() as i32, (item.max.z / GRID).floor() as i32);
            // A mesh wider than the whole grid search (ground, a merged street) goes in every cell it spans; past 64
            // cells a side it is kept once, in its corner cell's overflow list.
            if (x1 - x0) > 64 || (z1 - z0) > 64 {
                grid.entry((i32::MIN, i32::MIN)).or_default().push(i);
                continue;
            }
            for x in x0..=x1 {
                for z in z0..=z1 {
                    grid.entry((x, z)).or_default().push(i);
                }
            }
        }
        Self { items, grid }
    }

    /// Indices of meshes whose box meets `min..max`.
    fn query(&self, min: glam::Vec3, max: glam::Vec3) -> Vec<usize> {
        let mut found: Vec<usize> = Vec::new();
        let (x0, z0) = ((min.x / GRID).floor() as i32, (min.z / GRID).floor() as i32);
        let (x1, z1) = ((max.x / GRID).floor() as i32, (max.z / GRID).floor() as i32);
        let wide = self.grid.get(&(i32::MIN, i32::MIN)).into_iter().flatten();
        let cells = (x0..=x1).flat_map(|x| (z0..=z1).map(move |z| (x, z)));
        for i in cells.filter_map(|cell| self.grid.get(&cell)).flatten().chain(wide) {
            let item = &self.items[*i];
            if item.min.cmple(max).all() && item.max.cmpge(min).all() {
                found.push(*i);
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }
}

/// What a game drew itself last frame for entities with no mesh component (TSF's rooms: a name, a transform and its own
/// room component), as mesh items. The draw's sort key names the entity, by the convention the selection outline uses;
/// a draw whose box is nowhere near its entity is someone else's key (characters key by vertex offset) and is left out.
fn drawn_items(world: &World, renderer: &SomniumRenderer) -> Vec<MeshItem> {
    let mut items = Vec::new();
    for draw in renderer.last_frame_draws() {
        let index = draw.sort_key.0 as u32;
        if index == 0 {
            continue;
        }
        let Some(entity) = world.find_entity_by_index(index) else {
            continue;
        };
        if world.get::<MeshComponent>(entity).is_some() || crate::is_hidden(world, entity) {
            continue;
        }
        let Some((min, max)) = renderer.geometry.mesh_aabb(draw.vertex_offset) else {
            continue;
        };
        let (lo, hi) = transformed_box(draw.transform, glam::Vec3::from_array(min), glam::Vec3::from_array(max));
        let at = model_of(world, entity).map_or(glam::Vec3::ZERO, |m| m.w_axis.truncate());
        let near = at.cmpge(lo - glam::Vec3::ONE).all() && at.cmple(hi + glam::Vec3::ONE).all();
        if !near || !lo.is_finite() || !hi.is_finite() {
            continue;
        }
        let mesh = MeshComponent {
            vertex_offset: draw.vertex_offset,
            index_offset: draw.index_offset,
            index_count: draw.index_count,
        };
        items.push(MeshItem { entity, mesh, model: draw.transform, min: lo, max: hi });
    }
    items
}

/// Whether the box `model` (a unit cube transformed) meets the world box `min..max`: the world box's corners taken
/// into the cube's space, bounded, against the cube. Generous for a turned box, which is the safe side here.
fn box_overlaps(model: glam::Mat4, min: glam::Vec3, max: glam::Vec3) -> bool {
    let Some(inverse) = invert_affine(model) else {
        return false;
    };
    let (lo, hi) = transformed_box(inverse, min, max);
    lo.cmple(glam::Vec3::splat(0.5)).all() && hi.cmpge(glam::Vec3::splat(-0.5)).all()
}

/// A unit cube's triangles (corners at +-0.5), for box colliders.
fn unit_cube() -> TriangleBvh {
    let corners: Vec<glam::Vec3> = (0..8)
        .map(|i| glam::Vec3::new(if i & 1 == 0 { -0.5 } else { 0.5 }, if i & 2 == 0 { -0.5 } else { 0.5 }, if i & 4 == 0 { -0.5 } else { 0.5 }))
        .collect();
    let faces: [[u32; 4]; 6] = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]];
    let indices: Vec<u32> = faces.iter().flat_map(|f| [f[0], f[1], f[2], f[0], f[2], f[3]]).collect();
    TriangleBvh::build(&corners, &indices)
}

/// The nearest terrain hit along a unit ray.
fn terrain_hit(renderer: &SomniumRenderer, terrains: &[u32], origin: glam::Vec3, direction: glam::Vec3, max: f32) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for &id in terrains {
        let Some(terrain) = renderer.terrain(id) else {
            continue;
        };
        let Some(point) = terrain.raycast(origin, direction) else {
            continue;
        };
        let distance = (point - origin).dot(direction);
        if (0.0..=max).contains(&distance) && best.is_none_or(|b| distance < b.distance) {
            best = Some(Hit { distance, normal: glam::Vec3::Y });
        }
    }
    best
}

impl<G: GameApp> Engine<G> {
    // ─── The edit layer ─────────────────────────────────────────────────────

    /// A scene file is about to spawn: read its edit layer and apply it. Returns the edited document when there is
    /// one to apply.
    pub(super) fn edits_before_scene_spawn(&mut self, path: &str, document: &Value) -> Option<Value> {
        let scene = PathBuf::from(path);
        self.reset_editing(Some(scene.clone()));
        if !self.config.player_mode {
            self.editing.places = match crate::scene_places::read(&scene) {
                Ok((places, left_out)) => {
                    if left_out > 0 {
                        warn!(left_out, "places: rows without a name, an x or a z were left out of the Go to list");
                    }
                    places
                }
                Err(error) => {
                    warn!(%error, "places: the list could not be read");
                    Vec::new()
                }
            };
        }
        // A shipped game applies the layer and keeps nothing else: the bookkeeping below serves the editor.
        if !self.config.player_mode {
            let root = self.config.project_root.clone();
            self.editing.base = ids_in(document);
            if let Some(label) = made_by(document) {
                self.editing.made_by.insert(project_relative(root.as_deref(), &scene), label);
            }
        }
        let warning = (!self.config.player_mode)
            .then(|| composed_scene_warning(&scene, self.config.startup_scene.as_deref()))
            .flatten();
        let result = match SceneEdits::read(&scene) {
            Ok(Some(mut layer)) => {
                let mut edited = document.clone();
                let applied = scene_edits::apply(&mut edited, &mut layer, Scope::Scene);
                self.note_applied(&applied);
                let file = scene_edits::layer_path(&scene);
                let mut summary = applied.summary();
                let waiting = layer.entities.iter().filter(|entry| !entry.from.is_empty()).count();
                if waiting > 0 {
                    if !summary.is_empty() {
                        summary.push_str("; ");
                    }
                    summary.push_str(&format!("{waiting} more apply as their cells stream in"));
                }
                if !summary.is_empty() {
                    self.editing.summary = Some(format!(
                        "{}: {summary}",
                        file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
                    ));
                }
                for (id, why) in &applied.orphaned {
                    warn!(id = %id, why = %why, "edit layer entry not applied");
                }
                self.editing.layer = Some(layer);
                Some(edited)
            }
            Ok(None) => {
                self.editing.layer = Some(SceneEdits::new(&scene));
                None
            }
            Err(error) => {
                warn!(%error, "edit layer unreadable");
                self.editing.summary = Some(format!("Your edits file could not be read, so it was not applied: {error}"));
                self.editing.layer_error = Some(error);
                None
            }
        };
        if let Some(warning) = warning {
            warn!("{warning}");
            self.editing.summary = Some(match self.editing.summary.take() {
                Some(summary) => format!("{warning} ({summary})"),
                None => warning,
            });
        }
        result
    }

    /// After the scene spawned: say what its layer did.
    pub(super) fn edits_after_scene_spawn(&mut self) {
        if let Some(summary) = self.editing.summary.take() {
            // A player is never told about the editor's edit layer; the log keeps the line.
            if self.config.player_mode {
                info!("{summary}");
            } else {
                self.toast(&summary);
            }
        }
    }

    /// Forget the old scene's edit state (a new scene, a map, a different file).
    pub(super) fn reset_editing(&mut self, scene: Option<PathBuf>) {
        let parts = std::mem::take(&mut self.editing.parts);
        let prefix_tools = std::mem::take(&mut self.editing.prefix_tools);
        self.editing = EditingState { scene, parts, prefix_tools, ..EditingState::default() };
    }

    fn note_applied(&mut self, applied: &scene_edits::Applied) {
        for id in applied.changed.iter().chain(&applied.added) {
            if let Some(id) = PersistentId::parse_hex(id) {
                self.editing.edited.insert(id);
            }
        }
    }

    /// A streamed cell is about to spawn: its entries of the layer (and anything moved this session that has not
    /// been saved yet) go into its document.
    pub(super) fn edits_before_cell_spawn(&mut self, index: usize, document: &Value) -> Option<Value> {
        let root = self.config.project_root.clone();
        let file = project_relative(root.as_deref(), &self.cell_streamer.cells[index].file);
        if !self.config.player_mode {
            self.editing.base.extend(ids_in(document));
            if let Some(label) = made_by(document) {
                self.editing.made_by.insert(file.clone(), label);
            }
            if let Some(sources) = document.get("sources").and_then(Value::as_object) {
                for (id, label) in sources {
                    if let (Some(id), Some(label)) = (PersistentId::parse_hex(id), label.as_str()) {
                        self.editing.sources.insert(id, label.to_owned());
                    }
                }
            }
        }
        let layer = self.editing.layer.as_mut()?;
        if layer.entities.is_empty() {
            return None;
        }
        let mut edited = document.clone();
        let applied = scene_edits::apply(&mut edited, layer, Scope::Cell(&file));
        for (id, why) in &applied.orphaned {
            warn!(id = %id, why = %why, cell = %file, "edit layer entry not applied");
        }
        self.note_applied(&applied);
        Some(edited)
    }

    /// A streamed cell is about to unload: fold what was done to its entities into the layer first.
    pub(super) fn edits_before_cell_unload(&mut self, entities: &[Entity]) {
        let ids: HashSet<PersistentId> = entities
            .iter()
            .filter_map(|e| self.world.persistent_id(*e))
            .filter(|id| self.editing.baselines.contains_key(id))
            .collect();
        if !ids.is_empty() {
            self.fold_session(Some(&ids), true);
        }
    }

    /// A selected part of one level of detail takes the same part of the other levels with it, so a move, a turn, a
    /// delete or a save is of the prop and not of one of its copies ([`crate::lod_group::twins_of`]).
    pub(super) fn select_lod_twins(&mut self) {
        let in_a_level = |entity: &Entity| {
            self.world
                .get::<Parent>(*entity)
                .and_then(|level| self.world.get::<Parent>(level.entity))
                .is_some()
        };
        // Most selections are of things with no grandparent at all: no pass over the world for those.
        if self.world.entities_with::<crate::lod_group::LodGroupComponent>().next().is_none()
            || !self.selection.as_slice().iter().any(in_a_level)
        {
            return;
        }
        let twins = crate::lod_group::twins_of(&self.world, &children_map(&self.world), self.selection.as_slice());
        self.selection.add_followers(twins);
    }

    /// The script's gizmo gesture (`drag` in [`Self::run_editor_script`]): the pointer on the handle of `axis`,
    /// pressed, and taken to where a gesture of `amount` ends. False when no handle could be taken.
    fn script_drag(&mut self, mode: GizmoMode, axis: GizmoAxis, amount: f32) -> bool {
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.set_gizmo_mode(mode);
        }
        self.refresh_gizmo_anchor();
        let Some(renderer) = self.renderer.as_ref() else {
            return false;
        };
        let Some(at) = renderer.gizmo_world_pos else {
            return false;
        };
        let turned = if self.settings.editor().gizmo_local_space { renderer.gizmo_world_rotation } else { glam::Quat::IDENTITY };
        // The gizmo's size in the world: it is drawn at a constant size on screen (`gizmo_axis_under_ray`).
        let reach = (renderer.camera_pos - at).length().max(0.5) * 0.15;
        let along = turned * axis.world_dir();
        let (from, to) = match mode {
            GizmoMode::Translate => (at + along * (0.6 * reach), at + along * (0.6 * reach + amount)),
            GizmoMode::Scale => (at + along * (0.6 * reach), at + along * (0.6 * reach * amount)),
            GizmoMode::Rotate => {
                let (tangent, bitangent) = ring_plane_basis(axis);
                let (tangent, bitangent) = (turned * tangent, turned * bitangent);
                // Take the ring on the camera's side, where no other ring's handle lies in front of it.
                let toward = renderer.camera_pos - at;
                let near = toward.dot(bitangent).atan2(toward.dot(tangent));
                let on_ring = |angle: f32| at + (tangent * angle.cos() + bitangent * angle.sin()) * (0.84 * reach);
                (on_ring(near), on_ring(near + amount.to_radians()))
            }
        };
        let (width, height) = self.viewport_size();
        let view_proj = renderer.picking_view_proj();
        let to_pointer = |point: glam::Vec3| {
            let clip = view_proj * point.extend(1.0);
            (clip.w > 0.0).then(|| ((clip.x / clip.w + 1.0) * 0.5 * width, (1.0 - clip.y / clip.w) * 0.5 * height))
        };
        let (Some(press), Some(release)) = (to_pointer(from), to_pointer(to)) else {
            return false;
        };
        self.cursor_pos = press;
        if !self.begin_gizmo_drag() {
            return false;
        }
        self.cursor_pos = release;
        self.editing.script_release = true;
        true
    }

    // ─── Places ──────────────────────────────────────────────────────

    /// Each frame: the palette is given the scene's places once it is there, and an eye waiting over a place whose
    /// ground had not loaded drops onto it when it comes.
    pub(super) fn settle_places(&mut self) {
        if !self.editing.places_shown
            && let Some(ui) = self.ui_manager.as_mut()
        {
            ui.set_palette_places(
                self.editing.places.iter().map(|p| (p.name.clone(), p.group.clone(), p.built)).collect(),
            );
            self.editing.places_shown = true;
        }
        let Some(mut drop) = self.editing.place_drop else {
            return;
        };
        let now = std::time::Instant::now();
        let eye = |height: f32| glam::Vec3::new(drop.x, height + crate::scene_places::EYE_HEIGHT, drop.z);
        match drop.stood {
            // In the air: down onto the ground as soon as its tile is in.
            None => {
                if let Some(ground) = self.ground_height(drop.x, drop.z) {
                    self.camera_pose_request = Some((eye(ground), drop.yaw, drop.pitch));
                    (drop.stood, drop.since) = (Some(ground), now);
                    self.editing.place_drop = Some(drop);
                } else if now > drop.until {
                    self.editing.place_drop = None;
                    self.toast("No ground has loaded under that place: the camera stays in the air over it");
                }
            }
            // Down: once the place's cells are in, onto what is built there, if anything is.
            Some(stood) => {
                let settled = now.duration_since(drop.since).as_secs_f32() > 1.5 && self.cell_streamer.reading.is_empty();
                if settled || now > drop.until {
                    self.editing.place_drop = None;
                    if let Some(height) = self.standing_height(drop.x, drop.z)
                        && (height - stood).abs() > 0.2
                    {
                        self.camera_pose_request = Some((eye(height), drop.yaw, drop.pitch));
                    }
                }
            }
        }
    }

    /// What an eye stands on at scene (x, z): the ground, or what is built over it there (a bridge's deck, a quay, a
    /// stair) up to [`STRUCTURE_REACH`] above it. The support search End uses. `None` while no ground is loaded there.
    fn standing_height(&mut self, x: f32, z: f32) -> Option<f32> {
        let ground = self.ground_height(x, z)?;
        let renderer = self.renderer.as_ref()?;
        let index = MeshIndex::build(&self.world, renderer);
        let terrains = self.prepare_terrains();
        let over = glam::Vec3::new(x, ground + STRUCTURE_REACH, z);
        let half = glam::Vec3::splat(0.05);
        let support = self.support_for(&[over], over - half, over + half, &HashSet::new(), &index, &terrains);
        Some(support.gap.map_or(ground, |gap| (over.y - gap).max(ground)))
    }

    /// The palette's "Go to" row `index`: the camera stands at that place, eye height over its ground, facing the
    /// way the place names (or the way it faced).
    pub(super) fn go_to_place(&mut self, index: usize) {
        let Some(place) = self.editing.places.get(index).cloned() else {
            return;
        };
        // ponytail: Play stands the player under the editor camera when it starts, so going somewhere while playing
        // means stop, go, play. That is left to him (one key each) until he asks for it in one step.
        if self.play_session_active {
            self.toast("Stop Play first, then go to the place: Play starts where the camera stands");
            return;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        // The way the camera faces now, for a place that names no yaw.
        let (width, height) = self.viewport_size();
        let ahead = ndc_to_world(width * 0.5, height * 0.5, width, height, &renderer.picking_view_proj().inverse());
        let forward = (ahead - renderer.camera_pos).normalize_or_zero();
        let (yaw, pitch) = match place.camera_yaw() {
            Some(yaw) => (yaw, 0.0),
            None => (forward.z.atan2(forward.x).to_degrees(), forward.y.clamp(-1.0, 1.0).asin().to_degrees().clamp(-89.0, 89.0)),
        };
        let here = renderer.camera_pos.y;
        let now = std::time::Instant::now();
        let waiting = PlaceDrop {
            x: place.x,
            z: place.z,
            yaw,
            pitch,
            stood: None,
            since: now,
            until: now + std::time::Duration::from_secs(120),
        };
        self.editing.place_drop = None;
        if let Some(y) = place.y {
            // A place in the air names its own height.
            self.camera_pose_request = Some((glam::Vec3::new(place.x, y, place.z), yaw, pitch));
        } else if let Some(height) = self.standing_height(place.x, place.z) {
            // Loaded already: stand there now, and look once more when the streamer has settled round the eye.
            let eye = glam::Vec3::new(place.x, height + crate::scene_places::EYE_HEIGHT, place.z);
            self.camera_pose_request = Some((eye, yaw, pitch));
            self.editing.place_drop = Some(PlaceDrop { stood: Some(height), ..waiting });
        } else {
            // Its ground has not streamed in: go there at this height and come down when it has.
            self.camera_pose_request = Some((glam::Vec3::new(place.x, here, place.z), yaw, pitch));
            self.editing.place_drop = Some(waiting);
        }
        let note = if place.note.is_empty() { String::new() } else { format!(": {}", place.note) };
        self.toast(&format!("{}{note}", place.name));
    }

    /// Snapshot whatever has just been selected (and everything under it) before anything can change it.
    pub(super) fn touch_selection(&mut self) {
        if self.config.player_mode || self.play_session_active {
            return;
        }
        let generation = self.undo_stack.generation();
        if generation != self.editing.checked_generation {
            self.editing.checked_generation = generation;
            self.refresh_session_edits();
        }
        let selection = self.selection.as_slice();
        if selection == self.editing.touched_selection.as_slice() {
            return;
        }
        self.editing.touched_selection = selection.to_vec();
        let children = children_map(&self.world);
        let mut fresh: Vec<Entity> = Vec::new();
        for &entity in self.selection.as_slice() {
            for member in subtree(&children, entity) {
                let known = self
                    .world
                    .persistent_id(member)
                    .is_some_and(|id| self.editing.baselines.contains_key(&id));
                if !known && !fresh.contains(&member) {
                    fresh.push(member);
                }
            }
        }
        if fresh.is_empty() {
            return;
        }
        let snapshots = self.snapshots(&fresh);
        for entity in fresh {
            let Some(id) = self.world.persistent_id(entity) else {
                continue;
            };
            let Some(json) = snapshots.get(&id.to_string()).cloned() else {
                continue;
            };
            let origin = self.origin_of(entity);
            self.editing.baselines.insert(id, Baseline { json, origin });
        }
    }

    /// The scene documents of `entities`, by persistent id (minting ids where missing).
    fn snapshots(&mut self, entities: &[Entity]) -> HashMap<String, Value> {
        let alive: Vec<Entity> = entities.iter().copied().filter(|e| self.world.is_alive(*e)).collect();
        let Ok(document) = crate::scene_schema::entities_to_json(&mut self.world, &self.type_registry, &alive) else {
            return HashMap::new();
        };
        document["entities"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|entity| Some((entity.get("persistent_id")?.as_str()?.to_owned(), entity.clone())))
            .collect()
    }

    fn origin_of(&self, entity: Entity) -> Origin {
        let from = self
            .world
            .get::<crate::cell_streaming::Streamed>(entity)
            .and_then(|streamed| self.cell_streamer.cells.get(streamed.cell as usize))
            .map(|cell| project_relative(self.config.project_root.as_deref(), &cell.file))
            .unwrap_or_default();
        let parent = self
            .world
            .get::<Parent>(entity)
            .filter(|parent| self.world.is_alive(parent.entity))
            .and_then(|parent| self.world.get::<Name>(parent.entity))
            .map(|name| name.as_str().to_owned())
            .unwrap_or_default();
        Origin { from, parent }
    }

    /// Which touched entities differ from their snapshot now (for the marker), keeping the saved ones.
    fn refresh_session_edits(&mut self) {
        let touched: Vec<(PersistentId, Entity)> = self
            .editing
            .baselines
            .keys()
            .filter_map(|id| Some((*id, self.world.entity_by_persistent_id(*id)?)))
            .collect();
        if touched.is_empty() || touched.len() > 2_000 {
            return;
        }
        let entities: Vec<Entity> = touched.iter().map(|(_, e)| *e).collect();
        let now = self.with_isolation_lifted(|engine| engine.snapshots(&entities));
        for (id, _) in touched {
            let changed = now
                .get(&id.to_string())
                .zip(self.editing.baselines.get(&id))
                .is_some_and(|(now, baseline)| *now != baseline.json);
            let saved = self.editing.layer.as_ref().is_some_and(|layer| layer.entry(&id.to_string()).is_some());
            if changed || saved {
                self.editing.edited.insert(id);
            } else {
                self.editing.edited.remove(&id);
            }
        }
    }

    /// Fold this session's changes to the touched entities (all, or `only`) into the layer, and take fresh snapshots
    /// of the ones still alive. `unloading`: they are about to go with their cell, not deleted.
    fn fold_session(&mut self, only: Option<&HashSet<PersistentId>>, unloading: bool) {
        let Some(mut layer) = self.editing.layer.take() else {
            return;
        };
        let ids: Vec<PersistentId> = self
            .editing
            .baselines
            .keys()
            .filter(|id| only.is_none_or(|only| only.contains(id)))
            .copied()
            .collect();
        let alive: Vec<Entity> = ids.iter().filter_map(|id| self.world.entity_by_persistent_id(*id)).collect();
        let now = self.with_isolation_lifted(|engine| engine.snapshots(&alive));
        let gone: HashSet<String> = ids.iter().map(PersistentId::to_string).filter(|id| !now.contains_key(id)).collect();
        for id in ids {
            let hex = id.to_string();
            let Some(baseline) = self.editing.baselines.get_mut(&id) else {
                continue;
            };
            match now.get(&hex) {
                Some(current) => {
                    let added_before = layer.entry(&hex).is_some_and(|entry| entry.op == EditOp::Add);
                    if !self.editing.base.contains(&id) && !added_before {
                        scene_edits::record_add(&mut layer, current);
                    } else {
                        scene_edits::record_change(&mut layer, &baseline.json, current, &baseline.origin);
                    }
                    if unloading {
                        self.editing.baselines.remove(&id);
                    } else {
                        baseline.json = current.clone();
                    }
                }
                None => {
                    let baseline = self.editing.baselines.remove(&id).expect("looked up above");
                    let added = layer.entry(&hex).is_some_and(|entry| entry.op == EditOp::Add);
                    let parent_gone = scene_edits::parent_id_of(&baseline.json).is_some_and(|p| gone.contains(p));
                    if added || !parent_gone {
                        // An addition goes whole; a removal covers everything under it.
                        scene_edits::record_remove(&mut layer, &baseline.json, &baseline.origin);
                    }
                    self.editing.edited.remove(&id);
                }
            }
        }
        for entry in &layer.entities {
            if let Some(id) = PersistentId::parse_hex(&entry.id)
                && entry.op != EditOp::Remove
            {
                self.editing.edited.insert(id);
            }
        }
        self.editing.layer = Some(layer);
    }

    /// Ctrl+S: keep the edits beside the opened scene. A scene with no file of its own goes to Save As.
    pub(super) fn save_scene_edits(&mut self) {
        let Some(scene) = self.editing.scene.clone() else {
            self.save_scene_as();
            return;
        };
        if let Some(error) = self.editing.layer_error.clone() {
            self.toast(&format!(
                "Not saved: your edits file could not be read ({error}). Move it aside, open the scene again, then save."
            ));
            return;
        }
        self.flush_material_assets();
        self.fold_session(None, false);
        self.prune_painted_edits();
        let Some(layer) = self.editing.layer.as_ref() else {
            return;
        };
        let count = layer.entities.len() + layer.painted.len();
        match layer.write(&scene) {
            Ok(()) => {
                self.scene_dirty = false;
                self.autosave.saved();
                crate::autosave::clear(&self.config.content_root);
                self.remember_recent_scene(&scene);
                let file = scene_edits::layer_path(&scene);
                let file = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let mut message = if count == 0 {
                    "Saved: no edits to this scene".to_owned()
                } else {
                    format!("Saved {count} edited object{} to {file}", if count == 1 { "" } else { "s" })
                };
                if self.editing.terrain_touched {
                    message.push_str(" (terrain sculpting is not kept there: File > Save Scene As keeps it)");
                }
                self.toast(&message);
                if let Some(ui) = self.ui_manager.as_mut() {
                    ui.reset_inspector_baseline();
                }
            }
            Err(error) => self.toast(&format!("Save failed: {error}")),
        }
    }

    /// Write the whole scene, edits included, to a file the author picks; it becomes the opened scene.
    pub(super) fn save_scene_as(&mut self) {
        let start = self
            .editing
            .scene
            .as_ref()
            .and_then(|scene| scene.parent().map(Path::to_path_buf))
            .or_else(|| self.config.project_root.as_ref().map(|root| root.join("scenes")).filter(|dir| dir.is_dir()))
            .unwrap_or_else(|| self.config.content_root.clone());
        let Some(path) = rfd::FileDialog::new()
            .set_title("Save Scene As")
            .add_filter("Scene", &["json", "somnium"])
            .set_directory(&start)
            .save_file()
        else {
            return;
        };
        self.flush_material_assets();
        // Cell entries live outside this file and keep applying to the cells.
        self.fold_session(None, false);
        let kept_for_cells: Vec<_> = self
            .editing
            .layer
            .as_ref()
            .map(|layer| layer.entities.iter().filter(|entry| !entry.from.is_empty()).cloned().collect())
            .unwrap_or_default();
        self.end_isolation();
        let text = path.to_string_lossy().into_owned();
        if let Err(error) = crate::scene_schema::save_scene_schema(&mut self.world, &self.type_registry, &text) {
            self.toast(&format!("Save failed: {error}"));
            return;
        }
        self.save_terrain_sidecars(&text);
        // The file now holds every edit: an old layer beside it would apply twice.
        let old = scene_edits::layer_path(&path);
        if old.exists() {
            let aside = old.with_extension("json.replaced");
            let _ = std::fs::remove_file(&aside);
            let _ = std::fs::rename(&old, &aside);
        }
        self.reset_editing(Some(path.clone()));
        self.editing.base = self.world.entities_with::<PersistentId>().filter_map(|e| self.world.persistent_id(e)).collect();
        let mut layer = SceneEdits::new(&path);
        layer.entities = kept_for_cells;
        if !layer.entities.is_empty()
            && let Err(error) = layer.write(&path)
        {
            self.toast(&format!("The scene saved, but its streamed-cell edits did not: {error}"));
        }
        self.editing.layer = Some(layer);
        self.scene_dirty = false;
        self.autosave.saved();
        crate::autosave::clear(&self.config.content_root);
        self.remember_recent_scene(&path);
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
        if let Some(ui) = self.ui_manager.as_mut() {
            ui.set_scene_name(name.clone());
            ui.reset_inspector_baseline();
        }
        self.toast(&format!("Saved the whole scene to {}", name.unwrap_or_default()));
    }

    /// Heightmap and splatmap sidecars for the scene's own terrains (streamed tiles keep theirs).
    pub(super) fn save_terrain_sidecars(&mut self, path: &str) {
        let Some(renderer) = &self.renderer else {
            return;
        };
        let terrain_ids: Vec<u32> = self
            .world
            .iter_with::<TerrainComponent>()
            .filter(|(e, _)| self.world.get::<crate::cell_streaming::Streamed>(*e).is_none())
            .map(|(_, terrain)| terrain.terrain_id)
            .collect();
        for id in terrain_ids {
            if let Some(terrain) = renderer.terrain(id) {
                let sidecar = format!("{path}.terrain{id}.bin");
                match terrain.save_binary(&sidecar) {
                    Ok(()) => info!("Terrain {} data saved to {}", id, sidecar),
                    Err(e) => warn!("Failed to save terrain {}: {}", id, e),
                }
            }
        }
    }

    /// Terrain was sculpted or painted: Save warns that the layer does not hold it.
    pub(super) fn note_terrain_edit(&mut self) {
        self.editing.terrain_touched = true;
    }

    // ─── Picking ────────────────────────────────────────────────────────────

    /// The triangle BVH of a mesh range, read back once and shared by every entity drawing it.
    fn bvh_for(&mut self, mesh: MeshComponent) -> Option<Arc<TriangleBvh>> {
        let key = (mesh.vertex_offset, mesh.index_offset, mesh.index_count);
        if let Some(found) = self.editing.bvhs.get(&key) {
            return found.clone();
        }
        let built = (mesh.index_count / 3 <= BVH_TRIANGLE_LIMIT)
            .then(|| {
                let (renderer, ctx) = (self.renderer.as_ref()?, self.render_ctx.as_ref()?);
                let (vertices, indices) =
                    renderer.read_mesh(ctx, mesh.vertex_offset, mesh.index_offset, mesh.index_count)?;
                let positions: Vec<glam::Vec3> = vertices.iter().map(|v| glam::Vec3::from_array(v.position)).collect();
                Some(Arc::new(TriangleBvh::build(&positions, &indices)))
            })
            .flatten();
        self.editing.bvhs.insert(key, built.clone());
        built
    }

    /// What a plain click on the viewport selects: the nearest surface under the cursor (triangles, not boxes), as
    /// the whole object it belongs to. Clicking the selected object again picks the part.
    pub(super) fn click_pick(&mut self) -> Option<Entity> {
        let (origin, direction) = self.cursor_ray()?;
        let hit = self.pick_hit(origin, direction);
        // A painted copy (a tree, a shrub) or a ground-cover one (grass, a flower) nearer than any object is what was
        // clicked: not an entity, so the selection clears and the card names the copy.
        let (painted, cover) = self.pick_copies(origin, direction, hit.map(|(d, _)| d));
        (self.editing.painted, self.editing.cover) = (painted, cover);
        if painted.is_some() || cover.is_some() {
            return None;
        }
        hit.map(|(_, entity)| entity)
    }

    /// The painted or ground-cover copy a world ray meets first, before the object hit at `object` and the ground
    /// (a copy behind a hill is not under the cursor). At most one of the two.
    fn pick_copies(&mut self, origin: glam::Vec3, direction: glam::Vec3, object: Option<f32>) -> (Option<PaintedPick>, Option<CoverPick>) {
        let ground = self.renderer.as_ref().and_then(|renderer| {
            let terrains: Vec<u32> = self.world.iter_with::<TerrainComponent>().map(|(_, t)| t.terrain_id).collect();
            terrain_hit(renderer, &terrains, origin, direction, PAINTED_REACH)
        });
        let limit = object
            .into_iter()
            .chain(ground.map(|hit| hit.distance + GROUND_MARGIN))
            .fold(PAINTED_REACH, f32::min);
        let painted = self.pick_painted(origin, direction, limit, object.is_some());
        let cover = self.pick_cover(origin, direction, painted.map_or(limit, |p| p.distance), object.is_some());
        match cover {
            Some(cover) => (None, Some(cover)),
            None => (painted, None),
        }
    }

    /// F4: the object at the centre of the view. In Play, stop first and select it in the restored editor world (by
    /// persistent id), framed, so what you saw while playing is what you fix.
    pub(super) fn select_looked_at(&mut self) {
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let origin = renderer.camera_pos;
        let centre = renderer.picking_view_proj().inverse().project_point3(glam::Vec3::new(0.0, 0.0, 0.5));
        let direction = (centre - origin).normalize_or_zero();
        let primary = self.selection.primary.take();
        let found = (direction != glam::Vec3::ZERO).then(|| self.pick_hit(origin, direction)).flatten();
        self.selection.primary = primary;
        let (painted, cover) = if direction == glam::Vec3::ZERO {
            (None, None)
        } else {
            self.pick_copies(origin, direction, found.map(|(d, _)| d))
        };
        if painted.is_some() || cover.is_some() {
            if self.play_session_active {
                self.handle_editor_event(EditorEvent::StopSimulation);
            }
            self.selection.clear();
            self.after_selection_change();
            (self.editing.painted, self.editing.cover) = (painted, cover);
            let kind = painted.map_or_else(|| cover.map_or(0, |c| c.copy.kind), |p| p.kind);
            let name = self.foliage_kind_name(kind).to_owned();
            self.toast(&if painted.is_some() {
                format!("{name}: a painted copy (it lives with its ground, so it is not an object). The card says where it is from")
            } else {
                format!("{name}: ground cover, grown round the view by its rules (not stored). The card says which rules")
            });
            return;
        }
        let hit = found.map(|(_, entity)| entity);
        let id = hit.and_then(|e| self.world.persistent_id(e));
        let name = hit.map(|e| name_of(&self.world, e));
        if self.play_session_active {
            self.handle_editor_event(EditorEvent::StopSimulation);
        }
        let Some(name) = name else {
            self.toast("Nothing to select at the centre of the view (the ground and grass are not objects)");
            return;
        };
        match id.and_then(|id| self.world.entity_by_persistent_id(id)) {
            Some(entity) => {
                self.selection.set_single(Some(entity));
                self.after_selection_change();
                self.focus_camera_on_selection();
                self.reveal_selection_in_outliner(true);
                self.toast(&format!("{name}: selected, the thing you were looking at. End drops it; Ctrl+S keeps it"));
            }
            None => self.toast(&format!("{name} only exists while the game runs (the game made it), so there is nothing to edit")),
        }
    }

    /// A tile's painted copies as they install: the ones the open scene's layer took out stay out.
    pub(super) fn painted_before_install(&self, ground: &Path, copies: &mut Vec<somnium_renderer::terrain::foliage_paint::PaintedFoliage>) {
        let Some(layer) = self.editing.layer.as_ref().filter(|layer| !layer.painted.is_empty()) else { return };
        let ground = project_relative(self.config.project_root.as_deref(), ground);
        let hidden = hide_painted(&layer.painted, &ground, copies);
        if hidden > 0 {
            info!(ground = %ground, hidden, "painted copies kept out by the edit layer");
        }
    }

    /// Take the picked painted copy out of its ground and note it in the layer so Ctrl+S keeps it out: the copy, where
    /// it stood in the world, and the undo step (not yet pushed). `None` when nothing painted is picked.
    fn take_painted_copy(
        &mut self,
    ) -> Option<(
        somnium_renderer::terrain::foliage_paint::PaintedFoliage,
        glam::Mat4,
        Box<dyn crate::editor_commands::EditorCommand>,
    )> {
        let pick = self.editing.painted.take()?;
        self.editing.painted_card = None;
        let ground = self.ground_file(pick.terrain)?;
        let ground = project_relative(self.config.project_root.as_deref(), &ground);
        let terrain = self.renderer.as_mut().and_then(|r| r.terrain_mut(pick.terrain))?;
        let copy = terrain.painted_foliage.get(pick.index).copied().filter(|copy| copy.kind == pick.kind)?;
        let placed = terrain.model
            * glam::Mat4::from_scale_rotation_translation(
                glam::Vec3::splat(copy.scale),
                glam::Quat::from_rotation_y(copy.yaw) * glam::Quat::from_rotation_x(copy.tilt),
                copy.position,
            );
        let before = terrain.painted_foliage.clone();
        terrain.painted_foliage.remove(pick.index);
        terrain.edit_revision = terrain.edit_revision.wrapping_add(1);
        let after = terrain.painted_foliage.clone();
        let undo = crate::editor_commands::FoliageEditCmd::new(pick.terrain, before, after, self.terrain_restore_queue.clone());
        if let Some(layer) = self.editing.layer.as_mut() {
            layer.painted.push(scene_edits::PaintedEdit { ground, kind: copy.kind, at: copy.position.to_array() });
        }
        self.scene_dirty = true;
        Some((copy, placed, Box::new(undo)))
    }

    /// Delete with a painted copy picked (its card showing): take it out of its ground, undoable, and note it in the
    /// layer so Ctrl+S keeps it out. Returns whether there was one.
    pub(super) fn delete_painted_copy(&mut self) -> bool {
        let Some(kind) = self.editing.painted.map(|pick| pick.kind) else { return false };
        let name = self.foliage_kind_name(kind).to_owned();
        let Some((_, _, undo)) = self.take_painted_copy() else { return false };
        self.undo_stack.push_silent(undo);
        self.toast(&format!("Took out {name}: Ctrl+S keeps it out, Ctrl+Z brings it back"));
        true
    }

    /// Ctrl+Shift+O with a painted copy picked: it becomes an object. The copy goes (as Delete takes it out) and an
    /// object of its kind's model stands where it stood, selected, to move, turn, delete and save like any prop; Ctrl+S
    /// keeps both. One undo step.
    pub(super) fn make_painted_object(&mut self) {
        let Some(pick) = self.editing.painted else {
            self.toast("Click a painted tree or shrub first, then make it an object");
            return;
        };
        let name = self.foliage_kind_name(pick.kind).to_owned();
        let Some(entry) = self.project_foliage.get(&pick.kind) else {
            self.toast(&format!("{name} is one of the engine's own kinds, with no model file to make an object of"));
            return;
        };
        let source = project_relative(self.config.project_root.as_deref(), &entry.source);
        let pieces: Vec<(MeshComponent, u32, u32, glam::Mat4)> = self
            .foliage_meshes
            .get(usize::from(pick.kind))
            .and_then(Option::as_ref)
            .into_iter()
            .flatten()
            .zip(&entry.primitives)
            .map(|(part, node)| {
                let mesh = MeshComponent {
                    vertex_offset: part.vertex_offset,
                    index_offset: part.index_offset,
                    index_count: part.index_count,
                };
                (mesh, part.material_id, *node, part.local)
            })
            .collect();
        if pieces.is_empty() || pieces.len() != entry.primitives.len() {
            self.toast(&format!("{name}'s model is not loaded yet: try again in a moment"));
            return;
        }
        let Some((_, placed, hide)) = self.take_painted_copy() else { return };
        use crate::editor_commands::EditorCommand as _;
        let mut paste = crate::clipboard::PasteEntitiesCmd::new(painted_object(&name, &source, placed, &pieces), None);
        paste.execute(&mut self.world, &mut self.selection.primary);
        let roots = paste.roots().to_vec();
        self.undo_stack.push_silent(Box::new(crate::editor_commands::CommandGroup::new(
            "Make an Object",
            vec![hide, Box::new(paste)],
        )));
        self.selection.set_many(roots);
        self.after_selection_change();
        info!(kind = pick.kind, name = %name, source = %source, "a painted copy became an object");
        self.toast(&format!("{name} is an object now: move, turn or delete it, and Ctrl+S keeps it"));
    }

    /// Before a save: a taken-out copy that is back on its loaded ground (Ctrl+Z) is no longer taken out.
    fn prune_painted_edits(&mut self) {
        let Some(layer) = self.editing.layer.as_mut().filter(|layer| !layer.painted.is_empty()) else { return };
        let Some(renderer) = self.renderer.as_ref() else { return };
        let root = self.config.project_root.clone();
        let grounds: Vec<(String, u32)> = self
            .world
            .iter_with::<TerrainComponent>()
            .filter_map(|(entity, t)| {
                let file = self
                    .world
                    .get::<crate::cell_streaming::Streamed>(entity)
                    .and_then(|s| self.cell_streamer.cells.get(s.cell as usize))
                    .map(|cell| cell.file.clone())
                    .or_else(|| self.editing.scene.clone())?;
                Some((project_relative(root.as_deref(), &file), t.terrain_id))
            })
            .collect();
        layer.painted.retain(|edit| {
            let back = grounds.iter().filter(|(ground, _)| *ground == edit.ground).any(|(_, id)| {
                renderer.terrain(*id).is_some_and(|data| {
                    data.painted_foliage.iter().any(|copy| {
                        copy.kind == edit.kind
                            && glam::Vec3::from_array(edit.at).distance(copy.position) <= scene_edits::PAINTED_MATCH
                    })
                })
            });
            !back
        });
    }

    /// The object nearest along a world ray, triangle-accurate, as [`Self::click_pick`] means it, and how far.
    fn pick_hit(&mut self, origin: glam::Vec3, direction: glam::Vec3) -> Option<(f32, Entity)> {
        let renderer = self.renderer.as_ref()?;
        let pickable = |world: &World, entity: Entity| {
            world.get::<EditorFlags>(entity).is_none_or(|flags| !flags.locked && !flags.hidden)
        };
        let mut boxes: Vec<(f32, Entity)> = Vec::new();
        let proxies = self
            .world
            .entities_with::<crate::decal::DecalComponent>()
            .chain(self.world.entities_with::<LightComponent>())
            .chain(self.world.entities_with::<AudioEmitterComponent>())
            .chain(self.world.entities_with::<crate::ParticleEmitter>());
        for entity in self.world.entities_with::<MeshComponent>().chain(proxies) {
            if pickable(&self.world, entity)
                && let Some(d2) = entity_ray_hit_distance(&self.world, renderer, entity, origin, direction)
            {
                boxes.push((d2.sqrt(), entity));
            }
        }
        // What the game drew itself (TSF's rooms): picked by the same box-then-triangles test.
        let drawn = drawn_items(&self.world, renderer);
        let mut drawn_meshes: HashMap<Entity, (MeshComponent, glam::Mat4)> = HashMap::new();
        for item in &drawn {
            if !pickable(&self.world, item.entity) {
                continue;
            }
            let inverse = item.model.inverse();
            let (lo, hi) = renderer
                .geometry
                .mesh_aabb(item.mesh.vertex_offset)
                .map_or((glam::Vec3::ZERO, glam::Vec3::ZERO), |(a, b)| (glam::Vec3::from_array(a), glam::Vec3::from_array(b)));
            let local_origin = inverse.transform_point3(origin);
            let local_direction = inverse.transform_vector3(direction);
            if let Some(t) = ray_aabb_distance(local_origin, local_direction, lo, hi) {
                let hit = item.model.transform_point3(local_origin + local_direction * t);
                boxes.push((origin.distance(hit), item.entity));
                drawn_meshes.insert(item.entity, (item.mesh, item.model));
            }
        }
        boxes.sort_by(|a, b| a.0.total_cmp(&b.0));
        boxes.dedup_by_key(|(_, entity)| *entity);
        // Of a prop's levels of detail only the one on screen can be clicked: the others stand unseen in the same
        // place, and a click that took one of those moved a copy nobody was looking at.
        if self.world.entities_with::<crate::lod_group::LodGroupComponent>().next().is_some() {
            let children = children_map(&self.world);
            boxes.retain(|(_, entity)| !crate::lod_group::hidden_from(&self.world, &children, *entity, origin));
        }
        let mut best: Option<(f32, Entity)> = None;
        for (box_distance, entity) in boxes {
            if best.is_some_and(|(d, _)| box_distance > d) {
                break;
            }
            let own = self.world.get::<MeshComponent>(entity).copied().zip(model_of(&self.world, entity));
            let exact = if let Some((mesh, model)) = own.or_else(|| drawn_meshes.get(&entity).copied()) {
                match self.bvh_for(mesh) {
                    Some(bvh) => {
                        let surface = MeshSurface {
                            bvh: &bvh,
                            model,
                            inverse: model.inverse(),
                            min: glam::Vec3::splat(f32::NEG_INFINITY),
                            max: glam::Vec3::splat(f32::INFINITY),
                        };
                        surface.raycast(origin, direction, 100_000.0).map(|hit| hit.distance)
                    }
                    None => Some(box_distance),
                }
            } else if self.world.get::<crate::decal::DecalComponent>(entity).is_some() {
                decal_slab_distance(&self.world, entity, origin, direction)
            } else {
                Some(box_distance)
            };
            if let Some(d) = exact
                && best.is_none_or(|(b, _)| d < b)
            {
                best = Some((d, entity));
            }
        }
        let (distance, part) = best?;
        Some((distance, self.object_for_click(part)))
    }

    /// A painted kind's box in its own units: the union of its parts' boxes, each through its part transform.
    fn kind_box(&mut self, kind: u16) -> Option<(glam::Vec3, glam::Vec3)> {
        if let Some(found) = self.editing.kind_boxes.get(&kind) {
            return *found;
        }
        let renderer = self.renderer.as_ref()?;
        let parts = self.foliage_meshes.get(usize::from(kind)).and_then(Option::as_ref)?;
        let mut bounds: Option<(glam::Vec3, glam::Vec3)> = None;
        for part in parts {
            let Some((min, max)) = renderer.geometry.mesh_aabb(part.vertex_offset) else { continue };
            let (lo, hi) = transformed_box(part.local, glam::Vec3::from_array(min), glam::Vec3::from_array(max));
            bounds = Some(bounds.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
        }
        self.editing.kind_boxes.insert(kind, bounds);
        bounds
    }

    /// A foliage kind's parts as meshes, each with its transform within the kind.
    fn foliage_parts(&self, kind: u16) -> Vec<(MeshComponent, glam::Mat4)> {
        self.foliage_meshes
            .get(usize::from(kind))
            .and_then(Option::as_ref)
            .map(|parts| {
                parts
                    .iter()
                    .map(|p| (MeshComponent { vertex_offset: p.vertex_offset, index_offset: p.index_offset, index_count: p.index_count }, p.local))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The nearest painted copy a world ray meets before `limit` metres: each terrain's foliage grid is visited along
    /// the ray, every copy's box tested, then the nearest few by their triangles. Small kinds (grass, litter) never
    /// hide an object behind them (`behind_object`).
    fn pick_painted(&mut self, origin: glam::Vec3, direction: glam::Vec3, limit: f32, behind_object: bool) -> Option<PaintedPick> {
        // Copy out what the walk needs, so the box cache can borrow `self` afterwards.
        let mut copies: Vec<(u32, usize, somnium_renderer::terrain::foliage_paint::PaintedFoliage, glam::Mat4)> = Vec::new();
        {
            let renderer = self.renderer.as_ref()?;
            for (_, terrain) in self.world.iter_with::<TerrainComponent>() {
                let (Some(data), Some(grid)) = (renderer.terrain(terrain.terrain_id), self.foliage_grids.get(&terrain.terrain_id))
                else {
                    continue;
                };
                let inverse = data.model.inverse();
                let (local_origin, local_direction) =
                    (inverse.transform_point3(origin), inverse.transform_vector3(direction).normalize_or_zero());
                let mut seen: HashSet<usize> = HashSet::new();
                for step in 0..=(limit / PAINTED_STEP).ceil() as usize {
                    let p = local_origin + local_direction * (step as f32 * PAINTED_STEP);
                    grid.visit([p.x, p.z], PAINTED_STEP, |i| {
                        seen.insert(i);
                    });
                }
                for index in seen {
                    if let Some(copy) = data.painted_foliage.get(index) {
                        copies.push((terrain.terrain_id, index, *copy, data.model));
                    }
                }
            }
        }
        let mut entered: Vec<(f32, u32, usize, u16, glam::Mat4)> = Vec::new();
        for (terrain, index, copy, ground) in copies {
            let Some((lo, hi)) = self.kind_box(copy.kind) else { continue };
            if behind_object && (hi - lo).length() < PAINTED_SMALL {
                continue;
            }
            let model = ground
                * glam::Mat4::from_scale_rotation_translation(
                    glam::Vec3::splat(copy.scale),
                    glam::Quat::from_rotation_y(copy.yaw) * glam::Quat::from_rotation_x(copy.tilt),
                    copy.position,
                );
            let inverse = model.inverse();
            let (local_origin, local_direction) = (inverse.transform_point3(origin), inverse.transform_vector3(direction));
            if let Some(t) = ray_aabb_distance(local_origin, local_direction, lo, hi) {
                let distance = origin.distance(model.transform_point3(local_origin + local_direction * t));
                if distance < limit {
                    entered.push((distance, terrain, index, copy.kind, model));
                }
            }
        }
        entered.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut best: Option<PaintedPick> = None;
        for (box_distance, terrain, index, kind, model) in entered.into_iter().take(PAINTED_REFINE) {
            if best.is_some_and(|b| box_distance > b.distance) {
                break;
            }
            if let Some(distance) = self.copy_hit(kind, model, origin, direction, limit)
                && best.is_none_or(|b| distance < b.distance)
            {
                best = Some(PaintedPick { terrain, index, kind, distance });
            }
        }
        best
    }

    /// Where a world ray first meets a foliage copy's triangles (`model` places the copy), before `limit`.
    fn copy_hit(&mut self, kind: u16, model: glam::Mat4, origin: glam::Vec3, direction: glam::Vec3, limit: f32) -> Option<f32> {
        let mut best: Option<f32> = None;
        for (mesh, local) in self.foliage_parts(kind) {
            let Some(bvh) = self.bvh_for(mesh) else { continue };
            let world = model * local;
            let surface = MeshSurface {
                bvh: &bvh,
                model: world,
                inverse: world.inverse(),
                min: glam::Vec3::splat(f32::NEG_INFINITY),
                max: glam::Vec3::splat(f32::INFINITY),
            };
            if let Some(hit) = surface.raycast(origin, direction, limit)
                && best.is_none_or(|b| hit.distance < b)
            {
                best = Some(hit.distance);
            }
        }
        best
    }

    /// The nearest ground-cover copy (grass, flowers, pebbles: grown round the view) a world ray meets before `limit`
    /// metres, as it is drawn: thinned with distance, widened, sunk at its kind's edge. Each cell's column is tested
    /// first, then its copies' boxes, then the nearest few by their triangles. Between thin blades the box of a small
    /// kind is near enough. Small kinds never hide an object behind them (`behind_object`), as with painted copies.
    fn pick_cover(&mut self, origin: glam::Vec3, direction: glam::Vec3, limit: f32, behind_object: bool) -> Option<CoverPick> {
        let terrains: Vec<(u32, glam::Mat4)> = {
            let renderer = self.renderer.as_ref()?;
            self.world
                .iter_with::<TerrainComponent>()
                .filter_map(|(_, t)| renderer.terrain(t.terrain_id).map(|data| (t.terrain_id, data.model)))
                .collect()
        };
        // Each grown kind's box first: the box cache needs `self` mutably, the walk below only reads.
        let kinds: HashSet<u16> = terrains.iter().flat_map(|&(t, _)| self.ground_cover.cells(t).map(|((kind, _, _), _)| kind)).collect();
        let boxes: HashMap<u16, (glam::Vec3, glam::Vec3)> = kinds.into_iter().filter_map(|k| self.kind_box(k).map(|b| (k, b))).collect();
        let batched = crate::foliage_batches::enabled();
        // (box distance, the copy, where it is drawn, whether it is small)
        let mut entered: Vec<(f32, CoverPick, glam::Mat4, bool)> = Vec::new();
        for (terrain, ground) in terrains {
            let Some(focus) = self.ground_cover.focus(terrain) else { continue };
            let inverse = ground.inverse();
            let (local_origin, local_direction) = (inverse.transform_point3(origin), inverse.transform_vector3(direction));
            for ((kind, cx, cz), cell) in self.ground_cover.cells(terrain) {
                let Some(&(lo, hi)) = boxes.get(&kind) else { continue };
                let Some((radius, inner)) =
                    self.project_foliage.get(&kind).and_then(|e| e.ground_cover.as_ref()).map(|g| (g.radius, g.inner))
                else {
                    continue;
                };
                let small = (hi - lo).length() < PAINTED_SMALL;
                if cell.is_empty() || (behind_object && small) {
                    continue;
                }
                // The cell's column: its square, from its lowest copy's foot to its highest copy's top, as wide as the
                // widest copy can draw.
                let (mut low, mut high, mut largest) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f32);
                for c in cell {
                    low = low.min(c.instance.position.y);
                    high = high.max(c.instance.position.y);
                    largest = largest.max(c.instance.scale);
                }
                let reach = lo.abs().max(hi.abs()).max_element() * largest * crate::ground_cover::MOST_WIDEN;
                let size = crate::ground_cover::CELL;
                let column = (
                    glam::Vec3::new(cx as f32 * size - reach, low - reach, cz as f32 * size - reach),
                    glam::Vec3::new((cx + 1) as f32 * size + reach, high + reach, (cz + 1) as f32 * size + reach),
                );
                if ray_aabb_distance(local_origin, local_direction, column.0, column.1).is_none() {
                    continue;
                }
                for c in cell {
                    let scale = if batched {
                        crate::ground_cover::drawn_scale(c, focus, radius, inner)
                    } else {
                        let d = c.instance.position - focus;
                        let distance = d.x.hypot(d.z);
                        (distance <= radius && distance >= inner && c.instance.scale > 0.0).then(|| glam::Vec3::splat(c.instance.scale))
                    };
                    let Some(scale) = scale else { continue };
                    let model = ground * glam::Mat4::from_scale_rotation_translation(scale, c.rotation, c.instance.position);
                    let inverse = model.inverse();
                    let (o, d) = (inverse.transform_point3(origin), inverse.transform_vector3(direction));
                    if let Some(t) = ray_aabb_distance(o, d, lo, hi) {
                        let distance = origin.distance(model.transform_point3(o + d * t));
                        if distance < limit {
                            let pick = CoverPick { terrain, cell: (cx, cz), copy: c.instance, distance };
                            entered.push((distance, pick, model, small));
                        }
                    }
                }
            }
        }
        entered.sort_by(|a, b| a.0.total_cmp(&b.0));
        let nearest_small = entered.iter().find(|e| e.3).map(|e| e.1);
        let mut best: Option<CoverPick> = None;
        for (box_distance, pick, model, _) in entered.into_iter().take(PAINTED_REFINE) {
            if best.is_some_and(|b| box_distance > b.distance) {
                break;
            }
            if let Some(distance) = self.copy_hit(pick.copy.kind, model, origin, direction, limit)
                && best.is_none_or(|b| distance < b.distance)
            {
                best = Some(CoverPick { distance, ..pick });
            }
        }
        best.or(nearest_small)
    }

    /// A mesh's surface (`editor_ground::mesh_shape`), read back once and kept.
    fn mesh_shape_of(&mut self, mesh: MeshComponent) -> Option<MeshShape> {
        let key = (mesh.vertex_offset, mesh.index_offset, mesh.index_count);
        if let Some(found) = self.editing.shapes.get(&key) {
            return *found;
        }
        let shape = (mesh.index_count / 3 <= BVH_TRIANGLE_LIMIT)
            .then(|| {
                let (renderer, ctx) = (self.renderer.as_ref()?, self.render_ctx.as_ref()?);
                let (vertices, indices) = renderer.read_mesh(ctx, mesh.vertex_offset, mesh.index_offset, mesh.index_count)?;
                let positions: Vec<glam::Vec3> = vertices.iter().map(|v| glam::Vec3::from_array(v.position)).collect();
                Some(editor_ground::mesh_shape(&positions, &indices))
            })
            .flatten();
        self.editing.shapes.insert(key, shape);
        shape
    }

    /// The card's line on a selection's surface, every mesh under it together.
    fn shape_line(&mut self, entity: Entity) -> Option<String> {
        let children = children_map(&self.world);
        let mut meshes: Vec<MeshComponent> =
            subtree(&children, entity).into_iter().filter_map(|e| self.world.get::<MeshComponent>(e).copied()).collect();
        meshes.sort_by_key(|m| (m.vertex_offset, m.index_offset, m.index_count));
        meshes.dedup_by_key(|m| (m.vertex_offset, m.index_offset, m.index_count));
        if meshes.is_empty() {
            return None;
        }
        let triangles: u64 = meshes.iter().map(|m| u64::from(m.index_count / 3)).sum();
        if triangles > CARD_SHAPE_LIMIT {
            return Some(format!("shape     not checked on a click ({} triangles); Check Meshes (Level) does", thousands(triangles)));
        }
        let mut shape = MeshShape::default();
        for mesh in meshes {
            shape.add(&self.mesh_shape_of(mesh)?);
        }
        Some(format!("shape     {}", shape_text(&shape)))
    }

    /// Check the surface of every model the loaded objects draw: closed, open only at its base (fine where it sits in
    /// the ground), or open on a side or top (hollow from behind). Rocks and cliffs come first, since every one must
    /// be closed where it shows (the developer, 6 Nov: "they are hollow and one side is only there"). Written beside
    /// the scene as `<scene>.meshcheck.md`.
    pub(super) fn check_meshes(&mut self) {
        let started = std::time::Instant::now();
        let root = self.config.project_root.clone();
        let children = children_map(&self.world);
        // Each model file: its meshes, and the objects that draw it.
        let imported: Vec<(Entity, String, Option<MeshComponent>)> = self
            .world
            .iter_with::<crate::ImportedMesh>()
            .map(|(e, m)| (e, m.source.clone(), self.world.get::<MeshComponent>(e).copied()))
            .collect();
        let mut models: BTreeMap<String, (BTreeSet<(u32, u32, u32)>, HashSet<Entity>)> = BTreeMap::new();
        for (entity, source, mesh) in &imported {
            let entry = models.entry(source.clone()).or_default();
            if let Some(m) = mesh {
                entry.0.insert((m.vertex_offset, m.index_offset, m.index_count));
            }
            entry.1.insert(self.object_in(&children, *entity));
        }
        // (model, its surface, objects drawing it, examples, whether every mesh was read)
        let mut rows: Vec<(String, MeshShape, usize, Vec<String>, bool)> = Vec::new();
        for (source, (meshes, objects)) in models {
            let mut shape = MeshShape::default();
            let mut whole = true;
            for (vertex_offset, index_offset, index_count) in meshes {
                match self.mesh_shape_of(MeshComponent { vertex_offset, index_offset, index_count }) {
                    Some(s) => shape.add(&s),
                    None => whole = false,
                }
            }
            let mut objects_sorted: Vec<Entity> = objects.iter().copied().collect();
            objects_sorted.sort_by_key(|e| e.index());
            let examples: Vec<String> = objects_sorted.iter().take(3).map(|e| name_of(&self.world, *e)).collect();
            rows.push((source, shape, objects.len(), examples, whole));
        }
        // The files themselves, with the foliage palette's (its models, LODs and impostors): what will not load.
        let content = self.config.content_root.clone();
        let resolve = |source: &str| -> PathBuf {
            let path = Path::new(source);
            if path.is_absolute() {
                path.to_path_buf()
            } else if let Some(root) = root.as_ref().filter(|r| r.join(path).exists()) {
                root.join(path)
            } else {
                content.join(path)
            }
        };
        let mut files: BTreeMap<String, PathBuf> = rows.iter().map(|r| (r.0.clone(), resolve(&r.0))).collect();
        for entry in self.project_foliage.values() {
            for path in std::iter::once(&entry.source)
                .chain(entry.lod.as_ref().map(|l| &l.source))
                .chain(entry.impostor.as_ref().map(|l| &l.source))
            {
                let path = if path.is_absolute() { path.clone() } else { content.join(path) };
                files.insert(project_relative(root.as_deref(), &path), path);
            }
        }
        let file_problems: Vec<(String, Vec<String>)> = files
            .iter()
            .map(|(name, path)| (name.clone(), somnium_asset::gltf_file_problems(path)))
            .filter(|(_, problems)| !problems.is_empty())
            .collect();
        // Objects whose model failed to load: they keep their source and draw nothing.
        let mut drawing_nothing: BTreeMap<String, usize> = BTreeMap::new();
        for (entity, source, mesh) in &imported {
            if mesh.is_none() && self.world.get::<crate::ImportedMesh>(*entity).is_some() {
                *drawing_nothing.entry(source.clone()).or_default() += 1;
            }
        }
        let open_side = |r: &&(String, MeshShape, usize, Vec<String>, bool)| r.1.open_above_base > 0;
        let rocks: Vec<&(String, MeshShape, usize, Vec<String>, bool)> = rows.iter().filter(|r| rock_like(&r.0)).collect();
        let mut hollow_rocks: Vec<&(String, MeshShape, usize, Vec<String>, bool)> = rocks.iter().copied().filter(open_side).collect();
        hollow_rocks.sort_by(|a, b| b.2.cmp(&a.2).then(b.1.open_above_base.cmp(&a.1.open_above_base)));
        let mut others: Vec<&(String, MeshShape, usize, Vec<String>, bool)> =
            rows.iter().filter(|r| !rock_like(&r.0) && r.1.open_above_base > 0).collect();
        others.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
        // Slivers alone are harmless; fins and broken triangles (an index past the vertices) are not.
        let broken: Vec<&(String, MeshShape, usize, Vec<String>, bool)> =
            rows.iter().filter(|r| r.1.broken > 0 || r.1.non_manifold_edges > 0).collect();
        let unread: Vec<&(String, MeshShape, usize, Vec<String>, bool)> = rows.iter().filter(|r| !r.4).collect();
        let table = |items: &[&(String, MeshShape, usize, Vec<String>, bool)]| -> Vec<String> {
            let mut out = vec!["| objects | model | surface | e.g. |".to_owned(), "|---:|---|---|---|".to_owned()];
            out.extend(items.iter().map(|(source, shape, objects, examples, _)| {
                format!("| {objects} | {source} | {} | {} |", shape_text(shape), examples.join("; "))
            }));
            out
        };
        let mut report = vec![
            format!(
                "# Mesh check of {}",
                self.editing.scene.as_deref().map_or_else(|| "the open scene".to_owned(), |s| project_relative(root.as_deref(), s))
            ),
            String::new(),
            format!(
                "{} model files drawn by the loaded objects. \"Open on a side or top\" means an edge with one face above \
                 the model's lowest fifth: seen from behind it, the model is hollow. Open at the base only is fine where \
                 the model sits in the ground. Select one of its objects and use Select All Using This Model to find the rest.",
                rows.len()
            ),
            String::new(),
            format!("## Model files that will not load or do not match ({} of {} checked)", file_problems.len(), files.len()),
            String::new(),
        ];
        if file_problems.is_empty() {
            report.push("None: every model file the loaded objects and the foliage palette name reads whole.".to_owned());
        } else {
            report.push("| file | what is wrong |".to_owned());
            report.push("|---|---|".to_owned());
            for (name, problems) in &file_problems {
                report.push(format!("| {name} | {} |", problems.join("; ")));
            }
        }
        report.push(String::new());
        report.push(format!("## Objects that draw nothing: their model did not load, or is still streaming in ({})", drawing_nothing.values().sum::<usize>()));
        report.push(String::new());
        for (source, count) in &drawing_nothing {
            report.push(format!("- {source}: {count} object{}", if *count == 1 { "" } else { "s" }));
        }
        report.push(String::new());
        report.push(format!("## Rocks and cliffs open on a side or top ({} of {} rock-like models)", hollow_rocks.len(), rocks.len()));
        report.push(String::new());
        if hollow_rocks.is_empty() {
            report.push("None: every rock and cliff model loaded is closed where it shows.".to_owned());
        } else {
            report.extend(table(&hollow_rocks));
        }
        for (title, items) in [
            ("Other models open on a side or top (walls and roofs may be meant to be; foliage cards are)", &others),
            ("Fins (an edge with 3+ faces) or broken triangles (a corner past the vertices or not a number)", &broken),
            ("Not checked (too big to read back, or unreadable)", &unread),
        ] {
            report.push(String::new());
            report.push(format!("## {title} ({})", items.len()));
            report.push(String::new());
            if !items.is_empty() {
                report.extend(table(items));
            }
        }
        let text = report.join("\n") + "\n";
        let target = self
            .editing
            .scene
            .as_ref()
            .map(|s| PathBuf::from(format!("{}.meshcheck.md", s.display())))
            .or_else(|| root.as_ref().map(|r| r.join("meshcheck.md")));
        let summary = format!(
            "Check Meshes: {} model files will not load or do not match, {} objects draw nothing; {} of {} rock and cliff models are open on a side (hollow from behind); {} other models open on a side, {} with fins or broken triangles ({} files, {:.1} s)",
            file_problems.len(),
            drawing_nothing.values().sum::<usize>(),
            hollow_rocks.len(),
            rocks.len(),
            others.len(),
            broken.len(),
            rows.len(),
            started.elapsed().as_secs_f32()
        );
        match target.map(|path| std::fs::write(&path, &text).map(|()| path)) {
            Some(Ok(path)) => {
                info!("{summary}; written to {}", path.display());
                self.toast(&format!("{summary}. Written to {}", project_relative(root.as_deref(), &path)));
            }
            Some(Err(error)) => self.toast(&format!("{summary}. Not written: {error}")),
            None => self.toast(&summary),
        }
    }

    /// Models the importer refused since the last frame (`somnium_asset::take_refused`; it logged each once): one
    /// toast in the editor naming the first and the count. A game alone only drains the list: its log has them.
    pub(super) fn report_refused_models(&mut self) {
        let refused = somnium_asset::take_refused();
        let Some((file, reason)) = refused.first() else { return };
        if self.config.player_mode {
            return;
        }
        let name = project_relative(self.config.project_root.as_deref(), file);
        let reason: String = if reason.chars().count() > 160 {
            reason.chars().take(157).chain("...".chars()).collect()
        } else {
            reason.clone()
        };
        self.toast(&if refused.len() == 1 {
            format!("Skipped a broken model, kept off the GPU: {name}: {reason}")
        } else {
            format!(
                "Skipped {} broken models, kept off the GPU (the log names each): {name}: {reason}; and {} more",
                refused.len(),
                refused.len() - 1
            )
        });
    }

    /// The height of the loaded ground at world x, z, from the first terrain that covers the point.
    fn ground_height(&self, x: f32, z: f32) -> Option<f32> {
        let renderer = self.renderer.as_ref()?;
        self.world.iter_with::<TerrainComponent>().find_map(|(_, t)| {
            let data = renderer.terrain(t.terrain_id)?;
            let local = data.model.inverse().transform_point3(glam::Vec3::new(x, 0.0, z));
            let [wx, wz] = data.desc.world_size();
            ((0.0..=wx).contains(&local.x) && (0.0..=wz).contains(&local.z)).then(|| {
                let h = data.world_height_at(local.x, local.z);
                data.model.transform_point3(glam::Vec3::new(local.x, h, local.z)).y
            })
        })
    }

    /// The file a ground's data comes from: its streamed cell's, or the open scene's.
    fn ground_file(&self, terrain: u32) -> Option<PathBuf> {
        self.world
            .iter_with::<TerrainComponent>()
            .find(|(_, t)| t.terrain_id == terrain)
            .and_then(|(e, _)| self.world.get::<crate::cell_streaming::Streamed>(e).copied())
            .and_then(|s| self.cell_streamer.cells.get(s.cell as usize))
            .map(|cell| cell.file.clone())
            .or_else(|| self.editing.scene.clone())
    }

    /// Which of the project's tools writes a foliage kind's palette entry: a tool that names the palette file and the
    /// kind's name (without its number), quoted. Read once per session.
    fn palette_writer(&mut self, kind_name: &str) -> Option<String> {
        if self.editing.palette_tools.is_none() {
            let tools = self.config.project_root.as_ref().and_then(|root| std::fs::read_dir(root.join("tools")).ok());
            let mut found: Vec<(String, String)> = tools
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "py"))
                .filter_map(|entry| {
                    let text = std::fs::read_to_string(entry.path()).ok()?;
                    text.contains("foliage.palette.json").then(|| (format!("tools/{}", entry.file_name().to_string_lossy()), text))
                })
                .collect();
            found.sort();
            self.editing.palette_tools = Some(found);
        }
        let base = kind_name.trim_end_matches(|c: char| c.is_ascii_digit()).trim_end();
        let quoted = [format!("\"{base}\""), format!("'{base}'")];
        let writers: Vec<&String> = self
            .editing
            .palette_tools
            .iter()
            .flatten()
            .filter(|(_, text)| quoted.iter().any(|q| text.contains(q.as_str())))
            .map(|(name, _)| name)
            .collect();
        match writers.as_slice() {
            [] => None,
            [one] => Some((*one).clone()),
            [first, rest @ ..] => Some(format!("{first} (and {} more name it)", rest.len())),
        }
    }

    /// A ground slot's number and what the ground's own material set calls it (its `SOURCE.json`), with the set where
    /// it is not the project's. A set that names nothing falls back on the engine's built-in name for the slot, which a
    /// project's set need not follow.
    fn layer_label(&mut self, terrain: u32, layer: u8) -> String {
        let set = self
            .world
            .iter_with::<TerrainComponent>()
            .find(|(_, t)| t.terrain_id == terrain)
            .and_then(|(e, _)| self.world.get::<crate::cell_streaming::TerrainStreamingComponent>(e))
            .map(|s| s.layer_set.trim().to_owned())
            .unwrap_or_default();
        let terrain_root = self.config.content_root.join("terrain");
        let dir = crate::cell_streaming::layer_set_dir(&terrain_root, &set).unwrap_or(terrain_root);
        let names = match self.editing.set_names.get(&dir) {
            Some(names) => Arc::clone(names),
            None => {
                let names = Arc::new(set_slot_names(&dir));
                self.editing.set_names.insert(dir, Arc::clone(&names));
                names
            }
        };
        let in_set = if set.is_empty() { String::new() } else { format!(" in set {set}") };
        match names.get(&layer) {
            Some(name) => format!("slot {layer} \"{name}\"{in_set}"),
            None => {
                let engine = somnium_renderer::terrain::textures::LAYER_NAMES.get(usize::from(layer)).copied().unwrap_or("?");
                format!("slot {layer}{in_set} (the set names none; the engine's \"{engine}\")")
            }
        }
    }

    /// The card for a ground-cover copy: what kind, which model, the ground it grows on (its habitat), how dense it
    /// grows there and what grows beside it, the rules that grew it and the tool that wrote them.
    fn cover_card(&mut self, pick: CoverPick) -> Option<String> {
        if let Some((cached, card)) = &self.editing.cover_card
            && *cached == pick
        {
            return Some(card.clone());
        }
        let root = self.config.project_root.clone();
        let kind = pick.copy.kind;
        let entry = self.project_foliage.get(&kind)?.clone();
        let rules = entry.ground_cover.clone()?;
        let name = self.foliage_kind_name(kind).to_owned();
        let writer = self.palette_writer(&name);
        let habitat = self.layer_label(pick.terrain, entry.brush.layer);
        let file = self.ground_file(pick.terrain);
        let ground_entity = self.world.iter_with::<TerrainComponent>().find(|(_, t)| t.terrain_id == pick.terrain).map(|(e, _)| e);
        let streamed = ground_entity.is_some_and(|e| self.world.get::<crate::cell_streaming::Streamed>(e).is_some());
        // The tile's layer set ("" the project's own), which a kind's rules may name (`GroundCoverRules::layer_sets`).
        let tile_set = ground_entity
            .and_then(|e| self.world.get::<crate::cell_streaming::TerrainStreamingComponent>(e))
            .map(|s| s.layer_set.trim().to_owned())
            .unwrap_or_default();
        let renderer = self.renderer.as_ref()?;
        let data = renderer.terrain(pick.terrain)?;
        let (x, z) = (pick.copy.position.x, pick.copy.position.z);
        // A streamed tile's drifts follow the world (`GroundCover::update_seeded`).
        let origin = if streamed { glam::Vec2::new(data.model.w_axis.x, data.model.w_axis.z) } else { glam::Vec2::ZERO };
        let weight = data.layer_weight_at(x, z, entry.brush.layer);
        let here = crate::ground_cover::density_at(data, kind, &entry, x, z, origin);
        let full = rules.density * crate::ground_cover::density_scale();
        let focus = self.ground_cover.focus(pick.terrain).unwrap_or(pick.copy.position);
        let batched = crate::foliage_batches::enabled();
        let drawn = |c: &crate::ground_cover::Cover, radius: f32| {
            if batched {
                crate::ground_cover::drawn_scale(c, focus, radius, rules.inner).is_some()
            } else {
                let d = c.instance.position - focus;
                (rules.inner..=radius).contains(&d.x.hypot(d.z))
            }
        };
        // Its own cell, and every kind within a couple of metres of the copy.
        let (mut in_cell, mut drawn_in_cell) = (0usize, 0usize);
        let mut nearby: BTreeMap<u16, usize> = BTreeMap::new();
        let size = crate::ground_cover::CELL;
        for ((k, cx, cz), cell) in self.ground_cover.cells(pick.terrain) {
            if (k, cx, cz) == (kind, pick.cell.0, pick.cell.1) {
                in_cell = cell.len();
                drawn_in_cell = cell.iter().filter(|c| drawn(c, rules.radius)).count();
            }
            let (x0, z0) = (cx as f32 * size, cz as f32 * size);
            let gap = (x0 - x).max(x - (x0 + size)).max(0.0).hypot((z0 - z).max(z - (z0 + size)).max(0.0));
            if gap > COVER_NEARBY {
                continue;
            }
            let near = cell
                .iter()
                .filter(|c| (c.instance.position.x - x).hypot(c.instance.position.z - z) <= COVER_NEARBY)
                .count();
            if near > 0 {
                *nearby.entry(k).or_default() += near;
            }
        }
        let mut nearby: Vec<(u16, usize)> = nearby.into_iter().collect();
        nearby.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let together: usize = nearby.iter().map(|(_, n)| n).sum();
        let listed: Vec<String> = nearby.iter().take(5).map(|(k, n)| format!("{n} {}", self.foliage_kind_name(*k))).collect();
        let more = nearby.len().saturating_sub(5);
        // The kinds that grow on this slot of this tile's set: the habitat's mix.
        let same_layer: Vec<&str> = self
            .project_foliage
            .values()
            .filter(|e| e.brush.layer == entry.brush.layer && e.ground_cover.as_ref().is_some_and(|g| g.grows_on(&tile_set)))
            .map(|e| e.name.as_str())
            .collect();
        let sets = rules.layer_sets.as_ref().map_or_else(String::new, |sets| {
            let named: Vec<String> =
                sets.iter().map(|s| if s.trim().is_empty() { "the project's".to_owned() } else { s.trim().to_owned() }).collect();
            format!(", only on sets: {}", named.join(", "))
        });
        let at = data.model.transform_point3(pick.copy.position);
        let from_centre = (pick.copy.position.x - focus.x).hypot(pick.copy.position.z - focus.z);
        let mut lines = vec![
            format!("{name}   ground cover, kind {kind}: grown round the view by its rules, never stored"),
            format!("model     {}", project_relative(root.as_deref(), &entry.source)),
            format!(
                "habitat   ground {habitat}, weight {weight:.2} here (grows from {:.2}, on slopes to {:.0} deg); {} cover kinds grow on this slot of this set",
                entry.brush.min_layer_weight.max(1e-3),
                entry.brush.max_slope_deg,
                same_layer.len()
            ),
            format!(
                "density   {here:.2} a m2 here by the rules: {full:.2} at a full layer x weight {weight:.2}{}",
                if rules.clump.is_some() && weight * full > 0.0 {
                    format!(" x drift {:.2}", here / (weight * full))
                } else {
                    String::new()
                }
            ),
            format!(
                "grown     {in_cell} in its {size:.0} m cell ({:.2} a m2), {drawn_in_cell} drawn from {from_centre:.0} m",
                in_cell as f32 / (size * size)
            ),
            format!(
                "nearby    within {COVER_NEARBY:.0} m: {}{} ({together} copies)",
                listed.join(", "),
                if more > 0 { format!(" and {more} more kinds") } else { String::new() }
            ),
            format!(
                "rules     {}, kind {kind}: radius {:.0} m{}{}{sets}",
                project_relative(root.as_deref(), &self.config.content_root.join("foliage.palette.json")),
                rules.radius,
                rules.clump.map_or_else(String::new, |m| format!(", drifts {m:.0} m")),
                if rules.align_to_ground { ", leans with the slope" } else { ", upright" }
            ),
        ];
        if let Some(writer) = writer {
            lines.push(format!("written   by {writer}"));
        }
        match &file {
            Some(file) => lines.push(format!(
                "ground    {}{}",
                project_relative(root.as_deref(), file),
                if streamed { " (a streamed tile)" } else { "" }
            )),
            None => lines.push("ground    (a scene with no file yet)".to_owned()),
        }
        lines.push(format!(
            "at        x {:.1}  y {:.1}  z {:.1}   scale {:.2}   turned {:.0} deg",
            at.x,
            at.y,
            at.z,
            pick.copy.scale,
            pick.copy.yaw.to_degrees().rem_euclid(360.0)
        ));
        lines.push("Census (Level) lists every ground-cover kind grown round the view now".to_owned());
        let card = lines.join("\n");
        info!("cover card:\n{card}");
        self.editing.cover_card = Some((pick, card.clone()));
        Some(card)
    }

    /// The card for a painted copy: what kind, which model file, where it lives and which file keeps it.
    fn painted_card(&mut self, pick: PaintedPick) -> Option<String> {
        if let Some((cached, card)) = &self.editing.painted_card
            && *cached == pick
        {
            return Some(card.clone());
        }
        let root = self.config.project_root.clone();
        let name = self.foliage_kind_name(pick.kind).to_owned();
        if crate::foliage_palette::is_project_kind(pick.kind) {
            self.palette_writer(&name);
        }
        let renderer = self.renderer.as_ref()?;
        let data = renderer.terrain(pick.terrain)?;
        let copy = *data.painted_foliage.get(pick.index)?;
        let total = data.painted_foliage.len();
        let same = data.painted_foliage.iter().filter(|p| p.kind == pick.kind).count();
        let at = data.model.transform_point3(copy.position);
        let ground = self.world.iter_with::<TerrainComponent>().find(|(_, t)| t.terrain_id == pick.terrain).map(|(e, _)| e);
        let file: Option<PathBuf> = ground
            .and_then(|e| self.world.get::<crate::cell_streaming::Streamed>(e).copied())
            .and_then(|s| self.cell_streamer.cells.get(s.cell as usize))
            .map(|cell| cell.file.clone())
            .or_else(|| self.editing.scene.clone());
        let model = self.project_foliage.get(&pick.kind).map_or_else(
            || format!("built-in kind {} (the engine's own palette)", pick.kind),
            |entry| project_relative(root.as_deref(), &entry.source),
        );
        let mut lines = vec![
            format!("{name}   painted copy, kind {} (not an object: it lives with its ground)", pick.kind),
            format!("model     {model}"),
            format!("copy      {} of {total} on this ground; {same} of this kind here", pick.index + 1),
        ];
        if crate::foliage_palette::is_project_kind(pick.kind)
            && let Some(writer) = self.palette_writer(&name)
        {
            lines.push(format!("kind by   {writer} (its entry in the foliage palette)"));
        }
        match &file {
            Some(file) => {
                lines.push(format!("ground    {}", project_relative(root.as_deref(), file)));
                // The copies are kept beside the ground's file: `<file>.terrain<id>.bin.foliage.json`.
                let stem = file.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                let kept: Vec<String> = file
                    .parent()
                    .and_then(|dir| std::fs::read_dir(dir).ok())
                    .into_iter()
                    .flatten()
                    .flatten()
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .filter(|n| n.starts_with(&stem) && n.ends_with(".foliage.json"))
                    .collect();
                match kept.as_slice() {
                    [one] => lines.push(format!("kept in   {one}")),
                    [] => {}
                    many => lines.push(format!("kept in   one of {} (.foliage.json beside the ground)", many.len())),
                }
            }
            None => lines.push("ground    (a scene with no file yet)".to_owned()),
        }
        lines.push(format!(
            "at        x {:.1}  y {:.1}  z {:.1}   scale {:.2}   turned {:.0} deg",
            at.x,
            at.y,
            at.z,
            copy.scale,
            copy.yaw.to_degrees().rem_euclid(360.0)
        ));
        lines.push(if self.project_foliage.contains_key(&pick.kind) {
            "Ctrl+Shift+O makes it an object (move, turn, save it); Delete takes it out".to_owned()
        } else {
            "Delete takes it out; Census (Level) counts every painted kind and model loaded".to_owned()
        });
        let card = lines.join("\n");
        info!("painted card:\n{card}");
        self.editing.painted_card = Some((pick, card.clone()));
        Some(card)
    }

    /// Count what the loaded world is made of: painted copies by kind, and the model files objects draw (parts and
    /// whole objects). Written beside the scene as `<scene>.census.md`, with the tree-like models that come from
    /// outside the painted palette listed apart (the "old trees" question).
    pub(super) fn census(&mut self) {
        let root = self.config.project_root.clone();
        let mut painted: BTreeMap<u16, usize> = BTreeMap::new();
        let mut grounds = 0;
        if let Some(renderer) = self.renderer.as_ref() {
            for (_, terrain) in self.world.iter_with::<TerrainComponent>() {
                if let Some(data) = renderer.terrain(terrain.terrain_id) {
                    grounds += 1;
                    for copy in &data.painted_foliage {
                        *painted.entry(copy.kind).or_default() += 1;
                    }
                }
            }
        }
        let mut models: BTreeMap<String, (usize, HashSet<Entity>)> = BTreeMap::new();
        let imported: Vec<(Entity, String)> =
            self.world.iter_with::<crate::ImportedMesh>().map(|(e, m)| (e, m.source.clone())).collect();
        let children = children_map(&self.world);
        for (entity, source) in &imported {
            let object = self.object_in(&children, *entity);
            let entry = models.entry(source.clone()).or_default();
            entry.0 += 1;
            entry.1.insert(object);
        }
        let palette: HashSet<String> =
            self.project_foliage.values().map(|e| project_relative(root.as_deref(), &e.source)).collect();
        let copies: usize = painted.values().sum();
        // Ground cover is grown round the view and dropped behind it: what is grown now, and how much of it draws.
        let batched = crate::foliage_batches::enabled();
        let mut cover: BTreeMap<u16, (usize, usize, BTreeSet<u32>)> = BTreeMap::new();
        let terrain_ids: Vec<u32> = self.world.iter_with::<TerrainComponent>().map(|(_, t)| t.terrain_id).collect();
        for &terrain in &terrain_ids {
            let Some(focus) = self.ground_cover.focus(terrain) else { continue };
            for ((kind, _, _), cell) in self.ground_cover.cells(terrain) {
                let (radius, inner) =
                    self.project_foliage.get(&kind).and_then(|e| e.ground_cover.as_ref()).map_or((0.0, 0.0), |g| (g.radius, g.inner));
                let entry = cover.entry(kind).or_default();
                entry.2.insert(terrain);
                entry.0 += cell.len();
                entry.1 += cell
                    .iter()
                    .filter(|c| {
                        if batched {
                            crate::ground_cover::drawn_scale(c, focus, radius, inner).is_some()
                        } else {
                            (inner..=radius).contains(&(c.instance.position.x - focus.x).hypot(c.instance.position.z - focus.z))
                        }
                    })
                    .count();
            }
        }
        let cover_grown: usize = cover.values().map(|v| v.0).sum();
        let cover_drawn: usize = cover.values().map(|v| v.1).sum();
        let mut report = vec![
            format!(
                "# Census of {}",
                self.editing.scene.as_deref().map_or_else(|| "the open scene".to_owned(), |s| project_relative(root.as_deref(), s))
            ),
            String::new(),
            format!(
                "What is loaded now: {grounds} grounds and {} objects round the camera. Cells streamed out are not counted.",
                imported.iter().map(|(e, _)| self.object_in(&children, *e)).collect::<HashSet<_>>().len()
            ),
            String::new(),
            format!("## Painted copies by kind ({copies} in {} kinds; they live with their ground, not as objects)", painted.len()),
            String::new(),
            "| copies | kind | name | model |".to_owned(),
            "|---:|---:|---|---|".to_owned(),
        ];
        let mut by_count: Vec<(u16, usize)> = painted.iter().map(|(k, n)| (*k, *n)).collect();
        by_count.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (kind, count) in &by_count {
            let model = self
                .project_foliage
                .get(kind)
                .map_or_else(|| "built-in".to_owned(), |e| project_relative(root.as_deref(), &e.source));
            report.push(format!("| {count} | {kind} | {} | {model} |", self.foliage_kind_name(*kind)));
        }
        report.push(String::new());
        report.push(format!(
            "## Ground cover round the view now ({cover_grown} grown, {cover_drawn} drawn, in {} kinds; grown by rules, never stored)",
            cover.len()
        ));
        report.push(String::new());
        report.push("Click a copy for its card: habitat, density there, rules and the tool that wrote them.".to_owned());
        report.push(String::new());
        report.push("| grown | drawn | kind | name | ground | model |".to_owned());
        report.push("|---:|---:|---:|---|---|---|".to_owned());
        let mut cover_rows: Vec<(u16, (usize, usize, BTreeSet<u32>))> = cover.into_iter().collect();
        cover_rows.sort_by(|a, b| b.1.0.cmp(&a.1.0).then(a.0.cmp(&b.0)));
        for (kind, (grown, drawn, terrains)) in &cover_rows {
            let (model, layer) = self.project_foliage.get(kind).map_or_else(
                || ("built-in".to_owned(), 0),
                |e| (project_relative(root.as_deref(), &e.source), e.brush.layer),
            );
            // Every set it grows on: one kind on two biomes' slots shows here.
            let mut grounds: Vec<String> = terrains.iter().map(|&t| self.layer_label(t, layer)).collect();
            grounds.sort();
            grounds.dedup();
            let name = self.foliage_kind_name(*kind).to_owned();
            report.push(format!("| {grown} | {drawn} | {kind} | {name} | {} | {model} |", grounds.join("; ")));
        }
        report.push(String::new());
        report.push(format!("## Models the objects draw ({} files)", models.len()));
        report.push(String::new());
        report.push("| objects | parts | model |".to_owned());
        report.push("|---:|---:|---|".to_owned());
        let mut drawn: Vec<(&String, usize, usize)> = models.iter().map(|(s, (parts, objects))| (s, objects.len(), *parts)).collect();
        drawn.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
        for (source, objects, parts) in &drawn {
            report.push(format!("| {objects} | {parts} | {source} |"));
        }
        let outside: Vec<&(&String, usize, usize)> =
            drawn.iter().filter(|(s, _, _)| tree_like(s.as_str()) && !palette.contains(s.as_str())).collect();
        report.push(String::new());
        report.push(format!("## Tree-like models from outside the painted palette ({})", outside.len()));
        report.push(String::new());
        if outside.is_empty() {
            report.push("None: every tree-like model loaded is a painted palette kind.".to_owned());
        } else {
            report.push("Objects (places, ruins, points of interest) that draw a tree-like model the palette does not list. \
                         These are the likeliest \"old trees\". Select one and use Select All Using This Model.".to_owned());
            report.push(String::new());
            for (source, objects, _) in &outside {
                report.push(format!("- {source}: {objects} objects"));
            }
        }
        let text = report.join("\n") + "\n";
        let target = self
            .editing
            .scene
            .as_ref()
            .map(|s| PathBuf::from(format!("{}.census.md", s.display())))
            .or_else(|| root.as_ref().map(|r| r.join("census.md")));
        let summary = format!(
            "Census: {copies} painted copies in {} kinds, {cover_grown} ground-cover copies round the view in {} kinds, {} model \
             files; {} tree-like models from outside the palette",
            painted.len(),
            cover_rows.len(),
            models.len(),
            outside.len()
        );
        match target.map(|path| std::fs::write(&path, &text).map(|()| path)) {
            Some(Ok(path)) => {
                info!("{summary}; written to {}", path.display());
                self.toast(&format!("{summary}. Written to {}", project_relative(root.as_deref(), &path)));
            }
            Some(Err(error)) => self.toast(&format!("{summary}. Not written: {error}")),
            None => self.toast(&summary),
        }
    }

    /// Select every object that draws the selected object's model files.
    pub(super) fn select_same_asset(&mut self) {
        let Some(selected) = self.selection.primary.filter(|e| self.world.is_alive(*e)) else {
            let copy = self.editing.painted.is_some() || self.editing.cover.is_some();
            self.toast(if copy {
                "Painted and ground-cover copies are not objects; the Census lists every copy of each kind"
            } else {
                "Select something first"
            });
            return;
        };
        let children = children_map(&self.world);
        let object = self.object_of(selected);
        let sources: HashSet<String> = subtree(&children, object)
            .into_iter()
            .filter_map(|e| self.world.get::<crate::ImportedMesh>(e).map(|m| m.source.clone()))
            .collect();
        if sources.is_empty() {
            self.toast("It draws no model file");
            return;
        }
        let parts: Vec<Entity> = self
            .world
            .iter_with::<crate::ImportedMesh>()
            .filter(|(_, m)| sources.contains(&m.source))
            .map(|(e, _)| e)
            .collect();
        let mut objects: Vec<Entity> = parts.into_iter().map(|e| self.object_of(e)).collect();
        objects.sort_by_key(|e| e.index());
        objects.dedup();
        let count = objects.len();
        self.selection.set_many(objects);
        self.after_selection_change();
        let named = if sources.len() == 1 { sources.iter().next().cloned().unwrap_or_default() } else { format!("its {} model files", sources.len()) };
        self.toast(&format!("{count} objects draw {named}"));
    }

    /// The object a click on `part` means: its highest ancestor that is still one object (not a container of
    /// hundreds), or the part itself when that object is already selected.
    fn object_for_click(&self, part: Entity) -> Entity {
        let object = self.object_of(part);
        if object != part && self.selection.primary == Some(object) { part } else { object }
    }

    /// The object `part` belongs to: its highest ancestor that is still one object, not a container of hundreds.
    fn object_of(&self, part: Entity) -> Entity {
        self.object_in(&children_map(&self.world), part)
    }

    /// [`Self::object_of`] over a children map built once (a pass over thousands of parts).
    fn object_in(&self, children: &HashMap<Entity, Vec<Entity>>, part: Entity) -> Entity {
        let mut object = part;
        loop {
            let Some(parent) = self.world.get::<Parent>(object).map(|p| p.entity) else {
                break;
            };
            if parent == Entity::DANGLING || !self.world.is_alive(parent) || subtree(children, parent).len() > CONTAINER_PARTS
            {
                break;
            }
            object = parent;
        }
        object
    }

    /// The selection's objects: each selected entity, minus any under another selected one.
    fn selected_objects(&self) -> Vec<Entity> {
        let selected: HashSet<Entity> = self.selection.as_slice().iter().copied().collect();
        self.selection
            .as_slice()
            .iter()
            .copied()
            .filter(|&entity| {
                let mut current = entity;
                for _ in 0..64 {
                    match self.world.get::<Parent>(current).map(|p| p.entity) {
                        Some(parent) if self.world.is_alive(parent) => {
                            if selected.contains(&parent) {
                                return false;
                            }
                            current = parent;
                        }
                        _ => break,
                    }
                }
                true
            })
            .collect()
    }

    // ─── Ground ─────────────────────────────────────────────────────────────

    /// Live terrains, their models set for ray casts.
    fn prepare_terrains(&mut self) -> Vec<u32> {
        let terrains: Vec<(u32, glam::Mat4)> = self
            .world
            .iter_with::<TerrainComponent>()
            .filter(|(_, terrain)| terrain.terrain_id != u32::MAX)
            .map(|(e, terrain)| {
                (terrain.terrain_id, self.world.get::<Transform>(e).map_or(glam::Mat4::IDENTITY, Transform::to_matrix))
            })
            .collect();
        let Some(renderer) = self.renderer.as_mut() else {
            return Vec::new();
        };
        terrains
            .into_iter()
            .filter_map(|(id, model)| {
                renderer.terrain_mut(id)?.model = model;
                Some(id)
            })
            .collect()
    }

    /// The object's lowest vertices in the world, from its meshes' BVHs; its box's footprint when it has none.
    fn object_base(&mut self, members: &[Entity], min: glam::Vec3, max: glam::Vec3) -> Vec<glam::Vec3> {
        let mut vertices = Vec::new();
        for &member in members {
            let (Some(mesh), Some(model)) = (self.world.get::<MeshComponent>(member).copied(), model_of(&self.world, member))
            else {
                continue;
            };
            if let Some(bvh) = self.bvh_for(mesh) {
                vertices.extend(bvh.corners().map(|corner| model.transform_point3(corner)));
            }
        }
        let base = editor_ground::base_points(&vertices, 9);
        if base.is_empty() { editor_ground::footprint(min, max) } else { base }
    }

    /// What an object with bounds `min..max` and lowest points `base` stands on, leaving `excluded` out.
    fn support_for(
        &mut self,
        base: &[glam::Vec3],
        min: glam::Vec3,
        max: glam::Vec3,
        excluded: &HashSet<Entity>,
        index: &MeshIndex,
        terrains: &[u32],
    ) -> editor_ground::Support {
        let reach = editor_ground::CONTACT + 0.05;
        let column_min = glam::Vec3::new(min.x - reach, min.y - editor_ground::SEARCH_DEPTH, min.z - reach);
        let column_max = max + glam::Vec3::splat(reach);
        let near: Vec<usize> = index.query(column_min, column_max);
        let mut picked: Vec<(Arc<TriangleBvh>, usize)> = Vec::new();
        for i in near {
            let item = &index.items[i];
            if excluded.contains(&item.entity) {
                continue;
            }
            if let Some(bvh) = self.bvh_for(item.mesh) {
                picked.push((bvh, i));
            }
        }
        let mut meshes: Vec<MeshSurface<'_>> = picked
            .iter()
            .map(|(bvh, i)| {
                let item = &index.items[*i];
                MeshSurface { bvh, model: item.model, inverse: item.model.inverse(), min: item.min, max: item.max }
            })
            .collect();
        // Static box colliders: a game's walls and floors when they are colliders, not meshes (TSF's rooms). A box
        // inside the object's own bounds is its own collider, which floats with it and must not hold it up.
        let cube = self.editing.cube.get_or_insert_with(|| Arc::new(unit_cube())).clone();
        let boxes: Vec<(glam::Mat4, glam::Vec3, glam::Vec3)> = self
            .physics
            .as_ref()
            .map(|physics| physics.static_boxes())
            .unwrap_or_default()
            .into_iter()
            .map(|(half, position, rotation)| {
                let model = glam::Mat4::from_scale_rotation_translation(half * 2.0, rotation, position);
                let (lo, hi) = transformed_box(model, glam::Vec3::splat(-0.5), glam::Vec3::splat(0.5));
                (model, lo, hi)
            })
            .filter(|(_, lo, hi)| lo.cmple(column_max).all() && hi.cmpge(column_min).all())
            .filter(|(_, lo, hi)| {
                let inside = lo.cmpge(min - glam::Vec3::splat(0.1)).all() && hi.cmple(max + glam::Vec3::splat(0.1)).all();
                !inside
            })
            .collect();
        meshes.extend(boxes.iter().map(|&(model, lo, hi)| MeshSurface {
            bvh: &cube,
            model,
            inverse: model.inverse(),
            min: lo,
            max: hi,
        }));
        let renderer = self.renderer.as_ref();
        let terrain = move |origin: glam::Vec3, direction: glam::Vec3, max: f32| {
            renderer.and_then(|renderer| terrain_hit(renderer, terrains, origin, direction, max))
        };
        let terrain: &dyn Fn(glam::Vec3, glam::Vec3, f32) -> Option<Hit> = &terrain;
        let ground = Ground { meshes, terrain: Some(terrain) };
        let mut support = editor_ground::support_at(&ground, base, min, max);
        // Set into a wall collider (a sconce, a picture, a ceiling panel resting on the walls): touching, not floating.
        let contact = glam::Vec3::splat(editor_ground::CONTACT);
        support.touching |= boxes.iter().any(|(model, ..)| box_overlaps(*model, min - contact, max + contact));
        support
    }

    /// Drop (and with `align`, first tilt) each selected object onto what is under it, as one undoable step.
    pub(super) fn drop_selection_to_ground(&mut self, align: bool) {
        let objects = self.selected_objects();
        if objects.is_empty() {
            self.toast("Select an object first");
            return;
        }
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        crate::propagate_transforms(&mut self.world);
        let index = MeshIndex::build(&self.world, renderer);
        let terrains = self.prepare_terrains();
        let children = children_map(&self.world);
        let mut commands: Vec<Box<dyn EditorCommand>> = Vec::new();
        let mut moved: Vec<String> = Vec::new();
        let mut left: Vec<String> = Vec::new();
        for object in objects {
            let Some(start) = self.world.get::<Transform>(object).copied() else {
                continue;
            };
            let members = subtree(&children, object);
            let excluded: HashSet<Entity> = members.iter().copied().collect();
            let name = name_of(&self.world, object);
            let Some(renderer) = self.renderer.as_ref() else {
                return;
            };
            let Some((mut min, mut max)) = world_bounds(&self.world, renderer, &members) else {
                left.push(format!("{name} (no mesh)"));
                continue;
            };
            let parent_world = crate::editor_gizmo::parent_world_matrix(&self.world, object);
            let Some(parent_inverse) = crate::editor_gizmo::invert_affine(parent_world) else {
                left.push(format!("{name} (its parent is squashed flat)"));
                continue;
            };
            let mut transform = start;
            if align {
                let base = self.object_base(&members, min, max);
                let support = self.support_for(&base, min, max, &excluded, &index, &terrains);
                if support.gap.is_some() {
                    let world_model = model_of(&self.world, object).unwrap_or(glam::Mat4::IDENTITY);
                    let (_, world_rotation, _) = world_model.to_scale_rotation_translation();
                    let up = world_rotation * glam::Vec3::Y;
                    let tilt = glam::Quat::from_rotation_arc(up, support.normal);
                    let (_, parent_rotation, _) = parent_world.to_scale_rotation_translation();
                    transform.rotation = (parent_rotation.inverse() * tilt * world_rotation).normalize();
                    // Measure the tilted object where it now sits.
                    if let Some(t) = self.world.get_mut::<Transform>(object) {
                        *t = transform;
                    }
                    crate::propagate_transforms(&mut self.world);
                    let renderer = self.renderer.as_ref().expect("checked above");
                    if let Some(bounds) = world_bounds(&self.world, renderer, &members) {
                        (min, max) = bounds;
                    }
                }
            }
            let base = self.object_base(&members, min, max);
            let support = self.support_for(&base, min, max, &excluded, &index, &terrains);
            match editor_ground::drop_offset(&support) {
                Some(offset) => {
                    let world_now = model_of(&self.world, object).unwrap_or(glam::Mat4::IDENTITY).w_axis.truncate();
                    transform.translation = parent_inverse.transform_point3(world_now - glam::Vec3::Y * offset);
                    moved.push(format!("{name} {:+.2} m", -offset));
                }
                None if support.gap.is_none() => left.push(format!("{name} (nothing under it)")),
                None if !align => left.push(format!("{name} (already resting)")),
                None => moved.push(format!("{name} (tilted)")),
            }
            // Put it back; the command applies the change, so undo has both ends.
            if let Some(t) = self.world.get_mut::<Transform>(object) {
                *t = start;
            }
            let unchanged = transform.translation == start.translation
                && transform.rotation == start.rotation
                && transform.scale == start.scale;
            if !unchanged {
                commands.push(Box::new(SetTransformCmd::new(object.index(), start, transform)));
            }
        }
        crate::propagate_transforms(&mut self.world);
        let label = if align { "Align to Surface" } else { "Drop to Ground" };
        if !commands.is_empty() {
            self.undo_stack.push(
                Box::new(CommandGroup::named(label.to_owned(), commands)),
                &mut self.world,
                &mut self.selection.primary,
            );
            self.selection.reconcile();
            self.scene_dirty = true;
            crate::propagate_transforms(&mut self.world);
        }
        let mut message = format!("{label}: ");
        if !moved.is_empty() {
            message.push_str(&summarise(&moved));
        }
        if !left.is_empty() {
            if !moved.is_empty() {
                message.push_str("; ");
            }
            message.push_str("left ");
            message.push_str(&summarise(&left));
        }
        self.toast(&message);
    }

    /// Select every object within `FLOAT_RADIUS` of the camera that hangs in the air, worst first in the log.
    pub(super) fn find_floating_objects(&mut self) {
        let started = std::time::Instant::now();
        let Some(renderer) = self.renderer.as_ref() else {
            return;
        };
        let camera = renderer.camera_pos;
        crate::propagate_transforms(&mut self.world);
        let index = MeshIndex::build(&self.world, renderer);
        let terrains = self.prepare_terrains();
        let children = children_map(&self.world);
        let mut objects: Vec<Entity> = index
            .items
            .iter()
            .filter(|item| {
                let centre = (item.min + item.max) * 0.5;
                glam::Vec2::new(centre.x - camera.x, centre.z - camera.z).length() <= FLOAT_RADIUS
            })
            .map(|item| top_ancestor(&self.world, item.entity))
            .collect();
        objects.sort_unstable_by_key(|e| e.index());
        objects.dedup();
        let checked = objects.len();
        let mut found: Vec<(f32, Entity, String)> = Vec::new();
        for object in objects {
            if crate::is_hidden(&self.world, object) {
                continue;
            }
            let members = subtree(&children, object);
            if members.len() > CONTAINER_PARTS {
                continue;
            }
            let Some(renderer) = self.renderer.as_ref() else {
                return;
            };
            let Some((min, max)) = world_bounds(&self.world, renderer, &members) else {
                continue;
            };
            if (max.x - min.x).max(max.z - min.z) > editor_ground::PROP_SPAN {
                continue;
            }
            let excluded: HashSet<Entity> = members.iter().copied().collect();
            let base = self.object_base(&members, min, max);
            let support = self.support_for(&base, min, max, &excluded, &index, &terrains);
            if editor_ground::floats(&support, min, max) {
                found.push((support.gap.unwrap_or_default(), object, name_of(&self.world, object)));
            }
        }
        found.sort_by(|a, b| b.0.total_cmp(&a.0));
        self.editing.floating = found.iter().filter_map(|(_, e, _)| self.world.persistent_id(*e)).collect();
        let ms = started.elapsed().as_millis();
        if found.is_empty() {
            self.toast(&format!("No floating objects within {FLOAT_RADIUS:.0} m ({checked} objects checked, {ms} ms)"));
            return;
        }
        for (gap, object, name) in &found {
            info!(gap = *gap, entity = object.index(), "floating: {name}");
        }
        if let Some(ui) = self.ui_manager.as_mut() {
            ui.append_log(&format!("[floating] {} objects within {FLOAT_RADIUS:.0} m of the camera:", found.len()));
            for (gap, _, name) in found.iter().take(200) {
                ui.append_log(&format!("[floating]   {gap:.2} m  {name}"));
            }
        }
        let (worst_gap, _, worst) = &found[0];
        let message = format!(
            "{} floating object{} selected (worst {worst_gap:.2} m: {worst}); the Outliner's \"floating\" search \
             lists them. Look before pressing End: it drops every one selected.",
            found.len(),
            if found.len() == 1 { "" } else { "s" }
        );
        self.selection.set_many(found.iter().map(|(_, e, _)| *e));
        self.after_selection_change();
        self.toast(&message);
    }

    // ─── Hide, isolate, find ────────────────────────────────────────────────

    /// Hide every part of the selected objects, or show them all when they are all hidden. Undoable and saved.
    pub(super) fn toggle_hide_selection(&mut self) {
        let children = children_map(&self.world);
        let members: Vec<Entity> =
            self.selected_objects().into_iter().flat_map(|object| subtree(&children, object)).collect();
        if members.is_empty() {
            return;
        }
        let hide = !members.iter().all(|e| crate::is_hidden(&self.world, *e));
        let commands: Vec<Box<dyn EditorCommand>> = members
            .iter()
            .filter_map(|&entity| {
                let before = self.world.get::<EditorFlags>(entity).copied().unwrap_or_default();
                let after = EditorFlags { hidden: hide, ..before };
                (before != after).then(|| Box::new(SetEditorFlagsCmd::new(entity, before, after)) as Box<dyn EditorCommand>)
            })
            .collect();
        if commands.is_empty() {
            return;
        }
        let label = if hide { "Hide" } else { "Show" };
        self.undo_stack
            .push(Box::new(CommandGroup::named(label.to_owned(), commands)), &mut self.world, &mut self.selection.primary);
        self.selection.reconcile();
        self.scene_dirty = true;
        self.toast(&format!("{label} {} part{}", members.len(), if members.len() == 1 { "" } else { "s" }));
    }

    /// Hide every mesh and decal outside the selected objects until Show Everything. Pressed again, it ends.
    pub(super) fn isolate_selection(&mut self) {
        if !self.editing.isolated.is_empty() {
            self.end_isolation();
            return;
        }
        let children = children_map(&self.world);
        let mut keep: HashSet<Entity> = HashSet::new();
        for object in self.selected_objects() {
            keep.extend(subtree(&children, object));
        }
        if keep.is_empty() {
            return;
        }
        let others: Vec<Entity> = self
            .world
            .entities_with::<MeshComponent>()
            .chain(self.world.entities_with::<crate::decal::DecalComponent>())
            .filter(|e| !keep.contains(e) && !crate::is_hidden(&self.world, *e))
            .collect();
        for entity in others {
            let before = self.world.get::<EditorFlags>(entity).copied();
            self.editing.isolated.push((entity, before));
        }
        self.apply_isolation();
        self.toast("Isolated: Alt+H (or Shift+H again) shows everything. Isolation is never saved.");
    }

    fn apply_isolation(&mut self) {
        for (entity, before) in &self.editing.isolated {
            let flags = EditorFlags { hidden: true, ..before.unwrap_or_default() };
            if let Some(current) = self.world.get_mut::<EditorFlags>(*entity) {
                *current = flags;
            } else if self.world.is_alive(*entity) {
                let _ = self.world.insert_component(*entity, flags);
            }
        }
    }

    fn restore_isolated(&mut self) {
        for (entity, before) in &self.editing.isolated {
            if !self.world.is_alive(*entity) {
                continue;
            }
            match before {
                Some(flags) => {
                    if let Some(current) = self.world.get_mut::<EditorFlags>(*entity) {
                        *current = *flags;
                    }
                }
                None => {
                    let _ = self.world.remove_component::<EditorFlags>(*entity);
                }
            }
        }
    }

    /// Show what Isolate hid. Also runs before Play and before any whole-scene save.
    pub(super) fn end_isolation(&mut self) {
        if self.editing.isolated.is_empty() {
            return;
        }
        self.restore_isolated();
        self.editing.isolated.clear();
    }

    /// Run `f` with the isolation flags put back, so snapshots never record them.
    fn with_isolation_lifted<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        if self.editing.isolated.is_empty() {
            return f(self);
        }
        self.restore_isolated();
        let result = f(self);
        self.apply_isolation();
        result
    }

    /// Scroll the Outliner to the primary selection and open the rows above it.
    pub(super) fn reveal_selection_in_outliner(&mut self, quiet: bool) {
        let Some(entity) = self.selection.primary else {
            return;
        };
        let mut ancestors = Vec::new();
        let mut current = entity;
        for _ in 0..64 {
            match self.world.get::<Parent>(current).map(|p| p.entity) {
                Some(parent) if self.world.is_alive(parent) => {
                    ancestors.push(parent.index());
                    current = parent;
                }
                _ => break,
            }
        }
        let shown = self.ui_manager.as_mut().is_some_and(|ui| ui.reveal_outliner_row(entity.index(), &ancestors));
        if !shown && !quiet {
            self.toast("The Outliner's search hides it: clear the search to see it there");
        }
    }

    /// Select the primary's parent.
    pub(super) fn select_parent(&mut self) {
        let parent = self
            .selection
            .primary
            .and_then(|e| self.world.get::<Parent>(e).map(|p| p.entity))
            .filter(|p| self.world.is_alive(*p));
        match parent {
            Some(parent) => {
                self.selection.set_single(Some(parent));
                self.after_selection_change();
                self.reveal_selection_in_outliner(true);
            }
            None => self.toast("It is already at the top"),
        }
    }

    // ─── The selection card ─────────────────────────────────────────────────

    /// What the primary selection is, where it came from and what can be done to it, several lines.
    pub(super) fn selection_card(&mut self) -> Option<String> {
        if self.selection.primary.is_some() {
            self.editing.painted = None;
            self.editing.cover = None;
        } else if let Some(pick) = self.editing.painted {
            return self.painted_card(pick);
        } else if let Some(pick) = self.editing.cover {
            return self.cover_card(pick);
        }
        let entity = self.selection.primary.filter(|e| self.world.is_alive(*e))?;
        let id = self.world.persistent_id(entity);
        let key = CardKey {
            entity,
            generation: self.undo_stack.generation(),
            count: self.selection.len(),
            edited: id.is_some_and(|id| self.editing.edited.contains(&id)),
            dirty: self.scene_dirty,
        };
        if let Some((cached, card)) = &self.editing.card
            && *cached == key
        {
            return Some(card.clone());
        }
        let card = self.build_card(entity);
        info!("selection card:
{card}");
        self.editing.card = Some((key, card.clone()));
        Some(card)
    }

    /// `SOMNIUM_EDITOR_SCRIPT`: scripted editor input for proof captures in a background window, inert unless set.
    /// Steps are `<frame>:<action>`, separated by `;`: `pick=<x>,<y>` (a click at that fraction of the window),
    /// `select=<name part>` (the first entity whose name contains it), `cmd=<command id>`, `save`, `close`, `splash`,
    /// `gpu_error`, `panic` (a crash on the main thread), `native_crash` (an access violation),
    /// `eye=<x>,<z>,<yaw>,<pitch>` (the editor camera 1.7 m over the loaded ground at scene x, z), and `hour=<h>` (the
    /// scene's Time of Day set to that hour: the editor's clock stands still, so one launch can shoot a day).
    pub(super) fn run_editor_script(&mut self) {
        static SCRIPT: std::sync::LazyLock<Vec<(u64, String)>> = std::sync::LazyLock::new(|| {
            std::env::var("SOMNIUM_EDITOR_SCRIPT")
                .unwrap_or_default()
                .split(';')
                .filter_map(|step| {
                    let (frame, action) = step.trim().split_once(':')?;
                    Some((frame.trim().parse().ok()?, action.trim().to_owned()))
                })
                .collect()
        });
        if SCRIPT.is_empty() {
            return;
        }
        self.editing.script_frame += 1;
        let frame = self.editing.script_frame;
        if std::mem::take(&mut self.editing.script_release) {
            self.end_gizmo_drag();
        }
        for (_, action) in SCRIPT.iter().filter(|(at, _)| *at == frame) {
            info!(frame, action = %action, "editor script");
            match action.split_once('=') {
                Some(("pick", at)) => {
                    let Some((x, y)) = at.split_once(',').and_then(|(x, y)| Some((x.trim().parse::<f32>().ok()?, y.trim().parse::<f32>().ok()?)))
                    else {
                        warn!(action = %action, "editor script: pick wants x,y fractions");
                        continue;
                    };
                    let (width, height) = self.viewport_size();
                    self.cursor_pos = (x * width, y * height);
                    let hit = self.click_pick();
                    info!(picked = %hit.map_or_else(|| "nothing".to_owned(), |e| name_of(&self.world, e)), "editor script");
                    self.selection.set_single(hit);
                    self.after_selection_change();
                    self.reveal_selection_in_outliner(true);
                }
                None if action == "native_crash" => {
                    // A native crash (an access violation, as a driver's arrives), for the minidump's proof run.
                    crate::crash_report::raise_native_crash();
                }
                None if action == "panic" => {
                    // A crash on the main thread, for the crash report's proof run.
                    panic!("editor script: a panic on the main thread, asked for by the proof run");
                }
                Some(("pose", at)) => {
                    // The editor camera at scene x, y, z, turned to yaw and pitch (degrees; yaw 0 looks along +x,
                    // 90 along +z). Asks for no ground: a view from the air, or a first stand somewhere whose ground
                    // has yet to stream in (`eye` refuses a place with none loaded).
                    let numbers: Vec<f32> = at.split(',').filter_map(|v| v.trim().parse::<f32>().ok()).collect();
                    let [x, y, z, yaw, pitch] = numbers[..] else {
                        warn!(action = %action, "editor script: pose wants x,y,z,yaw,pitch");
                        continue;
                    };
                    self.camera_pose_request = Some((glam::Vec3::new(x, y, z), yaw, pitch.clamp(-89.0, 89.0)));
                }
                Some(("eye", at)) => {
                    // The editor camera 1.7 m over the ground at scene x, z, turned to yaw and pitch (degrees).
                    let numbers: Vec<f32> = at.split(',').filter_map(|v| v.trim().parse::<f32>().ok()).collect();
                    let [x, z, yaw, pitch] = numbers[..] else {
                        warn!(action = %action, "editor script: eye wants x,z,yaw,pitch");
                        continue;
                    };
                    match self.ground_height(x, z) {
                        Some(y) => {
                            info!(x, z, ground = y, yaw, pitch, "editor script: eye over the ground");
                            self.camera_pose_request = Some((glam::Vec3::new(x, y + 1.7, z), yaw, pitch.clamp(-89.0, 89.0)));
                        }
                        None => warn!(x, z, "editor script: no ground loaded under the eye"),
                    }
                }
                Some(("hour", h)) => match h.trim().parse::<f32>() {
                    Ok(hour) => {
                        let clocks: Vec<_> = self.world.entities_with::<crate::time_of_day::TimeOfDayComponent>().collect();
                        for e in clocks {
                            if let Some(t) = self.world.get_mut::<crate::time_of_day::TimeOfDayComponent>(e) {
                                t.hour = hour.rem_euclid(24.0);
                            }
                        }
                    }
                    Err(_) => warn!(action = %action, "editor script: hour wants a number"),
                },
                // The broken sun's form (`TimeOfDay.sun_wreck`: -1 whole, 0 cut, 1 shards, 2 bars), so one launch can
                // show them all. `SOMNIUM_SUN_FORM` still wins.
                Some(("form", f)) => match f.trim().parse::<i32>() {
                    Ok(form) => {
                        let clocks: Vec<_> = self.world.entities_with::<crate::time_of_day::TimeOfDayComponent>().collect();
                        for e in clocks {
                            if let Some(t) = self.world.get_mut::<crate::time_of_day::TimeOfDayComponent>(e) {
                                t.sun_wreck = form.clamp(-1, 2);
                            }
                        }
                    }
                    Err(_) => warn!(action = %action, "editor script: form wants a number"),
                },
                // A picture of the scene alone (no editor) at this frame, so one launch can take several. With
                // `SOMNIUM_CAPTURE_STILL_HDR=1` each also leaves its linear `.somcap`.
                Some(("shot", path)) => {
                    if let Some(renderer) = self.renderer.as_mut() {
                        if let Err(error) = renderer.request_authoring_capture(std::path::Path::new(path), false) {
                            warn!(path, %error, "editor script: shot not taken");
                        }
                    }
                }
                // A debug view by its id (`somnium_ui::debug::DEBUG_VIEWS`): lit, sun_only, ambient_only, albedo, ...
                // Or by its number, for the shaders' views that have no menu entry.
                Some(("view", id)) => {
                    match somnium_ui::debug::debug_view(id).map(|view| view.code).or_else(|| id.trim().parse().ok()) {
                        Some(code) => {
                            self.terrain_debug_view = code;
                            if let Some(renderer) = self.renderer.as_mut() {
                                renderer.shading_debug = code;
                            }
                        }
                        None => warn!(id, "editor script: no such debug view"),
                    }
                }
                // A gizmo gesture on the selection, through the code a mouse drag runs: `drag=turn,y,180` (degrees),
                // `drag=move,x,1.5` (metres), `drag=scale,y,2` (times). The pointer is put on the handle and pressed,
                // taken to where the gesture ends, and let go on the next frame.
                Some(("drag", what)) => {
                    let parts: Vec<&str> = what.split(',').map(str::trim).collect();
                    let gesture = match parts[..] {
                        [mode, axis, amount] => {
                            let mode = match mode {
                                "move" => Some(GizmoMode::Translate),
                                "turn" => Some(GizmoMode::Rotate),
                                "scale" => Some(GizmoMode::Scale),
                                _ => None,
                            };
                            let axis = match axis {
                                "x" => Some(GizmoAxis::X),
                                "y" => Some(GizmoAxis::Y),
                                "z" => Some(GizmoAxis::Z),
                                _ => None,
                            };
                            mode.zip(axis).zip(amount.parse::<f32>().ok())
                        }
                        _ => None,
                    };
                    match gesture {
                        Some(((mode, axis), amount)) if self.script_drag(mode, axis, amount) => {}
                        Some(_) => warn!(action = %action, "editor script: no gizmo handle could be taken (is something selected, and in view?)"),
                        None => warn!(action = %action, "editor script: drag wants move|turn|scale, x|y|z, amount"),
                    }
                }
                Some(("select", part)) => {
                    // `#` and the start of a persistent id names one entity among the many of one name.
                    if let Some(id) = part.strip_prefix('#') {
                        let found = self
                            .world
                            .entities()
                            .find(|e| self.world.persistent_id(*e).is_some_and(|p| p.to_string().starts_with(id)));
                        if found.is_none() {
                            warn!(id, "editor script: nothing loaded has that id");
                        }
                        self.selection.set_single(found);
                        self.after_selection_change();
                        self.reveal_selection_in_outliner(true);
                        continue;
                    }
                    let found = self.world.iter_with::<Name>().find(|(_, name)| name.as_str().contains(part)).map(|(e, _)| e);
                    if found.is_none() {
                        warn!(part, "editor script: nothing by that name");
                    }
                    self.selection.set_single(found);
                    self.after_selection_change();
                    self.reveal_selection_in_outliner(true);
                }
                Some(("cmd", id)) => {
                    let ran = self.ui_manager.as_mut().is_some_and(|ui| ui.run_command_id(id));
                    if !ran {
                        warn!(id, "editor script: no such command, or it is disabled");
                    }
                }
                None if action == "save" => self.handle_editor_event(EditorEvent::SaveScene),
                // The title bar's X: with unsaved edits, the editor asks before it closes.
                None if action == "close" => self.handle_editor_event(EditorEvent::CloseWindow),
                // The launch splash again, held (never told to finish), for a picture of its reveal.
                None if action == "splash" => {
                    if let Some(ui) = self.ui_manager.as_mut() {
                        ui.begin_splash("Loading the world");
                    }
                }
                // One deliberately invalid GPU call (a buffer both read- and write-mapped), for the proof that the
                // editor logs a GPU error, names the GPU, and stays up.
                None if action == "gpu_error" => {
                    if let Some(ctx) = self.render_ctx.as_ref() {
                        use somnium_ui::wgpu;
                        drop(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                            label: Some("editor script: deliberately invalid"),
                            size: 4,
                            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::MAP_WRITE,
                            mapped_at_creation: false,
                        }));
                    }
                }
                _ => warn!(action = %action, "editor script: unknown step"),
            }
        }
    }

    fn build_card(&mut self, entity: Entity) -> String {
        let root = self.config.project_root.clone();
        let mut lines: Vec<String> = Vec::new();
        let object = top_ancestor(&self.world, entity);
        let name = name_of(&self.world, entity);
        let id = self.world.persistent_id(entity);
        let mut title = if object == entity {
            name.clone()
        } else {
            format!("{}  \u{203a}  {name}", name_of(&self.world, object))
        };
        if self.selection.len() > 1 {
            title.push_str(&format!("   (+{} more selected)", self.selection.len() - 1));
        }
        if id.is_some_and(|id| self.editing.edited.contains(&id)) {
            title.push_str("   [edited]");
        }
        if id.is_some_and(|id| self.editing.floating.contains(&id)) {
            title.push_str("   [floating]");
        }
        lines.push(title);

        // Mesh, its size, and what it is made of.
        if let Some(imported) = self.world.get::<crate::ImportedMesh>(entity).cloned() {
            let mut line = format!("mesh      {}  part {}", imported.source, imported.node);
            if let Some(size) = self.size_of(entity) {
                line.push_str(&format!("  ({size})"));
            }
            lines.push(line);
            if let Some(part) = self.gltf_part(&imported.source, imported.node as usize) {
                if !part.node.is_empty() || !part.mesh.is_empty() {
                    lines.push(format!("node      \"{}\"  mesh \"{}\"", part.node, part.mesh));
                }
                let material = self.material_label(entity).unwrap_or_else(|| format!("\"{}\" (from the model)", part.material));
                lines.push(format!("material  {material}"));
                if !part.images.is_empty() {
                    lines.push(format!("textures  {}", part.images.join(", ")));
                }
            } else if let Some(material) = self.material_label(entity) {
                lines.push(format!("material  {material}"));
            }
        } else if self.world.get::<MeshComponent>(entity).is_some() {
            let kind = self.world.get::<MeshKind>(entity).map_or("mesh".to_owned(), |kind| format!("{kind:?}"));
            let size = self.size_of(entity).map(|s| format!("  ({s})")).unwrap_or_default();
            lines.push(format!("mesh      built-in {kind}{size}"));
            if let Some(material) = self.material_label(entity) {
                lines.push(format!("material  {material}"));
            }
        } else {
            let children = children_map(&self.world);
            let members = subtree(&children, entity);
            let parts = members.len() - 1;
            // What the parts draw: the asset files an object is made of.
            let mut sources: Vec<String> = Vec::new();
            for member in &members {
                if let Some(imported) = self.world.get::<crate::ImportedMesh>(*member)
                    && !sources.contains(&imported.source)
                {
                    sources.push(imported.source.clone());
                }
            }
            match sources.as_slice() {
                [] => {}
                [one] => lines.push(format!("mesh      {one}")),
                [first, rest @ ..] => lines.push(format!("mesh      {first} and {} more file{}", rest.len(), if rest.len() == 1 { "" } else { "s" })),
            }
            let size = self
                .renderer
                .as_ref()
                .and_then(|r| world_bounds(&self.world, r, &members))
                .map(|(lo, hi)| {
                    let s = hi - lo;
                    format!(", {:.2} x {:.2} x {:.2} m", s.x, s.y, s.z)
                })
                .unwrap_or_default();
            lines.push(format!("object    {parts} part{}{size}", if parts == 1 { "" } else { "s" }));
        }

        // Whether its surface is closed: a rock or cliff open on a side is hollow from behind.
        if let Some(line) = self.shape_line(entity) {
            lines.push(line);
        }

        // Where it came from.
        let streamed = self.world.get::<crate::cell_streaming::Streamed>(entity).copied();
        let file = match streamed.and_then(|s| self.cell_streamer.cells.get(s.cell as usize)) {
            Some(cell) => format!("cell {}", project_relative(root.as_deref(), &cell.file)),
            None => self
                .editing
                .scene
                .as_ref()
                .map(|scene| project_relative(root.as_deref(), scene))
                .unwrap_or_else(|| "(a scene with no file yet)".to_owned()),
        };
        let added = id.is_some_and(|id| {
            self.editing.layer.as_ref().and_then(|l| l.entry(&id.to_string())).is_some_and(|e| e.op == EditOp::Add)
                || (!self.editing.base.contains(&id) && streamed.is_none())
        });
        lines.push(if added { "file      added in the editor (kept in the edits file)".to_owned() } else { format!("file      {file}") });
        if let Some(made) = self.made_by_for(entity, &file) {
            lines.push(format!("made by   {made}"));
        }
        lines.push(format!(
            "id        entity {}  persistent {}",
            entity.index(),
            id.map_or_else(|| "(none yet)".to_owned(), |id| id.to_string())
        ));
        if let Some(line) = self.edit_line(entity) {
            lines.push(line);
        }
        lines.push("End drop to ground   Shift+End align   H hide   Shift+H isolate   Ctrl+Shift+F floating".to_owned());
        lines.join("\n")
    }

    fn size_of(&self, entity: Entity) -> Option<String> {
        let renderer = self.renderer.as_ref()?;
        let (lo, hi) = world_bounds(&self.world, renderer, &[entity])?;
        let s = hi - lo;
        Some(format!("{:.2} x {:.2} x {:.2} m", s.x, s.y, s.z))
    }

    fn material_label(&self, entity: Entity) -> Option<String> {
        let material = self.world.get::<MaterialComponent>(entity)?;
        if material.asset.raw() == 0 {
            return None;
        }
        let record = self.asset_gate.published().and_then(|db| db.get(material.asset));
        Some(record.map_or_else(|| format!("asset {:032x} (not in the content folder)", material.asset.raw()), |r| r.relative_path.clone()))
    }

    fn gltf_part(&mut self, source: &str, node: usize) -> Option<somnium_asset::GltfPart> {
        if !self.editing.parts.contains_key(source) {
            let parts = self
                .imported_source_path(source)
                .ok()
                .and_then(|path| somnium_asset::describe_gltf_parts(&path).ok())
                .map(Arc::new);
            self.editing.parts.insert(source.to_owned(), parts);
        }
        self.editing.parts.get(source)?.as_ref()?.get(node).cloned()
    }

    /// Who made it: a cell's per-object label, the file's own `made_by`, or the tools that write its name's prefix.
    fn made_by_for(&mut self, entity: Entity, file: &str) -> Option<String> {
        let object = top_ancestor(&self.world, entity);
        for candidate in [entity, object] {
            if let Some(label) = self.world.persistent_id(candidate).and_then(|id| self.editing.sources.get(&id)) {
                return Some(label.clone());
            }
        }
        let file = file.trim_start_matches("cell ");
        if let Some(label) = self.editing.made_by.get(file) {
            return Some(label.clone());
        }
        let name = name_of(&self.world, object);
        let prefix = name.split(" / ").next().filter(|p| p.len() >= 4 && *p != name)?.to_owned();
        if !self.editing.prefix_tools.contains_key(&prefix) {
            let tools = self.tools_writing(&prefix);
            self.editing.prefix_tools.insert(prefix.clone(), tools);
        }
        let tools = self.editing.prefix_tools.get(&prefix)?;
        match tools.as_slice() {
            [] => None,
            [one] => Some(format!("{one} (it names its objects \"{prefix} / ...\")")),
            [first, rest @ ..] => Some(format!(
                "{first} (it names its objects \"{prefix} / ...\"; {} more tool{} touch them)",
                rest.len(),
                if rest.len() == 1 { "" } else { "s" }
            )),
        }
    }

    /// The project's tools whose source mentions `prefix`, likeliest maker first: a tool that sets the prefix as a
    /// constant (`TAG = 'Furnish 20261005'`), then one whose file name shares a word with it, then the oldest, since
    /// the maker predates the fixers and audits that mention its objects later.
    fn tools_writing(&self, prefix: &str) -> Vec<String> {
        let Some(root) = self.config.project_root.as_ref() else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(root.join("tools")) else {
            return Vec::new();
        };
        let mut found: Vec<(u8, std::time::SystemTime, String)> = entries
            .filter_map(Result::ok)
            .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "py"))
            .filter_map(|entry| {
                let text = std::fs::read_to_string(entry.path()).ok()?;
                let stem = entry.path().file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let score = maker_score(prefix, &stem, &text)?;
                let when = entry.metadata().and_then(|m| m.modified()).unwrap_or(std::time::UNIX_EPOCH);
                Some((score, when, format!("tools/{}", entry.file_name().to_string_lossy())))
            })
            .collect();
        found.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        found.into_iter().map(|(_, _, name)| name).collect()
    }

    /// What has been done to it: saved in the layer, or changed since it was first selected.
    fn edit_line(&mut self, entity: Entity) -> Option<String> {
        let id = self.world.persistent_id(entity)?;
        let hex = id.to_string();
        let saved = self.editing.layer.as_ref().and_then(|layer| layer.entry(&hex)).cloned();
        let now = self.world.get::<Transform>(entity).map(|t| t.translation);
        let unsaved = self.editing.edited.contains(&id) && self.scene_dirty;
        let file = self
            .editing
            .scene
            .as_ref()
            .map(|s| scene_edits::layer_path(s))
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_default();
        match (saved, unsaved) {
            (Some(entry), _) if entry.op == EditOp::Change => {
                let moved = entry.was.zip(now).map(|(was, now)| (glam::Vec3::from_array(was) - now).length());
                let what = moved.filter(|m| *m > 0.005).map_or_else(|| "changed".to_owned(), |m| format!("moved {m:.2} m"));
                Some(format!("edits     {what} since the generator made it; kept in {file}{}", if unsaved { " (more unsaved)" } else { "" }))
            }
            (Some(_), _) => Some(format!("edits     added in the editor; kept in {file}")),
            (None, true) => Some("edits     changed, not saved yet: Ctrl+S keeps it beside the scene".to_owned()),
            (None, false) => None,
        }
    }

    /// The Outliner row's marks: `edited`, `floating`, and the asset path `asset:` searches.
    pub(super) fn decorate_outliner_row(&self, entity: Entity, row: &mut somnium_ui::OutlinerRow) {
        if let Some(id) = self.world.persistent_id(entity) {
            if self.editing.edited.contains(&id) {
                row.tags.push("edited");
                row.name.push_str("  [edited]");
            }
            if self.editing.floating.contains(&id) {
                row.tags.push("floating");
            }
        }
        row.asset.clear();
        if let Some(imported) = self.world.get::<crate::ImportedMesh>(entity) {
            row.asset.push_str(&imported.source);
        }
    }

    /// Put the card on the clipboard.
    pub(super) fn copy_identity(&mut self) {
        match self.selection_card() {
            Some(card) => match copy_to_clipboard(&card) {
                Ok(()) => self.toast("Copied what it is"),
                Err(error) => self.toast(&format!("Could not copy: {error}")),
            },
            None => self.toast("Select something first"),
        }
    }
}

/// Every persistent id in a scene document.
fn ids_in(document: &Value) -> HashSet<PersistentId> {
    document["entities"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|entity| PersistentId::parse_hex(entity.get("persistent_id")?.as_str()?))
        .collect()
}

/// A document's own note of the tool that wrote it.
fn made_by(document: &Value) -> Option<String> {
    [&document["made_by"], &document["generated_by"], &document["rimeborn"]["made_by"]]
        .into_iter()
        .find_map(|value| value.as_str().map(str::to_owned))
}

/// How likely a tool (file `stem`, source `text`) is to be the one that makes objects named `prefix / ...`: 2 for
/// setting it as a constant (`TAG = 'Furnish 20261005'`), 1 for a file name sharing a word with it; `None` when it
/// never mentions it.
fn maker_score(prefix: &str, stem: &str, text: &str) -> Option<u8> {
    if !text.contains(prefix) {
        return None;
    }
    let sets_it = text.lines().any(|line| {
        let Some((name, value)) = line.trim_start().split_once('=') else {
            return false;
        };
        let name = name.trim();
        !name.is_empty()
            && name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            && value.trim_start().trim_start_matches('f').trim_start_matches(['\'', '"']).starts_with(prefix)
    });
    let stem = stem.to_ascii_lowercase();
    let named = prefix
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|word| word.len() >= 4)
        .any(|word| stem.contains(&word.to_ascii_lowercase()));
    Some(u8::from(sets_it) * 2 + u8::from(named))
}

/// A short list for a toast: the first few, then a count.
fn summarise(items: &[String]) -> String {
    let shown: Vec<&str> = items.iter().take(3).map(String::as_str).collect();
    let mut text = shown.join(", ");
    if items.len() > 3 {
        text.push_str(&format!(" and {} more", items.len() - 3));
    }
    text
}

/// A decal is clicked on its own surface, not anywhere in its projection box: the box is thinned to 4 cm along its
/// shortest axis.
fn decal_slab_distance(world: &World, entity: Entity, origin: glam::Vec3, direction: glam::Vec3) -> Option<f32> {
    let model = model_of(world, entity)?;
    let (scale, _, _) = model.to_scale_rotation_translation();
    let thin = if scale.x <= scale.y && scale.x <= scale.z {
        0
    } else if scale.y <= scale.z {
        1
    } else {
        2
    };
    let mut half = glam::Vec3::splat(0.5);
    half[thin] = (0.02 / scale[thin].abs().max(1e-4)).min(0.5);
    let inverse = invert_affine(model)?;
    let local_origin = inverse.transform_point3(origin);
    let local_direction = inverse.transform_vector3(direction);
    let t = ray_aabb_distance(local_origin, local_direction, -half, half)?;
    Some(origin.distance(model.transform_point3(local_origin + local_direction * t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_painted_copy_taken_out_stays_out_on_its_own_ground_only() {
        use somnium_renderer::terrain::foliage_paint::PaintedFoliage;
        let copy = |kind: u16, x: f32| PaintedFoliage { kind, position: glam::Vec3::new(x, 2.0, 10.0), yaw: 0.0, tilt: 0.0, scale: 1.0 };
        let edits = vec![scene_edits::PaintedEdit { ground: "scenes/terrain_tiles/t_-14_0.scene.json".into(), kind: 410, at: [5.02, 2.0, 10.0] }];
        let mut copies = vec![copy(410, 5.0), copy(410, 9.0), copy(411, 5.0)];
        assert_eq!(hide_painted(&edits, "scenes/terrain_tiles/t_-14_0.scene.json", &mut copies), 1);
        assert_eq!(copies, vec![copy(410, 9.0), copy(411, 5.0)], "the other copy and the other kind stay");
        let mut other = vec![copy(410, 5.0)];
        assert_eq!(hide_painted(&edits, "scenes/terrain_tiles/t_-13_0.scene.json", &mut other), 0, "another tile");
        // A layer written before painted edits existed still reads; an empty list is not written.
        let old: scene_edits::SceneEdits =
            serde_json::from_str(r#"{"format":"somnium-scene-edits","version":1,"scene":"s","entities":[]}"#).unwrap();
        assert!(old.painted.is_empty());
        assert!(!serde_json::to_string(&old).unwrap().contains("painted"));
    }

    #[test]
    fn a_painted_copy_made_an_object_stands_where_it_stood_and_draws_its_model_again() {
        use crate::editor_commands::EditorCommand as _;
        let placed = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(1.3),
            glam::Quat::from_rotation_y(0.7),
            glam::Vec3::new(-15464.9, 28.2, 273.0),
        );
        let local = glam::Mat4::from_translation(glam::Vec3::new(0.0, 0.4, 0.0));
        let mesh = |offset| MeshComponent { vertex_offset: offset, index_offset: 0, index_count: 36 };
        let pieces = [(mesh(100), 7, 2, local), (mesh(200), 8, 5, glam::Mat4::IDENTITY)];
        let clipboard = painted_object("myrtle shrub", "assets/plants/myrtle.gltf", placed, &pieces);
        let mut world = World::new();
        let mut selected = None;
        let mut paste = crate::clipboard::PasteEntitiesCmd::new(clipboard, None);
        paste.execute(&mut world, &mut selected);
        let [root] = paste.roots() else { panic!("one object") };
        crate::propagate_transforms(&mut world);
        let parts: Vec<Entity> =
            world.iter_with::<Parent>().filter(|(_, parent)| parent.entity == *root).map(|(e, _)| e).collect();
        assert_eq!(parts.len(), 2, "a part for each piece of the model");
        let first = parts.iter().copied().find(|p| world.get::<MeshComponent>(*p).unwrap().vertex_offset == 100).unwrap();
        let drawn = world.get::<WorldTransform>(first).unwrap().0;
        assert!(drawn.abs_diff_eq(placed * local, 1e-3), "the part draws where the painted copy drew it");
        let source = world.get::<crate::ImportedMesh>(first).unwrap();
        assert_eq!((source.source.as_str(), source.node), ("assets/plants/myrtle.gltf", 2), "a reopened scene finds it");
        assert_eq!(world.get::<MaterialComponent>(first).unwrap().runtime_id, 7);
        assert!(world.persistent_id(*root).is_some() && world.persistent_id(first).is_some(), "Ctrl+S can keep it");
    }

    #[test]
    fn a_places_scene_warns_that_its_edits_never_reach_the_world() {
        let world = Path::new("F:/game/scenes/sunscar.scene.json");
        let places = Path::new("F:/game/scenes/places_kept_monargent.scene.json");
        let warning = composed_scene_warning(places, Some(world)).expect("warned");
        assert!(warning.contains("places_kept_monargent.scene.json") && warning.contains("Open sunscar.scene.json"));
        assert_eq!(composed_scene_warning(world, Some(world)), None, "the world itself");
        assert_eq!(composed_scene_warning(Path::new("scenes/town.scene.json"), Some(world)), None, "any other scene");
        assert_eq!(composed_scene_warning(places, None), None, "no startup scene: nothing to point at");
    }

    #[test]
    fn tree_like_is_whole_words_of_the_path() {
        assert!(tree_like("assets/foliage/trees/grown/holm_oak_b.gltf"));
        assert!(tree_like("assets/foliage/trees/grown/wall_fig.gltf"));
        assert!(tree_like("assets/props/Old_Pines_03.glb"));
        assert!(tree_like("assets/props/cypress.glb"));
        assert!(!tree_like("assets/architecture/places/monargent/silver_street_+1_0_01.gltf"));
        assert!(!tree_like("assets/props/airplane_wreck.glb"));
    }

    #[test]
    fn rocks_are_named_by_whole_words_and_a_surface_reads_plainly() {
        assert!(rock_like("assets/rocks/granite_tor_03.gltf"));
        assert!(rock_like("assets/pois/kit/Cliff_Face_B.glb"));
        assert!(rock_like("assets/props/pebbles.gltf"));
        assert!(!rock_like("assets/foliage/ground/pebbles.gltf"), "ground cover, not a placed rock");
        assert!(!rock_like("assets/architecture/stonemason_hut.gltf"), "a whole word, not a part of one");
        assert!(!rock_like("assets/foliage/trees/grown/rock_rose.gltf"), "a shrub");
        assert!(!rock_like("assets/props/stone_pine.gltf"), "a tree");
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(1_234_567), "1,234,567");
        let closed = MeshShape { triangles: 12_400, ..MeshShape::default() };
        assert_eq!(shape_text(&closed), "closed on every side; 12,400 triangles");
        let seated = MeshShape { triangles: 10, open_edges: 4, ..MeshShape::default() };
        assert!(shape_text(&seated).starts_with("open at its base only (4 edges)"));
        let hollow = MeshShape { triangles: 10, open_edges: 1_230, open_above_base: 1_100, degenerate: 2, ..MeshShape::default() };
        let text = shape_text(&hollow);
        assert!(text.starts_with("OPEN on a side or top: 1,230 edges") && text.contains("1,100 above") && text.ends_with("2 slivers"), "{text}");
    }

    #[test]
    fn a_ground_slot_is_named_by_its_own_material_set() {
        let dir = std::env::temp_dir().join(format!("somnium-set-names-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(set_slot_names(&dir).is_empty(), "no SOURCE.json: no names");
        std::fs::write(
            dir.join("SOURCE.json"),
            r#"{"made_by": "tools/look_terrain_set.py", "layers": [
                {"slot": 4, "file": "leafy_grass", "layer": "sunstraw"},
                {"slot": 31, "file": "snow_01", "layer": "moss"},
                {"slot": 300, "layer": "past the slots"},
                {"slot": 5, "layer": "  "}]}"#,
        )
        .unwrap();
        let names = set_slot_names(&dir);
        assert_eq!(names.get(&4).map(String::as_str), Some("sunstraw"));
        assert_eq!(names.get(&31).map(String::as_str), Some("moss"), "the set's name, not the engine's \"Hard Snow\"");
        assert_eq!(names.len(), 2, "a slot past 255 and a blank name are skipped: {names:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn a_cell_path_reads_the_same_however_it_was_joined() {
        let root = Path::new("F:/Somnium Games/rimeborn-fp");
        let joined = Path::new("F:/Somnium Games/rimeborn-fp/scenes/sunscar_cells/../terrain_tiles/t_0_0.scene.json");
        assert_eq!(project_relative(Some(root), joined), "scenes/terrain_tiles/t_0_0.scene.json");
        let cell = Path::new("F:/Somnium Games/rimeborn-fp/scenes/sunscar_cells/s_-64_9.scene.json");
        assert_eq!(project_relative(Some(root), cell), "scenes/sunscar_cells/s_-64_9.scene.json");
    }

    #[test]
    fn the_tool_that_names_a_prefix_is_its_maker_not_the_ones_that_mention_it() {
        let maker = "import json\nTAG = 'Furnish 20261005'\n";
        let fixer = "names = ('Furnish 20261005 / Design studio / chair',)\n";
        assert_eq!(maker_score("Furnish 20261005", "furnish_20261005", maker), Some(3));
        assert_eq!(maker_score("Furnish 20261005", "reach_fix_20261005", fixer), Some(1), "the date in its name");
        assert_eq!(maker_score("Furnish 20261005", "forest_o_20261029", fixer), Some(0));
        assert_eq!(maker_score("Furnish 20261005", "anything", "nothing here"), None);
    }

    #[test]
    fn a_sconce_set_into_a_wall_collider_touches_it() {
        // An 18 cm wall, 2.8 m tall and 12 m long, centred at the origin; a sconce half in it, one 20 cm off it.
        let wall = glam::Mat4::from_scale(glam::Vec3::new(0.18, 2.8, 12.0));
        assert!(box_overlaps(wall, glam::Vec3::new(0.05, 1.0, -0.1), glam::Vec3::new(0.25, 1.2, 0.1)));
        assert!(!box_overlaps(wall, glam::Vec3::new(0.29, 1.0, -0.1), glam::Vec3::new(0.45, 1.2, 0.1)));
    }

    #[test]
    fn a_unit_cube_is_hit_on_its_top_face() {
        let cube = unit_cube();
        let (t, _) = cube.raycast(glam::Vec3::new(0.1, 2.0, -0.2), glam::Vec3::NEG_Y, 10.0).unwrap();
        assert!((t - 1.5).abs() < 1e-5, "{t}");
        let (t, _) = cube.raycast(glam::Vec3::new(-2.0, 0.3, 0.0), glam::Vec3::X, 10.0).unwrap();
        assert!((t - 1.5).abs() < 1e-5, "{t}: and on its side");
    }

    #[test]
    fn a_toast_list_stays_short() {
        let items: Vec<String> = (0..5).map(|i| format!("crate {i}")).collect();
        assert_eq!(summarise(&items), "crate 0, crate 1, crate 2 and 2 more");
    }

    #[test]
    fn made_by_reads_each_place_tools_write_it() {
        let places = serde_json::json!({"entities": [], "rimeborn": {"made_by": "rimeborn-fp/tools/arch_montara.py"}});
        assert_eq!(made_by(&places).as_deref(), Some("rimeborn-fp/tools/arch_montara.py"));
        assert_eq!(made_by(&serde_json::json!({"made_by": "tools/compose.py"})).as_deref(), Some("tools/compose.py"));
        assert_eq!(made_by(&serde_json::json!({"entities": []})), None);
    }

    #[test]
    fn a_decal_is_clicked_on_its_face_not_in_front_of_it() {
        let mut world = World::new();
        // A 2 x 2 m decal projecting 1 m deep along z, at the origin.
        let decal = world.spawn((Transform {
            translation: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::new(2.0, 2.0, 1.0),
        },));
        let d = decal_slab_distance(&world, decal, glam::Vec3::new(0.0, 0.0, 5.0), glam::Vec3::NEG_Z).unwrap();
        assert!((d - 4.98).abs() < 1e-3, "{d}: the slab's face, 2 cm in front of the plane, not the box's at 0.5 m");
    }
}
