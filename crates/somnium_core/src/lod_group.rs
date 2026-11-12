//! Distance levels of detail for imported buildings and props (Rimeborn's WS2-1).
//!
//! A `somnium.LodGroup` on a root picks which one of its direct children draws: the first (in name order, so `LOD0`,
//! `LOD1`, `LOD2` sort as meant) while the camera is nearer than `distances.x`, the second until `distances.y`, the
//! third beyond, and nothing past `distances.z` when that is above zero. Each level is an ordinary subtree of parts
//! (a building's LOD0 with its cutaway groups, its LOD1 shell...). The levels not chosen are skipped at submission,
//! shadows included, through a runtime set: nothing authored or saved changes, and a scene without a `LodGroup`
//! draws exactly as before.

use std::collections::HashMap;

use glam::Vec3;
use somnium_ecs::component_schema;
use somnium_ecs::reflect::TypeRegistry;
use somnium_ecs::{Component, Entity, World};

use crate::{Name, Parent, Transform, WorldTransform};

/// A root whose children are levels of detail.
#[derive(Debug, Clone, PartialEq)]
pub struct LodGroupComponent {
    /// Off: every level draws (for looking at them side by side).
    pub enabled: bool,
    /// Camera distances, metres: x leaves the first level, y the second; past z (if above zero) nothing draws.
    pub distances: Vec3,
}

impl Component for LodGroupComponent {}

impl Default for LodGroupComponent {
    fn default() -> Self {
        Self { enabled: true, distances: Vec3::new(80.0, 200.0, 0.0) }
    }
}

pub(crate) fn register(registry: &mut TypeRegistry) {
    registry.register(component_schema! {
        LodGroupComponent as "somnium.LodGroup", display "LOD Group", version 1,
        fields {
            enabled { group: "LOD", doc: "Off: every level draws." },
            distances { unit: "m", group: "LOD", doc: "Camera distances: x leaves the first child level, y the second; past z (if above zero) nothing draws. Levels are the direct children in name order (LOD0, LOD1, LOD2)." },
        }
    });
}

/// Which level a camera at `distance` draws, of `levels`; `None` past the cull.
#[must_use]
pub fn level_for(group: &LodGroupComponent, distance: f32, levels: usize) -> Option<usize> {
    if levels == 0 || (group.distances.z > 0.0 && distance > group.distances.z) {
        return None;
    }
    let passed = usize::from(distance > group.distances.x) + usize::from(distance > group.distances.y);
    Some(passed.min(levels - 1))
}

/// The level of a `LodGroup` that `entity` sits in, as (the group's root, the level): the nearest group above it.
/// `None` for an entity in no group and for a group's root.
// ponytail: a group inside a group is asked about its inner group only; nothing nests them today.
fn level_of(world: &World, entity: Entity) -> Option<(Entity, Entity)> {
    let mut level = entity;
    loop {
        let parent = world.get::<Parent>(level).map(|p| p.entity)?;
        if parent == Entity::DANGLING || !world.is_alive(parent) {
            return None;
        }
        if world.get::<LodGroupComponent>(parent).is_some() {
            return Some((parent, level));
        }
        level = parent;
    }
}

fn name_of(world: &World, entity: Entity) -> String {
    world.get::<Name>(entity).map_or_else(String::new, |n| n.as_str().to_owned())
}

/// Whether `entity` is in a level of detail that a camera at `camera` is not shown. It is what [`Lods::is_hidden`]
/// says in the draw loop, worked out for one entity from the world alone: a click asks it of the few things under
/// the pointer, so that it takes the copy on screen and not the far copy standing unseen in the same place.
/// `children` maps each parent to its children.
#[must_use]
pub fn hidden_from(world: &World, children: &HashMap<Entity, Vec<Entity>>, entity: Entity, camera: Vec3) -> bool {
    let Some((root, level)) = level_of(world, entity) else { return false };
    let Some(group) = world.get::<LodGroupComponent>(root) else { return false };
    if !group.enabled {
        return false;
    }
    // The levels as `Lods` counts them: the root's direct children by name.
    let mut levels = children.get(&root).cloned().unwrap_or_default();
    levels.sort_by_key(|e| name_of(world, *e));
    let at = world
        .get::<WorldTransform>(root)
        .map(|w| w.0.w_axis.truncate())
        .or_else(|| world.get::<Transform>(root).map(|t| t.translation))
        .unwrap_or(Vec3::ZERO);
    let drawn = level_for(group, at.distance(camera), levels.len()).and_then(|i| levels.get(i).copied());
    drawn != Some(level)
}

/// The same part in a group's other levels: for each of `selected` that is a part of one level of a `LodGroup`, the
/// entity of the same name along the same path under each sibling level. Moved alone, a part leaves its far copy
/// behind, and from thirty metres off it is back where it was. A whole level, a group's root and anything outside a
/// group have no twins; nor has a part whose name two siblings share, since which is meant cannot be said.
#[must_use]
pub fn twins_of(world: &World, children: &HashMap<Entity, Vec<Entity>>, selected: &[Entity]) -> Vec<Entity> {
    let mut twins = Vec::new();
    for &entity in selected {
        let Some((root, level)) = level_of(world, entity) else { continue };
        // The names from the level down to the part.
        let mut path = Vec::new();
        let mut at = entity;
        while at != level {
            path.push(name_of(world, at));
            let Some(parent) = world.get::<Parent>(at).map(|p| p.entity) else { break };
            at = parent;
        }
        if path.is_empty() {
            continue;
        }
        path.reverse();
        for other in children.get(&root).into_iter().flatten().copied().filter(|l| *l != level) {
            let mut found = Some(other);
            for step in &path {
                found = found.and_then(|at| {
                    let mut same = children.get(&at).into_iter().flatten().copied().filter(|c| name_of(world, *c) == *step);
                    let first = same.next();
                    if same.next().is_some() { None } else { first }
                });
            }
            if let Some(twin) = found
                && !selected.contains(&twin)
                && !twins.contains(&twin)
            {
                twins.push(twin);
            }
        }
    }
    twins
}

struct Group {
    root: Entity,
    /// Every entity of each level's subtree, the level's own root first.
    levels: Vec<Vec<Entity>>,
    active: Option<Option<usize>>,
}

/// The levels skipped this frame.
#[derive(Default)]
pub struct Lods {
    groups: Vec<Group>,
    built_for: usize,
    /// By entity index, the generation + 1 of a skipped entity (0: drawn). A region's draw loop asks for every mesh
    /// every frame; a set of every part of every level not chosen cost a cache miss a mesh (`tests/draw_loop_scale.rs`).
    hidden: Vec<u32>,
}

impl Lods {
    /// Choose each group's level for a camera at `camera`; rebuilds the skip set only when a choice changes.
    pub fn update(&mut self, world: &World, camera: Vec3) {
        // ponytail: the level structure is rebuilt when the world's entity count changes; a reparent that keeps the
        // count is picked up at the next change. Hook the hierarchy's edits if that ever matters.
        if self.built_for != world.entity_count() || self.groups.iter().any(|g| !world.is_alive(g.root)) {
            self.rebuild(world);
        }
        let mut changed = false;
        for group in &mut self.groups {
            let Some(settings) = world.get::<LodGroupComponent>(group.root) else { continue };
            let choice = if settings.enabled {
                let at = world
                    .get::<WorldTransform>(group.root)
                    .map(|w| w.0.w_axis.truncate())
                    .or_else(|| world.get::<Transform>(group.root).map(|t| t.translation))
                    .unwrap_or(Vec3::ZERO);
                level_for(settings, at.distance(camera), group.levels.len())
            } else {
                Some(usize::MAX)
            };
            if group.active != Some(choice) {
                group.active = Some(choice);
                changed = true;
            }
        }
        if changed {
            self.hidden.fill(0);
            for group in &self.groups {
                let Some(choice) = group.active else { continue };
                if choice == Some(usize::MAX) {
                    continue;
                }
                for (i, level) in group.levels.iter().enumerate() {
                    if Some(i) != choice {
                        for entity in level {
                            let slot = entity.index() as usize;
                            if slot >= self.hidden.len() {
                                self.hidden.resize(slot + 1, 0);
                            }
                            // ponytail: a generation of u32::MAX wraps to "drawn"; four billion reuses of one slot.
                            self.hidden[slot] = entity.generation().wrapping_add(1);
                        }
                    }
                }
            }
        }
    }

    /// Whether this entity belongs to a level not drawn this frame.
    #[must_use]
    pub fn is_hidden(&self, entity: Entity) -> bool {
        self.hidden.get(entity.index() as usize).is_some_and(|&g| g != 0 && g == entity.generation().wrapping_add(1))
    }

    fn rebuild(&mut self, world: &World) {
        self.built_for = world.entity_count();
        self.hidden.fill(0);
        let roots: Vec<Entity> = world.entities_with::<LodGroupComponent>().collect();
        self.groups.clear();
        if roots.is_empty() {
            return;
        }
        let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for entity in world.entities_with::<Parent>() {
            if let Some(parent) = world.get::<Parent>(entity).map(|p| p.entity) {
                if parent != Entity::DANGLING {
                    children.entry(parent).or_default().push(entity);
                }
            }
        }
        let name = |e: Entity| world.get::<Name>(e).map_or_else(String::new, |n| n.as_str().to_owned());
        for root in roots {
            let mut direct = children.get(&root).cloned().unwrap_or_default();
            direct.sort_by_key(|e| name(*e));
            let levels = direct
                .into_iter()
                .map(|level| {
                    let mut all = vec![level];
                    let mut i = 0;
                    while i < all.len() {
                        if let Some(kids) = children.get(&all[i]) {
                            all.extend(kids.iter().copied());
                        }
                        i += 1;
                    }
                    all
                })
                .collect();
            self.groups.push(Group { root, levels, active: None });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn(world: &mut World, name: &str, parent: Option<Entity>, at: Vec3) -> Entity {
        let e = world.spawn((Name::new(name), Transform::from_translation(at)));
        if let Some(parent) = parent {
            world.insert_component(e, Parent { entity: parent }).unwrap();
        }
        e
    }

    #[test]
    fn the_level_follows_the_distance() {
        let g = LodGroupComponent { enabled: true, distances: Vec3::new(80.0, 200.0, 500.0) };
        assert_eq!(level_for(&g, 10.0, 3), Some(0));
        assert_eq!(level_for(&g, 120.0, 3), Some(1));
        assert_eq!(level_for(&g, 300.0, 3), Some(2));
        assert_eq!(level_for(&g, 300.0, 2), Some(1), "fewer levels: the last one holds");
        assert_eq!(level_for(&g, 600.0, 3), None, "past the cull nothing draws");
        assert_eq!(level_for(&LodGroupComponent::default(), 1.0e6, 3), Some(2), "z of 0 never culls");
    }

    #[test]
    fn only_the_chosen_level_draws() {
        let mut world = World::new();
        let house = spawn(&mut world, "cottage", None, Vec3::new(100.0, 0.0, 0.0));
        world.insert_component(house, LodGroupComponent::default()).unwrap();
        // Out of name order on purpose.
        let far = spawn(&mut world, "LOD1", Some(house), Vec3::ZERO);
        let near = spawn(&mut world, "LOD0", Some(house), Vec3::ZERO);
        let roof = spawn(&mut world, "Roof / slate", Some(near), Vec3::ZERO);
        let shell = spawn(&mut world, "Walls / shell", Some(far), Vec3::ZERO);
        let plain = spawn(&mut world, "well", None, Vec3::ZERO);

        let mut lods = Lods::default();
        lods.update(&world, Vec3::new(100.0, 30.0, 40.0));
        assert!(!lods.is_hidden(near) && !lods.is_hidden(roof));
        assert!(lods.is_hidden(far) && lods.is_hidden(shell));
        assert!(!lods.is_hidden(plain), "nothing outside a group is touched");

        lods.update(&world, Vec3::new(100.0, 30.0, 140.0));
        assert!(lods.is_hidden(near) && lods.is_hidden(roof));
        assert!(!lods.is_hidden(far) && !lods.is_hidden(shell));

        world.get_mut::<LodGroupComponent>(house).unwrap().enabled = false;
        lods.update(&world, Vec3::new(100.0, 30.0, 140.0));
        assert!(!lods.is_hidden(near) && !lods.is_hidden(far), "off: every level draws");
    }

    fn children_of(world: &World) -> HashMap<Entity, Vec<Entity>> {
        let mut map: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for (entity, parent) in world.iter_with::<Parent>() {
            map.entry(parent.entity).or_default().push(entity);
        }
        map
    }

    /// A yard with a bench in each level: the click's answer agrees with the draw loop's at every distance, and the
    /// bench of one level names the bench of the other as its twin.
    #[test]
    fn a_click_asks_what_the_draw_loop_shows_and_a_part_knows_its_twin() {
        let mut world = World::new();
        let yard = spawn(&mut world, "yard", None, Vec3::new(100.0, 0.0, 0.0));
        world.insert_component(yard, LodGroupComponent { enabled: true, distances: Vec3::new(25.0, 100_000.0, 90.0) }).unwrap();
        let far = spawn(&mut world, "LOD1", Some(yard), Vec3::ZERO);
        let near = spawn(&mut world, "LOD0", Some(yard), Vec3::ZERO);
        let bench = spawn(&mut world, "ks_bench", Some(near), Vec3::ZERO);
        let logs = spawn(&mut world, "ks_logs", Some(near), Vec3::ZERO);
        let far_bench = spawn(&mut world, "ks_bench", Some(far), Vec3::ZERO);
        let well = spawn(&mut world, "well", None, Vec3::ZERO);
        let children = children_of(&world);

        let mut lods = Lods::default();
        for distance in [5.0, 24.0, 26.0, 60.0, 95.0] {
            let camera = Vec3::new(100.0 + distance, 0.0, 0.0);
            lods.update(&world, camera);
            for entity in [near, far, bench, logs, far_bench, well, yard] {
                assert_eq!(hidden_from(&world, &children, entity, camera), lods.is_hidden(entity), "{entity:?} at {distance} m");
            }
        }

        assert_eq!(twins_of(&world, &children, &[bench]), vec![far_bench]);
        assert_eq!(twins_of(&world, &children, &[far_bench]), vec![bench]);
        assert!(twins_of(&world, &children, &[bench, far_bench]).is_empty(), "both are selected already");
        assert!(twins_of(&world, &children, &[logs]).is_empty(), "the far level has no logs");
        assert!(twins_of(&world, &children, &[near, yard, well]).is_empty(), "a level, a root and a well have none");
        // Two far benches of one name: which is the twin cannot be said.
        spawn(&mut world, "ks_bench", Some(far), Vec3::ZERO);
        assert!(twins_of(&world, &children_of(&world), &[bench]).is_empty());
    }
}
