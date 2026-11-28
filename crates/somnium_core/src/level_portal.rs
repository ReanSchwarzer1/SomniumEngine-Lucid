//! A level portal (Sunscar, 19 Nov 2026; the game's `design/engine/TERRAIN_HOLES_AND_PORTALS.md`): a box that names a
//! level. While the game plays, the eye entering the box is logged once per entry, and again only after it has left.
//! It loads nothing yet: caves place one in each mouth's throat, and a level loader will read them later.

use std::collections::HashSet;

use somnium_ecs::{Entity, World};

/// A box (the entity's Transform is its centre and turn; scale applies) that names a level.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelPortalComponent {
    /// The level's id, a project-relative name (no file has to exist yet).
    pub level: String,
    /// The box's half size, metres, in the entity's own axes.
    pub half_extents: [f32; 3],
}

impl Default for LevelPortalComponent {
    fn default() -> Self {
        Self { level: String::new(), half_extents: [1.0, 1.5, 0.5] }
    }
}

impl somnium_ecs::Component for LevelPortalComponent {}

/// Whether `eye` (world space) is inside the box of half size `half_extents` under the world matrix `model`.
#[must_use]
pub fn inside(model: glam::Mat4, half_extents: [f32; 3], eye: glam::Vec3) -> bool {
    let local = model.inverse().transform_point3(eye);
    local.abs().cmple(glam::Vec3::from(half_extents)).all()
}

/// The portals the eye stands in, so an entry is logged once.
#[derive(Default)]
pub struct Portals {
    inside: HashSet<Entity>,
}

impl Portals {
    /// One frame of play with the eye at `eye`: the portals just entered, with their level, each logged.
    pub fn update(&mut self, world: &World, eye: glam::Vec3) -> Vec<(Entity, String)> {
        let mut now = HashSet::new();
        let mut entered = Vec::new();
        for entity in world.entities_with::<LevelPortalComponent>() {
            let (Some(portal), Some(at)) = (world.get::<LevelPortalComponent>(entity), world.get::<crate::WorldTransform>(entity))
            else {
                continue;
            };
            if inside(at.0, portal.half_extents, eye) {
                now.insert(entity);
                if !self.inside.contains(&entity) {
                    let name = world.get::<crate::Name>(entity).map_or("unnamed", |n| n.as_str());
                    tracing::info!(level = %portal.level, portal = name, "level portal: entered");
                    entered.push((entity, portal.level.clone()));
                }
            }
        }
        self.inside = now;
        entered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_portal_is_entered_once_until_the_eye_leaves_it() {
        let mut world = World::new();
        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            glam::Vec3::new(10.0, 0.0, 5.0),
        );
        let portal = world.spawn((
            crate::Name::new("rose_adit_mouth_portal"),
            crate::WorldTransform(model),
            LevelPortalComponent { level: "caves/rose_adit_01".into(), half_extents: [1.0, 1.5, 0.5] },
        ));
        let mut portals = Portals::default();
        // Turned a quarter and scaled 2: its local z (half 0.5, so 1 m) lies along world x.
        assert!(portals.update(&world, glam::Vec3::new(20.0, 0.0, 5.0)).is_empty(), "outside");
        assert_eq!(portals.update(&world, glam::Vec3::new(10.8, 1.0, 5.0)), vec![(portal, "caves/rose_adit_01".to_owned())]);
        assert!(portals.update(&world, glam::Vec3::new(10.8, 1.0, 5.5)).is_empty(), "still inside: logged once");
        assert!(portals.update(&world, glam::Vec3::new(11.2, 1.0, 5.0)).is_empty(), "past its 1 m: out");
        assert_eq!(portals.update(&world, glam::Vec3::new(10.0, 0.0, 6.5)).len(), 1, "in again from its side");
    }
}
