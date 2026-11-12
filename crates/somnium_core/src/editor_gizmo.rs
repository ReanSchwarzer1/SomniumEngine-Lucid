//! Hierarchy-aware transform-gizmo transactions.
//!
//! The window host decides when a gesture begins and ends. This module owns
//! the difficult invariant in between: the gesture is solved in world space,
//! while every selected entity must be written in its own parent's local
//! space. Capturing is atomic, so a degenerate parent cannot leave half a
//! multi-selection moved.

use crate::{Parent, Transform, WorldTransform};
use somnium_ecs::{Entity, World};

#[derive(Clone)]
pub(crate) struct GizmoFollower {
    pub entity_index: u32,
    pub start_transform: Transform,
    pub start_world_translation: glam::Vec3,
    pub parent_world_inverse: glam::Mat4,
    /// The parent's world rotation, so a world-space turn can be written into the local rotation.
    pub parent_rotation: glam::Quat,
    /// It started with the primary's own local transform under a parent standing where the primary's stands (a part's
    /// twin in another level of detail): it is the same body, so it takes the primary's transform as it is.
    pub rides_primary: bool,
}

pub(crate) fn capture_followers(
    world: &World,
    selection: &[Entity],
    primary_index: u32,
) -> Option<Vec<GizmoFollower>> {
    let primary = selection
        .iter()
        .find(|entity| entity.index() == primary_index)
        .and_then(|entity| Some((*world.get::<Transform>(*entity)?, parent_world_matrix(world, *entity))));
    selection
        .iter()
        .filter(|entity| entity.index() != primary_index)
        .map(|entity| {
            let start_transform = *world.get::<Transform>(*entity)?;
            let parent_world = parent_world_matrix(world, *entity);
            let rides_primary = primary.is_some_and(|(start, parent)| {
                start.translation == start_transform.translation
                    && start.rotation == start_transform.rotation
                    && start.scale == start_transform.scale
                    && parent.abs_diff_eq(parent_world, 1e-3)
            });
            let start_world_translation = entity_world_matrix(world, *entity)
                .map(|model| model.to_scale_rotation_translation().2)
                .unwrap_or_else(|| parent_world.transform_point3(start_transform.translation));
            Some(GizmoFollower {
                entity_index: entity.index(),
                start_transform,
                start_world_translation,
                parent_world_inverse: invert_affine(parent_world)?,
                parent_rotation: parent_world.to_scale_rotation_translation().1,
                rides_primary,
            })
        })
        .collect()
}

pub(crate) fn apply_followers(
    world: &mut World,
    followers: &[GizmoFollower],
    world_offset: glam::Vec3,
    spin: glam::Quat,
    growth: glam::Vec3,
    pivot: Option<glam::Vec3>,
    primary: Transform,
) {
    for follower in followers {
        let Some(entity) = world.find_entity_by_index(follower.entity_index) else {
            continue;
        };
        let Some(transform) = world.get_mut::<Transform>(entity) else {
            continue;
        };
        if follower.rides_primary {
            *transform = primary;
            continue;
        }
        let mut moved = follower.start_transform;
        moved.translation = follower
            .parent_world_inverse
            .transform_point3(follower.start_world_translation + world_offset);
        moved.rotation = spin_in_parent(follower.parent_rotation, spin) * follower.start_transform.rotation;
        moved.scale = follower.start_transform.scale * growth;
        if let Some(pivot) = pivot {
            let arm = follower.start_world_translation - pivot;
            let world_translation = pivot + spin * (arm * growth) + world_offset;
            moved.translation = follower
                .parent_world_inverse
                .transform_point3(world_translation);
        }
        *transform = moved;
    }
}

/// The model transform local authoring is relative to. A dangling or missing
/// parent is a root, and a parent whose propagation has not run yet falls back
/// to its authored transform.
pub(crate) fn parent_world_matrix(world: &World, entity: Entity) -> glam::Mat4 {
    let Some(parent) = world.get::<Parent>(entity).map(|parent| parent.entity) else {
        return glam::Mat4::IDENTITY;
    };
    if !world.is_alive(parent) {
        return glam::Mat4::IDENTITY;
    }
    entity_world_matrix(world, parent).unwrap_or(glam::Mat4::IDENTITY)
}

pub(crate) fn entity_world_matrix(world: &World, entity: Entity) -> Option<glam::Mat4> {
    world
        .get::<WorldTransform>(entity)
        .map(|world| world.0)
        .or_else(|| world.get::<Transform>(entity).map(Transform::to_matrix))
}

pub(crate) fn world_to_local_translation(
    parent_world_inverse: glam::Mat4,
    world: glam::Vec3,
) -> glam::Vec3 {
    parent_world_inverse.transform_point3(world)
}

pub(crate) fn invert_affine(model: glam::Mat4) -> Option<glam::Mat4> {
    let determinant = model.determinant();
    (determinant.is_finite() && determinant.abs() >= 1e-8).then(|| model.inverse())
}

/// A world-space turn written in the frame of a local rotation: its parent's. A part of a turned object turns about
/// the axis the gizmo shows, not about that axis read in the parent's frame.
pub(crate) fn spin_in_parent(parent_rotation: glam::Quat, world_spin: glam::Quat) -> glam::Quat {
    parent_rotation.inverse() * world_spin * parent_rotation
}

/// Where the handle of a part of a merged model sits: the middle of its mesh, when the part's own origin lies outside
/// the mesh's box. A bench carved out of a yard model keeps the yard's origin, metres away at the yard's corner, and a
/// turn or a scale about that origin swings the bench across the yard. `None` when the origin is in the box, as it is
/// for every object that is its own model: the origin is the handle, as it always was. `min` and `max` are the mesh's
/// box in its own frame, `model` its world matrix.
pub(crate) fn part_pivot(model: glam::Mat4, min: glam::Vec3, max: glam::Vec3) -> Option<glam::Vec3> {
    if !(min.is_finite() && max.is_finite()) {
        return None;
    }
    // An origin on a face of the box (a prop standing on its origin, a door on its hinge line) counts as inside.
    let slack = (max - min) * 0.05 + glam::Vec3::splat(0.02);
    let origin_inside = (min - slack).cmple(glam::Vec3::ZERO).all() && (max + slack).cmpge(glam::Vec3::ZERO).all();
    (!origin_inside).then(|| model.transform_point3((min + max) * 0.5))
}

/// A local translation after the body is turned by `turn` (a world-space turn) about the world point `handle`: the
/// origin swings round the handle.
pub(crate) fn turned_about(
    parent_world: glam::Mat4,
    parent_world_inverse: glam::Mat4,
    start: glam::Vec3,
    handle: glam::Vec3,
    turn: glam::Quat,
) -> glam::Vec3 {
    let origin = parent_world.transform_point3(start);
    parent_world_inverse.transform_point3(handle + turn * (origin - handle))
}

/// A local translation after the local scale goes from `start.scale` to `scale` with the world point `handle` held
/// still.
pub(crate) fn scaled_about(
    parent_world_inverse: glam::Mat4,
    start: &Transform,
    handle: glam::Vec3,
    scale: glam::Vec3,
) -> glam::Vec3 {
    // The handle in the entity's own turned frame, at its starting scale.
    let reach = start.rotation.inverse() * (parent_world_inverse.transform_point3(handle) - start.translation);
    let ratio = |to: f32, from: f32| if from.abs() > 1e-6 { to / from } else { 1.0 };
    let kept = glam::Vec3::new(
        ratio(scale.x, start.scale.x),
        ratio(scale.y, start.scale.y),
        ratio(scale.z, start.scale.z),
    );
    start.translation + start.rotation * (reach * (glam::Vec3::ONE - kept))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bench that is part 6 of a yard model: its origin is the yard's corner, over five metres away. Its handle is
    /// the middle of its mesh. An object standing on its own origin keeps the origin.
    #[test]
    fn a_part_far_from_its_origin_is_handled_at_its_middle() {
        let model = glam::Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(0.5),
            glam::Vec3::new(100.0, 3.0, -40.0),
        );
        let (min, max) = (glam::Vec3::new(5.0, 0.0, 1.0), glam::Vec3::new(6.6, 0.9, 1.6));
        let handle = part_pivot(model, min, max).expect("the origin is outside the bench");
        assert!(handle.abs_diff_eq(model.transform_point3(glam::Vec3::new(5.8, 0.45, 1.3)), 1e-4));
        // A barrel standing on its own origin: the origin is on the bottom face of its box.
        assert_eq!(part_pivot(model, glam::Vec3::new(-0.3, 0.0, -0.3), glam::Vec3::new(0.3, 0.9, 0.3)), None);
        assert_eq!(part_pivot(model, glam::Vec3::splat(f32::NAN), max), None);
    }

    /// Turned half round about its handle the bench stays where it stood and faces the other way; grown about its
    /// handle it stays too. Turned about its origin (the old rule) it swung metres away.
    #[test]
    fn a_part_turns_and_grows_about_its_handle_and_not_about_its_origin() {
        let parent = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(2.0),
            glam::Quat::from_rotation_z(0.2) * glam::Quat::from_rotation_y(1.1),
            glam::Vec3::new(-15460.0, 310.0, 284.0),
        );
        let parent_rotation = parent.to_scale_rotation_translation().1;
        let start = Transform {
            translation: glam::Vec3::new(1.0, 0.0, 2.0),
            rotation: glam::Quat::from_rotation_y(0.3),
            scale: glam::Vec3::new(1.5, 1.0, 0.5),
        };
        // The middle of the bench's mesh and one of its corners, in the part's own frame.
        let (middle, corner) = (glam::Vec3::new(5.8, 0.45, 1.3), glam::Vec3::new(6.6, 0.45, 1.3));
        let world_of = |t: &Transform, p: glam::Vec3| parent.transform_point3(t.to_matrix().transform_point3(p));
        let handle = world_of(&start, middle);

        let turn = glam::Quat::from_rotation_y(std::f32::consts::PI);
        let mut turned = start;
        turned.rotation = spin_in_parent(parent_rotation, turn) * start.rotation;
        turned.translation = turned_about(parent, parent.inverse(), start.translation, handle, turn);
        assert!(world_of(&turned, middle).abs_diff_eq(handle, 2e-2), "the handle moved");
        let (before, after) = (world_of(&start, corner) - handle, world_of(&turned, corner) - handle);
        assert!(after.abs_diff_eq(turn * before, 2e-2), "the corner is not on the far side of the handle");
        let mut old = start;
        old.rotation = turned.rotation;
        assert!(world_of(&old, middle).distance(handle) > 5.0, "about its origin the bench leaves its place");

        let mut grown = start;
        grown.scale = start.scale * glam::Vec3::new(2.0, 1.0, 3.0);
        grown.translation = scaled_about(parent.inverse(), &start, handle, grown.scale);
        assert!(world_of(&grown, middle).abs_diff_eq(handle, 2e-2), "the handle moved as the bench grew");
    }

    /// A part of a turned object turns about the world axis the gizmo drew: its world rotation after the drag is the
    /// drag's turn times its world rotation before.
    #[test]
    fn a_part_of_a_turned_parent_turns_about_the_world_axis() {
        let parent = glam::Quat::from_rotation_z(0.7) * glam::Quat::from_rotation_x(0.4);
        let local = glam::Quat::from_rotation_y(0.3);
        let turn = glam::Quat::from_axis_angle(glam::Vec3::Y, 0.5);
        let after = parent * spin_in_parent(parent, turn) * local;
        assert!(after.abs_diff_eq(turn * parent * local, 1e-5));
        // The old rule, `turn * local`, turned it about the parent's Y instead.
        assert!(!(parent * turn * local).abs_diff_eq(turn * parent * local, 1e-3));
    }

    #[test]
    fn one_singular_parent_refuses_the_entire_capture() {
        let mut world = World::new();
        let good = world.spawn((Transform::default(),));
        let collapsed_parent = world.spawn((Transform {
            scale: glam::Vec3::ZERO,
            ..Transform::default()
        },));
        let bad = world.spawn((
            Transform::default(),
            Parent {
                entity: collapsed_parent,
            },
        ));

        assert!(capture_followers(&world, &[good, bad], u32::MAX).is_none());
    }

    /// A part's twin in the other level of detail starts on the part, in a frame standing where the part's stands: it
    /// ends each frame of a drag exactly on the part. Another selected thing follows by the gesture as before.
    #[test]
    fn a_twin_rides_the_primary_and_another_follower_does_not() {
        let mut world = World::new();
        let at = glam::Mat4::from_translation(glam::Vec3::new(-15460.0, 310.0, 284.0));
        let near = world.spawn((Transform::default(), WorldTransform(at)));
        let far = world.spawn((Transform::default(), WorldTransform(at)));
        let part = Transform::from_translation(glam::Vec3::new(5.0, 0.0, 1.0));
        let bench = world.spawn((part, Parent { entity: near }));
        let twin = world.spawn((part, Parent { entity: far }));
        let beside = glam::Vec3::new(7.0, 0.0, 1.0);
        let crate_ = world.spawn((
            Transform::from_translation(beside),
            WorldTransform(at * glam::Mat4::from_translation(beside)),
            Parent { entity: near },
        ));
        let followers = capture_followers(&world, &[twin, crate_, bench], bench.index()).unwrap();
        assert_eq!(followers.iter().map(|f| f.rides_primary).collect::<Vec<_>>(), vec![true, false]);

        let moved = Transform {
            translation: glam::Vec3::new(5.5, 0.0, 2.0),
            rotation: glam::Quat::from_rotation_y(1.0),
            scale: glam::Vec3::new(1.0, 2.0, 1.0),
        };
        apply_followers(&mut world, &followers, glam::Vec3::X, glam::Quat::IDENTITY, glam::Vec3::ONE, None, moved);
        let landed = *world.get::<Transform>(twin).unwrap();
        assert!(landed.translation == moved.translation && landed.rotation == moved.rotation && landed.scale == moved.scale);
        assert!(world.get::<Transform>(crate_).unwrap().translation.abs_diff_eq(glam::Vec3::new(8.0, 0.0, 1.0), 1e-2));
    }
}
