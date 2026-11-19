//! Things shown by the hour (sunscar-city, 19 Nov 2026): shop fronts open by day, shutters closed at night, washing
//! taken in by dusk.
//!
//! A `somnium.ShownByHour` on an entity shows it, and everything under it, while the scene clock's hour is in
//! `[from, to)`; a window whose `from` is past its `to` runs over midnight. It applies in play only: the editor shows
//! everything at every hour, so nothing vanishes while he edits. What the clock hides is skipped at submission,
//! shadows and the traced scene with it, through a runtime set as [`crate::lod_group::Lods`] keeps its levels: nothing
//! authored or saved changes. The engine staggers nothing; a street that opens over three quarters of an hour is
//! written as houses with their own hours.

use std::collections::HashMap;

use somnium_ecs::component_schema;
use somnium_ecs::reflect::TypeRegistry;
use somnium_ecs::{Component, Entity, World};

use crate::Parent;
use crate::time_of_day::TimeOfDayComponent;

/// An entity shown only between two hours of the scene's clock.
#[derive(Debug, Clone, PartialEq)]
pub struct ShownByHourComponent {
    /// The hour it appears, 0..24.
    pub from: f32,
    /// The hour it goes. Below `from`, the window runs past midnight.
    pub to: f32,
}

impl Component for ShownByHourComponent {}

impl Default for ShownByHourComponent {
    fn default() -> Self {
        Self { from: 7.0, to: 19.0 }
    }
}

pub(crate) fn register(registry: &mut TypeRegistry) {
    registry.register(component_schema! {
        ShownByHourComponent as "somnium.ShownByHour", display "Shown By Hour", version 1,
        fields {
            from { min: 0.0, max: 24.0, step: 0.25, precision: 2, unit: "h", group: "Hours", doc: "The hour of the scene clock it appears at, in play (the editor shows it at every hour)." },
            to { min: 0.0, max: 24.0, step: 0.25, precision: 2, unit: "h", group: "Hours", doc: "The hour it goes. Below From, the window runs past midnight: shutters closed at night are From 19.5, To 7.25." },
        }
    });
}

/// Whether `hour` falls in `[from, to)`, past midnight when `from` is past `to`. A window of no length shows at every
/// hour, so a mistyped one never hides a thing for good.
#[must_use]
pub fn shown_at(from: f32, to: f32, hour: f32) -> bool {
    let (from, to, hour) = (from.rem_euclid(24.0), to.rem_euclid(24.0), hour.rem_euclid(24.0));
    if from == to {
        true
    } else if from < to {
        (from..to).contains(&hour)
    } else {
        hour >= from || hour < to
    }
}

/// The scene clock's hour, from its first enabled `TimeOfDay`.
fn clock_hour(world: &World) -> Option<f32> {
    world.iter_with::<TimeOfDayComponent>().find(|(_, clock)| clock.enabled).map(|(_, clock)| clock.hour)
}

/// What the clock hides this frame.
#[derive(Default)]
pub struct Hours {
    /// Each `ShownByHour` entity with every entity under it, its own first.
    groups: Vec<(Entity, Vec<Entity>)>,
    built_for: usize,
    /// The clock's minute the set was made for (`None`: not in play, or not made yet).
    minute: Option<i64>,
    /// By entity index, the generation + 1 of a hidden entity (0: shown), as `Lods` keeps its levels.
    hidden: Vec<u32>,
}

impl Hours {
    /// Make the set for this frame. `playing` off shows everything. The set is made again only when the clock passes
    /// a minute or the world's entity count changes.
    // ponytail: an edited window or a reparent that keeps the count is picked up at the next minute or count change.
    pub fn update(&mut self, world: &World, playing: bool) {
        if self.built_for != world.entity_count() {
            self.rebuild(world);
        }
        if self.groups.is_empty() {
            return;
        }
        let hour = if playing { clock_hour(world) } else { None };
        let minute = hour.map(|h| (f64::from(h) * 60.0).floor() as i64);
        if minute == self.minute {
            return;
        }
        self.minute = minute;
        self.hidden.fill(0);
        let Some(hour) = hour else { return };
        let hidden = &mut self.hidden;
        for (root, all) in &self.groups {
            let Some(window) = world.get::<ShownByHourComponent>(*root) else { continue };
            if shown_at(window.from, window.to, hour) {
                continue;
            }
            for entity in all {
                let slot = entity.index() as usize;
                if slot >= hidden.len() {
                    hidden.resize(slot + 1, 0);
                }
                hidden[slot] = entity.generation().wrapping_add(1);
            }
        }
    }

    /// Whether the clock hides this entity this frame.
    #[must_use]
    pub fn is_hidden(&self, entity: Entity) -> bool {
        self.hidden.get(entity.index() as usize).is_some_and(|&g| g != 0 && g == entity.generation().wrapping_add(1))
    }

    // ponytail: a scan of every parented entity whenever the count changes, as Lods makes; share one children map
    // between the two if streaming's recounts ever show in a frame.
    fn rebuild(&mut self, world: &World) {
        self.built_for = world.entity_count();
        self.minute = None;
        self.hidden.fill(0);
        self.groups.clear();
        let roots: Vec<Entity> = world.entities_with::<ShownByHourComponent>().collect();
        if roots.is_empty() {
            return;
        }
        let mut children: HashMap<Entity, Vec<Entity>> = HashMap::new();
        for (entity, parent) in world.iter_with::<Parent>() {
            if parent.entity != Entity::DANGLING {
                children.entry(parent.entity).or_default().push(entity);
            }
        }
        for root in roots {
            let mut all = vec![root];
            let mut i = 0;
            while i < all.len() {
                if let Some(kids) = children.get(&all[i]) {
                    all.extend(kids.iter().copied());
                }
                i += 1;
            }
            self.groups.push((root, all));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Name, Transform};

    fn spawn(world: &mut World, name: &str) -> Entity {
        world.spawn((Name::new(name), Transform::from_translation(glam::Vec3::ZERO)))
    }

    #[test]
    fn a_window_holds_its_hours_and_wraps_past_midnight() {
        assert!(shown_at(7.25, 19.5, 12.0) && !shown_at(7.25, 19.5, 19.5) && !shown_at(7.25, 19.5, 3.0));
        assert!(shown_at(19.5, 7.25, 23.0) && shown_at(19.5, 7.25, 2.0) && !shown_at(19.5, 7.25, 12.0));
        assert!(shown_at(7.0, 24.0, 23.9) && !shown_at(7.0, 24.0, 6.9), "24 is midnight");
        assert!(shown_at(9.0, 9.0, 3.0), "a window of no length never hides");
    }

    /// A shop front with its awning under it and the shutter beside it: by day the front shows and the shutter
    /// hides; by night the other way; out of play, both show.
    #[test]
    fn the_clock_shows_the_front_by_day_and_the_shutter_by_night() {
        let mut world = World::new();
        let clock = spawn(&mut world, "Environment");
        world
            .insert_component(clock, TimeOfDayComponent { enabled: true, hour: 12.0, ..TimeOfDayComponent::default() })
            .unwrap();
        let front = spawn(&mut world, "shop_front");
        world.insert_component(front, ShownByHourComponent { from: 7.25, to: 19.5 }).unwrap();
        let awning = spawn(&mut world, "awning");
        world.insert_component(awning, Parent { entity: front }).unwrap();
        let shutter = spawn(&mut world, "shutter");
        world.insert_component(shutter, ShownByHourComponent { from: 19.5, to: 7.25 }).unwrap();
        let wall = spawn(&mut world, "wall");

        let mut hours = Hours::default();
        hours.update(&world, true);
        assert!(!hours.is_hidden(front) && !hours.is_hidden(awning) && hours.is_hidden(shutter), "noon");
        assert!(!hours.is_hidden(wall), "nothing without a window is touched");

        world.get_mut::<TimeOfDayComponent>(clock).unwrap().hour = 22.0;
        hours.update(&world, true);
        assert!(hours.is_hidden(front) && hours.is_hidden(awning) && !hours.is_hidden(shutter), "night");

        hours.update(&world, false);
        assert!(!hours.is_hidden(front) && !hours.is_hidden(awning) && !hours.is_hidden(shutter), "the editor");
    }
}
