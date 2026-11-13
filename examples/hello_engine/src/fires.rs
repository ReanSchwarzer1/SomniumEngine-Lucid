//! Fire for every made flame (Sunscar, 13 Nov 2026; `somnium_core::fire`). Where the engine no longer draws a
//! `kp_flame` mesh, a flame burns in the particle pass: flickering tongues, and over an open fire (a brazier, a
//! torch, a tar basket) embers and smoke that rise and go with the wind. Each burns while its lamp does: the light
//! it stands in (`somnium.Lamp`, dusk to dawn), or, for a flame with no light of its own, a small one made for it
//! and switched the same way, so the light wavers with the fire.
//!
//! What is made here is never saved: it has no persistent id, so no scene and no edits layer holds it, and it goes
//! when the flame's entity goes (its cell leaves, the scene closes).

use std::collections::HashSet;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use somnium_core::fire::{Flame, FlameSource};
use somnium_core::light_flicker::LampComponent;
use somnium_core::weather::WeatherComponent;
use somnium_core::{LightComponent, Name, ParticleEmitter, Parent, Transform, WorldTransform};
use somnium_ecs::curve::{Gradient, GradientStop};
use somnium_ecs::{Entity, World};

/// A light this near a flame is its own (a lantern's stands 3 cm over its wick), metres.
const OWN_LIGHT: f32 = 1.2;
/// Two flames this near are one (the same flame in two levels of detail), metres.
const SAME_SPOT: f32 = 0.05;

struct Burning {
    carrier: Entity,
    spot: (i32, i32, i32),
    /// Everything made for it: emitters, and a light when it had none.
    made: Vec<Entity>,
    /// The emitters that stop when the lamp is out.
    emitters: Vec<Entity>,
    smoke: Option<Entity>,
    /// The light it burns with.
    light: Option<Entity>,
    lit: Option<bool>,
    /// A fire's (brazier, torch) rather than a lamp's: its light flickers hard.
    open: bool,
}

#[derive(Default)]
pub struct Fires {
    burning: Vec<Burning>,
    carriers: HashSet<Entity>,
    spots: HashSet<(i32, i32, i32)>,
}

fn spot_of(at: Vec3) -> (i32, i32, i32) {
    let q = at / SAME_SPOT;
    (q.x.round() as i32, q.y.round() as i32, q.z.round() as i32)
}

/// A fraction in `0..1` from where a flame stands: no two flames flicker in step.
fn salt(at: Vec3) -> f32 {
    ((at.x * 12.9898 + at.y * 78.233 + at.z * 37.719).sin() * 43_758.547).fract().abs()
}

fn ramp(stops: &[(f32, [f32; 4])]) -> Gradient {
    Gradient::from_stops(stops.iter().map(|&(t, c)| GradientStop::new(t, c)).collect())
}

impl Fires {
    /// Once a frame, after the transforms: light what arrived, put out what left, and burn each while its lamp does.
    pub fn update(&mut self, world: &mut World, night: f32, time: f32) {
        // What left: its cell unloaded or the scene closed. What was made for it goes too.
        let mut gone = Vec::new();
        self.burning.retain(|fire| {
            let alive = world.is_alive(fire.carrier);
            if !alive {
                gone.push((fire.carrier, fire.spot, fire.made.clone()));
            }
            alive
        });
        for (carrier, spot, made) in gone {
            for entity in made {
                if world.is_alive(entity) {
                    world.despawn(entity);
                }
            }
            self.spots.remove(&spot);
            self.carriers.remove(&carrier);
        }
        self.carriers.retain(|&carrier| world.is_alive(carrier));

        // What arrived.
        let arrived: Vec<(Entity, Arc<[Flame]>)> = world
            .entities_with::<FlameSource>()
            .filter(|entity| !self.carriers.contains(entity))
            .filter_map(|entity| Some((entity, world.get::<FlameSource>(entity)?.flames.clone())))
            .collect();
        if !arrived.is_empty() {
            let lights: Vec<(Entity, Vec3)> = world
                .entities_with::<LightComponent>()
                .filter_map(|entity| Some((entity, world.get::<WorldTransform>(entity)?.0.w_axis.truncate())))
                .collect();
            // Smoke a place made already (the Kept's brazier smoke): an open fire there adds none of its own.
            let smoke: Vec<Vec3> = world
                .entities_with::<ParticleEmitter>()
                .filter(|&entity| world.get::<Name>(entity).is_none_or(|name| !name.as_str().starts_with("fire: ")))
                .filter_map(|entity| Some(world.get::<WorldTransform>(entity)?.0.w_axis.truncate()))
                .collect();
            for (carrier, flames) in arrived {
                self.carriers.insert(carrier);
                let Some(model) = world.get::<WorldTransform>(carrier).map(|w| w.0) else { continue };
                for flame in flames.iter() {
                    let at = model.transform_point3(flame.base);
                    let spot = spot_of(at);
                    if !self.spots.insert(spot) {
                        continue;
                    }
                    let own = lights
                        .iter()
                        .map(|&(light, p)| (light, p.distance(at)))
                        .filter(|&(_, d)| d <= OWN_LIGHT)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(light, _)| light);
                    // ponytail: a place's own smoke (the Kept's rose brazier) did not show in any still, so every
                    // open fire makes its own column; take `smoke` into account again if the two ever show doubled.
                    let smoked = smoke.iter().any(|p| p.distance(at) <= 1.0) && false;
                    let fire = burn(world, carrier, model, flame, spot, own, smoked);
                    self.burning.push(fire);
                }
            }
        }

        // Each burns while its lamp does; smoke goes with the wind, a third as fast.
        let wind = world
            .entities_with::<WeatherComponent>()
            .next()
            .and_then(|entity| world.get::<WeatherComponent>(entity))
            .map_or([0.0; 2], WeatherComponent::wind_vector);
        for fire in &mut self.burning {
            let lit = fire.light.is_none_or(|light| {
                match (world.get::<LampComponent>(light), world.get::<WorldTransform>(light)) {
                    (Some(lamp), Some(at)) => lamp.factor(night, at.0.w_axis.truncate(), time) > 0.02,
                    _ => true,
                }
            });
            // A fire's own light (the place's) flickers as hard as the fire does: the ground and walls with it.
            if fire.open
                && let Some(light) = fire.light
                && let Some(lamp) = world.get_mut::<LampComponent>(light)
                && lamp.flicker < 0.45
            {
                lamp.flicker = 0.45;
            }
            if fire.lit != Some(lit) {
                for &emitter in &fire.emitters {
                    if let Some(emitter) = world.get_mut::<ParticleEmitter>(emitter) {
                        emitter.enabled = lit;
                    }
                }
                fire.lit = Some(lit);
            }
            if let Some(smoke) = fire.smoke
                && let Some(emitter) = world.get_mut::<ParticleEmitter>(smoke)
            {
                let drift = glam::Vec2::new(wind[0], wind[1]) / 3.0;
                let drift = if drift.length() < 0.25 { drift.try_normalize().unwrap_or(glam::Vec2::X) * 0.25 } else { drift };
                emitter.velocity_bias = [drift.x, 0.0, drift.y];
            }
        }
    }
}

/// Make one flame's fire: its emitters (and a light if it has none) as children of the flame's entity, so they go
/// where it is moved. `model` is that entity's world matrix, `own` the light it stands in.
fn burn(
    world: &mut World,
    carrier: Entity,
    model: Mat4,
    flame: &Flame,
    spot: (i32, i32, i32),
    own: Option<Entity>,
    smoked: bool,
) -> Burning {
    // The model's scale up its height: sizes are in world metres, offsets in the flame's own space.
    let scale = model.y_axis.truncate().length().max(1e-3);
    let (height, radius) = (flame.height * scale, flame.radius.max(0.004) * scale);
    let open = flame.open();
    let at = model.transform_point3(flame.base);
    let seed = salt(at);
    let mut made = Vec::new();
    let mut emitters = Vec::new();
    let mut child = |world: &mut World, name: &str, rise: f32, emitter: Option<ParticleEmitter>, light: Option<(LightComponent, LampComponent)>| {
        let local = Transform::from_translation(flame.base + Vec3::Y * (rise / scale));
        let world_matrix = WorldTransform(model * Mat4::from_translation(local.translation));
        let entity = world.spawn((Name::new(name), local, world_matrix, Parent { entity: carrier }));
        if let Some(emitter) = emitter {
            let _ = world.insert_component(entity, emitter);
        }
        if let Some((light, lamp)) = light {
            let _ = world.insert_component(entity, light);
            let _ = world.insert_component(entity, lamp);
        }
        made.push(entity);
        entity
    };
    let smoke = if open {
        // A wood or tar fire is one body over its whole bed (the developer, 13 Nov 11:58, of separate cones on a pan:
        // "looks dogshit"): many small licks born all over the bed, rising, merging and splitting, torn into wisps at
        // the top (the shader), white-yellow low down, orange, then deep red as they die; a bed of glowing coals
        // under them; sparks; smoke soot-dark close over the flame and lit from below at night.
        let lick = (radius * 0.75).clamp(0.07, 0.22);
        let fire = ParticleEmitter {
            additive: true,
            world_up: true,
            flame: true,
            max_particles: 110,
            spawn_rate: 170.0,
            lifetime: 0.55,
            initial_speed: 0.25,
            spread_angle: 0.3,
            gravity: -1.6,
            size_start: lick,
            size_end: lick * 0.25,
            aspect: 1.7,
            flutter: 0.2,
            flutter_hz: 6.5,
            tip_tint: Vec3::new(1.0, 0.3, 0.06),
            spawn_extents: [radius * 0.7, 0.0, radius * 0.7],
            color_over_life: ramp(&[
                (0.0, [2.2, 1.9, 1.2, 0.0]),
                (0.08, [2.2, 1.8, 1.0, 0.9]),
                (0.35, [2.0, 1.1, 0.35, 0.8]),
                (0.7, [1.3, 0.35, 0.06, 0.55]),
                (1.0, [0.5, 0.08, 0.01, 0.0]),
            ]),
            spawn_accum: seed,
            ..Default::default()
        };
        emitters.push(child(world, "fire: body", lick * 1.7 * 0.4, Some(fire), None));
        let coals = ParticleEmitter {
            additive: true,
            max_particles: 16,
            spawn_rate: 12.0,
            lifetime: 1.3,
            initial_speed: 0.01,
            spread_angle: 0.0,
            gravity: 0.0,
            size_start: (radius * 0.8).max(0.06),
            size_end: (radius * 0.6).max(0.05),
            spawn_extents: [radius * 0.55, 0.0, radius * 0.55],
            color_over_life: ramp(&[(0.0, [1.4, 0.3, 0.04, 0.0]), (0.3, [1.6, 0.36, 0.05, 0.85]), (1.0, [0.7, 0.1, 0.01, 0.0])]),
            spawn_accum: seed,
            ..Default::default()
        };
        emitters.push(child(world, "fire: coals", 0.02, Some(coals), None));
        let glow = ParticleEmitter {
            additive: true,
            max_particles: 14,
            spawn_rate: 7.0,
            lifetime: 1.6,
            initial_speed: 0.45,
            spread_angle: 0.25,
            gravity: -0.1,
            size_start: (radius * 2.2).max(0.25),
            size_end: (radius * 4.0).max(0.7),
            spawn_extents: [radius * 0.3, 0.0, radius * 0.3],
            color_over_life: ramp(&[
                (0.0, [1.2, 0.5, 0.13, 0.0]),
                (0.15, [1.2, 0.5, 0.13, 0.5]),
                (0.55, [0.45, 0.17, 0.05, 0.25]),
                (1.0, [0.05, 0.02, 0.01, 0.0]),
            ]),
            spawn_accum: seed,
            ..Default::default()
        };
        emitters.push(child(world, "fire: glow", lick * 1.7 + 0.3, Some(glow), None));
        let sparks = ParticleEmitter {
            additive: true,
            max_particles: 10,
            spawn_rate: 2.5,
            lifetime: 1.6,
            initial_speed: 0.8,
            spread_angle: 0.5,
            gravity: -0.2,
            size_start: 0.014,
            size_end: 0.005,
            spawn_extents: [radius * 0.4, 0.0, radius * 0.4],
            color_over_life: ramp(&[(0.0, [6.0, 2.4, 0.5, 1.0]), (1.0, [3.0, 0.8, 0.1, 0.0])]),
            spawn_accum: seed,
            ..Default::default()
        };
        emitters.push(child(world, "fire: sparks", height * 0.4, Some(sparks), None));
        // A column of grey-brown smoke, lit by the sun and the sky: it rises, widens, thins and leans with the wind.
        let smoke = ParticleEmitter {
            lit: true,
            max_particles: 40,
            spawn_rate: 5.0,
            lifetime: 7.0,
            initial_speed: 0.45,
            spread_angle: 0.15,
            gravity: -0.03,
            size_start: (radius * 2.0).max(0.25),
            size_end: 1.8,
            rotation_spread: 3.1,
            color_over_life: ramp(&[
                (0.0, [0.03, 0.028, 0.025, 0.0]),
                (0.1, [0.035, 0.031, 0.028, 0.55]),
                (0.45, [0.1, 0.095, 0.085, 0.35]),
                (1.0, [0.16, 0.15, 0.14, 0.0]),
            ]),
            spawn_accum: seed,
            ..Default::default()
        };
        if smoked {
            None
        } else {
            let smoke = child(world, "fire: smoke", lick * 1.7 + 0.3, Some(smoke), None);
            emitters.push(smoke);
            Some(smoke)
        }
    } else {
        // A lantern's or a candle's flame: a few tongues over the wick, each a fraction of a second.
        let width = (radius * 2.4).max(0.012);
        let aspect = (height * 1.15 / width).clamp(1.6, 3.4);
        let tongues = ParticleEmitter {
            additive: true,
            world_up: true,
            flame: true,
            max_particles: 10,
            spawn_rate: 18.0,
            lifetime: 0.3,
            initial_speed: 0.03,
            spread_angle: 0.2,
            gravity: -0.35,
            size_start: width,
            size_end: width * 0.45,
            aspect,
            flutter: 0.12,
            flutter_hz: 7.5,
            tip_tint: Vec3::new(1.0, 0.42, 0.12),
            color_over_life: ramp(&[
                (0.0, [6.0, 4.0, 1.6, 0.0]),
                (0.12, [6.0, 4.0, 1.6, 1.0]),
                (0.6, [4.8, 2.6, 0.8, 0.85]),
                (1.0, [2.4, 0.8, 0.15, 0.0]),
            ]),
            spawn_accum: seed,
            ..Default::default()
        };
        emitters.push(child(world, "fire: flame", width * aspect * 0.5, Some(tongues), None));
        None
    };
    // A flame with no light of its own gets a small one, lit and put out as a lamp is and wavering with it; the
    // renderer keeps the lights that matter most where the eye is.
    let light = own.or_else(|| {
        let mut light = LightComponent::point(if open { 220.0 + 900.0 * height } else { 35.0 }, if open { 9.0 } else { 4.0 });
        light.color_temperature_k = if open { 1700.0 } else { 1900.0 };
        light.source_radius = radius.clamp(0.01, 0.2);
        let lamp = LampComponent { switched: true, flicker: if open { 0.45 } else { 0.15 } };
        Some(child(world, "fire: light", height * 0.6, None, Some((light, lamp))))
    });
    Burning { carrier, spot, made, emitters, smoke, light, lit: None, open }
}
