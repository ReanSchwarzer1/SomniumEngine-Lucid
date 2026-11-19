//! Shared particle simulation and designer-authored sprite appearance.
use crate::{Transform, WorldTransform};

/// A manual editor step advances particles once while the transport stays paused.
pub(crate) fn simulation_delta(
    clock: &crate::SimulationClock,
    stepping: bool,
    frame_delta: f32,
) -> f32 {
    if stepping {
        clock.fixed_delta_seconds
    } else if clock.state == crate::SimulationState::Paused {
        0.0
    } else {
        frame_delta
    }
}

// ─── Phase 11.5J: GPU Particle System ────────────────────────────────────────

/// Per-particle runtime state (CPU-side).
#[derive(Debug, Clone, Copy)]
pub struct ParticleState {
    /// World-space position, or offset from the emitter when `local_space` is set.
    pub position: glam::Vec3,
    /// World-space velocity (m/s).
    pub velocity: glam::Vec3,
    /// Current age in seconds (0 = just born).
    pub age: f32,
    /// Total lifetime in seconds.
    pub lifetime: f32,
    /// Stable random initial orientation in [-1, 1], independent of travel.
    pub rotation_seed: f32,
}

/// ECS component that drives a GPU particle emitter.
///
/// Add this component to an entity together with `Transform` and `WorldTransform`.
/// The engine simulates particles each frame and uploads the results to the
/// `ParticlePass` for instanced billboard rendering.
#[derive(Debug, Clone)]
pub struct ParticleEmitter {
    /// Stop emission while allowing existing particles to finish.
    pub enabled: bool,
    /// Sprite texture; NONE retains the original soft round particle.
    pub texture: somnium_asset::database::AssetId,
    /// Height divided by width of the camera-facing sprite.
    pub aspect: f32,
    /// Add emitted light instead of covering the background.
    pub additive: bool,
    /// Lit by the sun (a forward-scattering phase) and the sky's fill, the colour taken as an albedo: smoke, dust
    /// and mist read in the scene's own light at any exposure. Off, the colour is emitted HDR as before.
    pub lit: bool,
    /// Keep the sprite vertical in world space.
    pub world_up: bool,
    /// Birth positions remain relative to the moving emitter.
    pub local_space: bool,
    /// Duplicate visible sprites into the one active staged mirror.
    pub staged_reflection: bool,
    /// Additional colour multiplier at the sprite's top edge.
    pub tip_tint: glam::Vec3,
    /// Sprite-sheet columns, rows and usable frames.
    pub atlas_columns: u32,
    /// Number of sprite-sheet rows.
    pub atlas_rows: u32,
    /// Number of usable frames, bounded by columns times rows.
    pub atlas_frames: u32,
    /// Sprite-sheet frames/second; zero animates across particle lifetime.
    pub atlas_fps: f32,
    /// Random rotation range and spin in radians/second.
    pub rotation_spread: f32,
    /// Angular speed in radians per second after birth.
    pub spin: f32,
    /// Sideways sprite deformation in fractions of its width; zero disables it.
    pub flutter: f32,
    /// Deformation frequency in Hz, driven by simulation age (respects Pause).
    pub flutter_hz: f32,
    /// An untextured sprite drawn as a flame's tongue: round at its base, pointed at its tip, bent by its flutter
    /// (Sunscar's fire, 13 Nov 2026). Off, it is the soft round particle.
    pub flame: bool,
    /// One-shot burst, consumed by simulation.
    pub burst: u32,
    /// Resolved texture state, not persisted.
    pub texture_status: String,
    /// Runtime bindless slot from the common asset pipeline.
    pub texture_slot: i32,
    // ── Emitter parameters ────────────────────────────────────────────────────
    /// Maximum number of live particles at once.
    pub max_particles: u32,
    /// New particles spawned per second.
    pub spawn_rate: f32,
    /// Each particle's lifetime in seconds.
    pub lifetime: f32,
    /// Initial speed in m/s (direction is randomized within `spread_angle`).
    pub initial_speed: f32,
    /// Cone half-angle (radians) for direction randomization (0 = straight up).
    pub spread_angle: f32,
    /// Full billboard width at birth, in metres.
    pub size_start: f32,
    /// Particle size at end of life.
    pub size_end: f32,
    /// CONTROL-K: linear RGBA over the particle's life, `0` at birth and `1`
    /// at death.
    ///
    /// Replaces the `color_start`/`color_end` pair, which could express a
    /// straight line between two colours and nothing else — no flash, no
    /// fade-in-then-out, no hold. A two-stop ramp reproduces the old pair
    /// exactly, so the default is that pair.
    pub color_over_life: somnium_ecs::curve::Gradient,
    /// Downward gravity acceleration (m/s²).
    pub gravity: f32,
    /// Phase CONTROL-N: a constant velocity added to every particle at birth.
    ///
    /// The cone spawn is right for a fountain and useless for rain, which
    /// falls in one direction and is *sheared* by wind. One vector turns the
    /// same emitter into both, which is the plan's "precipitation through the
    /// existing particle emitter" rather than a second particle system.
    pub velocity_bias: [f32; 3],
    /// Phase CONTROL-N: half-extents of a box particles spawn in, around the
    /// emitter's origin.
    ///
    /// Zero is the point emitter every existing scene has. Non-zero makes the
    /// emitter a volume, which is what rain needs: a camera-anchored box
    /// overhead, so precipitation exists where the player is and nowhere else.
    pub spawn_extents: [f32; 3],

    /// A flame's card: one persistent instance from a simulated flipbook in place of spawned particles (Sunscar's
    /// fire, 19 Nov 2026; `texture` is the book's atlas). Never saved: the shell makes it.
    pub card: Option<FlameCard>,
    /// A bed of coals lying flat under an open fire, drawn by the coal shader in place of particles.
    pub coals: Option<CoalBed>,
    /// A fire's light on lit particles near it: smoke over a fire glows from below.
    pub glow: Option<Glow>,
    /// Seconds of a camera's shutter: above zero each particle is drawn as the streak it makes in that time (sparks).
    pub stretch: f32,

    // ── Runtime state (not user-facing) ──────────────────────────────────────
    /// Live particles owned by this emitter.
    pub particles: Vec<ParticleState>,
    /// Fractional carry-over for sub-frame spawning.
    pub spawn_accum: f32,
}

/// One flame drawn from its kind's simulated flipbook (`somnium_core::fire::FlameBook`): a card standing on the
/// emitter's point, its bed there, turning to face the eye and leaning toward it as the eye rises over the fire.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlameCard {
    /// Width and height, metres.
    pub size: [f32; 2],
    /// Where the bed sits on the card, down from its top (0..1): the emitter's point is there.
    pub bed_v: f32,
    /// Luminance where the book's R is 1, cd/m2.
    pub peak_cd_m2: f32,
    pub frames: u32,
    pub columns: u32,
    pub rows: u32,
    /// The first row of the 40-degree view (0: level only).
    pub down_row0: u32,
    pub fps: f32,
    /// Seconds into the book; the simulation advances it, so it pauses with the scene.
    pub clock: f32,
    /// A candle's or a lantern's book: its B channel is a blue foot, not motion.
    pub small: bool,
    /// Metres the card's top leans at its full height (the wind), x and z.
    pub lean: [f32; 2],
    /// 0..1: lit and put out with its lamp.
    pub fade: f32,
    /// The card's mean luminance over its peak (its book's intensity over peak times area): what a far card spreads.
    pub mean_rel: f32,
    /// The book's power at this frame over its mean (the shell's): a far card's flicker.
    pub power: f32,
}

/// A bed of embers: lumps with a dark crust and cracks glowing at `hot_k`, each breathing slowly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoalBed {
    /// Across, metres.
    pub diameter: f32,
    /// The hottest cracks' temperature, kelvin (a brazier's 1,150 K, a forge's under the blast 1,400 K).
    pub hot_k: f32,
    pub seed: f32,
    pub clock: f32,
    pub fade: f32,
}

/// A light that lights particles near it, as a point: where it is and its luminous intensity (candela, linear RGB).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glow {
    pub at: [f32; 3],
    pub candela: [f32; 3],
}

#[cfg(test)]
mod tests {
    use super::*;
    use somnium_ecs::World;
    #[test]
    fn staged_mirror_reuses_exact_particle_state_without_double_advancing() {
        let mut world = World::new();
        let entity = world.spawn((
            Transform::from_translation(glam::Vec3::new(1., 2., 3.)),
            ParticleEmitter {
                enabled: false,
                staged_reflection: true,
                local_space: true,
                flutter: 0.2,
                gravity: 0.0,
                particles: vec![ParticleState {
                    position: glam::Vec3::ZERO,
                    velocity: glam::Vec3::ZERO,
                    age: 0.2,
                    lifetime: 1.,
                    rotation_seed: 0.4,
                }],
                ..Default::default()
            },
        ));
        let matrix = glam::Mat4::from_scale(glam::Vec3::new(1., 1., -1.));
        let particles = simulate_particles_reflected(&mut world, 0.05, 1, Some(matrix));
        assert_eq!(particles.len(), 2);
        assert_eq!(particles[0].position, [1., 2., 3.]);
        assert_eq!(particles[1].position, [1., 2., -3.]);
        assert_eq!(particles[0].color, particles[1].color);
        assert_eq!(particles[0].flutter, particles[1].flutter);
        assert_eq!(particles[1].flags & 4, 4);
        assert_eq!(
            world.get::<ParticleEmitter>(entity).unwrap().particles[0].age,
            0.25
        );
        world
            .get_mut::<ParticleEmitter>(entity)
            .unwrap()
            .staged_reflection = false;
        assert_eq!(
            simulate_particles_reflected(&mut world, 0., 2, Some(matrix)).len(),
            1
        );
    }
    #[test]
    fn initial_rotation_stays_constant_as_particles_travel_without_spin() {
        let mut world = World::new();
        world.spawn((ParticleEmitter {
            enabled: false,
            burst: 1,
            gravity: 0.0,
            initial_speed: 0.0,
            velocity_bias: [2.0, 0.0, 0.0],
            rotation_spread: 1.0,
            spin: 0.0,
            flutter: 0.0,
            flutter_hz: 4.0,
            ..Default::default()
        },));
        let first = simulate_particles(&mut world, 0.01, 1);
        let moved = simulate_particles(&mut world, 0.05, 2);
        assert_ne!(first[0].position, moved[0].position);
        assert_eq!(first[0].rotation, moved[0].rotation);
    }

    #[test]
    fn manual_step_consumes_a_paused_burst_and_advances_only_one_fixed_tick() {
        let mut clock = crate::SimulationClock {
            state: crate::SimulationState::Paused,
            ..Default::default()
        };
        let mut world = World::new();
        let entity = world.spawn((ParticleEmitter {
            enabled: false,
            burst: 1,
            initial_speed: 1.0,
            gravity: 0.0,
            ..Default::default()
        },));
        simulate_particles(&mut world, simulation_delta(&clock, false, 0.04), 1);
        assert_eq!(world.get::<ParticleEmitter>(entity).unwrap().burst, 1);
        simulate_particles(&mut world, simulation_delta(&clock, true, 0.04), 2);
        assert_eq!(world.get::<ParticleEmitter>(entity).unwrap().burst, 0);
        simulate_particles(&mut world, simulation_delta(&clock, true, 0.04), 3);
        let emitter = world.get::<ParticleEmitter>(entity).unwrap();
        assert_eq!(emitter.particles.len(), 1);
        assert_eq!(emitter.particles[0].age, clock.fixed_delta_seconds);
        let position = emitter.particles[0].position;
        simulate_particles(&mut world, simulation_delta(&clock, false, 0.04), 4);
        let particle = world.get::<ParticleEmitter>(entity).unwrap().particles[0];
        assert_eq!(particle.age, clock.fixed_delta_seconds);
        assert_eq!(particle.position, position);
        for state in [
            crate::SimulationState::Editing,
            crate::SimulationState::Playing,
        ] {
            clock.state = state;
            assert_eq!(simulation_delta(&clock, false, 0.04), 0.04);
        }
    }
    #[test]
    fn local_particles_follow_emitter_but_world_smoke_keeps_its_birth_position() {
        let mut world = World::new();
        let entity = world.spawn((
            Transform::default(),
            ParticleEmitter {
                enabled: false,
                burst: 1,
                local_space: true,
                initial_speed: 0.0,
                gravity: 0.0,
                ..Default::default()
            },
        ));
        let first = simulate_particles(&mut world, 0.01, 1);
        world.get_mut::<Transform>(entity).unwrap().translation.x = 2.0;
        let second = simulate_particles(&mut world, 0.01, 2);
        assert_eq!(second[0].position[0] - first[0].position[0], 2.0);
        let p = world.get_mut::<ParticleEmitter>(entity).unwrap();
        p.particles.clear();
        p.local_space = false;
        p.burst = 1;
        let first = simulate_particles(&mut world, 0.01, 3);
        world.get_mut::<Transform>(entity).unwrap().translation.x = 5.0;
        let second = simulate_particles(&mut world, 0.01, 4);
        assert_eq!(first[0].position, second[0].position);
    }
    #[test]
    fn paused_burst_is_retained_and_missing_texture_never_becomes_a_round_proxy() {
        let mut world = World::new();
        let entity = world.spawn((ParticleEmitter {
            enabled: false,
            burst: 3,
            texture: somnium_asset::database::AssetId::from_relative_path(std::path::Path::new(
                "smoke.png",
            )),
            ..Default::default()
        },));
        assert!(simulate_particles(&mut world, 0.0, 0).is_empty());
        assert_eq!(world.get::<ParticleEmitter>(entity).unwrap().burst, 3);
        assert!(simulate_particles(&mut world, 0.01, 1).is_empty());
        let p = world.get_mut::<ParticleEmitter>(entity).unwrap();
        assert_eq!(p.particles.len(), 3);
        assert_eq!(p.burst, 0);
        p.texture_slot = 7;
        assert_eq!(simulate_particles(&mut world, 0.01, 2).len(), 3);
    }
    #[test]
    fn textured_flutter_changes_silhouette_over_time_and_respects_pause() {
        let mut world = World::new();
        let e = world.spawn((ParticleEmitter {
            enabled: false,
            burst: 1,
            gravity: 0.0,
            initial_speed: 0.0,
            flutter: 0.25,
            flutter_hz: 4.0,
            ..Default::default()
        },));
        let first = simulate_particles(&mut world, 0.01, 1)[0];
        let moved = simulate_particles(&mut world, 0.04, 2)[0];
        assert!((first.flutter - moved.flutter).abs() > 0.01);
        assert_eq!(first.position, moved.position);
        assert_eq!(
            moved.flutter,
            simulate_particles(&mut world, 0.0, 3)[0].flutter
        );
        world.get_mut::<ParticleEmitter>(e).unwrap().flutter = 0.0;
        assert_eq!(simulate_particles(&mut world, 0.04, 4)[0].flutter, 0.0);
    }
    #[test]
    fn a_flame_card_is_one_instance_on_its_clock_and_a_bed_lies_flat() {
        let mut world = World::new();
        let card = FlameCard {
            size: [0.6, 0.9],
            bed_v: 0.85,
            peak_cd_m2: 1100.0,
            frames: 96,
            columns: 12,
            rows: 8,
            down_row0: 8,
            fps: 30.0,
            clock: 0.0,
            small: false,
            lean: [0.1, 0.0],
            fade: 1.0,
            mean_rel: 0.2,
            power: 1.0,
        };
        let e = world.spawn((
            Transform::from_translation(glam::Vec3::new(1.0, 0.9, 2.0)),
            ParticleEmitter { card: Some(card), texture_slot: 3, ..Default::default() },
        ));
        world.spawn((ParticleEmitter {
            coals: Some(CoalBed { diameter: 0.32, hot_k: 1150.0, seed: 0.3, clock: 0.0, fade: 1.0 }),
            ..Default::default()
        },));
        let out = simulate_particles(&mut world, 0.05, 1);
        assert_eq!(out.len(), 2, "one card, one bed, nothing spawned");
        let c = out.iter().find(|p| p.flags & (1 << 5) != 0).expect("the card");
        assert_eq!(c.position, [1.0, 0.9, 2.0]);
        assert_eq!((c.uv_rect[0], c.uv_rect[2], c.uv_rect[3]), (1.0, 12.0, 8.0), "frame 1 at 1.5 frames in");
        assert!((c.uv_rect[1] - 0.5).abs() < 1e-4);
        assert_eq!(c.tip_tint, [0.85, 96.0, 8.0]);
        assert!((c.aspect - 1.5).abs() < 1e-4);
        assert!(out.iter().any(|p| p.flags == 1 << 6 && p.size == 0.32));
        assert!(world.get::<ParticleEmitter>(e).unwrap().particles.is_empty());
        // Out with its lamp: nothing drawn, but its clock still runs.
        world.get_mut::<ParticleEmitter>(e).unwrap().card.as_mut().unwrap().fade = 0.0;
        assert_eq!(simulate_particles(&mut world, 0.05, 2).len(), 1);
        assert!((world.get::<ParticleEmitter>(e).unwrap().card.unwrap().clock - 0.1).abs() < 1e-5);
    }

    #[test]
    fn sparks_carry_their_velocity_and_smoke_the_light_of_its_fire() {
        let mut world = World::new();
        world.spawn((ParticleEmitter {
            enabled: false,
            burst: 1,
            gravity: 0.0,
            initial_speed: 0.0,
            velocity_bias: [0.0, 2.0, 0.0],
            stretch: 1.0 / 60.0,
            ..Default::default()
        },));
        world.spawn((
            Transform::from_translation(glam::Vec3::new(0.0, 1.0, 0.0)),
            ParticleEmitter {
                enabled: false,
                burst: 1,
                gravity: 0.0,
                initial_speed: 0.0,
                lit: true,
                glow: Some(Glow { at: [0.0, 0.0, 0.0], candela: [30.0, 15.0, 5.0] }),
                ..Default::default()
            },
        ));
        let out = simulate_particles(&mut world, 0.01, 1);
        let spark = out.iter().find(|p| p.flags & (1 << 7) != 0).expect("a spark");
        assert_eq!(spark.extra, [0.0, 2.0, 0.0, 1.0 / 60.0]);
        let smoke = out.iter().find(|p| p.flags & (1 << 8) != 0).expect("lit smoke");
        assert!((smoke.extra[0] - 30.0).abs() < 1e-3 && (smoke.extra[3] - 1.0).abs() < 1e-6, "1 m from its fire: {:?}", smoke.extra);
    }

    #[test]
    fn flipbook_uv_tracks_lifetime_without_leaving_the_sheet() {
        let mut world = World::new();
        let entity = world.spawn((ParticleEmitter {
            enabled: false,
            atlas_columns: 4,
            atlas_rows: 2,
            atlas_frames: 6,
            ..Default::default()
        },));
        world
            .get_mut::<ParticleEmitter>(entity)
            .unwrap()
            .particles
            .push(ParticleState {
                rotation_seed: 0.0,
                position: glam::Vec3::ZERO,
                velocity: glam::Vec3::ZERO,
                age: 0.8,
                lifetime: 1.0,
            });
        let p = simulate_particles(&mut world, 0.0, 0)[0];
        assert_eq!(p.uv_rect, [0.0, 0.5, 0.25, 0.5]);
        assert_eq!(std::mem::size_of_val(&p), 96);
    }
}

impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            enabled: true,
            texture: somnium_asset::database::AssetId::NONE,
            aspect: 1.0,
            additive: false,
            lit: false,
            world_up: false,
            local_space: false,
            staged_reflection: false,
            tip_tint: glam::Vec3::ONE,
            atlas_columns: 1,
            atlas_rows: 1,
            atlas_frames: 1,
            atlas_fps: 0.0,
            rotation_spread: 0.0,
            spin: 0.0,
            flutter: 0.0,
            flutter_hz: 4.0,
            flame: false,
            burst: 0,
            texture_status: "Soft round particle".into(),
            texture_slot: -1,
            max_particles: 1000,
            spawn_rate: 100.0,
            lifetime: 3.0,
            initial_speed: 5.0,
            spread_angle: 0.8,
            size_start: 1.0,
            size_end: 0.2,
            color_over_life: somnium_ecs::curve::Gradient::ramp(
                [1.0, 0.4, 0.1, 1.0],
                [0.2, 0.0, 0.0, 0.0],
            ),
            velocity_bias: [0.0; 3],
            spawn_extents: [0.0; 3],
            gravity: 1.0,
            card: None,
            coals: None,
            glow: None,
            stretch: 0.0,
            particles: Vec::new(),
            spawn_accum: 0.0,
        }
    }
}
impl somnium_ecs::Component for ParticleEmitter {}

/// Simulate all particle emitters and return a flat list of GPU instances.
///
/// Call each frame in `about_to_wait` after physics and before `render()`.
/// `seed` increments each frame (used for deterministic pseudo-random spawn direction).
pub fn simulate_particles(
    world: &mut somnium_ecs::World,
    dt: f32,
    frame: u64,
) -> Vec<somnium_renderer::pass::particle::GpuParticle> {
    simulate_particles_reflected(world, dt, frame, None)
}

/// Advance once, then mirror opted-in instances without a second simulation.
/// The caller selects the visible staged mirror using the current camera.
pub(crate) fn simulate_particles_reflected(
    world: &mut somnium_ecs::World,
    dt: f32,
    frame: u64,
    reflection: Option<glam::Mat4>,
) -> Vec<somnium_renderer::pass::particle::GpuParticle> {
    use somnium_renderer::pass::particle::GpuParticle;

    let dt = if dt.is_finite() {
        dt.clamp(0.0, 0.1)
    } else {
        0.0
    };
    let mut gpu_particles = Vec::new();

    let emitter_entities: Vec<somnium_ecs::Entity> = world
        .entities_with::<ParticleEmitter>()
        .filter(|e| world.get::<ParticleEmitter>(*e).is_some())
        .collect();

    for entity in emitter_entities {
        // Borrow world piecemeal to satisfy the borrow checker.
        let origin = world
            .get::<WorldTransform>(entity)
            .map(|wt| glam::Vec3::new(wt.0.w_axis.x, wt.0.w_axis.y, wt.0.w_axis.z))
            .or_else(|| world.get::<Transform>(entity).map(|t| t.translation))
            .unwrap_or(glam::Vec3::ZERO);

        let Some(emitter) = world.get_mut::<ParticleEmitter>(entity) else {
            continue;
        };

        // A flame's card or a bed of coals: one instance where the emitter stands, its clock run by the simulation.
        if let Some(card) = emitter.card.as_mut() {
            card.clock += dt;
            let card = *card;
            if card.fade > 0.0 && emitter.texture_slot >= 0 && card.frames > 0 {
                let f = (card.clock * card.fps).max(0.0);
                let frame = (f.floor() as u64 % u64::from(card.frames)) as f32;
                gpu_particles.push(GpuParticle {
                    position: origin.to_array(),
                    size: card.size[0],
                    color: [card.peak_cd_m2, card.fade, card.lean[0], card.lean[1]],
                    tip_tint: [card.bed_v, card.frames as f32, card.down_row0 as f32],
                    aspect: (card.size[1] / card.size[0].max(1e-4)).clamp(0.01, 100.0),
                    uv_rect: [frame, f.fract(), card.columns.max(1) as f32, card.rows.max(1) as f32],
                    rotation: 0.0,
                    texture_index: emitter.texture_slot,
                    flags: 2 | (1 << 5) | (u32::from(card.small) << 9),
                    flutter: 0.0,
                    extra: [card.mean_rel, card.power, 0.0, 0.0],
                });
            }
            continue;
        }
        if let Some(bed) = emitter.coals.as_mut() {
            bed.clock += dt;
            let bed = *bed;
            if bed.fade > 0.0 {
                gpu_particles.push(GpuParticle {
                    position: origin.to_array(),
                    size: bed.diameter,
                    color: [bed.hot_k, bed.fade, bed.seed, bed.clock],
                    tip_tint: [0.0; 3],
                    aspect: 1.0,
                    uv_rect: [0.0; 4],
                    rotation: 0.0,
                    texture_index: -1,
                    flags: 1 << 6,
                    flutter: 0.0,
                    extra: [0.0; 4],
                });
            }
            continue;
        }

        // ── 1. Advance existing particles ─────────────────────────────────────
        let gravity = emitter.gravity;
        emitter.particles.retain_mut(|p| {
            p.age += dt;
            p.velocity.y -= gravity * dt;
            p.position += p.velocity * dt;
            p.age < p.lifetime
        });

        // ── 2. Spawn new particles ────────────────────────────────────────────
        emitter.spawn_accum += if emitter.enabled {
            emitter.spawn_rate.clamp(0.0, 100_000.0) * dt
        } else {
            0.0
        };
        let to_spawn = emitter.spawn_accum.floor() as u32;
        emitter.spawn_accum -= to_spawn as f32;
        let available = emitter
            .max_particles
            .min(10_000)
            .saturating_sub(emitter.particles.len() as u32);
        let burst = if dt > 0.0 {
            std::mem::take(&mut emitter.burst).min(10_000)
        } else {
            0
        };
        let count = to_spawn.saturating_add(burst).min(available);

        let speed = emitter.initial_speed;
        let spread = emitter.spread_angle;
        let lifetime = emitter.lifetime.clamp(0.001, 3600.0);

        for i in 0..count {
            // Deterministic LCG pseudo-random — good enough for particles.
            let seed = frame
                .wrapping_mul(1_000_003)
                .wrapping_add((i as u64).wrapping_mul(6_364_136_223_846_793_005))
                // Each emitter its own draw: with the frame and the particle alone, every emitter spawned the same
                // directions on the same frame, and a street of lanterns flickered in step (Sunscar, 13 Nov 2026).
                .wrapping_add(u64::from(entity.index()).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let r1 = ((seed >> 33) & 0xFFFF) as f32 / 65535.0; // 0..1
            let r2 = ((seed >> 17) & 0xFFFF) as f32 / 65535.0 * 2.0 * std::f32::consts::PI;
            let theta = r1 * spread;
            let dir = glam::Vec3::new(theta.sin() * r2.cos(), theta.cos(), theta.sin() * r2.sin());
            // CONTROL-N: a spawn volume and a constant velocity, both zero for
            // every emitter authored before this existed.
            let extents = glam::Vec3::from(emitter.spawn_extents);
            let jitter = if extents == glam::Vec3::ZERO {
                glam::Vec3::ZERO
            } else {
                let r3 = ((seed >> 5) & 0xFFFF) as f32 / 65535.0;
                let r4 = ((seed >> 41) & 0xFFFF) as f32 / 65535.0;
                let r5 = ((seed >> 23) & 0xFFFF) as f32 / 65535.0;
                (glam::Vec3::new(r3, r4, r5) * 2.0 - glam::Vec3::ONE) * extents
            };
            emitter.particles.push(ParticleState {
                position: if emitter.local_space {
                    jitter
                } else {
                    origin + jitter
                },
                velocity: dir * speed + glam::Vec3::from(emitter.velocity_bias),
                rotation_seed: r2 / std::f32::consts::PI - 1.0,
                age: 0.0,
                lifetime,
            });
        }

        // ── 3. Emit GPU instances ─────────────────────────────────────────────
        let size_start = emitter.size_start;
        let size_end = emitter.size_end;
        // CONTROL-K: the ramp is sampled per particle rather than baked into a
        // table, because an emitter's particle count is the small number here
        // and a table would need invalidating whenever the ramp was edited —
        // which is every frame of a drag.
        let ramp = &emitter.color_over_life;

        for p in &emitter.particles {
            if emitter.texture != somnium_asset::database::AssetId::NONE && emitter.texture_slot < 0
            {
                continue;
            }
            let frac = (p.age / p.lifetime).clamp(0.0, 1.0);
            let size = size_start + (size_end - size_start) * frac;
            let color = ramp.evaluate(frac);
            let cols = emitter.atlas_columns.clamp(1, 64);
            let rows = emitter.atlas_rows.clamp(1, 64);
            let frames = emitter.atlas_frames.clamp(1, cols * rows);
            let frame = if emitter.atlas_fps > 0.0 {
                (p.age * emitter.atlas_fps.min(240.0)) as u32 % frames
            } else {
                ((frac * frames as f32) as u32).min(frames - 1)
            };
            gpu_particles.push(GpuParticle {
                position: (p.position
                    + if emitter.local_space {
                        origin
                    } else {
                        glam::Vec3::ZERO
                    })
                .to_array(),
                size,
                color,
                tip_tint: emitter.tip_tint.to_array(),
                aspect: emitter.aspect.clamp(0.01, 100.0),
                // An untextured flame tongue reads no atlas: it takes its seed and age, for its turbulence.
                uv_rect: if emitter.flame && emitter.texture_slot < 0 {
                    [p.rotation_seed, p.age, 0.0, 0.0]
                } else {
                    [
                        (frame % cols) as f32 / cols as f32,
                        (frame / cols) as f32 / rows as f32,
                        1.0 / cols as f32,
                        1.0 / rows as f32,
                    ]
                },
                rotation: emitter.spin * p.age + emitter.rotation_spread * p.rotation_seed,
                texture_index: emitter.texture_slot,
                flags: u32::from(emitter.additive)
                    | (u32::from(emitter.world_up) << 1)
                    | (u32::from(emitter.lit) << 3)
                    | (u32::from(emitter.flame) << 4)
                    | (u32::from(emitter.stretch > 0.0) << 7)
                    | (u32::from(emitter.lit && emitter.glow.is_some()) << 8),
                flutter: emitter.flutter.clamp(0.0, 0.4)
                    * ((p.age * emitter.flutter_hz.clamp(0.0, 30.0) * std::f32::consts::TAU
                        + p.rotation_seed * 3.0)
                        .sin()
                        + 0.35
                            * (p.age * emitter.flutter_hz.clamp(0.0, 30.0) * 15.3
                                + p.rotation_seed * 7.0)
                                .sin())
                    / 1.35,
                // A spark: its velocity and the shutter it streaks over. Lit smoke near a fire: the illuminance the
                // fire puts on it, lux, its distance kept over 0.2 m (a puff is no point).
                extra: if emitter.stretch > 0.0 {
                    let v = p.velocity;
                    [v.x, v.y, v.z, emitter.stretch.min(0.1)]
                } else if let (true, Some(glow)) = (emitter.lit, emitter.glow) {
                    let at = p.position + if emitter.local_space { origin } else { glam::Vec3::ZERO };
                    let d2 = at.distance_squared(glam::Vec3::from(glow.at)).max(0.04);
                    [glow.candela[0] / d2, glow.candela[1] / d2, glow.candela[2] / d2, 1.0]
                } else {
                    [0.0; 4]
                },
            });
            if emitter.staged_reflection
                && let Some(matrix) = reflection
            {
                let mut reflected = *gpu_particles.last().unwrap();
                reflected.position = matrix
                    .transform_point3(glam::Vec3::from_array(reflected.position))
                    .to_array();
                reflected.rotation = -reflected.rotation;
                reflected.flags |= 4; // mirror the sprite texture horizontally
                gpu_particles.push(reflected);
            }
        }
    }

    gpu_particles
}
