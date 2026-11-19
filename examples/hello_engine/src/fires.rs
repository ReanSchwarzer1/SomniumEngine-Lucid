//! Fire for every made flame (Sunscar, 13 Nov 2026; `somnium_core::fire`). Where the engine no longer draws a
//! `kp_flame` mesh, a flame burns in the particle pass. Each burns while its lamp does: the light it stands in
//! (`somnium.Lamp`, dusk to dawn), or, for a flame with no light of its own, one made for it and switched the same
//! way.
//!
//! Since 19 Nov 2026 (sunscar-fire, `design/fire/FIRE.md`) a flame burns from its kind's simulated flipbook
//! (`assets/fire/<kind>.png` and `.fire.json`, our own Mantaflow runs): one card at the fire's real luminance, scaled
//! to the light it gives; over an open fire a bed of coals, smoke its fire lights from below, and sparks drawn as the
//! streaks a camera sees; and the light follows the book's own frames, its output and its place, so the ground
//! flickers as the flame does and a fire's shadows sway with its body. A kind with no book yet burns as on 13 Nov.
//! Only the near level of detail burns: a flame under a far level (`LOD1`...) is the same flame.
//!
//! What is made here is never saved: it has no persistent id, so no scene and no edits layer holds it, and it goes
//! when the flame's entity goes (its cell leaves, the scene closes).

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use somnium_asset::database::AssetId;
use somnium_core::fire::{Flame, FlameBook, FlameKind, FlameSource};
use somnium_core::light_flicker::LampComponent;
use somnium_core::weather::WeatherComponent;
use somnium_core::{CoalBed, FlameCard, Glow, LightComponent, Name, ParticleEmitter, Parent, Transform, WorldTransform};
use somnium_ecs::curve::{Gradient, GradientStop};
use somnium_ecs::{Entity, World};

/// A light this near a flame is its own (a lantern's stands 3 cm over its wick), metres.
const OWN_LIGHT: f32 = 1.2;
/// Two flames this near are one (the same flame in two levels of detail), metres.
const SAME_SPOT: f32 = 0.05;
/// Smoke is made within this of the eye, and sparks within the second, metres.
const SMOKE_REACH: f32 = 90.0;
const SPARK_REACH: f32 = 45.0;

struct Burning {
    carrier: Entity,
    spot: (i32, i32, i32),
    /// Everything made for it: emitters, and a light when it had none.
    made: Vec<Entity>,
    /// The emitters that stop when the lamp is out.
    emitters: Vec<Entity>,
    smoke: Option<Entity>,
    sparks: Option<Entity>,
    /// A book's card and its bed of coals.
    card: Option<Entity>,
    coals: Option<Entity>,
    /// The light it burns with.
    light: Option<Entity>,
    lit: Option<bool>,
    /// A fire's (brazier, torch) rather than a lamp's: its light flickers hard.
    open: bool,
    /// The book it burns from: the book, its kind, its scale over the book's own size, its light's lumens, the light's
    /// colour per candela (linear RGB) and its resting height over the bed (m).
    book: Option<BookFire>,
    /// Where it stands, world.
    at: Vec3,
}

struct BookFire {
    book: Arc<FlameBook>,
    kind: FlameKind,
    scale: f32,
    lumens: f32,
    colour: Vec3,
    rest: glam::Vec2,
}

#[derive(Default)]
pub struct Fires {
    burning: Vec<Burning>,
    carriers: HashSet<Entity>,
    spots: HashSet<(i32, i32, i32)>,
    /// Each kind's book and its atlas, read once the project is known.
    books: Option<HashMap<FlameKind, (Arc<FlameBook>, AssetId)>>,
    /// Lights a book drives this frame: their output over the lamp's, and how far the light stands from its rest.
    drive: HashMap<Entity, (f32, Vec3)>,
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

/// A blackbody's colour at `kelvin`, linear Rec.709 at unit luminance, negatives cut (as particle.wgsl's table).
fn blackbody(kelvin: f32) -> Vec3 {
    const T: [(f32, [f32; 3]); 6] = [
        (1000.0, [4.32, 0.12, 0.0]),
        (1200.0, [3.80, 0.28, 0.0]),
        (1400.0, [3.36, 0.41, 0.0]),
        (1600.0, [3.01, 0.51, 0.0]),
        (1800.0, [2.72, 0.59, 0.0]),
        (2000.0, [2.48, 0.66, 0.02]),
    ];
    let k = kelvin.clamp(T[0].0, T[5].0);
    let i = T.windows(2).position(|w| k <= w[1].0).unwrap_or(4);
    let (a, b) = (T[i], T[i + 1]);
    Vec3::from(a.1).lerp(Vec3::from(b.1), (k - a.0) / (b.0 - a.0))
}

/// Whether `entity` stands under a far level of detail: a `LOD1`, `LOD2`... below a `somnium.LodGroup` (the near
/// level `LOD0` carries the same flame, and burns at every distance).
fn under_far_level(world: &World, entity: Entity) -> bool {
    let mut at = entity;
    for _ in 0..32 {
        if world.get::<Name>(at).is_some_and(|name| {
            let name = name.as_str();
            name.strip_prefix("LOD").and_then(|n| n.parse::<u32>().ok()).is_some_and(|n| n > 0)
        }) {
            return true;
        }
        match world.get::<Parent>(at) {
            Some(parent) => at = parent.entity,
            None => return false,
        }
    }
    false
}

/// What each kind does besides its card: its light's colour, a made light's reach, its bed of coals' heat (0: none),
/// its smoke and its sparks. Numbers in `design/fire/FIRE.md`.
struct Look {
    kelvin: f32,
    range: f32,
    coal_k: f32,
    /// spawn rate /s, lifetime s, start and end size m, speed m/s, albedo, opacity, most at once
    smoke: Option<(f32, f32, f32, f32, f32, f32, f32, u32)>,
    /// spawn rate /s, lifetime s, speed m/s, spread rad, most at once
    sparks: Option<(f32, f32, f32, f32, u32)>,
}

fn look(kind: FlameKind) -> Look {
    match kind {
        FlameKind::Candle => Look { kelvin: 1900.0, range: 4.0, coal_k: 0.0, smoke: None, sparks: None },
        FlameKind::Lantern => Look { kelvin: 1900.0, range: 6.0, coal_k: 0.0, smoke: None, sparks: None },
        // Pitch burns sooty: thick black smoke from the flame's tips, a few drops and sparks.
        FlameKind::Torch => Look {
            kelvin: 1700.0,
            range: 9.0,
            coal_k: 0.0,
            smoke: Some((9.0, 3.0, 0.1, 0.9, 0.8, 0.05, 0.7, 40)),
            sparks: Some((1.2, 1.0, 1.1, 0.5, 6)),
        },
        FlameKind::Brazier => Look {
            kelvin: 1750.0,
            range: 10.0,
            coal_k: 1150.0,
            smoke: Some((5.0, 6.0, 0.25, 1.6, 0.45, 0.25, 0.45, 40)),
            sparks: Some((2.5, 1.6, 1.0, 0.5, 10)),
        },
        // A hearth's smoke goes up its flue: short-lived, it never leaves the chimney.
        FlameKind::Hearth => Look {
            kelvin: 1800.0,
            range: 9.0,
            coal_k: 1120.0,
            smoke: Some((2.0, 1.2, 0.2, 0.5, 0.35, 0.2, 0.35, 8)),
            sparks: Some((0.8, 1.0, 0.8, 0.4, 4)),
        },
        // Under the bellows the bed is the fire: little smoke, sparks in showers when it is worked.
        FlameKind::Forge => Look {
            kelvin: 1900.0,
            range: 9.0,
            coal_k: 1380.0,
            smoke: Some((2.0, 3.0, 0.2, 0.8, 0.35, 0.3, 0.25, 10)),
            sparks: Some((0.6, 0.8, 1.4, 0.6, 6)),
        },
        FlameKind::Bonfire => Look {
            kelvin: 1750.0,
            range: 25.0,
            coal_k: 1150.0,
            smoke: Some((4.0, 10.0, 0.8, 4.0, 1.4, 0.3, 0.4, 60)),
            sparks: Some((12.0, 2.2, 2.0, 0.6, 40)),
        },
    }
}

impl Fires {
    /// A light's output over its lamp's this frame, and how far from where it stands it shines (its book's frame).
    #[must_use]
    pub fn drive(&self, light: Entity) -> (f32, Vec3) {
        self.drive.get(&light).copied().unwrap_or((1.0, Vec3::ZERO))
    }

    /// Once a frame, after the transforms: light what arrived, put out what left, and burn each while its lamp does.
    /// `project` is where the books are; `eye` the camera's place, for what is made near it only.
    pub fn update(&mut self, world: &mut World, night: f32, time: f32, project: Option<&Path>, eye: Option<Vec3>) {
        if self.books.is_none()
            && let Some(project) = project
        {
            let books: HashMap<FlameKind, (Arc<FlameBook>, AssetId)> = FlameKind::ALL
                .into_iter()
                .filter_map(|kind| {
                    let book = FlameBook::load(project, kind)?;
                    let asset = AssetId::from_relative_path(std::path::Path::new(&book.texture));
                    Some((kind, (Arc::new(book), asset)))
                })
                .collect();
            tracing::info!(books = books.len(), "fire: the kinds' simulated books");
            self.books = Some(books);
        }
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
            for (carrier, flames) in arrived {
                self.carriers.insert(carrier);
                // The near level carries the same flame, and burns at every distance.
                if under_far_level(world, carrier) {
                    continue;
                }
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
                    let book = self.books.as_ref().and_then(|books| books.get(&flame.kind)).cloned();
                    let fire = match book {
                        Some((book, asset)) => burn_book(world, carrier, model, flame, spot, own, book, asset),
                        None => burn(world, carrier, model, flame, spot, own, false),
                    };
                    self.burning.push(fire);
                }
            }
        }

        // Each burns while its lamp does; smoke goes with the wind, a third as fast, and a fire leans into it.
        let wind = world
            .entities_with::<WeatherComponent>()
            .next()
            .and_then(|entity| world.get::<WeatherComponent>(entity))
            .map_or([0.0; 2], WeatherComponent::wind_vector);
        let wind = glam::Vec2::new(wind[0], wind[1]);
        self.drive.clear();
        for fire in &mut self.burning {
            let lit = fire.light.is_none_or(|light| {
                match (world.get::<LampComponent>(light), world.get::<WorldTransform>(light)) {
                    (Some(lamp), Some(at)) => lamp.factor(night, at.0.w_axis.truncate(), time) > 0.02,
                    _ => true,
                }
            });
            let near = eye.map_or(0.0, |eye| eye.distance(fire.at));
            if let Some(b) = &fire.book {
                // The book drives the light: the lamp's own flicker is the flame's now.
                if let Some(light) = fire.light
                    && let Some(lamp) = world.get_mut::<LampComponent>(light)
                    && lamp.flicker != 0.0
                {
                    lamp.flicker = 0.0;
                }
                let clock = fire
                    .card
                    .and_then(|card| world.get::<ParticleEmitter>(card))
                    .and_then(|emitter| emitter.card)
                    .map_or(0.0, |card| card.clock);
                let power = b.book.power_at(clock);
                // The light's sway: the book's light centre against its mean, across (as the card faces the eye) and up.
                let centre = b.book.centre_at(clock) * b.scale - b.rest;
                let across = eye.map_or(Vec3::X, |eye| {
                    let f = (eye - fire.at) * Vec3::new(1.0, 0.0, 1.0);
                    let f = f.try_normalize().unwrap_or(Vec3::Z);
                    Vec3::new(f.z, 0.0, -f.x)
                });
                let shift = across * centre.x + Vec3::Y * centre.y;
                let light_at = fire.light.and_then(|light| world.get::<WorldTransform>(light)).map_or(fire.at, |w| w.0.w_axis.truncate());
                if let Some(light) = fire.light {
                    self.drive.insert(light, (power, shift));
                }
                let fade = if lit { 1.0 } else { 0.0 };
                if let Some(card) = fire.card
                    && let Some(emitter) = world.get_mut::<ParticleEmitter>(card)
                    && let Some(card) = emitter.card.as_mut()
                {
                    card.fade = fade;
                    // A fire leans into the wind, a lantern's flame inside its glass does not.
                    let lean = if b.kind.open() { (0.12 * wind.length()).min(0.6) } else if b.kind == FlameKind::Candle { 0.02 * wind.length().min(2.0) } else { 0.0 };
                    let dir = wind.try_normalize().unwrap_or(glam::Vec2::ZERO);
                    card.lean = [dir.x * lean * card.size[1], dir.y * lean * card.size[1]];
                }
                if let Some(coals) = fire.coals
                    && let Some(emitter) = world.get_mut::<ParticleEmitter>(coals)
                    && let Some(bed) = emitter.coals.as_mut()
                {
                    bed.fade = fade;
                }
                let glow = Glow {
                    at: (light_at + shift).to_array(),
                    candela: (b.colour * (b.lumens / (4.0 * std::f32::consts::PI) * power)).to_array(),
                };
                if let Some(smoke) = fire.smoke
                    && let Some(emitter) = world.get_mut::<ParticleEmitter>(smoke)
                {
                    emitter.glow = Some(glow);
                    emitter.enabled = lit && near < SMOKE_REACH;
                }
                if let Some(sparks) = fire.sparks
                    && let Some(emitter) = world.get_mut::<ParticleEmitter>(sparks)
                {
                    emitter.enabled = lit && near < SPARK_REACH;
                }
            } else {
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
                }
            }
            fire.lit = Some(lit);
            if let Some(smoke) = fire.smoke
                && let Some(emitter) = world.get_mut::<ParticleEmitter>(smoke)
            {
                let drift = wind / 3.0;
                let drift = if drift.length() < 0.25 { drift.try_normalize().unwrap_or(glam::Vec2::X) * 0.25 } else { drift };
                emitter.velocity_bias = [drift.x, emitter.velocity_bias[1], drift.y];
            }
        }
    }
}

/// A child entity of the flame's carrier at `local` (the flame's own space), with an emitter or a light.
fn child(
    world: &mut World,
    carrier: Entity,
    model: Mat4,
    local: Vec3,
    name: &str,
    emitter: Option<ParticleEmitter>,
    light: Option<(LightComponent, LampComponent)>,
) -> Entity {
    let local = Transform::from_translation(local);
    let world_matrix = WorldTransform(model * Mat4::from_translation(local.translation));
    let entity = world.spawn((Name::new(name), local, world_matrix, Parent { entity: carrier }));
    if let Some(emitter) = emitter {
        let _ = world.insert_component(entity, emitter);
    }
    if let Some((light, lamp)) = light {
        let _ = world.insert_component(entity, light);
        let _ = world.insert_component(entity, lamp);
    }
    entity
}

/// Make one flame's fire from its kind's book: its card, and for an open fire a bed of coals, smoke and sparks; and a
/// light if it has none. All are children of the flame's entity, so they go where it is moved.
#[allow(clippy::too_many_arguments)]
fn burn_book(
    world: &mut World,
    carrier: Entity,
    model: Mat4,
    flame: &Flame,
    spot: (i32, i32, i32),
    own: Option<Entity>,
    book: Arc<FlameBook>,
    asset: AssetId,
) -> Burning {
    let kind = flame.kind;
    let lk = look(kind);
    // The model's scale up its height: sizes are in world metres, offsets in the flame's own space.
    let unit = model.y_axis.truncate().length().max(1e-3);
    let at = model.transform_point3(flame.base);
    let seed = salt(at);
    // The book's size to the made flame's: a lamp's or a torch's by its height, a fire on a bed by its spread.
    let scale = match kind {
        FlameKind::Candle | FlameKind::Lantern | FlameKind::Torch => flame.height * unit / book.flame_height_m.max(1e-3),
        _ => 2.0 * flame.radius * unit / book.bed.width_m.max(1e-3),
    }
    .clamp(0.4, 2.5);
    let lumens = own
        .and_then(|light| world.get::<LightComponent>(light).map(|l| l.intensity))
        .filter(|lumens| *lumens > 0.0)
        .unwrap_or(match kind {
            FlameKind::Candle => 12.0,
            FlameKind::Lantern => 30.0,
            _ => 4.0 * std::f32::consts::PI * book.intensity_cd * scale * scale,
        });
    // The card's light is its lamp's: its luminous intensity face on is the lamp's candela.
    let peak = book.peak_cd_m2 * (lumens / (4.0 * std::f32::consts::PI)) / (book.intensity_cd * scale * scale).max(1e-6);
    let mut made = Vec::new();
    let mut emitters = Vec::new();
    let card = ParticleEmitter {
        texture: asset,
        max_particles: 0,
        spawn_rate: 0.0,
        card: Some(FlameCard {
            size: [book.card_m[0] * scale, book.card_m[1] * scale],
            bed_v: book.bed.v,
            peak_cd_m2: peak,
            frames: book.frames,
            columns: book.columns,
            rows: book.rows,
            down_row0: book.down_row0(),
            fps: book.fps,
            clock: seed * book.frames as f32 / book.fps.max(1.0),
            small: !kind.open(),
            lean: [0.0; 2],
            fade: 0.0,
        }),
        ..Default::default()
    };
    let card = child(world, carrier, model, flame.base, "fire: card", Some(card), None);
    made.push(card);
    let coals = (lk.coal_k > 0.0).then(|| {
        let bed = ParticleEmitter {
            max_particles: 0,
            spawn_rate: 0.0,
            coals: Some(CoalBed {
                diameter: book.bed.width_m * scale * 0.95,
                hot_k: lk.coal_k,
                seed,
                clock: seed * 30.0,
                fade: 0.0,
            }),
            ..Default::default()
        };
        let bed = child(world, carrier, model, flame.base + Vec3::Y * (0.005 / unit), "fire: coals", Some(bed), None);
        made.push(bed);
        bed
    });
    let height = book.flame_height_m * scale;
    let smoke = lk.smoke.map(|(rate, life, size0, size1, speed, albedo, opacity, most)| {
        // Born at the flame's tips, soot-dark close over it and greyer as it thins; lit by the sun and the sky, and
        // from below by its own fire.
        let a = albedo;
        let smoke = ParticleEmitter {
            lit: true,
            max_particles: most,
            spawn_rate: rate,
            lifetime: life,
            initial_speed: speed * scale.sqrt(),
            spread_angle: 0.18,
            gravity: -0.05,
            size_start: size0 * scale,
            size_end: size1 * scale,
            rotation_spread: 3.1,
            spin: 0.15,
            spawn_extents: [book.bed.width_m * scale * 0.2, 0.0, book.bed.width_m * scale * 0.2],
            color_over_life: ramp(&[
                (0.0, [a * 0.5, a * 0.46, a * 0.42, 0.0]),
                (0.08, [a * 0.6, a * 0.56, a * 0.5, opacity]),
                (0.45, [a, a * 0.96, a * 0.9, opacity * 0.55]),
                (1.0, [a * 1.1, a * 1.08, a * 1.04, 0.0]),
            ]),
            spawn_accum: seed,
            ..Default::default()
        };
        let smoke = child(world, carrier, model, flame.base + Vec3::Y * (height * 0.85 / unit), "fire: smoke", Some(smoke), None);
        made.push(smoke);
        emitters.push(smoke);
        smoke
    });
    let sparks = lk.sparks.map(|(rate, life, speed, spread, most)| {
        // A spark is a speck of burning wood at 1,600-1,800 K cooling to 1,000 K as it rises: a 5 mm sprite carries a
        // 1 mm speck's light, drawn as the streak a 1/60 s shutter makes.
        let hot = blackbody(1750.0) * 2500.0;
        let warm = blackbody(1350.0) * 600.0;
        let cool = blackbody(1050.0) * 60.0;
        let sparks = ParticleEmitter {
            additive: true,
            max_particles: most,
            spawn_rate: rate,
            lifetime: life,
            initial_speed: speed * scale.sqrt(),
            spread_angle: spread,
            gravity: -0.6,
            size_start: 0.005,
            size_end: 0.003,
            stretch: 1.0 / 60.0,
            spawn_extents: [book.bed.width_m * scale * 0.35, 0.0, book.bed.width_m * scale * 0.35],
            color_over_life: ramp(&[
                (0.0, [hot.x, hot.y, hot.z, 1.0]),
                (0.35, [warm.x, warm.y, warm.z, 1.0]),
                (1.0, [cool.x, cool.y, cool.z, 0.0]),
            ]),
            spawn_accum: seed,
            ..Default::default()
        };
        let sparks = child(world, carrier, model, flame.base + Vec3::Y * (height * 0.25 / unit), "fire: sparks", Some(sparks), None);
        made.push(sparks);
        emitters.push(sparks);
        sparks
    });
    // A flame with no light of its own gets one, lit and put out as a lamp is, where the book's light stands on
    // average; the book moves it from there.
    let rest = {
        let n = book.light_centre_m.len().max(1) as f32;
        book.light_centre_m.iter().fold(glam::Vec2::ZERO, |s, c| s + glam::Vec2::from(*c)) / n * scale
    };
    let light = own.or_else(|| {
        let mut light = LightComponent::point(lumens, lk.range);
        light.color_temperature_k = lk.kelvin;
        light.source_radius = (book.card_m[0] * scale * 0.25).clamp(0.005, 0.5);
        let lamp = LampComponent { switched: true, flicker: 0.0 };
        let l = child(world, carrier, model, flame.base + Vec3::new(rest.x, rest.y, 0.0) / unit, "fire: light", None, Some((light, lamp)));
        made.push(l);
        Some(l)
    });
    let rest = if own.is_some() {
        // A place's own light stands where it was put: the book sways it about there.
        own.and_then(|l| world.get::<WorldTransform>(l)).map_or(rest, |w| {
            let d = w.0.w_axis.truncate() - at;
            glam::Vec2::new(0.0, d.y)
        })
    } else {
        rest
    };
    let colour = blackbody(lk.kelvin);
    Burning {
        carrier,
        spot,
        made,
        emitters,
        smoke,
        sparks,
        card: Some(card),
        coals,
        light,
        lit: None,
        open: kind.open(),
        book: Some(BookFire { book, kind, scale, lumens, colour: colour / colour.dot(Vec3::new(0.2126, 0.7152, 0.0722)), rest }),
        at,
    }
}

/// Make one flame's fire as on 13 Nov (a kind with no book yet): its emitters (and a light if it has none) as children
/// of the flame's entity, so they go where it is moved. `model` is that entity's world matrix, `own` the light it
/// stands in.
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
    let mut make = |world: &mut World, name: &str, rise: f32, emitter: Option<ParticleEmitter>, light: Option<(LightComponent, LampComponent)>| {
        let entity = child(world, carrier, model, flame.base + Vec3::Y * (rise / scale), name, emitter, light);
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
        emitters.push(make(world, "fire: body", lick * 1.7 * 0.4, Some(fire), None));
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
        emitters.push(make(world, "fire: coals", 0.02, Some(coals), None));
        // No glow sprite over the fire: it read as a soft orb hanging over the brazier (13 Nov 12:32, beside real
        // photographs); the smoke and the fire's own light carry it.
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
        emitters.push(make(world, "fire: sparks", height * 0.4, Some(sparks), None));
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
            let smoke = make(world, "fire: smoke", lick * 1.7 + 0.3, Some(smoke), None);
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
        emitters.push(make(world, "fire: flame", width * aspect * 0.5, Some(tongues), None));
        None
    };
    // A flame with no light of its own gets a small one, lit and put out as a lamp is and wavering with it; the
    // renderer keeps the lights that matter most where the eye is.
    let light = own.or_else(|| {
        let mut light = LightComponent::point(if open { 220.0 + 900.0 * height } else { 35.0 }, if open { 9.0 } else { 4.0 });
        light.color_temperature_k = if open { 1700.0 } else { 1900.0 };
        light.source_radius = radius.clamp(0.01, 0.2);
        let lamp = LampComponent { switched: true, flicker: if open { 0.45 } else { 0.15 } };
        Some(make(world, "fire: light", height * 0.6, None, Some((light, lamp))))
    });
    Burning {
        carrier,
        spot,
        made,
        emitters,
        smoke,
        sparks: None,
        card: None,
        coals: None,
        light,
        lit: None,
        open,
        book: None,
        at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flame_under_a_far_level_does_not_burn_twice() {
        let mut world = World::new();
        let root = world.spawn((Name::new("torch"),));
        let near = world.spawn((Name::new("LOD0"), Parent { entity: root }));
        let far = world.spawn((Name::new("LOD1"), Parent { entity: root }));
        let near_flame = world.spawn((Name::new("kp_flame"), Parent { entity: near }));
        let far_flame = world.spawn((Name::new("kp_flame"), Parent { entity: far }));
        assert!(!under_far_level(&world, near_flame));
        assert!(under_far_level(&world, far_flame));
        assert!(!under_far_level(&world, root));
    }

    #[test]
    fn blackbody_runs_from_red_to_yellow() {
        let (a, b) = (blackbody(1000.0), blackbody(1900.0));
        assert!(a.x / a.y.max(1e-3) > b.x / b.y, "cooler is redder");
        assert_eq!(blackbody(500.0), blackbody(1000.0));
    }
}
