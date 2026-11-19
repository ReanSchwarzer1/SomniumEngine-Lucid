//! Real fire in place of the made flame (Sunscar, 13 Nov 2026).
//!
//! Every flame in the world was one made mesh: a lathed teardrop of the emissive material `kp_flame`
//! (`tools/kept_kit.py` `flame()`, behind its lanterns, candles, braziers and the city's tar torches). Lit and still,
//! it read as a glowing crystal, by day and by night ("pls replace these weird crystals with actual fire particle
//! effects and smokes and a light source", the developer, 12 Nov). The engine no longer draws a primitive of that
//! material: [`scene_flames`] finds the flames in it as it is restored, the entity carries them as a
//! [`FlameSource`], and the game's shell burns them (flickering tongues, embers and smoke in the particle pass, and a
//! light that wavers with them).
//!
//! Since 19 Nov 2026 (sunscar-fire, `design/fire/FIRE.md`) each flame has a [`FlameKind`], named by its material
//! (`kp_flame_<kind>`) or inferred from its size, and each kind burns from its own simulated flipbook, a
//! [`FlameBook`] (`assets/fire/<kind>.png` and `.fire.json`, made by the game's `tools/fire/fire_book.py` from Blender
//! Mantaflow runs): a card at the fire's real luminance, its light driven by the book's own frames.

use std::collections::HashMap;
use std::sync::Arc;

/// The material a made flame is drawn with. A kind may follow it: `kp_flame_brazier` ([`FlameKind::from_material`]).
pub const FLAME_MATERIAL: &str = "kp_flame";

/// Whether a material is a made flame's: `kp_flame`, or `kp_flame_` and a kind's name.
#[must_use]
pub fn is_flame_material(name: &str) -> bool {
    name == FLAME_MATERIAL || FlameKind::from_material(name).is_some()
}

/// What burns. Each kind has its own simulated flipbook, smoke, sparks and light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FlameKind {
    /// A candle's flame: 2-3 cm, laminar, a dim blue foot.
    Candle,
    /// An oil lamp's or a lantern's flame: about 5 cm.
    Lantern,
    /// A tar torch: a wrapped head burning 0.3-0.5 m tall, sooty.
    Torch,
    /// A wood fire in a brazier's pan or a fire basket.
    Brazier,
    /// Logs on a hearth, under a chimney.
    Hearth,
    /// A smith's coal bed under the bellows' blast.
    Forge,
    /// A great open fire of piled wood.
    Bonfire,
}

impl FlameKind {
    /// Every kind, in the order of their size.
    pub const ALL: [FlameKind; 7] = [Self::Candle, Self::Lantern, Self::Torch, Self::Brazier, Self::Hearth, Self::Forge, Self::Bonfire];

    /// Its name, as in `kp_flame_<name>` and `assets/fire/<name>.png`.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Candle => "candle",
            Self::Lantern => "lantern",
            Self::Torch => "torch",
            Self::Brazier => "brazier",
            Self::Hearth => "hearth",
            Self::Forge => "forge",
            Self::Bonfire => "bonfire",
        }
    }

    /// The kind a material names (`kp_flame_<kind>`), if it names one.
    #[must_use]
    pub fn from_material(material: &str) -> Option<Self> {
        let kind = material.strip_prefix("kp_flame_")?;
        Self::ALL.into_iter().find(|k| k.name() == kind)
    }

    /// An open fire, rather than a lamp's or a candle's flame: it smokes and sparks.
    #[must_use]
    pub fn open(self) -> bool {
        !matches!(self, Self::Candle | Self::Lantern)
    }

    /// Lies on a bed of coals (a torch's head burns in the air).
    #[must_use]
    pub fn on_coals(self) -> bool {
        matches!(self, Self::Brazier | Self::Hearth | Self::Forge | Self::Bonfire)
    }

    /// How far a fire's tongues spread from the first and still burn as one, metres.
    #[must_use]
    pub fn spread(self) -> f32 {
        match self {
            Self::Bonfire => 2.0,
            Self::Hearth | Self::Forge => 0.7,
            _ => FIRE_SPREAD,
        }
    }

    /// The kind a made flame's size says, when its material names none: a single tongue under 3.5 cm is a candle's,
    /// under 9 cm a lantern's; an open fire spreading under 12 cm is a torch, wider a brazier. Forge, hearth and
    /// bonfire are never guessed: their models name them.
    #[must_use]
    pub fn infer(height: f32, radius: f32, tongues: u32) -> Self {
        if tongues <= 1 && height < CANDLE_HEIGHT {
            Self::Candle
        } else if tongues <= 1 && height < OPEN_HEIGHT {
            Self::Lantern
        } else if radius <= TORCH_SPREAD {
            Self::Torch
        } else {
            Self::Brazier
        }
    }
}

/// A candle's flame is shorter than this, metres (`kept_kit.candle_flame`: 2.5 cm; a lantern's 5 cm).
const CANDLE_HEIGHT: f32 = 0.035;
/// An open fire this narrow is a torch's (the city's tar torches spread 7-10 cm; the braziers' pans 13-20 cm).
const TORCH_SPREAD: f32 = 0.12;

/// Tongues at least this tall are a fire's (`brazier_fire`: 0.12 to 0.3 m); a lantern's is 5 cm, a candle's 2.5.
const OPEN_HEIGHT: f32 = 0.09;
/// A fire's tongues stand within this of the first, metres (a brazier's pan is up to 0.4 m across).
const FIRE_SPREAD: f32 = 0.45;

/// One flame, or one fire of several tongues, in its entity's own space (the mesh's).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Flame {
    /// Where it stands: the wick, or the middle of a fire's bed.
    pub base: glam::Vec3,
    /// Its tallest tongue, metres.
    pub height: f32,
    /// A tongue's radius, or how far a fire's tongues spread from its middle, metres.
    pub radius: f32,
    /// How many tongues the mesh had there.
    pub tongues: u32,
    /// What burns there: named by the material, or inferred from the size.
    pub kind: FlameKind,
}

impl Flame {
    /// A fire (a brazier, a torch, a tar basket) rather than a lamp's or a candle's flame: it smokes and sparks.
    #[must_use]
    pub fn open(&self) -> bool {
        self.kind.open()
    }
}

/// An entity whose mesh was a made flame: it is not drawn, and the shell burns these in its place.
#[derive(Debug, Clone)]
pub struct FlameSource {
    /// Its flames, in its own space.
    pub flames: Arc<[Flame]>,
}

impl somnium_ecs::Component for FlameSource {}

/// Whether the rule is on: `SOMNIUM_FIRE=0` keeps the made flames as meshes (a launch's before and after of it).
#[must_use]
pub fn rule_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SOMNIUM_FIRE").as_deref() != Ok("0"))
}

/// Every node of `scene` drawn with [`FLAME_MATERIAL`], by node index, with its flames.
#[must_use]
pub fn scene_flames(scene: &somnium_asset::LoadedScene) -> HashMap<u32, Arc<[Flame]>> {
    let mut out = HashMap::new();
    for (index, node) in scene.nodes.iter().enumerate() {
        let material = node.material_index.and_then(|m| scene.materials.get(m)).map(|m| m.name.as_str());
        let Some(material) = material.filter(|name| is_flame_material(name)) else { continue };
        let Some(mesh) = node.mesh_index.and_then(|m| scene.meshes.get(m)) else {
            continue;
        };
        let positions: Vec<glam::Vec3> = mesh.vertices.iter().map(|v| glam::Vec3::from(v.position)).collect();
        out.insert(index as u32, flames_of_kind(&positions, &mesh.indices, FlameKind::from_material(material)).into());
    }
    out
}

/// The flames in a made-flame mesh: each teardrop (the triangles that share corners) is a tongue; a lantern's or a
/// candle's tongue is a flame of its own, and tall tongues that stand together are one fire.
#[must_use]
pub fn flames_of(positions: &[glam::Vec3], indices: &[u32]) -> Vec<Flame> {
    flames_of_kind(positions, indices, None)
}

/// [`flames_of`] for a material that names its kind: an open kind joins every tongue within its spread into one fire
/// whatever their height (a forge's flames are a few centimetres), and every flame found takes the kind.
#[must_use]
pub fn flames_of_kind(positions: &[glam::Vec3], indices: &[u32], named: Option<FlameKind>) -> Vec<Flame> {
    // The kit writes every face with its own corners, so the teardrops are found by where corners coincide (to a
    // millimetre), not by shared indices.
    let key = |p: glam::Vec3| ((p.x * 1000.0).round() as i64, (p.y * 1000.0).round() as i64, (p.z * 1000.0).round() as i64);
    let mut slot: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let mut parent: Vec<usize> = Vec::new();
    let ids: Vec<usize> = positions
        .iter()
        .map(|&p| {
            *slot.entry(key(p)).or_insert_with(|| {
                parent.push(parent.len());
                parent.len() - 1
            })
        })
        .collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| ids.get(i as usize).copied());
        let (Some(a), Some(b), Some(c)) = (a, b, c) else { continue };
        let (ra, rb, rc) = (root(&mut parent, a), root(&mut parent, b), root(&mut parent, c));
        parent[rb] = ra;
        parent[rc] = ra;
    }
    // Each teardrop's corners: its axis is their middle (a lathed body is symmetric about it), its base the lowest.
    let mut groups: HashMap<usize, Vec<glam::Vec3>> = HashMap::new();
    let used: std::collections::HashSet<usize> = indices.iter().filter_map(|&i| ids.get(i as usize).copied()).collect();
    for (i, &p) in positions.iter().enumerate() {
        if used.contains(&ids[i]) {
            groups.entry(root(&mut parent, ids[i])).or_default().push(p);
        }
    }
    let mut tongues: Vec<Flame> = groups
        .into_values()
        .map(|points| {
            let n = points.len() as f32;
            let mid = points.iter().copied().sum::<glam::Vec3>() / n;
            let low = points.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
            let high = points.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max);
            let radius = points.iter().map(|p| glam::Vec2::new(p.x - mid.x, p.z - mid.z).length()).fold(0.0, f32::max);
            Flame { base: glam::Vec3::new(mid.x, low, mid.z), height: high - low, radius, tongues: 1, kind: FlameKind::Lantern }
        })
        .collect();
    // In a stable order, so a fire is built the same way every run.
    tongues.sort_by(|a, b| (a.base.x, a.base.z, a.base.y).partial_cmp(&(b.base.x, b.base.z, b.base.y)).unwrap_or(std::cmp::Ordering::Equal));
    let mut flames: Vec<Flame> = Vec::new();
    let mut fires: Vec<Vec<Flame>> = Vec::new();
    let open_named = named.filter(|kind| kind.open());
    let spread = open_named.map_or(FIRE_SPREAD, FlameKind::spread);
    // Tall tongues found the fires first; a short one then joins a fire within its spread (a brazier's tongues
    // shorten toward the pan's edge), and only a lone short tongue is a lamp's or a candle's flame.
    let (tall, short): (Vec<Flame>, Vec<Flame>) =
        tongues.into_iter().partition(|t| t.height >= OPEN_HEIGHT || open_named.is_some());
    let mut lone = Vec::new();
    for tongue in tall {
        let near = fires.iter_mut().find(|fire| {
            let first = fire[0];
            glam::Vec2::new(tongue.base.x - first.base.x, tongue.base.z - first.base.z).length() <= spread
                && (tongue.base.y - first.base.y).abs() <= 0.1_f32.max(spread * 0.25)
        });
        match near {
            Some(fire) => fire.push(tongue),
            None => fires.push(vec![tongue]),
        }
    }
    for tongue in short {
        let near = fires.iter_mut().find(|fire| {
            fire.iter().any(|t| {
                glam::Vec2::new(tongue.base.x - t.base.x, tongue.base.z - t.base.z).length() <= spread
                    && (tongue.base.y - t.base.y).abs() <= 0.1
            })
        });
        match near {
            Some(fire) => fire.push(tongue),
            None => lone.push(tongue),
        }
    }
    for tongue in lone {
        let kind = named.unwrap_or_else(|| FlameKind::infer(tongue.height, tongue.radius, 1));
        flames.push(Flame { kind, ..tongue });
    }
    for fire in fires {
        let n = fire.len() as f32;
        let mid = fire.iter().map(|t| t.base).sum::<glam::Vec3>() / n;
        let low = fire.iter().map(|t| t.base.y).fold(f32::INFINITY, f32::min);
        let height = fire.iter().map(|t| t.height).fold(0.0, f32::max);
        let radius = fire
            .iter()
            .map(|t| glam::Vec2::new(t.base.x - mid.x, t.base.z - mid.z).length() + t.radius)
            .fold(0.0, f32::max);
        let tongues = fire.len() as u32;
        // A torch or a brazier inferred from the size stays open even with one tongue: it was taller than a lamp's.
        let kind = open_named.unwrap_or_else(|| match FlameKind::infer(height, radius, tongues.max(2)) {
            FlameKind::Candle | FlameKind::Lantern => FlameKind::Torch,
            kind => kind,
        });
        flames.push(Flame { base: glam::Vec3::new(mid.x, low, mid.z), height, radius, tongues, kind });
    }
    flames
}

/// A kind's simulated flipbook (`assets/fire/<kind>.fire.json` beside its `.png` atlas), as `tools/fire/fire_book.py`
/// writes it. The atlas holds, per texel: R the luminance on a log scale (16 stops under `peak_cd_m2`), G the colour
/// temperature (800-2,200 K), and B and A the motion to the next frame (a candle's or a lantern's book: B its blue
/// foot's luminance, log). Frames run left to right, top to bottom; a second band of rows is the fire seen 40 degrees
/// from above.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct FlameBook {
    /// The atlas, relative to the project.
    pub texture: String,
    pub frames: u32,
    pub columns: u32,
    pub rows: u32,
    pub fps: f32,
    /// The views, level first; a second one is the 40-degree band, starting at its `row0`.
    pub views: Vec<BookView>,
    /// Luminance where R is 1, cd/m2.
    pub peak_cd_m2: f32,
    /// The card's width and height at the simulation's own size, metres.
    pub card_m: [f32; 2],
    /// Where the fire's bed (a candle's wick) sits on the card, and how wide it is, at the simulation's size.
    pub bed: BookBed,
    /// How high the flame reaches over its bed, metres, at the simulation's size.
    pub flame_height_m: f32,
    /// The card's luminous intensity face on, candela, at the simulation's size (its lumens are 4 pi times it).
    pub intensity_cd: f32,
    /// Each frame's emitted power over the book's mean: the light's flicker.
    pub power: Vec<f32>,
    /// Each frame's centre of light on the card, metres across and up from the bed, at the simulation's size.
    pub light_centre_m: Vec<[f32; 2]>,
}

/// One view of a book.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct BookView {
    pub elevation_deg: f32,
    pub row0: u32,
}

/// Where a book's bed is on its card: `u` across, `v` down from the top (0..1), and its width in metres.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
pub struct BookBed {
    pub u: f32,
    pub v: f32,
    pub width_m: f32,
}

impl FlameBook {
    /// The kind's book under the project, if it has been made.
    #[must_use]
    pub fn load(project: &std::path::Path, kind: FlameKind) -> Option<Self> {
        let path = project.join("assets").join("fire").join(format!("{}.fire.json", kind.name()));
        let book: Self = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
        (book.frames > 0 && book.power.len() == book.frames as usize && book.light_centre_m.len() == book.frames as usize).then_some(book)
    }

    /// The frame showing `clock` seconds in, and how far it is toward the next (the book loops).
    #[must_use]
    pub fn frame(&self, clock: f32) -> (u32, f32) {
        let f = (clock * self.fps).max(0.0);
        ((f.floor() as u64 % u64::from(self.frames.max(1))) as u32, f.fract())
    }

    /// The light's output over its mean, `clock` seconds in: the flame's own flicker.
    #[must_use]
    pub fn power_at(&self, clock: f32) -> f32 {
        let (a, t) = self.frame(clock);
        let n = self.power.len();
        if n == 0 {
            return 1.0;
        }
        let a = a as usize % n;
        self.power[a] * (1.0 - t) + self.power[(a + 1) % n] * t
    }

    /// The light's centre on the card, metres across and up from the bed, `clock` seconds in, at the book's size.
    #[must_use]
    pub fn centre_at(&self, clock: f32) -> glam::Vec2 {
        let (a, t) = self.frame(clock);
        let n = self.light_centre_m.len();
        if n == 0 {
            return glam::Vec2::ZERO;
        }
        let (p, q) = (self.light_centre_m[a as usize % n], self.light_centre_m[(a as usize + 1) % n]);
        glam::Vec2::from(p).lerp(glam::Vec2::from(q), t)
    }

    /// The row the 40-degree band starts at, or 0 when the book has only its level view.
    #[must_use]
    pub fn down_row0(&self) -> u32 {
        self.views.iter().find(|v| v.elevation_deg > 1.0).map_or(0, |v| v.row0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A teardrop as `kept_kit.flame()` lathes it, every face with its own corners: positions and indices.
    fn teardrop(base: glam::Vec3, h: f32, r: f32, into: &mut (Vec<glam::Vec3>, Vec<u32>)) {
        let profile = [(0.0, 0.0), (r * 0.55, h * 0.12), (r, h * 0.38), (r * 0.6, h * 0.72), (0.0, h)];
        let seg = 6;
        for pair in profile.windows(2) {
            let ((r0, y0), (r1, y1)) = (pair[0], pair[1]);
            for k in 0..seg {
                let (a0, a1) = (std::f32::consts::TAU * k as f32 / seg as f32, std::f32::consts::TAU * (k + 1) as f32 / seg as f32);
                let at = |r: f32, a: f32, y: f32| base + glam::Vec3::new(r * a.cos(), y, r * a.sin());
                let quad = [at(r0, a0, y0), at(r0, a1, y0), at(r1, a1, y1), at(r1, a0, y1)];
                for tri in [[0, 1, 2], [0, 2, 3]] {
                    for corner in tri {
                        into.1.push(into.0.len() as u32);
                        into.0.push(quad[corner]);
                    }
                }
            }
        }
    }

    #[test]
    fn lanterns_and_candles_are_flames_and_a_braziers_tongues_one_fire() {
        let mut mesh = (Vec::new(), Vec::new());
        // Two lanterns on one post, 0.5 m apart; a candle on a sill; a brazier's five tongues in a 0.13 m pan.
        teardrop(glam::Vec3::new(0.0, 2.2, 0.0), 0.05, 0.012, &mut mesh);
        teardrop(glam::Vec3::new(0.5, 2.2, 0.0), 0.05, 0.012, &mut mesh);
        teardrop(glam::Vec3::new(3.0, 1.0, 1.0), 0.025, 0.006, &mut mesh);
        for (dx, dz, h) in [(0.0, 0.0, 0.28), (0.07, 0.02, 0.2), (-0.05, 0.06, 0.18), (0.02, -0.08, 0.15), (-0.08, -0.03, 0.13)] {
            teardrop(glam::Vec3::new(10.0 + dx, 0.9, 4.0 + dz), h, 0.04, &mut mesh);
        }
        // and a short tongue at its rim, under a lamp's 9 cm: part of the fire, not a lamp
        teardrop(glam::Vec3::new(10.11, 0.9, 4.0), 0.06, 0.03, &mut mesh);
        let flames = flames_of(&mesh.0, &mesh.1);
        assert_eq!(flames.len(), 4, "{flames:?}");
        // The kinds by size: a lantern's 5 cm, a candle's 2.5 cm, and the brazier's pan of 13 cm a brazier.
        let kind_at = |x: f32| flames.iter().find(|f| (f.base.x - x).abs() < 0.2).map(|f| f.kind);
        assert_eq!((kind_at(0.0), kind_at(3.0), kind_at(10.0)), (Some(FlameKind::Lantern), Some(FlameKind::Candle), Some(FlameKind::Brazier)));
        let small: Vec<&Flame> = flames.iter().filter(|f| !f.open()).collect();
        assert_eq!(small.len(), 3, "two lanterns and a candle");
        for f in &small {
            assert_eq!(f.tongues, 1);
            assert!(f.height <= 0.05 + 1e-4 && f.radius <= 0.0121, "{f:?}");
        }
        let lantern = small.iter().find(|f| f.base.x.abs() < 1e-3).expect("the first lantern");
        assert!(lantern.base.distance(glam::Vec3::new(0.0, 2.2, 0.0)) < 1e-3, "at its wick: {lantern:?}");
        let fire = flames.iter().find(|f| f.open()).expect("the brazier");
        assert_eq!(fire.tongues, 6, "the rim's short tongue joins the fire");
        assert!((fire.base.y - 0.9).abs() < 1e-3 && (fire.height - 0.28).abs() < 1e-3, "{fire:?}");
        assert!(fire.base.distance(glam::Vec3::new(10.0, 0.9, 4.0)) < 0.05 && fire.radius < 0.16, "{fire:?}");
        // No mesh, no flames.
        assert!(flames_of(&[], &[]).is_empty());
    }

    #[test]
    fn a_material_names_its_kind_and_a_named_open_kind_joins_short_tongues() {
        assert!(is_flame_material("kp_flame") && is_flame_material("kp_flame_forge") && !is_flame_material("kp_flame_x"));
        assert_eq!(FlameKind::from_material("kp_flame_bonfire"), Some(FlameKind::Bonfire));
        assert_eq!(FlameKind::from_material("kp_flame"), None);
        // A forge's flames: five licks of 4-7 cm over a 0.4 m bed. Unnamed they are five lamps; named, one forge.
        let mut mesh = (Vec::new(), Vec::new());
        for (dx, dz, h) in [(0.0, 0.0, 0.07), (0.15, 0.05, 0.05), (-0.12, 0.1, 0.06), (0.05, -0.16, 0.04), (-0.1, -0.08, 0.05)] {
            teardrop(glam::Vec3::new(dx, 0.8, dz), h, 0.02, &mut mesh);
        }
        assert_eq!(flames_of(&mesh.0, &mesh.1).len(), 5);
        let forge = flames_of_kind(&mesh.0, &mesh.1, Some(FlameKind::Forge));
        assert_eq!(forge.len(), 1, "{forge:?}");
        assert_eq!((forge[0].kind, forge[0].tongues), (FlameKind::Forge, 5));
        // A torch inferred: one tall tongue of 0.3 m on a 5 cm head.
        let mut torch = (Vec::new(), Vec::new());
        teardrop(glam::Vec3::new(0.0, 1.6, 0.0), 0.3, 0.05, &mut torch);
        assert_eq!(flames_of(&torch.0, &torch.1)[0].kind, FlameKind::Torch);
    }

    #[test]
    fn a_book_loops_its_frames_and_its_light_follows_them() {
        let book = FlameBook {
            texture: "assets/fire/brazier.png".into(),
            frames: 4,
            columns: 2,
            rows: 2,
            fps: 30.0,
            views: vec![BookView { elevation_deg: 0.0, row0: 0 }, BookView { elevation_deg: 40.0, row0: 2 }],
            peak_cd_m2: 1000.0,
            card_m: [0.6, 0.9],
            bed: BookBed { u: 0.5, v: 0.9, width_m: 0.32 },
            flame_height_m: 0.5,
            intensity_cd: 30.0,
            power: vec![1.0, 1.2, 0.8, 1.0],
            light_centre_m: vec![[0.0, 0.2], [0.02, 0.25], [-0.02, 0.2], [0.0, 0.22]],
        };
        assert_eq!(book.frame(0.0), (0, 0.0));
        let (a, t) = book.frame(1.5 / 30.0);
        assert_eq!(a, 1);
        assert!((t - 0.5).abs() < 1e-4);
        assert_eq!(book.frame(4.0 / 30.0 + 1e-4).0, 0, "it loops");
        assert!((book.power_at(1.5 / 30.0) - 1.0).abs() < 1e-4, "halfway between 1.2 and 0.8");
        assert!((book.power_at(3.5 / 30.0) - 1.0).abs() < 1e-4, "the last frame runs into the first");
        assert!((book.centre_at(0.5 / 30.0).y - 0.225).abs() < 1e-4);
        assert_eq!(book.down_row0(), 2);
    }
}
