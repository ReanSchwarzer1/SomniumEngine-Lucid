//! Real fire in place of the made flame (Sunscar, 13 Nov 2026).
//!
//! Every flame in the world was one made mesh: a lathed teardrop of the emissive material `kp_flame`
//! (`tools/kept_kit.py` `flame()`, behind its lanterns, candles, braziers and the city's tar torches). Lit and still,
//! it read as a glowing crystal, by day and by night ("pls replace these weird crystals with actual fire particle
//! effects and smokes and a light source", the developer, 12 Nov). The engine no longer draws a primitive of that
//! material: [`scene_flames`] finds the flames in it as it is restored, the entity carries them as a
//! [`FlameSource`], and the game's shell burns them (flickering tongues, embers and smoke in the particle pass, and a
//! light that wavers with them).

use std::collections::HashMap;
use std::sync::Arc;

/// The material a made flame is drawn with.
pub const FLAME_MATERIAL: &str = "kp_flame";

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
}

impl Flame {
    /// A fire (a brazier, a torch, a tar basket) rather than a lamp's or a candle's flame: it smokes and sparks.
    #[must_use]
    pub fn open(&self) -> bool {
        self.tongues > 1 || self.height >= OPEN_HEIGHT
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
        let flame = node.material_index.and_then(|m| scene.materials.get(m)).is_some_and(|m| m.name == FLAME_MATERIAL);
        let Some(mesh) = node.mesh_index.and_then(|m| scene.meshes.get(m)).filter(|_| flame) else {
            continue;
        };
        let positions: Vec<glam::Vec3> = mesh.vertices.iter().map(|v| glam::Vec3::from(v.position)).collect();
        out.insert(index as u32, flames_of(&positions, &mesh.indices).into());
    }
    out
}

/// The flames in a made-flame mesh: each teardrop (the triangles that share corners) is a tongue; a lantern's or a
/// candle's tongue is a flame of its own, and tall tongues that stand together are one fire.
#[must_use]
pub fn flames_of(positions: &[glam::Vec3], indices: &[u32]) -> Vec<Flame> {
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
            Flame { base: glam::Vec3::new(mid.x, low, mid.z), height: high - low, radius, tongues: 1 }
        })
        .collect();
    // In a stable order, so a fire is built the same way every run.
    tongues.sort_by(|a, b| (a.base.x, a.base.z, a.base.y).partial_cmp(&(b.base.x, b.base.z, b.base.y)).unwrap_or(std::cmp::Ordering::Equal));
    let mut flames: Vec<Flame> = Vec::new();
    let mut fires: Vec<Vec<Flame>> = Vec::new();
    for tongue in tongues {
        if tongue.height < OPEN_HEIGHT {
            flames.push(tongue);
            continue;
        }
        let near = fires.iter_mut().find(|fire| {
            let first = fire[0];
            glam::Vec2::new(tongue.base.x - first.base.x, tongue.base.z - first.base.z).length() <= FIRE_SPREAD
                && (tongue.base.y - first.base.y).abs() <= 0.1
        });
        match near {
            Some(fire) => fire.push(tongue),
            None => fires.push(vec![tongue]),
        }
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
        flames.push(Flame { base: glam::Vec3::new(mid.x, low, mid.z), height, radius, tongues: fire.len() as u32 });
    }
    flames
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
        let flames = flames_of(&mesh.0, &mesh.1);
        assert_eq!(flames.len(), 4, "{flames:?}");
        let small: Vec<&Flame> = flames.iter().filter(|f| !f.open()).collect();
        assert_eq!(small.len(), 3, "two lanterns and a candle");
        for f in &small {
            assert_eq!(f.tongues, 1);
            assert!(f.height <= 0.05 + 1e-4 && f.radius <= 0.0121, "{f:?}");
        }
        let lantern = small.iter().find(|f| f.base.x.abs() < 1e-3).expect("the first lantern");
        assert!(lantern.base.distance(glam::Vec3::new(0.0, 2.2, 0.0)) < 1e-3, "at its wick: {lantern:?}");
        let fire = flames.iter().find(|f| f.open()).expect("the brazier");
        assert_eq!(fire.tongues, 5);
        assert!((fire.base.y - 0.9).abs() < 1e-3 && (fire.height - 0.28).abs() < 1e-3, "{fire:?}");
        assert!(fire.base.distance(glam::Vec3::new(10.0, 0.9, 4.0)) < 0.05 && fire.radius < 0.15, "{fire:?}");
        // No mesh, no flames.
        assert!(flames_of(&[], &[]).is_empty());
    }
}
