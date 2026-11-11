//! Storm bodies and lightning for the cloud march (Sunscar weather, 11 Nov 2026).
//!
//! The cloud layer is one tiling slab: no storm in it has a place, a shape or parts. A [`StormVolume`] is a body with
//! all three: an analytic shape in world metres with the cloud noise as its detail, evaluated in `clouds.wgsl`'s
//! `cloud_density`, so the screen march, the ground's cloud shadow and the sky light's capture all see it. A
//! [`StormFlash`] is a light inside the march: lightning lighting the cloud from within.
//!
//! A game hands both to [`crate::pass::clouds::CloudPass::storm`] every frame. Empty (the default) the march does one
//! compare and nothing else, which is every scene that does not ask for a storm.

/// Bodies the march takes in one frame.
pub const STORM_VOLUMES: usize = 16;
/// Lights inside the march in one frame.
pub const STORM_FLASHES: usize = 8;

const CELL: f32 = 1.0;
const WALL: f32 = 2.0;
const SHELF: f32 = 3.0;
const FUNNEL: f32 = 4.0;
const BOWL: f32 = 5.0;
const ANVIL: f32 = 6.0;
const SHAFT: f32 = 7.0;
const RING: f32 = 8.0;
const HOLE: f32 = 9.0;
const DECK: f32 = 10.0;

/// What a cloud droplet scatters of the light it meets (`CLOUD_ALBEDO` in the shader).
pub const WATER: [f32; 3] = [0.95, 0.95, 0.95];

/// One body, as `clouds.wgsl`'s `StormVolume` reads it. Build one with the constructors below.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StormVolume {
    /// xyz: the body's foot (y its bottom), w: kind.
    pub pos: [f32; 4],
    /// x: radius, y: height, z: density scale, w: metres per repeat of its noise.
    pub size: [f32; 4],
    /// Per kind (the constructors say).
    pub a: [f32; 4],
    pub b: [f32; 4],
    /// rgb: the share of light it scatters, by colour (water white, dust the soil's colour); w: turn, radians.
    pub albedo: [f32; 4],
    /// The cylinder that holds it: x, z of its axis, its radius; w: its finest feature in metres (the march never
    /// steps further than this inside the cylinder).
    pub bound: [f32; 4],
    /// The cylinder's bottom and top (world y).
    pub bound_y: [f32; 4],
}

impl StormVolume {
    fn new(kind: f32, foot: glam::Vec3, radius: f32, height: f32, density: f32, noise: f32) -> Self {
        Self {
            pos: [foot.x, foot.y, foot.z, kind],
            size: [radius.max(1.0), height.max(1.0), density.max(0.0), noise.max(1.0)],
            a: [0.0; 4],
            b: [0.0; 4],
            albedo: [WATER[0], WATER[1], WATER[2], 0.0],
            bound: [foot.x, foot.z, radius.max(1.0), (radius * 0.5).max(4.0)],
            bound_y: [foot.y, foot.y + height.max(1.0), 0.0, 0.0],
        }
    }

    /// What it scatters, by colour: dust and debris take the ground's colour.
    #[must_use]
    pub fn coloured(mut self, albedo: [f32; 3]) -> Self {
        self.albedo[..3].copy_from_slice(&albedo);
        self
    }

    /// Its turn about its own axis, radians (a game adds the rotation's rate times the frame's seconds).
    #[must_use]
    pub fn turned(mut self, radians: f32) -> Self {
        self.albedo[3] = radians;
        self
    }

    /// A cumulonimbus's tower: `radius` wide from its flat base at `base` up `height` metres, leaning `lean` metres
    /// `downwind` (the wind aloft, world XZ) at its top, where it flares into its anvil ([`Self::anvil`], a body of
    /// its own). `striation` 0..1 bands it and widens its base to a bell (a rotating updraft); `overshoot` metres is
    /// the dome above the anvil.
    #[must_use]
    pub fn cell(base: glam::Vec3, radius: f32, height: f32, downwind: glam::Vec2, lean: f32, striation: f32, overshoot: f32) -> Self {
        let dir = downwind.normalize_or(glam::Vec2::X);
        let mut v = Self::new(CELL, base, radius, height, 1.0, (radius * 2.2).clamp(600.0, 6000.0));
        v.a = [dir.x, dir.y, 0.0, 0.0];
        // b.z: it flares into an anvil at its top (1, the default); a game sets 0 for a tower with none.
        v.b = [lean, striation.clamp(0.0, 1.0), 1.0, overshoot.max(0.0)];
        // Its widest (the flare under the anvil, 1.63 radii with its bands) pushed out by all the noise may (1.35).
        let centre = glam::Vec2::new(base.x, base.z) + dir * (lean * 0.5);
        v.bound = [centre.x, centre.y, radius * 2.2 + lean.abs() * 0.5, (radius * 0.08).max(30.0)];
        // The noise lifts its top a little; the dome stands over that.
        v.bound_y = [base.y, base.y + height * 1.04 + overshoot.max(0.0), 0.0, 0.0];
        v
    }

    /// A cumulonimbus's anvil: ice spread `reach` metres `downwind` from the top of a tower (give it the tower's
    /// `base`, `radius` and `height`, so the two meet), a little way upwind and to the sides; its underside starts
    /// `anvil_base` (0..1) of the way up the tower and rises to its rim. `mammatus` 0..1 hangs lobes under it.
    ///
    /// A body of its own so that its cylinder is only as thick as it is: carried by the tower, it put ten kilometres
    /// of empty air under it into every ray's march (first stills: 12-23 ms a frame).
    #[must_use]
    pub fn anvil(base: glam::Vec3, radius: f32, height: f32, downwind: glam::Vec2, reach: f32, anvil_base: f32, mammatus: f32) -> Self {
        let dir = downwind.normalize_or(glam::Vec2::X);
        let under = anvil_base.clamp(0.3, 0.95);
        let mut v = Self::new(ANVIL, base, radius, height, 1.0, (radius * 2.2).clamp(600.0, 6000.0));
        v.a = [dir.x, dir.y, reach.max(0.0), under];
        v.b = [0.0, 0.0, mammatus.clamp(0.0, 1.0), 0.0];
        let centre = glam::Vec2::new(base.x, base.z) + dir * (reach * 0.42);
        v.bound = [centre.x, centre.y, reach * 0.62 + radius * 2.0, (radius * 0.15).max(60.0)];
        // From its lowest mammatus to its top.
        v.bound_y = [base.y + height * (under - 0.08), base.y + height, 0.0, 0.0];
        v
    }

    /// A storm's base: a flat-bottomed deck `thickness` metres deep whose underside is at `centre`'s height, `radius`
    /// wide across the wind and reaching `downwind` metres further along it (world XZ, the way its anvil blows),
    /// thick at the storm and thinning to a rim that breaks into scud. From under it, it is the sky: the dark, level
    /// base a wall cloud lowers from and rain falls out of. A tower alone in a clear sky is the view from twenty
    /// kilometres; this is the view from under it.
    #[must_use]
    pub fn deck(centre: glam::Vec3, radius: f32, thickness: f32, downwind: glam::Vec2) -> Self {
        let half = downwind.length() * 0.5;
        let dir = downwind.normalize_or(glam::Vec2::X);
        let mut v = Self::new(DECK, centre, radius, thickness, 1.0, 2600.0);
        v.a = [dir.x, dir.y, half, 0.0];
        let mid = glam::Vec2::new(centre.x, centre.z) + dir * half;
        v.bound = [mid.x, mid.y, radius + half, (thickness * 0.1).max(40.0)];
        // Its underside rolls a little.
        v.bound_y = [centre.y - 30.0, centre.y + thickness.max(1.0), 0.0, 0.0];
        v
    }

    /// A wall cloud: a block `radius` wide lowered `height` metres under a storm's base (its foot is its own bottom),
    /// with a tail cloud `tail` metres long toward `tail_toward` (world XZ; the rain's side). `ragged` 0..1 frays it.
    #[must_use]
    pub fn wall(bottom: glam::Vec3, radius: f32, height: f32, tail_toward: glam::Vec2, tail: f32, ragged: f32) -> Self {
        let dir = tail_toward.normalize_or(glam::Vec2::X);
        let mut v = Self::new(WALL, bottom, radius, height, 1.2, (radius * 1.6).clamp(250.0, 2500.0));
        v.a = [dir.x, dir.y, tail.max(0.0), ragged.clamp(0.0, 1.0)];
        // Its flare into the base (1.6 radii) pushed out by all the noise may (1.35).
        v.bound[2] = radius * 2.2 + tail;
        v.bound[3] = (radius * 0.1).max(15.0);
        // Its foot hangs in rags (the noise moves it a third of its height either way); its top is inside the base.
        v.bound_y = [bottom.y - height * 0.65, bottom.y + height * 1.85, 0.0, 0.0];
        v
    }

    /// A shelf cloud on a gust front: an arc of `radius` round `centre` (y its nose's bottom), facing `bearing`
    /// (radians from +Z toward +X) and reaching `half_arc` radians each way, `nose` metres thick at its leading edge
    /// and rising over `depth` metres behind it to its parent's base `height` metres up.
    #[must_use]
    pub fn shelf(centre: glam::Vec3, radius: f32, height: f32, bearing: f32, half_arc: f32, depth: f32, nose: f32) -> Self {
        let mut v = Self::new(SHELF, centre, radius, height, 1.1, (nose * 3.0).clamp(300.0, 2500.0));
        let arc = half_arc.clamp(0.05, std::f32::consts::PI);
        v.a = [bearing, arc, depth.max(1.0), nose.max(1.0)];
        v.bound[2] = radius + nose;
        v.bound[3] = (nose * 0.12).max(12.0);
        if arc < std::f32::consts::FRAC_PI_2 {
            // The cylinder round the arc alone, not the disc it is cut from: a narrow shelf seven kilometres out put
            // fourteen kilometres of empty air at its own fine step into every ray (and the deck behind it went
            // missing when the ray's iterations ran out). Its middle is half way between the arc's far point and its
            // inner corners' chord; its radius reaches whichever corner is furthest.
            let (outer, inner) = (radius + nose.max(1.0), (radius - depth.max(1.0)).max(0.0));
            let mid = (outer + inner * arc.cos()) * 0.5;
            let corner = |r: f32| ((r * arc.sin()).powi(2) + (r * arc.cos() - mid).powi(2)).sqrt();
            v.bound[0] = centre.x + bearing.sin() * mid;
            v.bound[1] = centre.z + bearing.cos() * mid;
            v.bound[2] = (outer - mid).max(corner(outer)).max(corner(inner));
        }
        v
    }

    /// A funnel from the ground at `foot` up `height` metres to the cloud: `radius` wide at the ground and
    /// `top_radius` where it meets the base, its top `lean` metres off its foot (world XZ), bowed `bend` metres
    /// between. `power` shapes it: 1 a cone, 3 a thin stem flaring late. Hollow: dense at its wall.
    #[must_use]
    pub fn funnel(foot: glam::Vec3, radius: f32, height: f32, top_radius: f32, lean: glam::Vec2, bend: f32, power: f32) -> Self {
        let wide = radius.max(top_radius);
        let mut v = Self::new(FUNNEL, foot, radius, height, 2.5, (wide * 2.5).clamp(60.0, 1500.0));
        v.a = [lean.x, lean.y, top_radius.max(1.0), power.clamp(0.3, 6.0)];
        v.b = [bend, 0.0, 0.0, 0.0];
        let centre = glam::Vec2::new(foot.x, foot.z) + lean * 0.5;
        v.bound = [centre.x, centre.y, wide * 1.15 + lean.length() * 0.5 + bend.abs(), (radius.min(top_radius) * 0.3).max(3.0)];
        v
    }

    /// The debris cloud and dust skirt at a vortex's foot: a turning bowl of lofted ground `radius` wide and `height`
    /// tall, `flare` times wider again along the ground, clear inside `inner` metres (where the funnel stands).
    #[must_use]
    pub fn bowl(foot: glam::Vec3, radius: f32, height: f32, inner: f32, flare: f32, dust: [f32; 3]) -> Self {
        let mut v = Self::new(BOWL, foot, radius, height, 0.55, (radius * 1.4).clamp(40.0, 1200.0)).coloured(dust);
        v.a = [inner.max(0.0), flare.max(0.0), 0.0, 0.0];
        v.bound[2] = radius * (1.0 + flare.max(0.0)) * 1.05;
        v.bound[3] = (radius * 0.12).max(4.0);
        v
    }

    /// A shaft of rain or hail from the cloud's base `height` metres above `foot` down to the ground, `radius` wide,
    /// its foot carried `drift` metres (world XZ) from under its top by the wind. `virga` 0..1 is the share of its
    /// height above the ground it never reaches, evaporated on the way (0.3 ends it 70% of the way down).
    /// `opacity` is extinction relative to cloud (rain about 0.03, a hail core 0.1).
    #[must_use]
    pub fn shaft(foot: glam::Vec3, radius: f32, height: f32, drift: glam::Vec2, virga: f32, opacity: f32, tint: [f32; 3]) -> Self {
        let mut v = Self::new(SHAFT, foot, radius, height, opacity, (radius * 0.9).clamp(150.0, 3000.0)).coloured(tint);
        v.a = [drift.x, drift.y, virga.clamp(0.0, 0.95), 1.0];
        let centre = glam::Vec2::new(foot.x, foot.z) - drift * 0.5;
        v.bound = [centre.x, centre.y, radius * 1.3 + drift.length() * 0.5, (radius * 0.2).max(30.0)];
        v
    }

    /// A ring of storm round `base` (y the cloud's underside): a wall `half_wall` metres thick each way about
    /// `radius`, `height` tall, leaning `lean` metres outward by its top (from inside, a stadium of cloud round a
    /// clear eye), with an anvil reaching `anvil` metres outward from its top whose underside starts `anvil_base`
    /// (0..1) of the way up, and rain under the wall down `rain_depth` metres at `rain` of the cloud's density.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn ring(base: glam::Vec3, radius: f32, height: f32, half_wall: f32, lean: f32, anvil: f32, anvil_base: f32, rain: f32, rain_depth: f32) -> Self {
        let mut v = Self::new(RING, base, radius, height, 1.0, (half_wall * 4.0).clamp(600.0, 6000.0));
        v.a = [half_wall.max(1.0), lean, anvil.max(0.0), anvil_base.clamp(0.3, 0.95)];
        v.b = [rain.clamp(0.0, 1.0), rain_depth.max(0.0), 0.0, 0.0];
        v.bound[2] = radius + lean.abs() + anvil.max(0.0) + half_wall * 1.6;
        v.bound[3] = (half_wall * 0.2).max(30.0);
        v.bound_y = [base.y - rain_depth.max(0.0), base.y + height, 0.0, 0.0];
        v
    }

    /// A hole in the cloud layer `radius` wide round `centre` (its height is not used), soft over `soft` metres: a
    /// storm's clear slot, a fallstreak hole. It takes cloud away from the slab; it adds none.
    #[must_use]
    pub fn hole(centre: glam::Vec3, radius: f32, soft: f32) -> Self {
        let mut v = Self::new(HOLE, centre, radius, 1.0, 0.0, 1000.0);
        v.a = [soft.max(1.0), 0.0, 0.0, 0.0];
        v.bound[2] = radius + soft.max(1.0);
        v.bound_y = [-1.0e6, 1.0e6, 0.0, 0.0];
        v
    }
}

/// A light inside the cloud march: a lightning channel's light on the cloud round it.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StormFlash {
    /// xyz: where; w: the source's radius in metres (a channel is a line, not a point: this softens the fall-off).
    pub pos: [f32; 4],
    /// rgb: luminous intensity, candela (a near flash averages 5e9 to 5e10 over a frame).
    pub color: [f32; 4],
}

impl StormFlash {
    #[must_use]
    pub fn new(at: glam::Vec3, radius: f32, candela: glam::Vec3) -> Self {
        Self { pos: [at.x, at.y, at.z, radius.max(1.0)], color: [candela.x, candela.y, candela.z, 0.0] }
    }
}

/// What the march is given this frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Storm {
    pub volumes: Vec<StormVolume>,
    pub flashes: Vec<StormFlash>,
}

/// `clouds.wgsl`'s `Storm` uniform.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StormGpu {
    counts: [u32; 4],
    volumes: [StormVolume; STORM_VOLUMES],
    flashes: [StormFlash; STORM_FLASHES],
}

impl Storm {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.volumes.is_empty() && self.flashes.is_empty()
    }

    /// Packed for the shader: the first [`STORM_VOLUMES`] bodies and [`STORM_FLASHES`] lights (a game sorts by what
    /// matters to the eye before it hands over more).
    pub(crate) fn pack(&self) -> StormGpu {
        let mut gpu: StormGpu = bytemuck::Zeroable::zeroed();
        let (v, f) = (self.volumes.len().min(STORM_VOLUMES), self.flashes.len().min(STORM_FLASHES));
        gpu.volumes[..v].copy_from_slice(&self.volumes[..v]);
        gpu.flashes[..f].copy_from_slice(&self.flashes[..f]);
        #[allow(clippy::cast_possible_truncation)]
        {
            gpu.counts = [v as u32, f as u32, 0, 0];
        }
        gpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_uniform_matches_the_shaders_layout() {
        // WGSL: 7 vec4 a body, 2 a flash, one vec4<u32> of counts.
        assert_eq!(std::mem::size_of::<StormVolume>(), 112);
        assert_eq!(std::mem::size_of::<StormFlash>(), 32);
        assert_eq!(std::mem::size_of::<StormGpu>(), 16 + 112 * STORM_VOLUMES + 32 * STORM_FLASHES);
    }

    #[test]
    fn a_bodys_cylinder_holds_it() {
        let dir = glam::Vec2::new(0.6, 0.8);
        let at = glam::Vec2::new(1000.0, -500.0);
        let cell = StormVolume::cell(glam::Vec3::new(1000.0, 900.0, -500.0), 800.0, 6000.0, dir, 600.0, 0.5, 400.0);
        // The tower's widest (under its anvil, leaning downwind) and its upwind edge are inside its cylinder, and the
        // cylinder is the tower's alone: nowhere near an anvil's reach.
        let centre = glam::Vec2::new(cell.bound[0], cell.bound[1]);
        assert!((at + dir * (600.0 + 800.0 * 1.63)).distance(centre) <= cell.bound[2], "tower's top outside");
        assert!((at - dir * 800.0 * 1.4).distance(centre) <= cell.bound[2], "tower's back outside");
        assert!(cell.bound[2] < 3000.0 && cell.bound_y[1] >= 900.0 + 6000.0 + 400.0);
        // The anvil's far tip is inside its own, which starts at its underside: the air below it is nobody's.
        let anvil = StormVolume::anvil(glam::Vec3::new(1000.0, 900.0, -500.0), 800.0, 6000.0, dir, 9000.0, 0.72, 0.5);
        let centre = glam::Vec2::new(anvil.bound[0], anvil.bound[1]);
        assert!((at + dir * 9000.0 * 1.04).distance(centre) <= anvil.bound[2], "anvil tip outside");
        assert!(anvil.bound_y[0] > 900.0 + 6000.0 * 0.6 && anvil.bound_y[1] >= 900.0 + 6000.0);

        // A deck's cylinder reaches its downwind end and stops at its own top.
        let deck = StormVolume::deck(glam::Vec3::new(1000.0, 900.0, -500.0), 7000.0, 900.0, dir * 6000.0);
        assert!((at + dir * 12_900.0).distance(glam::Vec2::new(deck.bound[0], deck.bound[1])) <= deck.bound[2]);
        assert!(deck.bound_y[0] < 900.0 && (deck.bound_y[1] - 1800.0).abs() < 1.0);

        // A narrow shelf's cylinder holds its arc (its nose's ends and its inner corners), and is nowhere near the
        // size of the disc the arc is cut from.
        let shelf = StormVolume::shelf(glam::Vec3::new(0.0, 400.0, 0.0), 7000.0, 500.0, 0.3, 0.18, 1100.0, 170.0);
        let middle = glam::Vec2::new(shelf.bound[0], shelf.bound[1]);
        for (r, a) in [(7170.0_f32, 0.3 - 0.18_f32), (7170.0, 0.3 + 0.18), (7170.0, 0.3), (5900.0, 0.3 - 0.18), (5900.0, 0.3 + 0.18)] {
            assert!((glam::Vec2::new(a.sin(), a.cos()) * r).distance(middle) <= shelf.bound[2] + 1.0, "a corner of the arc is outside");
        }
        assert!(shelf.bound[2] < 1600.0, "{}", shelf.bound[2]);

        let funnel = StormVolume::funnel(glam::Vec3::new(0.0, 300.0, 0.0), 60.0, 800.0, 220.0, glam::Vec2::new(300.0, 0.0), 40.0, 2.2);
        let top = glam::Vec2::new(300.0, 0.0);
        assert!(top.distance(glam::Vec2::new(funnel.bound[0], funnel.bound[1])) + 220.0 <= funnel.bound[2]);
        // Its finest feature is a fraction of its thinnest radius, so the march cannot step over it.
        assert!(funnel.bound[3] <= 60.0 * 0.5);

        // A ring's cylinder reaches its anvil's rim and down to the ground its rain falls on.
        let ring = StormVolume::ring(glam::Vec3::new(0.0, 700.0, 0.0), 1250.0, 4800.0, 450.0, 500.0, 3000.0, 0.78, 0.05, 650.0);
        assert!(ring.bound[2] >= 1250.0 + 500.0 + 3000.0 && ring.bound_y[0] <= 50.0 && ring.bound_y[1] >= 5500.0);
    }

    #[test]
    fn packing_keeps_only_what_the_shader_holds() {
        let one = StormVolume::hole(glam::Vec3::ZERO, 500.0, 100.0);
        let storm = Storm { volumes: vec![one; STORM_VOLUMES + 5], flashes: vec![StormFlash::new(glam::Vec3::Y, 10.0, glam::Vec3::ONE); 2] };
        let gpu = storm.pack();
        assert_eq!(gpu.counts, [STORM_VOLUMES as u32, 2, 0, 0]);
        assert!(Storm::default().is_empty() && !storm.is_empty());
    }
}
