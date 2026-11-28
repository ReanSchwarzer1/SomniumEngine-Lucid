// The planet over the horizon (somnium.SkyPlanet; Sunscar's sky, sunscar-sky 12 Nov 2026): a dead terrestrial world,
// drawn analytically on the sky behind the clouds at full resolution, as the sun's disc is. It is a sphere lit by the
// sun: its phase falls out of the light, its limb darkens, a thin air rims its sunlit edge, and its night side is
// black against the stars but for an ember glow along the fissures of a world still cooling and the faint light our
// own world throws on it.
//
// The surface: rust and ochre highlands, dark basalt plains, a great ringed impact basin, younger craters with pale
// rays, all from value noise on the sphere (no texture), turning slowly. Its brightness is physical: a Lommel-Seeliger
// and Lambert mix for rough rock, its geometric albedo, the sun's illuminance above the air.

fn planet_hash(p: vec3<f32>) -> f32 {
    let q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yzx + 33.33);
    return fract((r.x + r.y) * r.z);
}

fn planet_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(planet_hash(i), planet_hash(i + vec3(1.0, 0.0, 0.0)), u.x),
            mix(planet_hash(i + vec3(0.0, 1.0, 0.0)), planet_hash(i + vec3(1.0, 1.0, 0.0)), u.x), u.y),
        mix(mix(planet_hash(i + vec3(0.0, 0.0, 1.0)), planet_hash(i + vec3(1.0, 0.0, 1.0)), u.x),
            mix(planet_hash(i + vec3(0.0, 1.0, 1.0)), planet_hash(i + vec3(1.0, 1.0, 1.0)), u.x), u.y), u.z);
}

/// Fractal noise of `octaves` octaves, 0..1.
fn planet_fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    var total = 0.0;
    for (var i = 0; i < octaves; i = i + 1) {
        v += a * planet_noise(q);
        total += a;
        q = q * 2.03 + vec3<f32>(1.7, 9.2, 4.1);
        a *= 0.5;
    }
    return v / total;
}

/// Craters on the unit sphere: distance to the nearest of a cell's centres, as a bowl with a raised rim and rays.
/// Returns (bowl depth 0..1, rim 0..1, ray brightness 0..1).
fn planet_craters(n: vec3<f32>, cells: f32) -> vec3<f32> {
    let p = n * cells;
    let base = floor(p);
    var best = 9.0;
    var size = 0.0;
    var centre = vec3<f32>(0.0);
    for (var dx = -1; dx <= 1; dx = dx + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dz = -1; dz <= 1; dz = dz + 1) {
                let c = base + vec3<f32>(f32(dx), f32(dy), f32(dz));
                let h = planet_hash(c + 0.37);
                if h < 0.55 {
                    continue;
                }
                let at = c + vec3<f32>(planet_hash(c + 1.3), planet_hash(c + 2.9), planet_hash(c + 5.1));
                let d = length(p - at);
                let r = mix(0.18, 0.62, planet_hash(c + 7.7));
                if d / r < best {
                    best = d / r;
                    size = r;
                    centre = at;
                }
            }
        }
    }
    let bowl = 1.0 - smoothstep(0.0, 1.0, best);
    let rim = exp(-pow((best - 1.0) / 0.18, 2.0));
    // Rays: spokes from the young ones (one in three), fading out to three radii.
    let young = step(0.66, planet_hash(centre + 3.3));
    let away = normalize(p - centre + vec3<f32>(1e-4));
    let spokes = pow(planet_noise(away * 9.0 + centre), 3.0);
    let rays = young * spokes * (1.0 - smoothstep(1.0, 3.0, best)) * step(1.0, best) * size;
    return vec3<f32>(bowl, rim, rays);
}

/// The planet's albedo (linear RGB, about 1 on average: the component's geometric albedo scales it) and its ember
/// glow (0..1) at the point of its surface whose outward normal is `n`, in the planet's own turning frame.
struct PlanetSurface {
    albedo: vec3<f32>,
    ember: f32,
}

fn planet_surface(n: vec3<f32>, seed: f32) -> PlanetSurface {
    let s = vec3<f32>(seed * 17.31, seed * 5.77, seed * 11.13);
    // Highlands and basalt plains: the broad shapes, with ragged coasts.
    let plains = smoothstep(0.47, 0.6, planet_fbm(n * 1.6 + s, 5));
    // The great basin: a ringed scar a third of the disc across, set at a fixed place on the surface.
    let basin_at = normalize(vec3<f32>(0.42, 0.35, 0.84));
    let basin = acos(clamp(dot(n, basin_at), -1.0, 1.0));
    let rings = 0.5 + 0.5 * cos(basin * 46.0) * exp(-pow((basin - 0.55) / 0.22, 2.0));
    let basin_floor = 1.0 - smoothstep(0.32, 0.42, basin);
    let craters = planet_craters(n, 5.0) * 0.7 + planet_craters(n * 1.73, 13.0) * 0.45;
    let detail = planet_fbm(n * 14.0 + s, 3);

    let rust = vec3<f32>(1.32, 0.86, 0.58);
    let ochre = vec3<f32>(1.18, 1.02, 0.78);
    let basalt = vec3<f32>(0.42, 0.4, 0.38);
    var albedo = mix(mix(rust, ochre, smoothstep(0.35, 0.75, detail)), basalt, max(plains, basin_floor * 0.85));
    albedo *= 0.8 + 0.4 * rings * (1.0 - basin_floor);
    albedo *= 1.0 - 0.35 * craters.x;
    albedo += vec3<f32>(0.5, 0.46, 0.4) * (craters.y * 0.6 + craters.z * 1.4);
    albedo *= 0.85 + 0.3 * detail;

    // Fissures: thin ridged lines where the crust is still open, across the basalt and out of the basin.
    let ridge = 1.0 - abs(planet_fbm(n * 3.1 + s * 0.5, 4) * 2.0 - 1.0);
    let fissure = smoothstep(0.93, 0.985, ridge) * (0.35 + 0.65 * max(plains, basin_floor));
    return PlanetSurface(albedo, fissure);
}

/// The planet seen along `dir`: radiance (cd/m2, before the air between it and the eye) and its coverage (0..1, the
/// disc's edge antialiased), so the stars behind are hidden. `toward`: the direction to its centre; `radius`: its
/// angular radius (radians); `sun_dir`: toward the sun; `sun_lux`: the sun's illuminance above the air; `albedo`: its
/// geometric albedo; `surface`: (seed, turns of its spin, ember glow cd/m2 at full, our world's light on its dark
/// side as a share of its day side).
fn sky_planet(
    dir: vec3<f32>,
    toward: vec3<f32>,
    radius: f32,
    sun_dir: vec3<f32>,
    sun_lux: f32,
    albedo: f32,
    surface: vec4<f32>,
) -> vec4<f32> {
    let cos_r = cos(radius);
    let c = dot(dir, toward);
    let pixel = max(length(fwidth(dir)), 1e-6);
    if c < cos_r - 2.0 * pixel {
        return vec4<f32>(0.0);
    }
    // The disc's plane: the sphere's normal under this ray, its visible face toward the eye.
    let side = normalize(cross(toward, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(toward.y) > 0.99)));
    let up = cross(side, toward);
    let off = dir - toward * c;
    let sin_r = sin(radius);
    let x = dot(off, side) / sin_r;
    let y = dot(off, up) / sin_r;
    let rr = x * x + y * y;
    let coverage = 1.0 - smoothstep(1.0 - pixel / sin_r, 1.0 + pixel / sin_r, sqrt(rr));
    let z = sqrt(max(1.0 - rr, 0.0));
    let n = normalize(side * x + up * y - toward * z);

    // Its own frame turns about an axis tilted from the sky's up; a day of its own is `surface.y` turns.
    let spin = surface.y * 6.2831853;
    let axis = normalize(vec3<f32>(0.24, 0.94, 0.24));
    let local = n * cos(spin) + cross(axis, n) * sin(spin) + axis * dot(axis, n) * (1.0 - cos(spin));
    let ground = planet_surface(local, surface.x);

    // Rough rock lit by the sun: Lommel-Seeliger and Lambert, half each; the eye looks along -n's visible face.
    let mu0 = dot(n, sun_dir);
    let mu = max(-dot(n, dir), 1e-3);
    let lit = max(mu0, 0.0);
    let lommel = lit / (lit + mu);
    let brdf = (0.5 * lommel * 2.0 + 0.5 * lit) / 3.14159265;
    // The terminator is soft over a few degrees (relief and its thin air), not a ruled line.
    let terminator = smoothstep(-0.04, 0.06, mu0);
    var radiance = ground.albedo * albedo * sun_lux * brdf * terminator;

    // A thin air: a blue-white rim on the sunlit limb, thickest where the sun grazes it.
    let limb = pow(1.0 - z, 3.0);
    radiance += vec3<f32>(0.55, 0.72, 1.0) * albedo * sun_lux / 3.14159265 * 0.25 * limb * smoothstep(-0.15, 0.2, mu0);
    // The night side: our world's light on it, and the fissures' ember glow (about 1,000 K: deep orange).
    let night = 1.0 - smoothstep(-0.05, 0.08, mu0);
    radiance += ground.albedo * albedo * sun_lux * surface.w * (0.5 + 0.5 * mu) / 3.14159265 * night;
    radiance += vec3<f32>(1.0, 0.33, 0.07) * surface.z * ground.ember * night;
    return vec4<f32>(radiance * coverage, coverage);
}
