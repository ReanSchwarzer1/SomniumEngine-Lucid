// Standing water on the ground, and rain's rings on it (Sunscar weather, 11 Nov 2026).
//
// Wetness darkens and glosses the ground; it leaves no pools. Rain that keeps falling collects where the ground is
// flat and a little low, and what the eye then sees is not wet soil but a mirror: the sky in a sheet of water, with
// every stone that stands above it breaking the sheet, a damp rim round it, and while it rains, rings.
//
// One call from shading's fragment stage after wetness, on terrain only. The amount and the rain's rings arrive in
// `weather.w` (a game's `ShadingPass::set_standing_water`); zero, the default, is never called.
//
// The low places are not known (the terrain's own relief is finer than its heights): they are drawn by two scales of
// noise in world space, and the surface's own detail decides the rest, because a detail normal that leans is a stone
// or a ridge standing proud of the water. Rings: one drop a cell of 0.4 m about once a second, a ring spreading at
// 0.28 m/s and fading in under a second.

fn sw_pcg(v: u32) -> u32 {
    var h = v * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

/// Three numbers 0..1 for a cell.
fn sw_cell(c: vec2<i32>, salt: u32) -> vec3<f32> {
    let a = sw_pcg(bitcast<u32>(c.x) * 1597334677u ^ sw_pcg(bitcast<u32>(c.y) + salt));
    let b = sw_pcg(a);
    let d = sw_pcg(b);
    return vec3<f32>(f32(a >> 8u), f32(b >> 8u), f32(d >> 8u)) / 16777216.0;
}

/// Smooth noise 0..1 with one cell every `metres`.
fn sw_noise(xz: vec2<f32>, metres: f32) -> f32 {
    let p = xz / metres;
    let i = vec2<i32>(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = sw_cell(i, 7u).x;
    let b = sw_cell(i + vec2<i32>(1, 0), 7u).x;
    let c = sw_cell(i + vec2<i32>(0, 1), 7u).x;
    let d = sw_cell(i + vec2<i32>(1, 1), 7u).x;
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// The tilt rain's rings give the water at a place (x, z of the normal's offset). `rate` 0..1 is how many of the
/// cells have a drop at any time.
fn sw_rings(xz: vec2<f32>, time: f32, rate: f32) -> vec2<f32> {
    let cell = 0.4;
    var tilt = vec2<f32>(0.0);
    let here = vec2<i32>(floor(xz / cell));
    for (var j = -1; j <= 1; j = j + 1) {
        for (var i = -1; i <= 1; i = i + 1) {
            let c = here + vec2<i32>(i, j);
            let k = sw_cell(c, 13u);
            let period = 0.9 + 0.6 * k.z;
            let beat = time / period + k.x * 17.0;
            // Each beat's drop lands somewhere else in the cell, and only some beats have one.
            let drop = sw_cell(c, u32(floor(beat)) * 31u + 101u);
            if drop.z > rate {
                continue;
            }
            let age = fract(beat) * period;
            let centre = (vec2<f32>(c) + 0.15 + 0.7 * drop.xy) * cell;
            let off = xz - centre;
            let d = length(off);
            // A ring 2.5 cm wide at 0.28 m/s, dying over 0.9 s and as it widens.
            let x = (d - 0.28 * age) / 0.025;
            let ring = -x * exp(-x * x) * saturate(1.0 - age / 0.9) / (1.0 + 6.0 * d);
            tilt += off / max(d, 1e-4) * ring;
        }
    }
    return tilt * 0.9;
}

/// Water standing on the ground at `world`. `geo_n` is the ground's own normal (water stands only where it is nearly
/// level); `hold` 0..1 how well this ground holds water (mud and paving do, scree does not); `level` 0..1 how much
/// has collected; `rain` 0..1 the rings; `time` seconds.
fn apply_standing_water(surface_in: Surface, world: vec3<f32>, geo_n: vec3<f32>, hold: f32, level: f32, rain: f32, time: f32) -> Surface {
    var s = surface_in;
    // Within about four degrees of level it stands; by ten it has run off.
    let flat = smoothstep(0.985, 0.9975, geo_n.y) * saturate(hold * 1.5);
    if flat <= 0.0 || level <= 0.0 {
        return s;
    }
    // Where the ground is low: broad hollows a few metres across, broken by half-metre ones.
    let low = sw_noise(world.xz, 3.7) * 0.7 + sw_noise(world.xz + vec2<f32>(19.3, 7.1), 0.83) * 0.3;
    // More water reaches higher ground: a tenth of level ground under water in drizzle, near half in a cloudburst.
    let reach = mix(0.22, 0.52, level) * flat;
    // What stands proud of it: the detail normal leans where a stone or a ridge is.
    let proud = smoothstep(0.015, 0.09, 1.0 - saturate(dot(s.normal, geo_n)));
    let water = smoothstep(reach, reach - 0.035, low + proud * 0.12);
    // A damp rim round each pool: darker, a little glossier, not yet a mirror.
    let rim = smoothstep(reach + 0.09, reach, low) * (1.0 - water);
    s.albedo = s.albedo * (1.0 - 0.28 * rim);
    s.roughness = s.roughness * (1.0 - 0.45 * rim);
    if water <= 0.0 {
        return s;
    }
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if rain > 0.0 {
        let tilt = sw_rings(world.xz, time, rain);
        up = normalize(vec3<f32>(tilt.x, 1.0, tilt.y));
    }
    // Shallow, a little cloudy water over wet soil: the bed shows darker through it, and the surface is water's own
    // mirror (a dielectric of index 1.33).
    s.albedo = mix(s.albedo, s.albedo * 0.5, water);
    s.normal = normalize(mix(s.normal, up, water));
    s.roughness = mix(s.roughness, 0.035, water);
    s.metallic = mix(s.metallic, 0.0, water);
    s.f0 = mix(s.f0, vec3<f32>(0.02), water);
    return s;
}
