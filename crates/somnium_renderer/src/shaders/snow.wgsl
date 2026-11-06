// Snow lying on surfaces (sunscar-photoreal, 6-7 Nov 2026; PostProcess `snow*`). Included by shading.wgsl, which calls
// `apply_snow` once per opaque surface after terrain, decals and wetness, so terrain, rock, ruins, roofs, branches and
// the far stand-in cells all take it. Its settings are its own uniform (shading group 1 binding 34):
//   p0 = (on, snowline m, full-cover height m, cos of the slope where snow stops sliding off)
//   p1 = (shade bias, reserved, age 0 fresh .. 1 old and dirty, reserved)
//   p2 = (direction toward the sun, metres per pixel per metre of distance)
//   p3 = (wind direction x, z, scour, unused)
// Off (p0.x 0) costs one branch.

struct SnowParams {
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
}
@group(1) @binding(34) var<uniform> snow_params: SnowParams;

/// PCG on integer cells: far from the world's origin a float hash (fract(sin(big))) loses its randomness.
fn snow_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn snow_cell(c: vec2<i32>) -> f32 {
    return f32(snow_pcg(u32(c.x) * 73856093u ^ snow_pcg(u32(c.y)))) * 2.3283064e-10;
}

/// Smooth value noise over `p` in cells: (value 0..1, d/dp.x, d/dp.y).
fn snow_noise_d(p: vec2<f32>) -> vec3<f32> {
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * (3.0 - 2.0 * f);
    let du = 6.0 * f * (1.0 - f);
    let a = snow_cell(i);
    let b = snow_cell(i + vec2<i32>(1, 0));
    let c = snow_cell(i + vec2<i32>(0, 1));
    let d = snow_cell(i + vec2<i32>(1, 1));
    let k = a - b - c + d;
    return vec3<f32>(
        a + (b - a) * u.x + (c - a) * u.y + k * u.x * u.y,
        (b - a + k * u.y) * du.x,
        (c - a + k * u.x) * du.y,
    );
}

/// Smooth value noise over world xz at `cell` metres, 0..1.
fn snow_noise(xz: vec2<f32>, cell: f32) -> f32 {
    return snow_noise_d(xz / cell).x;
}

/// 1 while a pixel is small against `cell` metres, falling to 0 once it covers half a cell: finer than that the
/// pattern would only flicker, so it fades to its mean.
fn snow_fade(footprint: f32, cell: f32) -> f32 {
    return 1.0 - smoothstep(0.15 * cell, 0.5 * cell, footprint);
}

/// Streaks down the fall line, 0..1: noise `across_m` wide and `along_m` long, laid along the nearest two of eight
/// fixed bearings and blended between them, so the pattern holds still as the slope turns (a frame turned with the
/// slope itself would swing kilometres of noise past each pixel this far from the origin).
fn snow_streaks(xz: vec2<f32>, fall: vec2<f32>, across_m: f32, along_m: f32) -> f32 {
    let step_angle = 0.39269908; // pi / 8: streaks have no head or tail, so eight bearings cover the half turn
    let k = atan2(fall.y, fall.x) / step_angle;
    let k0 = floor(k);
    let t = k - k0;
    var sum = 0.0;
    for (var j = 0; j < 2; j++) {
        let index = (i32(k0) + j + 16) % 8;
        let angle = f32(index) * step_angle;
        let along = vec2<f32>(cos(angle), sin(angle));
        let across = vec2<f32>(-along.y, along.x);
        let q = vec2<f32>(dot(xz, across) / across_m, dot(xz, along) / along_m) + vec2<f32>(f32(index) * 17.3);
        sum += snow_noise_d(q).x * select(1.0 - t, t, j == 1);
    }
    return sum;
}

/// Where snow lies, from the geometric normal: (amount 0..1, nearness to its lower edge 0..1, metres per pixel).
/// Above the snowline, lower and deeper on faces turned from the sun and on lee faces, ragged at its edge (broad
/// lobes, finer breakup, and on slopes streaks down the fall line: gullies hold snow, ribs shed it), only where the
/// slope holds it (gullies hold it on steeper ground), stripped from faces the wind strikes.
fn snow_cover(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let footprint = distance(world, view.camera_pos) * snow_params.p2.w;
    let line = snow_params.p0.y;
    let full = max(snow_params.p0.z, line + 1.0);
    let steep = length(n.xz);
    let fall = select(vec2<f32>(1.0, 0.0), n.xz / max(steep, 1e-4), steep > 1e-3);
    let flat_sun = snow_params.p2.xz / max(length(snow_params.p2.xz), 1e-4);
    let away = 0.5 - 0.5 * dot(fall, flat_sun) * step(0.05, steep);
    let wind = snow_params.p3.xy / max(length(snow_params.p3.xy), 1e-4);
    let lee = dot(fall, wind) * steep;
    let lobes = (snow_noise(world.xz, 260.0) - 0.5) * snow_fade(footprint, 260.0);
    let mid = (snow_noise(world.xz, 23.0) - 0.5) * snow_fade(footprint, 23.0);
    let fine = (snow_noise(world.xz, 4.1) - 0.5) * snow_fade(footprint, 4.1);
    let on_slope = smoothstep(0.2, 0.55, steep);
    let gullies = ((snow_streaks(world.xz, fall, 16.0, 110.0) - 0.5) * snow_fade(footprint, 16.0)
        + (snow_streaks(world.xz, fall, 90.0, 520.0) - 0.5) * 0.8 * snow_fade(footprint, 90.0)) * on_slope;
    let height = world.y + away * snow_params.p1.x * 150.0 + snow_params.p3.z * max(lee, 0.0) * 120.0
        + lobes * 110.0 + mid * 55.0 + fine * 16.0 + gullies * 110.0;
    var amount = smoothstep(line, full, height);
    let hold = snow_params.p0.w - gullies * 0.2 - mid * 0.1;
    amount *= smoothstep(hold - 0.06, hold + 0.10, n.y);
    amount *= 1.0 - snow_params.p3.z * saturate(-lee) * 1.5;
    let edge = 1.0 - smoothstep(line, mix(line, full, 0.5), height);
    return vec3<f32>(saturate(amount), edge, footprint);
}

/// Lay the snow over a finished surface: its albedo goes to snow's (0.86-0.92; old snow at the lower edge greyer and
/// dirtier), rough and non-metallic, its normal is buried toward the geometric up (small stones and grain vanish
/// under it) and rippled by the wind near the eye, and the cover breaks against the surface's own relief so rock and
/// ridges show through with crisp edges.
fn apply_snow(surface_in: Surface, world: vec3<f32>, geo_n: vec3<f32>) -> Surface {
    var s = surface_in;
    let cover = snow_cover(world, geo_n);
    var amount = cover.x;
    if amount <= 0.0 {
        return s;
    }
    // Relief breaks through: where the detail normal stands off the geometric one (stones, ledges), snow is thin.
    let relief = 1.0 - saturate(dot(s.normal, geo_n));
    amount = saturate(amount * 1.25 - relief * 3.0 * (1.0 - amount));
    // A crisp edge: snow is there or it is not, over a narrow band.
    amount = smoothstep(0.25, 0.6, amount);
    let age = saturate(snow_params.p1.z + cover.y * 0.6);
    let fresh = vec3<f32>(0.88, 0.91, 0.94);
    let dirt = (s.albedo - vec3<f32>(dot(s.albedo, vec3<f32>(0.333)))) * 0.2 + s.albedo * 0.3;
    let snow_albedo = mix(fresh, fresh * 0.62 + dirt, age);
    s.albedo = mix(s.albedo, snow_albedo, amount);
    s.roughness = mix(s.roughness, mix(0.6, 0.85, age), amount);
    s.metallic = mix(s.metallic, 0.0, amount);
    let buried = normalize(mix(geo_n, vec3<f32>(0.0, 1.0, 0.0), 0.35));
    s.normal = normalize(mix(s.normal, buried, amount * 0.85));
    s.bent_normal = normalize(mix(s.bent_normal, buried, amount * 0.85));
    // Wind ripples, 0.7 m across the wind and 4 m along it, 3 cm high; gone before a pixel is too coarse for them.
    let wind = select(vec2<f32>(0.0, 1.0), normalize(snow_params.p3.xy), length(snow_params.p3.xy) > 1e-3);
    let side = vec2<f32>(-wind.y, wind.x);
    let ripple = snow_noise_d(vec2<f32>(dot(world.xz, wind) / 0.7, dot(world.xz, side) / 4.0));
    let slope = (wind * (ripple.y / 0.7) + side * (ripple.z / 4.0)) * 0.03 * amount * snow_fade(cover.z, 0.7);
    s.normal = normalize(s.normal - vec3<f32>(slope.x, 0.0, slope.y));
    // Snow fills cracks and hollows: less of the surface's own cavity darkening survives under it.
    s.occlusion = mix(s.occlusion, max(s.occlusion, 0.85), amount);
    return s;
}
