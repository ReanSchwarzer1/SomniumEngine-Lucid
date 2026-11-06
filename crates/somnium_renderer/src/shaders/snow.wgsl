// Snow lying on surfaces (sunscar-photoreal, 6 Nov 2026; `somnium.SnowCover`). Included by shading.wgsl, which calls
// `apply_snow` once per opaque surface after terrain, decals and wetness, so terrain, rock, ruins, roofs, branches and
// the far stand-in cells all take it. Its settings ride at the end of the View uniform (global_pool.wgsl `snow0..3`):
//   snow0 = (on, snowline m, full-cover height m, cos of the slope where snow stops sliding off)
//   snow1 = (shade bias, depth m, age 0 fresh .. 1 old and dirty, sparkle)
//   snow2 = (direction toward the sun, unused)
//   snow3 = (wind direction x, z, scour, unused)
// Off (snow0.x 0) costs one branch.

/// PCG on integer cells: far from the world's origin a float hash (fract(sin(big))) loses its randomness.
fn snow_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn snow_cell(c: vec2<i32>) -> f32 {
    return f32(snow_pcg(u32(c.x) * 73856093u ^ snow_pcg(u32(c.y)))) * 2.3283064e-10;
}

/// Smooth value noise over world xz at `cell` metres.
fn snow_noise(xz: vec2<f32>, cell: f32) -> f32 {
    let p = xz / cell;
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = snow_cell(i);
    let b = snow_cell(i + vec2<i32>(1, 0));
    let c = snow_cell(i + vec2<i32>(0, 1));
    let d = snow_cell(i + vec2<i32>(1, 1));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

/// How much snow lies here, 0..1: above the snowline (lower and deeper on faces turned from the sun, ragged with
/// noise at its edge), only where the surface is flat enough to hold it, thinned on ridges the wind scours.
fn snow_amount(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let line = view.snow0.y;
    let full = max(view.snow0.z, line + 1.0);
    // Faces turned from the sun keep snow lower down: up to `shade bias` x 150 m.
    let flat_n = n.xz / max(length(n.xz), 1e-4);
    let flat_sun = view.snow2.xz / max(length(view.snow2.xz), 1e-4);
    let away = 0.5 - 0.5 * dot(flat_n, flat_sun) * step(0.05, length(n.xz));
    let ragged = (snow_noise(world.xz, 23.0) - 0.5) * 70.0 + (snow_noise(world.xz, 4.1) - 0.5) * 18.0;
    let height = world.y + away * view.snow1.x * 150.0 + ragged;
    var amount = smoothstep(line, full, height);
    // Slope: snow slides off steep faces; none past the limit.
    amount *= smoothstep(view.snow0.w - 0.06, view.snow0.w + 0.10, n.y);
    // Wind: faces turned into the wind are scoured.
    let wind = view.snow3.xy / max(length(view.snow3.xy), 1e-4);
    amount *= 1.0 - view.snow3.z * saturate(-dot(flat_n, wind)) * (1.0 - n.y) * 2.0;
    return saturate(amount);
}

/// Lay the snow over a finished surface: its albedo goes to snow's (0.86-0.92, dimmer and dirtier with age), rough
/// and non-metallic, its normal is buried toward the geometric up (small stones and grain vanish under it), and the
/// cover breaks against the surface's own relief so rock and ridges show through with crisp edges.
fn apply_snow(surface_in: Surface, world: vec3<f32>, geo_n: vec3<f32>) -> Surface {
    var s = surface_in;
    var amount = snow_amount(world, geo_n);
    if amount <= 0.0 {
        return s;
    }
    // Relief breaks through: where the detail normal stands off the geometric one (stones, ledges), snow is thin.
    let relief = 1.0 - saturate(dot(s.normal, geo_n));
    amount = saturate(amount * 1.25 - relief * 3.0 * (1.0 - amount));
    // A crisp edge: snow is there or it is not, over a narrow band.
    amount = smoothstep(0.25, 0.6, amount);
    let age = saturate(view.snow1.z);
    let fresh = vec3<f32>(0.88, 0.91, 0.94);
    let dirt = (s.albedo - vec3<f32>(dot(s.albedo, vec3<f32>(0.333)))) * 0.2 + s.albedo * 0.25;
    let snow_albedo = mix(fresh, fresh * 0.82 + dirt, age);
    s.albedo = mix(s.albedo, snow_albedo, amount);
    s.roughness = mix(s.roughness, mix(0.6, 0.85, age), amount);
    s.metallic = mix(s.metallic, 0.0, amount);
    let buried = normalize(mix(geo_n, vec3<f32>(0.0, 1.0, 0.0), 0.35));
    s.normal = normalize(mix(s.normal, buried, amount * 0.85));
    s.bent_normal = normalize(mix(s.bent_normal, buried, amount * 0.85));
    // Snow fills cracks and hollows: less of the surface's own cavity darkening survives under it.
    s.occlusion = mix(s.occlusion, max(s.occlusion, 0.85), amount);
    return s;
}
