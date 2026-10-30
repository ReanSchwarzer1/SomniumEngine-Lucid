// Somnium Engine — foliage wind. The GPU half of `wind.rs`; change both together.
//
// A pure function with no bindings, so every pass that positions geometry can
// include it: visibility (the raster), shadow (the cascades), shading (the
// triangle reconstruction) and velocity (the motion vector). If one of them
// displaced and another did not, a leaf would be shaded where it was not
// drawn, or cast its shadow from where it no longer is.
//
// `wind` is the view buffer's vec4: xy wind velocity over the ground (m/s,
// world x and z), z strength (authored sway x preset gate; 0 stills
// everything), w fade distance in metres. `bend` and `flutter` come from the
// material (glTF extras `somnium_wind_bend` / `somnium_wind_flutter`).

fn wind_smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// A plant's sway phase in 0..2pi, an integer hash of its pivot to 1/64 m. `fract(sin(dot) * 43758)` hashed nothing
// kilometres from the origin: sin of a six-figure argument loses its low bits, so neighbouring tufts swayed in step
// along lines of constant 12.99x + 78.23z, and a meadow 13 km out showed rows running along x (Sunscar, 31 Oct 2026).
fn wind_phase(pivot: vec3<f32>) -> f32 {
    let c = vec2<i32>(floor(pivot.xz * 64.0));
    var h = (bitcast<u32>(c.x) * 0x8da6b343u) ^ (bitcast<u32>(c.y) * 0xd8163841u);
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    h = h ^ (h >> 16u);
    return f32(h >> 8u) * (6.2831855 / 16777216.0);
}

fn wind_offset(
    p: vec3<f32>,
    pivot: vec3<f32>,
    bend: f32,
    flutter: f32,
    time: f32,
    wind: vec4<f32>,
    camera: vec3<f32>,
) -> vec3<f32> {
    let speed = length(wind.xy);
    if (bend <= 0.0 && flutter <= 0.0) || speed < 0.01 || wind.z <= 0.0 {
        return vec3<f32>(0.0);
    }
    let fade = 1.0 - wind_smoothstep(wind.w * 0.65, wind.w, length(p.xz - camera.xz));
    if fade <= 0.0 {
        return vec3<f32>(0.0);
    }
    let dir = wind.xy / speed;
    let h = max(p.y - pivot.y, 0.0);
    let phase = wind_phase(pivot);
    // Gust fronts travel downwind across the stand at about 10 m/s.
    let front = dot(pivot.xz, dir) * 0.09 - time * 0.9;
    let gust = 0.55 + 0.45 * (0.5 + 0.5 * sin(front)) * (0.6 + 0.4 * sin(time * 0.37 + phase));
    // Drag grows with speed and saturates: a gale does not lay a pine flat.
    let drag = min(speed * 0.22, 1.6);
    let sway = 0.7 + 0.3 * sin(time * 0.9 + phase);
    let lean = bend * h * h * drag * gust * sway;
    var out = vec3<f32>(dir.x * lean, 0.0, dir.y * lean);
    // A bent stem keeps its length, so its tip drops as it leans.
    out.y = -0.5 * lean * lean / max(h, 0.25);
    let q = p * 1.7;
    let f = flutter * drag * clamp(h * 2.0, 0.0, 1.0) * (0.5 + 0.5 * gust);
    out += vec3<f32>(
        sin(time * 5.3 + q.x + q.y * 0.7 + phase),
        0.6 * sin(time * 6.7 + q.z * 1.3 + q.x * 0.5),
        sin(time * 4.9 + q.y + q.z),
    ) * f;
    return out * (fade * wind.z);
}
