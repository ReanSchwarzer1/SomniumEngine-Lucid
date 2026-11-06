// Phase 11.5K: tone-mapping + vignette post-process pass.
// Phase 24A/24B: physical exposure and a selectable tone-mapping curve.
//
// Reads an Rgba16Float HDR texture and writes tone-mapped LDR to the swapchain.
// Applied after the shading pass and grid overlay, before the UI overlay.

//!include "dream_lens.wgsl"
//!include "tone_curves.wgsl"

struct PostParams {
    /// Linear multiplier from scene luminance (cd/m²) to display range.
    /// Derived from EV100, not an arbitrary gain — see `light_units.rs`.
    exposure:          f32,
    vignette_strength: f32,
    /// Chromatic-aberration offset in UV units at the screen edge. 0 = off.
    ca_strength:       f32,
    /// 0 = AgX, 1 = ACES, 2 = Reinhard. Matches `Tonemapper::as_index`.
    tonemapper:        u32,
    /// Non-zero when `metered` should be used in place of `exposure`.
    auto_exposure:     u32,
    /// How much of the bloom chain reaches the image.
    bloom_intensity:   f32,
    // ── Colour grading (Phase 24Y) ──────────────────────────────────────────
    /// White balance, -1 cooler … +1 warmer.
    temperature:       f32,
    /// White balance on the green–magenta axis.
    tint:              f32,
    /// 1.0 is neutral; pivots around middle grey.
    contrast:          f32,
    /// 1.0 is neutral; 0 is monochrome.
    saturation:        f32,
    /// ASC CDL slope / offset / power. Neutral is (1, 0, 1).
    gain:              f32,
    lift:              f32,
    gamma:             f32,
    // ── Lens (Phase 24Z) ────────────────────────────────────────────────────
    /// Film grain strength. 0 = off.
    grain:             f32,
    /// Seconds, so grain moves rather than sitting as a static pattern.
    time:              f32,
    /// CONTROL-K. Non-zero when `response` holds an authored tone-response
    /// curve. Zero leaves grading exactly as Phase 24Y left it, which is what
    /// keeps this addition invisible until somebody authors a curve.
    response_enabled:  f32,
    /// 32 uniform samples of the authored response curve over `0..=1`, packed
    /// four to a vector because a uniform `array<f32, N>` has a 16-byte stride
    /// and would cost four times the space for the same table.
    response: array<vec4<f32>, 8>,
    dream: vec4<f32>,
    /// A transition over the whole picture (`PictureShift`): kind, progress, the wave's reach in metres.
    shift: vec4<f32>,
    /// Where the wave spreads from, in the world.
    shift_at: vec4<f32>,
    /// The camera's, to read the world back out of the depth buffer for the wave.
    inv_view_proj: mat4x4<f32>,
    /// x painterly amount, y ink-line amount (0 skips each); zw unused.
    stylise: vec4<f32>,
    /// x how much of the grading LUT to take (0 leaves it unread); y low-light vision (`low_light`); zw unused.
    lut: vec4<f32>,
    /// The photographic camera (sunscar-photoreal, 6 Nov 2026; `PostProcessComponent::photographic`): x 1 on (0: the
    /// chain as it was, untouched); y the lens's glare share; z sensor noise (1 a real sensor's); w unused.
    photo: vec4<f32>,
    /// x tan(half the vertical field of view), y aspect (width / height), for the lens's cos^4 fall-off; zw unused.
    lens: vec4<f32>,
    /// The camera's white balance: a linear-RGB matrix by rows (w unused); the identity when off.
    wb_r: vec4<f32>,
    wb_g: vec4<f32>,
    wb_b: vec4<f32>,
    /// Heat over hot ground (sunscar-photoreal, BIOMES 2.7): x shimmer, y the inferior mirage (0 skips each), z the sine
    /// of the vanishing line's dip below the horizon; w unused.
    heat: vec4<f32>,
    /// The camera-relative view-projection, to find where a bent ray's direction falls on the screen.
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var hdr_tex:  texture_2d<f32>;
@group(0) @binding(1) var hdr_samp: sampler;
@group(0) @binding(2) var<uniform> pp: PostParams;
/// `[0]` = metered exposure multiplier, `[1]` = adapted EV100 (Phase 24A-3).
@group(0) @binding(3) var<storage, read> metered: array<f32, 2>;
/// Blurred bloom chain (Phase 24T).
@group(0) @binding(4) var bloom_tex: texture_2d<f32>;
// The grading LUT: a .cube, sRGB-encoded in and out as such files are made.
@group(0) @binding(5) var grade_lut: texture_3d<f32>;

/// Colour as the eye sees it in dim light (ENGINE_AUDIT S6, for the Lid nights): under about 4 cd/m2 the cones give way
/// to the rods. Chroma fades with the light (Hunt), and toward rod vision everything goes to the rods' blue-grey,
/// where reds go dark first (Purkinje). Jensen, Premoze, Shirley et al., "A Spatial Post-Processing Algorithm for
/// Images of Night Scenes" (JGT 2002): scotopic luminance from XYZ, and the mesopic blend linear in log luminance
/// from 0.01 to about 4 cd/m2. `strength` scales how far it goes (1 the eye's own).
fn low_light(c: vec3<f32>, strength: f32) -> vec3<f32> {
    let x = dot(c, vec3<f32>(0.4124, 0.3576, 0.1805));
    let y = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let z = dot(c, vec3<f32>(0.0193, 0.1192, 0.9505));
    if y <= 0.0 || x <= 1e-6 {
        return c;
    }
    let scotopic = max(y * (1.33 * (1.0 + (y + z) / x) - 1.68), 0.0);
    let cones = saturate((log2(y) / log2(10.0) + 2.0) / 2.6);
    let s = 1.0 - clamp(strength, 0.0, 1.0) * (1.0 - cones);
    // The rods' blue-grey, at unit luminance (Jensen's chromaticity near x, y = 0.25).
    let rods = vec3<f32>(0.871, 1.002, 1.362);
    return c * s + rods * scotopic * (1.0 - s);
}

fn lut_encode(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3(0.0), vec3(1.0));
    return select(1.055 * pow(x, vec3(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3(0.0031308));
}

fn lut_decode(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(0.04045));
}
/// The visibility buffer and the paint words: a draw whose paint word carries `OUTSIDE_RETRO` (`vertex_paint.rs`) is
/// kept out of the retro picture and shown by a look of its own.
@group(1) @binding(0) var vis_buffer: texture_2d<u32>;
@group(1) @binding(1) var<storage, read> paint_words: array<u32>;
/// The scene's depth: the picture shift's wave finds each pixel's place in the world from it.
@group(1) @binding(2) var scene_depth: texture_depth_2d;

/// Whether the scene pixel under `uv` belongs to a draw kept out of the retro picture.
fn outside_retro(uv: vec2<f32>) -> bool {
    let dims = textureDimensions(vis_buffer);
    let p = min(vec2<u32>(uv * vec2<f32>(dims)), dims - vec2<u32>(1u));
    let visible = textureLoad(vis_buffer, p, 0).x;
    let table = paint_words[0];
    if visible == 0u || table == 0u {
        return false;
    }
    // Word 11: the table's length when instance batches sit past it (`vertex_paint.rs`), else 0.
    let covered = paint_words[11];
    if covered != 0u && visible - 1u >= covered {
        return false;
    }
    return (paint_words[table + visible - 1u] & 0x80000000u) != 0u;
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0)       uv:   vec2<f32>,
}

// Full-screen triangle: NDC (-1,-1), (3,-1), (-1,3).
// UV origin is top-left → UV.y = (1 - NDC.y) / 2.
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VOut {
    var xs = array<f32, 3>(-1.0,  3.0, -1.0);
    var ys = array<f32, 3>(-1.0, -1.0,  3.0);
    let p  = vec2(xs[vid], ys[vid]);
    let uv = vec2((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
    return VOut(vec4(p, 0.0, 1.0), uv);
}

/// The lens's own fall-off toward the corners (natural vignetting): cos^4 of the angle off the axis, taken
/// `vignette_strength` of the way (0 none; 1 an uncorrected lens, about 2 stops in a 55-degree view's corners).
fn lens_falloff(uv: vec2<f32>) -> f32 {
    let ndc = uv * 2.0 - vec2<f32>(1.0);
    let t = vec2<f32>(ndc.x * pp.lens.x * pp.lens.y, ndc.y * pp.lens.x);
    let cos2 = 1.0 / (1.0 + dot(t, t));
    return mix(1.0, cos2 * cos2, clamp(pp.vignette_strength, 0.0, 1.0));
}

/// PCG hash (O'Neill; Jarzynski and Olano, JCGT 2020): one well-mixed u32 from another. Integer, so it does not
/// lose its randomness at large pixel coordinates as a float hash does.
fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

/// Two independent standard normal numbers for a pixel, a frame and a stream (Box-Muller on two PCG draws).
fn gauss2(pixel: vec2<u32>, frame: u32, stream: u32) -> vec2<f32> {
    let base = pcg_hash(pixel.x + pcg_hash(pixel.y + pcg_hash(frame * 8u + stream)));
    let r = sqrt(-2.0 * log(max(f32(base) * 2.3283064e-10, 1e-7)));
    let a = 6.2831853 * f32(pcg_hash(base)) * 2.3283064e-10;
    return r * vec2<f32>(cos(a), sin(a));
}

/// Sensor noise on the exposed light (middle grey 0.18), as a camera makes it: photon noise, the square root of the
/// light gathered, and 3 electrons of read noise. The camera holds ISO 100 down to EV 10 and raises it a stop for
/// each stop less light, so a bright day is clean and a dim wood or a night grows grain. At ISO 100 middle grey
/// gathers about 5,000 electrons, as a full-frame sensor's photosite does; `strength` 1 is that sensor. The noise
/// is mostly luminance, with a third as much colour, as after a camera's own noise reduction.
fn sensor_noise(x: vec3<f32>, pixel: vec2<u32>, strength: f32, ev: f32) -> vec3<f32> {
    let electrons = 27800.0 / exp2(clamp(10.0 - ev, 0.0, 14.0));
    let luma = max(dot(x, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.0);
    let sigma = sqrt(luma * electrons + 9.0) / electrons;
    // A new draw every 60th of a second: noise moves with time, as a sensor's does, and never sits as a pattern.
    let frame = u32(fract(pp.time / 4096.0) * 4096.0 * 60.0);
    let g0 = gauss2(pixel, frame, 0u);
    let g1 = gauss2(pixel, frame, 1u);
    let n = g0.x;
    let noise = vec3<f32>(n) + 0.33 * (vec3<f32>(g0.y, g1.x, g1.y) - vec3<f32>(n));
    // In each channel's own share of the light, so a coloured surface keeps its colour; grey in the near-black.
    let share = select(vec3<f32>(1.0), x / max(luma, 1e-6), luma > 1e-4);
    return max(x + strength * sigma * noise * share, vec3<f32>(0.0));
}

/// Whether a depth-buffer value is the open sky (nothing drawn). One place, so a change of depth convention flips it once.
fn depth_is_sky(z: f32) -> bool {
    return z >= 0.99999;
}

/// The camera-relative view ray through `uv`, and how far the drawn surface there is (1e5 for the sky).
fn view_ray(uv: vec2<f32>) -> vec4<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let mid = pp.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    let dir = normalize(mid.xyz / mid.w);
    let dims = textureDimensions(scene_depth);
    let z = textureLoad(scene_depth, min(vec2<u32>(uv * vec2<f32>(dims)), dims - vec2<u32>(1u)), 0);
    var dist = 1.0e5;
    if !depth_is_sky(z) {
        let h = pp.inv_view_proj * vec4<f32>(ndc, z, 1.0);
        dist = length(h.xyz / h.w);
    }
    return vec4<f32>(dir, dist);
}

/// Heat shimmer (BIOMES 2.7): turbulent hot air refracts the light, so far ground seen along grazing rays wobbles at a
/// few hertz. The offset grows with distance and with how close the ray runs to the horizon; up to about 3 px at
/// strength 1 on far ground at eye level. Returns where to read the scene.
fn heat_shimmer(uv: vec2<f32>, ray: vec4<f32>) -> vec2<f32> {
    if ray.w >= 1.0e5 {
        return uv;
    }
    let graze = 1.0 - smoothstep(0.0, 0.12, abs(ray.y));
    let reach = smoothstep(20.0, 600.0, ray.w);
    let a = pp.heat.x * graze * reach * 0.0012;
    let t = pp.time;
    let wobble = vec2<f32>(sin(uv.y * 900.0 + t * 23.0 + 2.0 * sin(uv.x * 140.0 + t * 7.0)),
                           0.35 * sin(uv.x * 700.0 - t * 19.0 + 2.0 * sin(uv.y * 210.0 - t * 9.0)));
    return uv + wobble * a;
}

/// The inferior mirage (BIOMES 2.7): over hot flat ground, rays dipping less than the vanishing line (`pp.heat.z`, the
/// sine of its dip) are bent back up before they reach it, so the band between the horizon and that line shows the
/// sky and the far land above the horizon, mirrored about the line and shimmering, as a lake would. Only on drawn
/// ground past 80 m. Returns the light to show.
fn inferior_mirage(hdr: vec3<f32>, ray: vec4<f32>) -> vec3<f32> {
    let dip = pp.heat.z;
    if ray.w < 80.0 || ray.w >= 1.0e5 || ray.y > 0.0 || ray.y < -dip {
        return hdr;
    }
    // At the vanishing line the mirrored ray leaves level; toward the horizon it climbs to the dip's height above it.
    let up = (dip + ray.y) * (1.0 + 0.6 * sin(pp.time * 11.0 + ray.x * 4000.0));
    let flat = normalize(vec3<f32>(ray.x, 0.0, ray.z));
    let src = normalize(flat + vec3<f32>(0.0, max(up, 0.0), 0.0));
    let clip = pp.view_proj * vec4<f32>(src, 0.0);
    if clip.w <= 0.0 {
        return hdr;
    }
    let ndc = clip.xy / clip.w;
    let suv = clamp(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5), vec2<f32>(0.0), vec2<f32>(1.0));
    let mirrored = textureSampleLevel(hdr_tex, hdr_samp, suv, 0.0).rgb;
    // Fades in from the vanishing line and out at the horizon, never a hard mirror.
    let w = pp.heat.y * smoothstep(-dip, -0.6 * dip, ray.y) * (1.0 - smoothstep(-0.15 * dip, 0.0, ray.y)) * 0.85;
    return mix(hdr, mirrored, clamp(w, 0.0, 1.0));
}

/// The camera's grade on the scene's own light, after exposure and before the curve: the lens's fall-off, white
/// balance (a chromatic adaptation made on the CPU), temperature and tint as channel gains, gain, saturation about
/// the light's luminance, and contrast in log light about middle grey where the curve does not take it.
fn photo_grade(x_in: vec3<f32>, uv: vec2<f32>) -> vec3<f32> {
    var x = x_in * lens_falloff(uv);
    x = vec3<f32>(dot(pp.wb_r.xyz, x), dot(pp.wb_g.xyz, x), dot(pp.wb_b.xyz, x));
    let t = clamp(pp.temperature, -1.0, 1.0);
    let g = clamp(pp.tint, -1.0, 1.0);
    x = max(x * vec3<f32>(1.0 + t * 0.2, 1.0 + g * 0.1, 1.0 - t * 0.2) * pp.gain, vec3<f32>(0.0));
    let luma = dot(x, vec3<f32>(0.2126, 0.7152, 0.0722));
    x = max(vec3<f32>(luma) + (x - vec3<f32>(luma)) * pp.saturation, vec3<f32>(0.0));
    if pp.tonemapper != 3u && abs(pp.contrast - 1.0) > 1e-4 {
        x = 0.18 * pow(x / 0.18, vec3<f32>(max(pp.contrast, 0.05)));
    }
    return x;
}

/// What the photographic camera leaves to the display: the CDL's offset and power, then the authored response.
fn photo_display_grade(c_in: vec3<f32>) -> vec3<f32> {
    var c = pow(max(c_in + vec3<f32>(pp.lift), vec3<f32>(0.0)), vec3<f32>(max(pp.gamma, 1e-3)));
    if (pp.response_enabled > 0.5) {
        c = vec3<f32>(response_at(c.r), response_at(c.g), response_at(c.b));
    }
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Colour grading (Phase 24Y) ──────────────────────────────────────────────
//
// Applied *after* tone mapping, in display space. Grading before it would
// fight the curve: the tone mapper's job is to fit scene luminance into a
// display's range, and shifting colours beforehand changes what it is fitting.
//
// These are the controls a colourist actually uses. Exposure and the tone curve
// decide how bright the image is and how it rolls off; grading decides what it
// *feels* like, and no amount of the former substitutes for the latter.

/// White balance on the orange–blue and green–magenta axes.
///
/// Applied as a channel scale rather than a full chromatic-adaptation
/// transform. A true Bradford adaptation would be more correct, but this is a
/// look control rather than a colour-management step, and being able to reason
/// about what the slider does is worth more here than the last few percent.
fn white_balance(c: vec3<f32>, temperature: f32, tint: f32) -> vec3<f32> {
    let t = clamp(temperature, -1.0, 1.0);
    let g = clamp(tint, -1.0, 1.0);
    let scale = vec3<f32>(
        1.0 + t * 0.2,
        1.0 + g * 0.1,
        1.0 - t * 0.2,
    );
    return c * scale;
}

/// ASC CDL: slope, offset, power. The grading standard film uses.
fn color_decision_list(c: vec3<f32>, slope: f32, offset: f32, power: f32) -> vec3<f32> {
    let adjusted = c * slope + offset;
    return pow(max(adjusted, vec3<f32>(0.0)), vec3<f32>(max(power, 1e-3)));
}

fn apply_grading(c_in: vec3<f32>) -> vec3<f32> {
    var c = white_balance(c_in, pp.temperature, pp.tint);
    c = color_decision_list(c, pp.gain, pp.lift, pp.gamma);

    // Contrast pivots around middle grey rather than around black, so raising
    // it darkens shadows and lifts highlights instead of just brightening
    // everything.
    const MIDDLE_GREY: f32 = 0.18;
    c = (c - MIDDLE_GREY) * pp.contrast + MIDDLE_GREY;

    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c = mix(vec3<f32>(luma), c, pp.saturation);

    // CONTROL-K: the authored response curve, applied per channel after the
    // fixed grade. Per channel rather than on luminance so a curve can shape
    // colour as well as brightness — an S-curve on all three is the film look,
    // and the alternative would forbid ever curving one of them alone.
    if (pp.response_enabled > 0.5) {
        c = vec3<f32>(response_at(c.r), response_at(c.g), response_at(c.b));
    }

    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}

/// Sample the authored response table, linearly between its 32 entries.
///
/// Indexing is dynamic, which a uniform array supports; the vector component
/// is selected by comparison rather than by dynamic subscript, because WGSL
/// forbids dynamically indexing a vector in a uniform.
fn response_at(x: f32) -> f32 {
    let u = clamp(x, 0.0, 1.0) * 31.0;
    let i0 = u32(floor(u));
    let i1 = min(i0 + 1u, 31u);
    let f  = u - floor(u);
    return mix(response_sample(i0), response_sample(i1), f);
}

fn response_sample(i: u32) -> f32 {
    let v = pp.response[i / 4u];
    let c = i % 4u;
    if (c == 0u) { return v.x; }
    if (c == 1u) { return v.y; }
    if (c == 2u) { return v.z; }
    return v.w;
}

/// Hash for grain and dither.
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q += dot(q, q + 45.32);
    return fract(q.x * q.y);
}

/// The PS2's picture (`pp.dream.w` how far in, sub-phase 11): 448 lines, as its interlaced output had, of square
/// pixels across this screen's width. On the way there (a transition) the lines fall from the screen's own in
/// four steps, each a quarter of the octaves between.
fn ps2_grid(amount: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let octaves = max(log2(dims.y / 448.0), 0.0);
    let lines = 448.0 * exp2(octaves * floor((1.0 - amount) * 4.0 + 0.0001) / 4.0);
    return vec2<f32>(round(lines * dims.x / max(dims.y, 1.0)), round(lines));
}

/// 4x4 ordered (Bayer) threshold for a cell of that grid, in -0.5..0.5.
fn bayer4(cell: vec2<u32>) -> f32 {
    var m = array<f32, 16>(0.0, 8.0, 2.0, 10.0, 12.0, 4.0, 14.0, 6.0, 3.0, 11.0, 1.0, 9.0, 15.0, 7.0, 13.0, 5.0);
    return (m[(cell.y & 3u) * 4u + (cell.x & 3u)] + 0.5) / 16.0 - 0.5;
}

// ── The picture shift (`PictureShift`) ─────────────────────────────────────────────────────────────────────────────

/// Pushing into a television (kind 1), zoomed this much at `p`: in the end one phosphor triad is wider than the view.
fn screen_zoom(p: f32) -> f32 {
    return exp2(p * p * 9.0);
}

/// Where the scene is read for a screen pixel while pushing into a television: the glass bulges a little, the scene is
/// read once per phosphor triad (it is a picture made of them), and the view closes on the centre.
fn screen_uv(uv: vec2<f32>, p: f32) -> vec2<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let zoom = screen_zoom(p);
    let c = uv - vec2(0.5);
    let bulged = vec2(0.5) + c * (1.0 + 0.2 * dot(c, c) * smoothstep(0.0, 0.25, p));
    let cell = 3.0 * zoom;
    let snapped = mix(bulged, (floor(bulged * dims / cell) + 0.5) * cell / dims, smoothstep(0.02, 0.3, p));
    return vec2(0.5) + (snapped - vec2(0.5)) / zoom;
}

/// The television's own picture over `colour`: three upright stripes to a triad, red, green and blue, each lit only by
/// its channel and dark between; dark gaps between the scanlines; the tube's flicker and its rolling bar.
fn screen_mask(colour: vec3<f32>, uv: vec2<f32>, p: f32) -> vec3<f32> {
    let dims = vec2<f32>(textureDimensions(hdr_tex));
    let f = fract(uv * dims / (3.0 * screen_zoom(p)));
    let k = f.x * 3.0;
    let stripe = floor(k);
    var mask = vec3(0.12, 0.14, 1.0);
    if stripe < 0.5 {
        mask = vec3(1.0, 0.12, 0.08);
    } else if stripe < 1.5 {
        mask = vec3(0.1, 1.0, 0.12);
    }
    let edge = smoothstep(0.0, 0.16, fract(k)) * (1.0 - smoothstep(0.84, 1.0, fract(k)));
    let line = smoothstep(0.0, 0.1, f.y) * (1.0 - smoothstep(0.8, 1.0, f.y));
    let flicker = select(0.03 * sin(pp.time * 61.0), 0.0, pp.shift.w > 0.5);
    let roll = 0.88 + 0.12 * sin((uv.y - pp.time * 0.4) * 6.2832) + flicker;
    return mix(colour, colour * mask * (2.4 * edge * line * roll), smoothstep(0.02, 0.35, p));
}

/// The eyes closing (kind 2), shut `p` of the way: what the lids leave of the view. They meet a little below the
/// middle, the upper lid doing most of the closing, curved as lids are; the light through them is a dark red.
fn eyelids(colour: vec3<f32>, uv: vec2<f32>, p: f32) -> vec3<f32> {
    let x = uv.x * 2.0 - 1.0;
    let meet = 0.56;
    let gap = (1.0 - p) * 1.9 - 0.15 * p;
    let curve = 1.0 - 0.38 * x * x;
    let top = meet - gap * 0.62 * curve - 0.02;
    let bottom = meet + gap * 0.38 * curve + 0.02;
    let soft = 0.03 + 0.08 * p;
    let open = smoothstep(top - soft, top + soft, uv.y) * (1.0 - smoothstep(bottom - soft, bottom + soft, uv.y));
    let luma = dot(colour, vec3(0.2126, 0.7152, 0.0722));
    let lid = vec3(0.16, 0.025, 0.02) * (0.25 + luma) * (1.0 - 0.7 * p);
    return mix(lid, colour * (1.0 - 0.45 * p), open);
}

/// A wave of the retro picture through the world (kind 3): (x) 1 where its front has passed this pixel's surface,
/// (y) how near the front is, for its seam. The front is ragged in blocks, as the picture it brings is.
fn picture_wave(uv: vec2<f32>, p: f32) -> vec2<f32> {
    if p >= 0.999 {
        return vec2(1.0, 0.0);
    }
    let dims = textureDimensions(scene_depth);
    let texel = min(vec2<u32>(uv * vec2<f32>(dims)), dims - vec2<u32>(1u));
    let z = textureLoad(scene_depth, texel, 0);
    var d = 1.0e5;
    if !depth_is_sky(z) {
        let h = pp.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
        d = distance(h.xyz / h.w, pp.shift_at.xyz);
    }
    let front = p * pp.shift.z;
    let block = floor(uv * vec2<f32>(96.0, 64.0));
    let tick = select(floor(pp.time * 10.0), 0.0, pp.shift.w > 0.5);
    let ragged = d + (hash21(block + vec2(tick, 0.0)) - 0.5) * 0.7;
    let inside = select(0.0, 1.0, ragged < front);
    let seam = (1.0 - smoothstep(0.0, 0.6, abs(ragged - front))) * step(0.0005, p);
    return vec2(inside, seam);
}

/// A painterly picture (`pp.stylise.x`): the 4-sector Kuwahara filter. Each pixel takes the mean colour of whichever
/// quarter of its neighbourhood varies least, so flat areas flatten into patches and edges stay hard. Variance is
/// measured on a compressed copy (x / (1 + x)) so bright lights do not decide every pixel.
fn kuwahara(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<i32>(textureDimensions(hdr_tex));
    let c = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(0), dims - vec2<i32>(1));
    let r = clamp(i32(round(f32(dims.y) / 480.0)), 2, 3);
    var best = vec3<f32>(0.0);
    var least = 1.0e9;
    for (var q = 0; q < 4; q++) {
        let sx = select(-1, 1, (q & 1) != 0);
        let sy = select(-1, 1, (q & 2) != 0);
        var sum = vec3<f32>(0.0);
        var sum_c = vec3<f32>(0.0);
        var sq_c = vec3<f32>(0.0);
        for (var j = 0; j <= r; j++) {
            for (var i = 0; i <= r; i++) {
                let p = clamp(c + vec2<i32>(i * sx, j * sy), vec2<i32>(0), dims - vec2<i32>(1));
                let v = max(textureLoad(hdr_tex, p, 0).rgb, vec3<f32>(0.0));
                let k = v / (vec3<f32>(1.0) + v);
                sum += v;
                sum_c += k;
                sq_c += k * k;
            }
        }
        let n = f32((r + 1) * (r + 1));
        let mean_c = sum_c / n;
        let variance = sq_c / n - mean_c * mean_c;
        let score = variance.r + variance.g + variance.b;
        if score < least {
            least = score;
            best = sum / n;
        }
    }
    return best;
}

/// Ink lines (`pp.stylise.y`): how strongly this pixel sits on a silhouette. Read through the inverse view depth
/// (`w` of the depth unprojected by `inv_view_proj`), which runs exactly straight across any flat surface on screen,
/// so its second difference is zero on every plane, ground and wall alike, and jumps where one thing stands in front
/// of another. Relative to the pixel's own depth, so a 1 m step reads the same near or far; sky counts as infinitely
/// far, which outlines what stands against it. Fades out with distance.
fn ink(uv: vec2<f32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(scene_depth));
    let c = clamp(vec2<i32>(uv * vec2<f32>(dims)), vec2<i32>(3), dims - vec2<i32>(4));
    let w0 = nearest_inverse_depth(c, dims);
    if w0 <= 0.0 {
        return 0.0;
    }
    // Samples two texels out, each the nearest surface round it: the small holes of a leaf clump close up, so a
    // canopy is inked round its outline and not speckled inside.
    let h = abs(nearest_inverse_depth(c - vec2<i32>(2, 0), dims) + nearest_inverse_depth(c + vec2<i32>(2, 0), dims) - 2.0 * w0);
    let v = abs(nearest_inverse_depth(c - vec2<i32>(0, 2), dims) + nearest_inverse_depth(c + vec2<i32>(0, 2), dims) - 2.0 * w0);
    let edge = smoothstep(0.02, 0.08, max(h, v) / w0);
    // Near lines are full, far ones fade (the isometric camera sits 10 to 120 m away).
    return edge * (1.0 - smoothstep(60.0, 220.0, 1.0 / w0));
}

/// The nearest surface (largest 1 / view depth) at a texel and its four neighbours.
fn nearest_inverse_depth(p: vec2<i32>, dims: vec2<i32>) -> f32 {
    let q = clamp(p, vec2<i32>(1), dims - vec2<i32>(2));
    return max(max(inverse_depth(q, dims), max(inverse_depth(q - vec2<i32>(1, 0), dims), inverse_depth(q + vec2<i32>(1, 0), dims))),
        max(inverse_depth(q - vec2<i32>(0, 1), dims), inverse_depth(q + vec2<i32>(0, 1), dims)));
}

/// 1 / view depth at a depth texel (0 for the sky).
fn inverse_depth(p: vec2<i32>, dims: vec2<i32>) -> f32 {
    let z = textureLoad(scene_depth, p, 0);
    if depth_is_sky(z) {
        return 0.0;
    }
    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(dims);
    return abs((pp.inv_view_proj * vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0)).w);
}

fn tonemap(hdr: vec3<f32>) -> vec3<f32> {
    switch pp.tonemapper {
        case 1u: { return aces_film(hdr); }
        case 2u: { return reinhard(hdr); }
        // The photographic curve; under the photographic camera its slope takes `contrast`.
        case 3u: { return photographic(hdr, select(1.0, pp.contrast, pp.photo.x > 0.5)); }
        // 15: passthrough, for debug views (`postprocess::TONEMAP_PASSTHROUGH`). A curve would grade the very
        // values being inspected, and exposure would crush a 0/1 flag image to black.
        case 15u: { return clamp(hdr, vec3(0.0), vec3(1.0)); }
        default: { return agx(hdr); }
    }
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Chromatic aberration: split the RGB channels along the vector from the
    // screen centre, so fringing grows toward the edges like a real lens.
    // Written branch-free — at ca_strength = 0 all three taps land on the same
    // texel, which is exactly the un-aberrated result. (A branch around
    // textureSample would risk WGSL's uniform-control-flow rules.)
    // A transition over the whole picture (`pp.shift`; none is kind 0). Pushing into a television reads the scene
    // through its glass, closer and closer.
    let shift_kind = u32(pp.shift.x + 0.5);
    let shift_p = clamp(pp.shift.y, 0.0, 1.0);
    var view_uv = in.uv;
    if shift_kind == 1u {
        view_uv = screen_uv(in.uv, shift_p);
    }
    let centre = view_uv - vec2(0.5);
    let periphery = smoothstep(0.18, 0.66, length(centre));
    let amount = pp.dream.y * periphery;
    let t = pp.time * pp.dream.z;
    // A retro picture reads one scene pixel per cell of its own coarse grid: what rendering at that resolution
    // without antialiasing looks like, hard steps and all.
    var retro_amount = clamp(pp.dream.w, 0.0, 1.0);
    // The wave: where its front has passed, the retro picture already; at its front, a seam.
    var seam = 0.0;
    var waved = false;
    if shift_kind == 3u {
        let w = picture_wave(in.uv, shift_p);
        retro_amount = max(retro_amount, w.x);
        seam = w.y;
        waved = w.x > 0.5;
    }
    let retro = retro_amount > 0.0;
    // Behind the wave's front the machine can hold even less than a PS2's picture: half its lines.
    let grid = ps2_grid(retro_amount) * select(1.0, 0.5, waved);
    // What is kept out of the retro picture (`outside_retro`) is read at full resolution, every detail of it.
    let apart = retro && outside_retro(view_uv);
    var base_uv = view_uv;
    if retro && !apart {
        base_uv = (floor(view_uv * grid) + 0.5) / grid;
    }
    let mapped = dream_coordinates(base_uv, pp.dream, pp.time, vec2<f32>(textureDimensions(hdr_tex)));
    var scene_uv = mapped.uv;
    // Heat over hot ground (`pp.heat`): the shimmer moves where the scene is read. Off at zero.
    let heated = (pp.heat.x > 0.0 || pp.heat.y > 0.0) && !retro;
    var ray = vec4<f32>(0.0, 0.0, 1.0, 1.0e5);
    if heated {
        ray = view_ray(scene_uv);
        if pp.heat.x > 0.0 {
            scene_uv = heat_shimmer(scene_uv, ray);
        }
    }
    let fracture = mapped.fracture;
    let crack = mapped.crack;
    let ca_dir = scene_uv - vec2(0.5);
    let ca_off = ca_dir * (pp.ca_strength + 0.012 * fracture);
    var hdr = vec3(
        textureSample(hdr_tex, hdr_samp, scene_uv - ca_off).r,
        textureSample(hdr_tex, hdr_samp, scene_uv).g,
        textureSample(hdr_tex, hdr_samp, scene_uv + ca_off).b,
    );
    // The photographic camera (`pp.photo`). Its lateral chromatic aberration spreads along the spectrum rather than
    // in three hard copies: red and blue each read across their half of the offset, so a fringe softens into its
    // colour as a lens's does.
    let photo = pp.photo.x > 0.5 && !retro;
    if photo && pp.ca_strength > 0.0 {
        hdr = vec3(
            0.5 * (hdr.r + textureSampleLevel(hdr_tex, hdr_samp, scene_uv - 0.5 * ca_off, 0.0).r),
            hdr.g,
            0.5 * (hdr.b + textureSampleLevel(hdr_tex, hdr_samp, scene_uv + 0.5 * ca_off, 0.0).b),
        );
    }
    if apart {
        // Kept out of the retro picture, and shown wrong: its channels slip apart and back, and now and then a band of
        // it tears sideways, a signal that does not belong in this picture.
        let tick = floor(pp.time * 12.0);
        let tear = step(0.8, hash21(vec2(tick, 3.7))) * (hash21(vec2(floor(in.uv.y * 90.0), tick)) - 0.5) * 0.014;
        let slip = vec2(0.0016 + 0.0014 * sin(pp.time * 23.0), 0.0);
        let at = scene_uv + vec2(tear, 0.0);
        hdr = vec3(
            textureSampleLevel(hdr_tex, hdr_samp, at - slip, 0.0).r,
            textureSampleLevel(hdr_tex, hdr_samp, at, 0.0).g,
            textureSampleLevel(hdr_tex, hdr_samp, at + slip, 0.0).b,
        );
    }
    if shift_kind == 2u && shift_p > 0.02 {
        // Eyes closing: the view swims out of focus as they go.
        let r = vec2(0.014, 0.014 * f32(textureDimensions(hdr_tex).x) / f32(textureDimensions(hdr_tex).y)) * shift_p;
        hdr = (hdr * 2.0 + textureSampleLevel(hdr_tex, hdr_samp, scene_uv + vec2(r.x, 0.0), 0.0).rgb
            + textureSampleLevel(hdr_tex, hdr_samp, scene_uv - vec2(r.x, 0.0), 0.0).rgb
            + textureSampleLevel(hdr_tex, hdr_samp, scene_uv + vec2(0.0, r.y), 0.0).rgb
            + textureSampleLevel(hdr_tex, hdr_samp, scene_uv - vec2(0.0, r.y), 0.0).rgb) / 6.0;
    }
    if seam > 0.0 {
        // The wave's front tears the picture: its channels pulled apart along it.
        let slip = vec2(0.008 * seam, 0.0);
        hdr = mix(hdr, vec3(textureSampleLevel(hdr_tex, hdr_samp, scene_uv - slip, 0.0).r, hdr.g,
            textureSampleLevel(hdr_tex, hdr_samp, scene_uv + slip, 0.0).b), seam);
    }
    if retro && !apart {
        // The console's flicker filter: each line blended with the lines above and below it, which is what kept its
        // interlaced picture from shimmering on a television, and what gives it that soft vertical smear.
        let line = vec2<f32>(0.0, 1.0 / grid.y);
        let flicker = 0.25 * retro_amount;
        hdr = hdr * (1.0 - 2.0 * flicker) + flicker * (textureSampleLevel(hdr_tex, hdr_samp, scene_uv - line, 0.0).rgb
            + textureSampleLevel(hdr_tex, hdr_samp, scene_uv + line, 0.0).rgb);
    }

    // A spatial echo, deliberately bounded to the edge and never temporal history.
    if pp.dream.x > 1.5 && pp.dream.x < 2.5 && pp.dream.y > 0.0 {
        let echo_uv = clamp(scene_uv - centre * 0.022 * (0.65 + 0.35 * sin(t * 0.37)), vec2(0.002), vec2(0.998));
        let echo = textureSampleLevel(hdr_tex, hdr_samp, echo_uv, 0.0).rgb;
        hdr = mix(hdr, echo, amount * 0.22);
    }

    if fracture > 0.0 {
        // Double vision: the same picture again, a little to one side, drifting in and out of register; and the
        // cracks catch the light.
        let apart = vec2(0.020 * (0.55 + 0.45 * sin(t * 0.61)), 0.005 * sin(t * 0.43)) * fracture;
        let ghost = textureSampleLevel(hdr_tex, hdr_samp, clamp(scene_uv + apart, vec2(0.002), vec2(0.998)), 0.0).rgb;
        hdr = mix(hdr, max(hdr, ghost), min(fracture * 0.55, 0.6));
        hdr = hdr * (1.0 + 1.6 * crack) + vec3(0.012) * crack;
    }

    // The stylised look (Rimeborn): a painterly picture, before light and grading touch it. Off at zero.
    if pp.stylise.x > 0.0 && !retro {
        hdr = mix(hdr, kuwahara(scene_uv), clamp(pp.stylise.x, 0.0, 1.0));
    }

    // Phase 24T: bloom added *before* exposure and tone mapping, because it is
    // scattering inside the lens — it happens to the light on its way to the
    // sensor, not to the picture afterwards.
    if heated && pp.heat.y > 0.0 {
        hdr = inferior_mirage(hdr, ray);
    }
    let bloom = textureSample(bloom_tex, hdr_samp, view_uv).rgb;
    // The photographic camera's lens keeps the energy: the share it spreads into the glare halo (the chain, which the
    // CPU scales by that share over its level count) leaves the image it came from. No threshold there.
    hdr = hdr * select(1.0, 1.0 - pp.photo.y, photo) + bloom * pp.bloom_intensity;

    // Low-light vision, on the scene's own luminance before exposure, because the eye's rods answer to the light
    // and not to the camera. Off at zero.
    if pp.lut.y > 0.0 && !retro {
        hdr = low_light(hdr, pp.lut.y);
    }

    // Exposure. Physical now, not an arbitrary gain: either the value the
    // manual camera settings imply, or what the histogram metered this frame.
    let exposure = select(pp.exposure, metered[0], pp.auto_exposure != 0u);
    hdr *= exposure;
    // The photographic camera grades the exposed light before the curve, then the sensor adds its noise at the ISO
    // this exposure would take.
    if photo {
        hdr = photo_grade(hdr, in.uv);
        if pp.photo.z > 0.0 {
            let ev = select(log2(1.0 / (1.2 * max(pp.exposure, 1e-12))), metered[1], pp.auto_exposure != 0u);
            hdr = sensor_noise(hdr, vec2<u32>(max(in.clip.xy, vec2<f32>(0.0))), pp.photo.z, ev);
        }
    }

    // Tone map → LDR in [0, 1].
    let ldr = tonemap(hdr);

    // Phase 24Y: grading, in display space after the curve. The photographic camera graded the light already and
    // keeps only the CDL's offset and power and the authored response here.
    var graded = select(apply_grading(ldr), photo_display_grade(ldr), photo);

    // LUT grading (Sunscar, 5 Nov 2026): a look made in a grading tool, over the parametric grade. Texel centres:
    // 0 and 1 land on the first and last entries.
    if pp.lut.x > 0.0 && !retro {
        let n = f32(textureDimensions(grade_lut).x);
        let uvw = lut_encode(graded) * ((n - 1.0) / n) + 0.5 / n;
        let looked = lut_decode(textureSampleLevel(grade_lut, hdr_samp, uvw, 0.0).rgb);
        graded = mix(graded, looked, pp.lut.x);
    }

    // Ink lines over the graded picture: a darkening, so they keep the hue of what they edge. Off at zero.
    if pp.stylise.y > 0.0 && !retro {
        graded *= 1.0 - 0.75 * clamp(pp.stylise.y, 0.0, 1.0) * ink(scene_uv);
    }

    // Radial vignette: darkens screen edges.
    let uv_c = in.uv - vec2(0.5);
    let vign  = 1.0 - smoothstep(0.35, 0.75, length(uv_c) * 1.4 * pp.vignette_strength);
    graded *= select(vign, 1.0, photo);
    if shift_kind == 1u {
        graded = screen_mask(graded, in.uv, shift_p);
    } else if shift_kind == 2u {
        graded = eyelids(graded, in.uv, shift_p);
    }
    if seam > 0.0 {
        // The seam burns: a hot line of the old picture's light where the new one meets it.
        graded = mix(graded, vec3(1.0, 0.86, 0.8), 0.55 * seam * seam);
    }

    // Phase 24Z: film grain. Scaled by how dark the pixel is, because sensor
    // noise is most visible in shadows — applying it flat looks like dirt on
    // the lens rather than like film.
    if pp.grain > 0.0 && !photo {
        let n = hash21(in.uv * 1024.0 + fract(pp.time) * 137.0) - 0.5;
        let luma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
        graded += n * pp.grain * (1.0 - smoothstep(0.0, 0.7, luma));
    }

    // Phase 24Z: ordered dither before the 8-bit write.
    //
    // Not optional now that exposure is physical. A smooth dark gradient — a
    // night sky, a wall falling into shadow — quantises to visible bands at 8
    // bits, and the eye finds those bands far more objectionable than the noise
    // that hides them. Half a bit of noise is enough.
    if apart {
        // Drained to a cold near-grey, harder in its contrast than anything round it, crawling with fine noise.
        let luma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
        // Half its colour kept: the Epilogue's infestation stays a dark wet red on it.
        let cold = mix(vec3(luma), graded, 0.5) * vec3(0.9, 0.97, 1.08);
        let n = hash21(in.uv * 2048.0 + fract(pp.time * 7.3) * 251.0) - 0.5;
        graded = clamp((cold - 0.5) * 1.3 + 0.5, vec3(0.0), vec3(1.0)) + n * 0.09;
    }
    if retro && !apart {
        // 15-bit colour, five bits a channel, through a 4x4 ordered dither on the coarse grid: the PS2's
        // framebuffer, its banding broken into the familiar cross-hatch. (Blending three lines' quantisations, as its
        // flicker filter did after the framebuffer, was tried: three rows of the matrix leave vertical stripes.)
        // On the way there the colour falls a bit at a time, eight to five.
        let threshold = bayer4(vec2<u32>(floor(in.uv * grid)));
        var levels = exp2(round(mix(8.0, 5.0, retro_amount))) - 1.0;
        if waved {
            // Behind the wave: four bits a channel, and the colour draining out of it toward a sick grey-green.
            levels = 15.0;
            let luma = dot(graded, vec3(0.2126, 0.7152, 0.0722));
            graded = mix(graded, vec3(luma) * vec3(0.92, 1.0, 0.9), 0.45);
        }
        graded = floor(clamp(graded, vec3(0.0), vec3(1.0)) * levels + 0.5 + threshold) / levels;
    } else if photo {
        // The photographic camera dithers in the encoded picture, where the sRGB target quantises: a triangular ±1
        // step, new every frame. Half a step in linear light (below) is a quarter of a step in a bright sky, which
        // is why overcast skies banded (the lead's review, 6 Nov).
        let px = vec2<u32>(max(in.clip.xy, vec2<f32>(0.0)));
        let frame = u32(fract(pp.time / 4096.0) * 4096.0 * 60.0);
        let h = pcg_hash(px.x + pcg_hash(px.y + pcg_hash(frame * 8u + 5u)));
        let tri = (f32(h & 0xffffu) + f32(h >> 16u)) / 65535.0 - 1.0;
        graded = lut_decode(clamp(lut_encode(graded) + vec3<f32>(tri / 255.0), vec3<f32>(0.0), vec3<f32>(1.0)));
    } else {
        let dither = (hash21(in.uv * 2048.0) - 0.5) / 255.0;
        graded += dither;
    }

    return vec4(clamp(graded, vec3(0.0), vec3(1.0)), 1.0);
}
