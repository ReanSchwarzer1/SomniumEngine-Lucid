// ReSTIR GI's output gathered over neighbours and time on its own grid, before the shading pass upsamples it
// (Sunscar, 11 Nov 2026).
//
// Two things arrive per GI texel each frame, both one ray's worth:
//
// - the bounce off what the ray hit (`gi_in`: rgb, with the surface's camera distance in alpha), a reservoir's
//   estimate, noisy in blotches that neighbours and frames share;
// - the sky's share (`sky_in`): the environment's radiance along the ray, and the same again if the ray left the
//   scene; for a thin leaf, on alternate rays, the open sky its far side receives (restir_gi.wgsl `sky_out`).
//
// Shown as they come, they are the "black splotches": 15-30 pixel blobs on every surface the sky lights. Here each is
// first averaged over the 3 x 3 texels round it (nine rays a frame instead of one; the sky's ratio is exact, with no
// noise, wherever nothing is in the way), then with the same place a frame ago, found by moving this texel's surface
// point through last frame's camera (the world holds still), for up to `sky_frames` and `bounce_frames` frames.
//
// A place with no history of its own (a silhouette texel that traced the far side last frame, ground a trunk has just
// uncovered, the edge the camera is turning towards) borrows: first from the texels beside where it was that lie in
// the same region of space, then from whatever stood there, credited with a few frames only so its own rays soon
// outweigh the loan. What is left with nothing says so in its frame count, and the shading pass leans on its smooth
// fallback there rather than show one frame's rays.
//
// The bounce is bounded before it is averaged, by the two rules `gi_despeckle.wgsl` held (this is that pass, grown):
// no texel may exceed SPECK_RATIO times the second brightest of its same-surface neighbourhood, nor FIREFLY_RATIO
// times its mean.
//
// Out, three layers: 0 = bounce rgb and this frame's distance (0: nothing traced); 1 = the sky's four sums (near
// side num and den, far side num and count); 2 = the number of frames behind them, in red.

struct AccumulateParams {
    /// Camera-relative: inverse(projection x the view's rotation alone), this frame, without the frame's jitter.
    inv_view_proj: mat4x4<f32>,
    /// Last frame's projection x view rotation, for points relative to last frame's camera.
    prev_view_proj: mat4x4<f32>,
    /// This frame's camera less last frame's.
    camera_step: vec3<f32>,
    /// 0: no history (first frame, resize, the pass switched back on).
    history_valid: f32,
    /// 1 / the full resolution.
    inv_resolution: vec2<f32>,
    /// GI grid step in full-resolution pixels.
    scale: f32,
    frame: u32,
    /// Most frames the sky's share averages.
    sky_frames: f32,
    /// Most frames the bounce averages: fewer while the lights change.
    bounce_frames: f32,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var gi_in: texture_2d<f32>;
@group(0) @binding(1) var sky_in: texture_2d<f32>;
@group(0) @binding(2) var history: texture_2d_array<f32>;
@group(0) @binding(3) var gi_out: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(4) var<uniform> acc: AccumulateParams;

const SPECK_RATIO: f32 = 1.25;
const FIREFLY_RATIO: f32 = 2.5;
/// Frames a texel is credited with when its history is a loan from whatever stood beside it on screen: enough to
/// start smooth, few enough that its own rays outweigh the loan within a quarter of a second.
const BORROWED_FRAMES: f32 = 4.0;

fn accumulate_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Within the upsample's depth weight of 0.1 (shading.wgsl `gi_upsample`): the same surface.
fn same_surface(dist: f32, other: f32) -> bool {
    return other > 0.0 && abs(other - dist) < 2.3026 * (dist * 0.025 + 0.05);
}

/// The same region of space: within 8% of the distance and 30 cm. Wider than a surface on purpose. How much sky a
/// place sees changes over metres, not over the gap between two leaves of one crown, and a crown's texels are all
/// at slightly different depths: held to one surface they would never gather anything.
fn same_region(dist: f32, other: f32) -> bool {
    return other > 0.0 && abs(other - dist) < 0.08 * min(dist, other) + 0.3;
}

/// Where a GI texel sits on the full-resolution picture: the middle of its block, not the pixel it traced this frame
/// (restir_gi.wgsl `gi_full_coord`, which walks the block), and with no jitter. A camera that holds still then finds
/// each texel's history exactly on a texel: nothing is resampled, so nothing blurs however long the history.
fn texel_pixel(g: vec2<i32>) -> vec2<f32> {
    return (vec2<f32>(g) + 0.5) * max(acc.scale, 1.0);
}

@compute @workgroup_size(8, 8, 1)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(gi_in));
    let g = vec2<i32>(gid.xy);
    if g.x >= dims.x || g.y >= dims.y {
        return;
    }
    let c = textureLoad(gi_in, g, 0);
    if c.a <= 0.0 {
        textureStore(gi_out, g, 0, vec4<f32>(0.0));
        textureStore(gi_out, g, 1, vec4<f32>(0.0));
        textureStore(gi_out, g, 2, vec4<f32>(0.0));
        return;
    }

    // ── This frame: the neighbourhood's bounce, bounded and averaged; its sky sums, added ─────────────────────
    // `near` holds the same-surface texels' bounce with their luminance in alpha (the first is this texel's own).
    var near: array<vec4<f32>, 9>;
    near[0] = vec4<f32>(c.rgb, accumulate_luma(c.rgb));
    var count = 1;
    var sky = textureLoad(sky_in, g, 0);
    var sky_n = 1.0;
    for (var k = 0; k < 9; k = k + 1) {
        if k == 4 {
            continue;
        }
        let at = g + vec2<i32>(k % 3, k / 3) - vec2<i32>(1);
        if any(at < vec2<i32>(0)) || any(at >= dims) {
            continue;
        }
        let t = textureLoad(gi_in, at, 0);
        if same_surface(c.a, t.a) {
            near[count] = vec4<f32>(t.rgb, accumulate_luma(t.rgb));
            count = count + 1;
        }
        if same_region(c.a, t.a) {
            sky += textureLoad(sky_in, at, 0);
            sky_n += 1.0;
        }
    }
    var brightest = 0.0;
    var second = 0.0;
    var total = 0.0;
    for (var k = 0; k < count; k = k + 1) {
        let l = near[k].a;
        total += l;
        if l > brightest {
            second = brightest;
            brightest = l;
        } else {
            second = max(second, l);
        }
    }
    var ceiling = FIREFLY_RATIO * total / f32(count) + 1.0e-6;
    if count > 1 {
        ceiling = min(ceiling, SPECK_RATIO * second + 1.0e-6);
    }
    var bounce = vec3<f32>(0.0);
    for (var k = 0; k < count; k = k + 1) {
        bounce += near[k].rgb * min(1.0, ceiling / max(near[k].a, 1.0e-6));
    }
    bounce = bounce / f32(count);
    sky = sky / sky_n;
    var frames = 0.0;

    // ── Last frame: the same place ─────────────────────────────────────────────────────────────────────────
    if acc.history_valid > 0.5 {
        let ndc = texel_pixel(g) * acc.inv_resolution * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0);
        // Reversed depth: 1 is the near plane, and a point on it gives the pixel's ray.
        let ray = acc.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
        let before = normalize(ray.xyz / ray.w) * c.a + acc.camera_step;
        let clip = acc.prev_view_proj * vec4<f32>(before, 1.0);
        if clip.w > 1.0e-4 {
            let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
            let expected = length(before);
            // In GI texels by the grid's step, not by `dims`: the grid rounds up, so its last texel overhangs.
            let at = uv / (acc.inv_resolution * max(acc.scale, 1.0)) - 0.5;
            var old_bounce = vec3<f32>(0.0);
            var old_sky = vec4<f32>(0.0);
            var old_frames = 0.0;
            var weight = 0.0;
            if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
                // The four texels round where it was, each if it holds this region.
                let base = vec2<i32>(floor(at));
                let f = at - floor(at);
                for (var k = 0; k < 4; k = k + 1) {
                    let d = vec2<i32>(k & 1, k >> 1);
                    let tap = clamp(base + d, vec2<i32>(0), dims - 1);
                    let a = textureLoad(history, tap, 0, 0);
                    if !same_region(expected, a.a) {
                        continue;
                    }
                    let w = mix(1.0 - f.x, f.x, f32(d.x)) * mix(1.0 - f.y, f.y, f32(d.y)) + 1.0e-3;
                    old_bounce += a.rgb * w;
                    old_sky += textureLoad(history, tap, 1, 0) * w;
                    old_frames += textureLoad(history, tap, 2, 0).r * w;
                    weight += w;
                }
            }
            // Not there (a silhouette texel that traced the far side last frame): the texels beside it that hold this
            // region lend theirs. Failing that, whatever stood there does, for a few frames' credit.
            let nearest = clamp(vec2<i32>(round(at)), vec2<i32>(0), dims - 1);
            var borrowed = false;
            for (var round_index = 0; round_index < 2; round_index = round_index + 1) {
                if weight >= 0.25 || (round_index == 1 && weight > 0.0) {
                    break;
                }
                for (var k = 0; k < 9; k = k + 1) {
                    let tap = clamp(nearest + vec2<i32>(k % 3, k / 3) - vec2<i32>(1), vec2<i32>(0), dims - 1);
                    let a = textureLoad(history, tap, 0, 0);
                    if a.a <= 0.0 || (round_index == 0 && !same_region(expected, a.a)) {
                        continue;
                    }
                    old_bounce += a.rgb;
                    old_sky += textureLoad(history, tap, 1, 0);
                    old_frames += textureLoad(history, tap, 2, 0).r;
                    weight += 1.0;
                    borrowed = round_index == 1;
                }
            }
            if weight > 0.0 {
                frames = old_frames / weight;
                if borrowed {
                    frames = min(frames, BORROWED_FRAMES);
                }
                let keep_sky = 1.0 - 1.0 / min(frames + 1.0, max(acc.sky_frames, 1.0));
                let keep_bounce = 1.0 - 1.0 / min(frames + 1.0, max(acc.bounce_frames, 1.0));
                sky = mix(sky, old_sky / weight, keep_sky);
                bounce = mix(bounce, old_bounce / weight, keep_bounce);
            }
        }
    }

    textureStore(gi_out, g, 0, vec4<f32>(bounce, c.a));
    textureStore(gi_out, g, 1, sky);
    textureStore(gi_out, g, 2, vec4<f32>(min(frames + 1.0, max(acc.sky_frames, acc.bounce_frames)), 0.0, 0.0, 0.0));
}
