// Lightning channels (`pass/lightning.rs`; Sunscar weather, 11 Nov 2026): each span of a bolt as a ribbon facing the
// eye, added to the HDR scene behind the scene's own depth.
//
// A channel is a few centimetres wide and usually kilometres away: far thinner than a pixel. So a ribbon is never
// drawn thinner than MIN_PIXELS, and its radiance is the channel's light per metre divided by the width it is drawn
// at. The light arriving at the eye is then right at any distance, and a near bolt is thin and blinding while a far
// one is a hair: what a camera sees. The glow round a hot white channel is bloom's, from the same light.
//
// A cold coloured channel (a slow discharge in charged air) is mostly glow: `light.w` is the share of its light drawn
// as a soft profile GLOW_METRES wide round it. Its core burns out white on any exposure that shows the land, and
// bloom is a few hundredths of that: the glow is what keeps its colour (Sunscar, 12 Nov 2026: the game's stills
// showed a thin white line). The glow of a span is the span's own share of a line's: a Gaussian across it times the
// part of a Gaussian along it that the span covers, so spans end to end add up to one even glow with no beads at the
// bends and no gaps outside them.

struct LightningParams {
    /// Camera-relative, unjittered.
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    /// Radians a pixel spans.
    pixel_angle: f32,
    viewport: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0) var<uniform> lp: LightningParams;
@group(0) @binding(1) var scene_depth: texture_depth_2d;

/// The thinnest a ribbon is drawn, as the standard deviation of its profile, in pixels.
const MIN_PIXELS: f32 = 0.6;
/// A channel's real width (its luminous core), metres.
const CHANNEL_METRES: f32 = 0.08;
/// The ribbon's half-width in standard deviations: past this the profile is nothing.
const REACH: f32 = 3.0;
/// The standard deviation of a cold channel's glow, metres: the air and rain it lights round itself.
const GLOW_METRES: f32 = 4.5;
/// The least the glow's is, in the core's standard deviations (a far channel's glow is still wider than its core).
const GLOW_MIN: f32 = 4.0;
/// The glow's half-width in its own standard deviations.
const GLOW_REACH: f32 = 2.6;

struct VIn {
    /// xyz: the span's start, w: 0 this corner is at the start, 1 at the end.
    @location(0) a: vec4<f32>,
    /// xyz: the span's end, w: -1 or 1, the side.
    @location(1) b: vec4<f32>,
    /// rgb: candela per metre, w: the share of it drawn as glow (0 a hot channel: its core alone).
    @location(2) light: vec4<f32>,
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    /// Across the ribbon in standard deviations.
    @location(0) across: f32,
    /// Radiance at the ribbon's centre line.
    @location(1) radiance: vec3<f32>,
    /// rgb: the glow's radiance at the centre line of an endless channel, w: its standard deviation in metres (0 no
    /// glow).
    @location(2) glow: vec4<f32>,
    /// x: metres across the channel, y: metres along the span from its start, z: the span's length.
    @location(3) place: vec3<f32>,
}

/// The error function, near enough (within two hundredths).
fn erf_near(x: f32) -> f32 {
    return tanh(1.2025 * x);
}

@vertex
fn vs_main(in: VIn) -> VOut {
    let a = in.a.xyz - lp.camera_pos;
    let b = in.b.xyz - lp.camera_pos;
    let here = mix(a, b, in.a.w);
    let distance = max(length(here), 1.0);
    // Its profile's standard deviation in metres: the channel's own, or the least a pixel can show.
    let sigma = max(CHANNEL_METRES * 0.5, MIN_PIXELS * lp.pixel_angle * distance);
    // Square to the span and to the line of sight.
    let along = normalize(b - a);
    var side = cross(along, here / distance);
    let len = length(side);
    if len < 1e-4 {
        side = vec3<f32>(1.0, 0.0, 0.0);
    } else {
        side = side / len;
    }
    var out: VOut;
    let share = clamp(in.light.w, 0.0, 1.0);
    let span = length(b - a);
    if share <= 0.0 {
        out.clip = lp.view_proj * vec4<f32>(here + side * (in.b.w * REACH * sigma), 1.0);
        out.across = in.b.w * REACH;
        out.glow = vec4<f32>(0.0);
        out.place = vec3<f32>(0.0, 0.5 * span, span);
    } else {
        // Wide enough for the glow, and past both ends by as much: a span's glow rounds its ends.
        let wide = max(GLOW_METRES, GLOW_MIN * sigma);
        let reach = GLOW_REACH * wide;
        let past = (in.a.w * 2.0 - 1.0) * reach;
        out.clip = lp.view_proj * vec4<f32>(here + along * past + side * (in.b.w * reach), 1.0);
        out.across = in.b.w * reach / sigma;
        out.glow = vec4<f32>(in.light.rgb * share / (wide * 2.5066283), wide);
        out.place = vec3<f32>(in.b.w * reach, in.a.w * span + past, span);
    }
    // Candela per metre over the width it is spread across: a Gaussian of that sigma has area sigma * sqrt(2 pi).
    out.radiance = in.light.rgb * (1.0 - share) / (sigma * 2.5066283);
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    // Behind the scene's own surfaces it is hidden. Reversed depth: nearer is larger.
    let dims = vec2<i32>(textureDimensions(scene_depth));
    let scene = textureLoad(scene_depth, clamp(vec2<i32>(in.clip.xy), vec2<i32>(0), dims - 1), 0);
    if in.clip.z < scene {
        discard;
    }
    // The core, between the span's ends only (a glowing span's ribbon runs past them).
    var light = in.radiance * exp(-0.5 * in.across * in.across) * step(0.0, in.place.y) * step(in.place.y, in.place.z);
    if in.glow.w > 0.0 {
        let x = in.place.x / in.glow.w;
        // How much of a Gaussian along the channel this span covers, seen from here.
        let root = in.glow.w * 1.4142136;
        let covered = 0.5 * (erf_near(in.place.y / root) - erf_near((in.place.y - in.place.z) / root));
        // To nothing at the ribbon's edge, so no edge shows on a dark exposure.
        let fade = max(1.0 - x * x / (GLOW_REACH * GLOW_REACH), 0.0);
        light += in.glow.rgb * exp(-0.5 * x * x) * covered * fade * fade;
    }
    return vec4<f32>(light, 1.0);
}
