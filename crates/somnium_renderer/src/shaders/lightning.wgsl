// Lightning channels (`pass/lightning.rs`; Sunscar weather, 11 Nov 2026): each span of a bolt as a ribbon facing the
// eye, added to the HDR scene behind the scene's own depth.
//
// A channel is a few centimetres wide and usually kilometres away: far thinner than a pixel. So a ribbon is never
// drawn thinner than MIN_PIXELS, and its radiance is the channel's light per metre divided by the width it is drawn
// at. The light arriving at the eye is then right at any distance, and a near bolt is thin and blinding while a far
// one is a hair: what a camera sees. The glow round it is bloom's, from the same light.

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

struct VIn {
    /// xyz: the span's start, w: 0 this corner is at the start, 1 at the end.
    @location(0) a: vec4<f32>,
    /// xyz: the span's end, w: -1 or 1, the side.
    @location(1) b: vec4<f32>,
    /// rgb: candela per metre.
    @location(2) light: vec4<f32>,
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    /// Across the ribbon in standard deviations.
    @location(0) across: f32,
    /// Radiance at the ribbon's centre line.
    @location(1) radiance: vec3<f32>,
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
    out.clip = lp.view_proj * vec4<f32>(here + side * (in.b.w * REACH * sigma), 1.0);
    out.across = in.b.w * REACH;
    // Candela per metre over the width it is spread across: a Gaussian of that sigma has area sigma * sqrt(2 pi).
    out.radiance = in.light.rgb / (sigma * 2.5066283);
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
    return vec4<f32>(in.radiance * exp(-0.5 * in.across * in.across), 1.0);
}
