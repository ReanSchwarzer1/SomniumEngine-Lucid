// Small lights that hold up under a pixel (sunscar-photoreal, 12 Nov 2026). A lamp's flame or glass is a mesh a few
// centimetres across: a few tens of metres off it is smaller than a pixel, so the jittered, upscaled picture finds it
// in some frames and not in others, and a far lantern or a lit window vanishes or flickers. Each local light the eye
// can see also lands here as a point: its intensity over the distance squared, as radiance over one pixel's solid
// angle, shared bilinearly among the four pixels round where it falls (the same light for any sub-pixel position, so it
// neither flickers nor crawls), and fading out as its own source grows past a pixel and draws itself. Drawn into the
// HDR before bloom and the lens's glare, which give it its halo. Behind the scene's depth it adds nothing.

/// `cluster.rs` `GpuLocalLight`, as global_pool.wgsl declares it.
struct GpuLocalLight {
    position_ws: vec3<f32>,
    range: f32,
    color: vec3<f32>,
    light_type: u32,
    direction_ws: vec3<f32>,
    spot_cos_outer: f32,
    spot_cos_inner: f32,
    radius: f32,
    _pad1: f32,
    _pad2: f32,
}

struct GlintParams {
    view_proj: mat4x4<f32>,
    camera: vec4<f32>,  // the eye, then the near plane (reversed infinite Z: view depth = near / depth)
    frame: vec4<f32>,   // render width and height, one over a pixel's solid angle (1/sr), strength
    reach: vec4<f32>,   // extinction per metre, the farthest light drawn (m), a pixel's angle (rad), unused
}

@group(0) @binding(0) var<uniform> params: GlintParams;
@group(0) @binding(1) var<storage, read> lights: array<GpuLocalLight>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) radiance: vec3<f32>,
    @location(1) centre: vec2<f32>,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) li: u32) -> VOut {
    var out: VOut;
    out.pos = vec4<f32>(-2.0, -2.0, 0.0, 1.0);
    out.radiance = vec3<f32>(0.0);
    out.centre = vec2<f32>(0.0);
    let l = lights[li];
    // Point (0) and spot (1) lamps; area lights draw their own shapes.
    if l.light_type > 1u || max(l.color.x, max(l.color.y, l.color.z)) <= 0.0 {
        return out;
    }
    let to_light = l.position_ws - params.camera.xyz;
    let d = length(to_light);
    if d < 0.5 || d > params.reach.y {
        return out;
    }
    let clip = params.view_proj * vec4<f32>(l.position_ws, 1.0);
    if clip.w <= 0.0 {
        return out;
    }
    let ndc = clip.xyz / clip.w;
    if any(abs(ndc.xy) > vec2<f32>(1.02)) {
        return out;
    }
    let size = params.frame.xy;
    let px = (ndc.xy * vec2<f32>(0.5, -0.5) + 0.5) * size;
    // Behind the scene: what the light's pixel shows is nearer than the light by more than the lamp's own size.
    let texel = clamp(vec2<i32>(px), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let scene_dist = params.camera.w / max(textureLoad(scene_depth, texel, 0), 1.0e-9);
    if scene_dist < clip.w - max(l.radius, 0.05) - 0.3 {
        return out;
    }
    // A spot lamp sends this way only inside its cone.
    var toward = 1.0;
    if l.light_type == 1u {
        toward = smoothstep(l.spot_cos_outer, l.spot_cos_inner, dot(-to_light / d, l.direction_ws));
    }
    // A source wider than about a pixel draws itself.
    let across = 2.0 * max(l.radius, 0.0) / d / params.reach.z;
    let small = 1.0 - smoothstep(0.5, 1.5, across);
    let light = l.color * (toward * small * params.frame.w * exp(-params.reach.x * d) / (d * d) * params.frame.z);
    if max(light.x, max(light.y, light.z)) <= 0.0 {
        return out;
    }
    // The quad over the pixels round the point, a pixel and a half each way.
    let corner = vec2<f32>(f32(vi & 1u), f32(vi >> 1u)) * 2.0 - 1.0;
    let at = px + corner * 1.5;
    out.pos = vec4<f32>(at / size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.radiance = light;
    out.centre = px;
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    // Bilinear: each of the four pixels round the point takes its share, and the shares sum to one.
    let d = abs(in.pos.xy - in.centre);
    let w = max(1.0 - d.x, 0.0) * max(1.0 - d.y, 0.0);
    return vec4<f32>(in.radiance * w, 0.0);
}
