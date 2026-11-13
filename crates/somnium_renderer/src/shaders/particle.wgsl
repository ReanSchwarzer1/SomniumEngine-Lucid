// Linear HDR sprites; texture alpha, depth, lifetime and colour are independent.
enable wgpu_binding_array;
struct ParticleView {
    view_proj: mat4x4<f32>,
    camera_right: vec3<f32>, _pad0: f32,
    camera_up: vec3<f32>, _pad1: f32,
    camera_pos: vec3<f32>, _pad2: f32,
}
// The sun as the global pool holds it (`DirectionalLight`, its first 312 bytes; a prefix binds the same buffer).
struct ParticleSun {
    direction: vec3<f32>, _pad0: f32,
    color: vec3<f32>, _pad1: f32,
    view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    shadow_map_size: f32,
    ibl_intensity: f32,
}
struct GpuParticle {
    position: vec3<f32>, size: f32,
    color: vec4<f32>,
    tip_tint: vec3<f32>, aspect: f32,
    uv_rect: vec4<f32>,
    rotation: f32, texture_index: i32, flags: u32, flutter: f32,
}
@group(0) @binding(0) var<uniform> pview: ParticleView;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var sprite_sampler: sampler;
@group(1) @binding(4) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(6) var<storage, read> sun: ParticleSun;
const UV: array<vec2<f32>,6> = array<vec2<f32>,6>(vec2(0.,1.),vec2(1.,1.),vec2(1.,0.),vec2(0.,1.),vec2(1.,0.),vec2(0.,0.));
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) colour: vec4<f32>,
    @location(2) tip: vec3<f32>,
    @location(3) @interpolate(flat) texture_index: i32,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) local_uv: vec2<f32>,
    @location(6) @interpolate(flat) uv_rect: vec4<f32>,
    @location(7) @interpolate(flat) flutter: f32,
    @location(8) view_dir: vec3<f32>,
}
// A flame tongue's turbulence (bit 4).
fn flame_hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
fn flame_noise(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3. - 2. * f);
    return mix(mix(flame_hash(i), flame_hash(i + vec2(1., 0.)), u.x), mix(flame_hash(i + vec2(0., 1.)), flame_hash(i + vec2(1., 1.)), u.x), u.y);
}
@vertex fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> Out {
    let p = particles[iid]; let uv = UV[vid];
    let xy = vec2((uv.x - .5) * p.size, (.5 - uv.y) * p.size * p.aspect);
    let c = cos(p.rotation); let s = sin(p.rotation);
    let offset = vec2(c*xy.x-s*xy.y,s*xy.x+c*xy.y);
    var right = pview.camera_right; var up = pview.camera_up;
    if (p.flags & 2u) != 0u {
        let horizontal = vec3(right.x, 0., right.z);
        right = horizontal / max(length(horizontal),0.00001);
        if length(horizontal) < 0.00001 { right = vec3(1.,0.,0.); }
        up = vec3(0.,1.,0.);
    }
    var out: Out;
    let corner = p.position + right*offset.x + up*offset.y;
    out.position = pview.view_proj * vec4(corner,1.);
    out.view_dir = corner - pview.camera_pos;
    out.uv = p.uv_rect.xy + uv*p.uv_rect.zw;
    out.local_uv = uv; out.colour = p.color; out.tip = p.tip_tint;
    out.uv_rect = p.uv_rect; out.flutter = p.flutter;
    out.texture_index = p.texture_index; out.flags = p.flags;
    return out;
}
struct Fragment {
    @location(0) colour: vec4<f32>,
    @location(1) reactive: vec4<f32>,
}
@fragment fn fs_main(in: Out) -> Fragment {
    var sprite = vec4(1.);
    if in.texture_index >= 0 {
        // Bend the textured silhouette, with zero displacement at the wick.
        // Sampling is clamped to this atlas cell, never its neighbour.
        let height = 1. - in.local_uv.y;
        let uv = vec2(select(in.local_uv.x, 1. - in.local_uv.x, (in.flags & 4u) != 0u), in.local_uv.y);
        let warped = uv - vec2(in.flutter * height * height, 0.);
        let sample_uv = in.uv_rect.xy + clamp(warped, vec2(.001), vec2(.999)) * in.uv_rect.zw;
        sprite = textureSampleLevel(textures[in.texture_index],sprite_sampler,sample_uv,0.);
        if any(warped < vec2(0.)) || any(warped > vec2(1.)) { sprite.a = 0.; }
        // Shared standalone asset textures are RGBA8 UNORM; colour input is sRGB.
        sprite = vec4(pow(max(sprite.rgb,vec3(0.)),vec3(2.2)),sprite.a);
    } else if (in.flags & 16u) != 0u {
        // A flame's tongue (bit 4; Sunscar, 13 Nov 2026): round at its base, drawn to a point at its tip, bent by its
        // flutter with the base pinned as a textured flame's is, torn by turbulence that rises through it, its tip
        // licking up and down, white-yellow along its axis low down and orange-red at its edges. `uv_rect` carries
        // the tongue's seed and age (somnium_core particle.rs), so no two tongues burn alike.
        let seed = in.uv_rect.x;
        let age = in.uv_rect.y;
        let h = 1. - in.local_uv.y;
        let n1 = flame_noise(vec2(in.local_uv.x * 3. + seed * 7.1, h * 2.5 - age * 3.2));
        let n2 = flame_noise(vec2(in.local_uv.x * 7. - seed * 3.3, h * 5. - age * 5.7));
        let x = in.local_uv.x - .5 - in.flutter * h * h - ((n1 - .5) * .24 + (n2 - .5) * .1) * h;
        let tip = .74 + .24 * flame_noise(vec2(seed * 11. + 3., age * 4.5));
        let hh = h / tip;
        let half = .46 * sqrt(clamp(hh * 3.5, 0., 1.)) * pow(clamp((1. - hh) / .72, 0., 1.), .85) * (.8 + .4 * n1);
        // Torn at the top: the noise eats the tongue from its tip down, so it ends in wisps that break away.
        let tear = smoothstep(.3, .62, n1 * .6 + n2 * .4 + (1. - hh) * .55);
        let edge = (1. - smoothstep(half * .35, half + .001, abs(x))) * step(hh, 1.) * tear;
        let core = (1. - smoothstep(0., half * .55 + .001, abs(x))) * (1. - smoothstep(.15, .75, hh));
        let hot = mix(vec3(1., .5, .26), vec3(1.45, 1.4, 1.55), core);
        sprite = vec4(hot * (.75 + .5 * n2), edge);
    } else {
        sprite.a = 1. - smoothstep(.2,.5,distance(in.local_uv,vec2(.5)));
    }
    let alpha = clamp(in.colour.a * sprite.a,0.,1.);
    let tint = mix(vec3(1.),in.tip,smoothstep(.12,.72,1.-in.local_uv.y));
    var rgb = max(in.colour.rgb * tint * sprite.rgb,vec3(0.)) * alpha;
    // Lit (bit 3: smoke, dust, mist): the colour is an albedo, lit by the sun through a forward-scattering phase
    // and by the sky's fill (the engine's 0.08 cd/m2 per lux of sun, scaled as the IBL is), so a column of
    // chimney smoke reads in the scene's own light at any exposure (Sunscar, 6 Nov 2026).
    if (in.flags & 8u) != 0u {
        let v = normalize(in.view_dir);
        let l = normalize(sun.direction);
        let g = 0.5;
        let denom = 1. + g*g - 2.*g*dot(v, l);
        let phase = (1. - g*g) / (12.566371 * max(denom * sqrt(max(denom, 1e-6)), 1e-6));
        rgb *= sun.color * phase + sun.color * (0.08 * sun.ibl_intensity);
    }
    var out: Fragment;
    out.colour = vec4(rgb,select(alpha,0.,(in.flags & 1u) != 0u));
    // Texture alpha and scene depth decide actual visible coverage. Empty sprite
    // corners leave static history intact, including for additive flame sprites.
    // As strong as the sprite is (AMD's guidance: the alpha it is composited with), not all-or-nothing: a ground-mist
    // disc at 5% marked its whole six metres fully reactive, and under it FSR showed each raw jittered frame, so the
    // trunks shimmered and thin leaves blinked in and out (18 Oct 2026). TAA never read this, which is why it held.
    out.reactive = vec4(alpha,0.,0.,0.);
    return out;
}
