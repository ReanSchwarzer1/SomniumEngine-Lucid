// Visibility Buffer - Rasterization Pass
//!include "wind.wgsl"
//!include "depth_tie.wgsl"
//!include "terrain_morph.wgsl"

// wgpu 30 requires `binding_array<...>` to be behind an explicit enable
// directive; wgpu 29 accepted it without one. Found by MORROWIND-C, because
// MORROWIND-A2 bumped wgpu to 30 and left this crate's `naga` dev-dependency
// on 29 — so the validation test was checking these files with the *old*
// front end and passed. The resolver hoists and de-duplicates `enable`
// lines, so a module that includes this one inherits it.
enable wgpu_binding_array;

struct Vertex {
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    norm_x: f32,
    norm_y: f32,
    norm_z: f32,
    u: f32,
    v: f32,
}

struct Instance {
    model: mat4x4<f32>,
    material_id: u32,
    vertex_offset: u32,
    index_offset: u32,
    _padding: u32,
}

struct View {
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view:          mat4x4<f32>,
    camera_pos:    vec3<f32>,
    _padding:      f32,
    time:          f32,
    lod_scale:     f32,  // see global_pool.wgsl
    depth_tie_slope: f32,  // pass/visibility.rs `depth_tie_slope`
    _time_pad2:    f32,
    wind:          vec4<f32>,
    wind_time:     vec4<f32>,
}

struct Material {
    base_color: vec4<f32>,
    roughness: f32,
    metallic: f32,
    albedo_map: i32,
    normal_map: i32,
    metallic_roughness_map: i32,
    alpha_cutoff: f32,
    flags: u32,
    occlusion_map: i32,
    transmission: f32,
    // Three scalars, not a vec3.
    //
    // WGSL gives vec3<f32> a 16-byte alignment, so `emissive: vec3<f32>` here
    // sat at offset 64 and rounded the struct to 96 bytes, while Rust's
    // repr(C) packs [f32; 3] at offset 52 for a total of 80. Every material
    // past index 0 was therefore read from the wrong offset: `metallic` came
    // back as garbage, and a metallic reading of ~1 zeroes kD, so the sun's
    // diffuse term vanished on those materials and only IBL remained. That is
    // why primitives looked flat and showed no shadow (there was no sun term
    // left to darken), and why foliage rendered with wrong colours -- one bug,
    // scaling with material index.
    emissive_r: f32,
    emissive_g: f32,
    emissive_b: f32,
    emissive_map: i32,
    // Phase 25A-2: slot in the terrain-material array, or -1.
    terrain_index: i32,
    _pad1: f32,
    _pad2: f32,
    // Parallax-occlusion height map and relief depth (metres); see `pool.rs`.
    height_map: i32,
    height_depth: f32,
    _hpad0: f32,
    _hpad1: f32,
    // Foliage wind response (`pool.rs` `wind`): bend, flutter; see `wind.wgsl`.
    wind_bend: f32,
    wind_flutter: f32,
    _wind_pad0: f32,
    _wind_pad1: f32,
}

@group(0) @binding(0) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read> view: View;
// Phase 17D: alpha cutout needs the albedo texture here, not just in shading.
// The visibility buffer decides what exists at each pixel, so a leaf's cut-away
// corners have to be discarded now — resolving them later would be too late,
// the depth buffer would already show a solid quad.
@group(0) @binding(4) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(5) var<storage, read> materials: array<Material>;

@group(1) @binding(0) var cutout_sampler: sampler;
/// Each instance's place when two things' faces lie in one plane (renderer.rs `depth_tie`, 29 Oct 2026). The depth
/// test cannot tell such faces apart: which one a pixel keeps comes down to rounding a few steps of the depth buffer
/// either way, and FSR's or TAA's jitter moves the sample every frame, so the winner flips and the overlap shimmers
/// (the developer: "a lot of z fighting with tables stacked", gone with FSR off, whose resolve averages it away).
/// A higher tie is drawn that many steps nearer (depth_tie.wgsl, each step sized for the plane), so the same face wins
/// every frame: smaller things over larger (a sheet of paper over its desk, a trim over its wall), then by a hash of
/// the draw. A fixed step before was too little edge-on (one step apart still fought there) and too much face-on,
/// where it could put whole layers (a painting a few mm off its wall) behind each other.
@group(1) @binding(1) var<storage, read> depth_ties: array<u32>;

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) instance_id: u32,
    // Triangle index relative to the MESH, not to the draw.
    //
    // This used to come from `@builtin(primitive_index)`, which restarts at 0
    // for every draw call. That was fine while a draw was a whole mesh, but
    // Phase 15F splits a mesh across one draw per cluster, and the shading pass
    // uses this id to fetch the triangle out of the geometry pool — a
    // per-draw id would send it to the wrong triangle in every cluster after
    // the first.
    //
    // `vertex_index` includes `first_vertex`, which for a cluster draw is that
    // cluster's index offset within the mesh, so dividing by three recovers the
    // mesh-relative triangle. All three vertices of triangle k have indices
    // 3k, 3k+1 and 3k+2, so integer division gives k for each of them and the
    // flat interpolation is well defined.
    @location(1) @interpolate(flat) prim_id: u32,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) material_id: u32,
    // How much of an instance-batch copy shows across its LOD hand-over: 1 all (no hand-over), w in 0..1 a copy
    // fading out, -w one fading in (`foliage_fade_weight`). Alpha-tested parts dither by it in `fs_main`.
    @location(4) @interpolate(flat) fade: f32,
}

// LOD hand-over of instance-batch copies (Sunscar, 6 Nov 2026; instance.rs `foliage_fade`, kept in step with
// shadow.wgsl): a copy near its band's edge is in both batches, the near one fading out across FADE_WIDTH metres as
// the far one fades in, measured from the copy's origin to the eye, horizontally, as the bands are.
const FADE_WIDTH: f32 = 8.0;

/// Whether an instance's spare word is a hand-over: an edge set (the terrain LOD debug view writes 1..5 here), not
/// the contact-shadow flag, not a terrain chunk (which always carries a vertex count in bits 4..13).
fn foliage_fade_word(packed: u32) -> bool {
    return (packed >> 14u) != 0u && packed != 0x80000000u && ((packed >> 4u) & 1023u) == 0u;
}

/// How much of the copy shows at horizontal distance `d` from the eye: 1 inside its band, 0 past the edge.
fn foliage_fade_weight(packed: u32, d: f32) -> f32 {
    let edge = f32((packed >> 14u) & 65535u) / 16.0;
    let s = smoothstep(edge - FADE_WIDTH * 0.5, edge + FADE_WIDTH * 0.5, d);
    return select(1.0 - s, s, (packed & 1u) == 1u);
}

@vertex
fn vs_main(
    @builtin(vertex_index) v_idx: u32,
    @builtin(instance_index) inst_idx: u32
) -> VertexOutput {
    var out: VertexOutput;
    
    let instance = instances[inst_idx];
    let index = indices[instance.index_offset + v_idx];
    var vertex = vertices[instance.vertex_offset + index];

    // Phase 25C: the CDLOD morph (terrain_morph.wgsl), as the shading rebuild and the shadow pass apply it.
    let packed = instance._padding;
    let pos = terrain_morph(vec3<f32>(vertex.pos_x, vertex.pos_y, vertex.pos_z), index, instance.vertex_offset,
                            packed, instance.model, view.camera_pos);

    out.uv = vec2<f32>(vertex.u, vertex.v);
    out.material_id = instance.material_id;
    var world = (instance.model * vec4<f32>(pos, 1.0)).xyz;
    // Foliage wind. Terrain (packed != 0) never sways. The same displacement
    // runs in shadow.wgsl, the shading pass's reconstruction and velocity.wgsl.
    let handover = foliage_fade_word(packed);
    if packed == 0u || handover {
        let m = materials[instance.material_id];
        world += wind_offset(world, instance.model[3].xyz, m.wind_bend, m.wind_flutter,
                             view.wind_time.x, view.wind, view.camera_pos);
    }
    out.clip_pos = depth_tie_clip(view.view_proj, world, view.camera_pos);
    // Ties between faces in one plane go the same way every frame (`depth_ties`), in steps sized for this plane.
    // Word 0 counts the instances with a tie (`set_depth_ties`); an instance batch's copies lie past them and take 0.
    var tie = 0.0;
    if inst_idx < depth_ties[0] {
        tie = f32(depth_ties[inst_idx + 1u]);
    }
    if tie > 0.0 {
        let normal = (instance.model * vec4<f32>(vertex.norm_x, vertex.norm_y, vertex.norm_z, 0.0)).xyz;
        out.clip_pos.z += depth_tie_offset(tie, normal, world, out.clip_pos.w, view.view, view.camera_pos,
                                           view.depth_tie_slope);
    }
    out.instance_id = inst_idx;
    out.prim_id = v_idx / 3u;
    out.fade = 1.0;
    if handover {
        let w = foliage_fade_weight(packed, distance(instance.model[3].xz, view.camera_pos.xz));
        // Opaque parts (bark) switch at the edge, whole: two trunks dithered into each other would only shimmer,
        // and a discard in an opaque draw is not worth it. Past the edge the copy goes outside the clip box.
        if w <= 0.0 || (materials[instance.material_id].alpha_cutoff <= 0.0 && w < 0.5) {
            out.clip_pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        } else if materials[instance.material_id].alpha_cutoff > 0.0 {
            out.fade = select(w, -w, (packed & 1u) == 1u);
        }
    }
    
    return out;
}

@fragment
fn fs_main(
    @builtin(position) frag: vec4<f32>,
    @location(0) @interpolate(flat) instance_id: u32,
    @location(1) @interpolate(flat) prim_idx: u32,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(flat) material_id: u32,
    @location(4) @interpolate(flat) fade: f32,
) -> @location(0) vec2<u32> {
    // Derivatives are taken at top level, where control flow is uniform, and
    // fed to textureSampleGrad below. Sampling inside the branch directly would
    // break WGSL's uniformity rule, and dropping to LOD 0 instead would make
    // distant foliage crawl with aliasing.
    let ddx_uv = dpdx(uv) * view.lod_scale;
    let ddy_uv = dpdy(uv) * view.lod_scale;

    // A copy across its LOD hand-over: the same per-pixel noise keeps a pixel in the fading-out copy exactly where
    // the fading-in one drops it, so neither a gap nor a double shows (interleaved gradient noise, Jimenez 2014).
    // The pattern moves every frame (about 60 steps a second of engine time) so TAA blends it: a still one showed as
    // a stipple over the trees in the band.
    if abs(fade) < 1.0 {
        let step = f32(u32(view.time * 60.0) % 64u);
        let n = fract(52.9829189 * fract(dot(frag.xy + 5.588238 * step, vec2<f32>(0.06711056, 0.00583715))));
        if (fade >= 0.0 && n >= fade) || (fade < 0.0 && n <= 1.0 + fade) {
            discard;
        }
    }

    let material = materials[material_id];
    // Cutoff is 0 for OPAQUE and BLEND, so this costs one compare for
    // everything that is not alpha-tested.
    if material.alpha_cutoff > 0.0 && material.albedo_map >= 0 {
        let alpha = textureSampleGrad(
            textures[material.albedo_map], cutout_sampler, uv, ddx_uv, ddy_uv,
        ).a;
        var cutoff = material.alpha_cutoff;
        // Hashed alpha where minified (`MATERIAL_FLAG_HASHED_ALPHA`, bare twigs; Wyman & McGuire 2017): a twig a
        // texel wide is a fraction of a pixel far off, and a fixed cut-off keeps only the few texels that clear it,
        // so a bare crown broke into dark flecks. Past 2 texels a pixel the cut-off blends toward a per-pixel noise
        // that moves every frame, so each pixel shows its share of twig and TAA turns it into a haze.
        if (material.flags & 32u) != 0u {
            let size = vec2<f32>(textureDimensions(textures[material.albedo_map]));
            let texels = max(length(ddx_uv * size), length(ddy_uv * size));
            let minified = saturate(log2(max(texels, 1.0)) - 1.0);
            if minified > 0.0 {
                let step = f32(u32(view.time * 60.0) % 64u);
                // The screen term alone gave every overlapping card at a pixel the same threshold, so forty layers of
                // 10%-covered twigs still showed 10% and a far bare willow read as dead poles (lod2, 6 Nov). A hash
                // of the card's own UV decorrelates the layers, so coverage adds up as 1 - (1 - a)^n.
                let layer = fract(sin(dot(floor(uv * 4096.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
                let n = fract(layer + 52.9829189 * fract(dot(frag.xy + 5.588238 * step + vec2<f32>(13.0, 29.0),
                                                             vec2<f32>(0.06711056, 0.00583715))));
                cutoff = mix(cutoff, clamp(n, 0.02, 0.98), minified);
            }
        }
        if alpha < cutoff {
            discard;
        }
    }

    // Phase 15C: 16/16 split — 65 535 instances x 65 536 triangles per draw.
    // Was 10/22, which capped the whole scene at 1022 draws; triangle counts
    // that large belong in separate draws (and will become meshlets in 15D).
    //
    // Add 1 so 0 stays reserved as the sky/background sentinel (the vis buffer
    // clears to 0). Instance 0 therefore encodes as 0x00010000, never 0.
    // Separate channels: no packing, so no cap on either field. `+ 1` keeps 0
    // free as the "nothing here" value the shading pass tests for.
    return vec2<u32>(instance_id + 1u, prim_idx);
}
