// Somnium Engine — the ground's bare share (Sunscar, 19 Nov 2026; `pass/cover_census.rs`, SOMNIUM_COVER_CENSUS=1).
//
// Of the pixels within 10 m and within 10-30 m of the eye: terrain where grass can grow (the bare ground between
// plants), GPU grass, ground cover (any cut-out foliage below the eye less 0.3 m, so tree crowns are left out),
// everything else, and terrain where no grass grows (paths, rock: under 0.35 grassland weight in the grass pass's
// habitat map). The bare share is the first / (first + grass + cover). Reads the finished visibility buffer.

//!include "grass_pool.wgsl"

struct CoverParams {
    /// xy the buffer's size, z the grass pass's ground slots to look in (0: none, all terrain counts).
    size: vec4<u32>,
}

struct GrassTerrain {
    origin: vec2<f32>,
    cell: f32,
    verts: u32,
    height_off: u32,
    hab_off: u32,
    texels: u32,
    texel_size: f32,
}

@group(1) @binding(0) var vis_buffer: texture_2d<u32>;
@group(1) @binding(1) var depth_tex: texture_depth_2d;
@group(1) @binding(2) var<storage, read_write> counts: array<atomic<u32>, 10>;
@group(1) @binding(3) var<uniform> cp: CoverParams;
@group(1) @binding(4) var<storage, read> terrains: array<GrassTerrain>;
@group(1) @binding(5) var<storage, read> habitat_map: array<u32>;

var<workgroup> local_counts: array<atomic<u32>, 10>;

/// Whether grass can grow at world `xz`: its texel's grassland weight reaches the grass's presence threshold.
/// Outside every ground slot (or with no grass pass), yes: every terrain pixel counts, as before.
fn grassland(xz: vec2<f32>) -> bool {
    for (var i = 0u; i < cp.size.z; i++) {
        let t = terrains[i];
        if t.texels == 0u {
            continue;
        }
        let f = (xz - t.origin) / t.texel_size;
        if any(f < vec2<f32>(0.0)) || any(f >= vec2<f32>(f32(t.texels))) {
            continue;
        }
        let tex = vec2<u32>(f);
        let w = habitat_map[t.hab_off + tex.y * t.texels + tex.x];
        return f32((w >> 5u) & 255u) + f32((w >> 18u) & 255u) >= 0.35 * 255.0;
    }
    return true;
}

@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane < 10u {
        atomicStore(&local_counts[lane], 0u);
    }
    workgroupBarrier();
    // Every thread reaches both barriers: no early return.
    if gid.x < cp.size.x && gid.y < cp.size.y {
        let coord = vec2<i32>(gid.xy);
        let vis = textureLoad(vis_buffer, coord, 0).xy;
        let depth = textureLoad(depth_tex, coord, 0);
        if vis.x != 0u && depth > 0.0 {
            let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(cp.size.xy);
            let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
            // Camera-relative (pixel_class.wgsl `pc_world_at`).
            let h = view.inv_view_proj_rel * ndc;
            let rel = h.xyz / h.w;
            let d = length(rel);
            if d < 30.0 {
                let m = materials[instances[vis.x - 1u].material_id];
                var kind = 3u;
                if m.terrain_index >= 0 {
                    kind = select(4u, 0u, grassland(rel.xz + view.camera_pos.xz));
                } else if grass_material(m) {
                    kind = 1u;
                } else if m.alpha_cutoff > 0.0 && rel.y < -0.3 {
                    kind = 2u;
                }
                atomicAdd(&local_counts[select(5u, 0u, d < 10.0) + kind], 1u);
            }
        }
    }
    workgroupBarrier();
    if lane < 10u {
        let v = atomicLoad(&local_counts[lane]);
        if v > 0u {
            atomicAdd(&counts[lane], v);
        }
    }
}
