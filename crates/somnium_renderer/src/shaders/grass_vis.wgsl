// Somnium Engine — the GPU grass draw into the visibility buffer (Sunscar, 19 Nov 2026; `grass.rs`).
//
// One indirect draw per class (3, 2 or 1 segments): an instance per blade grass_gen.wgsl kept, 3 vertices per
// triangle, no index or vertex buffer. Drawn after the frame's final Hi-Z, into the same visibility buffer and depth
// (loaded, never cleared), two-sided and opaque: no alpha test, so early depth holds.

//!include "grass_pool.wgsl"

struct GrassDraw {
    /// The class's first slot in the span.
    slot_base: u32,
    capacity: u32,
    /// The grass instance this frame (`instances`).
    instance: u32,
    _pad: u32,
}

@group(1) @binding(0) var<uniform> draw: GrassDraw;

struct GrassOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) prim: u32,
}

@vertex
fn vs_grass(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> GrassOut {
    var out: GrassOut;
    out.prim = 0u;
    // Past the class's capacity (its counter ran over): nothing, all three corners at w = 0.
    if ii >= draw.capacity {
        out.clip = vec4<f32>(0.0);
        return out;
    }
    let slot = draw.slot_base + ii;
    let inst = instances[draw.instance];
    let m = materials[inst.material_id];
    let g = grass_load(inst.vertex_offset + slot);
    let tri = vi / 3u;
    let q = grass_strip_vertex(g, tri + vi % 3u, view.wind_time.x, view.wind, view.camera_pos, m.wind_bend,
                               m.wind_flutter);
    // Camera-relative, as the visibility pass projects (depth_tie.wgsl `depth_tie_clip`).
    let eye_clip = view.view_proj * vec4<f32>(view.camera_pos, 1.0);
    out.clip = view.view_proj * vec4<f32>(q.pos - view.camera_pos, 0.0) + eye_clip;
    out.prim = (slot << GRASS_TRI_BITS) | tri;
    return out;
}

@fragment
fn fs_grass(in: GrassOut) -> @location(0) vec2<u32> {
    return vec2<u32>(draw.instance + 1u, in.prim);
}
