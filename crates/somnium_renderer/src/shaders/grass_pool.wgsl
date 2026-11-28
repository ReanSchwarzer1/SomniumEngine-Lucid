// Somnium Engine — GPU grass blades read back through the shared pool (Sunscar, 19 Nov 2026; `grass.rs`).
//
// For the passes that bind `global_pool.wgsl`: the grass instance's material carries `MATERIAL_GRASS`, its
// `vertex_offset` is the blades' span, and the visibility buffer's primitive word is (slot << 3 | triangle).

//!include "global_pool.wgsl"
//!include "grass_blade.wgsl"

/// `material/pool.rs` `MATERIAL_FLAG_GRASS`.
const MATERIAL_GRASS: u32 = 128u;

fn grass_material(m: Material) -> bool {
    return (m.flags & MATERIAL_GRASS) != 0u;
}

fn grass_load(index: u32) -> GrassBlade {
    let v = vertices[index];
    return grass_decode(vec4<f32>(v.pos_x, v.pos_y, v.pos_z, v.norm_x), vec4<f32>(v.norm_y, v.norm_z, v.u, v.v));
}

struct GrassTri {
    v: array<Vertex, 3>,
    colour: vec3<f32>,
    root_ao: f32,
    root: vec3<f32>,
    wind: vec2<f32>,
}

/// The triangle the grass draw rasterised for visibility word `prim` of the grass instance (`base` its
/// `vertex_offset`), in world space with the wind at `clock`: what `indices` and `vertices` give any other mesh.
fn grass_pool_tri(base: u32, prim: u32, m: Material, clock: f32) -> GrassTri {
    let g = grass_load(base + (prim >> GRASS_TRI_BITS));
    let k = prim & ((1u << GRASS_TRI_BITS) - 1u);
    var out: GrassTri;
    for (var c = 0u; c < 3u; c++) {
        let q = grass_corner(g, k, c, clock, view.wind, view.camera_pos, m.wind_bend, m.wind_flutter);
        out.v[c] = Vertex(q.pos.x, q.pos.y, q.pos.z, q.normal.x, q.normal.y, q.normal.z, q.uv.x, q.uv.y);
    }
    out.colour = g.colour;
    out.root_ao = g.root_ao;
    out.root = g.root;
    out.wind = grass_wind(g, m.wind_bend, m.wind_flutter);
    return out;
}

/// A visibility-buffer triangle's corners in world space, as ReSTIR GI and lighting extra rebuild one for its normal:
/// a mesh's own (no wind, no morph, as they always took it), or a grass blade's.
fn vis_triangle_world(inst: Instance, prim: u32) -> array<vec3<f32>, 3> {
    let m = materials[inst.material_id];
    var out: array<vec3<f32>, 3>;
    if grass_material(m) {
        let g = grass_pool_tri(inst.vertex_offset, prim, m, view.wind_time.x);
        out[0] = vec3<f32>(g.v[0].pos_x, g.v[0].pos_y, g.v[0].pos_z);
        out[1] = vec3<f32>(g.v[1].pos_x, g.v[1].pos_y, g.v[1].pos_z);
        out[2] = vec3<f32>(g.v[2].pos_x, g.v[2].pos_y, g.v[2].pos_z);
        return out;
    }
    let base = inst.index_offset + prim * 3u;
    for (var c = 0u; c < 3u; c++) {
        let v = vertices[inst.vertex_offset + indices[base + c]];
        out[c] = (inst.model * vec4<f32>(v.pos_x, v.pos_y, v.pos_z, 1.0)).xyz;
    }
    return out;
}
