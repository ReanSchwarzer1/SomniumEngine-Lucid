// Cut-out cards for ray queries (Sunscar, 6 Nov 2026). Every BLAS is built opaque, so a ray that must see through a
// leaf card's holes asks for triangle candidates (ForceNonOpaque) and commits only the ones this calls covered. Used
// by `rt_hit.wgsl` (GI, water, the path tracer) and `restir_di.wgsl` (the traced sun). Needs the global pool
// (`instances`, `materials`, `vertices`, `indices`, `textures`) and a `default_sampler` from the includer.

// A ray that starts on the ground (Sunscar, 11 Nov 2026).
//
// The ground is drawn at a coarser level of detail with distance and traced at its finest, so a point on the drawn
// ground can lie under the traced one. A ray from it then starts inside the hill and meets the ground's own
// underside: GI's rays did, and every far slope and every crest wore rows of dark blobs as far apart as the coarse
// level's vertices (the developer's video: the hill behind Monargent). Lifting the start does not cure it, since how
// far under is anyone's guess on a sharp crest. But a ray that starts on a heightfield can only ever meet its top,
// so for such a ray a hit on the ground's underside is passed through, and that is exact.
//
// The tracer says so here before it traces, and takes it back after. Off, nothing changes for anyone.
/// The ray being traced started on the ground: hits on the ground's underside are passed through.
var<private> rt_ground_origin: bool = false;
/// That ray's direction, in the world.
var<private> rt_ray_dir: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

/// Whether a candidate hit on the ground is on its underside, for the ray `rt_ray_dir`.
fn rt_ground_underside(inst: Instance, isect: RayIntersection) -> bool {
    let base = inst.index_offset + isect.primitive_index * 3u;
    let v0 = vertices[inst.vertex_offset + indices[base]];
    let v1 = vertices[inst.vertex_offset + indices[base + 1u]];
    let v2 = vertices[inst.vertex_offset + indices[base + 2u]];
    let p0 = vec3<f32>(v0.pos_x, v0.pos_y, v0.pos_z);
    // The triangle's own normal, turned to face up whichever way it is wound: the ground's top is its upper side.
    var up = (inst.model * vec4<f32>(cross(vec3<f32>(v1.pos_x, v1.pos_y, v1.pos_z) - p0,
        vec3<f32>(v2.pos_x, v2.pos_y, v2.pos_z) - p0), 0.0)).xyz;
    if up.y < 0.0 {
        up = -up;
    }
    return dot(rt_ray_dir, up) > 0.0;
}

/// Whether a candidate hit lands on a covered texel: always, unless its material is cut out (`alpha_cutoff`), or it
/// is the ground's underside met by a ray that started on the ground (`rt_ground_origin`).
fn rt_candidate_is_covered(isect: RayIntersection) -> bool {
    let inst = instances[isect.instance_custom_data];
    let mat = materials[inst.material_id];
    if mat.terrain_index >= 0 {
        return !(rt_ground_origin && rt_ground_underside(inst, isect));
    }
    if mat.alpha_cutoff <= 0.0 || mat.albedo_map < 0 {
        return true;
    }
    let base = inst.index_offset + isect.primitive_index * 3u;
    let v0 = vertices[inst.vertex_offset + indices[base]];
    let v1 = vertices[inst.vertex_offset + indices[base + 1u]];
    let v2 = vertices[inst.vertex_offset + indices[base + 2u]];
    let b = isect.barycentrics;
    let uv = vec2<f32>(v0.u, v0.v) * (1.0 - b.x - b.y)
        + vec2<f32>(v1.u, v1.v) * b.x + vec2<f32>(v2.u, v2.v) * b.y;
    return textureSampleLevel(textures[mat.albedo_map], default_sampler, uv, 0.0).a >= mat.alpha_cutoff;
}
