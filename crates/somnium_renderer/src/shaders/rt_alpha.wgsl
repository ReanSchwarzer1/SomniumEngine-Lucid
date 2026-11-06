// Cut-out cards for ray queries (Sunscar, 6 Nov 2026). Every BLAS is built opaque, so a ray that must see through a
// leaf card's holes asks for triangle candidates (ForceNonOpaque) and commits only the ones this calls covered. Used
// by `rt_hit.wgsl` (GI, water, the path tracer) and `restir_di.wgsl` (the traced sun). Needs the global pool
// (`instances`, `materials`, `vertices`, `indices`, `textures`) and a `default_sampler` from the includer.

/// Whether a candidate hit lands on a covered texel: always, unless its material is cut out (`alpha_cutoff`).
fn rt_candidate_is_covered(isect: RayIntersection) -> bool {
    let inst = instances[isect.instance_custom_data];
    let mat = materials[inst.material_id];
    if mat.terrain_index >= 0 || mat.alpha_cutoff <= 0.0 || mat.albedo_map < 0 {
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
