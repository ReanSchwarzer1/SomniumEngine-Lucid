// Phase 25C: the CDLOD vertex morph, in one place so every pass that draws or rebuilds a terrain chunk sees the same
// ground. The visibility pass morphed alone, so the shading pass rebuilt the unmorphed triangle under a morphed pixel
// and the shadow pass drew the unmorphed chunk (Sunscar, 5 Nov 2026, turning the morph on).
//
// `packed` is the chunk's Instance._padding: lod 4 bits, verts 9, on 1, lod_base 8, morph_start 10 (renderer.rs, the
// terrain draw). `pos` is the vertex's chunk-local position and `index` its index in the chunk; `vertices` is the
// including module's vertex array. Off (the bit clear, or anything not a terrain chunk) returns `pos` unchanged.
fn terrain_morph(
    pos: vec3<f32>,
    index: u32,
    vertex_offset: u32,
    packed: u32,
    model: mat4x4<f32>,
    camera: vec3<f32>,
) -> vec3<f32> {
    let lod = packed & 15u;
    let verts = (packed >> 4u) & 511u;
    if ((packed >> 13u) & 1u) == 0u || verts <= 1u {
        return pos;
    }
    let lod_base = max(f32((packed >> 14u) & 255u), 1.0);
    let morph_start_frac = f32((packed >> 22u) & 1023u) / 1023.0;
    let world = (model * vec4<f32>(pos, 1.0)).xyz;
    let dist = length(world - camera);
    let range_start = lod_base * exp2(f32(lod));
    let range_end = lod_base * exp2(f32(lod + 1u));
    let morph_start = mix(range_start, range_end, morph_start_frac);
    let morph_k = saturate((dist - morph_start) / max(range_end - morph_start, 1.0));
    if morph_k <= 0.0 {
        return pos;
    }
    let cell = vertices[vertex_offset + 1u].pos_x - vertices[vertex_offset].pos_x;
    let next_cell = cell * f32(1u << (lod + 1u));
    let snapped = floor(pos.xz / next_cell + 0.5) * next_cell;
    let step = 1u << (lod + 1u);
    let vx = index % verts;
    let vz = index / verts;
    let i_c = (vz / step) * step * verts + (vx / step) * step;
    let coarse = vertices[vertex_offset + i_c];
    return vec3<f32>(mix(pos.x, snapped.x, morph_k), mix(pos.y, coarse.pos_y, morph_k), mix(pos.z, snapped.y, morph_k));
}
