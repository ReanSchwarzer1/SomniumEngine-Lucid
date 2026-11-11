// Meshes blending into the ground they stand on (sunscar-photoreal, 11 Nov 2026). Included by shading.wgsl, which calls
// `apply_ground_blend` once per opaque mesh surface (never terrain) whose blend height is above zero: the material's own
// (`Material::ground_blend`, glTF extra `somnium_ground_blend`, metres; negative never blends) or, where it says
// nothing, the scene's (`ground_blend.params.y`, PostProcess `ground_blend_m`; 0 is off), times the instance's scale.
//
// Within that height of the terrain under the pixel, the mesh's colour, roughness and normal cross-fade to the terrain's
// own surface at that point, on a ragged line (height plus noise), and the normal bends to the ground's: a boulder, a
// wall foot or a rubble heap sits in the ground instead of on it. The terrain under a pixel comes from binding 36, the
// live terrains of the frame: each one's world rectangle, the bindless id of its heights (R32Float metres above its
// origin, one texel per vertex) and its terrain index.

struct GroundTerrain {
    rect: vec4<f32>,  // world x, z of its corner, 1 / its size along x, z
    info: vec4<f32>,  // its origin's height, heights' texture id, terrain index, vertices along x
    grid: vec4<f32>,  // vertices along z, unused
}
struct GroundBlendList {
    params: vec4<f32>,  // live terrains in the list, the scene's blend height m, unused
    terrains: array<GroundTerrain, 32>,
}
@group(1) @binding(36) var<storage, read> ground_blend: GroundBlendList;

/// The list entry whose rectangle holds `xz`, or -1.
fn ground_terrain_at(xz: vec2<f32>) -> i32 {
    let count = min(u32(ground_blend.params.x), 32u);
    for (var i = 0u; i < count; i++) {
        let t = ground_blend.terrains[i];
        let uv = (xz - t.rect.xy) * t.rect.zw;
        if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0)) {
            return i32(i);
        }
    }
    return -1;
}

/// The ground's height at `uv` (0..1 over the terrain), bilinear between vertices as the mesh is.
fn ground_height(t: GroundTerrain, uv: vec2<f32>) -> f32 {
    let n = vec2<f32>(t.info.w, t.grid.x);
    let st = (clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)) * (n - 1.0) + 0.5) / n;
    return t.info.x + textureSampleLevel(textures[i32(t.info.y)], default_sampler, st, 0.0).r;
}

/// Cross-fade a finished mesh surface to the terrain's within `blend_m` of the ground.
fn apply_ground_blend(surface_in: Surface, world: vec3<f32>, geo_n: vec3<f32>, blend_m: f32) -> Surface {
    var s = surface_in;
    // Taken before any pixel of the quad leaves, so the ground's texture footprint stays defined at the band's edge.
    let ddx = dpdx(world.xz);
    let ddy = dpdy(world.xz);
    let found = ground_terrain_at(world.xz);
    if found < 0 {
        return s;
    }
    let t = ground_blend.terrains[found];
    let uv = (world.xz - t.rect.xy) * t.rect.zw;
    let ground_y = ground_height(t, uv);
    // Soil banks higher against faces turned up than against walls, and its edge is ragged: two octaves of noise over
    // the ground move the line by up to half the blend height either way.
    let reach = blend_m * mix(0.7, 1.25, saturate(geo_n.y * 0.5 + 0.5));
    let ragged = (snow_noise(world.xz, 0.45) - 0.5) * 0.7 + (snow_noise(world.xz + vec2<f32>(31.7), 0.13) - 0.5) * 0.35;
    var w = 1.0 - smoothstep(0.35, 1.0, (world.y - ground_y) / reach + ragged);
    if w <= 0.0 {
        return s;
    }
    // The ground's normal, by central differences one vertex apart.
    let step = vec2<f32>(1.0) / (vec2<f32>(t.info.w, t.grid.x) - 1.0);
    let span = step / t.rect.zw;
    let hx = ground_height(t, uv + vec2<f32>(step.x, 0.0)) - ground_height(t, uv - vec2<f32>(step.x, 0.0));
    let hz = ground_height(t, uv + vec2<f32>(0.0, step.y)) - ground_height(t, uv - vec2<f32>(0.0, step.y));
    let ground_n = normalize(vec3<f32>(-hx / (2.0 * span.x), 1.0, -hz / (2.0 * span.y)));
    // A face lying along the ground (a floor, paving, a road) is ground already: only faces meeting it at an angle
    // (a boulder's flank, a wall foot, a heap's sides) take the ground's surface.
    w *= smoothstep(0.985, 0.9, dot(geo_n, ground_n));
    if w <= 0.0 {
        return s;
    }
    // The terrain's own surface at this point, by the path the terrain itself is shaded with.
    let tm_idx = u32(t.info.z);
    let tm = terrain_materials[tm_idx];
    let ground_pos = vec3<f32>(world.x, ground_y, world.z);
    let terrain_xz = world.xz - tm.terrain_origin;
    let splat_uv = terrain_xz * tm.inv_world_size;
    var ground: TerrainSurface;
    if !enable_live_terrain || (enable_clipmap && tm.clipmap_enabled != 0u) {
        ground = evaluate_clipmap_material(tm, tm_idx, ground_pos, ground_n, splat_uv, ddx, ddy);
    } else {
        ground = evaluate_terrain_material(tm_idx, ground_pos, terrain_xz, ground_n, splat_uv, ddx, ddy);
    }
    s.albedo = mix(s.albedo, ground.albedo, w);
    s.roughness = mix(s.roughness, ground.roughness, w);
    s.metallic = mix(s.metallic, 0.0, w);
    s.normal = normalize(mix(s.normal, ground.normal, w));
    s.bent_normal = normalize(mix(s.bent_normal, ground_n, w));
    s.occlusion = mix(s.occlusion, s.occlusion * ground.occlusion, w);
    return s;
}
