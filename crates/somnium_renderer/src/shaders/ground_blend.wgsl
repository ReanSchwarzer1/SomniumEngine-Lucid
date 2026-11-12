// Meshes blending into the ground they stand on (sunscar-photoreal, 11 Nov 2026). Included by shading.wgsl, which calls
// `apply_ground_blend` once per opaque mesh surface (never terrain) whose blend height is above zero: the material's own
// (`Material::ground_blend`, glTF extra `somnium_ground_blend`, metres; negative never blends) or, where it says
// nothing and the mesh does not sway, the scene's (`ground_blend.params.y`, PostProcess `ground_blend_m`; 0 is off),
// times the instance's scale.
//
// Within that height of the terrain under the pixel, the mesh's colour, roughness, occlusion and normal cross-fade to
// the terrain's own surface, on a ragged line (height plus noise): a boulder, a wall foot or a rubble heap sits in the
// ground instead of on it. The terrain under a pixel comes from binding 36, the live terrains of the frame: each one's
// world rectangle, the bindless id of its heights (R32Float metres above its origin, one texel per vertex) and its
// terrain index.

struct GroundTerrain {
    rect: vec4<f32>,  // world x, z of its corner, 1 / its size along x, z
    info: vec4<f32>,  // its origin's height, heights' texture id, terrain index, vertices along x
    grid: vec4<f32>,  // vertices along z, unused
}
struct GroundBlendList {
    params: vec4<f32>,  // live terrains in the list, the scene's blend height m, every height's scale, open
    terrains: array<GroundTerrain, 32>,
}
@group(1) @binding(36) var<storage, read> ground_blend: GroundBlendList;

/// How far this pixel's screen-space occlusion and traced share of the sky open toward the ground's: the band's weight
/// times `ground_blend.params.w`, 0 off the band. Both were measured over the mesh's own hemisphere, which counts the
/// ground below the band as an occluder, while the band shades as that ground (the lead, 11 Nov: the band read darker
/// than the ground beside it). `gathered_light` opens the traced share by it.
var<private> ground_blend_open: f32 = 0.0;

/// The band's weight at this pixel (0 off the band): the traced reflection was traced off the mesh's own face, a wall's
/// mirror where the band shades as ground, so the band takes the sky's reflection along its own normal instead.
var<private> ground_blend_w: f32 = 0.0;

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
    let dwx = dpdx(world);
    let dwy = dpdy(world);
    let found = ground_terrain_at(world.xz);
    if found < 0 {
        return s;
    }
    let t = ground_blend.terrains[found];
    let uv = (world.xz - t.rect.xy) * t.rect.zw;
    let ground_y = ground_height(t, uv);
    let above = world.y - ground_y;
    let reach = blend_m * (1.0 - 0.4 * saturate(geo_n.y));
    // Most of a mesh stands far above its band: out before the slope's four reads and the noise. The band's top lies
    // under 2.03 reach measured across the ground (ragged line and crevices at their most), which is under 4.1 reach
    // straight up on ground as steep as 60 degrees.
    if above > reach * 4.1 {
        return s;
    }
    // The ground's slope and normal, by central differences one vertex apart.
    let step = vec2<f32>(1.0) / (vec2<f32>(t.info.w, t.grid.x) - 1.0);
    let span = step / t.rect.zw;
    let hx = ground_height(t, uv + vec2<f32>(step.x, 0.0)) - ground_height(t, uv - vec2<f32>(step.x, 0.0));
    let hz = ground_height(t, uv + vec2<f32>(0.0, step.y)) - ground_height(t, uv - vec2<f32>(0.0, step.y));
    let slope = vec2<f32>(hx / (2.0 * span.x), hz / (2.0 * span.y));
    let ground_n = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    // How high the pixel stands, measured across the ground rather than straight up (on a slope the vertical overstates
    // it), against the reach: lower on faces turned up, which would otherwise read as buried, and the line rises in the
    // mesh's own crevices (its material occlusion), where soil collects first. The edge is ragged: two octaves of noise
    // over the ground move the line by up to half the reach either way.
    let ragged = (snow_noise(world.xz, 0.45) - 0.5) * 0.7 + (snow_noise(world.xz + vec2<f32>(31.7), 0.13) - 0.5) * 0.35;
    let crevice = (1.0 - micro_occlusion) * 0.5;
    var w = 1.0 - smoothstep(0.35, 1.0, above * ground_n.y / reach + ragged - crevice);
    // A face lying along the ground (a floor, paving, a road) is ground already: only faces meeting it at an angle
    // (a boulder's flank, a wall foot, a heap's sides) take the ground's surface.
    w *= 1.0 - smoothstep(0.9, 0.985, dot(geo_n, ground_n));
    // Nothing stands on a cliff: on ground steeper than 45 degrees the band fades, and past 60 it is gone. Measured
    // across the ground, a shell cladding a canyon wall stood within its reach of the wall everywhere and wore a pale
    // fringe of the wall's own surface round every edge (gx4, the Vela's floor, 12 Nov).
    w *= smoothstep(0.5, 0.7, ground_n.y);
    if w <= 0.0 {
        return s;
    }
    // The face unfolded onto the ground: a point some way up it reads the ground as far out from its foot as it stands
    // along the face. So the band wears the terrain's own surface as the terrain shows it beside the foot (its layers,
    // far colour, octaves and wetness, at its own scale), meets it on the same texel at the foot, and carries no
    // vertical streaks, which the ground sampled straight down drew up every wall (11 Nov). The unfolding runs in the
    // ground's own plane: up the face by the height over the ground is out along the face's normal as it lies in that
    // plane, less the face's own run, k = n_t (1 + n.g) / |n_t|^2 with n_t the normal less its part along the ground's,
    // clamped where the face lies nearly along the ground, which the line above has faded out already. On level ground
    // that is out along the face's horizontal normal; on a wall it stays on the wall, where folding out level reached
    // the canyon's floor or bench and fringed a rock shell with their colour (12 Nov). The footprint follows by the
    // chain rule, the ground's height moving with the slope.
    let along = dot(geo_n, ground_n);
    let n_t = geo_n - ground_n * along;
    let k3 = n_t * (1.0 + along) / max(dot(n_t, n_t), 0.06);
    let k = k3.xz * ground_n.y;
    let fold = world.xz + k * max(above, 0.0);
    let fold_dx = dwx.xz + k * (dwx.y - dot(slope, dwx.xz));
    let fold_dy = dwy.xz + k * (dwy.y - dot(slope, dwy.xz));
    // The terrain's own surface there, by the path the terrain itself is shaded with.
    let tm_idx = u32(t.info.z);
    let tm = terrain_materials[tm_idx];
    let ground_pos = vec3<f32>(fold.x, ground_y + dot(slope, fold - world.xz), fold.y);
    let terrain_xz = fold - tm.terrain_origin;
    let splat_uv = terrain_xz * tm.inv_world_size;
    var ground: TerrainSurface;
    if !enable_live_terrain || (enable_clipmap && tm.clipmap_enabled != 0u) {
        ground = evaluate_clipmap_material(tm, tm_idx, ground_pos, ground_n, splat_uv, fold_dx, fold_dy);
    } else {
        ground = evaluate_terrain_material(tm_idx, ground_pos, terrain_xz, ground_n, splat_uv, fold_dx, fold_dy);
    }
    s.albedo = mix(s.albedo, ground.albedo, w);
    s.roughness = mix(s.roughness, ground.roughness, w);
    s.metallic = mix(s.metallic, 0.0, w);
    // Lit as the ground where it is the ground, so the foot meets the terrain without a step in the light; the sky is
    // gathered along the ground's normal there too.
    s.normal = normalize(mix(s.normal, ground.normal, w));
    s.bent_normal = normalize(mix(s.bent_normal, ground_n, w));
    // The ground's own occlusion in place of the mesh's, the screen's term kept, as paint swaps it: a scan's baked
    // contact shadow at its foot is the shadow of the ground it was scanned on. Multiplied together, the two put the
    // band below both sides (the lead, 11 Nov: 0.043 between a wall at 0.213 and ground at 0.090).
    let occlusion = mix(micro_occlusion, ground.occlusion, w);
    s.occlusion = s.occlusion / max(micro_occlusion, 0.05) * occlusion;
    micro_occlusion = occlusion;
    let open = w * ground_blend.params.w;
    ground_blend_open = open;
    ground_blend_w = w;
    let screen = mix(screen_occlusion, 1.0, open);
    s.occlusion = s.occlusion / max(screen_occlusion, 0.004) * screen;
    screen_occlusion = screen;
    return s;
}
