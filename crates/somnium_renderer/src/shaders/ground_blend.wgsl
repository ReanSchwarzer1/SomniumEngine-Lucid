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
    edge: vec4<f32>,    // the band's top line: its fade and its noise as shares of the reach, the joints' gain, unused
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

/// Whether the band's ground point shows on screen (1) or is hidden or off it (0), with its pixel, its distance
/// from the eye and the screen occlusion there. Seen, the band gathers the light that pixel gathered (shading.wgsl)
/// under that pixel's screen occlusion, so it is lit as the ground beside the foot is, in every weather and hour.
/// Unseen, it keeps the face's own light, opened by `ground_blend.params.w`.
var<private> ground_blend_seen: f32 = 0.0;
var<private> ground_blend_pixel: vec2<f32> = vec2<f32>(0.0);
var<private> ground_blend_dist: f32 = 0.0;
var<private> ground_blend_contact: f32 = 1.0;

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
    // under 2.35 reach measured across the ground (ragged line, crevices and joints at their most), which is under
    // 4.7 reach straight up on ground as steep as 60 degrees.
    if above > reach * 4.7 {
        return s;
    }
    // A face standing more than its reach under the ground is in front of a cut, not standing in soil: a retaining
    // wall at a scarp's toe, where the 2 m cells rise across its face. Below 0 the band would otherwise be whole.
    if above < -reach {
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
    //
    // The noise runs over the unfolded point (below), which moves along the face and out with the height, so the
    // line wanders up and down as well as along; its cells are 1.5 and 0.45 of the reach (0.45 and 0.13 m on a
    // 0.3 m wall, as before).
    let along = dot(geo_n, ground_n);
    let n_t = geo_n - ground_n * along;
    let k3 = n_t * (1.0 + along) / max(dot(n_t, n_t), 0.06);
    let k = k3.xz * ground_n.y;
    let fold = world.xz + k * max(above, 0.0);
    let ragged = ((snow_noise(fold, reach * 1.5) - 0.5) * 0.7
        + (snow_noise(fold + vec2<f32>(31.7), reach * 0.45) - 0.5) * 0.35) * ground_blend.edge.y;
    // Soil lies higher where the mesh's own surface turns away from its face: a joint between stones, a stone's
    // edge, a crack. So the line follows the masonry or the rock it climbs, stone by stone, and a stone's face stands
    // proud of it (19 Nov: a line moved by noise alone crossed stones and joints alike).
    let joint = saturate((1.0 - dot(s.normal, geo_n)) * 3.0) * ground_blend.edge.z;
    let crevice = (1.0 - micro_occlusion) * 0.5 + joint;
    // Soil is opaque, so the line is crisp: the fade takes `edge.x` of the reach (a soft fade of 0.4 showed the wall
    // through the ground like a ghost at 4 m, 19 Nov), but never less than a pixel and a half, so the line neither
    // crawls nor stairs far off.
    let line = above * ground_n.y / reach + ragged - crevice;
    let per_pixel = (abs(dwx.y) + abs(dwy.y)) * ground_n.y / reach;
    let half_fade = max(0.5 * ground_blend.edge.x, 0.75 * per_pixel);
    var w = 1.0 - smoothstep(0.8 - half_fade, 0.8 + half_fade, line);
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
    // The ground the band stands for, where it shows on screen: the unfolded point, but at least 8 cm and two pixels
    // out from the foot, so the band's lowest rows read the ground in front of them and not themselves. It counts as
    // seen when the surface at its pixel lies within 10 cm and 2% of its distance (the ground, or a blade or pebble
    // on it); a heap or a post in front of it does not.
    let probe_xz = world.xz + k * max(above, 0.08 + 2.0 * length(dwx));
    let probe = vec3<f32>(probe_xz.x, ground_y + dot(slope, probe_xz - world.xz), probe_xz.y) - view.camera_pos;
    let clip = view.view_proj * vec4<f32>(probe, 0.0) + view.view_proj * vec4<f32>(view.camera_pos, 1.0);
    if clip.w > 0.0 {
        let ndc = clip.xyz / clip.w;
        let probe_uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if all(probe_uv >= vec2<f32>(0.0)) && all(probe_uv < vec2<f32>(1.0)) {
            let dims = vec2<f32>(textureDimensions(scene_depth));
            let texel = vec2<i32>(probe_uv * dims);
            let scene_z = textureLoad(scene_depth, texel, 0);
            let there = view.inv_view_proj_rel * vec4<f32>(ndc.xy, scene_z, 1.0);
            let reach_m = length(probe);
            if scene_z > 0.0 && distance(there.xyz / there.w, probe) < 0.10 + 0.02 * reach_m {
                // What lies there. A mesh that is no leaf, card or far ground, that blends into no soil itself and
                // lies where the ground should show, is a floor: paving, a slab, a low step, a road. A foot standing
                // on it stands in no soil and takes no band, whatever height its material carries (the lead, 19 Nov:
                // "No band on paving, steps, sills or anything that stands on a mesh floor"). Nothing has been mixed
                // yet, so the surface leaves as it came, and the terrain's material is never read for it.
                let beside_vis = textureLoad(vis_buffer, texel, 0).x;
                if beside_vis != 0u {
                    let beside = materials[instances[beside_vis - 1u].material_id];
                    let beside_m = select(ground_blend.params.y, beside.ground_blend, beside.ground_blend != 0.0);
                    if beside.terrain_index < 0 && beside.alpha_cutoff <= 0.0 && (beside.flags & 118u) == 0u
                        && beside_m <= 0.0 {
                        return s;
                    }
                }
                ground_blend_seen = 1.0;
                ground_blend_pixel = probe_uv * dims;
                ground_blend_dist = reach_m;
                ground_blend_contact = textureLoad(gtao_tex, texel, 0).a;
            }
        }
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
    // The ground's sky as the terrain's own path gives it (shading.wgsl's terrain branch): its baked sky visibility
    // at the fold point, and the hue of sheltered ground there. Without them the band was bluer than the ground
    // beside it in every weather (MESH_BLENDING.md 2.3).
    let sky_vis = terrain_sky_visibility(tm, splat_uv);
    let sheltered = smoothstep(1.0, 0.88, sky_vis) * tm.macro_octave_strength.w;
    s.albedo = s.albedo * mix(vec3<f32>(1.0), vec3<f32>(0.88, 0.97, 0.78), sheltered * w);
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
    // The bake in the occlusion and in `baked_sky_vis`, as on the ground: the untraced sky takes it, and the traced
    // sky, which sees the ridges itself, takes it back out.
    let baked = mix(1.0, sky_vis, w);
    s.occlusion = s.occlusion * baked;
    baked_sky_vis = baked;
    ground_blend_w = w;
    // Seen (above), the band stands under its ground point's own screen occlusion and opens nothing by a constant.
    var open = w * ground_blend.params.w;
    var screen = mix(screen_occlusion, 1.0, open);
    if ground_blend_seen > 0.0 {
        screen = mix(screen_occlusion, ground_blend_contact, w);
        open = 0.0;
    }
    ground_blend_open = open;
    s.occlusion = s.occlusion / max(screen_occlusion, 0.004) * screen;
    screen_occlusion = screen;
    return s;
}
