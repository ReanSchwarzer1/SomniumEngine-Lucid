//!include "spot_cookie.wgsl"
// MORROWIND-C: composition is declared here rather than assembled by a
// `format!` of `include_str!` calls at this pass's construction site. The
// resolver (`somnium_shader`) emits each module once, in this order, and
// hoists every `enable` above everything.
//!include "rt_hit.wgsl"
//!include "global_pool.wgsl"
//!include "brdf.wgsl"
//!include "sampling.wgsl"
//!include "atmosphere.wgsl"
//!include "terrain_splat_core.wgsl"

enable wgpu_ray_query;

// Phase 24L: ReSTIR GI — ray-traced indirect diffuse.
//
// 24K resampled *direct* light: which of the sun's samples this pixel can see.
// This resamples the other half of the rendering equation — light that reached
// the pixel by bouncing off something else. It is what turns a constant ambient
// term into real coloured bounce: a red wall reddening the floor beside it, a
// hillside darkening the valley it overhangs, light spilling through an opening
// and falling off with distance. None of it baked.
//
// The estimator is the same one 24K used, applied to a different sample space.
// A DI reservoir holds a *direction to a light*. A GI reservoir holds a **point
// in the world** — where the ray landed, its normal, and the radiance leaving
// it toward us. That difference is the whole of ReSTIR GI, and it is why a
// neighbour's sample can be reused at all: two pixels a few centimetres apart
// see the same lit patch of world from slightly different angles, and the
// Jacobian below converts between those angles.
//
// # Reference
//
// `bevy_solari/src/realtime/restir_gi.wgsl` (`example_repo/bevy/bevy-main/`) —
// the reservoir contents, the pairwise MIS with the balance heuristic, the
// reconnection-shift Jacobian and its rejection threshold, and the two-pass
// split of (initial + temporal) then (spatial + shade). Bevy's version queries
// a world cache for the radiance at the sample point; this one lights the
// sample point directly from the sun, which is the `NO_WORLD_CACHE` path it
// also carries.
//
// Concatenated after `brdf.wgsl`, `sampling.wgsl`, `atmosphere.wgsl`, and the
// bounded `terrain_splat_core.wgsl`, and it binds the same
// `@group(0)` global pool the shading pass uses — so a ray hit resolves to
// geometry and material through the *same* `instances` array the visibility
// buffer resolves through, rather than through a second scene description that
// could disagree with it.

struct GiParams {
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    frame: u32,
    inv_resolution: vec2<f32>,
    /// Zero when history must be ignored (camera cut, resize, first frame).
    history_valid: f32,
    /// Scales the final indirect term. 0 disables the contribution without
    /// disabling the pass, which is what the A/B needs.
    intensity: f32,
    /// Metres beyond which an indirect ray is not worth tracing.
    max_distance: f32,
    // Three scalars, deliberately not a `vec3<f32>`: a vec3 aligns to 16, so it
    // would sit at offset 112 and round the struct to 128 against Rust's 112 —
    // which is exactly what wgpu rejected the first time this pass dispatched.
    // Same trap as `TerrainMaterial`'s trailing pad, one struct further on.
    /// GI grid step in full-resolution pixels (1 full, 2 half). Each GI texel
    /// owns a scale x scale block and traces one of its pixels, turning per frame.
    scale: f32,
    /// Most confidence a reservoir may carry in (`HISTORY_M_CAP*` in
    /// restir_gi.rs): lower while the lights flicker or move, instead of
    /// discarding history outright.
    history_m_cap: f32,
    _pad2: f32,
}

/// The full-resolution pixel a GI texel traces this frame.
fn gi_full_coord(g: vec2<u32>, full: vec2<u32>) -> vec2<i32> {
    let s = max(u32(gi.scale), 1u);
    let o = vec2<u32>(gi.frame % s, (gi.frame / s) % s);
    return vec2<i32>(min(g * s + o, full - vec2<u32>(1u)));
}

fn gi_grid(full: vec2<u32>) -> vec2<u32> {
    let s = max(u32(gi.scale), 1u);
    return (full + vec2<u32>(s - 1u)) / s;
}

/// A reservoir over *sample points* rather than over light directions.
///
/// 48 bytes. `radiance` is what leaves the sample point toward the shading
/// point; `w` is the unbiased contribution weight that lets the single kept
/// sample stand in for every candidate it beat; `m` is the confidence weight,
/// capped so the reservoir keeps responding to change.
struct GiReservoir {
    sample_pos: vec3<f32>,
    w_sum: f32,
    sample_normal: vec3<f32>,
    w: f32,
    radiance: vec3<f32>,
    m: f32,
}

@group(1) @binding(0) var accel:      acceleration_structure;
@group(1) @binding(1) var depth_tex:  texture_depth_2d;
@group(1) @binding(2) var vis_tex:    texture_2d<u32>;
@group(1) @binding(3) var out_tex:    texture_storage_2d<rgba16float, write>;
@group(1) @binding(4) var<uniform> gi: GiParams;
// Two buffers with fixed *roles*, not a ping-pong pair. `gi_a` holds the
// finished reservoir of the previous frame and is what pass 2 writes; `gi_b` is
// the handoff between the two passes. Roles rather than alternating ownership
// because pass 2 reads its neighbours' reservoirs: if it read and wrote the
// same buffer, a neighbour already processed this dispatch would hand back a
// reservoir that had been spatially resampled and shadowed, which is both a
// data race and a double-count.
// The terrain material and the albedo lookups need a sampler. Declared here
// rather than borrowed from the shading pass's group 1: this module is
// concatenated without `shading.wgsl`, and the pool it *does* share is group 0.
@group(1) @binding(7) var default_sampler: sampler;
@group(1) @binding(8) var grain_masks: texture_2d_array<f32>;
// The environment cube (`ibl.rs`): what an escaped ray sees.
@group(1) @binding(10) var gi_env: texture_cube<f32>;
@group(1) @binding(11) var gi_env_sampler: sampler;

// ── The sky's share (Sunscar, 11 Nov 2026) ───────────────────────────────────
//
// From 6 Nov an escaped ray put the sky's radiance into the reservoir, so the traced estimate held the whole of the
// diffuse light. A reservoir is one sample and its weight: right for a small bright source (a sunlit patch through a
// doorway), wrong for a dome that fills the hemisphere. Its noise does not average out, it is shared between
// neighbours and frames, and under a cloud deck or in shade, where the sky is all the light there is, it stood on
// every surface as dark smoky blotches 15-30 pixels wide (the developer, 7 and 11 Nov: "black splotches"). Inside a
// crown it was worse: nearly every borrowed sky sample failed its visibility ray, and the leaves went black.
//
// The sky is now estimated as what it is. Each texel's ray adds to two sums: the environment's radiance along it
// (`den`), and the same again if it left the scene (`num`). Their ratio is the share of the environment's light that
// gets past the scene, and it is exact, with no noise at all, wherever nothing is in the way: open ground stays as
// smooth as the cube. `gi_accumulate.wgsl` averages the sums over neighbours and over time, and the shading pass
// multiplies the cube's own irradiance by the ratio. The reservoir keeps what it is good at: the bounce off what the
// rays hit.
/// This frame's sums per GI texel. `r` over `g` is the share for the side of the surface the camera sees. `b` over
/// `a` is the open sky a thin leaf's far side receives (see `GiSurface.leaf`). Each ray's radiance is relative to
/// the sky's mean overhead: about 1 by day and by night, so the half floats that hold them never run out of range
/// or of precision, and a frame under a brightening sky weighs no more than the one before it.
@group(1) @binding(12) var sky_out: texture_storage_2d<rgba16float, write>;
/// The clouds' shadow on the ground (`clouds.wgsl` `cloud_shadow`) and its `[centre_x, centre_z, extent, strength]`:
/// the sun a bounce ray's hit receives comes through the deck, as the sun on every shaded surface does.
@group(1) @binding(13) var gi_cloud_shadow: texture_2d<f32>;
@group(1) @binding(14) var<uniform> gi_cloud_shadow_params: vec4<f32>;

/// The most one ray may weigh against the sky's mean: the sun's own texels in the cube stay a few times the mean,
/// and nothing brighter is let turn the share into "did the one ray at the sun get out".
const GI_SKY_CEILING: f32 = 16.0;
/// The cube's roughest mip (`ibl.rs` MIP_COUNT 6): its mean radiance about a normal, the sky's irradiance over pi.
const GI_ENV_ROUGHEST: f32 = 5.0;
/// How much of the sky a hit on a mesh is taken to see. ponytail: a guess (walls and trunks see about half the dome
/// and the cube's own ground term stands for the rest); read the accumulated ratio at the hit when it is on screen
/// if bounced sky light ever proves wrong.
const GI_HIT_SKY: f32 = 0.75;

/// The sun's share through the cloud deck at a world point (shading.wgsl `cloud_shadow_at`, the same field and rule).
fn gi_cloud_sun_at(p: vec3<f32>) -> f32 {
    if gi_cloud_shadow_params.w <= 0.0 {
        return 1.0;
    }
    let local = (p.xz - gi_cloud_shadow_params.xy) / max(gi_cloud_shadow_params.z, 1.0);
    let near = clamp(textureSampleLevel(gi_cloud_shadow, default_sampler,
        clamp(local, vec2<f32>(-1.0), vec2<f32>(1.0)) * 0.5 + 0.5, 0.0).r, 0.0, 1.0);
    let past = max(abs(local.x), abs(local.y)) - 1.0;
    if past <= 0.0 {
        return near;
    }
    var mean = 0.0;
    for (var i = 0; i < 16; i = i + 1) {
        let g = (vec2<f32>(f32(i & 3), f32(i >> 2)) + 0.5) / 4.0;
        mean += textureSampleLevel(gi_cloud_shadow, default_sampler, g, 0.0).r;
    }
    return mix(near, clamp(mean / 16.0, 0.0, 1.0), saturate(past * 2.0));
}

/// How much of the sky the ground sees at a terrain hit: the terrain's baked sky visibility (terrain_material.wgsl
/// `terrain_sky_visibility`), 1 without a bake.
fn gi_terrain_sky(terrain_index: u32, p: vec3<f32>) -> f32 {
    let tm = terrain_materials[terrain_index];
    if tm.skyvis_map < 0 {
        return 1.0;
    }
    let uv = (p.xz - tm.terrain_origin) * tm.inv_world_size;
    let seen = textureSampleLevel(textures[tm.skyvis_map], default_sampler, uv, 0.0).a;
    return mix(1.0, seen, clamp(tm.sky_visibility_strength, 0.0, 1.0));
}

@group(1) @binding(5) var<storage, read_write> gi_a: array<GiReservoir>;
@group(1) @binding(6) var<storage, read_write> gi_b: array<GiReservoir>;

fn gi_empty() -> GiReservoir {
    return GiReservoir(vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), 0.0);
}

/// Cheap hash-based uniform. Same generator as `restir_di.wgsl`.
fn gi_rand(seed: ptr<function, u32>) -> f32 {
    *seed = *seed * 747796405u + 2891336453u;
    var x = *seed;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return f32((x >> 22u) ^ x) / 4294967295.0;
}

/// Cosine-weighted direction about `n`.
///
/// Cosine rather than uniform (Bevy uses uniform with an explicit inverse PDF):
/// the diffuse BRDF's own cosine then cancels the sampling density exactly, so
/// the estimator loses a multiply and, more importantly, stops spending samples
/// on grazing directions that the cosine would have thrown away anyway.
fn gi_sample_cosine(n: vec3<f32>, u: vec2<f32>) -> vec3<f32> {
    let u1 = u.x;
    let u2 = u.y;
    let r = sqrt(u1);
    let theta = 6.28318530718 * u2;
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.99);
    let t = normalize(cross(up, n));
    let b = cross(n, t);
    return normalize(t * (r * cos(theta)) + b * (r * sin(theta)) + n * sqrt(max(1.0 - u1, 0.0)));
}

/// Trace one ray and resolve what it hit. Hit resolution lives in `rt_hit.wgsl`
/// so Halcyon reflections and this pass cannot drift apart (VV-D).
fn gi_trace(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, t_max: f32) -> RtHit {
    return rt_trace(origin, dir, t_min, t_max);
}

/// True when nothing blocks the segment between two points.
fn gi_visible(origin: vec3<f32>, to: vec3<f32>, t_min: f32) -> bool {
    let d = to - origin;
    let dist = length(d);
    if dist <= t_min {
        return true;
    }
    // Through a leaf card's holes, as the bounce ray that found the sample went (`rt_trace`). Opaque, every card was
    // a solid quad here: a canopy turned away nearly every sky sample and every sun ray to a hit under it, and the
    // floor of a wood went black (Sunscar, 6 Nov 2026).
    if rt_candidate_alpha {
        var aq: ray_query;
        rayQueryInitialize(&aq, accel, RayDesc(0x6u, 0x5u, t_min, dist * 0.999, origin, d / dist));
        while rayQueryProceed(&aq) {
            if rt_candidate_is_covered(rayQueryGetCandidateIntersection(&aq)) {
                rayQueryConfirmIntersection(&aq);
            }
        }
        return rayQueryGetCommittedIntersection(&aq).kind == RAY_QUERY_INTERSECTION_NONE;
    }
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        accel,
        // Terminate on first hit, and stop just short of the target so the
        // surface at `to` is not its own occluder.
        // Visibility rays see shadow casters; bounce-hit rays keep the full
        // geometry mask in rt_trace so non-casters still receive indirect light.
        // 0x1 opaque casters, 0x4 cut-out cards (raytrace.rs): leaves block GI's light too.
        RayDesc(0x4u, 0x5u, t_min, dist * 0.999, origin, d / dist),
    );
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind == RAY_QUERY_INTERSECTION_NONE;
}

struct GiLightSample {
    position: vec3<f32>,
    irradiance: vec3<f32>,
}

/// Incident irradiance divided by the uniform emitter-surface PDF. The pool
/// stores flux/(4*pi). A one-sided Lambertian panel therefore contributes
/// 4*color*cos(emitter)*cos(receiver)/distance^2 after cancelling its area.
/// This is sampled area transport, not a second inverse-square applied to LTC.
fn gi_sample_local(ll: GpuLocalLight, p: vec3<f32>, n: vec3<f32>, u: vec2<f32>) -> GiLightSample {
    var emitter_pos = ll.position_ws;
    let is_area = ll.light_type == 2u || ll.light_type == 3u;
    var axis = vec3<f32>(0.0, -1.0, 0.0);
    if ll.light_type != 0u {
        axis = normalize(ll.direction_ws);
    }
    if is_area {
        let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(axis.y) > 0.95);
        let t = normalize(cross(up, axis));
        let b = cross(axis, t);
        var xy = (2.0 * u - 1.0) * vec2<f32>(max(ll._pad1, 0.05), max(ll._pad2, 0.05));
        if ll.light_type == 3u {
            // Same inscribed octagon as the direct-light integrator. Each of
            // eight equal triangles is sampled uniformly in surface area.
            let sector = min(u.x * 8.0, 7.999999);
            let angle = floor(sector) * 0.785398163;
            let a = vec2<f32>(cos(angle), sin(angle));
            let c = vec2<f32>(cos(angle + 0.785398163), sin(angle + 0.785398163));
            xy = sqrt(fract(sector)) * mix(a, c, u.y) * max(ll.radius, 0.05);
        }
        emitter_pos += t * xy.x + b * xy.y;
    } else if ll.light_type == 4u {
        // Retain the renderer's closest-point tube approximation.
        let half_length = max(ll._pad1, 0.05);
        emitter_pos += axis * clamp(dot(p - emitter_pos, axis), -half_length, half_length);
    }
    let delta = emitter_pos - p;
    let distance = length(delta);
    let l = delta / max(distance, 1.0e-4);
    let range_distance = select(distance, length(ll.position_ws - p), is_area);
    let q = range_distance / max(ll.range, 1.0e-4);
    let fade = max(1.0 - q * q * q * q, 0.0);
    // Bounce sources stop brightening inside 25 cm of the emitter. A bounce ray
    // that lands on a lamp's own housing or post sees ~100x the irradiance of
    // the street it lights, so one rare hit per pixel became a gold firefly
    // that temporal reuse could not average while the camera moved. Direct
    // light is untouched; only that near-lamp bounce is held to its 25 cm value.
    let near_sq = 0.0625;
    var attenuation = fade * fade / max(distance * distance, near_sq);
    if is_area {
        attenuation *= 4.0 * max(dot(axis, -l), 0.0);
    } else if ll.light_type == 1u {
        attenuation *= spot_shape(axis, -l, ll.spot_cos_outer, ll.spot_cos_inner, ll._pad2);
    } else if ll.light_type == 4u {
        attenuation *= max(length(cross(l, axis)), 0.15);
    }
    return GiLightSample(emitter_pos, ll.color * (attenuation * max(dot(n, l), 0.0)));
}

/// Lights walked per bounce hit before the walk starts striding (see below).
const GI_LIGHT_WALK: u32 = 128u;

/// Every light that reaches the hit is a candidate, resampled using unoccluded
/// irradiance. Only the survivor needs a visibility ray.
///
/// It used to draw four of the scene's lights at random and scale by count/4.
/// A lamp reaches a few metres, so in an interior of 46 nearly every draw was a
/// lamp that does not reach the hit at all, and the rare draw that did came
/// back twelve times too bright: one bounce in a dozen carried all the light,
/// and the ones that landed beside a lamp were white specks that reuse carried
/// across the screen while the camera moved. Five lights outdoors never showed it.
/// Walking the list costs a range test per light and leaves nothing to chance
/// but which reaching lamp is kept. Past GI_LIGHT_WALK lights it strides from a
/// random start, each light equally likely, and scales by the stride.
fn gi_local_irradiance(p: vec3<f32>, n: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
    let count = cluster_params.num_local_lights;
    if count == 0u { return vec3<f32>(0.0); }
    let walked = min(count, GI_LIGHT_WALK);
    let stride = f32(count) / f32(walked);
    let start = gi_rand(seed) * stride;
    var selected = GiLightSample(p, vec3<f32>(0.0));
    var weight_sum = 0.0;
    for (var i = 0u; i < walked; i++) {
        let index = min(u32(start + f32(i) * stride), count - 1u);
        // Most lights are turned away here on their position, range and type
        // alone, so only those are read until one passes.
        let kind = local_lights[index].light_type;
        let to_light = local_lights[index].position_ws - p;
        var reach = local_lights[index].range;
        if kind == 4u {
            // A tube is lit from the nearest point of its axis, up to half its
            // length nearer than its centre.
            reach += max(local_lights[index]._pad1, 0.05);
        }
        // A point or a spot behind the surface lights nothing. The others have
        // extent and may still show an edge over it.
        if dot(to_light, to_light) >= reach * reach || (kind < 2u && dot(to_light, n) <= 0.0) { continue; }
        let uv = vec2<f32>(gi_rand(seed), gi_rand(seed));
        let candidate = gi_sample_local(local_lights[index], p, n, uv);
        let weight = gi_luma(candidate.irradiance);
        weight_sum += weight;
        if weight > 0.0 && gi_rand(seed) * weight_sum < weight {
            selected = candidate;
        }
    }
    if weight_sum <= 0.0 || !gi_visible(p + n * 0.02, selected.position, 0.02) {
        return vec3<f32>(0.0);
    }
    return selected.irradiance * (weight_sum * stride / gi_luma(selected.irradiance));
}

/// Direct radiance leaving a secondary hit: sun plus shadowed local fixtures.
/// One sun ray and at most one local-light shadow ray, independent of count.
fn gi_direct_at(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, terrain_index: i32, transmission: f32,
    seed: ptr<function, u32>) -> vec3<f32> {
    let l = normalize(light.direction);
    let ndl = dot(n, l);
    var irradiance = gi_local_irradiance(p, n, seed);
    // The sun reaches the hit through the cloud deck, as it reaches every shaded surface (`cloud_shadow_at`). Without
    // this a bounce under an overcast carried the full sun, several times the light of the sky that was really there.
    let sun = light.color * (light.ibl_intensity * gi_cloud_sun_at(p));
    if ndl > 0.0 && gi_luma(sun) > 1.0e-6 && gi_visible(p + n * 0.02, p + l * 4000.0, 0.02) {
        // Retain the established outdoor sun-bounce grading. Previously the
        // final IBL composite multiplied the whole reservoir by this value;
        // applying it here preserves that sun response while keeping practical
        // bounce independent of the sky-fill control.
        irradiance += sun * ndl;
    }
    // A leaf lit from the far side passes light through (thin Lambertian translucency, `Material.transmission`):
    // a canopy's underside glows green, and that, not the sky, is most of the light on a wood's floor (Sunscar,
    // 6 Nov 2026). The normal faces the ray, so the sun is behind it when ndl < 0.
    if ndl < 0.0 && transmission > 0.0 && gi_luma(sun) > 1.0e-6
        && gi_visible(p - n * 0.02, p + l * 4000.0, 0.02) {
        irradiance += sun * (-ndl * transmission);
    }
    // The sky lights the hit too: under a deck it is the only light the ground has to send on, and without it a
    // surface that faces the ground (an arch's underside, the lower leaves of a crown) had none at all. The cube's
    // roughest mip about the hit's normal is the sky's irradiance over pi; the hit sees its share of it.
    var seen = GI_HIT_SKY;
    if terrain_index >= 0 {
        seen = gi_terrain_sky(u32(terrain_index), p);
    }
    let sky = textureSampleLevel(gi_env, gi_env_sampler, n, GI_ENV_ROUGHEST).rgb * (light.ibl_intensity * seen);
    var direct_albedo = albedo;
    if rt_gi_deferred_terrain_albedo && terrain_index >= 0 {
        // Same lookup, position and arithmetic as eager rt_resolve; only its
        // timing changes. Ordinary textured/MASK hits still resolve there.
        direct_albedo = rt_terrain_albedo(u32(terrain_index), p);
    }
    // Lambert: albedo/π times irradiance.
    // `light.color` is already premultiplied by intensity (see
    // `DirectionalLight` in the pool) — the sort of detail that silently costs
    // a factor of 100 000 if assumed rather than checked.
    return direct_albedo * ((1.0 / 3.14159265) * irradiance + sky);
}

/// Luminance, the target function ReSTIR resamples against.
fn gi_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Reconnection-shift Jacobian.
///
/// A neighbour's sample point is a *point*, so reusing it here means looking at
/// the same patch of world from a different position: the solid angle it
/// subtends and the cosine at its surface both change, and the estimator is
/// only unbiased if that change is divided out. Ported from Bevy's `jacobian`,
/// including its rejection threshold — a large Jacobian means the two pixels
/// see the patch at wildly different angles, and keeping those samples explodes
/// the variance rather than reducing it.
fn gi_jacobian(new_pos: vec3<f32>, old_pos: vec3<f32>, sp: vec3<f32>, sn: vec3<f32>) -> f32 {
    let r = new_pos - sp;
    let q = old_pos - sp;
    let rl = length(r);
    let ql = length(q);
    if rl < 1e-6 || ql < 1e-6 {
        return 0.0;
    }
    let phi_r = saturate(dot(r / rl, sn));
    let phi_q = saturate(dot(q / ql, sn));
    if phi_q <= 0.0 {
        return 0.0;
    }
    let j = (phi_r * ql * ql) / (phi_q * rl * rl);
    if j != j || j > 1.0e6 {
        return 0.0;
    }
    return j;
}

/// Combine `other` into `r` at the shading point `p`/`n`.
///
/// Simplified from Bevy's full pairwise MIS: the target function here is the
/// canonical pixel's, and the neighbour's contribution is weighted by its own
/// confidence and Jacobian. That is the "talbot MIS" form the course notes give
/// as the practical default, and it is unbiased for the reuse this pass does.
fn gi_merge(
    r: ptr<function, GiReservoir>,
    p: vec3<f32>,
    n: vec3<f32>,
    other: GiReservoir,
    other_pos: vec3<f32>,
    seed: ptr<function, u32>,
) {
    if other.m <= 0.0 {
        return;
    }
    if other.w <= 0.0 {
        // Candidates that carried nothing are candidates still. They used all to carry the sky; now a ray that
        // leaves the scene is one of these, and passing them over forgot every miss: one hit in fifty rays stood
        // at a twenty-fifth of its light for good.
        (*r).m += other.m;
        return;
    }
    let j = gi_jacobian(p, other_pos, other.sample_pos, other.sample_normal);
    // Bevy rejects above 1.2. The same threshold, and for the same reason: past
    // it the shift is a bad approximation and the sample adds variance.
    if j <= 0.0 || j > 1.2 {
        return;
    }
    let wi = normalize(other.sample_pos - p);
    let ndl = saturate(dot(wi, n));
    if ndl <= 0.0 {
        return;
    }
    let p_hat = gi_luma(other.radiance) * ndl * j;
    let weight = p_hat * other.w * other.m;
    if weight <= 0.0 {
        return;
    }
    (*r).w_sum += weight;
    (*r).m += other.m;
    if gi_rand(seed) * (*r).w_sum <= weight {
        (*r).sample_pos = other.sample_pos;
        (*r).sample_normal = other.sample_normal;
        (*r).radiance = other.radiance;
    }
}

/// Cap on accumulated confidence, as in 24K.
const GI_M_CAP: f32 = 24.0;
/// Neighbours examined in the spatial pass.
const GI_SPATIAL_TAPS: u32 = 4u;
/// Spatial search radius in pixels. Bevy uses 30; smaller here because this
/// pass has no G-buffer to reject dissimilar neighbours with beyond depth.
const GI_SPATIAL_RADIUS: f32 = 10.0;   // GI texels (at half resolution, 20 pixels)

/// Reconstruct the primary surface from depth and the visibility buffer.
struct GiSurface {
    valid: bool,
    pos: vec3<f32>,
    normal: vec3<f32>,
    /// On a thin leaf (a two-sided cut-out), the share of light it lets through (`Material.transmission`); 0 on
    /// everything else. A leaf is lit from both sides: what the camera sees of it is the light its near side
    /// reflects and the light its far side lets through. The near side of a leaf that hangs over dark ground gets
    /// almost nothing, and with the far side left out every such leaf was black: the undersides of crowns, the
    /// insides of bushes, the low plants seen at a slant (the developer's "black splotches", 11 Nov 2026).
    leaf: f32,
    /// Metres the surface's rays start off it, along its normal: 5 cm, and more on far ground.
    ///
    /// The ground is drawn at a coarser level of detail with distance and traced at its finest, so a point on the
    /// drawn ground can lie under the traced one: its rays started inside the hill and met the ground's own back.
    /// It showed as rows of dark blobs on every far slope, as far apart as the coarse level's vertices (the
    /// developer's video, 11 Nov 2026: "smoky" blobs on the hill behind Monargent). The coarse level's vertex spacing
    /// doubles with each doubling of distance (terrain/mod.rs `select_lods`), and what a chord can miss of a curve
    /// grows as the square of its length, so the start is lifted by the square of the distance.
    lift: f32,
}

/// The lift that clears the ground's coarse levels: half a metre where the second level begins (192 m with the
/// default `lod_base_range` of 96 and 2 m cells), four times that at each doubling, never under 5 cm nor over 12 m.
/// ponytail: the terrain's own base range and cell size are not read; a terrain that sets them far from the defaults
/// needs them passed in its material.
fn gi_ground_lift(distance: f32) -> f32 {
    let levels = distance / 192.0;
    return clamp(0.5 * levels * levels, 0.05, 12.0);
}

// This cache stores exactly the geometry reconstructed by pass 1, at float32
// precision. Pass 2 can reuse it instead of loading and transforming the same
// triangle for every spatial neighbour. It is current-frame data, never history.
override gi_cache_surfaces: bool = false;
struct GiCachedSurface {
    position_valid: vec4<f32>,
    normal: vec4<f32>,
}
@group(1) @binding(9) var<storage, read_write> gi_surfaces: array<GiCachedSurface>;

fn gi_spatial_surface(index: u32, coord: vec2<i32>, dims: vec2<u32>) -> GiSurface {
    if gi_cache_surfaces {
        let cached = gi_surfaces[index];
        // `w` of the position is the lift, which is never zero on a surface.
        return GiSurface(cached.position_valid.w > 0.0, cached.position_valid.xyz, cached.normal.xyz,
            cached.normal.w, cached.position_valid.w);
    }
    return gi_primary_surface(coord, dims);
}

fn gi_primary_surface(coord: vec2<i32>, dims: vec2<u32>) -> GiSurface {
    var s: GiSurface;
    s.valid = false;
    s.pos = vec3<f32>(0.0);
    s.normal = vec3<f32>(0.0, 1.0, 0.0);
    s.leaf = 0.0;
    s.lift = 0.05;

    let depth = textureLoad(depth_tex, coord, 0);
    if depth <= 0.0 {
        return s;
    }
    let uv = (vec2<f32>(coord) + 0.5) * gi.inv_resolution;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    // Camera-relative (`inv_view_proj` is), then moved out.
    let world = gi.inv_view_proj * ndc;
    s.pos = world.xyz / world.w + gi.camera_pos;

    // The normal comes from the visibility buffer's triangle rather than from
    // depth derivatives: a compute shader has no quad to take derivatives
    // across, and a normal reconstructed from neighbouring depths is wrong
    // exactly at the silhouettes where indirect light matters most.
    let vis = textureLoad(vis_tex, coord, 0);
    if vis.x == 0u {
        return s;
    }
    let inst = instances[vis.x - 1u];
    let base = inst.index_offset + vis.y * 3u;
    let v0 = vertices[inst.vertex_offset + indices[base + 0u]];
    let v1 = vertices[inst.vertex_offset + indices[base + 1u]];
    let v2 = vertices[inst.vertex_offset + indices[base + 2u]];
    let p0 = (inst.model * vec4<f32>(v0.pos_x, v0.pos_y, v0.pos_z, 1.0)).xyz;
    let p1 = (inst.model * vec4<f32>(v1.pos_x, v1.pos_y, v1.pos_z, 1.0)).xyz;
    let p2 = (inst.model * vec4<f32>(v2.pos_x, v2.pos_y, v2.pos_z, 1.0)).xyz;
    // Geometric, not interpolated: this normal is used to push ray origins off
    // the surface, and a shading normal can point into the geometry it came
    // from on a low-poly silhouette.
    var gn = normalize(cross(p1 - p0, p2 - p0));
    if dot(gn, gi.camera_pos - s.pos) < 0.0 {
        gn = -gn;
    }
    s.normal = gn;
    s.valid = true;
    // Two-sided and cut out: a leaf, a blade, a frond (shading.wgsl's `thin_leaf`, the same test).
    if (materials[inst.material_id].flags & 1u) != 0u && materials[inst.material_id].alpha_cutoff > 0.0 {
        s.leaf = materials[inst.material_id].transmission;
    }
    if materials[inst.material_id].terrain_index >= 0 {
        s.lift = gi_ground_lift(distance(s.pos, gi.camera_pos));
    }
    return s;
}

// ── Pass 1: initial candidate + temporal reuse ───────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn initial_and_temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full = textureDimensions(depth_tex);
    let dims = gi_grid(full);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let coord = gi_full_coord(gid.xy, full);
    let index = gid.y * dims.x + gid.x;

    var seed = index * 9781u + gi.frame * 6271u + 17u;
    // The ray's direction from two uniforms. With the shared masks on they come from the mask at this texel of the
    // GI grid (not of the screen: the texels averaged together are grid neighbours), so neighbours and successive
    // frames spread over the hemisphere instead of clumping, and the sky's share settles in far fewer rays. The 64
    // slices repeat, so each round of them is turned by a further step of the R2 sequence.
    var u = vec2<f32>(gi_rand(&seed), gi_rand(&seed));
    if gi.history_valid >= 2.0 {
        let grain = textureLoad(grain_masks, vec2<i32>(gid.xy & vec2<u32>(63u)), i32(gi.frame & 63u), 0);
        seed = seed ^ u32(grain.a * 4294967295.0);
        u = fract(grain.rg + vec2<f32>(0.7548777, 0.5698403) * f32((gi.frame / 64u) & 1023u));
    }

    let surface = gi_primary_surface(coord, full);
    if gi_cache_surfaces {
        gi_surfaces[index] = GiCachedSurface(
            vec4<f32>(surface.pos, select(0.0, surface.lift, surface.valid)),
            vec4<f32>(surface.normal, surface.leaf));
    }
    if !surface.valid {
        gi_b[index] = gi_empty();
        textureStore(sky_out, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // ── Initial candidate: one bounce ────────────────────────────────────────
    var r = gi_empty();
    // A thin leaf's far side gets every other ray: a checkerboard that turns each frame, so each texel sees both
    // sides in two frames and each neighbourhood sees both in one.
    let far_side = surface.leaf > 0.0 && ((gid.x + gid.y + gi.frame) & 1u) == 1u;
    let ray_normal = select(surface.normal, -surface.normal, far_side);
    let dir = gi_sample_cosine(ray_normal, u);
    let origin = surface.pos + ray_normal * surface.lift;
    let hit = gi_trace(origin, dir, 0.05, gi.max_distance);
    // The sky's share: the environment along the ray, and again if the ray left the scene (see `sky_out`).
    let env_up = gi_luma(textureSampleLevel(gi_env, gi_env_sampler, vec3<f32>(0.0, 1.0, 0.0), GI_ENV_ROUGHEST).rgb);
    let env = min(gi_luma(textureSampleLevel(gi_env, gi_env_sampler, dir, 2.0).rgb) / max(env_up, 1.0e-9),
        GI_SKY_CEILING);
    let open = select(env, 0.0, hit.hit);
    textureStore(sky_out, vec2<i32>(gid.xy),
        select(vec4<f32>(open, env, 0.0, 0.0), vec4<f32>(0.0, 0.0, open, 1.0), far_side));
    // Bevy Solari's NO_WORLD_CACHE path rejects emissive hits. This estimator
    // does not importance-sample emissive geometry, so retaining the extremely
    // rare hits produces high-variance fireflies that wander under temporal
    // and spatial reuse. Emissive GI needs a light-sampling strategy of its own.
    if far_side {
        // No candidate this frame: the far side's ray counted the sky and nothing else, and the reservoir, which
        // holds the near side's bounce, goes on from its history as it stands.
    } else if hit.hit && all(hit.emissive <= vec3<f32>(0.0)) {
        let radiance = gi_direct_at(hit.pos, hit.normal, hit.albedo, hit.terrain_index, hit.transmission, &seed);
        let p_hat = gi_luma(radiance);
        if p_hat > 0.0 {
            r.sample_pos = hit.pos;
            r.sample_normal = hit.normal;
            r.radiance = radiance;
            // Cosine sampling cancels the BRDF's cosine, so the contribution
            // weight of a single candidate is 1/p_hat — the RIS weight over one
            // sample. `m` is 1: one candidate was drawn.
            r.w_sum = p_hat;
            r.m = 1.0;
            r.w = 1.0 / p_hat;
        } else {
            r.m = 1.0;
        }
    } else {
        // A ray that left the scene, or met something that glows, is a candidate that carries nothing: the sky it
        // saw is counted in `sky_out`, not here.
        r.m = 1.0;
    }

    // ── Temporal reuse ───────────────────────────────────────────────────────
    // Reprojection is the pixel's own history, as in 24K: the velocity buffer
    // (24AD) is still outstanding, so this is conservative under camera motion
    // rather than wrong. The similarity test is the sample point's distance
    // from the shading point, which rejects history that has slid onto
    // different geometry.
    if (u32(gi.history_valid) & 1u) != 0u {
        var prev = gi_a[index];
        if prev.m > 0.0 {
            prev.m = min(prev.m, min(GI_M_CAP, gi.history_m_cap));
            gi_merge(&r, surface.pos, surface.normal, prev, surface.pos, &seed);
        }
    }

    let p_hat_final = gi_luma(r.radiance) * saturate(dot(normalize(r.sample_pos - surface.pos), surface.normal));
    if p_hat_final > 0.0 && r.m > 0.0 {
        r.w = r.w_sum / (r.m * p_hat_final);
    } else {
        r.w = 0.0;
    }
    gi_b[index] = r;
}

// ── Pass 2: spatial reuse, visibility, shade ─────────────────────────────────

/// Keep the transport boundary representable by out_tex's rgba16float format.
/// This is the format's largest finite value, not an exposure/firefly cutoff.
/// In particular Inf * a zero bilateral weight must not poison a whole block
/// during full-resolution upsampling. Invalid channels contribute no energy.
fn gi_encode_output(radiance: vec3<f32>, distance: f32) -> vec4<f32> {
    let value = vec4<f32>(radiance, distance);
    let valid = select(vec4<f32>(0.0), value, value == value);
    return clamp(valid, vec4<f32>(0.0), vec4<f32>(65504.0));
}

@compute @workgroup_size(8, 8, 1)
fn spatial_and_shade(@builtin(global_invocation_id) gid: vec3<u32>) {
    let full = textureDimensions(depth_tex);
    let dims = gi_grid(full);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let g = vec2<i32>(gid.xy);
    let coord = gi_full_coord(gid.xy, full);
    let index = gid.y * dims.x + gid.x;

    var seed = index * 26699u + gi.frame * 15487u + 91u;
    if gi.history_valid >= 2.0 {
        let grain = textureLoad(grain_masks, vec2<i32>(coord & vec2<i32>(63)), i32(gi.frame & 63u), 0);
        seed = seed ^ u32(grain.r * 4294967295.0);
    }

    let surface = gi_spatial_surface(index, coord, full);
    if !surface.valid {
        gi_a[index] = gi_empty();
        textureStore(out_tex, g, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    var r = gi_b[index];

    // ── Spatial reuse ────────────────────────────────────────────────────────
    // The half 24K never got. It matters far more for GI than for DI: a direct
    // reservoir converges in a few frames because the sun is one small target,
    // while an indirect one is sampling the whole hemisphere and needs every
    // neighbour it can borrow.
    let depth_here = textureLoad(depth_tex, coord, 0);
    for (var i = 0u; i < GI_SPATIAL_TAPS; i = i + 1u) {
        let a = gi_rand(&seed) * 6.28318530718;
        let rad = sqrt(gi_rand(&seed)) * GI_SPATIAL_RADIUS;
        let ng = clamp(
            g + vec2<i32>(i32(cos(a) * rad), i32(sin(a) * rad)),
            vec2<i32>(0),
            vec2<i32>(i32(dims.x) - 1, i32(dims.y) - 1),
        );
        if ng.x == g.x && ng.y == g.y {
            continue;
        }
        let nc = gi_full_coord(vec2<u32>(ng), full);
        // Reject neighbours on different geometry. Without this a reservoir
        // crosses a silhouette and the background's bounce light bleeds onto
        // the foreground — the classic ReSTIR halo.
        // Reversed-Z (Sunscar, 6 Nov 2026): sky is 0, and `1 - depth` is what forward depth was within near / far,
        // so the tolerance is the one this was tuned with.
        let nd = textureLoad(depth_tex, nc, 0);
        if nd <= 0.0 || abs(nd - depth_here) > (1.0 - depth_here) * 0.02 {
            continue;
        }
        let ns = gi_spatial_surface(u32(ng.y) * dims.x + u32(ng.x), nc, full);
        if !ns.valid || dot(ns.normal, surface.normal) < 0.9 {
            continue;
        }
        let neighbour = gi_b[u32(ng.y) * dims.x + u32(ng.x)];
        gi_merge(&r, surface.pos, surface.normal, neighbour, ns.pos, &seed);
    }

    let wi = normalize(r.sample_pos - surface.pos);
    let ndl = saturate(dot(wi, surface.normal));
    let p_hat = gi_luma(r.radiance) * ndl;
    if p_hat > 0.0 && r.m > 0.0 {
        r.w = r.w_sum / (r.m * p_hat);
    } else {
        r.w = 0.0;
    }

    // ── Visibility ───────────────────────────────────────────────────────────
    // One ray, for the sample that survived resampling — the same bargain 24K
    // makes. Traced *after* the reservoir is stored, so the stored sample keeps
    // its unshadowed weight for next frame's reuse; shadowing it in the store
    // would make an occluded sample impossible to recover from and leave dark
    // trails behind moving geometry.
    gi_a[index] = r;
    if r.w > 0.0 && !gi_visible(surface.pos + surface.normal * surface.lift, r.sample_pos, 0.05) {
        r.w = 0.0;
    }

    // Cosine-weighted sampling already cancels the diffuse BRDF's own cosine,
    // so what is left is the sample's radiance times its contribution weight.
    // The shading pass multiplies by the surface albedo — this pass does not
    // know it, and must not, or two different albedos would be applied.
    let indirect = r.radiance * r.w * ndl * gi.intensity;
    // Alpha carries the traced surface's camera distance (0 = nothing traced):
    // the shading pass's upsample weights by it so light never crosses an edge.
    textureStore(out_tex, g, gi_encode_output(indirect, max(length(surface.pos - gi.camera_pos), 1.0e-3)));
}
