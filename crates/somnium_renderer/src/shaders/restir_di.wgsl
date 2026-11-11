//!include "global_pool.wgsl"
//!include "rt_alpha.wgsl"

enable wgpu_ray_query;

// Phase 24K: ReSTIR direct lighting.
//
// The shadow ray from 24J, plus the thing that makes rays affordable:
// *resampling*. Tracing one ray per light per pixel does not scale — a scene
// with a hundred lights would need a hundred rays. Resampled importance
// sampling instead draws a handful of cheap unshadowed candidates, keeps one in
// proportion to how much it contributes, and spends the single expensive ray
// confirming that one. The estimator stays unbiased because the kept sample
// carries the weight of everyone it beat.
//
// Reservoirs then let that work be *reused*. A pixel's chosen light is almost
// always a good choice for the same pixel next frame and for its neighbours, so
// combining reservoirs across time multiplies the effective sample count
// without tracing more rays. That reuse is the whole idea, and it is why this
// is the foundation 24L's indirect bounce is built on rather than a one-off.
//
// Compared with the shadow map this replaces: no cascades, no bias tuning, no
// peter-panning, and penumbra that comes from the light's actual angular size
// rather than from a filter kernel chosen to look about right.

struct RestirParams {
    /// Camera-relative: inverse(jittered projection x rotation alone). Add `camera_pos` for world space.
    inv_view_proj: mat4x4<f32>,
    /// Direction toward the sun, world space.
    sun_direction: vec3<f32>,
    /// Half the sun's angular diameter, radians (Phase 24E).
    sun_angular_radius: f32,
    inv_resolution: vec2<f32>,
    frame: u32,
    /// Zero when history should be ignored.
    history_valid: f32,
    camera_pos: vec3<f32>,
    /// How the sun's ray is traced (`occluded`). 0: opaque, as it was (debug view 38). 1: a ray from the ground
    /// passes the ground's underside (the default, and debug view 39). 2: that, and every ray passes through a
    /// cut-out's holes (a game that sets `SOMNIUM_SUN_CUTOUTS=1`).
    sun_rays: f32,
}

/// A reservoir over light samples.
///
/// `sample_dir` is the direction to the chosen sample, `p_hat` the target
/// function's value there (the ReSTIR papers' name for it), `w_sum` the total weight
/// of every candidate it competed against, `m` how many candidates that was,
/// and `w` the unbiased contribution weight that makes the single kept sample
/// stand in for all of them.
struct Reservoir {
    sample_dir: vec3<f32>,
    w_sum: f32,
    p_hat: f32,
    w: f32,
    m: f32,
    _pad: f32,
}

// Group 0 is the shared global pool (the scene's instances, materials and meshes), as for ReSTIR GI: a sun ray from
// the ground has to know the ground when it meets it.
@group(1) @binding(0) var accel:      acceleration_structure;
@group(1) @binding(1) var depth_tex:  texture_depth_2d;
@group(1) @binding(2) var out_tex:    texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var<uniform> params: RestirParams;
@group(1) @binding(4) var<storage, read>       prev_reservoirs: array<Reservoir>;
@group(1) @binding(5) var<storage, read_write> curr_reservoirs: array<Reservoir>;
@group(1) @binding(6) var grain_masks: texture_2d_array<f32>;
/// For `rt_alpha.wgsl`'s cut-out test.
@group(1) @binding(7) var default_sampler: sampler;
/// The visibility buffer: x = instance + 1 (0: nothing drawn), y = triangle.
@group(1) @binding(8) var vis_tex: texture_2d<u32>;

fn empty_reservoir() -> Reservoir {
    return Reservoir(vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0, 0.0);
}

/// Weighted reservoir sampling: keep `candidate` with probability
/// `weight / w_sum`.
///
/// This is what makes the whole scheme work in constant memory. Each candidate
/// either replaces the held sample or is discarded, but its weight is always
/// added — so the reservoir remembers the total even though it stores one.
fn reservoir_update(
    r: ptr<function, Reservoir>,
    candidate_dir: vec3<f32>,
    candidate_target: f32,
    weight: f32,
    rand: f32,
) {
    (*r).w_sum += weight;
    (*r).m += 1.0;
    if weight > 0.0 && rand * (*r).w_sum <= weight {
        (*r).sample_dir = candidate_dir;
        (*r).p_hat = candidate_target;
    }
}

/// Cheap hash-based uniform, adequate for candidate selection.
fn rand(seed: ptr<function, u32>) -> f32 {
    *seed = *seed * 747796405u + 2891336453u;
    var x = *seed;
    x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
    return f32((x >> 22u) ^ x) / 4294967295.0;
}

/// A direction inside the sun's disc.
///
/// The sun is not a point (Phase 24E), and sampling across its disc is what
/// produces a real penumbra: points that see all of it are fully lit, points
/// that see none are in umbra, and the gradient between is the soft edge that
/// PCSS has to approximate with a filter.
fn sample_sun_direction(seed: ptr<function, u32>) -> vec3<f32> {
    let dir = normalize(params.sun_direction);
    let u1 = rand(seed);
    let u2 = rand(seed);

    // Uniform point on a disc of the sun's angular radius, built in a frame
    // around the sun direction.
    let r = params.sun_angular_radius * sqrt(u1);
    let theta = 6.28318530 * u2;

    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(dir.y) > 0.99);
    let tangent = normalize(cross(up, dir));
    let bitangent = cross(dir, tangent);

    return normalize(dir + tangent * (r * cos(theta)) + bitangent * (r * sin(theta)));
}

/// How many pixel-footprints to push the ray start away from the surface.
///
/// Phase 25L. The origin is reconstructed from the depth buffer, so it carries
/// that buffer's precision error *and* sits wherever the pixel's centre lands on
/// a surface that may be nearly edge-on. A fixed 5 cm `t_min` is far below both
/// once the surface is more than a few metres away: the ray restarts inside the
/// geometry it came from and reports itself as its own occluder. On terrain that
/// showed as elongated black patches following the slopes — the artefact was in
/// the traced visibility, not in the shadow map, which is why no amount of
/// shadow-map bias touched it.
const RAY_BIAS_FOOTPRINTS: f32 = 6.0;

/// Trace a shadow ray. True when something blocks it. `ground`: the ray starts on the ground.
fn occluded(origin: vec3<f32>, dir: vec3<f32>, t_min: f32, ground: bool) -> bool {
    let cutouts = params.sun_rays > 1.5;
    if ground || cutouts {
        // Two things an opaque ray gets wrong, both seen on Sunscar's far slopes in sun (11 Nov 2026).
        //
        // The ground is drawn at a coarser level of detail with distance and traced at its finest, so a point on
        // the drawn ground can lie under the traced one. Its sun ray then starts inside the hill and meets the
        // ground's own underside: sunlit slopes wore dark bands as wide as the coarse level's cells, in steps with
        // saw-toothed edges. A ray from a heightfield can only meet its top, so the underside is passed through, as
        // GI's rays pass it (`rt_alpha.wgsl`).
        //
        // And every BLAS is opaque, so a leaf card shadowed as a whole card: dark rectangles beside every far shrub
        // and slabs under every crown. With cut-outs on, a candidate counts only where its texel is covered.
        rt_ground_origin = ground;
        rt_ray_dir = dir;
        var cq: ray_query;
        rayQueryInitialize(&cq, accel, RayDesc(0x6u, 0x1u, t_min, 10000.0, origin, dir));
        while rayQueryProceed(&cq) {
            let candidate = rayQueryGetCandidateIntersection(&cq);
            var blocks = true;
            if cutouts {
                blocks = rt_candidate_is_covered(candidate);
            } else {
                let inst = instances[candidate.instance_custom_data];
                blocks = materials[inst.material_id].terrain_index < 0 || !rt_ground_underside(inst, candidate);
            }
            if blocks {
                rayQueryConfirmIntersection(&cq);
            }
        }
        rt_ground_origin = false;
        return rayQueryGetCommittedIntersection(&cq).kind != RAY_QUERY_INTERSECTION_NONE;
    }
    var rq: ray_query;
    rayQueryInitialize(
        &rq,
        accel,
        // Terminate on first hit: a shadow ray only needs to know whether
        // anything is in the way, not what or how far.
        RayDesc(0x4u, 0x1u, t_min, 10000.0, origin, dir),
    );
    rayQueryProceed(&rq);
    return rayQueryGetCommittedIntersection(&rq).kind != RAY_QUERY_INTERSECTION_NONE;
}

/// Number of unshadowed candidates drawn before any ray is traced.
///
/// The point of the whole scheme: candidates are nearly free, rays are not.
const CANDIDATES: i32 = 8;
/// Cap on the history's candidate count.
///
/// Without it `m` grows without bound and the reservoir stops responding to
/// change — a light that switches off stays visible for as long as the history
/// has been accumulating, which is the classic ReSTIR failure.
const M_CAP: f32 = 20.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(depth_tex);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let coord = vec2<i32>(gid.xy);
    let index = gid.y * dims.x + gid.x;
    let depth = textureLoad(depth_tex, coord, 0);

    if depth <= 0.0 {
        // Sky (reversed-Z's cleared 0) is unshadowed by definition.
        textureStore(out_tex, coord, vec4<f32>(1.0, 1.0, 1.0, 1.0));
        curr_reservoirs[index] = empty_reservoir();
        return;
    }

    let uv = (vec2<f32>(coord) + 0.5) * params.inv_resolution;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = params.inv_view_proj * ndc;
    let rel = world.xyz / world.w;
    let origin = rel + params.camera_pos;

    // World size of one pixel at this depth: reconstruct the neighbour and
    // measure. That is the scale of both the reconstruction error and of how
    // far the true surface can sit from this point on a grazing slope, so it is
    // the right unit for the ray's start offset — and it needs no extra
    // uniforms, unlike a camera position or a per-pixel normal.
    let uv_dx = (vec2<f32>(coord) + vec2<f32>(1.5, 0.5)) * params.inv_resolution;
    let ndc_dx = vec4<f32>(uv_dx.x * 2.0 - 1.0, 1.0 - uv_dx.y * 2.0, depth, 1.0);
    let world_dx = params.inv_view_proj * ndc_dx;
    let footprint = length(world_dx.xyz / world_dx.w - rel);
    let ray_t_min = max(0.05, footprint * RAY_BIAS_FOOTPRINTS);

    var seed = index * 9781u + params.frame * 6271u;
    if params.history_valid >= 2.0 {
        let grain = textureLoad(grain_masks, coord & vec2<i32>(63), i32(params.frame & 63u), 0);
        seed = seed ^ u32(grain.b * 4294967295.0);
    }

    // ── Initial candidates ──────────────────────────────────────────────────
    // Drawn without tracing anything. The p_hat function is the unshadowed
    // contribution, which for a directional light reduces to how far the sample
    // direction sits from the disc centre; a full light set would evaluate each
    // light's intensity and falloff here instead.
    var r = empty_reservoir();
    for (var i = 0; i < CANDIDATES; i = i + 1) {
        let dir = sample_sun_direction(&seed);
        let p_hat = max(dot(dir, normalize(params.sun_direction)), 0.0);
        // Uniform source PDF over the disc, so the RIS weight is just the
        // p_hat function.
        reservoir_update(&r, dir, p_hat, p_hat, rand(&seed));
    }

    // Unbiased contribution weight: what makes one kept sample stand in for all
    // the candidates it beat.
    if r.p_hat > 0.0 {
        r.w = r.w_sum / (r.m * r.p_hat);
    }

    // ── Visibility ──────────────────────────────────────────────────────────
    // One ray, for the sample that survived resampling.
    let drawn = textureLoad(vis_tex, coord, 0);
    let on_ground = params.sun_rays > 0.5 && drawn.x != 0u
        && materials[instances[drawn.x - 1u].material_id].terrain_index >= 0;
    if r.w > 0.0 && occluded(origin, r.sample_dir, ray_t_min, on_ground) {
        // The running total has to go with the weight. Zeroing `w` alone left
        // `w_sum` holding the candidates' contribution, and the temporal
        // combine below recomputes `w = w_sum / (m * p_hat)` from it — which
        // resurrected the occluded sample at roughly w_sum/(m + prev_m) and
        // turned every shadow into a faint wash. Clearing the target function
        // as well keeps a shadowed pixel shadowed once its history agrees:
        // prev.w is then 0 too, so the reuse weight is 0 and the reservoir
        // stays empty rather than drifting back toward lit.
        r.w = 0.0;
        r.p_hat = 0.0;
        r.w_sum = 0.0;
    }

    // ── Temporal reuse ──────────────────────────────────────────────────────
    // Reprojection is the previous frame's own pixel: the camera-motion case is
    // handled by 24F's history and a full velocity buffer is still outstanding
    // (24AD), so reuse here is conservative rather than wrong.
    if (u32(params.history_valid) & 1u) != 0u {
        let prev = prev_reservoirs[index];
        if prev.m > 0.0 {
            var combined = r;
            let prev_m = min(prev.m, M_CAP);
            reservoir_update(
                &combined,
                prev.sample_dir,
                prev.p_hat,
                prev.p_hat * prev.w * prev_m,
                rand(&seed),
            );
            combined.m = r.m + prev_m;
            if combined.p_hat > 0.0 {
                combined.w = combined.w_sum / (combined.m * combined.p_hat);
            }
            r = combined;
        }
    }

    curr_reservoirs[index] = r;

    // Visibility in [0,1]. The reservoir's weight already carries the
    // resampling normalisation, so this is the fraction of the sun's disc the
    // point can see — a real penumbra rather than a filtered hard edge.
    let visibility = saturate(r.w * r.p_hat);
    textureStore(out_tex, coord, vec4<f32>(vec3<f32>(visibility), 1.0));
}
