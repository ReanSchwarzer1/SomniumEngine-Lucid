// MORROWIND-C: composition is declared here rather than assembled by a
// `format!` of `include_str!` calls at this pass's construction site. The
// resolver (`somnium_shader`) emits each module once, in this order, and
// hoists every `enable` above everything.
//!include "atmosphere.wgsl"
//!include "froxel_depth.wgsl"
//!include "storm.wgsl"

// Somnium Engine — Phase CONTROL-M: the volumetric cloud march.
//
// Concatenated after `atmosphere.wgsl`, whose LUT helpers and constants this
// reuses. That reuse is the point rather than a convenience: the clouds, the
// sky and the froxel volume must agree about the colour of the sun, and the
// only way to guarantee that is for all three to read the same transmittance
// LUT rather than each carrying its own sun tint.
//
// ## References
//
// - Schneider & Vos, SIGGRAPH 2015 — the shape model, the height gradients per
//   cloud type, Beer's law with the powder term, and the cone-sampled shadow
//   taps toward the sun.
// - Toft & Bowles, arXiv:1609.05344 — the adaptive step (large until density,
//   small inside), the early-out on transmittance, and the measured warning
//   that a per-pixel jittered ray start can cost more than the steps it saves
//   through texture-cache incoherence. `jitter_enabled` exists so that is a
//   measurement in this engine and not a belief.
// - Epic's Volumetric Cloud docs — Beer Shadow Maps as the ground-level cloud
//   shadow model, which is what `cloud_shadow` below writes.

struct CloudParams {
    /// Camera-relative: inverse(projection x the view's rotation alone). Rays start at `camera_pos`.
    inv_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    /// Metres from the ground to the bottom of the cloud layer.
    layer_bottom: f32,
    sun_direction: vec3<f32>,
    /// Layer thickness in metres.
    layer_thickness: f32,
    sun_illuminance: vec3<f32>,
    /// Density multiplier. Zero is a clear sky and skips the march entirely.
    density: f32,
    /// Wind offset in metres, accumulated on the CPU so the clouds keep moving
    /// across a pause in the same direction they were going.
    wind_offset: vec2<f32>,
    /// Metres of world per repeat of the weather map.
    weather_scale: f32,
    /// Metres of world per repeat of the base shape volume.
    shape_scale: f32,
    /// Strength of the high-frequency erosion, `0..1`.
    detail_strength: f32,
    /// Henyey–Greenstein forward lobe.
    phase_forward: f32,
    /// Henyey–Greenstein backward lobe. Negative.
    phase_backward: f32,
    /// Blend between the two lobes.
    phase_blend: f32,
    /// Ambient contribution from the sky, scaling the multiscatter LUT.
    ambient: f32,
    /// Extra absorption applied to precipitating columns.
    precipitation: f32,
    /// Non-zero applies the blue-noise ray-start offset.
    jitter_enabled: f32,
    /// Frame counter, for the temporal component of the jitter.
    frame: f32,
    /// Primary march steps at full quality.
    max_steps: f32,
    /// Light-march steps toward the sun.
    light_steps: f32,
    /// Half-extent, in metres, of the world-XZ cloud shadow map.
    shadow_extent: f32,
    /// `0..1`. Zero leaves the ground unshadowed by cloud.
    shadow_strength: f32,
    /// Distance in metres at which the march gives up. Clouds beyond the
    /// horizon are not worth the steps.
    max_distance: f32,
    /// Range the froxel volume spans, so aerial perspective can be applied to
    /// the cloud layer. Zero disables the lookup.
    volumetric_range: f32,
    /// Scale on the scattered light (`CloudSettings::brightness`).
    brightness: f32,
    /// How far a closed deck is carried down to the horizon, 0..1 (`CloudPass::horizon_closure`; `closed_dir`).
    horizon_closure: f32,
    /// How far a cloud is heaped, 0..1 (`CloudPass::heap`; `slab_density`).
    heap: f32,
    /// The share of the distance marched the step grows to, and the metres over which far cloud thins into the air
    /// (`CloudPass::far_step`, `far_fade`; 0 is off).
    far_step: f32,
    far_fade: f32,
    /// 1: `sun_illuminance` is the sun above the air (a running day clock, `Renderer::sky_above_air`), and each sample
    /// takes the sun through the air above it and the planet's shadow where it stands, so a high cloud is lit after the
    /// ground's sunset, and the sky's fill follows the twilight. 0: the sun on the ground, as it always was.
    above_air: f32,
    /// The planet over the horizon: toward it, w 1 by night (the sun too far down to light any cloud: the light
    /// march goes toward the planet); its light above the air (rgb, lux), w 1 when there is one.
    planet_direction: vec4<f32>,
    planet_light: vec4<f32>,
}

@group(0) @binding(0) var<uniform> cp: CloudParams;
@group(0) @binding(1) var base_noise_tex:   texture_3d<f32>;
@group(0) @binding(2) var detail_noise_tex: texture_3d<f32>;
@group(0) @binding(3) var weather_tex:      texture_2d<f32>;
@group(0) @binding(4) var cloud_sampler:    sampler;
@group(0) @binding(5) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(6) var multiscatter_lut:  texture_2d<f32>;
@group(0) @binding(7) var lut_sampler:       sampler;
@group(0) @binding(8) var scatter_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(9) var shadow_out:  texture_storage_2d<r16float, write>;
@group(0) @binding(10) var scene_depth: texture_depth_2d;
@group(0) @binding(11) var cloud_volumetrics: texture_3d<f32>;
/// Storm bodies and lightning (`storm.wgsl`). No bodies and no lights is every scene that asks for none.
@group(0) @binding(17) var<uniform> storm: Storm;

const CLOUD_PI: f32 = 3.14159265;
/// Transmittance below which the march stops. Anything darker is opaque.
const CLOUD_MIN_TRANSMITTANCE: f32 = 0.01;
/// Consecutive empty samples before the march goes back to long steps.
const CLOUD_EMPTY_BEFORE_ESCAPE: i32 = 8;
/// Samples a ray may light. The slab alone never reaches it (it is opaque long before).
const CLOUD_MAX_LIT: i32 = 96;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Shape
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn cloud_remap(v: f32, old_lo: f32, old_hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    return new_lo + (v - old_lo) / max(old_hi - old_lo, 1e-6) * (new_hi - new_lo);
}

/// The vertical density profile for a cloud type in `0..1`.
///
/// Three named shapes blended by the weather map's type channel: stratus is a
/// flat sheet low in the layer, cumulus is a rounded mass in the middle, and
/// cumulonimbus fills the whole layer. Returned as a multiplier on the shape
/// noise, which is what stops a cumulus from having a flat top.
fn height_gradient(height: f32, cloud_type: f32) -> f32 {
    let stratus = smoothstep(0.0, 0.08, height) * (1.0 - smoothstep(0.18, 0.32, height));
    let cumulus = smoothstep(0.02, 0.22, height) * (1.0 - smoothstep(0.60, 0.95, height));
    let cumulonimbus = smoothstep(0.0, 0.10, height) * (1.0 - smoothstep(0.85, 1.0, height));

    let low = mix(stratus, cumulus, clamp(cloud_type * 2.0, 0.0, 1.0));
    return mix(low, cumulonimbus, clamp((cloud_type - 0.5) * 2.0, 0.0, 1.0));
}

fn sample_weather(world_xz: vec2<f32>) -> vec4<f32> {
    let uv = (world_xz + cp.wind_offset) / max(cp.weather_scale, 1.0);
    return textureSampleLevel(weather_tex, cloud_sampler, uv, 0.0);
}

/// The slab's density at a world point. `cheap` skips the detail erosion, which is what
/// makes the long-step search affordable.
fn slab_density(world_pos: vec3<f32>, cheap: bool) -> f32 {
    let height = (world_pos.y - cp.layer_bottom) / max(cp.layer_thickness, 1.0);
    if height < 0.0 || height > 1.0 {
        return 0.0;
    }

    let weather = sample_weather(world_pos.xz);
    // A storm's holes take coverage, so their rims dissolve as a cloud's edge does.
    let coverage = weather.r * storm_slab(world_pos.xz);
    if coverage <= 0.001 {
        return 0.0;
    }
    // Cumulus heaps (`CloudPass::heap`; off by default). The height profile below is the same in every column, so
    // every cloud of a field ended at one height: white bricks with level tops, seen from the side (the lead,
    // 11 Nov 2026). A real one stands on a flat base, is tallest over its middle and falls away to its edge. The
    // column's profile is read as if the layer were thinner where the cloud's cover is thin: a third at its edge.
    var profile_height = height;
    if cp.heap > 0.0 {
        // (0.18 and a wide ramp: the smallest remnants of a field stood as upright dashes at 0.3.)
        profile_height = height / mix(1.0, mix(0.18, 1.0, smoothstep(0.1, 0.85, coverage)), cp.heap);
    }

    // The wind carries the *shape* as well as the weather field, and the two
    // move at slightly different rates so the sky does not read as one rigid
    // sheet sliding past.
    let shape_uvw = (world_pos + vec3<f32>(cp.wind_offset.x, 0.0, cp.wind_offset.y) * 1.35)
        / max(cp.shape_scale, 1.0);
    let base = textureSampleLevel(base_noise_tex, cloud_sampler, shape_uvw, 0.0);

    // Erode the Perlin–Worley base by its own higher-frequency Worley
    // channels, then by the height gradient, then by coverage. Order matters:
    // coverage last is what makes the coverage slider behave like a dissolve
    // rather than like a density multiply.
    let worley_fbm = base.g * 0.625 + base.b * 0.25 + base.a * 0.125;
    var shape = clamp(cloud_remap(base.r, worley_fbm - 1.0, 1.0, 0.0, 1.0), 0.0, 1.0);
    shape *= height_gradient(profile_height, weather.g);
    var density = clamp(cloud_remap(shape, 1.0 - coverage, 1.0, 0.0, 1.0), 0.0, 1.0) * coverage;

    if density <= 0.0 {
        return 0.0;
    }
    if cheap || cp.detail_strength <= 0.0 {
        return density * cp.density;
    }

    // Detail erosion, applied only near the edges. Schneider's trick: at the
    // core the high-frequency term is inverted so the interior gains billows
    // instead of being eaten away.
    let detail_uvw = (world_pos + vec3<f32>(cp.wind_offset.x, 0.0, cp.wind_offset.y) * 2.1)
        / max(cp.shape_scale * 0.11, 1.0);
    let detail = textureSampleLevel(detail_noise_tex, cloud_sampler, detail_uvw, 0.0);
    let detail_fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
    let modifier = mix(detail_fbm, 1.0 - detail_fbm, clamp(profile_height * 5.0, 0.0, 1.0));
    density = cloud_remap(
        density,
        modifier * cp.detail_strength * 0.35,
        1.0,
        0.0,
        1.0,
    );

    return clamp(density, 0.0, 1.0) * cp.density;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Storm bodies (Sunscar weather, 11 Nov 2026)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// The slab is one tiling field: no storm in it has a place or parts. These do: a tower with its anvil, a wall cloud
// under it, a shelf on its gust front, a funnel to the ground with its debris, shafts of rain. Each is a shape in
// world metres that says how full the body is at a point; the cloud's own noise then frays its edge exactly as
// coverage frays the slab's. Because they are part of `cloud_density`, the screen march, the ground's shadow field
// and the sky light's capture all see them.

fn storm_rot(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * p.x - s * p.y, s * p.x + c * p.y);
}

/// What the slab keeps of its coverage at a place: 1, less inside a storm's holes.
fn storm_slab(world_xz: vec2<f32>) -> f32 {
    var keep = 1.0;
    for (var i = 0u; i < storm.counts.x; i = i + 1u) {
        let v = storm.volumes[i];
        if storm_is_modifier(v) {
            keep *= smoothstep(v.size.x, v.size.x + v.a.x, length(world_xz - v.pos.xz));
        }
    }
    return keep;
}

// Each shape returns x: how full the body is at `q` (the point from its foot), 0 outside to 1 in its core, and
// y: how much its noise may fray it there (0 a smooth, laminar surface, 1 a ragged one).

/// A cumulonimbus's tower: from a flat base it draws in, climbs leaning downwind, and flares again where it spreads
/// into its anvil (`storm_anvil`, a body of its own), an overshooting dome on top. A rotating one is banded and
/// wider at its base: a bell.
fn storm_cell(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    let over = v.b.w / v.size.y;
    if h < 0.0 || h > 1.0 + over {
        return vec2<f32>(0.0);
    }
    let radius = v.size.x;
    let d = q.xz - v.a.xy * (v.b.x * h * h);
    let rho = length(d);
    // Bands round the tower where the updraft turns (a supercell's striations).
    let band = 1.0 + v.b.y * 0.07 * sin(h * 52.0 + atan2(d.y, d.x) * 2.0 - v.albedo.w);
    let bell = (0.18 + 0.3 * v.b.y) * (1.0 - smoothstep(0.0, 0.22, h));
    // A tower that has reached its anvil flares into it (`b.z` 1). One that has not (a flanking line's, a young
    // storm's) is a cumulus still: it draws in to a rounded head. (Flared and cut level, each stood as a drum with a
    // collar: the windowless pictures, 12 Nov.)
    let head = sqrt(max(1.0 - pow(smoothstep(0.4, 1.0, h), 2.0), 0.0));
    let rt = radius * mix((0.82 + bell) * head, 0.82 + bell + 0.7 * smoothstep(0.5, 0.95, h), v.b.z) * band;
    let dome = 0.97 + over * (1.0 - smoothstep(0.0, radius * 0.9, rho));
    // A convective tower's edge is hard: the shape fills over a sixth of its radius, and the noise frays only that.
    let tower = (1.0 - smoothstep(rt * 0.84, rt, rho)) * (1.0 - smoothstep(dome - 0.05, dome, h)) * smoothstep(0.0, 0.012, h);
    return vec2<f32>(tower, 0.7);
}

/// A cumulonimbus's anvil (it takes its tower's foot, radius and height, so the two meet): ice spread downwind and
/// thinning to its rim, its underside a wedge rising from the tower, its top falling a little toward the rim;
/// mammatus under it.
fn storm_anvil(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    let reach = v.a.z;
    if h < 0.0 || h > 1.0 || reach <= 0.0 {
        return vec2<f32>(0.0);
    }
    let dir = v.a.xy;
    let radius = v.size.x;
    let along = dot(q.xz, dir);
    let across = dot(q.xz, vec2<f32>(-dir.y, dir.x));
    let e = length(vec2<f32>((along - reach * 0.42) / (reach * 0.62 + radius), across / (reach * 0.34 + radius)));
    if e >= 1.0 {
        return vec2<f32>(0.0);
    }
    let under = mix(v.a.w, 0.9, smoothstep(0.1, 1.0, e));
    let top = 0.985 - 0.07 * smoothstep(0.2, 1.0, e);
    var anvil = (1.0 - smoothstep(0.6, 1.0, e)) * smoothstep(under - 0.02, under + 0.03, h) * (1.0 - smoothstep(top - 0.045, top, h));
    var fray = 1.0;
    if v.b.z > 0.0 && h < under + 0.02 && e < 0.95 {
        // Mammatus: smooth lobes about 350 m across hanging from the underside, away from the tower.
        let cells = textureSampleLevel(base_noise_tex, cloud_sampler, vec3<f32>(q.x, 0.0, q.z) / 2800.0 + vec3<f32>(0.0, 0.37, 0.0), 0.0).g;
        let hang = v.b.z * 0.06 * smoothstep(0.45, 0.85, cells) * smoothstep(radius * 1.8, radius * 3.2, length(q.xz));
        var lobe = 0.0;
        if hang > 0.0 {
            // Under the anvil's body, not out at its rim: there the sun reaches them, and a ring of white puffs
            // stood round the anvil (the windowless pictures, 12 Nov).
            lobe = (1.0 - smoothstep(0.45, 0.7, e)) * smoothstep(under - hang, under - hang * 0.55, h);
        }
        if lobe > anvil {
            anvil = lobe;
            fray = 0.15;
        }
    }
    // Ice: its rim thins away in fibres, it does not break into puffs.
    return vec2<f32>(anvil, fray * 0.08);
}

/// A storm's base: a level deck, an ellipse drawn out downwind, thick at the storm and thinning to a rim that the
/// noise breaks into scud; its underside smooth near the storm and rolling a little.
fn storm_deck(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = (q.y + 25.0 * sin(q.x * 0.0041) * sin(q.z * 0.0033)) / v.size.y;
    if h < 0.0 || h > 1.0 {
        return vec2<f32>(0.0);
    }
    let along = dot(q.xz, v.a.xy) - v.a.z;
    let across = dot(q.xz, vec2<f32>(-v.a.y, v.a.x));
    let e = length(vec2<f32>(along / (v.size.x + v.a.z), across / v.size.x));
    if e >= 1.0 {
        return vec2<f32>(0.0);
    }
    let top = mix(1.0, 0.4, smoothstep(0.3, 1.0, e));
    let m = (1.0 - smoothstep(0.7, 1.0, e)) * smoothstep(0.0, 0.04, h) * (1.0 - smoothstep(top - 0.25, top, h));
    return vec2<f32>(m, mix(0.35, 1.0, smoothstep(0.45, 0.9, e)));
}

/// A wall cloud: a block lowered under the base, narrow at its ragged foot and flaring into the base it hangs from,
/// with a tail cloud reaching toward the rain at its lower levels.
fn storm_wall(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    if h < -0.3 || h > 1.5 {
        return vec2<f32>(0.0);
    }
    let hc = clamp(h, 0.0, 1.0);
    let r = v.size.x * (0.7 + 0.9 * hc * hc);
    var m = (1.0 - smoothstep(r * 0.6, r, length(q.xz))) * smoothstep(-0.25, 0.2, h);
    let along = dot(q.xz, v.a.xy);
    if v.a.z > 0.0 && along > 0.0 {
        let u = along / v.a.z;
        let across = dot(q.xz, vec2<f32>(-v.a.y, v.a.x));
        let tube = v.size.x * 0.34 * (1.0 - 0.65 * u);
        let from_axis = length(vec2<f32>(across, (q.y - v.size.y * (0.3 + 0.4 * u)) * 1.5));
        m = max(m, (1.0 - smoothstep(tube * 0.45, tube, from_axis)) * (1.0 - smoothstep(0.8, 1.0, u)));
    }
    return vec2<f32>(m, mix(0.5, 1.0, v.a.w));
}

/// A shelf cloud: a wedge along an arc, its nose low at the leading edge, its upper surface smooth and banded and
/// climbing back to the parent's base, its underside ragged and lifting behind the nose.
fn storm_shelf(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let height = v.size.y;
    let nose = v.a.w;
    if q.y < -0.2 * nose || q.y > height * 1.05 {
        return vec2<f32>(0.0);
    }
    var off = atan2(q.x, q.z) - v.a.x;
    off = off - 6.2831853 * round(off / 6.2831853);
    let arc = 1.0 - smoothstep(v.a.y * 0.85, v.a.y, abs(off));
    let behind = v.size.x - length(q.xz);
    if arc <= 0.0 || behind < -nose || behind > v.a.z {
        return vec2<f32>(0.0);
    }
    let u = clamp(behind / v.a.z, 0.0, 1.0);
    // Its upper face is stacked plates: tiers that step back as it climbs, each a ledge with a shadow under it. Its
    // underside is torn by the air rolling under it, in lumps as deep as its nose.
    let along_arc = off * v.size.x;
    let rough = textureSampleLevel(base_noise_tex, cloud_sampler, vec3<f32>(along_arc, 0.0, behind) / (nose * 6.0) + vec3<f32>(0.23, 0.61, 0.07), 0.0);
    let tiers = 0.5 + 0.5 * sin(u * 26.0 + (rough.r - 0.5) * 3.0);
    let top = nose + (height - nose) * pow(u, 0.75) - nose * 0.45 * tiers * smoothstep(0.02, 0.12, u) * (1.0 - smoothstep(0.7, 1.0, u));
    let under = height * 0.55 * u * u * (3.0 - 2.0 * u) + nose * (1.1 * rough.g - 0.35) * smoothstep(0.0, 0.2, u);
    let inside = smoothstep(under - 0.12 * nose, under + 0.25 * nose, q.y) * (1.0 - smoothstep(top - 0.1 * nose, top, q.y));
    let front = smoothstep(-0.5 * nose, 0.3 * nose, behind) * (1.0 - smoothstep(0.85, 1.0, u));
    // Laminar bands along the face near the leading edge.
    let bands = 0.78 + 0.22 * sin(q.y / max(nose, 1.0) * 31.0 + behind * 0.01);
    let m = arc * front * inside * mix(1.0, bands, 1.0 - smoothstep(0.0, 0.3, u));
    let ragged = 1.0 - smoothstep(under, under + 0.6 * nose, q.y);
    return vec2<f32>(m, mix(0.12, 1.0, ragged));
}

/// Where a funnel's axis stands at a height (0 its foot, 1 its top), from its foot: leaning, and bowed between.
fn storm_funnel_axis(v: StormVolume, h: f32) -> vec2<f32> {
    let lean = v.a.xy;
    let len = length(lean);
    var side = vec2<f32>(1.0, 0.0);
    if len > 1.0 {
        side = vec2<f32>(-lean.y, lean.x) / len;
    }
    return lean * h + side * v.b.x * sin(h * 3.14159265);
}

/// A funnel: a tube from the ground to the cloud, dense at its wall and thin inside, so it reads lighter down its
/// middle.
fn storm_funnel(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    if h < -0.02 || h > 1.04 {
        return vec2<f32>(0.0);
    }
    let hc = clamp(h, 0.0, 1.0);
    let rho = length(q.xz - storm_funnel_axis(v, hc));
    let r = mix(v.size.x, v.a.z, pow(hc, v.a.w));
    let wall = 1.0 - smoothstep(r * 0.74, r, rho);
    let core = 1.0 - smoothstep(r * 0.2, r * 0.6, rho);
    return vec2<f32>(wall * (1.0 - 0.65 * core), 0.3);
}

/// The debris cloud and dust skirt at a vortex's foot: widest along the ground, drawn in as it rises, with inflow
/// bands spiralling in.
fn storm_bowl(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    if h < -0.05 || h > 1.0 {
        return vec2<f32>(0.0);
    }
    let hc = clamp(h, 0.0, 1.0);
    let rho = length(q.xz);
    let r = v.size.x * mix(1.0 + v.a.y, 0.4, pow(hc, 0.6));
    var m = (1.0 - smoothstep(r * 0.4, r, rho)) * (1.0 - smoothstep(0.5, 1.0, hc)) * smoothstep(-0.05, 0.02, h);
    let spiral = 0.5 + 0.5 * sin(atan2(q.z, q.x) * 3.0 + log(max(rho, 1.0)) * 5.0 + v.albedo.w * 2.0);
    m *= mix(1.0, 0.45 + 0.55 * spiral, (1.0 - hc) * smoothstep(v.size.x * 0.5, v.size.x, rho));
    if v.a.x > 0.0 {
        m *= mix(0.35, 1.0, smoothstep(v.a.x * 0.6, v.a.x * 1.4, rho));
    }
    return vec2<f32>(m, 1.0);
}

/// A shaft of rain or hail under a base: its foot lags its top (the drops fall into slower air), a little wider at
/// the ground; virga ends part of the way down.
fn storm_shaft(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    if h < 0.0 || h > 1.03 {
        return vec2<f32>(0.0);
    }
    let lag = 1.0 - (1.0 - h) * (1.0 - h);
    let r = v.size.x * (1.0 + 0.2 * (1.0 - h));
    var m = 1.0 - smoothstep(r * 0.35, r, length(q.xz + v.a.xy * lag));
    if v.a.z > 0.001 {
        m *= smoothstep(v.a.z - 0.12, v.a.z + 0.12, h);
    }
    return vec2<f32>(m, 1.0);
}

/// A ring of storm: a wall round a clear eye that leans outward as it climbs (from inside, a stadium), smooth on its
/// inner face and ragged outside, an anvil spreading outward from its top, and a curtain of rain under it.
fn storm_ring(v: StormVolume, q: vec3<f32>) -> vec2<f32> {
    let h = q.y / v.size.y;
    let rho = length(q.xz);
    if h > 1.02 {
        return vec2<f32>(0.0);
    }
    if h < 0.0 {
        if -q.y > v.b.y || v.b.x <= 0.0 {
            return vec2<f32>(0.0);
        }
        return vec2<f32>(1.0 - smoothstep(0.5, 1.0, abs(rho - v.size.x) / (v.a.x * 1.3)), 1.0);
    }
    let mid = v.size.x + v.a.y * h * h;
    let s = (rho - mid) / (v.a.x * (1.0 + 0.5 * h));
    var wall = (1.0 - smoothstep(0.75, 1.0, abs(s))) * (1.0 - smoothstep(0.93, 1.0, h)) * smoothstep(0.0, 0.02, h);
    var fray = mix(0.25, 1.0, smoothstep(-0.5, 0.5, s));
    if v.a.z > 0.0 && rho > mid {
        let e = (rho - mid) / v.a.z;
        let under = mix(v.a.w, 0.92, smoothstep(0.1, 1.0, e));
        let anvil = (1.0 - smoothstep(0.6, 1.0, e)) * smoothstep(under - 0.02, under + 0.03, h) * (1.0 - smoothstep(0.95, 0.99, h));
        if anvil > wall {
            wall = anvil;
            fray = 1.0;
        }
    }
    return vec2<f32>(wall, fray);
}

/// One body's density at a world point (not a hole's: `storm_slab`).
fn storm_body(v: StormVolume, p: vec3<f32>, cheap: bool) -> f32 {
    let q = p - v.pos.xyz;
    let turn = v.albedo.w;
    var m = vec2<f32>(0.0);
    // Where its noise is read: turned with it, and for a shaft drawn out along its fall.
    var np = q;
    // A cloud this dense is opaque at a tenth of its density, so thinning its edge with noise leaves its outline the
    // shape's own: a tower came out a smooth drum (first stills, 11 Nov). The noise has to move the outline. For the
    // convective bodies the point is pushed in or out from the axis by the noise's large lumps (turrets about half
    // a radius across) and its next octave, and up or down a little, before the shape is asked.
    var qs = q;
    let kind = i32(v.pos.w + 0.5);
    if kind == 1 || kind == 2 || kind == 5 {
        let lump = textureSampleLevel(base_noise_tex, cloud_sampler, q / v.size.w + vec3<f32>(0.31, 0.0, 0.17), 0.0);
        // The cell noise's octaves are round lumps with creases between, a cauliflower's: they carry most of it.
        let swell = clamp(1.0 + (lump.r - 0.5) * 0.3 + (lump.g - 0.6) * 0.5 + (lump.b - 0.6) * 0.22, 0.7, 1.35);
        // Up or down a little too (not at a tower's flat base); a wall cloud's foot hangs in rags.
        let lift = select(0.06 * step(0.05 * v.size.y, q.y), 0.7, kind == 2) * v.size.y;
        qs = vec3<f32>(q.x / swell, q.y + (lump.a - 0.5) * lift, q.z / swell);
    } else if kind == 10 {
        // A storm's base is not a ceiling: it is hollowed and domed from below, in lumps a kilometre across and
        // smaller ones on them, and its rim is torn. (It only ever rises: the body's cylinder starts at its base.)
        let at = vec3<f32>(q.x, 0.0, q.z);
        let big = textureSampleLevel(base_noise_tex, cloud_sampler, at / v.size.w + vec3<f32>(0.31, 0.21, 0.17), 0.0);
        let small = textureSampleLevel(base_noise_tex, cloud_sampler, at / (v.size.w * 0.31) + vec3<f32>(0.11, 0.57, 0.43), 0.0);
        let swell = clamp(1.0 + (big.g - 0.6) * 0.5 + (small.g - 0.6) * 0.2, 0.75, 1.3);
        let hollow = min(0.3 * v.size.y, 340.0) * clamp(0.75 * big.g + 0.45 * small.g - 0.25, 0.0, 1.0);
        qs = vec3<f32>(q.x / swell, q.y - hollow, q.z / swell);
        if v.a.w > 0.0 {
            // A deck that stands on the ground is a wall (dust): its face and top billow, lobes some hundreds of
            // metres across with creases between, cut back into it by up to half a kilometre.
            let lobe = textureSampleLevel(base_noise_tex, cloud_sampler, q / 1100.0 + vec3<f32>(0.37, 0.11, 0.59), 0.0);
            let knot = textureSampleLevel(base_noise_tex, cloud_sampler, q / 380.0 + vec3<f32>(0.71, 0.29, 0.13), 0.0);
            let cut = v.a.w * 520.0 * clamp(1.2 * lobe.g + 0.45 * knot.g - 0.5, 0.0, 1.0);
            let mid = v.a.xy * v.a.z;
            let from_mid = q.xz - mid;
            let out = mid + from_mid * (1.0 + cut / max(length(from_mid), 1.0));
            qs = vec3<f32>(out.x, q.y + 0.6 * cut, out.y);
        }
    } else if kind == 4 {
        // A funnel's wall is condensing, turning air: it bulges and necks along its length.
        let hc = clamp(q.y / v.size.y, 0.0, 1.0);
        let axis = storm_funnel_axis(v, hc);
        let lump = textureSampleLevel(base_noise_tex, cloud_sampler, vec3<f32>(q.x, q.y * 0.5 - turn * v.size.x, q.z) / v.size.w + vec3<f32>(0.47, 0.0, 0.29), 0.0);
        let swell = clamp(1.0 + (lump.g - 0.6) * 0.55 + (lump.b - 0.6) * 0.3, 0.72, 1.3);
        let rel = (q.xz - axis) / swell;
        qs = vec3<f32>(axis.x + rel.x, q.y, axis.y + rel.y);
    } else if kind == 7 {
        // Rain comes down from the parts of a base that rain hardest: a shaft's edge wanders, in and out by a third.
        let wander = textureSampleLevel(base_noise_tex, cloud_sampler, vec3<f32>(q.x, q.y * 0.15, q.z) / (v.size.x * 2.4) + vec3<f32>(0.13, 0.0, 0.41), 0.0);
        let swell = clamp(0.62 + 0.75 * wander.g, 0.6, 1.35);
        qs = vec3<f32>(q.x / swell, q.y, q.z / swell);
    }
    switch kind {
        case 1 {
            m = storm_cell(v, qs);
            let r = storm_rot(q.xz, turn * (1.0 - 0.5 * clamp(q.y / v.size.y, 0.0, 1.0)));
            np = vec3<f32>(r.x, q.y, r.y);
        }
        case 2 {
            m = storm_wall(v, qs);
            let r = storm_rot(q.xz, turn);
            np = vec3<f32>(r.x, q.y, r.y);
        }
        case 3 {
            m = storm_shelf(v, q);
            np = vec3<f32>(q.x, q.y * 2.0, q.z);
        }
        case 4 {
            m = storm_funnel(v, qs);
            let hc = clamp(q.y / v.size.y, 0.0, 1.0);
            // The wall turns, faster low down, and its streaks climb.
            let r = storm_rot(q.xz - storm_funnel_axis(v, hc), turn * (1.5 - 0.5 * hc));
            np = vec3<f32>(r.x, q.y - turn * v.size.x * 0.6, r.y);
        }
        case 5 {
            m = storm_bowl(v, qs);
            let r = storm_rot(q.xz, turn * mix(1.6, 0.8, clamp(length(q.xz) / v.size.x, 0.0, 1.0)));
            np = vec3<f32>(r.x, q.y - turn * 12.0, r.y);
        }
        case 6 {
            m = storm_anvil(v, q);
            // Its ice is drawn out along the wind aloft: fibres, not lumps.
            np = vec3<f32>(dot(q.xz, v.a.xy) * 0.3, q.y * 1.5, dot(q.xz, vec2<f32>(-v.a.y, v.a.x)));
        }
        case 7 {
            m = storm_shaft(v, qs);
            np = vec3<f32>(q.x, q.y * 0.07 + turn, q.z);
        }
        case 8 {
            m = storm_ring(v, q);
            if q.y < 0.0 {
                // Its rain, drawn out along its fall.
                np = vec3<f32>(q.x, q.y * 0.07 + turn * 30.0, q.z);
            } else {
                let r = storm_rot(q.xz, turn * (1.0 - 0.5 * clamp(q.y / v.size.y, 0.0, 1.0)));
                np = vec3<f32>(r.x, q.y, r.y);
            }
        }
        case 10 {
            m = storm_deck(v, qs);
            np = vec3<f32>(q.x, q.y * 2.0, q.z);
        }
        default {}
    }
    if m.x <= 0.0 {
        return 0.0;
    }
    let n = textureSampleLevel(base_noise_tex, cloud_sampler, np / v.size.w, 0.0);
    let worley = n.g * 0.625 + n.b * 0.25 + n.a * 0.125;
    let shape = clamp(cloud_remap(n.r, worley - 1.0, 1.0, 0.0, 1.0), 0.0, 1.0);
    // Frayed as coverage frays the slab; where the surface is laminar the shape alone holds.
    let frayed = clamp(cloud_remap(shape, 1.0 - m.x, 1.0, 0.0, 1.0), 0.0, 1.0) * m.x;
    var density = mix(m.x * 0.6, frayed, m.y);
    if !cheap && density > 0.0 {
        let detail = textureSampleLevel(detail_noise_tex, cloud_sampler, np / (v.size.w * 0.13), 0.0);
        let fbm = detail.r * 0.625 + detail.g * 0.25 + detail.b * 0.125;
        density = clamp(cloud_remap(density, fbm * 0.3 * m.y, 1.0, 0.0, 1.0), 0.0, 1.0);
        if kind == 7 {
            // Combed into fall streaks a hundred metres or so across, drawn out along the fall.
            let comb = textureSampleLevel(detail_noise_tex, cloud_sampler, vec3<f32>(q.x, q.y * 0.04 + turn, q.z) / 420.0, 0.0);
            density *= 0.3 + 1.1 * comb.r;
        }
    }
    // A ring's rain is thin beside its cloud.
    if v.pos.w > 7.5 && q.y < 0.0 {
        density *= v.b.x;
    }
    return density * v.size.z;
}

/// The bodies at a point: their density, what they scatter of the light they meet (weighted by density), and the
/// finest step any of them asks of a march there.
struct StormSample {
    density: f32,
    albedo: vec3<f32>,
    step: f32,
}

fn storm_sample(p: vec3<f32>, cheap: bool) -> StormSample {
    var s = StormSample(0.0, vec3<f32>(0.0), 1e9);
    for (var i = 0u; i < storm.counts.x; i = i + 1u) {
        let v = storm.volumes[i];
        let o = p.xz - v.bound.xy;
        if storm_is_modifier(v) || p.y < v.bound_y.x || p.y > v.bound_y.y || dot(o, o) > v.bound.z * v.bound.z {
            continue;
        }
        let d = storm_body(v, p, cheap);
        if d > 0.0 {
            s.density += d;
            s.albedo += v.albedo.rgb * d;
            s.step = min(s.step, v.bound.w);
        }
    }
    if s.density > 0.0 {
        s.albedo = s.albedo / s.density;
    }
    return s;
}

/// Density at a world point: the slab and every storm body.
fn cloud_density(world_pos: vec3<f32>, cheap: bool) -> f32 {
    var density = slab_density(world_pos, cheap);
    if storm.counts.x > 0u {
        density += storm_sample(world_pos, cheap).density;
    }
    return density;
}

/// Where a ray is in cloud: the slab's interval widened to every storm body's cylinder it crosses.
fn cloud_interval(origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    var interval = slab_interval(origin, dir);
    for (var i = 0u; i < storm.counts.x; i = i + 1u) {
        let v = storm.volumes[i];
        if storm_is_modifier(v) {
            continue;
        }
        let c = storm_cylinder(v, origin, dir);
        let exit = min(c.y, cp.max_distance);
        if exit > c.x {
            if interval.y <= interval.x {
                interval = vec2<f32>(c.x, exit);
            } else {
                interval = vec2<f32>(min(interval.x, c.x), max(interval.y, exit));
            }
        }
    }
    return interval;
}

/// Optical depth toward the sun from inside a storm body, in `light_march`'s units: four taps that treble in reach
/// from the body's own finest step (to 40 of them), so a funnel a hundred metres wide shades itself and still sees
/// the storm two kilometres above it. The slab's march steps by the layer's thickness, which would miss both.
fn storm_light_march(origin: vec3<f32>, sun: vec3<f32>, finest: f32) -> f32 {
    var optical_depth = 0.0;
    var width = finest;
    var near = 0.0;
    for (var i = 0; i < 4; i = i + 1) {
        optical_depth += cloud_density(origin + sun * (near + width * 0.5), true) * width * 0.01;
        near += width;
        width *= 3.0;
    }
    return optical_depth;
}

/// Lightning's light on the cloud at a point: each flash by the inverse square, through the cloud between (what is
/// here and what is halfway there, as an optical depth). Direct light dies in a few free paths; the light the cloud
/// scatters carries on as a glow, which is why a whole tower lights from one channel.
fn storm_flashes(p: vec3<f32>, density: f32) -> vec3<f32> {
    var light = vec3<f32>(0.0);
    for (var i = 0u; i < storm.counts.y; i = i + 1u) {
        let f = storm.flashes[i];
        let to = f.pos.xyz - p;
        let d2 = dot(to, to);
        let tau = (density + cloud_density(p + to * 0.5, true)) * 0.5 * sqrt(d2) * 0.05;
        let through = 0.4 * exp(-tau) + 0.6 / (1.0 + 0.15 * tau);
        light += f.color.rgb * through / (d2 + f.pos.w * f.pos.w);
    }
    // Scattered evenly: deep in cloud the light has no direction left.
    return light / (4.0 * CLOUD_PI);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Lighting
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

fn hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * CLOUD_PI * max(denom * sqrt(max(denom, 1e-6)), 1e-6));
}

/// Dual-lobe phase: a strong forward lobe for the silver lining and a weak
/// backward one so a cloud away from the sun is not flat black.
fn dual_lobe(cos_theta: f32) -> f32 {
    return mix(
        hg(cos_theta, cp.phase_forward),
        hg(cos_theta, cp.phase_backward),
        clamp(cp.phase_blend, 0.0, 1.0),
    );
}

/// The dual lobe with both eccentricities scaled by `k`: a multiple-scattering octave's flatter phase.
fn dual_lobe_scaled(cos_theta: f32, k: f32) -> f32 {
    return mix(
        hg(cos_theta, cp.phase_forward * k),
        hg(cos_theta, cp.phase_backward * k),
        clamp(cp.phase_blend, 0.0, 1.0),
    );
}

/// Multiple scattering inside the cloud as octaves of the single-scatter sun term (Wrenninge et al., "Oz", 2013;
/// Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016). Each octave
/// halves the scattering, the extinction toward the sun and the phase's eccentricity, so light that has bounced
/// inside the cloud reaches its body and base. Single scattering alone left an overcast deck's base near black
/// where real cloud is mid-grey (Sunscar, 6 Nov 2026). The first octave is the old term exactly.
const CLOUD_MS_OCTAVES: i32 = 3;
/// Weight of the diffusion term (`cloud_diffusion`): 1 the two-stream estimate as it stands.
const CLOUD_DIFFUSE: f32 = 1.0;
/// What a cloud droplet scatters of the light it intercepts: water absorbs almost none.
const CLOUD_ALBEDO: f32 = 0.95;
/// The clear sky's radiance per lux of sun, as the engine's IBL scale has it (`ibl.rs`: ~8000 cd/m2 under 100 000 lux).
const CLOUD_SKY_PER_LUX: f32 = 0.08;

fn sun_octaves(optical_depth: f32, cos_theta: f32) -> f32 {
    var energy = 0.0;
    var a = 1.0;
    for (var i = 0; i < CLOUD_MS_OCTAVES; i = i + 1) {
        let phase = select(dual_lobe_scaled(cos_theta, a), dual_lobe(cos_theta), i == 0);
        energy += a * beer_powder(optical_depth * a) * phase;
        a *= 0.5;
    }
    return energy;
}

/// The diffusion limit: deep in a thick cloud, light that has scattered many times falls off as
/// 1 / (1 + 0.75 tau (1 - g)) (two-stream; Bohren 1987), not exponentially as every octave does. It is why a
/// cloud is white from every side and an overcast deck's base is grey, not black.
fn cloud_diffusion(optical_depth: f32) -> f32 {
    return 1.0 / (1.0 + 0.75 * optical_depth * (1.0 - 0.85));
}

/// Beer's law with Schneider's powder term.
///
/// Beer alone makes the sun-facing side of a cloud *darker* the denser it is,
/// which is backwards for what the eye expects at the edge of a cumulus. The
/// powder term restores the dark-edge/bright-core reading.
fn beer_powder(optical_depth: f32) -> f32 {
    let beer = exp(-optical_depth);
    let powder = 1.0 - exp(-optical_depth * 2.0);
    return beer * mix(1.0, powder * 2.0, 0.5);
}

/// Six cone-spread taps toward the sun, accumulating optical depth.
///
/// Cone-spread rather than a straight line because a straight march
/// under-samples the neighbouring cloud that is actually casting the shadow,
/// and the error shows up as a cloud lit as though it were alone in the sky.
/// A body's light at a cloud sample `p`, per unit of its light above the air: through the air above the sample toward
/// it, and past the planet's shadow where the sample stands, eased over `soft` (the body's radius, in its cosine). A
/// kilometre up and tens of kilometres off, a cloud keeps a set sun's light after the ground has lost it, and reddens.
/// `r_eye`: the eye's radius from the planet's centre, km.
fn cloud_light_at(p: vec3<f32>, r_eye: f32, toward: vec3<f32>, soft: f32) -> vec3<f32> {
    let k = (p - cp.camera_pos) * 0.001;
    let at = vec3<f32>(k.x, r_eye + k.y, k.z);
    let r_at = length(at);
    let mu_at = dot(at, toward) / r_at;
    let horizon_at = -sqrt(max(1.0 - (GROUND_RADIUS / r_at) * (GROUND_RADIUS / r_at), 0.0));
    return sample_transmittance(transmittance_lut, lut_sampler, r_at, max(mu_at, horizon_at))
        * smoothstep(horizon_at - soft, horizon_at + soft, mu_at);
}

fn light_march(origin: vec3<f32>, sun: vec3<f32>) -> f32 {
    let steps = max(i32(cp.light_steps), 1);
    // Spread across the layer rather than across the ray, for the same reason
    // the primary march does: a light ray at a shallow sun angle must not
    // sample the cloud more coarsely than one at noon.
    let step_len = max(cp.layer_thickness / f32(steps) * 1.5, 1.0);
    var optical_depth = 0.0;
    var t = 0.0;
    for (var i = 0; i < steps; i = i + 1) {
        // The cone widens with distance; the offsets are a fixed low-discrepancy
        // set rather than random so the result does not shimmer.
        let spread = f32(i) / f32(steps) * step_len * 0.6;
        let jitter = vec3<f32>(
            sin(f32(i) * 2.399) * spread,
            cos(f32(i) * 1.117) * spread * 0.3,
            cos(f32(i) * 2.399) * spread,
        );
        t += step_len;
        let p = origin + sun * t + jitter;
        optical_depth += cloud_density(p, true) * step_len * 0.01;
    }
    // One long tap far along the ray catches a distant bank that the short
    // cone misses entirely.
    let far = origin + sun * step_len * f32(steps) * 4.0;
    optical_depth += cloud_density(far, true) * step_len * 0.04;
    return optical_depth;
}

/// Where the ray enters and leaves the cloud slab, in metres along the ray.
///
/// A flat slab rather than two concentric spheres. The curvature of a 1 km
/// layer over a scene a kilometre across is well under a pixel, and the slab
/// form has no case where a grazing ray produces an enormous interval.
fn slab_interval(origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let bottom = cp.layer_bottom;
    let top = cp.layer_bottom + cp.layer_thickness;
    if abs(dir.y) < 1e-4 {
        // Parallel to the slab: either inside it for ever or never in it.
        if origin.y >= bottom && origin.y <= top {
            return vec2<f32>(0.0, cp.max_distance);
        }
        return vec2<f32>(-1.0, -1.0);
    }
    let t0 = (bottom - origin.y) / dir.y;
    let t1 = (top - origin.y) / dir.y;
    let enter = max(min(t0, t1), 0.0);
    let exit = max(t0, t1);
    if exit <= 0.0 {
        return vec2<f32>(-1.0, -1.0);
    }
    return vec2<f32>(enter, min(exit, cp.max_distance));
}

/// Interleaved-gradient noise — the cheapest blue-noise-like offset there is,
/// and the one whose cost Toft & Bowles measured.
fn ign(pixel: vec2<f32>, frame: f32) -> f32 {
    let p = pixel + 5.588238 * (frame % 64.0);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

/// Aerial perspective for the cloud layer, sampled at its own depth.
///
/// Applied to the cloud layer **separately** from the scene layer and only
/// then composited, because applying it once after compositing is wrong: the
/// scene behind a cloud is further away than the cloud is, and one fetch
/// cannot be right for both.
fn cloud_aerial(view_uv: vec2<f32>, distance: f32) -> vec4<f32> {
    if cp.volumetric_range <= 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let w = clamp(froxel_w(distance, cp.volumetric_range), 0.0, 1.0);
    let sample = textureSampleLevel(
        cloud_volumetrics, lut_sampler, vec3<f32>(view_uv, w), 0.0);
    // Log-space inscatter, matching `volumetric.wgsl`'s storage.
    return vec4<f32>(exp(sample.rgb), clamp(sample.a, 0.0, 1.0));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// The march
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Sine of the elevation a closed deck is carried down from (2.6 degrees): above it a ray still meets the layer
/// inside the march's reach.
const HORIZON_FLOOR: f32 = 0.045;

/// The direction to march for a ray that looks at the sky: its own, or, under a closed deck and within a few degrees
/// of the horizon, the same bearing at 2.6 degrees. The layer is a flat slab and the march gives up at its reach, so
/// under about two and a half degrees no ray meets a deck that in truth runs to the horizon and beyond; and a ray
/// level with the deck's base or under it looks out below the deck for ever. A strip of the bare bright sky stood
/// there (the open-land stills, 11 Nov 2026). What is there is the deck's far underside: the same grey, drawn down.
/// Rays well below the horizon are the ground's and keep their own way.
fn closed_dir(dir: vec3<f32>) -> vec3<f32> {
    let level = length(dir.xz);
    if cp.horizon_closure <= 0.0 || dir.y >= HORIZON_FLOOR || dir.y < -0.02 || level < 1e-4 {
        return dir;
    }
    let reach = sqrt(1.0 - HORIZON_FLOOR * HORIZON_FLOOR) / level;
    return normalize(mix(dir, vec3<f32>(dir.x * reach, HORIZON_FLOOR, dir.z * reach), cp.horizon_closure));
}

/// One ray through the layer from the camera: in-scattered light, transmittance and the transmittance-weighted
/// depth. `jitter` (0..1) moves the first sample within a long step. Shared by the screen march and the sky
/// light's capture (`cloud_env`).
struct CloudRay {
    scatter: vec3<f32>,
    transmittance: f32,
    depth: f32,
}

fn march_ray(ray_dir: vec3<f32>, interval: vec2<f32>, jitter: f32, flashes: bool) -> CloudRay {
    let span = interval.y - interval.x;
    let steps = max(i32(cp.max_steps), 8);

    // The storm bodies this ray crosses: where, and the finest step each asks for. Inside one the march steps by the
    // body's own size, not the slab's: a funnel is a hundred metres wide and the slab's long step is two hundred.
    var hits = 0;
    var hit_enter: array<f32, 16>;
    var hit_exit: array<f32, 16>;
    var hit_step: array<f32, 16>;
    for (var i = 0u; i < storm.counts.x; i = i + 1u) {
        let v = storm.volumes[i];
        if storm_is_modifier(v) {
            continue;
        }
        let c = storm_cylinder(v, cp.camera_pos, ray_dir);
        if c.y > c.x {
            hit_enter[hits] = c.x;
            hit_exit[hits] = c.y;
            hit_step[hits] = v.bound.w;
            hits = hits + 1;
        }
    }

    // **The step is a distance in metres, derived from the layer's own
    // thickness — not `span / steps`.**
    //
    // The first cut divided the whole ray, and the whole ray is enormous: a
    // shallow ray through a slab 2 km thick can span 60 km, so `span / 48` was
    // a coarse step over a kilometre long and a cloud got sampled three or
    // four times through its entire depth. That is what the blocky,
    // stair-stepped cloud edges were — not the quarter-resolution buffer, which
    // was the obvious suspect and the wrong one.
    //
    // Sampling density is now constant in metres however the ray is angled, so
    // a cloud on the horizon is shaded like a cloud overhead. The cost is
    // bounded by the iteration cap and the transmittance early-out rather than
    // by the step count, which is the right way round: a ray that leaves the
    // slab immediately does almost no work whatever `max_steps` says.
    let long_step = max(cp.layer_thickness / f32(steps) * 3.0, 1.0);
    let short_step = long_step * 0.35;
    // Enough iterations to cross the layer several times at the fine step,
    // capped so a grazing ray cannot run away. Each storm body crossed gets its own share.
    let iteration_cap = steps * 6 + hits * 48;

    // Toft & Bowles' jitter. Their measurement is that this can cost
    // 2.3 → 7.5 ms through texture-cache incoherence, so whether it pays is a
    // number this engine takes rather than a belief it inherits.
    //
    // `cp.frame` is zero unless *temporal* jitter is on, and it defaults off.
    // With no cloud-history filter — the clouds ride in the TAA buffer, and TAA
    // has no motion vector for a sky pixel to reproject — a pattern that moved
    // every frame was not dithering, it was visible shimmer. Frame-stable
    // spatial jitter still breaks up the banding it was added for.
    var t = interval.x;
    var first = true;

    let sun = normalize(cp.sun_direction);
    let cos_theta = dot(ray_dir, sun);

    // Ambient comes from the same multiscatter LUT the sky uses, so a cloud at
    // dusk is lit by the dusk sky rather than by a constant.
    let r = GROUND_RADIUS + max(cp.camera_pos.y, 0.0) * 0.001;
    var sun_transmittance = vec3<f32>(0.0);
    var ambient_sky = vec3<f32>(0.0);
    if !ray_hits_ground(r, sun.y) {
        sun_transmittance = sample_transmittance(transmittance_lut, lut_sampler, r, sun.y);
        ambient_sky = sample_multiscatter(multiscatter_lut, lut_sampler, r, sun.y);
    }
    // Light on the cloud, in physical units (Sunscar, 6 Nov 2026). The sun's illuminance through the air, and the
    // clear sky's fill: its colour from the multiscatter LUT, its level the engine's per-lux sky radiance. Before,
    // the fill had no sun in it (it came to a fraction of a nit) and the march divided by extinction without the
    // scattering coefficient, so a deck came out ~1/extinction (17x at density 1.15) too bright, the sky light
    // that captures it lit the land 2 stops over, and the clear sky between clouds read navy.
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let sun_e = sun_transmittance * cp.sun_illuminance;
    var sky_fill = ambient_sky / max(dot(ambient_sky, luma), 1e-6) * (CLOUD_SKY_PER_LUX * dot(sun_e, luma));
    // Above the air (a running day clock): the fill follows the twilight. Its level is the clear sky's per lux of sun
    // at noon, times what is left of the multiply scattered light for this sun (the planet's shadow is in that table),
    // so it eases through dusk instead of stopping with the camera's last ray of sun.
    let above_air = cp.above_air > 0.5;
    if above_air {
        let scattered = sample_multiscatter_twilight(multiscatter_lut, lut_sampler, r, sun.y);
        let scattered_noon = sample_multiscatter(multiscatter_lut, lut_sampler, r, 1.0);
        let noon = sample_transmittance(transmittance_lut, lut_sampler, r, 1.0) * cp.sun_illuminance;
        sky_fill = scattered / max(dot(scattered_noon, luma), 1e-9) * (CLOUD_SKY_PER_LUX * dot(noon, luma));
    }
    // The planet over the horizon: a second light. By night (the sun too far down to light any cloud) the light march
    // goes toward it and it lights the cloud as the sun does; by day and at dusk its light, ten thousand times less
    // than the sun's, is diffused light only. Its glow on the air fills the cloud from the sky, as the sun's does.
    let planet_on = cp.planet_light.w > 0.5;
    let planet_night = planet_on && cp.planet_direction.w > 0.5;
    let planet_dir = normalize(cp.planet_direction.xyz + vec3<f32>(0.0, 1e-6, 0.0));
    let light_dir = select(sun, planet_dir, planet_night);
    let cos_planet = dot(ray_dir, planet_dir);
    if planet_on {
        let glow = sample_multiscatter_twilight(multiscatter_lut, lut_sampler, r, planet_dir.y);
        let glow_noon = sample_multiscatter(multiscatter_lut, lut_sampler, r, 1.0);
        let noon_through = sample_transmittance(transmittance_lut, lut_sampler, r, 1.0);
        sky_fill += glow / max(dot(glow_noon, luma), 1e-9)
            * (CLOUD_SKY_PER_LUX * dot(noon_through * cp.planet_light.rgb, luma));
    }

    var scatter = vec3<f32>(0.0);
    var transmittance = 1.0;
    var empty_runs = 0;
    var lit = 0;
    var inside = false;
    // The last optical depth marched toward the sun inside a storm body, and the samples since.
    var body_light = 0.0;
    var body_light_age = 99;
    // Transmittance-weighted mean depth: a cloud has no single depth, and this
    // is what the aerial-perspective fetch and CONTROL-N's motion vectors both
    // need in place of one.
    var depth_accum = 0.0;
    var depth_weight = 0.0;

    for (var i = 0; i < iteration_cap; i = i + 1) {
        if t >= interval.y || transmittance < CLOUD_MIN_TRANSMITTANCE {
            break;
        }
        var step_len = select(long_step, short_step, inside);
        if cp.far_step > 0.0 {
            // Far cloud is small in the picture: the step grows with the distance, and the iterations reach it.
            step_len = max(step_len, t * cp.far_step * select(1.0, 0.35, inside));
        }
        for (var k = 0; k < hits; k = k + 1) {
            if t + step_len >= hit_enter[k] && t <= hit_exit[k] {
                // No finer than a body's own detail, and no finer than the eye can tell at that range (a body ten
                // kilometres off is not stepped as one at one).
                step_len = min(step_len, max(hit_step[k], t * 0.012) * select(2.5, 1.0, inside));
            }
        }
        if first {
            // The start's jitter, as a share of the step the march starts with: of the slab's long step it skipped up
            // to a hundred metres of a body's near side, differently in every pixel.
            t += jitter * step_len;
            first = false;
        }
        let p = cp.camera_pos + ray_dir * (t + step_len * 0.5);

        if !inside {
            // Searching: cheap density only, long steps.
            if cloud_density(p, true) > 0.0 {
                // Step back one step and switch to fine sampling, or the
                // cloud's leading edge is cut off flat.
                inside = true;
                empty_runs = 0;
                t = max(t - step_len, interval.x);
                continue;
            }
            t += step_len;
            continue;
        }

        // A ray grazing a storm's thin anvil for kilometres would light hundreds of samples: past a ceiling the rest of
        // it is left unmarched (first stills: 10 ms a frame with one cell in view).
        lit = lit + 1;
        if lit > CLOUD_MAX_LIT {
            break;
        }
        let slab = slab_density(p, false);
        var body = StormSample(0.0, vec3<f32>(0.0), 1e9);
        if hits > 0 {
            body = storm_sample(p, false);
        }
        let density = slab + body.density;
        if density <= 0.0 {
            empty_runs += 1;
            if empty_runs >= CLOUD_EMPTY_BEFORE_ESCAPE {
                inside = false;
            }
            t += step_len;
            continue;
        }
        empty_runs = 0;

        // Precipitating columns absorb more, which is what makes the base of a
        // rain cloud read as dark rather than merely thick.
        var raining = 1.0;
        if slab > 0.0 {
            raining = 1.0 + sample_weather(p.xz).b * cp.precipitation * 2.0;
        }
        let extinction = (slab * raining + body.density) * 0.05;
        // Inside a storm body the light is marched at the body's own scale. Where the body is thin (a rain shaft is
        // crossed in twenty samples that all see the same sky) the light of two samples back still holds.
        var optical_depth = 0.0;
        if body.density > 0.0 {
            if body_light_age >= 2 || extinction * step_len > 0.3 {
                body_light = storm_light_march(p, light_dir, body.step) * 4.0;
                body_light_age = 0;
            } else {
                body_light_age = body_light_age + 1;
            }
            optical_depth = body_light;
        } else {
            optical_depth = light_march(p, light_dir) * 4.0;
        }
        let sun_energy = sun_octaves(optical_depth, cos_theta);

        // Radiance arriving at the sample: the sun through its octaves, the sun again as diffused light (Lambertian,
        // half-lit on average, and only where the cloud is dense enough to diffuse: wisps stay single-scatter), and
        // the sky's fill, which reaches inside as diffused light does.
        let diffusion = cloud_diffusion(optical_depth);
        let thick = 1.0 - exp(-4.0 * density);
        var sun_here = sun_e;
        if above_air {
            // The sun at this sample, through the air above it and past the planet's shadow where it stands.
            sun_here = cloud_light_at(p, r, sun, 0.0047) * cp.sun_illuminance;
        }
        if planet_night {
            sun_here = vec3<f32>(0.0);
        }
        var luminance = sun_here * (sun_energy + CLOUD_DIFFUSE * thick * diffusion * 0.5 / CLOUD_PI)
            + sky_fill * cp.ambient * diffusion;
        if planet_on {
            let planet_here = cloud_light_at(p, r, planet_dir, 0.06) * cp.planet_light.rgb;
            let planet_energy = select(0.0, sun_octaves(optical_depth, cos_planet), planet_night);
            luminance += planet_here * (planet_energy + CLOUD_DIFFUSE * thick * diffusion * 0.5 / CLOUD_PI);
        }
        if flashes && storm.counts.y > 0u {
            luminance += storm_flashes(p, density);
        }

        // Far cloud thins into the air in front of it (`far_fade`), and past three lengths there is none to march.
        var thinning = 1.0;
        if cp.far_fade > 0.0 {
            thinning = exp(-t / cp.far_fade);
            if thinning < 0.05 {
                break;
            }
        }
        let step_transmittance = exp(-extinction * thinning * step_len);
        // Analytic segment integration with the scattering coefficient in (sigma_s = albedo sigma_t): what the
        // segment scatters toward the eye is albedo x radiance x its opacity. Water scatters nearly all it meets;
        // a storm body may scatter less, and by colour (dust).
        let albedo = (vec3<f32>(CLOUD_ALBEDO) * slab + body.albedo * body.density) / density;
        let integrated = albedo * luminance * (1.0 - step_transmittance);
        scatter += transmittance * integrated;

        let weight = transmittance * (1.0 - step_transmittance);
        depth_accum += t * weight;
        depth_weight += weight;

        transmittance *= step_transmittance;
        t += step_len;
    }

    let depth = select(interval.x, depth_accum / max(depth_weight, 1e-6), depth_weight > 1e-6);
    return CloudRay(scatter * cp.brightness, transmittance, depth);
}

@compute @workgroup_size(8, 8, 1)
fn march(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(scatter_out);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // A clear sky costs one store per quarter-res pixel and nothing else.
    if cp.density <= 0.0 && storm.counts.x == 0u {
        textureStore(scatter_out, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Reversed-Z (Sunscar, 6 Nov 2026): depth 1 is the near plane and 0 infinity, where a perspective's point has
    // w = 0 and its xyz is the ray's direction (taken undivided, an orthographic view's far plane less its near).
    let near = cp.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far = cp.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let own_dir = normalize(far.xyz - near.xyz / near.w * far.w);
    // Under a closed deck a texel that sees sky follows the deck to the horizon (`closed_dir`). Whether it sees sky is
    // known once its depth taps are read, below; a texel that is all ground goes back to its own way there.
    var ray_dir = closed_dir(own_dir);

    var interval = cloud_interval(cp.camera_pos, ray_dir);
    if interval.y <= interval.x {
        textureStore(scatter_out, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Occlusion by geometry, over the texel's whole footprint. The depth buffer
    // is full resolution and this pass is quarter. A point sample at the texel
    // centre stamped "no cloud" into every texel whose centre fell on a leaf,
    // so through a canopy only the few texels centred on sky carried cloud and
    // the composite rebuilt the sky from them in visible blocks. Now a texel is
    // clamped by its farthest occluder and left unclamped when any of its
    // footprint is sky; the composite decides per full-resolution pixel which
    // geometry the cloud is in front of.
    let depth_dims = textureDimensions(scene_depth);
    let footprint = vec2<f32>(depth_dims) / vec2<f32>(dims);
    let corner = vec2<f32>(gid.xy) * footprint;
    var sky_seen = false;
    var farthest = 0.0;
    var taps = array<vec2<f32>, 5>(vec2(.5, .5), vec2(.1, .1), vec2(.9, .1), vec2(.1, .9), vec2(.9, .9));
    for (var k = 0; k < 5; k = k + 1) {
        let o = taps[k];
        let px = min(vec2<i32>(corner + o * footprint), vec2<i32>(depth_dims) - 1);
        let d = textureLoad(scene_depth, px, 0);
        if d <= 0.0 || d >= 1.0 {
            sky_seen = true;
            break;
        }
        let tap_ndc = (vec2<f32>(px) + .5) / vec2<f32>(depth_dims);
        let occ_clip = cp.inv_view_proj * vec4<f32>(tap_ndc.x * 2.0 - 1.0, 1.0 - tap_ndc.y * 2.0, d, 1.0);
        // Camera-relative (`inv_view_proj` is): the distance is the length.
        farthest = max(farthest, length(occ_clip.xyz / occ_clip.w));
    }
    if !sky_seen {
        if cp.horizon_closure > 0.0 {
            ray_dir = own_dir;
            interval = cloud_interval(cp.camera_pos, ray_dir);
        }
        interval.y = min(interval.y, farthest);
        if interval.y <= interval.x {
            textureStore(scatter_out, vec2<i32>(gid.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
            return;
        }
    }

    var jitter = 0.0;
    if cp.jitter_enabled > 0.5 {
        jitter = ign(vec2<f32>(gid.xy), cp.frame);
    }
    let cloud = march_ray(ray_dir, interval, jitter, true);
    var scatter = cloud.scatter;
    let transmittance = cloud.transmittance;

    // Aerial perspective, at the cloud's own transmittance-weighted depth.
    let cloud_distance = cloud.depth;
    let aerial = cloud_aerial(uv, cloud_distance);
    scatter = scatter * aerial.a + aerial.rgb * (1.0 - transmittance);
    // The target is 16-bit float: cloud beside a flash can pass its ceiling, and an infinity there becomes a NaN in
    // the history's variance and black in the picture.
    scatter = min(scatter, vec3<f32>(6.0e4));

    textureStore(
        scatter_out,
        vec2<i32>(gid.xy),
        vec4<f32>(scatter, clamp(transmittance, 0.0, 1.0)),
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// The sky light's clouds
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// The march for each texel of a small cube round the camera, which `ibl_gen.wgsl` lays over the sky before it
// prefilters. The ground's ambient, its specular and every GI miss then see the deck the camera sees. Before, an overcast
// dome pinned to the clear sky's zenith lit the land 3.5-5 stops under its own clouds (Sunscar, 5 Nov 2026).
@group(0) @binding(16) var env_out: texture_storage_2d_array<rgba16float, write>;

/// `ibl_gen.wgsl`'s `cube_dir`, so a capture texel and the environment texel it lands on look the same way.
fn env_cube_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {
    let st = uv * 2.0 - vec2(1.0);
    var d: vec3<f32>;
    switch face {
        case 0u:  { d = vec3( 1.0,  -st.y, -st.x); }
        case 1u:  { d = vec3(-1.0,  -st.y,  st.x); }
        case 2u:  { d = vec3( st.x,  1.0,   st.y); }
        case 3u:  { d = vec3( st.x, -1.0,  -st.y); }
        case 4u:  { d = vec3( st.x, -st.y,  1.0);  }
        default:  { d = vec3(-st.x, -st.y, -1.0);  }
    }
    return normalize(d);
}

@compute @workgroup_size(8, 8, 1)
fn cloud_env(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(env_out);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    // The capture closes a deck to the horizon as the picture does: the far land and sea fade toward what it holds.
    let dir = closed_dir(env_cube_dir(gid.z, (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims)));
    var out = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let interval = cloud_interval(cp.camera_pos, dir);
    if (cp.density > 0.0 || storm.counts.x > 0u) && interval.y > interval.x {
        // Without the flashes: the sky light gathers over seconds, and lightning lasts a few frames (the land takes
        // it as a light of its own).
        let cloud = march_ray(dir, interval, 0.5, false);
        out = vec4<f32>(cloud.scatter, clamp(cloud.transmittance, 0.0, 1.0));
    }
    textureStore(env_out, vec2<i32>(gid.xy), i32(gid.z), out);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Cloud shadows
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A world-XZ transmittance field, centred on the camera.
///
/// The Beer Shadow Map shape Epic recommends for ground-level viewing, reduced
/// to its useful core: one scalar per column saying how much sun reaches the
/// ground there. Terrain and water read it in `shading.wgsl`, so a cloud's
/// shadow crosses a beach onto the sea without either surface knowing what a
/// cloud is.
@compute @workgroup_size(8, 8, 1)
fn cloud_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(shadow_out);
    if gid.x >= dims.x || gid.y >= dims.y {
        return;
    }
    if (cp.density <= 0.0 && storm.counts.x == 0u) || cp.shadow_strength <= 0.0 {
        textureStore(shadow_out, vec2<i32>(gid.xy), vec4<f32>(1.0));
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let ground = vec2<f32>(
        cp.camera_pos.x + (uv.x * 2.0 - 1.0) * cp.shadow_extent,
        cp.camera_pos.z + (uv.y * 2.0 - 1.0) * cp.shadow_extent,
    );

    var sun = normalize(cp.sun_direction);
    if cp.planet_light.w > 0.5 && cp.planet_direction.w > 0.5 {
        // By night the deck's shadow on the land is the planet's, the night's light.
        sun = normalize(cp.planet_direction.xyz);
    }
    if sun.y <= 0.02 {
        // A sun on or below the horizon casts no cloud shadow worth the name,
        // and the slab intersection degenerates.
        textureStore(shadow_out, vec2<i32>(gid.xy), vec4<f32>(1.0));
        return;
    }

    let origin = vec3<f32>(ground.x, 0.0, ground.y);
    let enter = (cp.layer_bottom - origin.y) / sun.y;
    let exit = (cp.layer_bottom + cp.layer_thickness - origin.y) / sun.y;
    let steps = 8;
    let step_len = (exit - enter) / f32(steps);
    var optical_depth = 0.0;
    for (var i = 0; i < steps; i = i + 1) {
        let t = enter + (f32(i) + 0.5) * step_len;
        optical_depth += slab_density(origin + sun * t, true) * step_len * 0.05;
    }
    // The storm bodies between this ground and the sun, over their own reach (they stand under and over the slab).
    var near = 1e30;
    var far = -1e30;
    for (var i = 0u; i < storm.counts.x; i = i + 1u) {
        let v = storm.volumes[i];
        if storm_is_modifier(v) {
            continue;
        }
        let c = storm_cylinder(v, origin, sun);
        if c.y > c.x {
            near = min(near, c.x);
            far = max(far, c.y);
        }
    }
    if far > near {
        let body_step = (far - near) / 12.0;
        for (var i = 0; i < 12; i = i + 1) {
            optical_depth += storm_sample(origin + sun * (near + (f32(i) + 0.5) * body_step), true).density * body_step * 0.05;
        }
    }
    let transmittance = mix(1.0, exp(-optical_depth), clamp(cp.shadow_strength, 0.0, 1.0));
    textureStore(shadow_out, vec2<i32>(gid.xy), vec4<f32>(transmittance));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// History (22 Oct 2026)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// The developer: "the clouds are too blocky and choppy". The march is a
// quarter of the scene and takes few steps, and its ray start was the same
// every frame: banding that stood still on the screen while the sky moved
// behind it, and no TAA to settle it (a game running FSR has TAA off, and FSR
// has no motion for a sky pixel). Now each frame starts its rays somewhere
// else (`cp.frame`) and this pass keeps most of what the frames before found,
// as Schneider's and Hillaire's clouds do: last frame's resolved clouds,
// looked up where this texel's cloud was a frame ago, held inside the spread
// of what the march sees round it now (variance clipping, Salvi 2016) so a
// cloud that moved or a tree that crossed it leaves no trail.

struct ResolveParams {
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    /// Share of last frame kept. Zero on the first frame, after a resize and
    /// after a cut: this frame's march alone.
    keep: f32,
    layer_bottom: f32,
    layer_top: f32,
    max_distance: f32,
    _pad0: f32,
}

@group(0) @binding(12) var resolve_current: texture_2d<f32>;
@group(0) @binding(13) var resolve_history: texture_2d<f32>;
@group(0) @binding(14) var resolve_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(15) var<uniform> rp: ResolveParams;

/// How far along the ray the cloud is, near enough to reproject by: the
/// layer's near face, or from inside it half its thickness. A cloud is
/// hundreds of metres off at the least, so a walking camera's parallax is a
/// small part of a texel whichever depth inside it the cloud really has.
fn resolve_distance(origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    if origin.y >= rp.layer_bottom && origin.y <= rp.layer_top {
        return min((rp.layer_top - rp.layer_bottom) * 0.5, rp.max_distance);
    }
    if abs(dir.y) < 1e-4 {
        return rp.max_distance;
    }
    let t0 = (rp.layer_bottom - origin.y) / dir.y;
    let t1 = (rp.layer_top - origin.y) / dir.y;
    if max(t0, t1) <= 0.0 {
        return rp.max_distance;
    }
    return clamp(max(min(t0, t1), 0.0), 1.0, rp.max_distance);
}

/// Last frame's resolved clouds at `uv`, bilinear.
fn resolve_history_at(uv: vec2<f32>, dims: vec2<i32>) -> vec4<f32> {
    let p = uv * vec2<f32>(dims) - 0.5;
    let base = floor(p);
    let f = p - base;
    let lo = clamp(vec2<i32>(base), vec2<i32>(0), dims - 1);
    let hi = clamp(vec2<i32>(base) + 1, vec2<i32>(0), dims - 1);
    let a = textureLoad(resolve_history, vec2<i32>(lo.x, lo.y), 0);
    let b = textureLoad(resolve_history, vec2<i32>(hi.x, lo.y), 0);
    let c = textureLoad(resolve_history, vec2<i32>(lo.x, hi.y), 0);
    let d = textureLoad(resolve_history, vec2<i32>(hi.x, hi.y), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(8, 8, 1)
fn cloud_resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(resolve_out);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let dims = vec2<i32>(size);
    let here = vec2<i32>(gid.xy);
    let current = textureLoad(resolve_current, here, 0);
    if rp.keep <= 0.0 {
        textureStore(resolve_out, here, current);
        return;
    }

    // The spread of what this frame's march found round this texel.
    var mean = vec4<f32>(0.0);
    var square = vec4<f32>(0.0);
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let s = textureLoad(resolve_current, clamp(here + vec2<i32>(x, y), vec2<i32>(0), dims - 1), 0);
            mean += s;
            square += s * s;
        }
    }
    mean /= 9.0;
    let spread = sqrt(max(square / 9.0 - mean * mean, vec4<f32>(0.0)));
    let low = min(mean - 1.25 * spread, current);
    let high = max(mean + 1.25 * spread, current);

    // Where this texel's cloud was on the screen a frame ago.
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Reversed-Z: the near plane at 1, the ray's direction at 0 (as in the march above).
    let near = rp.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let far = rp.inv_view_proj * vec4<f32>(ndc, 0.0, 1.0);
    let dir = normalize(far.xyz - near.xyz / near.w * far.w);
    // Camera-relative, as `prev_view_proj` takes it (it carries the camera's step).
    let world = dir * resolve_distance(rp.camera_pos, dir);
    let before = rp.prev_view_proj * vec4<f32>(world, 1.0);
    if before.w <= 1e-6 {
        textureStore(resolve_out, here, current);
        return;
    }
    let before_uv = vec2<f32>(before.x / before.w * 0.5 + 0.5, 0.5 - before.y / before.w * 0.5);
    if any(before_uv < vec2<f32>(0.0)) || any(before_uv > vec2<f32>(1.0)) {
        textureStore(resolve_out, here, current);
        return;
    }
    let history = resolve_history_at(before_uv, dims);
    // A NaN or an infinity (an f16 overflow) is not trusted. Tested on the
    // exponent's bits: Vulkan lowers `x != x` to an ordered compare, false for
    // a NaN, so the obvious test never fires there.
    let bits = bitcast<vec4<u32>>(history) & vec4<u32>(0x7f800000u);
    if any(bits == vec4<u32>(0x7f800000u)) {
        textureStore(resolve_out, here, current);
        return;
    }
    textureStore(resolve_out, here, mix(current, clamp(history, low, high), rp.keep));
}
