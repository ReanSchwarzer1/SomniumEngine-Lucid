// Rain, snow and hail near the eye (`pass/precipitation.rs`; Sunscar weather, 11 Nov 2026).
//
// Each drop is its own instance, with no simulation: where it is comes from its number and the clock (a start in a
// box round the eye, carried by its own fall and wrapped back into the box), so the field costs nothing on the CPU
// and is the same field however the camera moves through it.
//
// What a camera sees of a falling drop is a streak as long as the drop travels while the shutter is open, and almost
// transparent: a 2 mm drop at 9 m/s covers any point of its 18 cm streak for a hundredth of the exposure. It is
// bright only because a drop is a lens: it shows the whole sky and land behind it, so against a dark wall it is a
// pale line and against a bright sky it is nothing. Toward the sun, and in lightning, it glitters, because a drop
// throws most of the light it meets forward. Snow is the opposite: slow, opaque, white, lit as any white thing is.
// (Garg and Nayar, "Photorealistic Rendering of Rain Streaks", 2006; Marshall and Palmer 1948 for the sizes.)

struct PrecipParams {
    /// Camera-relative, unjittered.
    view_proj: mat4x4<f32>,
    /// Its inverse: a pixel's depth back to where it is, camera-relative.
    inverse_view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    /// Radians a pixel spans.
    pixel_angle: f32,
    /// The box's centre (world) and the clock, seconds.
    centre: vec3<f32>,
    time: f32,
    /// The box's half-size, and how many drops.
    extent: vec3<f32>,
    count: f32,
    /// The mean drop's velocity (world, metres a second) and the exposure, seconds.
    velocity: vec3<f32>,
    exposure: f32,
    /// The camera's own velocity (a streak is the drop's path across the picture), and how far a drop wanders from
    /// its fall, metres a second (snow drifts).
    camera_velocity: vec3<f32>,
    wander: f32,
    sun_direction: vec3<f32>,
    /// 0 rain, 1 snow, 2 hail, 3 dust.
    kind: f32,
    /// The sun's illuminance by colour, lux, before any cloud shadow.
    sun_light: vec3<f32>,
    /// The mean drop's diameter, metres.
    size: f32,
    tint: vec3<f32>,
    /// A scale on how much of the picture the drops cover (1 as physics gives it).
    opacity: f32,
    /// Drops in the near, middle and far box (three and nine times the near one, hollow), and the splashes' strength.
    layers: vec4<f32>,
    /// What one drawn drop of each box stands for, and splashes a square metre a second.
    folds: vec4<f32>,
    /// x: 1 when the eye is under a roof (what the picture cannot show above is taken as covered).
    extra: vec4<f32>,
    /// Lightning and lamps: xyz where, w 1 if lit; rgb candela.
    flash_pos: array<vec4<f32>, 4>,
    flash_color: array<vec4<f32>, 4>,
}

@group(0) @binding(0) var<uniform> pp: PrecipParams;
@group(0) @binding(1) var env_cube: texture_cube<f32>;
@group(0) @binding(2) var env_sampler: sampler;
@group(0) @binding(3) var cloud_shadow_tex: texture_2d<f32>;
/// centre x, z, half-extent, strength (`CloudPass::shadow_params`).
@group(0) @binding(4) var<uniform> cloud_shadow_params: vec4<f32>;
@group(0) @binding(5) var scene_depth: texture_depth_2d;

const PRECIP_PI: f32 = 3.14159265;
/// The thinnest a streak is drawn, pixels.
const MIN_PIXELS: f32 = 0.75;

fn pcg(n: u32) -> u32 {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

/// Three numbers 0..1 from one.
fn hash3(n: u32) -> vec3<f32> {
    let a = pcg(n);
    let b = pcg(a);
    let c = pcg(b);
    return vec3<f32>(f32(a >> 8u), f32(b >> 8u), f32(c >> 8u)) / 16777216.0;
}

fn hg(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let d = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / (4.0 * PRECIP_PI * max(d * sqrt(max(d, 1e-6)), 1e-6));
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    /// Across the streak, -1..1.
    @location(0) across: f32,
    /// Its radiance, and how much of the picture behind it covers.
    @location(1) radiance: vec3<f32>,
    @location(2) cover: f32,
    /// Along the streak, -1..1 over the whole quad (its round caps included).
    @location(3) along: f32,
    /// The share of the quad's length that is the streak's straight middle; the rest is its two round caps.
    @location(4) @interpolate(flat) middle: f32,
    /// 1 for a flake (a soft round fleck), 0 for a drop.
    @location(5) @interpolate(flat) flake: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) drop: u32) -> VOut {
    var ends = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    var sides = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let end = ends[vertex];
    let side = sides[vertex];

    let h = hash3(drop);
    let k = hash3(drop ^ 0x9E3779B9u);
    // Its box: the near one, or a hollow one three or nine times as large round it (sunscar-particles, 26 Nov).
    var layer = 0u;
    var scale = 1.0;
    var fold = pp.folds.x;
    if f32(drop) >= pp.layers.x + pp.layers.y {
        layer = 2u;
        scale = 9.0;
        fold = pp.folds.z;
    } else if f32(drop) >= pp.layers.x {
        layer = 1u;
        scale = 3.0;
        fold = pp.folds.y;
    }
    // Most drops are small and a few are large (Marshall and Palmer); a larger drop falls faster.
    let big = k.x * k.x;
    let diameter = pp.size * mix(0.5, 2.2, big);
    let fall = pp.velocity * mix(0.75, 1.3, big);
    // Its wander: a velocity for its streak, and that velocity's own integral for where it is, which stays within a
    // metre of its fall. (The two were once one sum multiplied by the clock: snow drifted faster every second.)
    let wa = vec3<f32>(pp.time * 1.3 + k.y * 40.0, pp.time * 0.9 + k.z * 40.0, pp.time * 1.1 + k.y * 23.0);
    var velocity = fall + pp.wander * vec3<f32>(sin(wa.x), 0.3 * sin(wa.y), cos(wa.z));
    var strayed = pp.wander * vec3<f32>(-cos(wa.x) / 1.3, -0.3 * cos(wa.y) / 0.9, sin(wa.z) / 1.1);
    // Gusts: the wind's share of the fall swells and slackens, mostly together, so the slant breathes.
    let gust_w = 0.7 + 0.3 * k.z;
    let gust_a = pp.time * gust_w + k.y * 1.2;
    let wind_xz = vec3<f32>(fall.x, 0.0, fall.z) * 0.35;
    velocity += wind_xz * sin(gust_a);
    strayed += -wind_xz * cos(gust_a) / gust_w;
    // A flake flutters: a small fast spiral on its slow wander, and it glints as it turns.
    var glint = 1.0;
    if pp.kind > 0.5 && pp.kind < 1.5 {
        let fw = 9.0 + 9.0 * k.x;
        let fa = pp.time * fw + k.z * 31.0;
        let fr = 0.025 + 0.03 * k.y;
        velocity += fr * fw * vec3<f32>(cos(fa), 0.0, -sin(fa));
        strayed += fr * vec3<f32>(sin(fa), 0.0, cos(fa));
        glint = 0.7 + 0.45 * max(sin(fa * 0.5 + k.x * 7.0), 0.0);
    }
    if pp.kind > 2.5 {
        // A grain on the wind rises and sinks on the eddies, more than it falls (sunscar-particles, 27 Nov).
        let ew = 1.5 + 2.0 * k.z;
        let ea = pp.time * ew + k.x * 17.0;
        let er = 0.15 + 0.25 * k.y;
        velocity.y += er * ew * cos(ea);
        strayed.y += er * sin(ea);
    }

    // Where it is: its own start, carried by its fall, wrapped into the box round the eye.
    let box = pp.extent * 2.0 * scale;
    let u = fract(h + (fall * pp.time + strayed) / box - pp.centre / box + 0.5) - 0.5;
    let head = pp.centre + u * box - pp.camera_pos;
    var fade = smoothstep(0.0, 0.15, 1.0 - 2.0 * max(max(abs(u.x), abs(u.y)), abs(u.z)));
    if layer > 0u {
        // Hollow: inside the box within it the smaller box's drops are already falling.
        let inner = abs(u * 2.0 * scale);
        if all(inner < vec3<f32>(scale / 3.0)) {
            fade = 0.0;
        }
    }
    // Under a roof or a canopy no drop falls: the picture's depth says so where it shows what is above.
    fade *= open_above(head);

    // The streak: from where it was when the shutter opened to where it is.
    // Snow in a wind drawn over a film's whole shutter is a field of long dashes (the p3 stills): flakes are seen as
    // flecks, as a quarter of it shows them.
    var shutter = pp.exposure;
    if pp.kind > 0.5 && pp.kind < 1.5 {
        shutter = pp.exposure * 0.12;
    } else if pp.kind > 2.5 {
        // Grains are seen as short dashes along the wind, not threads.
        shutter = pp.exposure * 0.35;
    }
    let travel = (velocity - pp.camera_velocity) * shutter;
    let tail = head - travel;
    let here = mix(head, tail, end);
    let distance = max(length(here), 0.05);
    let width = max(diameter, MIN_PIXELS * pp.pixel_angle * distance);
    let length_m = length(travel);
    var along = vec3<f32>(0.0, 1.0, 0.0);
    if length_m > 1e-4 {
        along = -travel / length_m;
    }
    var aside = cross(along, here / distance);
    let aside_len = length(aside);
    if aside_len > 1e-4 {
        aside = aside / aside_len;
    } else {
        aside = vec3<f32>(1.0, 0.0, 0.0);
    }
    // Capped by half its width at each end, so a slow flake is a dot, not a hairline.
    let at = here + aside * (side * width * 0.5) + along * ((0.5 - end) * width);

    // How much of the picture behind a point on the streak the drop covers: the share of the exposure it is over
    // that point, thinned again where it is drawn wider than it is.
    let cover = min(clamp(diameter / (length_m + diameter), 0.0, 1.0) * (diameter / width) * fold, 1.0) * fade;

    // Its light. The sky and land all round it (a drop is a lens; a flake is lit by them), the sun where the clouds
    // let it through, and lightning.
    let sky = textureSampleLevel(env_cube, env_sampler, vec3<f32>(0.0, 1.0, 0.0), 6.0).rgb;
    let land = textureSampleLevel(env_cube, env_sampler, vec3<f32>(0.0, -1.0, 0.0), 6.0).rgb;
    var sun = pp.sun_light;
    if cloud_shadow_params.w > 0.0 {
        let uv = (pp.camera_pos.xz - cloud_shadow_params.xy) / max(cloud_shadow_params.z, 1.0) * 0.5 + 0.5;
        sun *= textureSampleLevel(cloud_shadow_tex, env_sampler, uv, 0.0).r;
    }
    let toward = here / distance;
    var radiance = vec3<f32>(0.0);
    if pp.kind < 0.5 {
        // Rain: what it refracts, and the light it throws forward.
        // Mostly the sky: a drop seen against dark ground shows the bright sky behind and above the eye, bent
        // through it, and so a shower reads pale against a dark field (the lead's 27 Nov read of p4: too faint).
        radiance = 0.8 * sky + 0.2 * land + sun * hg(dot(toward, normalize(pp.sun_direction)), 0.8);
    } else if pp.kind > 2.5 {
        // Dust: a matte grain, lit by the sky and the sun, brightest toward the sun (it throws light forward like
        // haze); its colour is the tint, the soil's.
        let mu = dot(toward, normalize(pp.sun_direction));
        radiance = 0.9 * (0.7 * sky + 0.3 * land) / PRECIP_PI + sun * (0.2 / PRECIP_PI + 0.5 * hg(mu, 0.6));
    } else {
        // Snow and hail: white, lit from above and by what light gets through them.
        radiance = 0.85 * (sky + sun * (0.25 + 0.5 * max(dot(toward, normalize(pp.sun_direction)), 0.0)) / PRECIP_PI);
    }
    for (var i = 0; i < 4; i = i + 1) {
        if pp.flash_pos[i].w > 0.5 {
            let to_flash = pp.flash_pos[i].xyz - (here + pp.camera_pos);
            let lux = pp.flash_color[i].rgb / (dot(to_flash, to_flash) + 25.0);
            if pp.kind < 0.5 {
                // A drop throws a flash's light forward in a narrow lobe: the rain blazes round the bolt, not all
                // over the picture.
                radiance += lux * hg(dot(toward, normalize(to_flash)), 0.92);
            } else {
                radiance += 0.85 * lux / PRECIP_PI;
            }
        }
    }

    var out: VOut;
    out.clip = pp.view_proj * vec4<f32>(at, 1.0);
    // A streak that reaches the eye's own plane projects to a line across the whole picture (the fans of the first
    // stills): such a drop is too near to see. Collapse it off the picture.
    let w_head = (pp.view_proj * vec4<f32>(head, 1.0)).w;
    let w_tail = (pp.view_proj * vec4<f32>(tail, 1.0)).w;
    if min(w_head, w_tail) < 0.3 {
        out.clip = vec4<f32>(4.0, 4.0, 0.5, 1.0);
    }
    out.across = side;
    out.radiance = radiance * pp.tint * glint;
    out.cover = cover;
    out.along = end * 2.0 - 1.0;
    out.middle = length_m / (length_m + width);
    out.flake = select(0.0, 1.0, (pp.kind > 0.5 && pp.kind < 1.5) || pp.kind > 2.5);
    return out;
}

struct FOut {
    @location(0) colour: vec4<f32>,
    /// FSR's reactive mask, as the sprites write it.
    @location(1) reactive: vec4<f32>,
}

@fragment
fn fs_main(in: VOut) -> FOut {
    // A capsule: round across its width and at both ends, densest down the middle (the quad's hard ends drew snow
    // as small white rectangles; sunscar-particles, 27 Nov). A flake is softer still: a round fleck with no edge.
    let cap = max(abs(in.along) - in.middle, 0.0) / max(1.0 - in.middle, 1e-4);
    let r2 = in.across * in.across + cap * cap;
    var body = max(1.0 - r2, 0.0);
    if in.flake > 0.5 {
        body = body * body;
    }
    // The shape after the clamp: a far drop that stands for many covers fully, and its edge must still fall away.
    let cover = min(in.cover * 1.5, 1.0) * body;
    if cover <= 0.0 {
        discard;
    }
    var out: FOut;
    out.colour = vec4<f32>(in.radiance * cover, cover);
    out.reactive = vec4<f32>(clamp(cover * 4.0, 0.0, 1.0), 0.0, 0.0, 1.0);
    return out;
}

// ---- Under cover, and the splashes (sunscar-particles, 26 Nov 2026) ----

/// Where a camera-relative point falls in the depth picture: pixel, and its own depth (reversed: 1 near, 0 far). w < 0
/// when it is off the picture or behind the eye.
fn to_screen(p: vec3<f32>) -> vec4<f32> {
    let c = pp.view_proj * vec4<f32>(p, 1.0);
    if c.w <= 1e-4 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let ndc = c.xyz / c.w;
    if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }
    let dims = vec2<f32>(textureDimensions(scene_depth));
    return vec4<f32>((ndc.x * 0.5 + 0.5) * dims.x, (0.5 - ndc.y * 0.5) * dims.y, ndc.z, 1.0);
}

/// 1 if nothing the picture shows lies over a point (camera-relative), 0 if a roof or a canopy does: steps up from it
/// and asks whether the scene stands in front of a step, near it. Off the picture it is open, unless the eye itself
/// is under a roof. (A screen-space guess: a post in front of a step can hide a drop that should fall.)
fn open_above(p: vec3<f32>) -> f32 {
    let steps = array<f32, 5>(0.5, 1.2, 2.5, 5.0, 10.0);
    let dims = vec2<i32>(textureDimensions(scene_depth));
    for (var i = 0; i < 5; i = i + 1) {
        let q = to_screen(p + vec3<f32>(0.0, steps[i], 0.0));
        if q.w < 0.0 {
            if pp.extra.x > 0.5 {
                return 0.0;
            }
            continue;
        }
        let pix = clamp(vec2<i32>(q.xy), vec2<i32>(0), dims - vec2<i32>(1));
        let d = textureLoad(scene_depth, pix, 0);
        // Reversed depth is 1 / distance: the scene in front of the step, by more than 2% and less than four times.
        if d > q.z * 1.02 && d < q.z * 4.0 {
            return 0.0;
        }
    }
    return 1.0;
}

@vertex
fn vs_splash(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((vertex << 1u) & 2u), f32(vertex & 2u));
    return vec4<f32>(xy * 2.0 - 1.0, 0.0, 1.0);
}

fn cell_hash(c: vec2<i32>, lane: u32, slot: u32) -> vec3<f32> {
    return hash3(pcg((bitcast<u32>(c.x) * 73856093u) ^ (bitcast<u32>(c.y) * 19349663u) ^ (lane * 83492791u)) ^ (slot * 2654435761u));
}

/// Rain splashing on what is open to the sky: on every surface that faces up within 25 m, a grid of 11 cm cells, each
/// with two splashes in turn at its own moments (as often as the rain lands on it). A splash is a bright dot that
/// opens into a ring in a tenth of a second, lit by the sky as the drops are. Where a splash is smaller than a pixel it
/// is folded into the pixel's share, so far ground fizzes faintly rather than sparkles.
@fragment
fn fs_splash(@builtin(position) frag: vec4<f32>) -> FOut {
    let pix = vec2<i32>(frag.xy);
    let d = textureLoad(scene_depth, pix, 0);
    let dims = vec2<f32>(textureDimensions(scene_depth));
    let ndc = vec2<f32>(frag.x / dims.x * 2.0 - 1.0, 1.0 - frag.y / dims.y * 2.0);
    let hp = pp.inverse_view_proj * vec4<f32>(ndc, max(d, 1e-7), 1.0);
    let p = hp.xyz / hp.w;
    // Derivatives first, while every pixel still runs.
    var n = cross(dpdy(p), dpdx(p));
    var out: FOut;
    out.colour = vec4<f32>(0.0);
    out.reactive = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    let dist = length(p);
    let nl = length(n);
    if d <= 0.0 || dist > 25.0 || nl < 1e-9 {
        return out;
    }
    n = n / nl;
    if dot(n, p) > 0.0 {
        n = -n;
    }
    let up = n.y;
    if up < 0.7 {
        return out;
    }
    let open = open_above(p);
    if open <= 0.0 {
        return out;
    }

    let cell = 0.11;
    let life = 0.11;
    let world = (p.xz + pp.camera_pos.xz) / cell;
    let c = vec2<i32>(floor(world));
    let f = world - floor(world);
    // How often one lane of a cell splashes: half of what lands on a cell, never so often that two overlap.
    let period = max(life * 1.3, 2.0 / max(pp.folds.w * cell * cell, 1e-3));
    // A pixel's footprint on the ground, metres.
    let foot = max(pp.pixel_angle * dist / max(up, 0.2), 0.002);
    var light = 0.0;
    for (var lane = 0u; lane < 2u; lane = lane + 1u) {
        let r = cell_hash(c, lane, 0u);
        let tt = pp.time / period + r.x;
        let slot = u32(floor(tt));
        let tau = fract(tt) * period;
        if tau < life {
            let q = cell_hash(c, lane, slot + 1u);
            let centre = 0.25 + 0.5 * q.xy;
            let at = (f - centre) * cell;
            let age = tau / life;
            let radius = 0.02 * mix(0.5, 1.4, q.z) * sqrt(age);
            let w = max(foot, 0.003);
            let rr = length(at);
            let ring = 1.0 - smoothstep(0.0, w, abs(rr - radius));
            let dot_ = (1.0 - smoothstep(0.0, radius * 0.4 + w, rr)) * (1.0 - age);
            // Smaller than a pixel: the mark's area over the pixel's.
            let thin = min(1.0, 0.004 / w);
            // Mostly the bright crown at the start, a faint ring after: a still catches a fleck, not a doily.
            light += (0.25 * ring + dot_) * (1.0 - age) * thin * mix(0.3, 1.0, q.y);
        }
    }
    let near = 1.0 - smoothstep(14.0, 25.0, dist);
    let cover = clamp(0.5 * light * pp.layers.w * near * open * smoothstep(0.7, 0.85, up), 0.0, 1.0);
    if cover <= 0.0 {
        return out;
    }
    let sky = textureSampleLevel(env_cube, env_sampler, vec3<f32>(0.0, 1.0, 0.0), 6.0).rgb;
    let radiance = 0.9 * sky * pp.tint;
    out.colour = vec4<f32>(radiance * cover, cover * 0.25);
    out.reactive = vec4<f32>(clamp(cover * 4.0, 0.0, 1.0), 0.0, 0.0, 1.0);
    return out;
}
