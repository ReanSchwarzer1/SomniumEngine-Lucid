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
    /// 0 rain, 1 snow, 2 hail.
    kind: f32,
    /// The sun's illuminance by colour, lux, before any cloud shadow.
    sun_light: vec3<f32>,
    /// The mean drop's diameter, metres.
    size: f32,
    tint: vec3<f32>,
    /// A scale on how much of the picture the drops cover (1 as physics gives it).
    opacity: f32,
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
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) drop: u32) -> VOut {
    var ends = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
    var sides = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
    let end = ends[vertex];
    let side = sides[vertex];

    let h = hash3(drop);
    let k = hash3(drop ^ 0x9E3779B9u);
    // Most drops are small and a few are large (Marshall and Palmer); a larger drop falls faster.
    let big = k.x * k.x;
    let diameter = pp.size * mix(0.5, 2.2, big);
    let fall = pp.velocity * mix(0.75, 1.3, big);
    // Its wander: a velocity for its streak, and that velocity's own integral for where it is, which stays within a
    // metre of its fall. (The two were once one sum multiplied by the clock: snow drifted faster every second.)
    let wa = vec3<f32>(pp.time * 1.3 + k.y * 40.0, pp.time * 0.9 + k.z * 40.0, pp.time * 1.1 + k.y * 23.0);
    let velocity = fall + pp.wander * vec3<f32>(sin(wa.x), 0.3 * sin(wa.y), cos(wa.z));
    let strayed = pp.wander * vec3<f32>(-cos(wa.x) / 1.3, -0.3 * cos(wa.y) / 0.9, sin(wa.z) / 1.1);

    // Where it is: its own start, carried by its fall, wrapped into the box round the eye.
    let box = pp.extent * 2.0;
    let u = fract(h + (fall * pp.time + strayed) / box - pp.centre / box + 0.5) - 0.5;
    let head = pp.centre + u * box - pp.camera_pos;
    let fade = smoothstep(0.0, 0.15, 1.0 - 2.0 * max(max(abs(u.x), abs(u.y)), abs(u.z)));

    // The streak: from where it was when the shutter opened to where it is.
    let travel = (velocity - pp.camera_velocity) * pp.exposure;
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
    let cover = clamp(diameter / (length_m + diameter), 0.0, 1.0) * (diameter / width) * fade * pp.opacity;

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
        radiance = 0.5 * (sky + land) + sun * hg(dot(toward, normalize(pp.sun_direction)), 0.8);
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
    out.across = side;
    out.radiance = radiance * pp.tint;
    out.cover = cover;
    return out;
}

struct FOut {
    @location(0) colour: vec4<f32>,
    /// FSR's reactive mask, as the sprites write it.
    @location(1) reactive: vec4<f32>,
}

@fragment
fn fs_main(in: VOut) -> FOut {
    // Round across its width: densest down the middle.
    let cover = min(in.cover * (1.0 - in.across * in.across) * 1.5, 1.0);
    var out: FOut;
    out.colour = vec4<f32>(in.radiance * cover, cover);
    out.reactive = vec4<f32>(clamp(cover * 4.0, 0.0, 1.0), 0.0, 0.0, 1.0);
    return out;
}
