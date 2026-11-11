// The fragmented sun (Sunscar; sunscar-sky, 12 Nov 2026; the forms of `design/sky/SYNTHESIS.md` part A). The Lid broke:
// its wreck rings the world, and a cloud of it hangs sunward, so the sun is always seen through lanes of its plates and
// dust. Everything here is lit as a thing in space lit by a star, not painted on: across the disc a lane is backlit and
// black; beside the disc only its dust shows, by forward scattering, so the lane glows as a streak that fades over a few
// degrees from the sun.
//
// Three forms, by `variant` (the clock's `sun_wreck`, or `SOMNIUM_SUN_FORM`):
//   0 the cut sun (the default): a disc 1.0 degree wide cut into three or four pieces by two drifting lanes, a third
//     thin one clipping its limb now and then; about 30% of it hidden. The light is the whole sun's.
//   1 five shards: the Heart, 0.55 degrees wide with 53% of the light, and four lesser pieces in a 2 degree circle.
//   2 the true-size sun: a 32 arcminute disc crossed by two bars 4 and 5 arcminutes wide that take 30% of its light
//     (the light itself is cut to match: 56,000 of 80,000 lux).
// The directional light stays one light from the disc's centre, its size the form's disc (so the shadows' penumbra
// follows it). The lanes drift; they never flicker.
//
// Coordinates: the tangent plane round the sun, in degrees, its axes held to the celestial pole, so the wreck turns
// slowly through the day as anything fixed on the sky does.

fn sun_hash2(p: vec2<f32>) -> f32 {
    let q = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    let r = q + dot(q, q.yzx + 33.33);
    return fract((r.x + r.y) * r.z);
}

/// Henyey-Greenstein, per steradian.
fn sun_hg(cos_theta: f32, g: f32) -> f32 {
    let d = max(1.0 + g * g - 2.0 * g * cos_theta, 1e-7);
    return (1.0 - g * g) / (4.0 * 3.14159265 * d * sqrt(d));
}

/// One lane of the wreck: (cover 0..1, dust 0..1) at `p` (degrees from the sun). `angle` its direction, `offset` its
/// centre line's distance from the sun and `width` its opaque core, degrees; `ragged` how far its edges wander (a share
/// of the width); `reach` how far its dust trails on its two sides (degrees, along +/- its direction).
fn wreck_lane(p: vec2<f32>, angle: f32, offset: f32, width: f32, ragged: f32, reach: vec2<f32>, px: f32,
              seed: f32) -> vec2<f32> {
    let along = vec2<f32>(cos(angle), sin(angle));
    let across = vec2<f32>(-along.y, along.x);
    let a = dot(p, along);
    let wander = width * ragged * (0.6 * sin(a * 23.0 + seed * 7.0) + 0.4 * sin(a * 61.0 + seed * 13.0));
    let s = abs(dot(p, across) - offset - wander);
    let half_width = width * 0.5;
    let aa = max(px, 0.002);
    let cover = 1.0 - smoothstep(half_width - aa, half_width + width * 0.25 + aa, s);
    // Its dust beside the disc: a soft glow some three times the core's width, smooth (the edges' wander is the
    // plates', not the dust's), trailing off along the lane, further on one side than the other.
    let soft = abs(dot(p, across) - offset) / (width * 3.0);
    let trail = exp(-abs(a) / select(reach.y, reach.x, a > 0.0));
    let dust = exp(-soft * soft) * trail;
    return vec2<f32>(cover, dust);
}

/// The cut sun's lanes and the true-size sun's bars at `p`: (cover, dust weighted by each lane's dust), `t` minutes.
fn wreck_lanes(p: vec2<f32>, variant: i32, t: f32, px: f32) -> vec2<f32> {
    if variant == 2 {
        // Two members of the Lid's frame, straight-edged: 5 and 4 arcminutes wide, nearly square to each other.
        let a = wreck_lane(p, 0.35 + 0.03 * sin(t / 13.0), 0.07 + 0.05 * sin(t / 9.0 + 1.0), 0.083, 0.02,
                           vec2<f32>(2.0, 0.5), px, 1.0);
        let b = wreck_lane(p, -1.05 + 0.02 * sin(t / 17.0), -0.06 + 0.04 * sin(t / 11.0 + 2.0), 0.067, 0.02,
                           vec2<f32>(0.4, 0.4), px, 2.0);
        return vec2<f32>(1.0 - (1.0 - a.x) * (1.0 - b.x), a.y + 0.2 * b.y);
    }
    // Two lanes of plates and dust crossing at sixty degrees; where they cross drifts in and out of the disc, so it is
    // in three pieces or four. A thin third lane sways past the limb.
    let a = wreck_lane(p, 0.42 + 0.05 * sin(t / 9.0), 0.16 + 0.12 * sin(t / 7.0 + 1.0), 0.14, 0.16,
                       vec2<f32>(3.5, 0.6), px, 3.0);
    let b = wreck_lane(p, -0.62 + 0.04 * sin(t / 13.0), -0.20 + 0.10 * sin(t / 11.0 + 2.0), 0.11, 0.18,
                       vec2<f32>(0.5, 0.9), px, 4.0);
    let c = wreck_lane(p, 1.35 + 0.03 * sin(t / 19.0), 0.56 + 0.14 * sin(t / 5.0 + 4.0), 0.045, 0.25,
                       vec2<f32>(0.8, 0.8), px, 5.0);
    let cover = 1.0 - (1.0 - a.x) * (1.0 - b.x) * (1.0 - c.x);
    return vec2<f32>(cover, a.y + 0.25 * b.y);
}

/// The share of a disc of `radius` degrees the lanes leave open, by 61 fixed points over it (a sunflower spiral): the
/// disc is brightened by its inverse, so it gives the light it stands for.
fn wreck_open_share(radius: f32, variant: i32, t: f32) -> f32 {
    var open = 0.0;
    for (var i = 0; i < 61; i = i + 1) {
        let fi = f32(i) + 0.5;
        let a = fi * 2.3999632;
        let at = vec2<f32>(cos(a), sin(a)) * sqrt(fi / 61.0) * radius;
        open += 1.0 - wreck_lanes(at, variant, t, 0.002).x;
    }
    return max(open / 61.0, 0.3);
}

/// A shard of the sun in the five-shard form: 1 inside, 0 outside (antialiased over `px` degrees), at `p` degrees from
/// the group's centre. A disc of `radius` at `at` cut by two straight breaks, so its edge is part round, part broken.
fn sun_shard(p: vec2<f32>, at: vec2<f32>, radius: f32, cut: vec2<f32>, px: f32) -> f32 {
    let q = p - at;
    let round = 1.0 - smoothstep(radius - px, radius + px, length(q));
    let n1 = vec2<f32>(cos(cut.x), sin(cut.x));
    let n2 = vec2<f32>(cos(cut.y), sin(cut.y));
    let b1 = 1.0 - smoothstep(-px, px, dot(q, n1) - radius * 0.55);
    let b2 = 1.0 - smoothstep(-px, px, dot(q, n2) - radius * 0.7);
    return round * b1 * b2;
}

/// The five shards (form 1), each the sun's own surface brightness: the Heart 0.55 degrees wide, and four lesser
/// pieces (16, 13, 10 and 8% of the light) inside a 2 degree circle, slowly turning about it. Coverage 0..1.
fn five_shards(p: vec2<f32>, t: f32, px: f32) -> f32 {
    var lit = sun_shard(p, vec2<f32>(0.0), 0.275, vec2<f32>(0.6, 2.9), px);
    let turn = t / 40.0;
    lit = max(lit, sun_shard(p, vec2<f32>(cos(0.7 + turn), sin(0.7 + turn)) * 0.66, 0.17, vec2<f32>(2.2, 4.4), px));
    lit = max(lit, sun_shard(p, vec2<f32>(cos(2.3 + turn), sin(2.3 + turn)) * 0.78, 0.152, vec2<f32>(5.1, 0.9), px));
    lit = max(lit, sun_shard(p, vec2<f32>(cos(3.8 + turn), sin(3.8 + turn)) * 0.6, 0.133, vec2<f32>(1.4, 3.6), px));
    lit = max(lit, sun_shard(p, vec2<f32>(cos(5.2 + turn), sin(5.2 + turn)) * 0.86, 0.12, vec2<f32>(3.0, 5.7), px));
    return lit;
}

/// The sun seen along `dir` through the Lid's wreck. `sun_dir` its centre; `radius` the form's disc, radians; `lux` its
/// light on the ground per channel (the air's colour in it); `beside` its light above the air, dimmed by the air along
/// this ray (the lanes are in space); `pole` the celestial pole; `time` seconds; `variant` 0..2. Radiance, cd/m2.
fn fragmented_sun(dir: vec3<f32>, sun_dir: vec3<f32>, radius: f32, lux: vec3<f32>, beside: vec3<f32>, pole: vec3<f32>,
                  time: f32, variant: i32) -> vec3<f32> {
    let c = dot(dir, sun_dir);
    // Nothing reaches past seven degrees.
    if c < 0.99254 {
        return vec3<f32>(0.0);
    }
    let east = normalize(cross(pole, sun_dir) + vec3<f32>(1e-5, 0.0, 0.0));
    let north = cross(sun_dir, east);
    let off = dir - sun_dir * c;
    let p = vec2<f32>(dot(off, east), dot(off, north)) * 57.29578;
    let px = max(length(fwidth(dir)) * 57.29578, 1e-5);
    let t = time / 60.0;
    let r = length(p);
    let disc_deg = radius * 57.29578;

    var light = vec3<f32>(0.0);
    var lanes = vec2<f32>(0.0);
    if variant == 1 {
        // The shards keep sky_detail's sun (0.6 cd/m2 per lux for a 0.2667 degree disc) spread over their area, about
        // 1.9 times the Heart's, so the glare they throw is the whole sun's.
        if r < 1.1 {
            light += lux * (0.6 * 0.2667 * 0.2667 / (0.275 * 0.275 * 1.89)) * five_shards(p, t, px);
        }
        // The dust they shine through, a soft halo fading over a degree and a half.
        light += beside * 0.0006 * exp(-r / 1.5) * sun_hg(c, 0.95);
    } else {
        lanes = wreck_lanes(p, variant, t, px);
        if r < disc_deg + 2.0 * px {
            // Limb darkening (the sun's own, Eddington's grey: 0.4 + 0.6 mu), normalised to the disc's mean; the disc
            // keeps sky_detail's sun spread over this form's disc, brightened by what the lanes leave.
            let rr = r / disc_deg;
            let mu = sqrt(max(1.0 - rr * rr, 0.0));
            let limb = (0.4 + 0.6 * mu) / 0.8;
            let edge = 1.0 - smoothstep(disc_deg - px, disc_deg + px, r);
            let spread = 0.6 * (0.2667 / disc_deg) * (0.2667 / disc_deg);
            light += lux * spread * limb * edge * (1.0 - lanes.x) / wreck_open_share(disc_deg, variant, t);
        }
        // Beside the disc the lanes' dust glows by forward scattering (large grains: g 0.95, so a streak falls tenfold
        // from one degree to six). Over the disc it is nothing against the disc and is left out.
        let tau = select(0.001, 0.0005, variant == 2);
        light += beside * tau * lanes.y * sun_hg(c, 0.95) * smoothstep(disc_deg, disc_deg + 0.15, r);
    }
    // fp16 targets end at 65,504.
    return min(light, vec3<f32>(60000.0));
}

/// The debris ring (the halo road): the Lid's fallen plates and their dust, swept into an orbit round the world in its
/// equatorial plane, from 1.55 to 2.7 world radii out. Seen from the ground at the clock's latitude it is a broad arch
/// across the sky on the side of the celestial equator, its outer edge running a few degrees under the sun's daily path.
/// It is lit by the sun wherever the world's shadow does not fall on it: at dusk the whole arch glows, lit from below the
/// horizon; by night the shadow cuts a dark gap in it opposite the sun. Its dust throws light mostly far forward, so by
/// day the sky hides it but for the arc beside the sun, which glitters where plates turn their faces to it.
/// Radiance, cd/m2, before our own air. `pole` the celestial pole; `sun_dir` toward the sun; `lux` the sun's light above
/// the air (rgb); `time` seconds.
fn debris_ring(dir: vec3<f32>, sun_dir: vec3<f32>, pole: vec3<f32>, lux: vec3<f32>, time: f32) -> vec3<f32> {
    // In world radii: the eye on the surface, the world's centre one radius below it.
    let centre = vec3<f32>(0.0, -1.0, 0.0);
    let toward_plane = dot(dir, pole);
    if abs(toward_plane) < 1e-4 {
        return vec3<f32>(0.0);
    }
    let t = dot(centre, pole) / toward_plane;
    if t <= 0.0 {
        return vec3<f32>(0.0);
    }
    let hit = dir * t - centre;
    let radius = length(hit);
    let inner = 1.55;
    let outer = 2.7;
    if radius < inner || radius > outer {
        return vec3<f32>(0.0);
    }
    // The world's shadow: a cylinder of one radius behind it, away from the sun (the sun is far: its rays are parallel).
    let along_sun = dot(hit, sun_dir);
    let off_axis = length(hit - sun_dir * along_sun);
    let lit = 1.0 - (1.0 - smoothstep(0.97, 1.03, off_axis)) * step(along_sun, 0.0);
    // Its density: a broad band, full to near its outer edge, a soft gap three-fifths out and faint lanes where the
    // plates' orbits crowd and clear; no hard stripes.
    let u = (radius - inner) / (outer - inner);
    let body = smoothstep(0.0, 0.15, u) * (1.0 - smoothstep(0.9, 1.0, u));
    let lanes = 0.85 + 0.15 * sin(u * 29.0 + 1.3) * sin(u * 9.0 + 0.4);
    let gap = 1.0 - 0.55 * exp(-pow((u - 0.63) / 0.04, 2.0));
    let tau = 3.0e-4 * body * lanes * gap;
    // Dust and plates: most of the light far forward (grains and flakes, g 0.97), a tenth broadly (g 0.6).
    let cos_scatter = dot(dir, sun_dir);
    let phase = 0.9 * sun_hg(cos_scatter, 0.97) + 0.1 * sun_hg(cos_scatter, 0.6);
    let slant = max(abs(toward_plane), 0.05);
    var light = lux * tau * phase / slant * lit;
    // Plates turning their faces to the sun: a glint a few times the dust round it, seen only near the sun, where the
    // forward light is strong; each turns in some seconds.
    let e1 = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), pole) + vec3<f32>(0.0, 0.0, 1e-5));
    let e2 = cross(pole, e1);
    let around = atan2(dot(hit, e2), dot(hit, e1));
    // Cells about a pixel across (0.02 by 0.035 degrees seen from the ground), one in two hundred a plate.
    let cell = vec2<f32>(floor(around * 3000.0), floor(u * 600.0));
    let h = sun_hash2(cell + vec2<f32>(5.0, 11.0));
    if h > 0.995 {
        let spin = pow(0.5 + 0.5 * sin(time * (0.4 + 0.8 * sun_hash2(cell)) + h * 50.0), 16.0);
        light += lux * tau * 12.0 * sun_hg(cos_scatter, 0.97) / slant * spin * lit;
    }
    return light;
}
