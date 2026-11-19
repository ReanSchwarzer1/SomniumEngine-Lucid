// Linear HDR sprites; texture alpha, depth, lifetime and colour are independent.
enable wgpu_binding_array;
struct ParticleView {
    view_proj: mat4x4<f32>,
    // _pad0: one pixel's angle, radians (Sunscar's fire: a far card's size).
    camera_right: vec3<f32>, _pad0: f32,
    camera_up: vec3<f32>, _pad1: f32,
    camera_pos: vec3<f32>, _pad2: f32,
    // Sunscar's fire (19 Nov 2026): x the manual EV100 (under -999 when auto exposure meters it), y the camera's near
    // plane (reversed infinite depth: view depth = near / depth), z the EV a fire's own light is seen at, w how far
    // a flame, a bed or a puff fades in front of what it meets, metres.
    fire: vec4<f32>,
}
// The sun as the global pool holds it (`DirectionalLight`, its first 312 bytes; a prefix binds the same buffer).
struct ParticleSun {
    direction: vec3<f32>, _pad0: f32,
    color: vec3<f32>, _pad1: f32,
    view_proj: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,
    shadow_map_size: f32,
    ibl_intensity: f32,
}
struct GpuParticle {
    position: vec3<f32>, size: f32,
    color: vec4<f32>,
    tip_tint: vec3<f32>, aspect: f32,
    uv_rect: vec4<f32>,
    rotation: f32, texture_index: i32, flags: u32, flutter: f32,
    extra: vec4<f32>,
}
@group(0) @binding(0) var<uniform> pview: ParticleView;
@group(0) @binding(1) var<storage, read> particles: array<GpuParticle>;
@group(0) @binding(2) var sprite_sampler: sampler;
@group(1) @binding(4) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(6) var<storage, read> sun: ParticleSun;
// This frame's scene depth (the attachment the pass tests against, read-only) and the auto exposure's reading
// ([0] its multiplier, [1] the adapted EV100, a frame old).
@group(2) @binding(0) var scene_depth: texture_depth_2d;
@group(2) @binding(1) var<storage, read> metered: array<f32>;
const UV: array<vec2<f32>,6> = array<vec2<f32>,6>(vec2(0.,1.),vec2(1.,1.),vec2(1.,0.),vec2(0.,1.),vec2(1.,0.),vec2(0.,0.));
struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) colour: vec4<f32>,
    @location(2) tip: vec3<f32>,
    @location(3) @interpolate(flat) texture_index: i32,
    @location(4) @interpolate(flat) flags: u32,
    @location(5) local_uv: vec2<f32>,
    @location(6) @interpolate(flat) uv_rect: vec4<f32>,
    @location(7) @interpolate(flat) flutter: f32,
    @location(8) view_dir: vec3<f32>,
    @location(9) @interpolate(flat) extra: vec4<f32>,
    @location(10) view_depth: f32,
}
// A flame tongue's turbulence (bit 4).
fn flame_hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
fn flame_noise(p: vec2<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3. - 2. * f);
    return mix(mix(flame_hash(i), flame_hash(i + vec2(1., 0.)), u.x), mix(flame_hash(i + vec2(0., 1.)), flame_hash(i + vec2(1., 1.)), u.x), u.y);
}

// ── Sunscar's fire, 19 Nov 2026 (design/fire/FIRE.md) ─────────────────────────────────────────────────────────────
// A blackbody from 800 to 2,200 K in steps of 100 K (Planck through the CIE 1931 observer; the game's
// tools/fire/fire_book.py): its colour in linear Rec.709 at unit luminance, and its luminance's log2 in cd/m2.
const BB_RGB = array<vec3<f32>, 15>(vec3(4.8649, -0.0386, -0.0919), vec3(4.5980, 0.0410, -0.0942), vec3(4.3193, 0.1240, -0.0953), vec3(4.0502, 0.2039, -0.0944), vec3(3.7996, 0.2780, -0.0911), vec3(3.5704, 0.3456, -0.0854), vec3(3.3626, 0.4065, -0.0771), vec3(3.1751, 0.4612, -0.0663), vec3(3.0060, 0.5101, -0.0532), vec3(2.8535, 0.5539, -0.0379), vec3(2.7157, 0.5931, -0.0206), vec3(2.5909, 0.6283, -0.0016), vec3(2.4776, 0.6599, 0.0190), vec3(2.3745, 0.6883, 0.0410), vec3(2.2805, 0.7139, 0.0641));
const BB_LOG2_CD = array<f32, 15>(-6.9350, -2.3061, 1.4390, 4.5328, 7.1324, 9.3481, 11.2593, 12.9251, 14.3900, 15.6883, 16.8471, 17.8877, 18.8274, 19.6802, 20.4576);
// A candle's foot glows blue (CH and C2 near 431 and 516 nm): linear Rec.709 at unit luminance.
const BLUE_FOOT = vec3<f32>(0.53, 0.88, 3.53);
// The book's encoding (fire_book.py): R and a small book's B are log2(Y / peak) over 16 stops; motion +-12 px.
const BOOK_STOPS = 16.0;
const BOOK_MOTION_PX = 12.0;
// A flame's card is never drawn smaller than this, pixels tall: past it, it is a soft point of its own whole light.
const FAR_PX = 2.5;

fn blackbody_rgb(t: f32) -> vec3<f32> {
    let x = clamp((t - 800.) / 100., 0., 13.999);
    let i = u32(floor(x));
    return max(mix(BB_RGB[i], BB_RGB[i + 1u], fract(x)), vec3(0.));
}
fn blackbody_cd(t: f32) -> f32 {
    let x = clamp((t - 800.) / 100., 0., 13.999);
    let i = u32(floor(x));
    return exp2(mix(BB_LOG2_CD[i], BB_LOG2_CD[i + 1u], fract(x)));
}
// A fire's own light as an eye fixed on it sees it: when the camera exposes darker than `fire.z` (the auto exposure's
// EV in use, [1]: after its adaptation, compensation and clamp), the fire's own pixels are taken down by the
// difference; its light on the world is never changed. A night at EV -2 against EV 5 keeps a flame's core white, its
// edges yellow and orange and its tips red instead of one white blob.
fn fire_eye() -> f32 {
    let ev = select(metered[1], pview.fire.x, pview.fire.x > -999.0);
    return exp2(min(ev - pview.fire.z, 0.0));
}
// How far in front of what it meets this fragment is, as a 0..1 fade over `reach` metres.
fn soft_depth(position: vec4<f32>, view_depth: f32, reach: f32) -> f32 {
    let d = textureLoad(scene_depth, vec2<i32>(position.xy), 0);
    let scene = pview.fire.y / max(d, 1.0e-9);
    return clamp((scene - view_depth) / max(reach, 1.0e-3), 0., 1.);
}
fn book_decode(code: f32) -> f32 {
    return select(0., exp2((code - 1.) * BOOK_STOPS), code > 1.5 / 255.);
}
fn book_texel(index: i32, frame: u32, band0: u32, cols: u32, cell: vec2<f32>, uv: vec2<f32>, lod: f32) -> vec4<f32> {
    let origin = vec2(f32(frame % cols), f32(frame / cols + band0)) * cell;
    return textureSampleLevel(textures[index], sprite_sampler, origin + clamp(uv, vec2(.002), vec2(.998)) * cell, lod);
}
// Voronoi cells for the coal bed: the nearest and second-nearest feature and the nearest cell's id.
fn coal_cells(p: vec2<f32>) -> vec3<f32> {
    let i = floor(p);
    var d1 = 8.0; var d2 = 8.0; var id = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let c = i + vec2(f32(x), f32(y));
            let h = vec2(flame_hash(c), flame_hash(c + vec2(17.3, 5.1)));
            let d = distance(p, c + 0.15 + 0.7 * h);
            if d < d1 { d2 = d1; d1 = d; id = flame_hash(c + vec2(3.7, 9.2)); } else if d < d2 { d2 = d; }
        }
    }
    return vec3(d1, d2, id);
}

@vertex fn vs_main(@builtin(vertex_index) vid: u32, @builtin(instance_index) iid: u32) -> Out {
    let p = particles[iid]; let uv = UV[vid];
    var out: Out;
    var corner: vec3<f32>;
    out.flutter = p.flutter;
    if (p.flags & 32u) != 0u {
        // A flame's card (bit 5): it stands on its bed, turns to face the eye about the vertical, and as the eye rises
        // over the fire tilts toward it, up to 40 degrees, where the book's 40-degree view takes over; its top leans
        // with the wind.
        let w = p.size; let h = p.size * p.aspect; let bed_v = p.tip_tint.x;
        let to_eye = pview.camera_pos - p.position;
        let flat_len = length(to_eye.xz);
        var f = vec3(0., 0., 1.);
        if flat_len > 1.0e-4 { f = vec3(to_eye.x, 0., to_eye.z) / flat_len; }
        let has_down = p.tip_tint.z > .5;
        let tilt = select(0., clamp(atan2(to_eye.y, max(flat_len, 1.0e-4)), 0., .6981317), has_down);
        let right = vec3(f.z, 0., -f.x);
        let up = cos(tilt) * vec3(0., 1., 0.) - sin(tilt) * f;
        // A card smaller than FAR_PX pixels grows about its light's centre to that size, its light spread over the
        // larger card (the fragment's far blob), so a far lamp is a steady point of its own light, never a sprite
        // that falls between pixels and blinks under FSR (photoreal's glints draw only the 256 nearest lamps).
        let centre_y = .35 * h * bed_v;
        let seen = h / max((pview.view_proj * vec4(p.position + up * centre_y, 1.)).w * pview._pad0, 1.0e-6);
        let grow = max(1., FAR_PX / max(seen, 1.0e-4));
        let x = (uv.x - .5) * w * grow;
        let y = centre_y + ((bed_v - uv.y) * h - centre_y) * grow;
        let rise = clamp(y / max(h * bed_v, 1.0e-4), 0., 1.);
        corner = p.position + right * x + up * y + vec3(p.color.z, 0., p.color.w) * pow(rise, 1.5);
        out.flutter = tilt / .6981317;
        out.extra = vec4(p.extra.x, p.extra.y, grow, 0.);
    } else if (p.flags & 64u) != 0u {
        // A bed of coals (bit 6): a disc lying flat; `uv` is metres across it.
        let offset = (uv - .5) * p.size;
        corner = p.position + vec3(offset.x, 0., offset.y);
        out.uv = offset;
    } else if (p.flags & 128u) != 0u {
        // A spark (bit 7): the streak it draws in its shutter, from where it was to where it is, as wide as it is.
        let travel = p.extra.xyz * p.extra.w;
        let to_eye = normalize(pview.camera_pos - p.position);
        var along = travel;
        if length(along) < p.size * .25 { along = pview.camera_up * p.size * .25; }
        let side = normalize(cross(along, to_eye)) * p.size * .5;
        let dir = normalize(along);
        corner = p.position - along * uv.y + dir * p.size * .5 * (1. - 2. * uv.y) + side * (uv.x * 2. - 1.);
    } else {
        let xy = vec2((uv.x - .5) * p.size, (.5 - uv.y) * p.size * p.aspect);
        let c = cos(p.rotation); let s = sin(p.rotation);
        let offset = vec2(c*xy.x-s*xy.y,s*xy.x+c*xy.y);
        var right = pview.camera_right; var up = pview.camera_up;
        if (p.flags & 2u) != 0u {
            let horizontal = vec3(right.x, 0., right.z);
            right = horizontal / max(length(horizontal),0.00001);
            if length(horizontal) < 0.00001 { right = vec3(1.,0.,0.); }
            up = vec3(0.,1.,0.);
        }
        corner = p.position + right*offset.x + up*offset.y;
    }
    out.position = pview.view_proj * vec4(corner,1.);
    out.view_depth = out.position.w;
    out.view_dir = corner - pview.camera_pos;
    if (p.flags & 64u) == 0u { out.uv = p.uv_rect.xy + uv*p.uv_rect.zw; }
    out.local_uv = uv; out.colour = p.color; out.tip = p.tip_tint;
    out.uv_rect = p.uv_rect;
    if (p.flags & 32u) == 0u { out.extra = p.extra; }
    out.texture_index = p.texture_index; out.flags = p.flags;
    return out;
}
struct Fragment {
    @location(0) colour: vec4<f32>,
    @location(1) reactive: vec4<f32>,
}
@fragment fn fs_main(in: Out) -> Fragment {
    var out: Fragment;
    // Derivatives first, in uniform control flow: a book's level of detail.
    let duv_x = dpdx(in.local_uv);
    let duv_y = dpdy(in.local_uv);
    if (in.flags & 32u) != 0u && in.texture_index >= 0 {
        // A flame's card from its simulated book (bit 5): frame A carried forward along its motion and frame B back,
        // blended; luminance from R, colour from the temperature in G through a blackbody, at the book's peak
        // luminance scaled to its lamp; the 40-degree band blended in as the eye rises; faded where it meets the
        // pan or the wick.
        let frames = max(u32(in.tip.y + .5), 1u);
        let down0 = u32(in.tip.z + .5);
        let cols = max(u32(in.uv_rect.z + .5), 1u);
        let rows = max(u32(in.uv_rect.w + .5), 1u);
        let fa = u32(in.uv_rect.x + .5) % frames;
        let fb = (fa + 1u) % frames;
        let t = in.uv_rect.y;
        let dims = vec2<f32>(textureDimensions(textures[in.texture_index]));
        let cell = vec2(1. / f32(cols), 1. / f32(rows + down0));
        let frame_px = dims * cell;
        let footprint = max(length(duv_x * frame_px), length(duv_y * frame_px));
        let lod = max(log2(max(footprint, 1.0e-4)), 0.);
        let small = (in.flags & 512u) != 0u;
        let down = clamp(in.flutter, 0., 1.) * select(0., 1., down0 > 0u);
        var y = 0.; var yt = 0.; var blue = 0.;
        for (var band = 0u; band < 2u; band++) {
            let share = select(1. - down, down, band == 1u);
            if share <= .001 { continue; }
            let band0 = select(0u, down0, band == 1u);
            var mv = vec2(0.);
            if !small {
                let m = book_texel(in.texture_index, fa, band0, cols, cell, in.local_uv, lod);
                mv = (m.ba - .5) * 2. * BOOK_MOTION_PX / frame_px;
            }
            let a = book_texel(in.texture_index, fa, band0, cols, cell, in.local_uv - mv * t, lod);
            let b = book_texel(in.texture_index, fb, band0, cols, cell, in.local_uv + mv * (1. - t), lod);
            let ya = book_decode(a.r) * (1. - t); let yb = book_decode(b.r) * t;
            y += (ya + yb) * share;
            yt += (ya * (800. + 1400. * a.g) + yb * (800. + 1400. * b.g)) * share;
            if small { blue += (book_decode(a.b) * (1. - t) + book_decode(b.b) * t) * share; }
        }
        let temperature = yt / max(y, 1.0e-12);
        let gain = in.colour.x * clamp(in.colour.y, 0., 1.) * fire_eye()
            * soft_depth(in.position, in.view_depth, pview.fire.w);
        var rgb = (y * blackbody_rgb(temperature) + blue * BLUE_FOOT) * gain;
        // Far (bit 5 grown, `extra.z`): the card's whole light, its mean over its peak (`extra.x`) at this frame's power
        // (`extra.y`), as a soft blob over the grown card, so a point's light is the flame's at any distance.
        let grow = in.extra.z;
        if grow > 1.001 {
            let c = vec2(.5, in.tip.x * .65);
            let q = (in.local_uv - c) / .18;
            let blob = exp(-.5 * dot(q, q)) / .2036;
            let far = in.extra.x * in.extra.y / (grow * grow) * blob;
            rgb = mix(rgb, far * blackbody_rgb(select(1750., 1800., small)) * gain, smoothstep(1., 2., grow));
            y = mix(y, far, smoothstep(1., 2., grow));
        }
        out.colour = vec4(max(rgb, vec3(0.)), 0.);
        // Fast-moving light: as reactive as it is bright, so FSR does not smear a tongue's last place.
        out.reactive = vec4(clamp(log2(max(y, 1.0e-6)) / 8. + 1., 0., 1.) * clamp(in.colour.y, 0., 1.), 0., 0., 0.);
        return out;
    }
    if (in.flags & 64u) != 0u {
        // A bed of coals (bit 6): lumps with a dark crust over cracks glowing at the bed's heat, each lump breathing
        // slowly, the rim cooler and ashen; emission at the blackbody's own luminance, opaque over the pan's floor.
        let radius = length(in.local_uv * 2. - 1.);
        let hot = in.colour.x; let fade = clamp(in.colour.y, 0., 1.); let seed = in.colour.z; let clock = in.colour.w;
        let q = in.uv / .035 + seed * 17.;
        let c = coal_cells(q);
        let crack = 1. - smoothstep(0., .1, c.y - c.x);
        let breath = 1. + .035 * sin(clock * .45 * 6.2831853 + c.z * 6.2831853) + .02 * sin(clock * 1.7 + c.z * 40.);
        var temperature = mix(mix(880., hot - 90., pow(c.z, 1.5)) * breath, hot, crack);
        temperature -= 160. * smoothstep(.55, 1., radius);
        let crust = smoothstep(.35, .8, flame_noise(q * 2.3 + seed)) * (1. - crack) * (1. - .7 * c.z);
        let emissivity = mix(.9, .08, crust);
        let coverage = (1. - smoothstep(.86, 1., radius)) * soft_depth(in.position, in.view_depth, .03);
        let rgb = blackbody_rgb(temperature) * (emissivity * blackbody_cd(temperature) * fade * fire_eye());
        out.colour = vec4(rgb * coverage, coverage);
        out.reactive = vec4(coverage, 0., 0., 0.);
        return out;
    }
    var sprite = vec4(1.);
    if (in.flags & 128u) != 0u {
        // A spark's streak (bit 7): soft across, brightest at its head, fading toward its tail.
        let across = 1. - abs(in.local_uv.x * 2. - 1.);
        sprite = vec4(1., 1., 1., across * across * mix(1., .35, in.local_uv.y));
    } else if in.texture_index >= 0 {
        // Bend the textured silhouette, with zero displacement at the wick.
        // Sampling is clamped to this atlas cell, never its neighbour.
        let height = 1. - in.local_uv.y;
        let uv = vec2(select(in.local_uv.x, 1. - in.local_uv.x, (in.flags & 4u) != 0u), in.local_uv.y);
        let warped = uv - vec2(in.flutter * height * height, 0.);
        let sample_uv = in.uv_rect.xy + clamp(warped, vec2(.001), vec2(.999)) * in.uv_rect.zw;
        sprite = textureSampleLevel(textures[in.texture_index],sprite_sampler,sample_uv,0.);
        if any(warped < vec2(0.)) || any(warped > vec2(1.)) { sprite.a = 0.; }
        // Shared standalone asset textures are RGBA8 UNORM; colour input is sRGB.
        sprite = vec4(pow(max(sprite.rgb,vec3(0.)),vec3(2.2)),sprite.a);
    } else if (in.flags & 16u) != 0u {
        // A flame's tongue (bit 4; Sunscar, 13 Nov 2026): round at its base, drawn to a point at its tip, bent by its
        // flutter with the base pinned as a textured flame's is, torn by turbulence that rises through it, its tip
        // licking up and down, white-yellow along its axis low down and orange-red at its edges. `uv_rect` carries
        // the tongue's seed and age (somnium_core particle.rs), so no two tongues burn alike.
        let seed = in.uv_rect.x;
        let age = in.uv_rect.y;
        let h = 1. - in.local_uv.y;
        let n1 = flame_noise(vec2(in.local_uv.x * 3. + seed * 7.1, h * 2.5 - age * 3.2));
        let n2 = flame_noise(vec2(in.local_uv.x * 7. - seed * 3.3, h * 5. - age * 5.7));
        let x = in.local_uv.x - .5 - in.flutter * h * h - ((n1 - .5) * .24 + (n2 - .5) * .1) * h;
        let tip = .74 + .24 * flame_noise(vec2(seed * 11. + 3., age * 4.5));
        let hh = h / tip;
        let half = .46 * sqrt(clamp(hh * 3.5, 0., 1.)) * pow(clamp((1. - hh) / .72, 0., 1.), .85) * (.8 + .4 * n1);
        // Torn at the top: the noise eats the tongue from its tip down, so it ends in wisps that break away.
        let tear = smoothstep(.3, .62, n1 * .6 + n2 * .4 + (1. - hh) * .55);
        let edge = (1. - smoothstep(half * .35, half + .001, abs(x))) * step(hh, 1.) * tear;
        let core = (1. - smoothstep(0., half * .55 + .001, abs(x))) * (1. - smoothstep(.15, .75, hh));
        let hot = mix(vec3(1., .5, .26), vec3(1.45, 1.4, 1.55), core);
        sprite = vec4(hot * (.75 + .5 * n2), edge);
    } else {
        sprite.a = 1. - smoothstep(.2,.5,distance(in.local_uv,vec2(.5)));
    }
    var alpha = clamp(in.colour.a * sprite.a,0.,1.);
    // Puffs and sparks near a fire fade where they meet a wall or the ground instead of being cut by it.
    if (in.flags & 384u) != 0u { alpha *= soft_depth(in.position, in.view_depth, pview.fire.w * 3.); }
    let tint = mix(vec3(1.),in.tip,smoothstep(.12,.72,1.-in.local_uv.y));
    var rgb = max(in.colour.rgb * tint * sprite.rgb,vec3(0.)) * alpha;
    // Lit (bit 3: smoke, dust, mist): the colour is an albedo, lit by the sun through a forward-scattering phase
    // and by the sky's fill (the engine's 0.08 cd/m2 per lux of sun, scaled as the IBL is), so a column of
    // chimney smoke reads in the scene's own light at any exposure (Sunscar, 6 Nov 2026).
    if (in.flags & 8u) != 0u {
        let v = normalize(in.view_dir);
        let l = normalize(sun.direction);
        let g = 0.5;
        let denom = 1. + g*g - 2.*g*dot(v, l);
        let phase = (1. - g*g) / (12.566371 * max(denom * sqrt(max(denom, 1e-6)), 1e-6));
        let albedo = rgb;
        rgb = albedo * (sun.color * phase + sun.color * (0.08 * sun.ibl_intensity));
        // And by its fire (bit 8): the illuminance its fire's light puts on it, scattered evenly (1 / 4 pi); smoke
        // over a brazier at night glows orange from below. Physical, as everything the fire lights (photoreal).
        if (in.flags & 256u) != 0u { rgb += albedo * in.extra.xyz * 0.0795775; }
    }
    // A spark is the fire's own light too.
    if (in.flags & 128u) != 0u { rgb *= fire_eye(); }
    out.colour = vec4(rgb,select(alpha,0.,(in.flags & 1u) != 0u));
    // Texture alpha and scene depth decide actual visible coverage. Empty sprite
    // corners leave static history intact, including for additive flame sprites.
    // As strong as the sprite is (AMD's guidance: the alpha it is composited with), not all-or-nothing: a ground-mist
    // disc at 5% marked its whole six metres fully reactive, and under it FSR showed each raw jittered frame, so the
    // trunks shimmered and thin leaves blinked in and out (18 Oct 2026). TAA never read this, which is why it held.
    out.reactive = vec4(alpha,0.,0.,0.);
    return out;
}
