// Somnium Engine — one GPU grass blade: its record and its triangles (Sunscar, 19 Nov 2026; `grass.rs`).
//
// A blade lives in one slot of the shared vertex pool: eight f32s, the size of one `Vertex`. Each is a real number or
// an integer under 2^24, which an f32 holds exactly, so no pass bitcasts a float a driver may have flushed as a
// denormal on the way through a register. grass_gen.wgsl writes them; the grass draw (grass_vis.wgsl) and every pass
// that rebuilds a surface from the visibility buffer (shading, velocity, ReSTIR GI, lighting extra, the path tracer)
// read them through `grass_strip_vertex`, so a blade is shaded, lit and moved exactly where it was drawn.
//
//   0..2  root, world xyz
//   3     height (m)
//   4     width (m), already widened for its distance
//   5     facing (16 bits, a turn) | lean (8 bits, 0..90 degrees) << 16
//   6     colour, the square root of linear albedo in 8 bits each: r | g << 8 | b << 16
//   7     bend (6 bits) | stiffness (4) << 6 | root occlusion (4) << 10 | segments - 1 (2) << 14 | seed (8) << 16
//
// The visibility buffer names a blade's triangle as (grass instance + 1, slot << 3 | triangle).
// A blade of S segments is a strip of 2S + 1 points (pairs up the blade, one at the tip): 2S - 1 triangles, triangle
// k on points k, k + 1, k + 2. Its winding alternates; the blade is two-sided.

//!include "wind.wgsl"

const GRASS_TRI_BITS: u32 = 3u;

struct GrassBlade {
    root: vec3<f32>,
    height: f32,
    width: f32,
    facing: f32,
    lean: f32,
    colour: vec3<f32>,
    bend: f32,
    stiffness: f32,
    root_ao: f32,
    segments: u32,
    seed: u32,
}

struct GrassVertex {
    pos: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
}

fn grass_int(x: f32) -> u32 {
    return u32(max(x, 0.0) + 0.5);
}

fn grass_decode(a: vec4<f32>, b: vec4<f32>) -> GrassBlade {
    var g: GrassBlade;
    g.root = a.xyz;
    g.height = a.w;
    g.width = b.x;
    let fl = grass_int(b.y);
    g.facing = f32(fl & 0xffffu) * (6.2831855 / 65536.0);
    g.lean = f32((fl >> 16u) & 255u) * (1.5707964 / 255.0);
    let c = grass_int(b.z);
    let s = vec3<f32>(f32(c & 255u), f32((c >> 8u) & 255u), f32((c >> 16u) & 255u)) / 255.0;
    g.colour = s * s;
    let m = grass_int(b.w);
    g.bend = f32(m & 63u) / 63.0;
    g.stiffness = f32((m >> 6u) & 15u) / 15.0;
    g.root_ao = f32((m >> 10u) & 15u) / 15.0;
    g.segments = ((m >> 14u) & 3u) + 1u;
    g.seed = (m >> 16u) & 255u;
    return g;
}

/// The record's second half (words 4..7); the first is (root, height).
fn grass_encode_tail(g: GrassBlade) -> vec4<f32> {
    let facing = u32(fract(g.facing / 6.2831855) * 65535.0 + 0.5);
    let lean = u32(clamp(g.lean / 1.5707964, 0.0, 1.0) * 255.0 + 0.5);
    let s = sqrt(clamp(g.colour, vec3<f32>(0.0), vec3<f32>(1.0)));
    let c = vec3<u32>(s * 255.0 + 0.5);
    let m = u32(clamp(g.bend, 0.0, 1.0) * 63.0 + 0.5)
        | (u32(clamp(g.stiffness, 0.0, 1.0) * 15.0 + 0.5) << 6u)
        | (u32(clamp(g.root_ao, 0.0, 1.0) * 15.0 + 0.5) << 10u)
        | ((clamp(g.segments, 1u, 4u) - 1u) << 14u)
        | ((g.seed & 255u) << 16u);
    return vec4<f32>(g.width, f32(facing | (lean << 16u)), f32(c.x | (c.y << 8u) | (c.z << 16u)), f32(m));
}

/// Triangles in a blade of `segments` segments.
fn grass_tris(segments: u32) -> u32 {
    return 2u * segments - 1u;
}

/// Where along the blade (0 root, 1 tip) its j-th pair of points lies: closer together towards the curling tip.
fn grass_seg_t(segments: u32, j: u32) -> f32 {
    if j == 0u {
        return 0.0;
    }
    if segments == 2u {
        return 0.55;
    }
    if segments == 3u {
        return select(0.75, 0.4, j == 1u);
    }
    return f32(j) / f32(segments);
}

/// The wind a blade takes, from its material's (`[bend, flutter]`): a stiff reed bends less than lodged straw.
fn grass_wind(g: GrassBlade, base_bend: f32, base_flutter: f32) -> vec2<f32> {
    return vec2<f32>(base_bend * mix(1.6, 0.35, g.stiffness), base_flutter * mix(1.0, 0.3, g.stiffness));
}

/// Point `j` of the blade's strip (0 .. 2 * segments): world position with the wind at `clock`, its normal rounded
/// across the blade (Ghost of Tsushima's trick: a flat blade lit as if it were curved), uv = (across, along).
fn grass_strip_vertex(g: GrassBlade, j: u32, clock: f32, wind: vec4<f32>, eye: vec3<f32>, base_bend: f32,
                      base_flutter: f32) -> GrassVertex {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let side = vec3<f32>(cos(g.facing), 0.0, sin(g.facing));
    let fwd = vec3<f32>(-side.z, 0.0, side.x);
    let tip = g.root + g.height * (fwd * sin(g.lean) + up * cos(g.lean));
    // Straight (bend 0) to arched (1): rising steeply from the root and curling over to the tip.
    let ctrl = mix(0.5 * (g.root + tip), g.root + up * (0.75 * g.height), g.bend);
    let at_tip = j >= 2u * g.segments;
    let t = select(grass_seg_t(g.segments, j >> 1u), 1.0, at_tip);
    let omt = 1.0 - t;
    var p = omt * omt * g.root + 2.0 * t * omt * ctrl + t * t * tip;
    let tangent = normalize(2.0 * omt * (ctrl - g.root) + 2.0 * t * (tip - ctrl) + vec3<f32>(0.0, 1e-6, 0.0));
    var n = normalize(cross(side, tangent));
    var u = 0.5;
    if !at_tip {
        let s = select(-1.0, 1.0, (j & 1u) == 1u);
        // The same area whatever the segments: 3 cover 0.646 of width x height, 2 cover 0.624, 1 (a triangle) 0.5, so
        // without this the sward thinned by a fifth where blades turn to one triangle at 24 m (the lead, 21:42: a
        // darker band near 20 m).
        let area = select(select(1.0, 1.035, g.segments == 2u), 1.29, g.segments == 1u);
        p += side * (s * 0.5 * g.width * area * (1.0 - t * t));
        n = normalize(n + side * (0.45 * s));
        u = 0.5 + 0.5 * s;
    }
    let w = grass_wind(g, base_bend, base_flutter);
    p += wind_offset(p, g.root, w.x, w.y, clock, wind, eye);
    var out: GrassVertex;
    out.pos = p;
    out.normal = n;
    out.uv = vec2<f32>(u, t);
    return out;
}
