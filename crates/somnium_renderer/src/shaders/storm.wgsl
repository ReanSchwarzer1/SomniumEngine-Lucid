// Storm bodies and lightning (`pass/storm.rs`; Sunscar weather, 11 Nov 2026): what the cloud march (`clouds.wgsl`)
// and its composite (`clouds_composite.wgsl`) share. Each binds the one uniform itself.

/// A body: an analytic shape in world metres with the cloud noise as its detail. Kinds: 1 cell (a cumulonimbus's
/// tower), 2 wall cloud, 3 shelf cloud, 4 funnel, 5 bowl (debris cloud and dust skirt), 6 anvil, 7 shaft (rain, hail, virga),
/// 8 ring (a wall of storm round a clear eye, with its anvil and its rain), 9 hole (takes cloud from the slab; adds
/// none), 10 deck (a storm's wide, level base).
struct StormVolume {
    /// xyz: its foot (y its bottom), w: kind.
    pos: vec4<f32>,
    /// x: radius, y: height, z: density scale, w: metres per repeat of its noise.
    size: vec4<f32>,
    /// Per kind (`storm.rs`'s constructors).
    a: vec4<f32>,
    b: vec4<f32>,
    /// rgb: the share of light it scatters; w: its turn, radians.
    albedo: vec4<f32>,
    /// The cylinder that holds it: axis x, z, radius; w: the march's finest step inside it, metres.
    bound: vec4<f32>,
    /// The cylinder's bottom and top, world y.
    bound_y: vec4<f32>,
}

/// A light inside the march. pos: xyz and the source's radius; color: luminous intensity, candela.
struct StormFlash {
    pos: vec4<f32>,
    color: vec4<f32>,
}

struct Storm {
    /// x: bodies, y: lights.
    counts: vec4<u32>,
    volumes: array<StormVolume, 16>,
    flashes: array<StormFlash, 8>,
}

/// A body that only changes the slab (a hole): it has no place along a ray.
fn storm_is_modifier(v: StormVolume) -> bool {
    return v.pos.w > 8.5 && v.pos.w < 9.5;
}

/// Where a ray crosses a body's cylinder: (enter, exit) in metres along it; exit < enter when it misses.
fn storm_cylinder(v: StormVolume, origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    var enter = 0.0;
    var exit = 1e30;
    if abs(dir.y) < 1e-5 {
        if origin.y < v.bound_y.x || origin.y > v.bound_y.y {
            return vec2<f32>(1.0, -1.0);
        }
    } else {
        let a = (v.bound_y.x - origin.y) / dir.y;
        let b = (v.bound_y.y - origin.y) / dir.y;
        enter = max(min(a, b), 0.0);
        exit = max(a, b);
    }
    let o = origin.xz - v.bound.xy;
    let dd = dot(dir.xz, dir.xz);
    let c = dot(o, o) - v.bound.z * v.bound.z;
    if dd < 1e-10 {
        if c > 0.0 {
            return vec2<f32>(1.0, -1.0);
        }
    } else {
        let b = dot(o, dir.xz);
        let disc = b * b - dd * c;
        if disc < 0.0 {
            return vec2<f32>(1.0, -1.0);
        }
        let s = sqrt(disc);
        enter = max(enter, (-b - s) / dd);
        exit = min(exit, (-b + s) / dd);
    }
    return vec2<f32>(enter, exit);
}
