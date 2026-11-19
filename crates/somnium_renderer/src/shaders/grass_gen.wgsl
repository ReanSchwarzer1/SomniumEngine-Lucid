// Somnium Engine — GPU grass, laid out each frame around the eye (Sunscar, 19 Nov 2026; `grass.rs`).
//
// The CPU lists the 8 m tiles in view with grass in them, and for each how many candidates every 0.5 m cell of it
// visits (its nearest distance sets that). One workgroup is one (tile, k): its 256 threads are the tile's 256 cells,
// each looking at the k-th candidate of its cell. A candidate is the k-th point of a rotated R2 sequence in its cell,
// and it grows while (k + 0.5) / CANDIDATES is under its keep. So a thinner sward is always a subset of a denser one:
// walking, the cut moves through a cell's points in the same order, and a point near the cut shrinks rather than pops.
//
// Its keep is the habitat's blades per m² (groundplants' `data/foliage/grass_habitats.json`, through the per-texel
// habitat map `grass.rs` builds from the splat), its presence (the grassland's summed weight), the distance (full to
// `d_lod`, then a quarter per doubling) and the slope. Past `d_lod` the blades widen as they thin: a ray from the eye
// crosses twice the sward per doubling of distance, so the ground stays as covered. A blade is never under
// `min_px` pixels wide; it is widened, and its keep cut to match, instead.
//
// Then frustum and Hi-Z (this frame's final pyramid) culling, and the blade goes to its class (3, 2 or 1 segments by
// distance): counted in workgroup memory, then one add per class per workgroup on the draw's counter. A class that
// runs over its capacity drops the rest; the draw clamps its count.

//!include "grass_blade.wgsl"

struct BladeSlot {
    a: vec4<f32>,
    b: vec4<f32>,
}

struct GrassParams {
    view_proj: mat4x4<f32>,
    eye_clip: vec4<f32>,
    /// xyz the eye, w the radians one pixel spans.
    eye: vec4<f32>,
    /// Camera-relative frustum planes, inside where dot(n, p) + d >= 0.
    planes: array<vec4<f32>, 6>,
    /// x d_lod (m), y range (m), z where the far fade starts (m), w the density cap (blades per m²).
    lod: vec4<f32>,
    /// x three segments within (m), y two within (m), z the narrowest blade (pixels), w candidates per cell at the cap.
    seg: vec4<f32>,
    /// xy the pyramid's size, z its levels, w 1 when it holds this frame's depth.
    hiz: vec4<f32>,
    /// x the span's first slot, y..w each class's capacity.
    base: vec4<u32>,
    /// x tiles, y chunks, z habitats, w the frame.
    misc: vec4<u32>,
}

struct GrassTile {
    origin: vec2<f32>,
    terrain: u32,
    candidates: u32,
}

struct GrassTerrain {
    origin: vec2<f32>,
    cell: f32,
    verts: u32,
    height_off: u32,
    hab_off: u32,
    texels: u32,
    texel_size: f32,
}

struct GrassHabitat {
    density: f32,
    h_min: f32,
    h_max: f32,
    w_min: f32,
    w_max: f32,
    lean_min: f32,
    lean_max: f32,
    lodged: f32,
    clump_m: f32,
    contrast: f32,
    seed_h_min: f32,
    seed_h_max: f32,
    share_green: f32,
    share_dead: f32,
    bend_min: f32,
    bend_max: f32,
    stiffness: f32,
    root_ao: f32,
    bare: f32,
    w_mean: f32,
    green: vec4<f32>,
    green_alt: vec4<f32>,
    dead: vec4<f32>,
    dead_alt: vec4<f32>,
    seed: vec4<f32>,
}

struct DrawArgs {
    vertex_count: u32,
    instance_count: atomic<u32>,
    first_vertex: u32,
    first_instance: u32,
}

@group(0) @binding(0) var<storage, read_write> blades: array<BladeSlot>;
@group(0) @binding(1) var<uniform> params: GrassParams;
@group(0) @binding(2) var<storage, read> tiles: array<GrassTile>;
@group(0) @binding(3) var<storage, read> chunks: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read> terrains: array<GrassTerrain>;
@group(0) @binding(5) var<storage, read> heights: array<f32>;
@group(0) @binding(6) var<storage, read> habitat_map: array<u32>;
@group(0) @binding(7) var<storage, read> habitats: array<GrassHabitat>;
@group(0) @binding(8) var hiz: texture_2d<f32>;
@group(0) @binding(9) var<storage, read_write> draws: array<DrawArgs, 3>;

const CELL: f32 = 0.5;
const TILE_CELLS: u32 = 16u;
const NO_HABITAT: u32 = 31u;

fn hash(x: u32) -> u32 {
    let v = x * 747796405u + 2891336453u;
    let w = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (w >> 22u) ^ w;
}

fn hash2(c: vec2<i32>, seed: u32) -> u32 {
    return hash(bitcast<u32>(c.x) ^ hash(bitcast<u32>(c.y) ^ seed));
}

fn unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

/// The terrain's height at world xz (bilinear), and the ground's normal.
fn ground(t: GrassTerrain, xz: vec2<f32>) -> vec4<f32> {
    let last = f32(t.verts - 1u);
    let f = clamp((xz - t.origin) / t.cell, vec2<f32>(0.0), vec2<f32>(last - 0.001));
    let i = vec2<u32>(f);
    let fr = f - vec2<f32>(i);
    let row = t.verts;
    let k = t.height_off + i.y * row + i.x;
    let h00 = heights[k];
    let h10 = heights[k + 1u];
    let h01 = heights[k + row];
    let h11 = heights[k + row + 1u];
    let h = mix(mix(h00, h10, fr.x), mix(h01, h11, fr.x), fr.y);
    let dx = mix(h10 - h00, h11 - h01, fr.y);
    let dz = mix(h01 - h00, h11 - h10, fr.x);
    let n = normalize(vec3<f32>(-dx, t.cell, -dz));
    return vec4<f32>(n, h);
}

/// Whether a blade (root `p` camera-relative, `h` tall, `r` its reach) is hidden behind this frame's depth.
fn occluded(p: vec3<f32>, h: f32, r: f32) -> bool {
    if params.hiz.w < 0.5 {
        return false;
    }
    let a = params.view_proj * vec4<f32>(p, 0.0) + params.eye_clip;
    let b = params.view_proj * vec4<f32>(p + vec3<f32>(0.0, h, 0.0), 0.0) + params.eye_clip;
    if a.w <= 0.05 || b.w <= 0.05 {
        return false;
    }
    let na = a.xyz / a.w;
    let nb = b.xyz / b.w;
    var lo = vec2<f32>(min(na.x, nb.x), min(na.y, nb.y)) * vec2<f32>(0.5, -0.5) + 0.5;
    var hi = vec2<f32>(max(na.x, nb.x), max(na.y, nb.y)) * vec2<f32>(0.5, -0.5) + 0.5;
    let tmp = lo.y;
    lo.y = min(lo.y, hi.y);
    hi.y = max(tmp, hi.y);
    // The blade's reach on screen, at its nearer end, and a pixel's margin.
    let reach = r / (min(a.w, b.w) * params.eye.w) / params.hiz.xy + 1.0 / params.hiz.xy;
    lo = clamp(lo - reach, vec2<f32>(0.0), vec2<f32>(1.0));
    hi = clamp(hi + reach, vec2<f32>(0.0), vec2<f32>(1.0));
    let extent = max(max((hi.x - lo.x) * params.hiz.x, (hi.y - lo.y) * params.hiz.y), 1.0);
    let level = clamp(i32(ceil(log2(extent))) - 1, 0, i32(params.hiz.z) - 1);
    let dims = vec2<f32>(textureDimensions(hiz, level));
    let t0 = vec2<i32>(clamp(lo * dims, vec2<f32>(0.0), dims - 1.0));
    let t1 = vec2<i32>(clamp(hi * dims, vec2<f32>(0.0), dims - 1.0));
    var furthest = 1.0;
    furthest = min(furthest, textureLoad(hiz, t0, level).r);
    furthest = min(furthest, textureLoad(hiz, vec2<i32>(t1.x, t0.y), level).r);
    furthest = min(furthest, textureLoad(hiz, vec2<i32>(t0.x, t1.y), level).r);
    furthest = min(furthest, textureLoad(hiz, t1, level).r);
    if furthest <= 0.0 {
        return false;
    }
    // Reversed-Z: nearer is larger. The blade's nearest point behind everything the pyramid recorded there.
    return max(na.z, nb.z) < furthest;
}

struct Grown {
    ok: bool,
    lod_class: u32,
    g: GrassBlade,
}

/// Candidate `k` of `cell` in `tile`: the blade it grows, or `ok` false.
fn grow(cell: u32, tile: GrassTile, k: u32, rank: f32) -> Grown {
    var nothing: Grown;
    nothing.ok = false;
    let c = vec2<u32>(cell % TILE_CELLS, cell / TILE_CELLS);
    let cell_origin = tile.origin + vec2<f32>(c) * CELL;
    let gc = vec2<i32>(floor(cell_origin / CELL + 0.5));
    let shift = vec2<f32>(unit(hash2(gc, 0x9e3779b9u)), unit(hash2(gc, 0x85ebca6bu)));
    // R2 (Roberts 2018): any prefix of it is evenly spread.
    let r2 = fract(shift + f32(k) * vec2<f32>(0.7548776662, 0.5698402910));
    let xz = cell_origin + r2 * CELL;

    let eye = params.eye.xyz;
    let d = distance(xz, eye.xz);
    if d >= params.lod.y {
        return nothing;
    }
    let terrain = terrains[tile.terrain];

    // The habitat here: one of the texel's two by weight, read at a texel picked with bilinear odds, so a habitat's
    // edge is a smooth blend of blades rather than 2 m squares.
    let h0 = hash2(gc, k * 0x27d4eb2du + 17u);
    let jitter = vec2<f32>(unit(hash(h0 ^ 0x632be5abu)), unit(hash(h0 ^ 0x2545f491u)));
    let tex_f = (xz - terrain.origin) / terrain.texel_size - 0.5 + jitter;
    let last = i32(terrain.texels) - 1;
    let tex = clamp(vec2<i32>(floor(tex_f)), vec2<i32>(0), vec2<i32>(last));
    let word = habitat_map[terrain.hab_off + u32(tex.y) * terrain.texels + u32(tex.x)];
    let hab_a = word & 31u;
    let w_a = f32((word >> 5u) & 255u) / 255.0;
    let hab_b = (word >> 13u) & 31u;
    let w_b = f32((word >> 18u) & 255u) / 255.0;
    let total = w_a + w_b;
    if total <= 0.0 || hab_a == NO_HABITAT {
        return nothing;
    }
    var hab_id = hab_a;
    if hab_b != NO_HABITAT && unit(hash(h0 ^ 0x1b873593u)) * total >= w_a {
        hab_id = hab_b;
    }
    let hab = habitats[min(hab_id, params.misc.z - 1u)];

    // Presence by the grassland's summed weight (groundplants' rule), not by each slot's own against a gate.
    let presence = smoothstep(0.35, 0.7, total);
    let g0 = ground(terrain, xz);
    let slope = smoothstep(0.55, 0.8, g0.y);
    let d_lod = params.lod.x;
    let s = max(1.0, d / d_lod);
    // A blade at least `min_px` pixels wide: widened further, and that many fewer of them.
    let px = params.seg.z * d * params.eye.w;
    let widen = max(1.0, px / max(hab.w_mean * s, 1e-5));
    let far = 1.0 - smoothstep(params.lod.z, params.lod.y, d);
    let keep = min(hab.density, params.lod.w) / params.lod.w * presence * slope * far / (s * s * widen);
    if rank >= keep {
        return nothing;
    }
    // The last eighth of the kept ranks grows in: a blade crossing the cut as the eye moves rises, never pops.
    let rise = 1.0 - smoothstep(keep * 0.875, keep, rank);

    // Clumps: the nearest of the jittered points on a `clump_m` grid sets height, tone and facing for its patch.
    let cq = xz / max(hab.clump_m, 0.25);
    let cb = vec2<i32>(floor(cq));
    var best = 1e9;
    var clump = 0u;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = cb + vec2<i32>(dx, dz);
            let hq = hash2(q, 0x68e31da4u);
            let centre = vec2<f32>(q) + vec2<f32>(unit(hq), unit(hash(hq)));
            let dd = dot(cq - centre, cq - centre);
            if dd < best {
                best = dd;
                clump = hq;
            }
        }
    }
    if unit(hash(clump ^ 0xb5297a4du)) < hab.bare {
        return nothing;
    }
    let c_height = unit(hash(clump ^ 0x1b56c4e9u)) * 2.0 - 1.0;
    let c_tone = unit(hash(clump ^ 0x7f4a7c15u));
    let c_face = unit(hash(clump ^ 0x94d049bbu)) * 6.2831855;

    // The blade itself.
    let h1 = hash(h0 ^ 0xc2b2ae35u);
    let h2 = hash(h1);
    let h3 = hash(h2);
    let h4 = hash(h3);
    let h5 = hash(h4);
    let h6 = hash(h5);
    let state = unit(h1);
    var g: GrassBlade;
    var height = mix(hab.h_min, hab.h_max, pow(unit(h2), 1.3)) * (1.0 + hab.contrast * c_height);
    var width = mix(hab.w_min, hab.w_max, unit(h3));
    var colour: vec3<f32>;
    if state < hab.share_green {
        colour = mix(hab.green.rgb, hab.green_alt.rgb, c_tone);
    } else if state < hab.share_green + hab.share_dead {
        colour = mix(hab.dead.rgb, hab.dead_alt.rgb, c_tone);
    } else {
        colour = hab.seed.rgb;
        height = mix(hab.seed_h_min, hab.seed_h_max, unit(h2));
        width *= 0.5;
    }
    colour *= 0.92 + 0.16 * unit(h4);
    g.root = vec3<f32>(xz.x, g0.w - 0.01 * height, xz.y);
    g.height = max(height * rise, 0.002);
    g.width = width * s * widen;
    g.facing = c_face + (unit(h5) - 0.5) * 3.7699;
    if unit(h6) < hab.lodged {
        g.lean = mix(1.05, 1.48, unit(hash(h6)));
    } else {
        g.lean = mix(hab.lean_min, hab.lean_max, unit(hash(h6)) * unit(hash(h6)));
    }
    g.colour = colour;
    g.bend = mix(hab.bend_min, hab.bend_max, unit(hash(h5 ^ 0x5bd1e995u)));
    g.stiffness = clamp(hab.stiffness + 0.3 * (unit(hash(h4 ^ 0x2c1b3c6du)) - 0.5), 0.0, 1.0);
    g.root_ao = hab.root_ao;
    g.seed = h6 & 255u;
    var lod_class = 2u;
    if d < params.seg.x {
        lod_class = 0u;
    } else if d < params.seg.y {
        lod_class = 1u;
    }
    g.segments = 3u - lod_class;

    // Frustum, then this frame's depth.
    let rel = g.root - eye;
    let reach = g.height + 0.5 * g.width;
    let centre = rel + vec3<f32>(0.0, 0.5 * g.height, 0.0);
    for (var i = 0u; i < 6u; i++) {
        let pl = params.planes[i];
        if dot(pl.xyz, centre) + pl.w < -reach {
            return nothing;
        }
    }
    if occluded(rel, g.height, reach) {
        return nothing;
    }

    var out: Grown;
    out.ok = true;
    out.lod_class = lod_class;
    out.g = g;
    return out;
}

var<workgroup> wg_count: array<atomic<u32>, 3>;
var<workgroup> wg_base: array<u32, 3>;

@compute @workgroup_size(256, 1, 1)
fn grass_gen(@builtin(local_invocation_index) cell: u32, @builtin(workgroup_id) wg: vec3<u32>,
             @builtin(num_workgroups) nwg: vec3<u32>) {
    // The dispatch's last row runs past the list: whole workgroups leave, before any barrier.
    let chunk_index = wg.x + wg.y * nwg.x;
    if chunk_index >= params.misc.y {
        return;
    }
    if cell < 3u {
        atomicStore(&wg_count[cell], 0u);
    }
    workgroupBarrier();
    let chunk = chunks[chunk_index];
    let grown = grow(cell, tiles[chunk.x], chunk.y, (f32(chunk.y) + 0.5) / params.seg.w);
    var local_slot = 0u;
    if grown.ok {
        local_slot = atomicAdd(&wg_count[grown.lod_class], 1u);
    }
    workgroupBarrier();
    // One add per class per workgroup on the draw's counter.
    if cell < 3u {
        let n = atomicLoad(&wg_count[cell]);
        var base = 0u;
        if n > 0u {
            base = atomicAdd(&draws[cell].instance_count, n);
        }
        wg_base[cell] = base;
    }
    workgroupBarrier();
    if !grown.ok {
        return;
    }
    let lod_class = grown.lod_class;
    let slot = wg_base[lod_class] + local_slot;
    if slot >= params.base[lod_class + 1u] {
        return;
    }
    var start = params.base.x;
    if lod_class >= 1u {
        start += params.base.y;
    }
    if lod_class >= 2u {
        start += params.base.z;
    }
    let g = grown.g;
    blades[start + slot] = BladeSlot(vec4<f32>(g.root, g.height), grass_encode_tail(g));
}
