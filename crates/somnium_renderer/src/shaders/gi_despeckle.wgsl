// ReSTIR GI's output despeckled on its own grid, once per texel, before the shading pass upsamples it (Sunscar,
// 6 Nov 2026). The same two bounds used to run per screen pixel inside `gi_upsample` (shading.wgsl): sixteen taps,
// each checked against its eight neighbours, which was 11 of a dense wood's 19.6 ms of shading at native 1440p.
//
// One reservoir per texel is still a stochastic estimate, and a texel that drew a rare bright sample (a bounce off a
// lamp housing, a sunlit leaf seen through a gap) comes out a speck. Both bounds are relative, so they need no
// exposure: a texel may not exceed SPECK_RATIO times its brightest same-surface neighbour (an isolated hot texel is
// exactly a speck, while real bounce is smooth and its neighbours vouch for it), nor FIREFLY_RATIO times the
// same-surface mean round it (the backstop for small clusters). Alpha is the traced surface's camera distance
// (0 = nothing traced) and passes through.

@group(0) @binding(0) var gi_in: texture_2d<f32>;
@group(0) @binding(1) var gi_out: texture_storage_2d<rgba16float, write>;

const SPECK_RATIO: f32 = 1.25;
const FIREFLY_RATIO: f32 = 2.5;

fn despeckle_luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

/// Within the upsample's depth weight of 0.1 (exp(-|d| / (dist * 0.025 + 0.05)) > 0.1).
fn same_surface(dist: f32, other: f32) -> bool {
    return other > 0.0 && abs(other - dist) < 2.3026 * (dist * 0.025 + 0.05);
}

@compute @workgroup_size(8, 8, 1)
fn despeckle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<i32>(textureDimensions(gi_in));
    let g = vec2<i32>(gid.xy);
    if g.x >= dims.x || g.y >= dims.y {
        return;
    }
    let c = textureLoad(gi_in, g, 0);
    if c.a <= 0.0 {
        textureStore(gi_out, g, c);
        return;
    }
    let l = despeckle_luma(c.rgb);
    var neighbour_max = -1.0;
    var sum = l;
    var count = 1.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            if dx == 0 && dy == 0 {
                continue;
            }
            let t = textureLoad(gi_in, clamp(g + vec2<i32>(dx, dy), vec2<i32>(0), dims - 1), 0);
            if same_surface(c.a, t.a) {
                let tl = despeckle_luma(t.rgb);
                neighbour_max = max(neighbour_max, tl);
                sum = sum + tl;
                count = count + 1.0;
            }
        }
    }
    var ceiling = FIREFLY_RATIO * sum / count + 1.0e-6;
    if neighbour_max >= 0.0 {
        ceiling = min(ceiling, SPECK_RATIO * neighbour_max + 1.0e-6);
    }
    textureStore(gi_out, g, vec4<f32>(c.rgb * min(1.0, ceiling / max(l, 1.0e-6)), c.a));
}
