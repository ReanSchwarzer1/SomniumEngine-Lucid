// Where the fog volume's slices lie along a view ray (volumetric.wgsl writes them; shading, water and the clouds read
// them). Exponential rather than even: even slices over a 1.2 km range were 37.5 m deep, so valley fog a few metres
// thick at the feet fell inside one slice. With FROXEL_BIAS 4 the first of 64 slices is about 3.7 m deep and the last
// about 53 m at that range (Sunscar, 6 Nov 2026; ENGINE_AUDIT item 3). `w` is 0..1 across the slices.
const FROXEL_BIAS: f32 = 4.0;

/// Distance along the ray at slice coordinate `w`.
fn froxel_t(w: f32, range: f32) -> f32 {
    return range * (exp2(FROXEL_BIAS * w) - 1.0) / (exp2(FROXEL_BIAS) - 1.0);
}

/// Slice coordinate of a distance along the ray (inverse of `froxel_t`).
fn froxel_w(t: f32, range: f32) -> f32 {
    return log2(1.0 + max(t, 0.0) / max(range, 1e-3) * (exp2(FROXEL_BIAS) - 1.0)) / FROXEL_BIAS;
}
