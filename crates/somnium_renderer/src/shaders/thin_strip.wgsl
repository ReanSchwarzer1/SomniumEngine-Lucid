// Thin strips keep about a pixel (Sunscar, 11 Nov 2026; the lead's review 17:05). A tall grass's culm is a strip
// 2-4 mm wide; within a few metres it falls under a pixel and the raster drops it, while its 2-4 cm seed head stays,
// so the heads floated over the ground. A foliage triangle that is a strip (thin along one UV axis, its corners on that
// axis's two edges) is widened in world space to `view.strip_pixel` at its distance (the renderer's
// `set_min_strip_px`, 0 = off), and the visibility pass keeps each pixel of it with its true coverage, so the culm
// fades with its head instead of vanishing first. A bare crown's last twigs (three-sided tubes) are strips too.
//
// Worked out from the triangle's rest pose, so every pass that rebuilds the drawn triangle gets the same offsets: the
// visibility pass (each corner from its own triangle), shading.wgsl and path_tracer.wgsl (all three). Wind is added
// before the offset in each, as it is here.

struct ThinStrip {
    o0: vec3<f32>,
    o1: vec3<f32>,
    o2: vec3<f32>,
    // The share of the widened strip that is strip: the visibility pass keeps a pixel with this chance.
    coverage: f32,
}

// Foliage only (it sways): bark, rock and buildings keep their shapes.
fn thin_strip_applies(strip_pixel: f32, wind_bend: f32, wind_flutter: f32) -> bool {
    return strip_pixel > 0.0 && (wind_bend > 0.0 || wind_flutter > 0.0);
}

fn thin_strip(r0: vec3<f32>, r1: vec3<f32>, r2: vec3<f32>, t0: vec2<f32>, t1: vec2<f32>, t2: vec2<f32>,
              eye: vec3<f32>, strip_pixel: f32) -> ThinStrip {
    var out = ThinStrip(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 1.0);
    let e1 = r1 - r0;
    let e2 = r2 - r0;
    let d1 = t1 - t0;
    let d2 = t2 - t0;
    let det = d1.x * d2.y - d2.x * d1.y;
    if abs(det) < 1e-12 {
        return out;
    }
    // The world vector one unit of u (and of v) spans on this triangle, times the triangle's own range of each.
    let tmin = min(t0, min(t1, t2));
    let tmax = max(t0, max(t1, t2));
    let wu = (e1 * d2.y - e2 * d1.y) / det * (tmax.x - tmin.x);
    let wv = (e2 * d1.x - e1 * d2.x) / det * (tmax.y - tmin.y);
    let lu = length(wu);
    let lv = length(wv);
    // A strip: its narrow axis under a fifth of its long one.
    let across_u = lu < lv;
    let w = select(wv, wu, across_u);
    let narrow = min(lu, lv);
    if narrow <= 0.0 || narrow > 0.2 * max(lu, lv) {
        return out;
    }
    let mid = (r0 + r1 + r2) / 3.0;
    let to_mid = mid - eye;
    let dist = length(to_mid);
    let sight = to_mid / max(dist, 1e-4);
    // Its width seen from the eye: the part of it across the line of sight, in pixels.
    let px = length(w - sight * dot(w, sight)) / max(dist * strip_pixel, 1e-9);
    if px >= 1.0 {
        return out;
    }
    // Up to sixteen times its width: a strip past that fades out by its coverage alone.
    let grow = min(1.0 / max(px, 1e-4), 16.0);
    let lo = select(tmin.y, tmin.x, across_u);
    let half = 0.5 * (select(tmax.y, tmax.x, across_u) - lo);
    let centre = lo + half;
    let a = vec3<f32>(select(t0.y, t0.x, across_u), select(t1.y, t1.x, across_u), select(t2.y, t2.x, across_u));
    // Each corner out along the strip's width by its side: -1 on one edge, +1 on the other, 0 between.
    let side = clamp((a - centre) / max(half, 1e-9), vec3<f32>(-1.0), vec3<f32>(1.0));
    let s = 0.5 * (grow - 1.0);
    out.o0 = w * side.x * s;
    out.o1 = w * side.y * s;
    out.o2 = w * side.z * s;
    out.coverage = 1.0 / grow;
    return out;
}
