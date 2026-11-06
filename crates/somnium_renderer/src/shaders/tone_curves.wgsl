// The tone curves (postprocess.wgsl's, moved here 6 Nov 2026 by sunscar-photoreal so a GPU test can evaluate
// the very curve the picture uses: tests/tone_curves.rs). Pure functions of the exposed light; no bindings.

// ACES filmic tone-mapping approximation (Narkowicz 2015).
fn aces_film(x: vec3<f32>) -> vec3<f32> {
    return clamp(
        (x * (2.51 * x + vec3(0.03))) / (x * (2.43 * x + vec3(0.59)) + vec3(0.14)),
        vec3(0.0), vec3(1.0)
    );
}

// ── AgX (Phase 24B) ─────────────────────────────────────────────────────────
//
// Troy Sobotka's AgX, as the analytic form rather than the 3-D LUT the
// reference implementation ships. A LUT would mean shipping and binding a KTX2
// asset for what is a closed-form curve; the polynomial below matches it to
// well within 8-bit output precision.
//
// Why replace ACES: ACES pushes very bright saturated colours toward the
// primaries, so an intense warm light skews orange and then clips to white
// rather than desaturating the way film does. That was tolerable when the sun
// was an arbitrary 3.0, and stops being tolerable at 100 000 lux with a sun
// disc in frame. AgX desaturates smoothly into the highlight instead, which is
// what keeps a sunrise looking photographed.

// Rec.709 → AgX working space. Columns, matching WGSL's mat3x3 constructor.
const AGX_INSET = mat3x3<f32>(
    vec3<f32>(0.8566271533, 0.1373189729, 0.1118982130),
    vec3<f32>(0.0951212405, 0.7612419906, 0.0767994186),
    vec3<f32>(0.0482516061, 0.1014390365, 0.8113023684),
);

// Inverse of the above, back to Rec.709.
const AGX_OUTSET = mat3x3<f32>(
    vec3<f32>( 1.1271005818, -0.1413297635, -0.1413297635),
    vec3<f32>(-0.1106066431,  1.1578237022, -0.1106066431),
    vec3<f32>(-0.0164939387, -0.0164939387,  1.2519364066),
);

/// Dynamic range AgX maps, in stops around middle grey.
const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

/// Sixth-order fit to AgX's default contrast sigmoid.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2
         - 40.14 * x4 * x
         + 31.96 * x4
         - 6.868 * x2 * x
         + 0.4298 * x2
         + 0.1191 * x
         - 0.00232;
}

fn agx(color_in: vec3<f32>) -> vec3<f32> {
    var color = AGX_INSET * max(color_in, vec3(0.0));

    // Log-encode across AgX's range. The floor keeps log2 finite on black.
    color = clamp(log2(max(color, vec3(1e-10))), vec3(AGX_MIN_EV), vec3(AGX_MAX_EV));
    color = (color - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV);

    color = agx_contrast(color);
    color = AGX_OUTSET * color;

    // AgX emits display-encoded values, but this pass writes to an sRGB target
    // that applies its own encode. Undo AgX's so the two do not compound.
    return clamp(pow(max(color, vec3(0.0)), vec3(2.2)), vec3(0.0), vec3(1.0));
}

/// Plain Reinhard, kept as a neutral reference point for comparisons.
fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return clamp(x / (vec3(1.0) + x), vec3(0.0), vec3(1.0));
}

// ── The photographic camera (sunscar-photoreal, 6 Nov 2026) ────────────────────────────────────────────────────
//
// A tone response measured off a camera, in AgX's working space. Per channel, in stops from middle grey, two halves
// of the smooth clamp u / (1 + u^p)^(1/p) meet at the pivot: the toe below it, the shoulder above. Fitted (rimeborn-fp
// tools/photoreal) to a digital camera's JPEG response: within a tenth of a stop from 7 stops under middle grey to 2
// over, so a dim wood keeps the detail a photograph keeps, and it never reaches black (10 stops under is still about
// 1/255). The shoulder is a third of a stop softer than a camera's at +3 to +4, which keeps the form of bright cloud,
// and AgX's matrices keep hue as AgX does: a fire's orange drifts +12 degrees two stops over (ACES: +29).

/// Middle grey on the curve, display-encoded (gamma 2.2): 0.4647^2.2 = 0.185.
const PHOTO_PIVOT: f32 = 0.4647;
/// Display-encoded rise per stop just below and just above middle grey.
const PHOTO_SLOPE_LO: f32 = 0.1242;
const PHOTO_SLOPE_HI: f32 = 0.1400;
/// How sharply the toe and the shoulder turn (the clamp's p).
const PHOTO_TOE: f32 = 2.146;
const PHOTO_SHOULDER: f32 = 3.0;

fn photo_clamp(u: vec3<f32>, p: f32) -> vec3<f32> {
    let v = max(u, vec3<f32>(0.0));
    return v / pow(vec3<f32>(1.0) + pow(v, vec3<f32>(p)), vec3<f32>(1.0 / p));
}

/// The photographic curve. `contrast` scales its slope about middle grey (1 as fitted): steeper deepens the darks
/// and brightens the lights, but the toe still only approaches black, so no contrast can crush them.
fn photographic(color_in: vec3<f32>, contrast: f32) -> vec3<f32> {
    let c = AGX_INSET * max(color_in, vec3<f32>(0.0));
    let stops = log2(max(c, vec3<f32>(1e-10)) / 0.18);
    let k = max(contrast, 0.05);
    let lo = PHOTO_PIVOT - PHOTO_PIVOT * photo_clamp(-stops * (PHOTO_SLOPE_LO * k / PHOTO_PIVOT), PHOTO_TOE);
    let hi = PHOTO_PIVOT + (1.0 - PHOTO_PIVOT) * photo_clamp(stops * (PHOTO_SLOPE_HI * k / (1.0 - PHOTO_PIVOT)), PHOTO_SHOULDER);
    let encoded = AGX_OUTSET * select(hi, lo, stops < vec3<f32>(0.0));
    return clamp(pow(max(encoded, vec3<f32>(0.0)), vec3<f32>(2.2)), vec3<f32>(0.0), vec3<f32>(1.0));
}
