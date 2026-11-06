//! The engine's own mark, at sizes the operating system asks for.
//!
//! MORROWIND-J step 2. The editor has drawn its mark in its own title bar since
//! Phase 26; what it has never had is an icon the *system* can show, so the
//! taskbar button, the alt-tab card and the window list all fell back to the
//! blank default. Those are the places a person looks to find a window among
//! twenty others, and a blank one is the hardest to find.
//!
//! The source is Lucid's clear-ray mark, `assets/brand/lucid-mark-mono.svg`, the
//! same drawing [`crate::icons::IconId::EngineMark`] uses, and below 21 px its
//! hand-hinted 16-unit drawing. One drawing rather than a hand-made bitmap beside
//! it, because a mark that drifts from the one in the title bar is worse than no
//! mark: it reads as two applications. The set and its build script are in
//! `F:/Somnium Games/rimeborn-fp/design/lucid_logo/final`.

/// One rasterised mark: RGBA8, `size` square, ready for `winit::window::Icon`.
pub struct MarkImage {
    pub rgba: Vec<u8>,
    pub size: u32,
}

/// Lucid's one accent, the ray's cyan: the window and taskbar icon, the
/// title-bar mark, and the ray and the crescent C in the splash and on About.
/// It reads on dark and light taskbars alike (5.6:1 and 2.9:1).
pub const LUCID_CYAN: [u8; 4] = [0x0B, 0x9C, 0xC2, 0xFF];

/// The mark hinted on a 16-unit grid, for 20 px and under, where the master's
/// tapering ray thins to nothing.
const MICRO: &str = include_str!("../assets/brand/lucid-mark-micro-16.svg");

/// The size Windows asks for in the taskbar and alt-tab (`ICON_BIG`).
///
/// 64 rather than 256: the mark is drawn on a 64-unit grid, so a larger raster
/// buys nothing but memory, and Windows downsamples cleanly.
pub const LARGE: u32 = 64;

/// The size a title bar and a window-list row ask for (`ICON_SMALL`).
pub const SMALL: u32 = 32;

/// Rasterise the engine mark at `size`, tinted and on a transparent field.
///
/// Returns `None` only if the vendored source fails to parse or the pixmap
/// cannot be allocated, which is a broken build rather than a runtime
/// condition — callers carry on without an icon rather than refusing to open a
/// window over a missing decoration.
#[must_use]
pub fn mark(size: u32, tint: [u8; 4]) -> Option<MarkImage> {
    let source = if size <= 20 {
        MICRO
    } else {
        crate::icon_svg::source_for(crate::icons::IconId::EngineMark)?
    };
    // The shared rasteriser keeps coverage and discards colour, which is
    // exactly what is wanted: the mark fills in `currentColor`, and the tint
    // belongs to whoever is drawing it.
    let coverage = crate::icon_svg::rasterize(source, size)?;
    let mut rgba = Vec::with_capacity(coverage.len() * 4);
    for alpha in coverage {
        // Straight alpha, not premultiplied: `Icon::from_rgba` documents 32bpp
        // RGBA and Windows composites it itself. Premultiplying here dims the
        // edge pixels of every stroke.
        rgba.extend_from_slice(&[tint[0], tint[1], tint[2], alpha]);
    }
    Some(MarkImage { rgba, size })
}

/// The mark in Lucid's cyan, which is what the title bar draws.
#[must_use]
pub fn accent_mark(size: u32) -> Option<MarkImage> {
    mark(size, LUCID_CYAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mark_rasterises_at_both_sizes_the_system_asks_for() {
        for size in [SMALL, LARGE] {
            let image = mark(size, [0x4C, 0x8D, 0xFF, 0xFF]).expect("vendored source parses");
            assert_eq!(image.size, size);
            assert_eq!(
                image.rgba.len(),
                (size * size * 4) as usize,
                "Icon::from_rgba rejects any other length"
            );
        }
    }

    #[test]
    fn the_mark_is_drawn_rather_than_a_transparent_square() {
        // The failure this catches is silent: a source that parsed but rendered
        // nothing gives a fully transparent icon, which looks exactly like
        // having no icon at all.
        let image = accent_mark(LARGE).expect("vendored source parses");
        let lit = image.rgba.chunks_exact(4).filter(|px| px[3] > 8).count();
        let total = (LARGE * LARGE) as usize;
        assert!(
            lit > total / 20,
            "only {lit} of {total} pixels have any coverage"
        );
        assert!(
            lit < total * 9 / 10,
            "{lit} of {total} pixels covered; the mark is a lens and a ray, not a filled square"
        );
    }

    #[test]
    fn sixteen_pixels_take_the_hinted_drawing() {
        // The master's ray tapers to nothing at 16 px; the hinted one keeps it a 2 px bar.
        let small = mark(16, LUCID_CYAN).expect("vendored micro mark parses");
        let hinted = crate::icon_svg::rasterize(MICRO, 16).expect("rasterises");
        assert_eq!(small.rgba.chunks_exact(4).map(|px| px[3]).collect::<Vec<_>>(), hinted);
        let ray_row = (8 * 16..9 * 16).filter(|&i| small.rgba[i * 4 + 3] > 200).count();
        assert!(ray_row >= 10, "the ray stays solid across the 16 px icon ({ray_row} px)");
    }

    #[test]
    fn the_tint_reaches_every_covered_pixel() {
        let tint = [0x11, 0x22, 0x33, 0xFF];
        let image = mark(SMALL, tint).expect("vendored source parses");
        for px in image.rgba.chunks_exact(4).filter(|px| px[3] > 0) {
            assert_eq!([px[0], px[1], px[2]], [tint[0], tint[1], tint[2]]);
        }
    }
}
