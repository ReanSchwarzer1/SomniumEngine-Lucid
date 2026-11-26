//! The light's switches (sunscar-engine, 26 Nov 2026, for sunscar-photoreal): numbers the look is tuned with, read
//! once from the environment at start. Each default is the value the engine had before, so a launch without them draws
//! the same picture: no shader's text is touched then, and no texture changes size or format.
//!
//! | Variable | Default | What it sets |
//! |---|---|---|
//! | `SOMNIUM_MIE_SCALE` | 1 | The atmosphere's aerosols: Hillaire's Mie scattering (3.996e-3 per km) and extinction (4.4e-3) times this. |
//! | `SOMNIUM_MIE_G` | 0.8 | The aerosols' phase asymmetry: how tight the glow round the sun is. |
//! | `SOMNIUM_GI_HIT_SKY` | 0.75 | How much of the sky a ReSTIR GI hit on a mesh sees (`restir_gi.wgsl`'s `GI_HIT_SKY`). |
//! | `SOMNIUM_GTAO_FORMAT` | `rgba8` | `rgba16f` keeps GTAO's visibility and bent normal in half floats (8 bits banded). |
//! | `SOMNIUM_ENV_CUBE` | 256 | The environment cube's face size, a power of two from 64 to 2048 (both cubes). |
//! | `SOMNIUM_GI_SCALE` | 2 | The GI grid step in pixels, 1 for full resolution: read in `pass/restir_gi.rs`, listed so the table is whole. |
//!
//! The aerosol finding behind the first switch is lucid/sky-aerosol 5291071 (sunscar-photoreal and sunscar-sky, 19 Nov
//! 2026): Hillaire's published Mie is a very clean day, an aerosol optical depth of about 0.005 at 550 nm, against 0.1
//! to 0.25 for late-autumn air over Italy. At the clean value a clear sky put about 4,300 lux of sky light on open
//! ground under a 35-degree sun, against 8,000-12,000 in life; shade sat 3.6 stops under sun against 2.2-2.6, and dusk
//! came out 1.5-2.5 times published twilight. 5291071 is `SOMNIUM_MIE_SCALE=25` (an optical depth of about 0.13): its
//! extinction was 0.111 against 0.110 here, because this switch keeps Hillaire's single-scattering albedo (0.908)
//! where 5291071 set 0.9. The one value feeds the sky's light, the dusk, the froxel haze and the aerial perspective
//! past the volume, so each look's own fog may come down to keep its visibility.

use std::borrow::Cow;
use std::sync::LazyLock;

/// The values one launch runs with.
#[derive(Clone, Debug, PartialEq)]
pub struct LightSwitches {
    pub mie_scale: f32,
    pub mie_g: f32,
    pub gi_hit_sky: f32,
    pub gtao_rgba16f: bool,
    pub env_cube: u32,
}

impl Default for LightSwitches {
    fn default() -> Self {
        Self { mie_scale: 1.0, mie_g: 0.8, gi_hit_sky: 0.75, gtao_rgba16f: false, env_cube: 256 }
    }
}

/// The exact lines the switches replace, as the shaders carry them. A test holds each one to its file, so an edit
/// of a line fails the build's tests instead of leaving its switch silently dead.
const MIE_SCATTERING: &str = "const MIE_SCATTERING: f32 = 3.996e-3;";
const MIE_EXTINCTION: &str = "const MIE_EXTINCTION: f32 = 4.4e-3;";
const MIE_G: &str = "const MIE_G:          f32 = 0.8;";
const GI_HIT_SKY: &str = "const GI_HIT_SKY: f32 = 0.75;";
const GTAO_STORAGE: &str = "texture_storage_2d<rgba8unorm, write>";

impl LightSwitches {
    fn from_env() -> Self {
        let d = Self::default();
        let float = |name: &str, fallback: f32, ok: fn(f32) -> bool| match std::env::var(name) {
            Err(_) => fallback,
            Ok(text) => match text.trim().parse::<f32>() {
                Ok(v) if v.is_finite() && ok(v) => v,
                _ => {
                    tracing::warn!(variable = name, value = %text, fallback, "light switch out of range: the default stands");
                    fallback
                }
            },
        };
        let gtao_rgba16f = match std::env::var("SOMNIUM_GTAO_FORMAT").as_deref().map(str::trim) {
            Err(_) | Ok("rgba8") => false,
            Ok("rgba16f") => true,
            Ok(other) => {
                tracing::warn!(value = other, "SOMNIUM_GTAO_FORMAT is rgba8 or rgba16f: rgba8 stands");
                false
            }
        };
        let env_cube = match std::env::var("SOMNIUM_ENV_CUBE") {
            Err(_) => d.env_cube,
            Ok(text) => match text.trim().parse::<u32>() {
                Ok(v) if v.is_power_of_two() && (64..=2048).contains(&v) => v,
                _ => {
                    tracing::warn!(value = %text, "SOMNIUM_ENV_CUBE is a power of two from 64 to 2048: 256 stands");
                    d.env_cube
                }
            },
        };
        Self {
            mie_scale: float("SOMNIUM_MIE_SCALE", d.mie_scale, |v| v > 0.0 && v <= 1000.0),
            mie_g: float("SOMNIUM_MIE_G", d.mie_g, |v| v > -0.999 && v < 0.999),
            gi_hit_sky: float("SOMNIUM_GI_HIT_SKY", d.gi_hit_sky, |v| (0.0..=1.0).contains(&v)),
            gtao_rgba16f,
            env_cube,
        }
    }

    /// True when some shader's text differs from the default's (the format and the cube change only textures).
    pub fn changes_shaders(&self) -> bool {
        let d = Self::default();
        self.mie_scale != d.mie_scale || self.mie_g != d.mie_g || self.gi_hit_sky != d.gi_hit_sky || self.gtao_rgba16f
    }

    /// One line for the log and the shader fingerprint.
    pub fn summary(&self) -> String {
        let gi_scale = std::env::var("SOMNIUM_GI_SCALE").unwrap_or_else(|_| "2 (default)".into());
        format!(
            "mie_scale {} mie_g {} gi_hit_sky {} gtao {} env_cube {} gi_scale {}",
            self.mie_scale,
            self.mie_g,
            self.gi_hit_sky,
            if self.gtao_rgba16f { "rgba16f" } else { "rgba8" },
            self.env_cube,
            gi_scale.trim()
        )
    }

    /// A composed module's text with these values in place of the defaults; borrowed when nothing changes.
    pub fn tune<'a>(&self, module: &str, text: &'a str) -> Cow<'a, str> {
        if !self.changes_shaders() {
            return Cow::Borrowed(text);
        }
        let d = Self::default();
        let mut out = text.to_string();
        if self.mie_scale != d.mie_scale {
            // In f64, so 25 prints as 0.0999 and not as the f32 product's 0.09990001.
            let scaled = |base: f64| (base * f64::from(self.mie_scale)) as f32;
            out = out.replace(MIE_SCATTERING, &format!("const MIE_SCATTERING: f32 = {:?};", scaled(3.996e-3)));
            out = out.replace(MIE_EXTINCTION, &format!("const MIE_EXTINCTION: f32 = {:?};", scaled(4.4e-3)));
        }
        if self.mie_g != d.mie_g {
            out = out.replace(MIE_G, &format!("const MIE_G: f32 = {:?};", self.mie_g));
        }
        if self.gi_hit_sky != d.gi_hit_sky {
            out = out.replace(GI_HIT_SKY, &format!("const GI_HIT_SKY: f32 = {:?};", self.gi_hit_sky));
        }
        // Only GTAO's own targets: other passes keep their rgba8 storage.
        if self.gtao_rgba16f && module == "gtao.wgsl" {
            out = out.replace(GTAO_STORAGE, "texture_storage_2d<rgba16float, write>");
        }
        Cow::Owned(out)
    }
}

static SWITCHES: LazyLock<LightSwitches> = LazyLock::new(|| {
    let switches = LightSwitches::from_env();
    let defaults = switches == LightSwitches::default();
    tracing::info!(defaults, "light switches: {}", switches.summary());
    switches
});

/// This launch's switches (read and logged once).
pub fn get() -> &'static LightSwitches {
    &SWITCHES
}

#[cfg(test)]
mod tests {
    use super::*;

    const ATMOSPHERE: &str = include_str!("shaders/atmosphere.wgsl");
    const RESTIR_GI: &str = include_str!("shaders/restir_gi.wgsl");
    const GTAO: &str = include_str!("shaders/gtao.wgsl");

    #[test]
    fn each_replaced_line_is_in_its_shader() {
        assert_eq!(ATMOSPHERE.matches(MIE_SCATTERING).count(), 1);
        assert_eq!(ATMOSPHERE.matches(MIE_EXTINCTION).count(), 1);
        assert_eq!(ATMOSPHERE.matches(MIE_G).count(), 1);
        assert_eq!(RESTIR_GI.matches(GI_HIT_SKY).count(), 1);
        assert_eq!(GTAO.matches(GTAO_STORAGE).count(), 2, "GTAO's raw and denoised targets");
    }

    #[test]
    fn the_defaults_leave_every_shader_as_it_was() {
        let d = LightSwitches::default();
        assert!(!d.changes_shaders());
        assert!(matches!(d.tune("atmosphere.wgsl", ATMOSPHERE), Cow::Borrowed(_)));
    }

    #[test]
    fn the_values_land_in_the_text() {
        let s = LightSwitches { mie_scale: 25.0, mie_g: 0.7, gi_hit_sky: 0.5, gtao_rgba16f: true, ..Default::default() };
        let atmosphere = s.tune("atmosphere.wgsl", ATMOSPHERE);
        assert!(atmosphere.contains("const MIE_SCATTERING: f32 = 0.0999;"));
        assert!(atmosphere.contains("const MIE_EXTINCTION: f32 = 0.11;"));
        assert!(atmosphere.contains("const MIE_G: f32 = 0.7;"));
        assert!(s.tune("restir_gi.wgsl", RESTIR_GI).contains("const GI_HIT_SKY: f32 = 0.5;"));
        let gtao = s.tune("gtao.wgsl", GTAO);
        assert_eq!(gtao.matches("texture_storage_2d<rgba16float, write>").count(), 2);
        assert!(!gtao.contains(GTAO_STORAGE));
        // Another pass's rgba8 storage is not GTAO's.
        assert_eq!(s.tune("other.wgsl", GTAO_STORAGE), GTAO_STORAGE);
    }
}
