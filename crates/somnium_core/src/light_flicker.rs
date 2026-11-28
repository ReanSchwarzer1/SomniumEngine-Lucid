//! Failing-fixture flicker for local lights.
//!
//! A multiplier on the light's output, evaluated where lights are submitted to
//! the renderer. The authored [`LightComponent`](crate::LightComponent) is never
//! written, so a flickering lamp does not dirty the scene or save a random
//! intensity, and turning the component off restores the exact authored value.
//!
//! The pattern is a deterministic function of time and `seed`: a low buzz, and
//! occasional bursts where the fixture stutters toward dark. Two lamps with
//! different seeds never flicker in step.

use somnium_ecs::Component;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightFlickerComponent {
    pub enabled: bool,
    /// Depth of the continuous buzz, `0..1` of the light's output.
    pub strength: f32,
    /// Bursts per second, on average.
    pub rate: f32,
    /// Chance that a burst takes the fixture nearly dark, `0..1`.
    pub dropout: f32,
    /// Decorrelates fixtures that share every other setting.
    pub seed: f32,
}

impl Component for LightFlickerComponent {}

impl Default for LightFlickerComponent {
    fn default() -> Self {
        Self {
            enabled: true,
            strength: 0.25,
            rate: 0.6,
            dropout: 0.35,
            seed: 0.0,
        }
    }
}

fn hash(x: f32) -> f32 {
    ((x * 12.9898).sin() * 43_758.547).fract().abs()
}

fn value_noise(t: f32, seed: f32) -> f32 {
    let i = t.floor();
    let f = t - i;
    let s = f * f * (3.0 - 2.0 * f);
    hash(i + seed * 17.13) * (1.0 - s) + hash(i + 1.0 + seed * 17.13) * s
}

impl LightFlickerComponent {
    /// Output multiplier at `time` seconds, in `0..=1`.
    #[must_use]
    pub fn factor(&self, time: f32) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        let strength = self.strength.clamp(0.0, 1.0);
        // Mains buzz: fast, shallow, never regular.
        let buzz = 1.0 - strength * 0.5 * value_noise(time * 11.0, self.seed);
        let rate = self.rate.max(0.0);
        if rate <= 0.0 {
            return buzz.clamp(0.0, 1.0);
        }
        // Bursts: each window of 1/rate seconds may hold one short stutter.
        let window = time * rate + self.seed * 3.7;
        let index = window.floor();
        let phase = window - index;
        let burst_len = 0.35;
        if phase < burst_len && hash(index + 0.5) < self.dropout.clamp(0.0, 1.0) {
            // Square on/off chatter inside the burst, like a failing starter.
            let chatter = hash((time * 24.0).floor() + index * 7.0);
            let depth = if chatter < 0.55 { 0.04 } else { 0.55 + 0.3 * chatter };
            return (buzz * depth).clamp(0.0, 1.0);
        }
        buzz.clamp(0.0, 1.0)
    }
}

/// A light of the living (an oil lantern, a candle, a watch fire): it wavers as a flame does, and a lamp burns while
/// the open sky is dark and is out by day. A tag beside a `LightComponent`; the light itself is never written.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LampComponent {
    /// A lamp, lit from dusk to dawn. Off for a fire that burns all day.
    pub switched: bool,
    /// How far the flame dips, `0..1` of the light's output.
    pub flicker: f32,
}

impl Component for LampComponent {}

impl Default for LampComponent {
    fn default() -> Self {
        Self { switched: true, flicker: 0.15 }
    }
}

/// Lamps are lit when the open sky gives less than this, lux (dusk, a storm's dark), and put out when it gives
/// more than [`LAMP_OFF_LUX`]: between the two they stay as they are, so a cloud's edge does not flick a village.
pub const LAMP_LUX: f32 = 1000.0;
pub const LAMP_OFF_LUX: f32 = 1200.0;
/// Seconds over which a land's lamps are lit, one after another, as dusk falls, and put out as the day comes.
pub const LAMP_STAGGER_S: f32 = 90.0;
/// In less light than [`LAMP_NIGHT_LUX`] the night has come already, and in more than [`LAMP_DAY_LUX`] the day (a
/// clock moved on by hand, a still taken at midnight): the lamps still to be lit or put out are, one after another,
/// within [`LAMP_CATCH_UP_S`]. A picture taken seconds after the hour was set to midnight showed no lantern lit.
pub const LAMP_NIGHT_LUX: f32 = 10.0;
pub const LAMP_DAY_LUX: f32 = 10_000.0;
pub const LAMP_CATCH_UP_S: f32 = 4.0;

impl LampComponent {
    /// Output multiplier for the lamp at `position`, `time` seconds in, when the lamps are `night` of the way lit
    /// ([`LampSwitch::night`]). Each lamp has its own turn in the lighting (by where it stands) and catches over a
    /// few seconds.
    #[must_use]
    pub fn factor(&self, night: f32, position: glam::Vec3, time: f32) -> f32 {
        let turn = hash(position.x * 0.731 + position.y * 0.117 + position.z * 1.933);
        let lit = if self.switched { ((night - 0.02 - 0.93 * turn) / 0.03).clamp(0.0, 1.0) } else { 1.0 };
        let flame = 0.6 * value_noise(time * 3.1, turn * 40.0) + 0.4 * value_noise(time * 9.7, turn * 91.0);
        lit * (1.0 - self.flicker.clamp(0.0, 1.0) * flame)
    }
}

/// Whether the living's lamps burn: `night` runs from 0 (all out) to 1 (all lit) over [`LAMP_STAGGER_S`] once the
/// open sky gives under [`LAMP_LUX`], and back when the light returns. One per game, stepped once a frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct LampSwitch {
    night: f32,
    dark: bool,
    /// The last step's time, seconds; `None` before the first, which sets `night` whole.
    last: Option<f32>,
}

impl LampSwitch {
    /// Step to `time` seconds under an open sky giving `lux`.
    pub fn step(&mut self, lux: f32, time: f32) {
        self.dark = if lux < LAMP_LUX { true } else if lux > LAMP_OFF_LUX { false } else { self.dark };
        self.night = match self.last {
            None => f32::from(u8::from(self.dark)),
            Some(last) => {
                let come = if self.dark { lux < LAMP_NIGHT_LUX } else { lux > LAMP_DAY_LUX };
                let turn = (time - last).clamp(0.0, 1.0) / if come { LAMP_CATCH_UP_S } else { LAMP_STAGGER_S };
                (self.night + if self.dark { turn } else { -turn }).clamp(0.0, 1.0)
            }
        };
        self.last = Some(time);
    }

    /// How far the lamps are lit, `0..=1`.
    #[must_use]
    pub fn night(&self) -> f32 {
        self.night
    }
}

/// The light an open, level place takes from the sky, lux: the sun through the air (`sun_lux`, at its own bearing)
/// on the level, and the sky's own light as a share of the sun above the air (`sun_above_air_lux`) that follows the
/// twilight: about an eighth with the sun high, 0.4% with it on the horizon, a hundredth of that six degrees under.
/// `sun_up` is the sine of the sun's elevation. With the sun at 80,000 lux it passes 1,000 lux about two degrees up.
#[must_use]
pub fn open_sky_lux(sun_above_air_lux: f32, sun_lux: f32, sun_up: f32) -> f32 {
    // (elevation in degrees, ln of the sky's share)
    const SKY: [(f32, f32); 6] =
        [(-12.0, -16.3), (-6.0, -10.3), (0.0, -5.52), (5.0, -3.5), (15.0, -2.5), (90.0, -2.0)];
    let elevation = sun_up.clamp(-1.0, 1.0).asin().to_degrees();
    let share = match SKY.windows(2).find(|pair| elevation < pair[1].0) {
        _ if elevation <= SKY[0].0 => 0.0,
        Some(pair) => (pair[0].1 + (pair[1].1 - pair[0].1) * (elevation - pair[0].0) / (pair[1].0 - pair[0].0)).exp(),
        None => SKY[5].1.exp(),
    };
    sun_lux * sun_up.max(0.0) + sun_above_air_lux * share
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lamps_light_one_by_one_at_dusk_and_go_out_at_dawn() {
        // Noon, the sun on the horizon, and six degrees under: 80,000 lux above the air.
        assert!(open_sky_lux(80_000.0, 70_000.0, 0.57) > 40_000.0);
        let dusk = open_sky_lux(80_000.0, 2_000.0, 0.0);
        assert!((200.0..LAMP_LUX).contains(&dusk), "{dusk}");
        assert!(open_sky_lux(80_000.0, 0.0, (-6.0f32).to_radians().sin()) < 5.0);
        assert!(open_sky_lux(80_000.0, 20_000.0, 5.0f32.to_radians().sin()) > LAMP_LUX, "by day they are out");
        // A game that starts in the dark starts with its lamps lit; day puts them out over the stagger.
        let lamp = LampComponent { flicker: 0.0, ..Default::default() };
        let places: Vec<glam::Vec3> = (0..40).map(|i| glam::Vec3::new(i as f32 * 7.3, 2.0, i as f32 * -3.1)).collect();
        let lit = |switch: &LampSwitch| places.iter().filter(|p| lamp.factor(switch.night(), **p, 0.0) >= 1.0).count();
        let mut switch = LampSwitch::default();
        switch.step(dusk, 0.0);
        assert_eq!((switch.night(), lit(&switch)), (1.0, 40));
        let mut counts = Vec::new();
        for second in 1..=100 {
            switch.step(5_000.0, second as f32);
            counts.push(lit(&switch));
        }
        assert!(counts.windows(2).all(|pair| pair[1] <= pair[0]), "{counts:?}");
        assert!(counts[30] > 5 && counts[30] < 35, "a third of the way through some are out: {}", counts[30]);
        assert_eq!((counts[95], switch.night()), (0, 0.0));
        // The hour set to midnight by hand: the night has come, and they are all lit within a few seconds, still
        // one after another. Set back to noon, they are out as soon.
        let mut after = Vec::new();
        for tenth in 1..=50 {
            switch.step(0.5, 100.0 + tenth as f32 * 0.1);
            after.push(lit(&switch));
        }
        assert!(after[5] > 0 && after[5] < 40 && after[45] == 40, "{after:?}");
        for tenth in 1..=50 {
            switch.step(60_000.0, 105.0 + tenth as f32 * 0.1);
        }
        assert_eq!((lit(&switch), switch.night()), (0, 0.0));
        // Between 1,000 and 1,200 lux nothing changes: lit lamps stay lit, and lamps that are out stay out.
        let mut dusk_edge = LampSwitch::default();
        dusk_edge.step(900.0, 0.0);
        dusk_edge.step(1100.0, 1.0);
        assert_eq!(dusk_edge.night(), 1.0);
        dusk_edge.step(1300.0, 2.0);
        assert!(dusk_edge.night() < 1.0);
        let mut dawn_edge = LampSwitch::default();
        dawn_edge.step(5000.0, 0.0);
        dawn_edge.step(1100.0, 1.0);
        assert_eq!(dawn_edge.night(), 0.0);
        // A fire is never switched: it burns by day; a flame dips no deeper than its flicker.
        assert_eq!(LampComponent { switched: false, flicker: 0.0 }.factor(0.0, places[3], 1.0), 1.0);
        let wavering = LampComponent::default();
        let dips: Vec<f32> = (0..500).map(|i| wavering.factor(1.0, places[3], i as f32 * 0.05)).collect();
        assert!(dips.iter().all(|f| (0.85..=1.0).contains(f)) && dips.iter().any(|f| *f < 0.95));
    }

    #[test]
    fn disabled_is_exactly_the_authored_light() {
        let flicker = LightFlickerComponent {
            enabled: false,
            ..Default::default()
        };
        assert_eq!(flicker.factor(12.3), 1.0);
    }

    #[test]
    fn the_factor_stays_in_range_and_actually_dips() {
        let flicker = LightFlickerComponent {
            dropout: 1.0,
            ..Default::default()
        };
        let samples: Vec<f32> = (0..4000).map(|i| flicker.factor(i as f32 * 0.013)).collect();
        assert!(samples.iter().all(|f| (0.0..=1.0).contains(f)));
        assert!(samples.iter().any(|&f| f < 0.1), "bursts must reach near-dark");
        assert!(samples.iter().any(|&f| f > 0.8), "the lamp must also be lit");
    }

    #[test]
    fn seeds_decorrelate_neighbouring_fixtures() {
        let a = LightFlickerComponent { seed: 1.0, ..Default::default() };
        let b = LightFlickerComponent { seed: 2.0, ..Default::default() };
        let differ = (0..500)
            .filter(|i| (a.factor(*i as f32 * 0.05) - b.factor(*i as f32 * 0.05)).abs() > 1.0e-3)
            .count();
        assert!(differ > 250);
    }
}
