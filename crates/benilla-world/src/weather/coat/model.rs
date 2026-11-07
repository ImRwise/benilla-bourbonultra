//! fork: the weather coat's arithmetic, with no engine in it: how wet and how snow-covered one
//! unit is, how both change each step, and the one `u32` the shader reads per rig slot.
//!
//! A [`Coat`] is two scalars in 0..1 that belong to a unit, not to the ground under it:
//!
//! - `wet` rises while rain reaches the unit (a soaked coat in about [`WET_UP_S`] s of full rain,
//!   against the ground's 90 s: a body wets faster than a road), jumps to full when the unit is
//!   in deep water, and dries after the rain stops or the unit steps under cover.
//! - `snow` builds while snow reaches a unit that is standing still, builds slowly while it
//!   walks, is shaken off while it runs, and melts into `wet` indoors or in the warm.
//!
//! "Reaches" is the unit's `open` share: 0 under a roof or indoors, 1 under open sky. The ground
//! model's wetness ([`super::super::wetness`]) is what a unit that has just been spawned starts
//! from, so a creature that streams in mid-storm is already as wet as the world.

/// Seconds from dry to soaked under full rain.
pub const WET_UP_S: f32 = 20.0;
/// Seconds from soaked to dry once the rain stops, standing still.
pub const DRY_S: f32 = 120.0;
/// Seconds for deep water to soak a unit through.
pub const SOAK_S: f32 = 2.0;
/// Seconds from bare to the snowfall's full cover on a unit that stands still in full snow.
pub const SNOW_UP_S: f32 = 40.0;
/// Seconds for snow to melt off a unit that is indoors or under cover.
pub const MELT_IN_S: f32 = 150.0;
/// Seconds for snow to melt off a unit that is outdoors with the snow stopped.
pub const MELT_OUT_S: f32 = 600.0;
/// Seconds for rain to wash a full snow cover off a unit in full rain.
pub const RAIN_WASH_S: f32 = 40.0;
/// Wetness added per unit of snow cover melted.
pub const MELT_WET_GAIN: f32 = 1.5;
/// The speed (yd/s) at which snow no longer settles and begins to be shaken off: a run.
pub const RUN_SPEED: f32 = 7.0;
/// Cover shed per yard travelled above [`SHED_FLOOR`], so a walk keeps its snow and a run loses it.
pub const SHED_PER_YD: f32 = 0.006;
/// The speed (yd/s) below which nothing is shed.
pub const SHED_FLOOR: f32 = 1.0;
/// The longest step one call integrates, seconds: a stall must not soak or dry a unit in one jump.
pub const MAX_STEP_S: f32 = 10.0;
/// Fastest speed (yd/s) the model believes: a teleport or a knockback is not a sprint.
pub const MAX_SPEED: f32 = 30.0;

/// Bit 31 of the shader word: this slot belongs to a unit the coat owns, so the shader skips the
/// zone-wide wet look for it and reads the coat instead. Always set for a coated unit, even a dry
/// one, so a unit under a roof is not wetted by the world's wetness.
pub const MANAGED: u32 = 1 << 31;

/// How wet and how snowy one unit is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Coat {
    pub wet: f32,
    pub snow: f32,
}

/// What reaches a unit this step.
#[derive(Clone, Copy, Debug, Default)]
pub struct Conditions {
    /// The rain rate in the zone, 0..1.
    pub rain: f32,
    /// The snowfall rate in the zone, 0..1.
    pub snowfall: f32,
    /// The unit's share of open sky: 0 indoors or under a roof, 1 in the open.
    pub open: f32,
    /// How far into deep water the unit stands: 0 dry, 1 swimming or chest deep.
    pub submerged: f32,
    /// How fast the unit moves, yd/s.
    pub speed: f32,
}

fn q(x: f32) -> u32 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u32
}

impl Coat {
    /// The coat a unit streams in with: the world's wetness and snow cover, as much of it as the
    /// unit's own sky share lets reach it.
    pub fn seeded(ambient_wet: f32, ambient_snow: f32, open: f32) -> Self {
        let open = open.clamp(0.0, 1.0);
        Self {
            wet: (ambient_wet * open).clamp(0.0, 1.0),
            snow: (ambient_snow * open).clamp(0.0, 1.0),
        }
    }

    /// One step of `dt` seconds under `c`.
    pub fn step(&mut self, c: Conditions, dt: f32) {
        let dt = dt.clamp(0.0, MAX_STEP_S);
        if dt == 0.0 {
            return;
        }
        let open = c.open.clamp(0.0, 1.0);
        let rain = c.rain.clamp(0.0, 1.0) * open;
        let fall = c.snowfall.clamp(0.0, 1.0) * open;
        let sub = c.submerged.clamp(0.0, 1.0);
        let speed = c.speed.clamp(0.0, MAX_SPEED);

        // Wet: toward the rain's level (a drizzle never quite soaks), or the water's, whichever
        // is wetter; away from it at the drying rate, a little faster on the move.
        let mut target = 0.0;
        let mut up = 0.0;
        if rain > 0.0 {
            target = 0.55 + 0.45 * rain;
            up = (0.35 + 0.65 * rain) / WET_UP_S;
        }
        if sub > 0.0 && sub >= target {
            target = sub;
            up = 1.0 / SOAK_S;
        }
        if self.wet < target {
            self.wet = (self.wet + up * dt).min(target);
        } else {
            let dry = (1.0 + speed.min(10.0) / 20.0) / DRY_S;
            self.wet = (self.wet - dry * dt).max(target);
        }

        // Snow: settles while it falls on the unit (less the faster it moves), is shaken off
        // above a walk, is washed by rain, drowned by deep water and melted by warmth.
        let moving = (speed / RUN_SPEED).clamp(0.0, 1.0);
        if fall > 0.0 {
            let cap = 0.5 + 0.5 * fall;
            if self.snow < cap {
                self.snow = (self.snow + fall * (1.0 - 0.85 * moving) / SNOW_UP_S * dt).min(cap);
            }
        } else {
            // Warmth wins over the cold: indoors it goes in MELT_IN_S, outdoors in MELT_OUT_S.
            let rate = (1.0 - open) / MELT_IN_S + open / MELT_OUT_S;
            let melt = (rate * dt).min(self.snow);
            self.snow -= melt;
            // Melt wets the coat. Snow is bulky: a cover's worth of it leaves more damp cloth
            // than its own weight in water, and indoors the gain outruns the drying rate, so a
            // unit that comes in from a blizzard is left damp, then dries.
            self.wet = (self.wet + melt * MELT_WET_GAIN).min(1.0);
        }
        self.snow -= (speed - SHED_FLOOR).max(0.0) * SHED_PER_YD * dt;
        self.snow -= rain * dt / RAIN_WASH_S;
        self.snow -= sub * dt / SOAK_S;
        self.wet = self.wet.clamp(0.0, 1.0);
        self.snow = self.snow.clamp(0.0, 1.0);
    }

    /// The shader word: [`MANAGED`], snow in bits 8..16, wet in bits 0..8.
    pub fn pack(self) -> u32 {
        MANAGED | (q(self.snow) << 8) | q(self.wet)
    }

    /// The coat a shader word holds, `None` for a slot the coat does not own.
    pub fn unpack(word: u32) -> Option<Self> {
        if word & MANAGED == 0 {
            return None;
        }
        Some(Self {
            wet: (word & 0xff) as f32 / 255.0,
            snow: ((word >> 8) & 0xff) as f32 / 255.0,
        })
    }
}

/// The zone's snow cover, the snow counterpart of [`super::super::wetness::Wetness`]: what a
/// freshly spawned unit starts from, and the `snow` row of the environment block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Snowpack {
    pub cover: f32,
}

/// Seconds from bare to the snowfall's cover in full snow.
pub const PACK_UP_S: f32 = 120.0;
/// Seconds from a full cover to bare once the snow stops.
pub const PACK_MELT_S: f32 = 300.0;

impl Snowpack {
    /// One real-time step; `snowfall` is the snow rate 0..1, 0 when it is not snowing.
    pub fn step(&mut self, snowfall: f32, dt: f32) {
        let f = snowfall.clamp(0.0, 1.0);
        if f > 0.0 {
            let target = 0.5 + 0.5 * f;
            let rate = (0.35 + 0.65 * f) / PACK_UP_S;
            if self.cover < target {
                self.cover = (self.cover + rate * dt).min(target);
            } else {
                self.cover = (self.cover - dt / PACK_MELT_S).max(target);
            }
        } else {
            self.cover = (self.cover - dt / PACK_MELT_S).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(c: &mut Coat, cond: Conditions, secs: f32) {
        let steps = (secs * 10.0) as usize;
        for _ in 0..steps {
            c.step(cond, 0.1);
        }
    }

    const FULL_RAIN: Conditions = Conditions {
        rain: 1.0,
        snowfall: 0.0,
        open: 1.0,
        submerged: 0.0,
        speed: 0.0,
    };
    const FULL_SNOW: Conditions = Conditions {
        rain: 0.0,
        snowfall: 1.0,
        open: 1.0,
        submerged: 0.0,
        speed: 0.0,
    };

    #[test]
    fn a_unit_in_full_rain_soaks_in_about_twenty_seconds() {
        let mut c = Coat::default();
        run(&mut c, FULL_RAIN, 19.5);
        assert!(c.wet > 0.95 && c.wet < 1.0, "{}", c.wet);
        run(&mut c, FULL_RAIN, 5.0);
        assert_eq!(c.wet, 1.0);
    }

    #[test]
    fn a_drizzle_never_quite_soaks() {
        let mut c = Coat::default();
        let drizzle = Conditions {
            rain: 0.2,
            ..FULL_RAIN
        };
        run(&mut c, drizzle, 300.0);
        assert!((c.wet - 0.64).abs() < 1e-3, "{}", c.wet);
    }

    #[test]
    fn a_roof_keeps_the_rain_off() {
        let mut c = Coat::default();
        let under_roof = Conditions {
            open: 0.0,
            ..FULL_RAIN
        };
        run(&mut c, under_roof, 120.0);
        assert_eq!(c.wet, 0.0);
    }

    #[test]
    fn stepping_under_cover_dries_it_slowly_not_at_once() {
        let mut c = Coat {
            wet: 1.0,
            snow: 0.0,
        };
        let cover = Conditions {
            open: 0.0,
            ..Conditions::default()
        };
        run(&mut c, cover, 60.0);
        assert!(
            (c.wet - 0.5).abs() < 0.01,
            "half dry at a minute: {}",
            c.wet
        );
        run(&mut c, cover, 61.0);
        assert_eq!(c.wet, 0.0);
    }

    #[test]
    fn running_dries_faster_than_standing() {
        let mut still = Coat {
            wet: 1.0,
            snow: 0.0,
        };
        let mut runner = still;
        run(&mut still, Conditions::default(), 30.0);
        run(
            &mut runner,
            Conditions {
                speed: 7.0,
                ..Conditions::default()
            },
            30.0,
        );
        assert!(runner.wet < still.wet);
    }

    #[test]
    fn deep_water_soaks_through_in_two_seconds_and_drowns_the_snow() {
        let mut c = Coat {
            wet: 0.0,
            snow: 0.8,
        };
        let swim = Conditions {
            submerged: 1.0,
            open: 0.0,
            ..Conditions::default()
        };
        run(&mut c, swim, 2.1);
        assert_eq!(c.wet, 1.0);
        assert_eq!(c.snow, 0.0);
    }

    #[test]
    fn a_unit_standing_in_snow_gathers_a_cover_a_runner_does_not() {
        let mut still = Coat::default();
        run(&mut still, FULL_SNOW, 60.0);
        assert!(still.snow > 0.95, "{}", still.snow);
        assert_eq!(still.wet, 0.0, "snow settles dry");

        let mut runner = Coat::default();
        run(
            &mut runner,
            Conditions {
                speed: RUN_SPEED,
                ..FULL_SNOW
            },
            60.0,
        );
        assert!(runner.snow < 0.02, "{}", runner.snow);
    }

    #[test]
    fn a_walker_slowly_gathers_snow() {
        let mut c = Coat::default();
        run(
            &mut c,
            Conditions {
                speed: 2.5,
                ..FULL_SNOW
            },
            60.0,
        );
        assert!(c.snow > 0.1 && c.snow < 0.9, "{}", c.snow);
    }

    #[test]
    fn a_flurry_covers_only_half() {
        let mut c = Coat::default();
        run(
            &mut c,
            Conditions {
                snowfall: 0.05,
                ..FULL_SNOW
            },
            600.0,
        );
        assert!((c.snow - 0.525).abs() < 0.01, "{}", c.snow);
    }

    #[test]
    fn indoors_the_snow_melts_into_water() {
        let mut c = Coat {
            wet: 0.0,
            snow: 1.0,
        };
        let indoors = Conditions {
            open: 0.0,
            ..Conditions::default()
        };
        let mut damp: f32 = 0.0;
        for _ in 0..(160 * 10) {
            c.step(indoors, 0.1);
            damp = damp.max(c.wet);
        }
        assert_eq!(c.snow, 0.0);
        assert!(damp > 0.15, "the melt wets the coat: {damp}");
        // ...and a damp coat indoors then dries.
        run(&mut c, indoors, 120.0);
        assert_eq!(c.wet, 0.0);
    }

    #[test]
    fn outdoors_the_snow_outlasts_the_flakes() {
        let mut c = Coat {
            wet: 0.0,
            snow: 1.0,
        };
        run(
            &mut c,
            Conditions {
                open: 1.0,
                ..Conditions::default()
            },
            120.0,
        );
        assert!(c.snow > 0.7, "{}", c.snow);
    }

    #[test]
    fn rain_washes_snow_away() {
        let mut c = Coat {
            wet: 0.0,
            snow: 1.0,
        };
        run(&mut c, FULL_RAIN, 45.0);
        assert!(c.snow < 0.05, "{}", c.snow);
        assert!(c.wet > 0.9);
    }

    #[test]
    fn a_stall_cannot_soak_or_dry_in_one_step() {
        let mut c = Coat::default();
        c.step(FULL_RAIN, 1000.0);
        assert!(
            c.wet < 0.6,
            "one stall steps at most {MAX_STEP_S} s: {}",
            c.wet
        );
    }

    #[test]
    fn a_new_unit_starts_as_wet_as_the_world_lets_it() {
        let c = Coat::seeded(0.8, 0.6, 1.0);
        assert_eq!((c.wet, c.snow), (0.8, 0.6));
        let indoors = Coat::seeded(0.8, 0.6, 0.0);
        assert_eq!(indoors, Coat::default());
    }

    #[test]
    fn the_word_round_trips_and_a_dry_coat_is_still_owned() {
        let c = Coat {
            wet: 0.5,
            snow: 1.0,
        };
        let back = Coat::unpack(c.pack()).unwrap();
        assert!((back.wet - 0.5).abs() < 1.0 / 255.0);
        assert_eq!(back.snow, 1.0);
        let dry = Coat::default().pack();
        assert_ne!(dry, 0, "a dry unit is not the unowned zero word");
        assert_eq!(Coat::unpack(dry), Some(Coat::default()));
        assert_eq!(Coat::unpack(0), None);
    }

    #[test]
    fn snowpack_builds_and_melts_like_the_ground_does() {
        let mut p = Snowpack::default();
        for _ in 0..(300 * 10) {
            p.step(1.0, 0.1);
        }
        assert_eq!(p.cover, 1.0);
        for _ in 0..(151 * 10) {
            p.step(0.0, 0.1);
        }
        assert!((p.cover - 0.5).abs() < 0.01, "{}", p.cover);
        for _ in 0..(151 * 10) {
            p.step(0.0, 0.1);
        }
        assert_eq!(p.cover, 0.0);
    }
}
