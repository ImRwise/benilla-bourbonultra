//! fork: weather that sticks to units. Rain soaks and snow settles on a player, an NPC, a
//! creature, a pet, a mount and everything worn or carried, and it stays there after the unit
//! walks under a roof, then dries or melts.
//!
//! The ground's wetness ([`super::wetness`]) is a property of the zone: every model reads it, and
//! a unit that steps under a porch is dry the instant it does. This module gives each unit a
//! [`WeatherCoat`] of its own, a pair of scalars ([`model::Coat`]) that build while the weather
//! reaches the unit and fall while it does not, so a unit can arrive wet, drip dry indoors, carry
//! snow on its shoulders and shake it off at a run.
//!
//! - **Owner.** The coat lives on the unit's net entity, the root that holds its
//!   [`GroundShade`]. Every rig below it (the body, a mount, a worn helm, a held weapon, a pet
//!   that is a child of it) finds that root by walking [`ChildOf`] and
//!   [`ParentModel`](crate::model_fade::ParentModel) up, so the whole unit wears one coat.
//!   Effects and doodads without a unit root above them have none and are untouched.
//! - **Inputs.** The zone's rain and snow rates ([`WeatherState`]), the unit's sky share (the
//!   shelter grid's roofs, and the interior classifier's verdict for units beyond the grid) and
//!   its depth in water, read each frame from the roots' transforms.
//! - **GPU.** One word a rig slot ([`RigCoats`], bit layout in [`model::Coat::pack`]) in a region
//!   of the shared light buffer, read by `wow_model.wgsl` through `benilla::wet_hook`. A slot whose
//!   word has [`model::MANAGED`] set takes its wet and snow looks from the coat and skips the
//!   zone-wide wet look; a slot with word 0 behaves exactly as before.
//!
//! The `rainSurfaces` cvar ([`super::wetness::RainSurfaces`]) gates the whole system: off writes
//! zero words and the models fall back to the zone-wide look. Capture knobs: `WOW_COAT_WET=<0..1>`
//! and `WOW_COAT_SNOW=<0..1>` pin every coat (a capture cannot wait minutes for the ramp).
//!
//! Limits: a unit more than ~48 yd from the camera is outside the shelter grid and counts as
//! open sky unless it is indoors; doodads without a canopy collider do not shelter (the grid's
//! own limit); deep water is any liquid, lava included.

use std::sync::Arc;

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::renderer::RenderQueue;
use bevy::render::{Render, RenderApp, RenderSystems};

use super::shelter::ShelterGrid;
use super::wetness::{RainSurfaces, Wetness};
use super::{WeatherKind, WeatherState};
use crate::entity_shade::GroundShade;
use crate::mesh_tag::MAX_RIG_SLOTS;
use crate::model_fade::ParentModel;
use crate::rig_palette::RigSkin;

pub mod model;
pub use model::{Coat, Conditions, Snowpack};

/// Bytes per slot in the coat region: one packed word ([`Coat::pack`]).
const SLOT_BYTES: u64 = 4;

/// Byte offset of the coat region in a `wow_light`-layout buffer, after the straddle clip table
/// and before the palette rows; it must match the shaders' struct order.
pub(crate) fn region_offset() -> u64 {
    crate::straddle::region_offset() + crate::straddle::region_bytes()
}

/// Bytes this region adds to every `wow_light`-layout buffer (8 KB at 2048 slots).
pub(crate) fn region_bytes() -> u64 {
    MAX_RIG_SLOTS as u64 * SLOT_BYTES
}

/// How far above a unit's feet its sky share is read, yd: waist height, so a roof a yard or two
/// overhead shelters it and the ground it stands on does not.
const PROBE_LIFT: f32 = 0.5;
/// Water depth (yd, surface above the feet) where a unit starts to be soaked, and where it is.
const SUBMERGE_FROM: f32 = 0.5;
const SUBMERGE_FULL: f32 = 1.5;
/// The bound on an owner walk; real chains are a handful of links, the rest is a cycle backstop.
const MAX_WALK: usize = 12;

/// The zone's snow cover, the snow counterpart of [`Wetness`]: what a unit that streams in starts
/// from, and the `snow` row of the environment block. Stepped by `wetness_tick`.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct ZoneSnow(pub Snowpack);

/// A unit's own wetness and snow cover, on its net entity (the one holding its [`GroundShade`]).
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct WeatherCoat {
    coat: Coat,
    /// Whether the first sight seeded it from the zone.
    seeded: bool,
    /// The root's position at the last step, for its speed.
    last_pos: Vec3,
    has_last: bool,
}

impl WeatherCoat {
    /// How wet and how snowy the unit is.
    pub fn coat(&self) -> Coat {
        self.coat
    }
}

/// The per-slot coat words, indexed by the `MeshTag` rig slot; `Arc`-shared for a cheap extract
/// and generation-stamped so a dry, snowless world uploads nothing.
#[derive(Resource, Clone, ExtractResource)]
pub struct RigCoats {
    slots: Arc<Vec<u32>>,
    generation: u64,
}

impl Default for RigCoats {
    fn default() -> Self {
        Self {
            slots: Arc::new(vec![0; MAX_RIG_SLOTS]),
            generation: 0,
        }
    }
}

impl RigCoats {
    /// Sets a slot's word (`0` leaves the slot to the zone-wide look). Slot 0, shared by
    /// everything unskinned, is never written.
    pub fn set(&mut self, slot: u16, word: u32) {
        let i = slot as usize;
        if i == 0 || i >= MAX_RIG_SLOTS || self.slots[i] == word {
            return;
        }
        Arc::make_mut(&mut self.slots)[i] = word;
        self.generation += 1;
    }

    /// Clears a slot; `RigSkin`'s free hook calls it, so a recycled slot never inherits a dead
    /// unit's coat.
    pub(crate) fn clear(&mut self, slot: u16) {
        self.set(slot, 0);
    }

    pub fn get(&self, slot: u16) -> u32 {
        self.slots.get(slot as usize).copied().unwrap_or(0)
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn env_unit(key: &str) -> Option<f32> {
    crate::launch_env::var(key)?
        .trim()
        .parse::<f32>()
        .ok()
        .map(|v| v.clamp(0.0, 1.0))
}

/// The capture pins, read once.
fn pins() -> (Option<f32>, Option<f32>) {
    static PINS: std::sync::OnceLock<(Option<f32>, Option<f32>)> = std::sync::OnceLock::new();
    *PINS.get_or_init(|| (env_unit("WOW_COAT_WET"), env_unit("WOW_COAT_SNOW")))
}

/// Steps every coat one real-time frame: what the weather brings against the unit's sky share,
/// its water depth and its speed.
#[allow(clippy::too_many_arguments)]
fn coat_tick(
    time: Res<Time<Real>>,
    state: Res<WeatherState>,
    cvar: Res<RainSurfaces>,
    wetness: Res<Wetness>,
    zone_snow: Res<ZoneSnow>,
    grid: Res<ShelterGrid>,
    water_index: Option<Res<crate::liquid::WaterIndex>>,
    water: Query<&crate::liquid::WaterChunkInfo>,
    mut coats: Query<(&GlobalTransform, &mut WeatherCoat, Option<&GroundShade>)>,
) {
    if !super::wetness::enabled(&cvar) {
        return;
    }
    let dt = time.delta_secs();
    let rain = if state.effect_kind == WeatherKind::Rain {
        state.effect_density.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let snowfall = if state.effect_kind == WeatherKind::Snow {
        state.effect_density.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (pin_wet, pin_snow) = pins();
    for (gt, mut wc, shade) in &mut coats {
        let pos = gt.translation();
        let open = if shade.is_some_and(|s| s.indoor) {
            0.0
        } else {
            1.0 - grid.shelter_at(pos + Vec3::Y * PROBE_LIFT)
        };
        if !wc.seeded {
            wc.coat = Coat::seeded(wetness.wetness, zone_snow.0.cover, open);
            wc.seeded = true;
        }
        let speed = if wc.has_last && dt > 1.0e-4 {
            (pos - wc.last_pos).length() / dt
        } else {
            0.0
        };
        wc.last_pos = pos;
        wc.has_last = true;
        let depth = water_index
            .as_deref()
            .and_then(|index| super::shelter::water_surface_y(index, &water, pos.xz()))
            .map_or(f32::NEG_INFINITY, |surface| surface - pos.y);
        wc.coat.step(
            Conditions {
                rain,
                snowfall,
                open,
                submerged: smoothstep(SUBMERGE_FROM, SUBMERGE_FULL, depth),
                speed,
            },
            dt,
        );
        if let Some(w) = pin_wet {
            wc.coat.wet = w;
        }
        if let Some(s) = pin_snow {
            wc.coat.snow = s;
        }
    }
}

/// One entity's links on the way up to its unit.
#[derive(Clone, Copy, Default)]
struct Link {
    parent_model: Option<Entity>,
    child_of: Option<Entity>,
    /// Holds a [`GroundShade`]: it is a unit's net entity.
    owner: bool,
    /// Holds a [`WeatherCoat`]: its packed word.
    word: Option<u32>,
}

/// Walks from a rig up its [`ParentModel`] and [`ChildOf`] links to the first unit net entity or
/// coated entity.
fn find_owner(start: Entity, link: impl Fn(Entity) -> Option<Link>) -> Option<(Entity, Link)> {
    let mut at = start;
    for _ in 0..MAX_WALK {
        let l = link(at)?;
        if l.owner || l.word.is_some() {
            return Some((at, l));
        }
        at = l.parent_model.or(l.child_of)?;
    }
    None
}

type LinkQuery<'w, 's> = Query<
    'w,
    's,
    (
        Option<&'static ChildOf>,
        Option<&'static ParentModel>,
        Has<GroundShade>,
        Option<&'static WeatherCoat>,
    ),
>;

/// Gives every unit that has a rig a coat, and writes each rig slot's word from its unit's coat.
/// Every [`RigSkin`] is rewritten each frame, as the aura tint is, so a recycled slot never keeps
/// a dead unit's word.
fn coat_publish(
    cvar: Res<RainSurfaces>,
    rigs: Query<(Entity, &RigSkin)>,
    links: LinkQuery,
    mut table: ResMut<RigCoats>,
    mut commands: Commands,
) {
    let on = super::wetness::enabled(&cvar);
    for (rig, skin) in &rigs {
        let word = if on {
            let owner = find_owner(rig, |e| {
                let (child_of, parent, owner, coat) = links.get(e).ok()?;
                Some(Link {
                    parent_model: parent.map(|p| p.0),
                    child_of: child_of.map(ChildOf::parent),
                    owner,
                    word: coat.map(|c| c.coat.pack()),
                })
            });
            match owner {
                Some((_, Link { word: Some(w), .. })) => w,
                // A unit with a rig and no coat yet: it gets one now and the word next frame.
                Some((root, _)) => {
                    commands
                        .entity(root)
                        .try_insert_if_new(WeatherCoat::default());
                    0
                }
                None => 0,
            }
        } else {
            0
        };
        table.set(skin.slot, word);
    }
}

/// Render world: writes the whole region to the shared buffer on a change; studio buffers stay
/// zero (their slots are unowned, so they keep the zone-wide look).
fn upload_rig_coats(
    queue: Res<RenderQueue>,
    shared: Option<Res<crate::lighting::SharedLightBuffer>>,
    coats: Option<Res<RigCoats>>,
    mut last: Local<Option<u64>>,
) {
    let (Some(shared), Some(coats)) = (shared, coats) else {
        return;
    };
    if *last == Some(coats.generation) {
        return;
    }
    *last = Some(coats.generation);
    queue.write_buffer(
        &shared.0,
        region_offset(),
        bytemuck::cast_slice(coats.slots.as_slice()),
    );
}

pub(super) fn register(app: &mut App) {
    app.init_resource::<ZoneSnow>()
        .init_resource::<RigCoats>()
        .add_plugins(ExtractResourcePlugin::<RigCoats>::default())
        // After the shelter grid has this frame's roofs and the wetness has this frame's rates.
        .add_systems(
            Update,
            (coat_tick, coat_publish)
                .chain()
                .after(super::wetness::wetness_tick)
                .after(super::shelter::tick_shelter),
        );
    if let Some(render) = app.get_sub_app_mut(RenderApp) {
        render.add_systems(
            Render,
            upload_rig_coats.in_set(RenderSystems::PrepareResources),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn entities(n: usize) -> Vec<Entity> {
        let mut world = World::new();
        (0..n).map(|_| world.spawn_empty().id()).collect()
    }

    #[test]
    fn a_worn_item_finds_its_unit_through_parent_model_then_child_of() {
        let e = entities(4);
        let (helm, body, unit, stray) = (e[0], e[1], e[2], e[3]);
        let mut g = HashMap::new();
        g.insert(
            helm,
            Link {
                parent_model: Some(body),
                ..Default::default()
            },
        );
        g.insert(
            body,
            Link {
                child_of: Some(unit),
                ..Default::default()
            },
        );
        g.insert(
            unit,
            Link {
                owner: true,
                ..Default::default()
            },
        );
        g.insert(stray, Link::default());
        let owner = |from| find_owner(from, |x| g.get(&x).copied()).map(|(e, _)| e);
        assert_eq!(owner(helm), Some(unit));
        assert_eq!(owner(body), Some(unit));
        assert_eq!(owner(unit), Some(unit));
        assert_eq!(
            owner(stray),
            None,
            "an effect with no unit above it has no coat"
        );
    }

    #[test]
    fn a_mount_under_the_unit_shares_its_coat_word() {
        let e = entities(3);
        let (mount, unit, _) = (e[0], e[1], e[2]);
        let mut g = HashMap::new();
        g.insert(
            mount,
            Link {
                child_of: Some(unit),
                ..Default::default()
            },
        );
        let word = Coat {
            wet: 1.0,
            snow: 0.5,
        }
        .pack();
        g.insert(
            unit,
            Link {
                owner: true,
                word: Some(word),
                ..Default::default()
            },
        );
        let (root, link) = find_owner(mount, |x| g.get(&x).copied()).unwrap();
        assert_eq!(root, unit);
        assert_eq!(link.word, Some(word));
    }

    #[test]
    fn a_cycle_terminates() {
        let e = entities(2);
        let (a, b) = (e[0], e[1]);
        let mut g = HashMap::new();
        g.insert(
            a,
            Link {
                parent_model: Some(b),
                ..Default::default()
            },
        );
        g.insert(
            b,
            Link {
                parent_model: Some(a),
                ..Default::default()
            },
        );
        assert!(find_owner(a, |x| g.get(&x).copied()).is_none());
    }

    #[test]
    fn slot_zero_is_never_written_and_generation_tracks_changes_only() {
        let mut t = RigCoats::default();
        t.set(0, Coat::default().pack());
        assert_eq!(t.get(0), 0);
        assert_eq!(t.generation, 0);
        t.set(5, Coat::default().pack());
        assert_eq!(t.generation, 1);
        t.set(5, Coat::default().pack());
        assert_eq!(t.generation, 1, "same word, no upload");
        t.clear(5);
        assert_eq!(t.generation, 2);
        t.clear(5);
        assert_eq!(t.generation, 2);
        t.set(u16::MAX, 7);
        assert_eq!(t.generation, 2, "an out-of-range slot is dropped");
    }

    #[test]
    fn the_region_is_one_word_a_slot_after_the_straddle_table() {
        assert_eq!(region_bytes(), (MAX_RIG_SLOTS * 4) as u64);
        assert_eq!(
            region_offset(),
            crate::straddle::region_offset() + crate::straddle::region_bytes()
        );
    }

    #[test]
    fn smoothstep_edges() {
        assert_eq!(smoothstep(0.5, 1.5, 0.0), 0.0);
        assert_eq!(smoothstep(0.5, 1.5, 2.0), 1.0);
        assert!((smoothstep(0.5, 1.5, 1.0) - 0.5).abs() < 1e-6);
    }
}
