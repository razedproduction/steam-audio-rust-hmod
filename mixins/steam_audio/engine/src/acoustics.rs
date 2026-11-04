//! World acoustics for the Horizon renderer: how each source reaches the listener through the blocks between them
//! (occlusion and transmission), and how reverberant the listener's surroundings are.
//!
//! Both come from rays through the block grid, cast on the game thread (the world lives there):
//!
//! - **Occlusion** follows Steam Audio's volumetric occlusion: rays from the listener to the source and to points
//!   around it; the fraction that gets through unobstructed is the occlusion value. **Transmission** follows Steam
//!   Audio's frequency-dependent transmission: along the central ray, every blocking cell multiplies each band by
//!   its material's transmission coefficient. The direct path's band gains are then Steam Audio's
//!   `occlusion + (1 - occlusion) · transmission` (`DirectEffect`).
//! - **Reverb**: rays in every direction from the listener measure how enclosed the place is, how far the walls
//!   are and what they are made of. Sabine's formula turns that into a reverb time per band for Steam Audio's
//!   parametric [`ReverbEffect`](crate::effects::ReverbEffect); open places get little reverb.
//!
//! A full cube of collision is a wall; everything else (slabs, fences, leaves, fluids) lets sound through. A wall's
//! material is one of Steam Audio's presets, chosen from the block's sound type.
//!
//! The game thread [`publish`]es its results; the audio thread takes them with [`Snapshot::update`], which never
//! waits for the game thread.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::bands::NUM_BANDS;
use crate::scene::Material;

/// What a wall is made of: Steam Audio's material presets (its Unity and Unreal plugins' `Generic`, `Brick`...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Surface {
    Generic,
    Brick,
    Concrete,
    Ceramic,
    Gravel,
    Carpet,
    Glass,
    Plaster,
    Wood,
    Metal,
    Rock,
}

impl Surface {
    pub const ALL: [Surface; 11] = [
        Self::Generic,
        Self::Brick,
        Self::Concrete,
        Self::Ceramic,
        Self::Gravel,
        Self::Carpet,
        Self::Glass,
        Self::Plaster,
        Self::Wood,
        Self::Metal,
        Self::Rock,
    ];

    /// Steam Audio's preset: absorption (low, mid, high), scattering, transmission (low, mid, high).
    pub fn material(self) -> Material {
        let (absorption, transmission) = match self {
            Self::Generic => ([0.10, 0.20, 0.30], [0.100, 0.050, 0.030]),
            Self::Brick => ([0.03, 0.04, 0.07], [0.015, 0.015, 0.015]),
            Self::Concrete => ([0.05, 0.07, 0.08], [0.015, 0.002, 0.001]),
            Self::Ceramic => ([0.01, 0.02, 0.02], [0.060, 0.044, 0.011]),
            Self::Gravel => ([0.60, 0.70, 0.80], [0.031, 0.012, 0.008]),
            Self::Carpet => ([0.24, 0.69, 0.73], [0.020, 0.005, 0.003]),
            Self::Glass => ([0.06, 0.03, 0.02], [0.060, 0.044, 0.011]),
            Self::Plaster => ([0.12, 0.06, 0.04], [0.056, 0.056, 0.004]),
            Self::Wood => ([0.11, 0.07, 0.06], [0.070, 0.014, 0.005]),
            Self::Metal => ([0.20, 0.07, 0.06], [0.200, 0.025, 0.010]),
            Self::Rock => ([0.13, 0.20, 0.24], [0.015, 0.002, 0.001]),
        };
        Material {
            absorption,
            scattering: 0.05,
            transmission,
        }
    }

    /// The surface of a block from its id (`minecraft:white_concrete`) and its break sound
    /// (`minecraft:block.stone.break`), which names its sound type.
    pub fn classify(block: &str, break_sound: &str) -> Self {
        let block = block.rsplit(':').next().unwrap_or(block);
        let sound = break_sound.rsplit(':').next().unwrap_or(break_sound);
        let sound = sound
            .strip_prefix("block.")
            .unwrap_or(sound)
            .trim_end_matches(".break");
        let any = |names: &[&str], text: &str| names.iter().any(|n| text.contains(n));
        if block.contains("concrete") && !block.contains("powder") {
            return Self::Concrete;
        }
        if block.contains("terracotta") {
            return Self::Ceramic;
        }
        if any(&["bricks", "brick_"], block) || any(&["bricks"], sound) {
            return Self::Brick;
        }
        if any(&["glass", "amethyst", "ice"], sound) || any(&["glass", "ice"], block) {
            return Self::Glass;
        }
        if any(
            &[
                "metal",
                "copper",
                "anvil",
                "chain",
                "netherite",
                "lantern",
                "iron",
                "heavy_core",
                "vault",
                "trial_spawner",
            ],
            sound,
        ) {
            return Self::Metal;
        }
        if any(&["wool", "moss", "snow", "sponge", "carpet"], sound) {
            return Self::Carpet;
        }
        if any(
            &[
                "wood",
                "bamboo",
                "scaffolding",
                "ladder",
                "bookshelf",
                "stem",
                "fungus",
            ],
            sound,
        ) {
            return Self::Wood;
        }
        if any(
            &[
                "gravel", "sand", "grass", "dirt", "mud", "soul", "roots", "nylium", "wart",
            ],
            sound,
        ) {
            return Self::Gravel;
        }
        if any(&["calcite", "bone"], sound) {
            return Self::Ceramic;
        }
        if any(
            &[
                "stone",
                "deepslate",
                "tuff",
                "basalt",
                "netherrack",
                "dripstone",
                "ancient_debris",
                "lodestone",
                "polished",
            ],
            sound,
        ) {
            return Self::Rock;
        }
        Self::Generic
    }
}

/// How one source reaches the listener: the fraction of its volume in plain sight and, along the line of sight,
/// what gets through the walls in each band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceAcoustics {
    pub occlusion: f32,
    pub transmission: [f32; NUM_BANDS],
}

impl SourceAcoustics {
    /// Nothing in the way.
    pub const OPEN: Self = Self {
        occlusion: 1.0,
        transmission: [1.0; NUM_BANDS],
    };

    /// Band gains of the direct path (Steam Audio's `DirectEffect` with frequency-dependent transmission).
    pub fn direct_gains(&self) -> [f32; NUM_BANDS] {
        let o = self.occlusion.clamp(0.0, 1.0);
        self.transmission.map(|t| o + (1.0 - o) * t.clamp(0.0, 1.0))
    }

    fn approach(&mut self, target: &Self, rate: f32) {
        self.occlusion += (target.occlusion - self.occlusion) * rate;
        for (t, &goal) in self.transmission.iter_mut().zip(&target.transmission) {
            *t += (goal - *t) * rate;
        }
    }
}

/// The listener's surroundings as a parametric reverb.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentReverb {
    /// Time for the energy to decay by 60 dB, per band, in seconds.
    pub reverb_times: [f32; NUM_BANDS],
    /// Level of the reverb relative to what sources send it (0: dry).
    pub wet: f32,
}

impl EnvironmentReverb {
    /// No reverb.
    pub const DRY: Self = Self {
        reverb_times: [MIN_REVERB_TIME; NUM_BANDS],
        wet: 0.0,
    };

    fn approach(&mut self, target: &Self, rate: f32) {
        self.wet += (target.wet - self.wet) * rate;
        for (t, &goal) in self.reverb_times.iter_mut().zip(&target.reverb_times) {
            *t += (goal - *t) * rate;
        }
    }
}

/// Steam Audio's shortest reverb time.
pub const MIN_REVERB_TIME: f32 = 0.1;
/// The longest reverb time the probe reports.
pub const MAX_REVERB_TIME: f32 = 8.0;

/// Tuning of the world acoustics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcousticsSettings {
    /// Exponent applied to each wall's transmission coefficients: 1 is Steam Audio's physical values (a stone wall
    /// lets through about −36 dB), lower is gentler. Minecraft is played by ear, so the default keeps walls audible.
    pub transmission_exponent: f32,
    /// Most walls counted along a ray.
    pub max_walls: u32,
    /// Distance from a source of the points the occlusion rays aim at besides its centre, in blocks.
    pub source_radius: f64,
    /// Rays the reverb probe casts, and how far.
    pub reverb_rays: usize,
    pub reverb_range: f64,
    /// Reverb level in a fully enclosed place.
    pub reverb_level: f32,
}

impl Default for AcousticsSettings {
    fn default() -> Self {
        Self {
            transmission_exponent: 0.35,
            max_walls: 8,
            source_radius: 0.6,
            reverb_rays: 48,
            reverb_range: 40.0,
            reverb_level: 0.5,
        }
    }
}

/// Visits the cells a segment crosses from `from` to `to` (Amanatides and Woo), with the distance from `from` at
/// which it enters each, leaving out the cells of both ends. Stops when `visit` returns false.
pub fn cells_between(from: [f64; 3], to: [f64; 3], mut visit: impl FnMut([i32; 3], f64) -> bool) {
    let start = from.map(|v| v.floor() as i32);
    let end = to.map(|v| v.floor() as i32);
    let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if start == end || length <= 0.0 {
        return;
    }
    let mut cell = start;
    let mut step = [0i32; 3];
    let mut next = [f64::INFINITY; 3];
    let mut delta = [f64::INFINITY; 3];
    for a in 0..3 {
        let dir = d[a] / length;
        if dir > 0.0 {
            step[a] = 1;
            delta[a] = 1.0 / dir;
            next[a] = (f64::from(cell[a]) + 1.0 - from[a]) / dir;
        } else if dir < 0.0 {
            step[a] = -1;
            delta[a] = -1.0 / dir;
            next[a] = (f64::from(cell[a]) - from[a]) / dir;
        }
    }
    let budget = (0..3)
        .map(|a| (end[a] - start[a]).unsigned_abs())
        .sum::<u32>();
    for _ in 0..budget {
        let a = if next[0] <= next[1] && next[0] <= next[2] {
            0
        } else if next[1] <= next[2] {
            1
        } else {
            2
        };
        let t = next[a];
        if t > length {
            return;
        }
        cell[a] += step[a];
        next[a] += delta[a];
        if cell == end || !visit(cell, t) {
            return;
        }
    }
}

/// How `source` reaches `listener` through the walls `wall` reports (`None`: no wall in that cell).
pub fn source_acoustics(
    wall: &mut impl FnMut([i32; 3]) -> Option<Surface>,
    listener: [f64; 3],
    source: [f64; 3],
    settings: &AcousticsSettings,
) -> SourceAcoustics {
    let r = settings.source_radius;
    // The centre, around it and above it: points below a source on the ground would be in the ground.
    let offsets = [
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [-r, 0.0, 0.0],
        [0.0, 0.0, r],
        [0.0, 0.0, -r],
        [0.0, r, 0.0],
    ];
    let source_cell = source.map(|v| v.floor() as i32);
    let (mut seen, mut aimed) = (0u32, 0u32);
    for (i, o) in offsets.iter().enumerate() {
        let point = [source[0] + o[0], source[1] + o[1], source[2] + o[2]];
        let cell = point.map(|v| v.floor() as i32);
        // A point inside a wall other than the source's own block is not part of the source.
        if i > 0 && cell != source_cell && wall(cell).is_some() {
            continue;
        }
        aimed += 1;
        let mut clear = true;
        cells_between(listener, point, |c, _| {
            if c != source_cell && wall(c).is_some() {
                clear = false;
            }
            clear
        });
        seen += u32::from(clear);
    }
    let occlusion = if aimed == 0 {
        0.0
    } else {
        seen as f32 / aimed as f32
    };
    let mut transmission = [1.0f32; NUM_BANDS];
    if occlusion < 1.0 {
        let mut walls = 0;
        cells_between(listener, source, |c, _| {
            if c != source_cell
                && let Some(surface) = wall(c)
            {
                let m = surface.material();
                for (t, &coefficient) in transmission.iter_mut().zip(&m.transmission) {
                    *t *= coefficient.max(1e-4).powf(settings.transmission_exponent);
                }
                walls += 1;
            }
            walls < settings.max_walls
        });
    }
    SourceAcoustics {
        occlusion,
        transmission,
    }
}

/// The reverb of the place around `listener`.
pub fn environment_reverb(
    wall: &mut impl FnMut([i32; 3]) -> Option<Surface>,
    listener: [f64; 3],
    settings: &AcousticsSettings,
) -> EnvironmentReverb {
    let rays = settings.reverb_rays.max(1);
    let (mut hits, mut distance) = (0usize, 0.0f64);
    // Rays going up, and how many of them meet a ceiling: under the open sky there is little reverb.
    let (mut up, mut up_hits) = (0usize, 0usize);
    let mut absorption = [0.0f32; NUM_BANDS];
    // Directions spread evenly over the sphere (a Fibonacci lattice).
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    for i in 0..rays {
        let y = 1.0 - 2.0 * (i as f64 + 0.5) / rays as f64;
        let radius = (1.0 - y * y).sqrt();
        let theta = golden * i as f64;
        let dir = [radius * theta.cos(), y, radius * theta.sin()];
        let to = [
            listener[0] + dir[0] * settings.reverb_range,
            listener[1] + dir[1] * settings.reverb_range,
            listener[2] + dir[2] * settings.reverb_range,
        ];
        let upward = y > 0.0;
        up += usize::from(upward);
        cells_between(listener, to, |c, t| match wall(c) {
            Some(surface) => {
                hits += 1;
                up_hits += usize::from(upward);
                distance += t.max(0.5);
                for (a, &m) in absorption.iter_mut().zip(&surface.material().absorption) {
                    *a += m;
                }
                false
            }
            None => true,
        });
    }
    if hits == 0 {
        return EnvironmentReverb::DRY;
    }
    let enclosed = hits as f32 / rays as f32;
    let mean_distance = (distance / hits as f64) as f32;
    // Sabine: T = 0.161 V / (S α). Seen from inside, V / S of a room is about a third of the distance to its walls
    // (a sphere's); what escapes to the open counts as fully absorbed.
    let reverb_times = absorption.map(|a| {
        let alpha = enclosed * (a / hits as f32) + (1.0 - enclosed);
        (0.161 * mean_distance / 3.0 / alpha.max(0.01)).clamp(MIN_REVERB_TIME, MAX_REVERB_TIME)
    });
    let covered = if up == 0 {
        1.0
    } else {
        up_hits as f32 / up as f32
    };
    EnvironmentReverb {
        reverb_times,
        wet: settings.reverb_level * enclosed * covered,
    }
}

#[derive(Debug)]
struct Shared {
    sources: HashMap<u64, SourceAcoustics>,
    reverb: EnvironmentReverb,
    /// Bumped on each publish.
    version: u64,
}

static SHARED: LazyLock<Mutex<Shared>> = LazyLock::new(|| {
    Mutex::new(Shared {
        sources: HashMap::new(),
        reverb: EnvironmentReverb::DRY,
        version: 0,
    })
});

/// How much of the way to a new value one publish moves: changes settle over a few updates instead of jumping
/// when the camera passes a corner.
const SMOOTHING: f32 = 0.5;

/// Publishes the game thread's latest results: the sources' (by mixer channel id; others are forgotten) and the
/// listener's reverb (`None`: keep the last).
pub fn publish(sources: &[(u64, SourceAcoustics)], reverb: Option<EnvironmentReverb>) {
    let Ok(mut shared) = SHARED.lock() else {
        return;
    };
    let previous = std::mem::take(&mut shared.sources);
    for (id, target) in sources {
        let mut value = previous.get(id).copied().unwrap_or(*target);
        value.approach(target, SMOOTHING);
        shared.sources.insert(*id, value);
    }
    if let Some(target) = reverb {
        let mut value = shared.reverb;
        value.approach(&target, SMOOTHING);
        shared.reverb = value;
    }
    shared.version += 1;
}

/// Forgets everything published (no world).
pub fn clear() {
    if let Ok(mut shared) = SHARED.lock() {
        shared.sources.clear();
        shared.reverb = EnvironmentReverb::DRY;
        shared.version += 1;
    }
}

/// The audio thread's copy of the published acoustics.
#[derive(Debug)]
pub struct Snapshot {
    pub sources: HashMap<u64, SourceAcoustics>,
    pub reverb: EnvironmentReverb,
    version: u64,
    /// Set by [`Snapshot::fixed`]: ignores what is published.
    fixed: bool,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            sources: HashMap::new(),
            reverb: EnvironmentReverb::DRY,
            version: u64::MAX,
            fixed: false,
        }
    }
}

impl Snapshot {
    /// Acoustics that stay as given (tests, tools).
    pub fn fixed(sources: HashMap<u64, SourceAcoustics>, reverb: EnvironmentReverb) -> Self {
        Self {
            sources,
            reverb,
            version: u64::MAX,
            fixed: true,
        }
    }

    /// Copies what was published since the last update; never waits (a busy lock keeps the last copy).
    pub fn update(&mut self) {
        if self.fixed {
            return;
        }
        let Ok(shared) = SHARED.try_lock() else {
            return;
        };
        if shared.version == self.version {
            return;
        }
        self.version = shared.version;
        self.sources.clone_from(&shared.sources);
        self.reverb = shared.reverb;
    }

    /// The direct path's band gains for source `id` (open when unknown).
    pub fn direct_gains(&self, id: u64) -> [f32; NUM_BANDS] {
        self.sources
            .get(&id)
            .map_or([1.0; NUM_BANDS], SourceAcoustics::direct_gains)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A world of `surface` cells for which `solid` holds.
    fn world(solid: impl Fn([i32; 3]) -> bool) -> impl FnMut([i32; 3]) -> Option<Surface> {
        move |c| solid(c).then_some(Surface::Rock)
    }

    #[test]
    fn the_walk_visits_the_cells_between_the_ends() {
        let mut cells = Vec::new();
        cells_between([0.5, 0.5, 0.5], [4.5, 0.5, 0.5], |c, t| {
            cells.push((c, t));
            true
        });
        assert_eq!(
            cells,
            vec![([1, 0, 0], 0.5), ([2, 0, 0], 1.5), ([3, 0, 0], 2.5)]
        );
        let mut diagonal = 0;
        cells_between([0.5, 0.5, 0.5], [-2.5, 3.5, 0.5], |_, _| {
            diagonal += 1;
            true
        });
        assert_eq!(diagonal, 5);
        let mut none = 0;
        cells_between([0.2, 0.2, 0.2], [0.8, 0.9, 0.1], |_, _| {
            none += 1;
            true
        });
        assert_eq!(none, 0);
    }

    #[test]
    fn a_wall_occludes_and_muffles_more_at_high_frequencies() {
        let settings = AcousticsSettings::default();
        let listener = [0.5, 64.6, 0.5];
        let source = [6.5, 64.5, 0.5];
        let open = source_acoustics(&mut world(|_| false), listener, source, &settings);
        assert_eq!(open, SourceAcoustics::OPEN);
        // A wall at x = 3, three blocks high and wide.
        let mut wall = world(|c| c[0] == 3 && (63..=66).contains(&c[1]) && c[2].abs() <= 2);
        let walled = source_acoustics(&mut wall, listener, source, &settings);
        assert_eq!(walled.occlusion, 0.0);
        let gains = walled.direct_gains();
        assert!(gains[0] > gains[1] && gains[1] > gains[2], "{gains:?}");
        assert!(gains[0] < 0.5, "{gains:?}");
        // A source's own block (a note block, a furnace) does not occlude it.
        let own = source_acoustics(&mut world(|c| c == [6, 64, 0]), listener, source, &settings);
        assert_eq!(own.occlusion, 1.0);
    }

    #[test]
    fn a_corner_occludes_partly() {
        let settings = AcousticsSettings::default();
        // A wall whose edge is at the line of sight: some rays pass beside it.
        let mut edge = world(|c| c[0] == 3 && c[2] >= 1);
        let partial = source_acoustics(&mut edge, [0.5, 64.5, 0.5], [6.5, 64.5, 0.9], &settings);
        assert!(
            partial.occlusion > 0.0 && partial.occlusion < 1.0,
            "{partial:?}"
        );
    }

    #[test]
    fn caves_reverberate_and_fields_stay_dry() {
        let settings = AcousticsSettings::default();
        let field = environment_reverb(&mut world(|c| c[1] < 64), [0.5, 65.6, 0.5], &settings);
        // A 9 x 5 x 9 stone room around the listener.
        let mut cave = world(|c| c[0].abs() > 4 || c[2].abs() > 4 || c[1] < 62 || c[1] > 66);
        let room = environment_reverb(&mut cave, [0.5, 63.6, 0.5], &settings);
        let mut hall = world(|c| c[0].abs() > 16 || c[2].abs() > 16 || c[1] < 50 || c[1] > 80);
        let large = environment_reverb(&mut hall, [0.5, 63.6, 0.5], &settings);
        assert!(field.wet < 0.2 * room.wet, "{field:?} {room:?}");
        assert!(room.wet > 0.4, "{room:?}");
        assert!(
            room.reverb_times[1] > 0.3 && room.reverb_times[1] < 2.0,
            "{room:?}"
        );
        assert!(
            large.reverb_times[1] > 2.0 * room.reverb_times[1],
            "{large:?}"
        );
        assert!(field.reverb_times[1] < room.reverb_times[1]);
        assert_eq!(
            environment_reverb(&mut world(|_| false), [0.5, 64.0, 0.5], &settings),
            EnvironmentReverb::DRY
        );
    }

    #[test]
    fn blocks_get_their_materials() {
        let cases = [
            (
                "minecraft:stone",
                "minecraft:block.stone.break",
                Surface::Rock,
            ),
            (
                "minecraft:deepslate",
                "minecraft:block.deepslate.break",
                Surface::Rock,
            ),
            (
                "minecraft:oak_planks",
                "minecraft:block.wood.break",
                Surface::Wood,
            ),
            (
                "minecraft:white_wool",
                "minecraft:block.wool.break",
                Surface::Carpet,
            ),
            (
                "minecraft:glass",
                "minecraft:block.glass.break",
                Surface::Glass,
            ),
            (
                "minecraft:iron_block",
                "minecraft:block.metal.break",
                Surface::Metal,
            ),
            (
                "minecraft:bricks",
                "minecraft:block.stone.break",
                Surface::Brick,
            ),
            (
                "minecraft:mud_bricks",
                "minecraft:block.mud_bricks.break",
                Surface::Brick,
            ),
            (
                "minecraft:dirt",
                "minecraft:block.gravel.break",
                Surface::Gravel,
            ),
            (
                "minecraft:white_concrete",
                "minecraft:block.stone.break",
                Surface::Concrete,
            ),
            (
                "minecraft:white_concrete_powder",
                "minecraft:block.sand.break",
                Surface::Gravel,
            ),
            (
                "minecraft:terracotta",
                "minecraft:block.stone.break",
                Surface::Ceramic,
            ),
            (
                "minecraft:slime_block",
                "minecraft:block.slime_block.break",
                Surface::Generic,
            ),
        ];
        for (block, sound, surface) in cases {
            assert_eq!(Surface::classify(block, sound), surface, "{block}");
        }
    }

    #[test]
    fn published_values_settle_smoothly() {
        let walled = SourceAcoustics {
            occlusion: 0.0,
            transmission: [0.2, 0.1, 0.05],
        };
        publish(&[(42, SourceAcoustics::OPEN)], None);
        publish(&[(42, walled)], None);
        let mut snapshot = Snapshot::default();
        snapshot.update();
        let halfway = snapshot.sources[&42];
        assert!(
            halfway.occlusion > 0.4 && halfway.occlusion < 0.6,
            "{halfway:?}"
        );
        assert_eq!(snapshot.direct_gains(7), [1.0; NUM_BANDS]);
        publish(&[], None);
        snapshot.update();
        assert!(snapshot.sources.is_empty());
    }
}
