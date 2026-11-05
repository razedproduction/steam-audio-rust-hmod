//! World acoustics for the Horizon renderer: how each source reaches the listener through the blocks between them
//! (occlusion and transmission). The way round a wall is found in [`crate::paths`], the reverb of the surroundings
//! is simulated in [`crate::reflections`].
//!
//! Rays through the block grid, cast on the game thread (the world lives there): **occlusion** follows Steam Audio's
//! volumetric occlusion: rays from the listener to the source and to points around it; the fraction that gets
//! through unobstructed is the occlusion value. **Transmission** follows Steam Audio's frequency-dependent
//! transmission: along the central ray, every blocking cell multiplies each band by its material's transmission
//! coefficient. The direct path's band gains are then Steam Audio's `occlusion + (1 - occlusion) · transmission`
//! (`DirectEffect`).
//!
//! A full cube of collision is a wall, and so are the blocks [`Surface::classify_partial`] counts (liquids, slabs,
//! stairs, doors...); leaves and fences only filter what goes through them. A wall that is not a full cube has the
//! shape of its block state ([`Shape`]: a bottom slab, a closed or an open door), and rays pass beside it. A wall's
//! material is one of Steam Audio's presets, chosen from the block's sound type.
//!
//! The game thread [`publish`]es its results; the audio thread takes them with [`Snapshot::update`], which never
//! waits for the game thread.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::bands::NUM_BANDS;
use crate::paths::PathAcoustics;
use crate::scene::Material;

/// What a wall is made of: Steam Audio's material presets (its Unity and Unreal plugins' `Generic`, `Brick`...),
/// and three of the mod's own for what is not a full block of something.
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
    /// Water: its surface reflects almost everything, and little crosses it.
    Water,
    /// Leaves: sound goes through, losing its highs on the way; nothing to reflect off.
    Foliage,
    /// Fences, panes, bars, chains, walls: thin, sound goes through and round them almost whole.
    Thin,
}

impl Surface {
    pub const ALL: [Surface; 14] = [
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
        Self::Water,
        Self::Foliage,
        Self::Thin,
    ];

    /// Whether it is a wall: it blocks rays (occlusion, reflections, the probes' sight). Foliage and thin blocks
    /// are not: they only filter the sound that goes through them.
    pub fn is_solid(self) -> bool {
        !matches!(self, Self::Foliage | Self::Thin)
    }

    /// Steam Audio's preset: absorption (low, mid, high), scattering, transmission (low, mid, high). For foliage
    /// and thin blocks, transmission is what one block lets through.
    pub fn material(self) -> Material {
        let (absorption, transmission) = match self {
            Self::Water => ([0.01, 0.01, 0.02], [0.030, 0.030, 0.030]),
            Self::Foliage => ([0.20, 0.40, 0.60], [0.90, 0.75, 0.55]),
            Self::Thin => ([0.10, 0.10, 0.10], [0.97, 0.94, 0.90]),
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

    /// The surface of a block without full collision, if it counts acoustically: liquids (water is [`Water`], lava
    /// [`Rock`](Self::Rock)), slabs, stairs, doors, trapdoors and other blocks that are mostly solid (as
    /// [`classify`](Self::classify) says), leaves ([`Foliage`]) and fences, panes, bars, chains and walls ([`Thin`]).
    /// Anything else (flowers, torches, carpets, rails...) lets sound through untouched.
    ///
    /// [`Water`]: Self::Water
    /// [`Foliage`]: Self::Foliage
    /// [`Thin`]: Self::Thin
    pub fn classify_partial(block: &str, break_sound: &str) -> Option<Self> {
        let id = block.rsplit(':').next().unwrap_or(block);
        let any = |names: &[&str]| names.iter().any(|n| id.contains(n));
        if id == "water" || id == "bubble_column" {
            return Some(Self::Water);
        }
        if id == "lava" {
            return Some(Self::Rock);
        }
        if any(&["leaves"]) {
            return Some(Self::Foliage);
        }
        // Before the solid ones: `*_wall` but not `*_wall_sign`, `*_wall_torch`, `*_wall_banner`...
        if any(&["fence", "pane", "iron_bars", "chain"]) || id.ends_with("_wall") {
            return Some(Self::Thin);
        }
        // Doors and trapdoors are walls of their material where they are: their shape says whether they are open.
        if any(&["door"]) && !any(&["sign"]) {
            return Some(Self::classify(block, break_sound));
        }
        if any(&[
            "slab",
            "stairs",
            "chest",
            "barrel",
            "enchanting_table",
            "end_portal_frame",
            "daylight_detector",
            "stonecutter",
            "lectern",
            "composter",
            "cauldron",
            "anvil",
            "bed",
            "farmland",
            "dirt_path",
            "soul_sand",
            "mud",
            "honey_block",
            "cactus",
            "sculk_shrieker",
            "sculk_sensor",
            "bell",
            "grindstone",
        ]) && !any(&["bedrock", "sign"])
        {
            return Some(Self::classify(block, break_sound));
        }
        None
    }
}

/// The shape of a block state that is not a full cube: its boxes in block space (`[min x, y, z, max x, y, z]`, 0 to
/// 1), as the game's outline has them. A bottom slab is `[0, 0, 0, 1, 0.5, 1]`; a door, closed or open, is a panel
/// 3/16 thick on the side its state says.
pub type Shape = std::sync::Arc<[[f64; 6]]>;

/// Whether the segment `from` → `to` passes through one of the boxes of `shape` in `cell` (slab test; a point,
/// `from == to`, whether it is inside one).
pub fn segment_hits_shape(
    from: [f64; 3],
    to: [f64; 3],
    cell: [i32; 3],
    shape: &[[f64; 6]],
) -> bool {
    let d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    shape.iter().any(|b| {
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for a in 0..3 {
            let lo = f64::from(cell[a]) + b[a];
            let hi = f64::from(cell[a]) + b[a + 3];
            if d[a].abs() < 1e-12 {
                if from[a] < lo || from[a] > hi {
                    return false;
                }
                continue;
            }
            let (mut ta, mut tb) = ((lo - from[a]) / d[a], (hi - from[a]) / d[a]);
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return false;
            }
        }
        true
    })
}

/// The axes (bit 0 x, 1 y, 2 z) along which `shape` closes its cell: seen along the axis, its boxes cover the whole
/// face. A bottom slab closes y; a door closes the axis across its panel; a fence post closes none.
pub fn shape_seals(shape: &[[f64; 6]]) -> u8 {
    // Block shapes are made of sixteenths: a 16 × 16 raster of each face is exact.
    const N: usize = 16;
    let mut seals = 0;
    for axis in 0..3 {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut covered = [[false; N]; N];
        for b in shape {
            if b[axis + 3] - b[axis] <= 1e-9 {
                continue;
            }
            for (i, row) in covered.iter_mut().enumerate() {
                let cu = (i as f64 + 0.5) / N as f64;
                if cu < b[u] || cu > b[u + 3] {
                    continue;
                }
                for (j, cell) in row.iter_mut().enumerate() {
                    let cv = (j as f64 + 0.5) / N as f64;
                    if cv >= b[v] && cv <= b[v + 3] {
                        *cell = true;
                    }
                }
            }
        }
        if covered.iter().all(|row| row.iter().all(|&c| c)) {
            seals |= 1 << axis;
        }
    }
    seals
}

/// Whether a block of `shape` that closes its cell along an axis ([`shape_seals`]) stands in the way along it:
/// `full(offset)` says whether the neighbour at that offset is a full wall. A closed door between two open cells
/// does; an open one, whose panel lies along the wall beside it, does not, nor does a slab on the ground. The
/// reflection and pathing grids, which have whole cells only, count such blocks as walls and no others.
pub fn shape_blocks_cell(seals: u8, mut full: impl FnMut([i32; 3]) -> bool) -> bool {
    (0..3).any(|axis| {
        if seals & (1 << axis) == 0 {
            return false;
        }
        let mut step = [0; 3];
        step[axis] = 1;
        let back = step.map(|v| -v);
        !full(step) && !full(back)
    })
}

/// How one source reaches the listener: the fraction of its volume in plain sight and, along the line of sight,
/// what gets through the walls in each band; and, when it is hidden, the way round ([`crate::paths`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceAcoustics {
    pub occlusion: f32,
    pub transmission: [f32; NUM_BANDS],
    /// What the leaves, fences and panes along the line of sight let through, in each band (they do not occlude).
    pub through: [f32; NUM_BANDS],
    /// The sound paths around what hides the source, heard for the part of it that is hidden (`1 − occlusion`).
    pub path: Option<PathAcoustics>,
}

impl SourceAcoustics {
    /// Nothing in the way.
    pub const OPEN: Self = Self {
        occlusion: 1.0,
        transmission: [1.0; NUM_BANDS],
        through: [1.0; NUM_BANDS],
        path: None,
    };

    /// The gain of the way round, relative to the direct sound with nothing in the way: the paths' gain for the
    /// hidden part of the source (0 without paths).
    pub fn path_gain(&self) -> f32 {
        self.path.map_or(0.0, |p| {
            (1.0 - self.occlusion.clamp(0.0, 1.0)) * p.gain.clamp(0.0, 1.0)
        })
    }

    /// Band gains of the direct path (Steam Audio's `DirectEffect` with frequency-dependent transmission), through
    /// the leaves and fences in the way.
    pub fn direct_gains(&self) -> [f32; NUM_BANDS] {
        let o = self.occlusion.clamp(0.0, 1.0);
        std::array::from_fn(|b| {
            (o + (1.0 - o) * self.transmission[b].clamp(0.0, 1.0)) * self.through[b].clamp(0.0, 1.0)
        })
    }

    fn approach(&mut self, target: &Self, rate: f32) {
        self.occlusion += (target.occlusion - self.occlusion) * rate;
        for (t, &goal) in self.transmission.iter_mut().zip(&target.transmission) {
            *t += (goal - *t) * rate;
        }
        for (t, &goal) in self.through.iter_mut().zip(&target.through) {
            *t += (goal - *t) * rate;
        }
        // A way round that appears fades in from where it is, one that goes fades out where it was.
        self.path = match (self.path, target.path) {
            (Some(mut current), Some(goal)) => {
                current.approach(&goal, rate);
                Some(current)
            }
            (Some(mut current), None) => {
                current.gain *= 1.0 - rate;
                (current.gain > 1e-3).then_some(current)
            }
            (None, Some(goal)) => Some(PathAcoustics {
                gain: goal.gain * rate,
                ..goal
            }),
            (None, None) => None,
        };
    }
}

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
}

impl Default for AcousticsSettings {
    fn default() -> Self {
        Self {
            transmission_exponent: 0.35,
            max_walls: 8,
            source_radius: 0.6,
        }
    }
}

/// Most leaves and fences counted along a ray (a forest muffles, it does not silence).
const MAX_FILTERS: u32 = 12;

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

/// How `source` reaches `listener` through the walls `wall` reports (`None`: no wall in that cell), all full cubes.
pub fn source_acoustics(
    wall: &mut impl FnMut([i32; 3]) -> Option<Surface>,
    listener: [f64; 3],
    source: [f64; 3],
    settings: &AcousticsSettings,
) -> SourceAcoustics {
    source_acoustics_shaped(
        &mut |c| wall(c).map(|s| (s, None)),
        listener,
        source,
        settings,
    )
}

/// How `source` reaches `listener` through the walls `wall` reports: each with its surface and, when it is not a
/// full cube, its [`Shape`], which a ray must go through to be stopped or filtered by it.
pub fn source_acoustics_shaped(
    wall: &mut impl FnMut([i32; 3]) -> Option<(Surface, Option<Shape>)>,
    listener: [f64; 3],
    source: [f64; 3],
    settings: &AcousticsSettings,
) -> SourceAcoustics {
    // What of the cell `c` the segment `from` → `to` meets.
    let mut wall = |c: [i32; 3], from: [f64; 3], to: [f64; 3]| -> Option<Surface> {
        let (surface, shape) = wall(c)?;
        match shape {
            Some(shape) if !segment_hits_shape(from, to, c, &shape) => None,
            _ => Some(surface),
        }
    };
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
        if i > 0 && cell != source_cell && wall(cell, point, point).is_some_and(Surface::is_solid) {
            continue;
        }
        aimed += 1;
        let mut clear = true;
        cells_between(listener, point, |c, _| {
            if c != source_cell && wall(c, listener, point).is_some_and(Surface::is_solid) {
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
    // Along the line of sight: the walls' transmission (for the occluded part), and what the leaves and fences let
    // through (for all of it).
    let mut transmission = [1.0f32; NUM_BANDS];
    let mut through = [1.0f32; NUM_BANDS];
    let (mut walls, mut filters) = (0, 0);
    cells_between(listener, source, |c, _| {
        if c == source_cell {
            return true;
        }
        match wall(c, listener, source) {
            Some(surface)
                if surface.is_solid() && occlusion < 1.0 && walls < settings.max_walls =>
            {
                let m = surface.material();
                for (t, &coefficient) in transmission.iter_mut().zip(&m.transmission) {
                    *t *= coefficient.max(1e-4).powf(settings.transmission_exponent);
                }
                walls += 1;
            }
            Some(surface) if !surface.is_solid() && filters < MAX_FILTERS => {
                let m = surface.material();
                for (t, &coefficient) in through.iter_mut().zip(&m.transmission) {
                    *t *= coefficient.max(1e-4).powf(settings.transmission_exponent);
                }
                filters += 1;
            }
            _ => {}
        }
        walls < settings.max_walls || filters < MAX_FILTERS
    });
    SourceAcoustics {
        occlusion,
        transmission,
        through,
        path: None,
    }
}

#[derive(Debug)]
struct Shared {
    sources: HashMap<u64, SourceAcoustics>,
    underwater: bool,
    /// Bumped on each publish.
    version: u64,
}

static SHARED: LazyLock<Mutex<Shared>> = LazyLock::new(|| {
    Mutex::new(Shared {
        sources: HashMap::new(),
        underwater: false,
        version: 0,
    })
});

/// What one band of the sound loses when the listener is under water: water carries the lows, and the ears, full of
/// it, hear little above them (Steam Audio's bands: up to 800 Hz, to 8 kHz, above).
pub const UNDERWATER_GAINS: [f32; NUM_BANDS] = [0.9, 0.3, 0.06];

/// Publishes whether the listener is under water.
pub fn publish_underwater(underwater: bool) {
    if let Ok(mut shared) = SHARED.lock()
        && shared.underwater != underwater
    {
        shared.underwater = underwater;
        shared.version += 1;
    }
}

/// How much of the way to a new value one publish moves: changes settle over a few updates instead of jumping
/// when the camera passes a corner.
const SMOOTHING: f32 = 0.5;

/// Publishes the game thread's latest results: the sources' (by mixer channel id; others are forgotten).
pub fn publish(sources: &[(u64, SourceAcoustics)]) {
    let Ok(mut shared) = SHARED.lock() else {
        return;
    };
    let previous = std::mem::take(&mut shared.sources);
    for (id, target) in sources {
        let mut value = previous.get(id).copied().unwrap_or(*target);
        value.approach(target, SMOOTHING);
        shared.sources.insert(*id, value);
    }
    shared.version += 1;
}

/// Forgets everything published (no world).
pub fn clear() {
    if let Ok(mut shared) = SHARED.lock() {
        shared.sources.clear();
        shared.underwater = false;
        shared.version += 1;
    }
}

/// The audio thread's copy of the published acoustics.
#[derive(Debug)]
pub struct Snapshot {
    pub sources: HashMap<u64, SourceAcoustics>,
    /// Whether the listener is under water.
    pub underwater: bool,
    version: u64,
    /// Set by [`Snapshot::fixed`]: ignores what is published.
    fixed: bool,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            sources: HashMap::new(),
            underwater: false,
            version: u64::MAX,
            fixed: false,
        }
    }
}

impl Snapshot {
    /// Acoustics that stay as given (tests, tools).
    pub fn fixed(sources: HashMap<u64, SourceAcoustics>) -> Self {
        Self {
            sources,
            underwater: false,
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
        self.underwater = shared.underwater;
    }

    /// The direct path's band gains for source `id` (open when unknown).
    pub fn direct_gains(&self, id: u64) -> [f32; NUM_BANDS] {
        self.sources
            .get(&id)
            .map_or([1.0; NUM_BANDS], SourceAcoustics::direct_gains)
    }

    /// The acoustics of source `id`, if known.
    pub fn get(&self, id: u64) -> Option<&SourceAcoustics> {
        self.sources.get(&id)
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
    fn partial_blocks_and_liquids_count_as_what_they_are() {
        let cases = [
            ("minecraft:water", "", Some(Surface::Water)),
            ("minecraft:lava", "", Some(Surface::Rock)),
            (
                "minecraft:oak_leaves",
                "minecraft:block.grass.break",
                Some(Surface::Foliage),
            ),
            (
                "minecraft:oak_fence",
                "minecraft:block.wood.break",
                Some(Surface::Thin),
            ),
            (
                "minecraft:glass_pane",
                "minecraft:block.glass.break",
                Some(Surface::Thin),
            ),
            (
                "minecraft:cobblestone_wall",
                "minecraft:block.stone.break",
                Some(Surface::Thin),
            ),
            (
                "minecraft:stone_slab",
                "minecraft:block.stone.break",
                Some(Surface::Rock),
            ),
            (
                "minecraft:oak_stairs",
                "minecraft:block.wood.break",
                Some(Surface::Wood),
            ),
            (
                "minecraft:chest",
                "minecraft:block.wood.break",
                Some(Surface::Wood),
            ),
            (
                "minecraft:oak_wall_sign",
                "minecraft:block.wood.break",
                None,
            ),
            (
                "minecraft:oak_door",
                "minecraft:block.wooden_door.break",
                Some(Surface::Wood),
            ),
            (
                "minecraft:iron_trapdoor",
                "minecraft:block.metal.break",
                Some(Surface::Metal),
            ),
            ("minecraft:wall_torch", "minecraft:block.wood.break", None),
            ("minecraft:poppy", "minecraft:block.grass.break", None),
            ("minecraft:white_carpet", "minecraft:block.wool.break", None),
        ];
        for (block, sound, surface) in cases {
            assert_eq!(Surface::classify_partial(block, sound), surface, "{block}");
        }
        assert!(
            Surface::Water.is_solid() && !Surface::Foliage.is_solid() && !Surface::Thin.is_solid()
        );
    }

    #[test]
    fn leaves_muffle_without_occluding() {
        let settings = AcousticsSettings::default();
        let listener = [0.5, 64.6, 0.5];
        let source = [8.5, 64.5, 0.5];
        // A hedge three blocks thick at x = 3..5.
        let mut hedge = |c: [i32; 3]| ((3..=5).contains(&c[0])).then_some(Surface::Foliage);
        let a = source_acoustics(&mut hedge, listener, source, &settings);
        assert_eq!(a.occlusion, 1.0);
        assert_eq!(a.transmission, [1.0; NUM_BANDS]);
        let gains = a.direct_gains();
        assert!(gains[0] > gains[2] && gains[0] < 1.0, "{gains:?}");
        assert!(
            gains[2] > 0.3,
            "a hedge muffles, it does not silence: {gains:?}"
        );
    }
    // Door panels 3/16 thick: across x (closed in a doorway along x) and along it (open, against the side).
    const PANEL_X: [[f64; 6]; 1] = [[0.0, 0.0, 0.0, 0.1875, 1.0, 1.0]];
    const PANEL_Z: [[f64; 6]; 1] = [[0.0, 0.0, 0.0, 1.0, 1.0, 0.1875]];
    const BOTTOM_SLAB: [[f64; 6]; 1] = [[0.0, 0.0, 0.0, 1.0, 0.5, 1.0]];

    #[test]
    fn shapes_are_hit_where_they_are() {
        let cell = [3, 64, 0];
        // Along x at mid height: a bottom slab is passed over, a panel across the way is not.
        let (from, to) = ([0.5, 64.75, 0.5], [6.5, 64.75, 0.5]);
        assert!(!segment_hits_shape(from, to, cell, &BOTTOM_SLAB));
        assert!(segment_hits_shape(
            [0.5, 64.25, 0.5],
            [6.5, 64.25, 0.5],
            cell,
            &BOTTOM_SLAB
        ));
        assert!(segment_hits_shape(from, to, cell, &PANEL_X));
        assert!(!segment_hits_shape(
            [0.5, 64.5, 0.6],
            [6.5, 64.5, 0.6],
            cell,
            &PANEL_Z
        ));
        // A point.
        assert!(segment_hits_shape(
            [3.5, 64.2, 0.5],
            [3.5, 64.2, 0.5],
            cell,
            &BOTTOM_SLAB
        ));
        assert!(!segment_hits_shape(
            [3.5, 64.8, 0.5],
            [3.5, 64.8, 0.5],
            cell,
            &BOTTOM_SLAB
        ));

        assert_eq!(shape_seals(&BOTTOM_SLAB), 0b010);
        assert_eq!(shape_seals(&PANEL_X), 0b001);
        // A fence post closes nothing.
        assert_eq!(shape_seals(&[[0.375, 0.0, 0.375, 0.625, 1.5, 0.625]]), 0);
        // Stairs facing north: a bottom slab and the back half above it.
        let stairs = [
            [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
            [0.0, 0.5, 0.0, 1.0, 1.0, 0.5],
        ];
        assert_eq!(shape_seals(&stairs), 0b110);
    }

    #[test]
    fn a_closed_door_muffles_and_an_open_one_does_not() {
        let settings = AcousticsSettings::default();
        let (listener, source) = ([0.5, 64.6, 0.5], [6.5, 64.5, 0.5]);
        // A stone wall at x = 3 with a doorway two blocks high at z = 0, a wooden door in it.
        let door = |panel: &'static [[f64; 6]]| {
            move |c: [i32; 3]| {
                if c[0] != 3 || !(63..=66).contains(&c[1]) || c[2].abs() > 2 {
                    return None;
                }
                if c[2] == 0 && (64..=65).contains(&c[1]) {
                    return Some((Surface::Wood, Some(Shape::from(panel))));
                }
                Some((Surface::Rock, None))
            }
        };
        let closed = source_acoustics_shaped(&mut door(&PANEL_X), listener, source, &settings);
        let open = source_acoustics_shaped(&mut door(&PANEL_Z), listener, source, &settings);
        assert!(closed.occlusion < 0.5, "{closed:?}");
        assert!(
            open.occlusion > closed.occlusion,
            "open {open:?}, closed {closed:?}"
        );
        assert!(open.direct_gains()[2] > 2.0 * closed.direct_gains()[2]);

        // The grids: a closed door between two open cells closes its cell; the open one lies along the wall.
        let wall_beside = |o: [i32; 3]| o[2] != 0;
        assert!(shape_blocks_cell(shape_seals(&PANEL_X), wall_beside));
        assert!(!shape_blocks_cell(shape_seals(&PANEL_Z), wall_beside));
        // A slab on the ground does not, a slab floor over a void does.
        assert!(!shape_blocks_cell(shape_seals(&BOTTOM_SLAB), |o| o[1] < 0));
        assert!(shape_blocks_cell(shape_seals(&BOTTOM_SLAB), |_| false));
    }

    #[test]
    fn published_values_settle_smoothly() {
        let path = PathAcoustics {
            direction: [1.0, 0.0, 0.0],
            gain: 0.8,
            eq: [0.7, 0.5, 0.3],
            distance: 12.0,
        };
        let walled = SourceAcoustics {
            occlusion: 0.0,
            transmission: [0.2, 0.1, 0.05],
            through: [1.0; NUM_BANDS],
            path: Some(path),
        };
        publish(&[(42, SourceAcoustics::OPEN)]);
        publish(&[(42, walled)]);
        let mut snapshot = Snapshot::default();
        snapshot.update();
        let halfway = snapshot.sources[&42];
        assert!(
            halfway.occlusion > 0.4 && halfway.occlusion < 0.6,
            "{halfway:?}"
        );
        // The way round fades in, from where it is.
        let p = halfway.path.expect("fading in");
        assert!(
            p.gain > 0.3 && p.gain < 0.5 && p.direction == path.direction,
            "{p:?}"
        );
        assert!(halfway.path_gain() < p.gain);
        // ...and out.
        publish(&[(42, SourceAcoustics::OPEN)]);
        snapshot.update();
        let fading = snapshot.sources[&42].path.expect("fading out");
        assert!(fading.gain < p.gain, "{fading:?}");
        assert_eq!(snapshot.direct_gains(7), [1.0; NUM_BANDS]);
        publish(&[]);
        snapshot.update();
        assert!(snapshot.sources.is_empty());
    }
}
