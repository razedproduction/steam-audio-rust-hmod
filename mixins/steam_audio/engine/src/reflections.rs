//! Real-time reflections and reverb of the listener's surroundings, simulated by Steam Audio from the blocks.
//!
//! This is Steam Audio's *listener-centric reverb*: a source placed at the listener, rays from the listener bouncing
//! off the blocks around it ([`ReflectionSimulator`]), the energy coming back to the listener over time per band and
//! direction ([`EnergyField`], order-1 Ambisonics), averaged over runs while nothing changes. From the field:
//!
//! - the reverb time of each band ([`reverb_estimator::estimate`]);
//! - an Ambisonics impulse response ([`Reconstructor`]) of the first [`TRANSITION_TIME`], crossfaded into a parametric
//!   tail ([`HybridReverbEstimator`]) and partitioned for convolution ([`OverlapSavePartitioner`]).
//!
//! Up to [`MAX_SOURCES`] sources near the listener ([`submit_sources`]) are simulated with it, each from where it
//! is, and get reflections of their own ([`Reflections::sources`]); the others share the listener's.
//!
//! The renderer plays them through Steam Audio's [`HybridReverbEffect`](crate::effects::HybridReverbEffect): the
//! early reflections from where the walls are, rotated with the head and rendered binaurally, the late tail
//! parametric.
//!
//! The blocks live on the game thread: it copies a [`GRID_SIZE`] block of them around the listener, a slice per frame
//! ([`GridBuilder`]), into [`Voxels`], a [`RayTracer`] that walks the grid cell by cell. Finished grids and the
//! listener's position go to a thread of its own ([`submit_grid`], [`submit_listener`]), which simulates at most
//! every [`MIN_INTERVAL`] and publishes [`Reflections`] for the audio thread ([`take`]), which never waits for it.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, LazyLock, Mutex, Once};
use std::time::{Duration, Instant};

use crate::acoustics::Surface;
use crate::air_absorption::AirAbsorptionModel;
use crate::bands::NUM_BANDS;
use crate::directivity::Directivity;
use crate::effects::{OverlapSaveFir, OverlapSavePartitioner};
use crate::geometry::Ray;
use crate::math::{CoordinateSpace3, Vector3};
use crate::random::CRand;
use crate::scene::{Hit, Material, RayTracer};
use crate::sh;
use crate::simulation::reverb_estimator;
use crate::simulation::{
    EnergyField, HybridReverbEstimator, ImpulseResponse, ReconstructionParams, ReconstructionType,
    Reconstructor, ReflectionSimulationParams, ReflectionSimulator, ReflectionSimulatorSettings,
    Reverb,
};
use crate::thread_pool::ThreadPool;

/// Blocks copied around the listener (x, y, z). Rays leaving it escape, as into the open.
pub const GRID_SIZE: [usize; 3] = [96, 64, 96];
/// The grid's corner is snapped to multiples of this, so grids copied while the listener walks about are the same
/// grid and the simulation keeps refining.
const GRID_SNAP: i32 = 8;
/// The listener must be at least this far inside a grid (blocks) for it to be simulated in.
const GRID_MARGIN: f32 = 6.0;

/// Ambisonics order of the reflections.
pub const ORDER: usize = 1;
/// Channels of the reflections' impulse response.
pub const NUM_CHANNELS: usize = 4;
/// Seconds of reflections simulated (the reverb times are fitted over them).
const DURATION: f32 = 2.0;
/// Seconds of impulse response convolved; the parametric reverb takes over after it.
pub const TRANSITION_TIME: f32 = 1.0;
/// Last fraction of the transition over which the impulse response crossfades into the parametric reverb.
const OVERLAP_FRACTION: f32 = 0.25;

/// Rays and bounces of one simulation (Steam Audio's real-time defaults are 4096 rays and 16 bounces; half the rays
/// here, made up for by averaging the runs).
const NUM_RAYS: usize = 2048;
const NUM_BOUNCES: usize = 16;
const NUM_DIFFUSE_SAMPLES: usize = 32;
const NUM_THREADS: usize = 2;
/// Runs averaged while the listener and the blocks stay the same; then the simulation rests until something changes.
const MAX_RUNS: usize = 16;
/// The listener moving less than this (blocks) is not a change.
const LISTENER_TOLERANCE: f64 = 0.25;
/// Shortest time between two simulations (longer when they take long: the thread works at most a quarter of the time).
pub const MIN_INTERVAL: Duration = Duration::from_millis(250);

/// The blocks around the listener: which cells are walls, and of what. Positions given to its [`RayTracer`] methods
/// are relative to its corner ([`Voxels::to_local`]): world coordinates are too large for `f32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Voxels {
    origin: [i32; 3],
    size: [usize; 3],
    /// Per cell, 0 for no wall, else 1 + the index of its [`Surface`] in [`Surface::ALL`].
    cells: Vec<u8>,
}

/// Steam Audio's material of each [`Surface::ALL`].
static MATERIALS: LazyLock<[Material; Surface::ALL.len()]> =
    LazyLock::new(|| Surface::ALL.map(Surface::material));

impl Voxels {
    /// A grid with no walls, its corner at block `origin`.
    pub fn new(origin: [i32; 3], size: [usize; 3]) -> Self {
        Self {
            origin,
            size,
            cells: vec![0; size[0] * size[1] * size[2]],
        }
    }

    /// The grid `f` describes: `f` is asked for every block in it.
    pub fn from_fn(
        origin: [i32; 3],
        size: [usize; 3],
        mut f: impl FnMut([i32; 3]) -> Option<Surface>,
    ) -> Self {
        let mut grid = Self::new(origin, size);
        for index in 0..grid.cells.len() {
            grid.fill(index, &mut f);
        }
        grid
    }

    pub fn origin(&self) -> [i32; 3] {
        self.origin
    }

    pub fn size(&self) -> [usize; 3] {
        self.size
    }

    /// A position in the world, relative to the grid's corner.
    pub fn to_local(&self, position: [f64; 3]) -> Vector3 {
        let p: [f32; 3] = std::array::from_fn(|a| (position[a] - f64::from(self.origin[a])) as f32);
        Vector3::from_array(p)
    }

    /// The wall at world block `cell`, if any (none outside the grid).
    pub fn surface(&self, cell: [i32; 3]) -> Option<Surface> {
        let local: [i32; 3] = std::array::from_fn(|a| cell[a] - self.origin[a]);
        self.local_index(local)
            .and_then(|i| self.cells[i].checked_sub(1))
            .map(|s| Surface::ALL[usize::from(s)])
    }

    /// Sets the wall at world block `cell` (ignored outside the grid).
    pub fn set(&mut self, cell: [i32; 3], surface: Option<Surface>) {
        let local: [i32; 3] = std::array::from_fn(|a| cell[a] - self.origin[a]);
        if let Some(i) = self.local_index(local) {
            self.cells[i] = encode(surface);
        }
    }

    /// Whether the cell `local` (relative to the corner) is a wall; outside the grid, it is not.
    pub fn is_wall_local(&self, local: [i32; 3]) -> bool {
        self.local_index(local).is_some_and(|i| self.cells[i] != 0)
    }

    /// `point` (local), or the centre of a free cell next to it if it is inside a wall (a note block, a furnace: a
    /// sound comes out of its block's sides). `None` if it is walled in.
    pub fn free_point(&self, point: Vector3) -> Option<Vector3> {
        let c = [point.x, point.y, point.z].map(|v| v.floor() as i32);
        if !self.is_wall_local(c) {
            return Some(point);
        }
        [
            [0, 1, 0],
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 1],
            [0, 0, -1],
            [0, -1, 0],
        ]
        .into_iter()
        .map(|o: [i32; 3]| -> [i32; 3] { std::array::from_fn(|a| c[a] + o[a]) })
        .find(|n| !self.is_wall_local(*n))
        .map(|n| Vector3::new(n[0] as f32 + 0.5, n[1] as f32 + 0.5, n[2] as f32 + 0.5))
    }

    /// Whether `position` (local) is inside the grid at all.
    pub fn inside(&self, position: Vector3) -> bool {
        (0..3).all(|a| position[a] >= 0.0 && position[a] < self.size[a] as f32)
    }

    /// Whether `position` (local) is far enough inside the grid to be simulated in.
    pub fn contains(&self, position: Vector3) -> bool {
        (0..3)
            .all(|a| position[a] >= GRID_MARGIN && position[a] <= self.size[a] as f32 - GRID_MARGIN)
    }

    /// Cells are stored x-major, then z, then y.
    fn local_index(&self, c: [i32; 3]) -> Option<usize> {
        if (0..3).any(|a| c[a] < 0 || c[a] as usize >= self.size[a]) {
            return None;
        }
        let [x, y, z] = c.map(|v| v as usize);
        Some((x * self.size[2] + z) * self.size[1] + y)
    }

    fn cell_of_index(&self, index: usize) -> [i32; 3] {
        let y = index % self.size[1];
        let z = (index / self.size[1]) % self.size[2];
        let x = index / (self.size[1] * self.size[2]);
        [
            self.origin[0] + x as i32,
            self.origin[1] + y as i32,
            self.origin[2] + z as i32,
        ]
    }

    fn fill(&mut self, index: usize, f: &mut impl FnMut([i32; 3]) -> Option<Surface>) {
        self.cells[index] = encode(f(self.cell_of_index(index)));
    }

    /// The first wall along `ray` between `min_distance` and `max_distance`: its distance, its face's normal (facing
    /// the ray) and its surface index. A ray starting inside a wall hits it at once.
    fn trace(
        &self,
        ray: &Ray,
        min_distance: f32,
        max_distance: f32,
    ) -> Option<(f32, Vector3, usize)> {
        let o = ray.origin;
        let d = ray.direction;
        let size: [f32; 3] = self.size.map(|s| s as f32);
        // The part of the ray inside the grid, and the axis it enters through.
        let mut t0 = min_distance.max(0.0);
        let mut t1 = max_distance;
        let mut entry = None;
        for a in 0..3 {
            if d[a] == 0.0 {
                if o[a] < 0.0 || o[a] > size[a] {
                    return None;
                }
                continue;
            }
            let inverse = 1.0 / d[a];
            let (mut near, mut far) = (-o[a] * inverse, (size[a] - o[a]) * inverse);
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            if near > t0 {
                t0 = near;
                entry = Some(a);
            }
            t1 = t1.min(far);
        }
        if t0.is_nan() || t1.is_nan() || t0 > t1 {
            return None;
        }
        let start = ray.point_at_distance(t0);
        let mut cell: [i32; 3] =
            std::array::from_fn(|a| (start[a].floor() as i32).clamp(0, self.size[a] as i32 - 1));
        let face = |a: usize, step: f32| {
            let mut n = [0.0f32; 3];
            n[a] = -step;
            Vector3::from_array(n)
        };
        let solid = |c: [i32; 3]| {
            let [x, y, z] = c.map(|v| v as usize);
            self.cells[(x * self.size[2] + z) * self.size[1] + y]
        };
        let here = solid(cell);
        if here != 0 {
            let normal = match entry {
                Some(a) => face(a, d[a].signum()),
                None => (-d).normalize_or_zero(),
            };
            return Some((t0, normal, usize::from(here - 1)));
        }
        let mut step = [0i32; 3];
        let mut next = [f32::INFINITY; 3];
        let mut delta = [f32::INFINITY; 3];
        for a in 0..3 {
            if d[a] > 0.0 {
                step[a] = 1;
                delta[a] = 1.0 / d[a];
                next[a] = t0 + (cell[a] as f32 + 1.0 - start[a]) / d[a];
            } else if d[a] < 0.0 {
                step[a] = -1;
                delta[a] = -1.0 / d[a];
                next[a] = t0 + (cell[a] as f32 - start[a]) / d[a];
            }
        }
        loop {
            let a = if next[0] <= next[1] && next[0] <= next[2] {
                0
            } else if next[1] <= next[2] {
                1
            } else {
                2
            };
            let t = next[a];
            if t > t1 {
                return None;
            }
            cell[a] += step[a];
            if cell[a] < 0 || cell[a] as usize >= self.size[a] {
                return None;
            }
            let s = solid(cell);
            if s != 0 {
                return Some((t, face(a, step[a] as f32), usize::from(s - 1)));
            }
            next[a] += delta[a];
        }
    }
}

/// A cell's code: only walls ([`Surface::is_solid`]) are kept; rays go through leaves and fences.
fn encode(surface: Option<Surface>) -> u8 {
    surface.filter(|s| s.is_solid()).map_or(0, |s| {
        1 + Surface::ALL.iter().position(|&x| x == s).unwrap_or(0) as u8
    })
}

impl RayTracer for Voxels {
    fn closest_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> Hit {
        match self.trace(ray, min_distance, max_distance) {
            Some((distance, normal, surface)) => Hit {
                distance,
                triangle_index: Some(0),
                object_index: None,
                material_index: Some(surface),
                normal,
                material: Some(MATERIALS[surface]),
            },
            None => Hit::default(),
        }
    }

    fn any_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> bool {
        self.trace(ray, min_distance, max_distance)
            .is_some_and(|(t, _, _)| t < max_distance)
    }
}

/// Copies the blocks around the listener into [`Voxels`] a slice at a time, on the game thread.
#[derive(Debug, Default)]
pub struct GridBuilder {
    pass: Option<Voxels>,
    next: usize,
}

impl GridBuilder {
    pub const fn new() -> Self {
        Self {
            pass: None,
            next: 0,
        }
    }

    /// The corner of the grid around `listener`.
    pub fn origin_for(listener: [f64; 3]) -> [i32; 3] {
        std::array::from_fn(|a| {
            let snapped = (listener[a].floor() as i32).div_euclid(GRID_SNAP) * GRID_SNAP;
            snapped + GRID_SNAP / 2 - GRID_SIZE[a] as i32 / 2
        })
    }

    /// Copies up to `budget` blocks (`wall` says what each is) and returns the grid when it is complete. A grid the
    /// listener has left before it is complete is started over around it.
    pub fn step(
        &mut self,
        listener: [f64; 3],
        budget: usize,
        wall: &mut impl FnMut([i32; 3]) -> Option<Surface>,
    ) -> Option<Voxels> {
        let fresh = match &self.pass {
            Some(pass) => !pass.contains(pass.to_local(listener)),
            None => true,
        };
        if fresh {
            self.pass = Some(Voxels::new(Self::origin_for(listener), GRID_SIZE));
            self.next = 0;
        }
        let pass = self.pass.as_mut()?;
        let end = (self.next + budget.max(1)).min(pass.cells.len());
        for index in self.next..end {
            pass.fill(index, wall);
        }
        self.next = end;
        if end < pass.cells.len() {
            return None;
        }
        self.next = 0;
        self.pass.take()
    }
}

/// Most sources simulated with reflections of their own (the nearest, which the game thread picks): their early
/// reflections come from where they are, not from where the listener is. The other sources share the listener's.
pub const MAX_SOURCES: usize = 4;
/// A source moving less than this (blocks) keeps refining its reflections; farther, they start over.
const SOURCE_TOLERANCE: f64 = 1.0;

/// One set of reflections ready to play: the impulse response of the first [`TRANSITION_TIME`] ([`NUM_CHANNELS`]
/// Ambisonics channels in world space), ready for convolution, and the parametric reverb taking over after it
/// (reverb times, level per band, delay in samples).
#[derive(Debug)]
pub struct ReverbIr {
    pub fir: OverlapSaveFir,
    pub reverb: Reverb,
    pub eq: [f32; NUM_BANDS],
    pub delay: usize,
}

/// What the renderer plays: the listener-centric reverb (`fir`, `reverb`, `eq`, `delay`) for the sources without
/// reflections of their own, and those of the sources that have them.
#[derive(Debug)]
pub struct Reflections {
    /// The sampling rate it was made for.
    pub rate: u32,
    /// The impulse response of the first [`TRANSITION_TIME`], [`NUM_CHANNELS`] Ambisonics channels in world space.
    pub fir: OverlapSaveFir,
    /// The parametric reverb's reverb times, level per band and delay in samples.
    pub reverb: Reverb,
    pub eq: [f32; NUM_BANDS],
    pub delay: usize,
    /// The reflections of each source simulated on its own (by mixer channel id), at most [`MAX_SOURCES`].
    pub sources: Vec<(u64, ReverbIr)>,
}

impl Reflections {
    /// Every impulse response it holds, to be reused.
    pub fn into_firs(self) -> impl Iterator<Item = OverlapSaveFir> {
        std::iter::once(self.fir).chain(self.sources.into_iter().map(|(_, s)| s.fir))
    }
}

/// Samples of impulse response convolved at `rate`.
pub fn ir_size(rate: u32) -> usize {
    (TRANSITION_TIME * rate as f32).ceil() as usize
}

/// The energy field of a source simulated on its own, averaged while it and the listener stay put.
#[derive(Debug)]
struct SourceState {
    id: u64,
    /// Where it was simulated (world).
    position: [f64; 3],
    accumulated: EnergyField,
    runs: usize,
    ir: ImpulseResponse,
    estimator: HybridReverbEstimator,
}

/// Simulates reflections at one sampling rate, synchronously (the thread behind [`submit_grid`] runs one).
#[derive(Debug)]
pub struct Simulation {
    rate: u32,
    frame_size: usize,
    simulator: ReflectionSimulator,
    pool: ThreadPool,
    /// One per simulated source: the listener-centric one first, then [`MAX_SOURCES`].
    fields: Vec<EnergyField>,
    accumulated: EnergyField,
    runs: usize,
    reconstructor: Reconstructor,
    estimator: HybridReverbEstimator,
    partitioner: OverlapSavePartitioner,
    ir: ImpulseResponse,
    air: AirAbsorptionModel,
    /// The grid and the listener position (world) the accumulated fields are for.
    last: Option<(Arc<Voxels>, [f64; 3])>,
    sources: Vec<SourceState>,
}

impl Simulation {
    pub fn new(rate: u32, frame_size: usize) -> Self {
        let settings = ReflectionSimulatorSettings {
            max_num_rays: NUM_RAYS,
            num_diffuse_samples: NUM_DIFFUSE_SAMPLES,
            max_duration: DURATION,
            max_order: ORDER,
            max_num_sources: 1 + MAX_SOURCES,
            num_threads: NUM_THREADS,
        };
        Self {
            rate,
            frame_size,
            simulator: ReflectionSimulator::new(&settings, 5489),
            pool: ThreadPool::new(NUM_THREADS),
            fields: (0..=MAX_SOURCES)
                .map(|_| EnergyField::new(DURATION, ORDER))
                .collect(),
            accumulated: EnergyField::new(DURATION, ORDER),
            runs: 0,
            reconstructor: Reconstructor::new(TRANSITION_TIME, rate),
            estimator: Self::new_estimator(rate, frame_size),
            partitioner: OverlapSavePartitioner::new(frame_size),
            ir: ImpulseResponse::new(TRANSITION_TIME, ORDER, rate),
            air: AirAbsorptionModel::default(),
            last: None,
            sources: Vec::new(),
        }
    }

    fn new_estimator(rate: u32, frame_size: usize) -> HybridReverbEstimator {
        HybridReverbEstimator::new(TRANSITION_TIME, rate, frame_size, &mut CRand::default())
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Whether a run for `listener` and `sources` in `grid` would refine the last one (or start over: something
    /// changed).
    pub fn wants(
        &self,
        grid: &Arc<Voxels>,
        listener: [f64; 3],
        sources: &[(u64, [f64; 3])],
    ) -> bool {
        if !self.same(grid, listener) || self.runs < MAX_RUNS {
            return true;
        }
        let usable = usable_sources(grid, sources);
        usable.len() != self.sources.len()
            || usable
                .iter()
                .zip(&self.sources)
                .any(|(&(id, _, at), state)| !state.matches(id, at) || state.runs < MAX_RUNS)
    }

    fn same(&self, grid: &Arc<Voxels>, listener: [f64; 3]) -> bool {
        self.last.as_ref().is_some_and(|(g, l)| {
            (Arc::ptr_eq(g, grid) || **g == **grid)
                && (0..3).all(|a| (l[a] - listener[a]).abs() < LISTENER_TOLERANCE)
        })
    }

    /// Runs the simulation for `listener` in `grid`, and for the `sources` (mixer channel id, world position; at most
    /// [`MAX_SOURCES`] of those inside the grid are), and makes the reflections. Impulse responses are taken from
    /// `spare` when it has some of the right size. `None` if the listener is not inside the grid.
    pub fn run(
        &mut self,
        grid: &Arc<Voxels>,
        listener: [f64; 3],
        sources: &[(u64, [f64; 3])],
        spare: &mut Vec<OverlapSaveFir>,
    ) -> Option<Reflections> {
        let position = grid.to_local(listener);
        if !grid.contains(position) {
            return None;
        }
        if !self.same(grid, listener) {
            self.runs = 0;
            self.last = Some((Arc::clone(grid), listener));
            for state in &mut self.sources {
                state.runs = 0;
            }
        }
        // Each source keeps its average while it stays put; new ones start one.
        let usable = usable_sources(grid, sources);
        let mut previous = std::mem::take(&mut self.sources);
        for &(id, _, at) in &usable {
            let state = match previous.iter().position(|s| s.id == id) {
                Some(i) => {
                    let mut state = previous.swap_remove(i);
                    if !state.matches(id, at) {
                        state.position = at;
                        state.runs = 0;
                    }
                    state
                }
                None => SourceState {
                    id,
                    position: at,
                    accumulated: EnergyField::new(DURATION, ORDER),
                    runs: 0,
                    ir: ImpulseResponse::new(TRANSITION_TIME, ORDER, self.rate),
                    estimator: Self::new_estimator(self.rate, self.frame_size),
                },
            };
            self.sources.push(state);
        }

        let at = CoordinateSpace3::canonical_at(position);
        let spaces: Vec<CoordinateSpace3> = std::iter::once(at)
            .chain(
                usable
                    .iter()
                    .map(|&(_, p, _)| CoordinateSpace3::canonical_at(p)),
            )
            .collect();
        let directivities = vec![Directivity::default(); spaces.len()];
        let params = ReflectionSimulationParams {
            sources: &spaces,
            directivities: &directivities,
            listener: at,
            num_rays: NUM_RAYS,
            num_bounces: NUM_BOUNCES,
            duration: DURATION,
            order: ORDER,
            irradiance_min_distance: 1.0,
        };
        self.simulator.simulate(
            &**grid,
            &params,
            &mut self.fields[..spaces.len()],
            &self.pool,
        );
        // The average of the runs so far (Steam Audio's accumulation over frames).
        accumulate(&mut self.accumulated, &self.fields[0], &mut self.runs);
        for (state, field) in self.sources.iter_mut().zip(&self.fields[1..]) {
            accumulate(&mut state.accumulated, field, &mut state.runs);
        }

        let size = ir_size(self.rate);
        let blocks = OverlapSaveFir::new(1, size, self.frame_size).num_blocks();
        let mut take_fir = || {
            spare
                .iter()
                .position(|f| f.num_channels() == NUM_CHANNELS && f.num_blocks() == blocks)
                .map(|i| spare.swap_remove(i))
                .unwrap_or_else(|| OverlapSaveFir::new(NUM_CHANNELS, size, self.frame_size))
        };
        let mut make = |field: &EnergyField,
                        ir: &mut ImpulseResponse,
                        estimator: &mut HybridReverbEstimator,
                        fir: OverlapSaveFir| {
            make_ir(
                &mut self.reconstructor,
                &mut self.partitioner,
                &self.air,
                field,
                ir,
                estimator,
                fir,
            )
        };
        let listener_ir = make(
            &self.accumulated,
            &mut self.ir,
            &mut self.estimator,
            take_fir(),
        );
        let sources = self
            .sources
            .iter_mut()
            .map(|s| {
                let ir = make(&s.accumulated, &mut s.ir, &mut s.estimator, take_fir());
                (s.id, ir)
            })
            .collect();
        Some(Reflections {
            rate: self.rate,
            fir: listener_ir.fir,
            reverb: listener_ir.reverb,
            eq: listener_ir.eq,
            delay: listener_ir.delay,
            sources,
        })
    }

    /// The listener-centric impulse response of the last run (channel 0 is the omnidirectional one).
    pub fn impulse_response(&self) -> &ImpulseResponse {
        &self.ir
    }

    /// The listener-centric energy field averaged over the runs so far.
    pub fn energy_field(&self) -> &EnergyField {
        &self.accumulated
    }

    /// The energy field of source `id` averaged over the runs so far, if it is simulated on its own.
    pub fn source_energy_field(&self, id: u64) -> Option<&EnergyField> {
        self.sources
            .iter()
            .find(|s| s.id == id)
            .map(|s| &s.accumulated)
    }
}

impl SourceState {
    fn matches(&self, id: u64, at: [f64; 3]) -> bool {
        self.id == id && (0..3).all(|a| (self.position[a] - at[a]).abs() < SOURCE_TOLERANCE)
    }
}

/// The first [`MAX_SOURCES`] of `sources` inside the grid, with their local position (out of their block if they are
/// in one) and world position.
fn usable_sources(grid: &Voxels, sources: &[(u64, [f64; 3])]) -> Vec<(u64, Vector3, [f64; 3])> {
    sources
        .iter()
        .filter_map(|&(id, at)| {
            let local = grid.to_local(at);
            if !grid.inside(local) {
                return None;
            }
            Some((id, grid.free_point(local)?, at))
        })
        .take(MAX_SOURCES)
        .collect()
}

/// Adds `field` to the average `accumulated` of `runs` runs.
fn accumulate(accumulated: &mut EnergyField, field: &EnergyField, runs: &mut usize) {
    let n = *runs as f32;
    if *runs == 0 {
        accumulated.copy_from(field);
    } else {
        accumulated.scale_in_place(n);
        accumulated.add_assign(field);
        accumulated.scale_in_place(1.0 / (n + 1.0));
    }
    *runs = (*runs + 1).min(MAX_RUNS);
}

/// From an energy field: the reverb times, the impulse response of the first [`TRANSITION_TIME`] crossfaded into the
/// parametric reverb, partitioned into `fir`.
fn make_ir(
    reconstructor: &mut Reconstructor,
    partitioner: &mut OverlapSavePartitioner,
    air: &AirAbsorptionModel,
    field: &EnergyField,
    ir: &mut ImpulseResponse,
    estimator: &mut HybridReverbEstimator,
    mut fir: OverlapSaveFir,
) -> ReverbIr {
    let reverb = reverb_estimator::estimate(field, air);
    reconstructor.reconstruct(
        field,
        ir,
        &ReconstructionParams {
            kind: ReconstructionType::Linear,
            duration: TRANSITION_TIME,
            order: ORDER,
            air_absorption: air,
            distance_attenuation_correction: None,
        },
    );
    let estimate = estimator.estimate(
        Some(field),
        &reverb,
        ir,
        TRANSITION_TIME,
        OVERLAP_FRACTION,
        ORDER,
    );
    partitioner.partition(ir, NUM_CHANNELS, ir.num_samples(), &mut fir);
    ReverbIr {
        fir,
        reverb,
        eq: estimate.eq_coeffs,
        delay: estimate.delay,
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The simulation thread
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Request {
    grid: Option<Arc<Voxels>>,
    listener: Option<[f64; 3]>,
    sources: Vec<(u64, [f64; 3])>,
}

static REQUEST: LazyLock<(Mutex<Request>, Condvar)> =
    LazyLock::new(|| (Mutex::new(Request::default()), Condvar::new()));
/// The latest reflections, for the audio thread.
static OUTPUT: Mutex<Option<Box<Reflections>>> = Mutex::new(None);
/// Impulse responses the audio thread is done with, for the next runs to reuse.
static SPARE: Mutex<Vec<OverlapSaveFir>> = Mutex::new(Vec::new());
/// Most impulse responses kept for reuse: two runs' worth.
const MAX_SPARE: usize = 2 * (1 + MAX_SOURCES);
/// The renderer's sampling rate (0 until there is one).
static RATE: AtomicU32 = AtomicU32::new(0);
/// The renderer's frame size.
static FRAME_SIZE: AtomicU32 = AtomicU32::new(512);
static STARTED: Once = Once::new();

/// Gives the simulation a newly copied grid (game thread; the grid is shared with [`crate::paths`]).
pub fn submit_grid(grid: &Arc<Voxels>) {
    start();
    let (lock, signal) = &*REQUEST;
    if let Ok(mut request) = lock.lock() {
        if request.grid.as_deref() != Some(&**grid) {
            request.grid = Some(Arc::clone(grid));
            signal.notify_one();
        }
    }
}

/// Tells the simulation where the listener is (game thread).
pub fn submit_listener(listener: [f64; 3]) {
    let (lock, signal) = &*REQUEST;
    if let Ok(mut request) = lock.lock() {
        let moved = request
            .listener
            .is_none_or(|l| (0..3).any(|a| (l[a] - listener[a]).abs() >= LISTENER_TOLERANCE));
        request.listener = Some(listener);
        if moved {
            signal.notify_one();
        }
    }
}

/// Tells the simulation which sources to simulate on their own (game thread): mixer channel id and world position,
/// the most important first; at most [`MAX_SOURCES`] of them are.
pub fn submit_sources(sources: &[(u64, [f64; 3])]) {
    let (lock, signal) = &*REQUEST;
    if let Ok(mut request) = lock.lock() {
        let changed = request.sources.len() != sources.len()
            || request.sources.iter().zip(sources).any(|(a, b)| {
                a.0 != b.0 || (0..3).any(|i| (a.1[i] - b.1[i]).abs() >= SOURCE_TOLERANCE)
            });
        request.sources.clear();
        request.sources.extend_from_slice(sources);
        if changed {
            signal.notify_one();
        }
    }
}

/// The renderer's sampling rate and frame size, which the reflections are made for (audio thread; never waits).
pub fn set_rate(rate: u32, frame_size: usize) {
    FRAME_SIZE.store(frame_size as u32, Ordering::Relaxed);
    RATE.store(rate, Ordering::Relaxed);
}

/// The reflections published since the last call, if any (audio thread; never waits).
pub fn take() -> Option<Box<Reflections>> {
    OUTPUT.try_lock().ok()?.take()
}

/// Hands back an impulse response the audio thread is done with (never waits; dropped if the lock is busy).
pub fn recycle(fir: OverlapSaveFir) {
    if let Ok(mut spare) = SPARE.try_lock()
        && spare.len() < MAX_SPARE
    {
        spare.push(fir);
    }
}

fn start() {
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("steam-audio-reflections".into())
            .spawn(run_thread);
        if let Err(e) = spawned {
            log::warn!("steam_audio: cannot start the reflection simulation ({e}); no reverb");
        }
    });
}

fn run_thread() {
    let mut simulation: Option<Simulation> = None;
    let mut last_run: Option<(Instant, Duration)> = None;
    let mut spare = Vec::new();
    loop {
        // At least `MIN_INTERVAL` between runs, and the thread busy at most a quarter of the time.
        if let Some((t, took)) = last_run {
            let wait = MIN_INTERVAL.max(took * 3);
            let elapsed = t.elapsed();
            if elapsed < wait {
                std::thread::sleep(wait - elapsed);
            }
        }
        // Wait for a grid and a listener the simulation would do something with.
        let (grid, listener, sources, rate) = {
            let (lock, signal) = &*REQUEST;
            let Ok(mut request) = lock.lock() else {
                return;
            };
            loop {
                let rate = RATE.load(Ordering::Relaxed);
                if rate != 0
                    && let (Some(grid), Some(listener)) = (&request.grid, request.listener)
                    && simulation.as_ref().is_none_or(|s| {
                        s.rate() != rate || s.wants(grid, listener, &request.sources)
                    })
                {
                    break (Arc::clone(grid), listener, request.sources.clone(), rate);
                }
                match signal.wait_timeout(request, Duration::from_millis(500)) {
                    Ok((r, _)) => request = r,
                    Err(_) => return,
                }
            }
        };
        let frame_size = FRAME_SIZE.load(Ordering::Relaxed) as usize;
        let simulation = match &mut simulation {
            Some(s) if s.rate() == rate => s,
            other => other.insert(Simulation::new(rate, frame_size)),
        };
        let started = Instant::now();
        if let Ok(mut s) = SPARE.lock() {
            spare.append(&mut s);
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            simulation.run(&grid, listener, &sources, &mut spare)
        }));
        last_run = Some((started, started.elapsed()));
        match result {
            Ok(Some(reflections)) => {
                if let Ok(mut output) = OUTPUT.lock()
                    && let Some(old) = output.replace(Box::new(reflections))
                {
                    drop(output);
                    spare.extend(old.into_firs());
                }
            }
            Ok(None) => {}
            Err(_) => {
                log::warn!("steam_audio: the reflection simulation failed; no reverb");
                return;
            }
        }
        spare.truncate(MAX_SPARE);
    }
}

/// The order-1 Ambisonics coefficients of a sound arriving from `direction` (world), for tests and tools.
pub fn coefficients(direction: Vector3) -> [f32; NUM_CHANNELS] {
    let mut c = [0.0; NUM_CHANNELS];
    sh::project_single_point(direction, ORDER, &mut c);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    fn room(half: [i32; 3], surface: Surface) -> impl FnMut([i32; 3]) -> Option<Surface> {
        // Air inside |x| <= half, 64 <= y < 64 + height, |z| <= half; walls around it.
        move |c| {
            let inside = c[0].abs() <= half[0]
                && (64..64 + half[1]).contains(&c[1])
                && c[2].abs() <= half[2];
            (!inside).then_some(surface)
        }
    }

    fn grid(listener: [f64; 3], f: impl FnMut([i32; 3]) -> Option<Surface>) -> Arc<Voxels> {
        Arc::new(Voxels::from_fn(
            GridBuilder::origin_for(listener),
            GRID_SIZE,
            f,
        ))
    }

    #[test]
    fn rays_stop_at_the_first_wall_facing_them() {
        let mut voxels = Voxels::new([0, 0, 0], [8, 8, 8]);
        voxels.set([5, 2, 2], Some(Surface::Wood));
        let ray = Ray::new(Vector3::new(1.5, 2.5, 2.5), Vector3::X_AXIS);
        let hit = voxels.closest_hit(&ray, 0.0, f32::INFINITY);
        assert!((hit.distance - 3.5).abs() < 1e-5, "{hit:?}");
        assert_eq!(hit.normal, Vector3::new(-1.0, 0.0, 0.0));
        assert_eq!(hit.material, Some(Surface::Wood.material()));
        assert!(voxels.any_hit(&ray, 0.0, 4.0));
        assert!(!voxels.any_hit(&ray, 0.0, 3.0));
        // Diagonal, from outside the grid, and leaving it.
        let diagonal = Ray::new(
            Vector3::new(-2.0, 0.5, 2.5),
            Vector3::new(1.0, 0.25, 0.0).normalize_or_zero(),
        );
        let hit = voxels.closest_hit(&diagonal, 0.0, f32::INFINITY);
        assert!(
            (hit.distance - 7.0 * 1.0625f32.sqrt()).abs() < 1e-4,
            "{hit:?}"
        );
        assert_eq!(hit.normal, Vector3::new(-1.0, 0.0, 0.0));
        let away = Ray::new(Vector3::new(1.5, 2.5, 2.5), -Vector3::X_AXIS);
        assert!(!voxels.closest_hit(&away, 0.0, f32::INFINITY).is_valid());
        assert_eq!(voxels.surface([5, 2, 2]), Some(Surface::Wood));
        assert_eq!(voxels.surface([50, 2, 2]), None);
    }

    #[test]
    fn the_builder_copies_the_grid_around_the_listener_in_slices() {
        let listener = [100.3, 70.0, -20.7];
        let mut builder = GridBuilder::default();
        let mut wall = room([5, 6, 5], Surface::Rock);
        let total = GRID_SIZE.iter().product::<usize>();
        let budget = total / 3 + 1;
        assert!(builder.step(listener, budget, &mut wall).is_none());
        assert!(builder.step(listener, budget, &mut wall).is_none());
        let grid = builder.step(listener, budget, &mut wall).expect("complete");
        assert_eq!(
            grid,
            Voxels::from_fn(grid.origin(), GRID_SIZE, room([5, 6, 5], Surface::Rock))
        );
        assert!(grid.contains(grid.to_local(listener)));
        // Walking within the snap keeps the same grid.
        assert_eq!(
            GridBuilder::origin_for(listener),
            GridBuilder::origin_for([101.9, 70.5, -19.2])
        );
    }

    /// Runs the simulation until it settles and returns the reverb and the impulse response's energy.
    fn simulate(grid: &Arc<Voxels>, listener: [f64; 3]) -> (Reverb, f32, Simulation) {
        let mut simulation = Simulation::new(48_000, 512);
        let mut last = None;
        for _ in 0..4 {
            last = simulation.run(grid, listener, &[], &mut Vec::new());
        }
        let reflections = last.expect("inside the grid");
        let energy = simulation
            .impulse_response()
            .channel(0)
            .iter()
            .map(|s| s * s)
            .sum::<f32>();
        (reflections.reverb, energy, simulation)
    }

    #[test]
    fn caves_reverberate_and_fields_stay_dry() {
        let listener = [0.5, 65.6, 0.5];
        let field = grid(listener, |c| (c[1] < 64).then_some(Surface::Gravel));
        let small = grid(listener, room([4, 5, 4], Surface::Rock));
        let large = grid(listener, room([16, 20, 16], Surface::Rock));
        let (field_reverb, field_energy, _) = simulate(&field, listener);
        let (small_reverb, small_energy, _) = simulate(&small, listener);
        let (large_reverb, large_energy, _) = simulate(&large, listener);
        eprintln!("field {field_reverb:?} {field_energy}");
        eprintln!("small {small_reverb:?} {small_energy}");
        eprintln!("large {large_reverb:?} {large_energy}");
        // In the open only the ground reflects, once.
        assert!(
            small_energy > 5.0 * field_energy,
            "{small_energy} {field_energy}"
        );
        assert_eq!(field_reverb.reverb_times, [0.1; NUM_BANDS]);
        assert!(
            small_reverb.reverb_times[1] > 0.3 && small_reverb.reverb_times[1] < 3.0,
            "{small_reverb:?}"
        );
        assert!(
            large_reverb.reverb_times[1] > 1.5 * small_reverb.reverb_times[1],
            "{large_reverb:?} {small_reverb:?}"
        );
        // Wool rooms are deader than stone ones.
        let wool = grid(listener, room([4, 5, 4], Surface::Carpet));
        let (wool_reverb, _, _) = simulate(&wool, listener);
        assert!(
            wool_reverb.reverb_times[2] < 0.5 * small_reverb.reverb_times[2],
            "{wool_reverb:?} {small_reverb:?}"
        );
    }

    #[test]
    fn a_wall_close_by_sends_its_reflection_from_its_side() {
        // Open everywhere but a big wall at x = 3 (to the listener's east) and the ground.
        let listener = [0.5, 65.6, 0.5];
        let walled = grid(listener, |c| {
            (c[1] < 64 || (c[0] == 3 && c[1] < 80 && c[2].abs() < 16)).then_some(Surface::Rock)
        });
        let (_, _, simulation) = simulate(&walled, listener);
        let field = simulation.energy_field();
        // The order-1 channels, projected on the direction of the wall and on the opposite one.
        let toward = |direction: Vector3| {
            let c = coefficients(direction);
            (1..NUM_CHANNELS)
                .map(|ch| {
                    c[ch]
                        * (0..NUM_BANDS)
                            .map(|b| field.band(ch, b).iter().sum::<f32>())
                            .sum::<f32>()
                })
                .sum::<f32>()
        };
        let (east, north) = (toward(Vector3::X_AXIS), toward(-Vector3::Z_AXIS));
        assert!(
            east > 0.0 && east > 3.0 * north.abs(),
            "east {east}, north {north}"
        );
    }

    #[test]
    fn nothing_is_simulated_outside_the_grid() {
        let listener = [0.5, 65.6, 0.5];
        let g = grid(listener, room([4, 5, 4], Surface::Rock));
        let mut simulation = Simulation::new(48_000, 512);
        assert!(
            simulation
                .run(&g, [500.0, 65.0, 0.0], &[], &mut Vec::new())
                .is_none()
        );
        assert!(simulation.run(&g, listener, &[], &mut Vec::new()).is_some());
        assert!(simulation.wants(&g, listener, &[]));
        for _ in 0..MAX_RUNS {
            simulation.run(&g, listener, &[], &mut Vec::new());
        }
        assert!(!simulation.wants(&g, listener, &[]));
        assert!(simulation.wants(&g, [2.0, 65.6, 0.5], &[]));
    }

    #[test]
    fn a_big_cave_rings_longer_than_a_smaller_one() {
        // 81 blocks across: larger than the grid of 0.4.0 (64), inside this one.
        let listener = [0.5, 65.6, 0.5];
        let (big, _, _) = simulate(&grid(listener, room([40, 20, 40], Surface::Rock)), listener);
        let (smaller, _, _) =
            simulate(&grid(listener, room([16, 20, 16], Surface::Rock)), listener);
        assert!(
            big.reverb_times[1] > 1.2 * smaller.reverb_times[1],
            "{big:?} {smaller:?}"
        );
    }

    #[test]
    fn leaves_are_not_walls_in_the_grid() {
        let voxels = Voxels::from_fn([0, 0, 0], [4, 4, 4], |c| {
            (c[0] == 1).then_some(Surface::Foliage)
        });
        assert_eq!(voxels.surface([1, 1, 1]), None);
        let water = Voxels::from_fn([0, 0, 0], [4, 4, 4], |c| {
            (c[1] == 0).then_some(Surface::Water)
        });
        assert_eq!(water.surface([1, 0, 1]), Some(Surface::Water));
    }
    #[test]
    fn sources_get_reflections_of_their_own() {
        let listener = [0.5, 65.6, 0.5];
        let g = grid(listener, room([6, 5, 6], Surface::Rock));
        let mut simulation = Simulation::new(48_000, 512);
        // A source in the cave, one inside a wall of it (heard from the free cell beside it), one outside the grid.
        let sources = [
            (7, [4.5, 64.5, 0.5]),
            (8, [7.5, 65.5, 0.5]),
            (9, [900.5, 65.5, 0.5]),
        ];
        let reflections = simulation
            .run(&g, listener, &sources, &mut Vec::new())
            .expect("inside the grid");
        let ids: Vec<u64> = reflections.sources.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, [7, 8]);
        let energy = |field: &EnergyField| {
            (0..NUM_BANDS)
                .map(|b| field.band(0, b).iter().sum::<f32>())
                .sum::<f32>()
        };
        let own = energy(simulation.source_energy_field(7).expect("simulated"));
        assert!(own > 0.0, "{own}");
        // Staying put refines, moving starts over, a new set of sources is a change.
        for _ in 0..MAX_RUNS {
            simulation.run(&g, listener, &sources, &mut Vec::new());
        }
        assert!(!simulation.wants(&g, listener, &sources));
        assert!(!simulation.wants(&g, listener, &[(7, [4.9, 64.5, 0.5]), sources[1]]));
        assert!(simulation.wants(&g, listener, &[(7, [1.5, 64.5, 0.5]), sources[1]]));
        assert!(simulation.wants(&g, listener, &sources[..1]));
        // The impulse responses are reused.
        let mut spare: Vec<OverlapSaveFir> = reflections.into_firs().collect();
        assert_eq!(spare.len(), 3);
        simulation.run(&g, listener, &sources, &mut spare);
        assert!(spare.is_empty());
    }
    #[test]
    #[ignore = "measurement"]
    fn measure_simulation_time() {
        let listener = [0.5, 65.6, 0.5];
        for (name, g) in [
            ("small", grid(listener, room([4, 5, 4], Surface::Rock))),
            ("large", grid(listener, room([16, 20, 16], Surface::Rock))),
            (
                "field",
                grid(listener, |c| (c[1] < 64).then_some(Surface::Gravel)),
            ),
        ] {
            let mut simulation = Simulation::new(48_000, 512);
            let started = Instant::now();
            for _ in 0..8 {
                simulation.run(&g, listener, &[], &mut Vec::new());
            }
            eprintln!("{name}: {:?} per run", started.elapsed() / 8);
            let sources: Vec<(u64, [f64; 3])> = (0..MAX_SOURCES as u64)
                .map(|i| (i, [listener[0] + 2.0 + i as f64, 64.5, listener[2] - 1.0]))
                .collect();
            let started = Instant::now();
            for _ in 0..8 {
                simulation.run(&g, listener, &sources, &mut Vec::new());
            }
            eprintln!(
                "{name}: {:?} per run with {} sources of their own",
                started.elapsed() / 8,
                sources.len()
            );
            let at = CoordinateSpace3::canonical_at(g.to_local(listener));
            let params = ReflectionSimulationParams {
                sources: &[at],
                directivities: &[Directivity::default()],
                listener: at,
                num_rays: NUM_RAYS,
                num_bounces: NUM_BOUNCES,
                duration: DURATION,
                order: ORDER,
                irradiance_min_distance: 1.0,
            };
            let started = Instant::now();
            for _ in 0..8 {
                simulation.simulator.simulate(
                    &*g,
                    &params,
                    &mut simulation.fields[..1],
                    &simulation.pool,
                );
            }
            eprintln!("{name}: {:?} of it tracing", started.elapsed() / 8);
            let s = &mut simulation;
            let started = Instant::now();
            let reverb = reverb_estimator::estimate(&s.accumulated, &s.air);
            let t_reverb = started.elapsed();
            let started = Instant::now();
            s.reconstructor.reconstruct(
                &s.accumulated,
                &mut s.ir,
                &ReconstructionParams {
                    kind: ReconstructionType::Linear,
                    duration: TRANSITION_TIME,
                    order: ORDER,
                    air_absorption: &s.air,
                    distance_attenuation_correction: None,
                },
            );
            let t_reconstruct = started.elapsed();
            let started = Instant::now();
            s.estimator.estimate(
                Some(&s.accumulated),
                &reverb,
                &mut s.ir,
                TRANSITION_TIME,
                OVERLAP_FRACTION,
                ORDER,
            );
            let t_hybrid = started.elapsed();
            let started = Instant::now();
            let mut fir = OverlapSaveFir::new(NUM_CHANNELS, ir_size(48_000), 512);
            let t_alloc = started.elapsed();
            let started = Instant::now();
            s.partitioner
                .partition(&s.ir, NUM_CHANNELS, s.ir.num_samples(), &mut fir);
            eprintln!(
                "{name}: reverb {t_reverb:?}, reconstruct {t_reconstruct:?}, hybrid {t_hybrid:?}, fir {t_alloc:?}, \
                 partition {:?}",
                started.elapsed()
            );
        }
        let started = Instant::now();
        let _ = grid(listener, room([4, 5, 4], Surface::Rock));
        eprintln!("grid: {:?}", started.elapsed());
    }
}
