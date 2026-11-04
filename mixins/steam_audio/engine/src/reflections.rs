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
pub const GRID_SIZE: [usize; 3] = [64, 48, 64];
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
/// Shortest time between two simulations.
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

fn encode(surface: Option<Surface>) -> u8 {
    surface.map_or(0, |s| {
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

/// What the renderer plays: the early reflections' impulse response, ready for convolution, and the parametric
/// reverb taking over after it.
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
}

/// Samples of impulse response convolved at `rate`.
pub fn ir_size(rate: u32) -> usize {
    (TRANSITION_TIME * rate as f32).ceil() as usize
}

/// Simulates reflections at one sampling rate, synchronously (the thread behind [`submit_grid`] runs one).
#[derive(Debug)]
pub struct Simulation {
    rate: u32,
    frame_size: usize,
    simulator: ReflectionSimulator,
    pool: ThreadPool,
    field: EnergyField,
    accumulated: EnergyField,
    runs: usize,
    reconstructor: Reconstructor,
    estimator: HybridReverbEstimator,
    partitioner: OverlapSavePartitioner,
    ir: ImpulseResponse,
    air: AirAbsorptionModel,
    /// The grid and the listener position (world) the accumulated field is for.
    last: Option<(Arc<Voxels>, [f64; 3])>,
}

impl Simulation {
    pub fn new(rate: u32, frame_size: usize) -> Self {
        let settings = ReflectionSimulatorSettings {
            max_num_rays: NUM_RAYS,
            num_diffuse_samples: NUM_DIFFUSE_SAMPLES,
            max_duration: DURATION,
            max_order: ORDER,
            max_num_sources: 1,
            num_threads: NUM_THREADS,
        };
        Self {
            rate,
            frame_size,
            simulator: ReflectionSimulator::new(&settings, 5489),
            pool: ThreadPool::new(NUM_THREADS),
            field: EnergyField::new(DURATION, ORDER),
            accumulated: EnergyField::new(DURATION, ORDER),
            runs: 0,
            reconstructor: Reconstructor::new(TRANSITION_TIME, rate),
            estimator: HybridReverbEstimator::new(
                TRANSITION_TIME,
                rate,
                frame_size,
                &mut CRand::default(),
            ),
            partitioner: OverlapSavePartitioner::new(frame_size),
            ir: ImpulseResponse::new(TRANSITION_TIME, ORDER, rate),
            air: AirAbsorptionModel::default(),
            last: None,
        }
    }

    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Whether a run for `listener` in `grid` would refine the last one (or start over: something changed).
    pub fn wants(&self, grid: &Arc<Voxels>, listener: [f64; 3]) -> bool {
        !self.same(grid, listener) || self.runs < MAX_RUNS
    }

    fn same(&self, grid: &Arc<Voxels>, listener: [f64; 3]) -> bool {
        self.last.as_ref().is_some_and(|(g, l)| {
            (Arc::ptr_eq(g, grid) || **g == **grid)
                && (0..3).all(|a| (l[a] - listener[a]).abs() < LISTENER_TOLERANCE)
        })
    }

    /// Runs the simulation for `listener` in `grid` and makes the reflections, writing the impulse response into
    /// `fir` (one of [`ir_size`] samples and [`NUM_CHANNELS`] channels, or a new one if `None`). `None` if the listener
    /// is not inside the grid.
    pub fn run(
        &mut self,
        grid: &Arc<Voxels>,
        listener: [f64; 3],
        fir: Option<OverlapSaveFir>,
    ) -> Option<Reflections> {
        let position = grid.to_local(listener);
        if !grid.contains(position) {
            return None;
        }
        if !self.same(grid, listener) {
            self.runs = 0;
            self.last = Some((Arc::clone(grid), listener));
        }
        let at = CoordinateSpace3::canonical_at(position);
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
        self.simulator.simulate(
            &**grid,
            &params,
            std::slice::from_mut(&mut self.field),
            &self.pool,
        );
        // The average of the runs so far (Steam Audio's accumulation over frames).
        let n = self.runs as f32;
        if self.runs == 0 {
            self.accumulated.copy_from(&self.field);
        } else {
            self.accumulated.scale_in_place(n);
            self.accumulated.add_assign(&self.field);
            self.accumulated.scale_in_place(1.0 / (n + 1.0));
        }
        self.runs = (self.runs + 1).min(MAX_RUNS);

        let reverb = reverb_estimator::estimate(&self.accumulated, &self.air);
        self.reconstructor.reconstruct(
            &self.accumulated,
            &mut self.ir,
            &ReconstructionParams {
                kind: ReconstructionType::Linear,
                duration: TRANSITION_TIME,
                order: ORDER,
                air_absorption: &self.air,
                distance_attenuation_correction: None,
            },
        );
        let estimate = self.estimator.estimate(
            Some(&self.accumulated),
            &reverb,
            &mut self.ir,
            TRANSITION_TIME,
            OVERLAP_FRACTION,
            ORDER,
        );
        let size = ir_size(self.rate);
        let mut fir = fir
            .filter(|f| {
                f.num_channels() == NUM_CHANNELS
                    && f.num_blocks() == OverlapSaveFir::new(1, size, self.frame_size).num_blocks()
            })
            .unwrap_or_else(|| OverlapSaveFir::new(NUM_CHANNELS, size, self.frame_size));
        self.partitioner
            .partition(&self.ir, NUM_CHANNELS, self.ir.num_samples(), &mut fir);
        Some(Reflections {
            rate: self.rate,
            fir,
            reverb,
            eq: estimate.eq_coeffs,
            delay: estimate.delay,
        })
    }

    /// The impulse response of the last run (channel 0 is the omnidirectional one).
    pub fn impulse_response(&self) -> &ImpulseResponse {
        &self.ir
    }

    /// The energy field averaged over the runs so far.
    pub fn energy_field(&self) -> &EnergyField {
        &self.accumulated
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The simulation thread
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Default)]
struct Request {
    grid: Option<Arc<Voxels>>,
    listener: Option<[f64; 3]>,
}

static REQUEST: LazyLock<(Mutex<Request>, Condvar)> =
    LazyLock::new(|| (Mutex::new(Request::default()), Condvar::new()));
/// The latest reflections, for the audio thread.
static OUTPUT: Mutex<Option<Box<Reflections>>> = Mutex::new(None);
/// Impulse responses the audio thread is done with, for the next runs to reuse.
static SPARE: Mutex<Vec<OverlapSaveFir>> = Mutex::new(Vec::new());
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
        && spare.len() < 2
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
    let mut last_run: Option<Instant> = None;
    loop {
        if let Some(t) = last_run {
            let elapsed = t.elapsed();
            if elapsed < MIN_INTERVAL {
                std::thread::sleep(MIN_INTERVAL - elapsed);
            }
        }
        // Wait for a grid and a listener the simulation would do something with.
        let (grid, listener, rate) = {
            let (lock, signal) = &*REQUEST;
            let Ok(mut request) = lock.lock() else {
                return;
            };
            loop {
                let rate = RATE.load(Ordering::Relaxed);
                if rate != 0
                    && let (Some(grid), Some(listener)) = (&request.grid, request.listener)
                    && simulation
                        .as_ref()
                        .is_none_or(|s| s.rate() != rate || s.wants(grid, listener))
                {
                    break (Arc::clone(grid), listener, rate);
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
        last_run = Some(Instant::now());
        let fir = SPARE.lock().ok().and_then(|mut spare| spare.pop());
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            simulation.run(&grid, listener, fir)
        }));
        match result {
            Ok(Some(reflections)) => {
                if let Ok(mut output) = OUTPUT.lock()
                    && let Some(old) = output.replace(Box::new(reflections))
                {
                    drop(output);
                    recycle(old.fir);
                }
            }
            Ok(None) => {}
            Err(_) => {
                log::warn!("steam_audio: the reflection simulation failed; no reverb");
                return;
            }
        }
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
            last = simulation.run(grid, listener, None);
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
        assert!(simulation.run(&g, [500.0, 65.0, 0.0], None).is_none());
        assert!(simulation.run(&g, listener, None).is_some());
        assert!(simulation.wants(&g, listener));
        for _ in 0..MAX_RUNS {
            simulation.run(&g, listener, None);
        }
        assert!(!simulation.wants(&g, listener));
        assert!(simulation.wants(&g, [2.0, 65.6, 0.5]));
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
                simulation.run(&g, listener, None);
            }
            eprintln!("{name}: {:?} per run", started.elapsed() / 8);
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
                    std::slice::from_mut(&mut simulation.field),
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
