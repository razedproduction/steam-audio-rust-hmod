//! Sound paths around obstacles, found by Steam Audio's pathing in the blocks around the listener.
//!
//! A source behind a wall is heard through the wall ([`crate::acoustics`]: occlusion and transmission), and also
//! around it: through the doorway, round the corner, down the corridor. That is Steam Audio's *pathing*:
//!
//! 1. **Probes.** Over the grid of blocks the reverb is simulated in ([`Voxels`]), a probe stands every
//!    [`SPACING`] blocks, [`HEIGHT`] above every floor with room for a player (Steam Audio's uniform floor
//!    placement, done on the cells instead of with downward rays). At most [`MAX_PROBES`], the nearest to the
//!    grid's centre.
//! 2. **Bake.** [`PathBaker`] builds the graph of the probes that see each other (within [`VIS_RANGE`], rays through
//!    the cells) and the shortest path between every pair of probes. It runs on a thread of its own each time the
//!    grid changes ([`submit_grid`]), at most every [`MIN_INTERVAL`], and publishes the result ([`latest`]).
//! 3. **Paths.** For an occluded source, [`Paths::find`] asks [`PathSimulator`] for the baked paths from the probes
//!    around the source to those around the listener (Steam Audio's `findPaths`): where they arrive from, how long
//!    they are, and what bending round the corners takes off each band (the UTD deviation model).
//!
//! The game thread asks for paths with the sources' occlusion ([`PathAcoustics`] goes with
//! [`SourceAcoustics`](crate::acoustics::SourceAcoustics)), and the renderer plays them binaurally from where they
//! arrive.

use std::sync::{Arc, Condvar, LazyLock, Mutex, Once};
use std::time::{Duration, Instant};

use crate::baking::{BakedDataIdentifier, BakedDataType, BakedDataVariation};
use crate::bands::NUM_BANDS;
use crate::deviation::DeviationModel;
use crate::distance_attenuation::DistanceAttenuationModel;
use crate::geometry::Sphere;
use crate::math::Vector3;
use crate::pathing::{PathBakeParams, PathBaker, PathSimulationParams, PathSimulator};
use crate::probes::{ProbeBatch, ProbeNeighborhood};
use crate::reflections::Voxels;
use crate::thread_pool::ThreadPool;

/// Blocks between probes, along x and z.
pub const SPACING: i32 = 4;
/// Height of a probe above its floor (about a player's ears).
pub const HEIGHT: f32 = 1.5;
/// Radius of a probe's influence: the probes around a point are those within it.
const INFLUENCE: f32 = 1.5 * SPACING as f32;
/// Most probes baked: the bake keeps a path for every pair of them.
pub const MAX_PROBES: usize = 768;
/// Probes farther apart than this are not joined directly (blocks).
pub const VIS_RANGE: f32 = 16.0;
/// Paths longer than this are not baked (blocks).
const PATH_RANGE: f32 = 96.0;
const NUM_THREADS: usize = 2;
/// Shortest time between two bakes (longer when they take long: the thread works at most a quarter of the time).
pub const MIN_INTERVAL: Duration = Duration::from_millis(500);
/// The order-0 spherical harmonic: the simulator's one SH coefficient is this times the paths' total weight.
const Y00: f32 = 0.282_094_8;

/// How an occluded source is heard around what hides it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathAcoustics {
    /// Where the paths arrive from: unit direction from the listener, in world axes.
    pub direction: [f32; 3],
    /// Gain of the paths relative to the direct sound with nothing in the way: the share of the probes around the
    /// listener a path reaches (up to 1), times what the longer way loses in distance attenuation (set by the
    /// caller, which knows the source's attenuation; 1 from [`Paths::find`]).
    pub gain: f32,
    /// What the paths' bends take off each band (Steam Audio's deviation model; at most 1).
    pub eq: [f32; NUM_BANDS],
    /// Length of the way round (blocks), at least the straight distance.
    pub distance: f32,
}

impl PathAcoustics {
    /// Moves a fraction `rate` of the way to `target`.
    pub(crate) fn approach(&mut self, target: &Self, rate: f32) {
        let mix = |a: f32, b: f32| a + (b - a) * rate;
        let direction = Vector3::from_array(std::array::from_fn(|a| {
            mix(self.direction[a], target.direction[a])
        }))
        .normalize_or_zero();
        self.direction = if direction.length() > 0.5 {
            direction.to_array()
        } else {
            target.direction
        };
        self.gain = mix(self.gain, target.gain);
        for (e, &t) in self.eq.iter_mut().zip(&target.eq) {
            *e = mix(*e, t);
        }
        self.distance = mix(self.distance, target.distance);
    }
}

/// The identifier of the pathing layer of a batch.
fn identifier() -> BakedDataIdentifier {
    BakedDataIdentifier {
        variation: BakedDataVariation::Dynamic,
        data_type: BakedDataType::Pathing,
        endpoint_influence: Sphere::default(),
    }
}

/// Where probes stand in `grid` (local coordinates), [`HEIGHT`] above floors (a wall with two free cells over it) or
/// in the middle of a one-block gap (crawl spaces, windows): every [`SPACING`] blocks, and in the doorways between
/// them (a floor between two walls, open on the other axis), which the regular probes would miss. At most
/// [`MAX_PROBES`], the nearest to `center`.
pub fn probe_points(grid: &Voxels, center: Vector3) -> Vec<Vector3> {
    let size = grid.size().map(|s| s as i32);
    let wall = |x: i32, y: i32, z: i32| grid.is_wall_local([x, y, z]);
    // A probe over the wall at (x, y, z): `HEIGHT` above it with two free cells over it, or in the middle of a
    // single free cell under another wall (a crawl space, a window in a wall).
    let point = |x: i32, y: i32, z: i32| {
        if !wall(x, y, z) || wall(x, y + 1, z) {
            return None;
        }
        let height = if wall(x, y + 2, z) { 0.5 } else { HEIGHT };
        Some(Vector3::new(
            x as f32 + 0.5,
            (y + 1) as f32 + height,
            z as f32 + 0.5,
        ))
    };
    let mut points = Vec::new();
    for x in (SPACING / 2..size[0]).step_by(SPACING as usize) {
        for z in (SPACING / 2..size[2]).step_by(SPACING as usize) {
            for y in 0..size[1] - 2 {
                points.extend(point(x, y, z));
            }
        }
    }
    // Doorways: walls on both sides along one axis at foot level, free cells on both sides along the other. A
    // doorway probe close to another doorway probe is not needed (corridors would get one per block).
    const MIN_GAP: f32 = 2.5;
    let regular = points.len();
    for x in 1..size[0] - 1 {
        for z in 1..size[2] - 1 {
            for y in 0..size[1] - 2 {
                let Some(p) = point(x, y, z) else {
                    continue;
                };
                let f = y + 1;
                let walled_x = wall(x - 1, f, z) && wall(x + 1, f, z);
                let walled_z = wall(x, f, z - 1) && wall(x, f, z + 1);
                let open_x = !wall(x - 1, f, z) && !wall(x + 1, f, z);
                let open_z = !wall(x, f, z - 1) && !wall(x, f, z + 1);
                if !((walled_x && open_z) || (walled_z && open_x)) {
                    continue;
                }
                if points[regular..]
                    .iter()
                    .all(|q: &Vector3| (*q - p).length_squared() >= MIN_GAP * MIN_GAP)
                {
                    points.push(p);
                }
            }
        }
    }
    if points.len() > MAX_PROBES {
        points.sort_by(|a, b| {
            (*a - center)
                .length_squared()
                .total_cmp(&(*b - center).length_squared())
        });
        points.truncate(MAX_PROBES);
    }
    points
}

/// The paths baked in one grid.
#[derive(Debug)]
pub struct Paths {
    grid: Arc<Voxels>,
    batch: ProbeBatch,
    simulator: PathSimulator,
    /// The simulator's distance attenuation: none (the caller attenuates with the source's own model).
    attenuation: DistanceAttenuationModel,
    deviation: DeviationModel,
}

impl Paths {
    /// Places probes in `grid` and bakes the paths between them on the threads of `pool`. `None` if there are fewer
    /// than two probes.
    pub fn bake(grid: Arc<Voxels>, pool: &ThreadPool) -> Option<Self> {
        let center = Vector3::from_array(grid.size().map(|s| s as f32 / 2.0));
        let points = probe_points(&grid, center);
        if points.len() < 2 {
            return None;
        }
        let mut batch = ProbeBatch::new();
        for point in points {
            batch.add_probe(Sphere::new(point, INFLUENCE));
        }
        batch.commit();
        let down = -Vector3::Y_AXIS;
        PathBaker::new().bake(
            &*grid,
            &identifier(),
            &PathBakeParams {
                // Point-to-point visibility: the cells are exact, and probes stand where a player could.
                num_samples: 1,
                radius: 0.0,
                threshold: 0.0,
                vis_range: VIS_RANGE,
                vis_range_real_time: VIS_RANGE,
                path_range: PATH_RANGE,
                asymmetric_vis_range: false,
                down,
                prune_vis_graph: false,
            },
            pool,
            &mut batch,
            |_| {},
        );
        let simulator = PathSimulator::new(&batch, 1, false, down);
        Some(Self {
            grid,
            batch,
            simulator,
            attenuation: DistanceAttenuationModel::Callback(Arc::new(|_| 1.0)),
            deviation: DeviationModel::default(),
        })
    }

    /// The grid the paths were baked in.
    pub fn grid(&self) -> &Arc<Voxels> {
        &self.grid
    }

    pub fn num_probes(&self) -> usize {
        self.batch.num_probes()
    }

    /// The paths from `source` to `listener` (world positions) through the probes, as if the source were hidden
    /// (the caller asks for occluded sources). `None` when there is none: either is outside the grid or sees no
    /// probe, or no baked path joins their probes.
    pub fn find(&self, listener: [f64; 3], source: [f64; 3]) -> Option<PathAcoustics> {
        self.find_traced(listener, source, None)
    }

    /// Where the probes stand (world).
    pub fn probe_positions(&self) -> Vec<[f64; 3]> {
        self.batch
            .probes()
            .iter()
            .map(|p| self.grid.to_world(p.influence.center))
            .collect()
    }

    /// [`Paths::find`], and with `legs` the legs of the paths it found, probe to probe (world, each once), for the
    /// debug mode ([`crate::debug`]).
    pub fn find_traced(
        &self,
        listener: [f64; 3],
        source: [f64; 3],
        legs: Option<&mut Vec<([f64; 3], [f64; 3])>>,
    ) -> Option<PathAcoustics> {
        let listener = self.grid.free_point(self.grid.to_local(listener))?;
        let source = self.grid.free_point(self.grid.to_local(source))?;
        let batches = [&self.batch];
        let around = |point: Vector3| {
            let mut probes = ProbeNeighborhood::default();
            probes.resize(ProbeNeighborhood::MAX_PROBES_PER_BATCH);
            self.batch.influencing_probes(point, &mut probes, 0, 0);
            probes.check_occlusion(&*self.grid, point, &batches);
            probes.calc_weights(point, &batches);
            probes
        };
        let (source_probes, listener_probes) = (around(source), around(listener));
        let params = PathSimulationParams {
            source,
            listener,
            radius: 0.0,
            threshold: 0.0,
            vis_range: VIS_RANGE,
            order: 0,
            enable_validation: false,
            find_alternate_paths: false,
            simplify_paths: false,
            real_time_vis: false,
            distance_attenuation_model: &self.attenuation,
            deviation_model: &self.deviation,
            paths_from_all_source_probes: true,
            force_direct_occlusion: true,
        };
        let mut sh = [0.0f32; 1];
        // Without validation the simulator only reports each leg of the paths it takes, as visible.
        let grid = &self.grid;
        let mut record = legs.map(|legs| {
            move |a: Vector3, b: Vector3, _: bool| {
                let leg = (grid.to_world(a), grid.to_world(b));
                if !legs
                    .iter()
                    .any(|l| *l == leg || (l.0 == leg.1 && l.1 == leg.0))
                {
                    legs.push(leg);
                }
            }
        });
        let result = self.simulator.find_paths(
            &*self.grid,
            &batches,
            0,
            &source_probes,
            &listener_probes,
            &params,
            &mut sh,
            record
                .as_mut()
                .map(|r| r as &mut dyn FnMut(Vector3, Vector3, bool)),
        )?;
        // The paths' total weight: 1 when every pair of probes around the two is joined.
        let weight = sh[0] / Y00;
        if weight.is_nan() || weight <= 1e-3 || result.direction.length() < 0.5 {
            return None;
        }
        let straight = (source - listener).length();
        let ratio = (result.distance_ratio / weight).clamp(1e-3, 1.0);
        let eq = result.eq_gains.map(|g| (g / weight).clamp(0.0, 1.0));
        Some(PathAcoustics {
            direction: result.direction.to_array(),
            gain: weight.min(1.0),
            eq,
            distance: (straight / ratio).max(straight),
        })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The bake thread
// ---------------------------------------------------------------------------------------------------------------------

/// The latest grid submitted.
static REQUEST: LazyLock<(Mutex<Option<Arc<Voxels>>>, Condvar)> =
    LazyLock::new(|| (Mutex::new(None), Condvar::new()));
/// The latest paths baked.
static LATEST: Mutex<Option<Arc<Paths>>> = Mutex::new(None);
static STARTED: Once = Once::new();

/// Gives the bake a newly copied grid (game thread).
pub fn submit_grid(grid: &Arc<Voxels>) {
    start();
    let (lock, signal) = &*REQUEST;
    if let Ok(mut request) = lock.lock()
        && request.as_deref() != Some(&**grid)
    {
        *request = Some(Arc::clone(grid));
        signal.notify_one();
    }
}

/// The latest paths baked, if any (game thread; the lock is only held to swap them).
pub fn latest() -> Option<Arc<Paths>> {
    LATEST.lock().ok()?.clone()
}

fn start() {
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new()
            .name("steam-audio-pathing".into())
            .spawn(run_thread);
        if let Err(e) = spawned {
            log::warn!("steam_audio: cannot start the pathing bake ({e}); no sound paths");
        }
    });
}

fn run_thread() {
    let pool = ThreadPool::new(NUM_THREADS);
    let mut baked: Option<Arc<Voxels>> = None;
    let mut last_run: Option<(Instant, Duration)> = None;
    loop {
        // At least `MIN_INTERVAL` between bakes, and the thread busy at most a quarter of the time.
        if let Some((t, took)) = last_run {
            let wait = MIN_INTERVAL.max(took * 3);
            let elapsed = t.elapsed();
            if elapsed < wait {
                std::thread::sleep(wait - elapsed);
            }
        }
        let grid = {
            let (lock, signal) = &*REQUEST;
            let Ok(mut request) = lock.lock() else {
                return;
            };
            loop {
                if let Some(grid) = request.as_ref()
                    && baked
                        .as_ref()
                        .is_none_or(|b| !Arc::ptr_eq(b, grid) && **b != **grid)
                {
                    break Arc::clone(grid);
                }
                match signal.wait_timeout(request, Duration::from_millis(500)) {
                    Ok((r, _)) => request = r,
                    Err(_) => return,
                }
            }
        };
        let started = Instant::now();
        baked = Some(Arc::clone(&grid));
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Paths::bake(grid, &pool)));
        last_run = Some((started, started.elapsed()));
        match result {
            Ok(paths) => {
                crate::debug::publish_paths(
                    paths.as_ref().map_or(0, Paths::num_probes),
                    started.elapsed(),
                );
                if let Ok(mut latest) = LATEST.lock() {
                    *latest = paths.map(Arc::new);
                }
            }
            Err(_) => {
                log::warn!("steam_audio: the pathing bake failed; no sound paths");
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::Surface;
    use crate::reflections::{GRID_SIZE, GridBuilder};

    fn grid(listener: [f64; 3], f: impl FnMut([i32; 3]) -> Option<Surface>) -> Arc<Voxels> {
        Arc::new(Voxels::from_fn(
            GridBuilder::origin_for(listener),
            GRID_SIZE,
            f,
        ))
    }

    /// Ground below y = 64; two stone rooms of 9 × 4 × 9 (inside) side by side along x, joined by a doorway in the
    /// wall between them at z = `door_z` (two blocks high, one wide), or not joined.
    fn two_rooms(door_z: Option<i32>) -> impl FnMut([i32; 3]) -> Option<Surface> {
        move |c| {
            let [x, y, z] = c;
            if y < 64 {
                return Some(Surface::Rock);
            }
            let inside_x = (-9..=9).contains(&x);
            let inside_z = (-4..=4).contains(&z);
            let shell = (-10..=10).contains(&x) && (-5..=5).contains(&z) && y <= 68;
            if !shell {
                return None;
            }
            if y == 68 || !inside_x || !inside_z {
                return Some(Surface::Rock);
            }
            // The wall between the rooms, at x = 0.
            if x == 0 && !(door_z == Some(z) && y < 66) {
                return Some(Surface::Rock);
            }
            None
        }
    }

    #[test]
    fn probes_stand_on_floors_with_room_above() {
        let listener = [0.5, 65.6, 0.5];
        let flat = grid(listener, |c| (c[1] < 64).then_some(Surface::Gravel));
        let center = Vector3::from_array(GRID_SIZE.map(|s| s as f32 / 2.0));
        let points = probe_points(&flat, center);
        let columns = |s: usize| s.div_ceil(SPACING as usize);
        assert_eq!(points.len(), columns(GRID_SIZE[0]) * columns(GRID_SIZE[2]));
        let y = flat.to_local([0.0, 64.0 + f64::from(HEIGHT), 0.0]).y;
        assert!(
            points.iter().all(|p| (p.y - y).abs() < 1e-4),
            "{:?}",
            points[0]
        );
        // Solid rock: nowhere to stand.
        let rock = grid(listener, |_| Some(Surface::Rock));
        assert!(probe_points(&rock, center).is_empty());
    }

    #[test]
    fn a_hidden_source_is_heard_through_the_doorway() {
        let listener = [-4.5, 65.6, 0.5];
        let source = [4.5, 64.5, 0.5];
        let pool = ThreadPool::new(2);
        let paths = Paths::bake(grid(listener, two_rooms(Some(-3))), &pool).expect("probes");
        let path = paths
            .find(listener, source)
            .expect("a path through the doorway");
        eprintln!("{path:?} ({} probes)", paths.num_probes());
        // The doorway is at x = 0, z = -3: ahead-right of the listener towards -z, not straight at the source (+x).
        let d = Vector3::from_array(path.direction);
        assert!(d.x > 0.3 && d.z < -0.2, "{path:?}");
        let straight = ((source[0] - listener[0]).powi(2) + 1.0).sqrt() as f32;
        assert!(path.distance > straight, "{path:?}");
        assert!(path.distance < 2.0 * straight, "{path:?}");
        assert!(path.gain > 0.3 && path.gain <= 1.0, "{path:?}");
        // Bending round the doorway takes more off the high frequencies.
        assert!(path.eq[2] <= path.eq[0], "{path:?}");
        // The debug mode's legs: the same path, through probes in both rooms, and one leg through the doorway.
        let mut legs = Vec::new();
        assert_eq!(
            paths.find_traced(listener, source, Some(&mut legs)),
            Some(path)
        );
        assert!(!legs.is_empty());
        let crosses = |a: f64, b: f64| (a < 0.0) != (b < 0.0);
        assert!(
            legs.iter()
                .any(|(a, b)| crosses(a[0], b[0]) && a[2] < 0.0 && b[2] < 0.0),
            "{legs:?}"
        );
        let probes = paths.probe_positions();
        assert_eq!(probes.len(), paths.num_probes());
        assert!(
            legs.iter()
                .all(|(a, b)| probes.contains(a) && probes.contains(b))
        );

        // Without the doorway the rooms are sealed: no path.
        let sealed = Paths::bake(grid(listener, two_rooms(None)), &pool).expect("probes");
        assert!(sealed.find(listener, source).is_none());
    }

    #[test]
    fn a_window_and_a_crawl_space_join_the_rooms() {
        let listener = [-4.5, 65.6, 0.5];
        let source = [4.5, 64.5, 0.5];
        let pool = ThreadPool::new(2);
        // A one-block window in the wall between the rooms, a block above the floor.
        let mut sealed = two_rooms(None);
        let window = grid(
            listener,
            move |c| {
                if c == [0, 65, -3] { None } else { sealed(c) }
            },
        );
        let paths = Paths::bake(window, &pool).expect("probes");
        assert!(paths.find(listener, source).is_some(), "through the window");
        // A crawl space under the wall: a trench one block deep across it.
        let mut sealed = two_rooms(None);
        let crawl = grid(listener, move |c| {
            if c[1] == 63 && c[2] == -3 && (-1..=1).contains(&c[0]) {
                None
            } else {
                sealed(c)
            }
        });
        let paths = Paths::bake(crawl, &pool).expect("probes");
        assert!(
            paths.find(listener, source).is_some(),
            "through the crawl space"
        );
    }
    #[test]
    fn a_source_inside_its_block_is_heard_from_the_block_beside_it() {
        let listener = [-4.5, 65.6, 0.5];
        let mut rooms = two_rooms(Some(-3));
        // A note block at the source.
        let walled = grid(listener, move |c| {
            if c == [4, 64, 0] {
                Some(Surface::Wood)
            } else {
                rooms(c)
            }
        });
        let paths = Paths::bake(walled, &ThreadPool::new(2)).expect("probes");
        assert!(paths.find(listener, [4.5, 64.5, 0.5]).is_some());
        // Outside the grid: nothing.
        assert!(paths.find(listener, [500.5, 64.5, 0.5]).is_none());
    }

    #[test]
    fn the_thread_bakes_what_is_submitted() {
        let listener = [-4.5, 65.6, 0.5];
        let g = grid(listener, two_rooms(Some(-3)));
        submit_grid(&g);
        let started = Instant::now();
        while latest().is_none_or(|p| !Arc::ptr_eq(p.grid(), &g)) {
            assert!(started.elapsed() < Duration::from_secs(20), "no bake");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[test]
    #[ignore = "measurement"]
    fn measure_bake_time() {
        let listener = [0.5, 65.6, 0.5];
        let pool = ThreadPool::new(NUM_THREADS);
        // Open ground; the two rooms; a cave system with stacked floors (a 3D maze of tunnels).
        let maze = |c: [i32; 3]| {
            let [x, y, z] = c;
            let tunnel = (y.rem_euclid(6) < 3 && (x.rem_euclid(5) < 2 || z.rem_euclid(7) < 2))
                || (x.rem_euclid(11) == 0 && z.rem_euclid(11) == 0);
            (!tunnel).then_some(Surface::Rock)
        };
        for (name, g) in [
            (
                "field",
                grid(listener, |c| (c[1] < 64).then_some(Surface::Gravel)),
            ),
            ("rooms", grid(listener, two_rooms(Some(-3)))),
            ("maze", grid(listener, maze)),
        ] {
            let started = Instant::now();
            let paths = Paths::bake(g, &pool).expect("probes");
            let bake = started.elapsed();
            let started = Instant::now();
            let mut found = 0;
            for i in 0..32 {
                let source = [
                    f64::from(i % 8) * 3.0 - 10.0,
                    65.5,
                    f64::from(i / 8) * 4.0 - 6.0,
                ];
                found += usize::from(paths.find(listener, source).is_some());
            }
            eprintln!(
                "{name}: {} probes, bake {bake:?}, 32 queries {:?} ({found} paths)",
                paths.num_probes(),
                started.elapsed()
            );
        }
    }
}
