//! Reflections and reverb by Monte Carlo path tracing (upstream `reflection_simulator.{h,cpp}`): rays leave the
//! listener, bounce around the scene, and at every bounce the energy reaching each source is added to that source's
//! [`EnergyField`].
//!
//! Upstream has two simulators for its own ray tracer and custom scenes: [`ReflectionSimulator`] traces one ray at a
//! time, [`BatchedReflectionSimulator`] a batch of rays bounce by bounce, through the batched queries of
//! [`RayTracer`]. Both draw from a random number generator per thread, so results depend on the seed, on the number of
//! threads and on which thread runs which batch of rays (risk R2).

use std::borrow::BorrowMut;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

use crate::bands::NUM_BANDS;
use crate::directivity::Directivity;
use crate::geometry::{Ray, Sphere};
use crate::job_graph::JobGraph;
use crate::math::{CoordinateSpace3, PI, Vector3, cpp_max};
use crate::propagation_medium::SPEED_OF_SOUND;
use crate::random::Mt19937;
use crate::sampling;
use crate::scene::{Hit, Material, RayTracer};
use crate::sh;
use crate::thread_pool::ThreadPool;

use super::{DirectSimulator, EnergyField};

/// Offset of a bounce's starting point from the surface, along the normal (upstream `kHitSurfaceOffset`).
pub const HIT_SURFACE_OFFSET: f32 = 1e-2;
/// Exponent of the specular lobe (upstream `kSpecularExponent`).
pub const SPECULAR_EXPONENT: f32 = 1e2;
/// Radius of the sphere around a source that ends a ray after its first bounce (upstream `kSourceRadius`).
pub const SOURCE_RADIUS: f32 = 0.1;
/// Radius of the sphere around the listener: hits closer than this are ignored, and the sphere ends a ray after its
/// first bounce (upstream `kListenerRadius`).
pub const LISTENER_RADIUS: f32 = 0.1;

/// Number of rays per job of [`ReflectionSimulator`] (upstream `ReflectionSimulator::kRayBatchSize`).
const RAY_BATCH_SIZE: usize = 32;

/// Sizes a reflection simulator is built for (the constructor arguments of upstream's simulators).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReflectionSimulatorSettings {
    /// Most rays traced per simulation. Their directions are fixed when the simulator is built.
    pub max_num_rays: usize,
    /// Number of directions a diffuse bounce chooses from.
    pub num_diffuse_samples: usize,
    /// Length of the energy fields accumulated per thread, in seconds.
    pub max_duration: f32,
    /// Ambisonics order of the energy fields accumulated per thread.
    pub max_order: usize,
    /// Most sources simulated at once; more are ignored.
    pub max_num_sources: usize,
    /// Number of threads, each with its own random number generator and energy fields. A [`ThreadPool`] passed to
    /// `simulate` may have at most this many threads.
    pub num_threads: usize,
}

/// Inputs of a simulation (the arguments of upstream `simulate`, for one listener).
#[derive(Debug, Clone, Copy)]
pub struct ReflectionSimulationParams<'a> {
    pub sources: &'a [CoordinateSpace3],
    /// One per source.
    pub directivities: &'a [Directivity],
    pub listener: CoordinateSpace3,
    /// At most the simulator's `max_num_rays`.
    pub num_rays: usize,
    pub num_bounces: usize,
    /// Rays stop bouncing once they have travelled this long, in seconds (checked before each bounce).
    pub duration: f32,
    /// Not used, as upstream: every channel of the simulator's maximum order is accumulated.
    pub order: usize,
    /// Sources closer than this to a hit point receive nothing from it, and the energy reaching a source falls off
    /// with distance from this distance on.
    pub irradiance_min_distance: f32,
}

/// What both simulators compute once, when they are built.
#[derive(Debug, Clone)]
struct Samples {
    /// Direction of each ray leaving the listener.
    listener_samples: Vec<Vector3>,
    /// Directions a diffuse bounce chooses from, around −z.
    diffuse_samples: Vec<Vector3>,
    /// Spherical harmonics of each listener sample, up to the maximum order: `max_num_rays × num_coeffs`.
    listener_coeffs: Vec<f32>,
    num_coeffs: usize,
}

impl Samples {
    fn new(settings: &ReflectionSimulatorSettings) -> Self {
        let mut listener_samples = vec![Vector3::ZERO; settings.max_num_rays];
        let mut diffuse_samples = vec![Vector3::ZERO; settings.num_diffuse_samples];
        sampling::sphere_samples(&mut listener_samples);
        sampling::hemisphere_samples(&mut diffuse_samples);

        let num_coeffs = sh::num_coeffs_for_order(settings.max_order);
        let mut listener_coeffs = Vec::with_capacity(settings.max_num_rays * num_coeffs);
        for &sample in &listener_samples {
            for l in 0..=settings.max_order {
                let l_signed = l as i32;
                for m in -l_signed..=l_signed {
                    listener_coeffs.push(sh::evaluate(l, m, sample));
                }
            }
        }

        Self {
            listener_samples,
            diffuse_samples,
            listener_coeffs,
            num_coeffs,
        }
    }

    fn coeffs(&self, ray: usize) -> &[f32] {
        &self.listener_coeffs[ray * self.num_coeffs..(ray + 1) * self.num_coeffs]
    }
}

/// The inputs of one simulation, shared by its jobs.
struct Frame<'a> {
    samples: &'a Samples,
    sources: &'a [CoordinateSpace3],
    directivities: &'a [Directivity],
    listener: CoordinateSpace3,
    num_bounces: usize,
    duration: f32,
    irradiance_min_distance: f32,
    /// Weight of one ray: the sphere's solid angle shared among the rays.
    scalar: f32,
}

impl Frame<'_> {
    /// Whether a hit at `hit_distance` along `ray` ends the ray: nothing hit, travelled too far, too close to the
    /// listener, or (after the first bounce) passing through a source's or the listener's sphere first.
    fn escaped(&self, bounce: usize, ray: &Ray, hit: &Hit, accum_distance: f32) -> bool {
        if !hit.is_valid() {
            return true;
        }

        if accum_distance > self.duration * SPEED_OF_SOUND {
            return true;
        }

        if hit.distance <= LISTENER_RADIUS {
            return true;
        }

        if bounce > 0 {
            for source in self.sources {
                if (self.listener.origin - source.origin).length() > SOURCE_RADIUS {
                    let source_hit_distance =
                        ray.intersect_sphere(&Sphere::new(source.origin, SOURCE_RADIUS));
                    if 0.0 <= source_hit_distance && source_hit_distance < hit.distance {
                        return true;
                    }
                }
            }

            let listener_hit_distance =
                ray.intersect_sphere(&Sphere::new(self.listener.origin, LISTENER_RADIUS));
            if 0.0 <= listener_hit_distance && listener_hit_distance < hit.distance {
                return true;
            }
        }

        false
    }

    /// The energy per band reaching source `source_index` from a hit whose shadow ray towards it (of length
    /// `hit_to_source_distance`) is clear, and its delay relative to the direct path.
    #[allow(clippy::too_many_arguments)]
    fn shade(
        &self,
        source_index: usize,
        ray: &Ray,
        hit: &Hit,
        hit_point: Vector3,
        shadow_direction: Vector3,
        hit_to_source_distance: f32,
        accum_energy: &[f32; NUM_BANDS],
        accum_distance: f32,
    ) -> ([f32; NUM_BANDS], f32) {
        let material = material(hit);
        let source = &self.sources[source_index];

        let diffuse_term =
            (1.0 / PI) * material.scattering * cpp_max(hit.normal.dot(shadow_direction), 0.0);
        let half_vector = ((shadow_direction - ray.direction) * 0.5).normalize_or_zero();
        let specular_term = ((SPECULAR_EXPONENT + 2.0) / (8.0 * PI))
            * (1.0 - material.scattering)
            * half_vector.dot(hit.normal).powf(SPECULAR_EXPONENT);
        let attenuation = 1.0 / cpp_max(hit_to_source_distance, self.irradiance_min_distance);
        let distance_term = (1.0 / (4.0 * PI)) * (attenuation * attenuation);
        let directivity_term = self.directivities[source_index].evaluate_at(hit_point, source);
        let frequency_independent_term =
            self.scalar * distance_term * directivity_term * (diffuse_term + specular_term);

        let energy = std::array::from_fn(|j| {
            frequency_independent_term * (1.0 - material.absorption[j]) * accum_energy[j]
        });

        let distance = accum_distance + hit.distance + hit_to_source_distance;
        let delay = (distance / SPEED_OF_SOUND)
            - DirectSimulator::direct_path_delay(self.listener.origin, source.origin);

        (energy, delay)
    }

    /// Continues `ray` from `hit_point`: diffusely (a random hemisphere sample around the normal) with the
    /// material's scattering probability, specularly otherwise.
    #[allow(clippy::too_many_arguments)]
    fn bounce(
        &self,
        hit: &Hit,
        hit_point: Vector3,
        rng: &mut Mt19937,
        ray: &mut Ray,
        accum_energy: &mut [f32; NUM_BANDS],
        accum_distance: &mut f32,
    ) {
        let material = material(hit);
        for (e, a) in accum_energy.iter_mut().zip(material.absorption) {
            *e *= 1.0 - a;
        }

        *accum_distance += hit.distance;

        ray.origin = hit_point;

        if rng.uniform_f32(0.0, 1.0) < material.scattering {
            let diffuse_samples = &self.samples.diffuse_samples;
            let index = rng.uniform_int() as usize % diffuse_samples.len();
            ray.direction =
                sampling::transform_hemisphere_sample(diffuse_samples[index], hit.normal);
        } else {
            ray.direction = Vector3::reflect(ray.direction, hit.normal);
        }
    }

    /// Adds `energy`, arriving along listener sample `ray` after `delay`, to `field`; nothing if the delay falls
    /// outside the field.
    fn accumulate(
        &self,
        field: &mut EnergyField,
        ray: usize,
        energy: &[f32; NUM_BANDS],
        delay: f32,
    ) {
        let Some(bin) = bin_index(delay, field.num_bins()) else {
            return;
        };

        let coeffs = self.samples.coeffs(ray);
        for (channel, &coeff) in coeffs.iter().enumerate().take(field.num_channels()) {
            for (band, e) in energy.iter().enumerate() {
                field.band_mut(channel, band)[bin] += coeff * e;
            }
        }
    }
}

/// `floorf(delay / kBinDuration)` converted to `int` as MSVC does (NaN and out-of-range values give `INT_MIN`), if it
/// lies in `0..num_bins`.
fn bin_index(delay: f32, num_bins: usize) -> Option<usize> {
    let bin = (delay / EnergyField::BIN_DURATION).floor();
    (bin >= 0.0 && bin < num_bins as f32).then_some(bin as usize)
}

fn material(hit: &Hit) -> Material {
    hit.material
        .expect("a hit reported by the scene has a material")
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// What `simulate` does before adding jobs, common to both simulators: checks the inputs, resets the output and
/// per-thread fields, and returns the number of sources to simulate.
fn begin<S: AsMut<[EnergyField]>, F: BorrowMut<EnergyField>>(
    settings: &ReflectionSimulatorSettings,
    params: &ReflectionSimulationParams<'_>,
    energy_fields: &mut [F],
    thread_fields: &mut [Mutex<S>],
    pool: &ThreadPool,
) -> usize {
    assert_eq!(
        params.sources.len(),
        params.directivities.len(),
        "one directivity per source"
    );
    assert!(
        params.num_rays <= settings.max_num_rays,
        "{} rays requested from a simulator built for {}",
        params.num_rays,
        settings.max_num_rays
    );
    assert!(
        pool.num_threads() <= settings.num_threads,
        "a pool of {} threads for a simulator built for {}",
        pool.num_threads(),
        settings.num_threads
    );

    // Sources past the maximum are ignored (upstream logs a warning).
    let num_sources = params.sources.len().min(settings.max_num_sources);
    assert!(
        energy_fields.len() >= num_sources,
        "one energy field per source"
    );

    for (i, field) in energy_fields.iter_mut().enumerate().take(num_sources) {
        field.borrow_mut().reset();
        for state in thread_fields.iter_mut() {
            state.get_mut().unwrap_or_else(|e| e.into_inner()).as_mut()[i].reset();
        }
    }

    num_sources
}

/// Adds every thread's fields to the outputs (upstream `finalizeJob`).
fn finalize<S: AsMut<[EnergyField]>, F: BorrowMut<EnergyField>>(
    energy_fields: &mut [F],
    thread_fields: &mut [Mutex<S>],
    num_sources: usize,
) {
    for (i, field) in energy_fields.iter_mut().enumerate().take(num_sources) {
        for state in thread_fields.iter_mut() {
            field
                .borrow_mut()
                .add_assign(&state.get_mut().unwrap_or_else(|e| e.into_inner()).as_mut()[i]);
        }
    }
}

/// Adds a job per batch of rays to a graph, runs it on `pool`, and returns whether every job ran (they do unless the
/// pool is cancelled), in which case upstream's last job finalizes.
fn run_jobs<S: Send>(
    num_rays: usize,
    batch_size: usize,
    thread_states: &[Mutex<S>],
    pool: &ThreadPool,
    job: impl Fn(&mut S, usize, usize, &AtomicBool) + Sync,
) -> bool {
    let jobs_completed = AtomicUsize::new(0);
    let mut graph = JobGraph::new();
    for start in (0..num_rays).step_by(batch_size) {
        let end = num_rays.min(start + batch_size);
        let (job, jobs_completed) = (&job, &jobs_completed);
        graph.add_job(move |thread_id, cancel| {
            job(&mut lock(&thread_states[thread_id]), start, end, cancel);
            jobs_completed.fetch_add(1, Ordering::SeqCst);
        });
    }

    let num_jobs = graph.num_jobs();
    pool.process(&mut graph);
    num_jobs > 0 && jobs_completed.load(Ordering::SeqCst) == num_jobs
}

// ---------------------------------------------------------------------------------------------------------------------
// ReflectionSimulator
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ThreadState {
    rng: Mt19937,
    energy_fields: Vec<EnergyField>,
}

impl AsMut<[EnergyField]> for ThreadState {
    fn as_mut(&mut self) -> &mut [EnergyField] {
        &mut self.energy_fields
    }
}

/// Traces one ray at a time, in jobs of 32 rays (upstream `ReflectionSimulator`, used for upstream's own ray
/// tracer).
#[derive(Debug)]
pub struct ReflectionSimulator {
    settings: ReflectionSimulatorSettings,
    samples: Samples,
    thread_states: Vec<Mutex<ThreadState>>,
}

impl ReflectionSimulator {
    /// A simulator whose threads' random number generators all start from `seed` (upstream seeds them all with
    /// `time(NULL)`).
    pub fn new(settings: &ReflectionSimulatorSettings, seed: u32) -> Self {
        let thread_states = (0..settings.num_threads)
            .map(|_| {
                Mutex::new(ThreadState {
                    rng: Mt19937::new(seed),
                    energy_fields: (0..settings.max_num_sources)
                        .map(|_| EnergyField::new(settings.max_duration, settings.max_order))
                        .collect(),
                })
            })
            .collect();

        Self {
            settings: *settings,
            samples: Samples::new(settings),
            thread_states,
        }
    }

    pub fn settings(&self) -> &ReflectionSimulatorSettings {
        &self.settings
    }

    /// Simulates reflections from `params.sources` to the listener on `pool`, writing one energy field per source
    /// into `energy_fields` (upstream `simulate` with energy fields, followed by `ThreadPool::process`). Only the
    /// channels and bins both an output field and the simulator's fields have are written.
    ///
    /// If the pool is cancelled before every job has run, the outputs are left silent. The fields may be owned
    /// (`&mut [EnergyField]`) or borrowed from elsewhere (`&mut [&mut EnergyField]`).
    ///
    /// # Panics
    ///
    /// If `params` has more rays than the simulator or not one directivity per source, `energy_fields` has fewer
    /// fields than the sources simulated, `pool` has more threads than the simulator, or a diffuse bounce happens
    /// with no diffuse samples.
    pub fn simulate<F: BorrowMut<EnergyField>>(
        &mut self,
        scene: &dyn RayTracer,
        params: &ReflectionSimulationParams<'_>,
        energy_fields: &mut [F],
        pool: &ThreadPool,
    ) {
        let num_sources = begin(
            &self.settings,
            params,
            energy_fields,
            &mut self.thread_states,
            pool,
        );

        let frame = Frame {
            samples: &self.samples,
            sources: &params.sources[..num_sources],
            directivities: &params.directivities[..num_sources],
            listener: params.listener,
            num_bounces: params.num_bounces,
            duration: params.duration,
            irradiance_min_distance: params.irradiance_min_distance,
            scalar: (4.0 * PI) / params.num_rays as f32,
        };

        let completed = run_jobs(
            params.num_rays,
            RAY_BATCH_SIZE,
            &self.thread_states,
            pool,
            |state, start, end, cancel| simulate_job(scene, &frame, state, start, end, cancel),
        );

        if completed {
            finalize(energy_fields, &mut self.thread_states, num_sources);
        }
    }
}

fn simulate_job(
    scene: &dyn RayTracer,
    frame: &Frame<'_>,
    state: &mut ThreadState,
    start: usize,
    end: usize,
    cancel: &AtomicBool,
) {
    let cancelled = || cancel.load(Ordering::Relaxed);

    for i in start..end {
        let mut ray = Ray::new(frame.listener.origin, frame.samples.listener_samples[i]);
        let mut accum_energy = [1.0f32; NUM_BANDS];
        let mut accum_distance = 0.0f32;

        for j in 0..frame.num_bounces {
            let Some((hit, hit_point)) = trace(scene, frame, &ray, j, accum_distance) else {
                break;
            };

            if cancelled() {
                return;
            }

            for k in 0..frame.sources.len() {
                let hit_to_source = frame.sources[k].origin - hit_point;
                if hit.normal.dot(hit_to_source) < 0.0 {
                    continue;
                }

                let hit_to_source_distance = hit_to_source.length();
                if hit_to_source_distance <= frame.irradiance_min_distance {
                    continue;
                }

                let shadow_ray = Ray::new(hit_point, hit_to_source / hit_to_source_distance);
                if scene.any_hit(&shadow_ray, 0.0, hit_to_source_distance) {
                    continue;
                }

                let (energy, delay) = frame.shade(
                    k,
                    &ray,
                    &hit,
                    hit_point,
                    shadow_ray.direction,
                    hit_to_source_distance,
                    &accum_energy,
                    accum_distance,
                );
                frame.accumulate(&mut state.energy_fields[k], i, &energy, delay);

                if cancelled() {
                    return;
                }
            }

            if j + 1 < frame.num_bounces {
                frame.bounce(
                    &hit,
                    hit_point,
                    &mut state.rng,
                    &mut ray,
                    &mut accum_energy,
                    &mut accum_distance,
                );

                if cancelled() {
                    return;
                }
            }
        }
    }
}

/// The hit ending this segment of `ray`, with its normal facing the ray, and the point the next bounce starts from;
/// `None` if the ray ends here.
fn trace(
    scene: &dyn RayTracer,
    frame: &Frame<'_>,
    ray: &Ray,
    bounce: usize,
    accum_distance: f32,
) -> Option<(Hit, Vector3)> {
    let mut hit = scene.closest_hit(ray, 0.0, f32::INFINITY);
    if frame.escaped(bounce, ray, &hit, accum_distance) {
        return None;
    }

    if hit.normal.dot(ray.direction) > 0.0 {
        hit.normal *= -1.0;
    }

    let hit_point = ray.point_at_distance(hit.distance) + HIT_SURFACE_OFFSET * hit.normal;
    Some((hit, hit_point))
}

// ---------------------------------------------------------------------------------------------------------------------
// BatchedReflectionSimulator
// ---------------------------------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BatchedThreadState {
    rays: Vec<Ray>,
    min_distances: Vec<f32>,
    max_distances: Vec<f32>,
    hits: Vec<Hit>,
    shadow_rays: Vec<Ray>,
    shadow_ray_min_distances: Vec<f32>,
    shadow_ray_max_distances: Vec<f32>,
    occluded: Vec<bool>,
    escaped: Vec<bool>,
    hit_points: Vec<Vector3>,
    energy: Vec<[f32; NUM_BANDS]>,
    delay: Vec<f32>,
    accum_energy: Vec<[f32; NUM_BANDS]>,
    accum_distance: Vec<f32>,
    rng: Mt19937,
    energy_fields: Vec<EnergyField>,
}

impl AsMut<[EnergyField]> for BatchedThreadState {
    fn as_mut(&mut self) -> &mut [EnergyField] {
        &mut self.energy_fields
    }
}

/// Traces batches of rays bounce by bounce through [`RayTracer::closest_hits`] and [`RayTracer::any_hits`]
/// (upstream `BatchedReflectionSimulator`, used for custom scenes). One job traces one batch.
///
/// Its results differ from [`ReflectionSimulator`]'s only in the order random numbers are drawn: here each bounce
/// draws for every ray of the batch in turn.
#[derive(Debug)]
pub struct BatchedReflectionSimulator {
    settings: ReflectionSimulatorSettings,
    ray_batch_size: usize,
    samples: Samples,
    thread_states: Vec<Mutex<BatchedThreadState>>,
}

impl BatchedReflectionSimulator {
    /// A simulator tracing `ray_batch_size` rays per job, whose threads' random number generators all start from
    /// `seed` (upstream seeds them all with `time(NULL)`).
    ///
    /// # Panics
    ///
    /// If `ray_batch_size` is 0.
    pub fn new(settings: &ReflectionSimulatorSettings, ray_batch_size: usize, seed: u32) -> Self {
        assert!(ray_batch_size > 0, "a ray batch holds at least one ray");
        let n = ray_batch_size;
        let thread_states = (0..settings.num_threads)
            .map(|_| {
                Mutex::new(BatchedThreadState {
                    rays: vec![Ray::new(Vector3::ZERO, Vector3::ZERO); n],
                    min_distances: vec![0.0; n],
                    max_distances: vec![0.0; n],
                    hits: vec![Hit::default(); n],
                    shadow_rays: vec![Ray::new(Vector3::ZERO, Vector3::ZERO); n],
                    shadow_ray_min_distances: vec![0.0; n],
                    shadow_ray_max_distances: vec![0.0; n],
                    occluded: vec![false; n],
                    escaped: vec![false; n],
                    hit_points: vec![Vector3::ZERO; n],
                    energy: vec![[0.0; NUM_BANDS]; n],
                    delay: vec![0.0; n],
                    accum_energy: vec![[0.0; NUM_BANDS]; n],
                    accum_distance: vec![0.0; n],
                    rng: Mt19937::new(seed),
                    energy_fields: (0..settings.max_num_sources)
                        .map(|_| EnergyField::new(settings.max_duration, settings.max_order))
                        .collect(),
                })
            })
            .collect();

        Self {
            settings: *settings,
            ray_batch_size,
            samples: Samples::new(settings),
            thread_states,
        }
    }

    pub fn settings(&self) -> &ReflectionSimulatorSettings {
        &self.settings
    }

    pub fn ray_batch_size(&self) -> usize {
        self.ray_batch_size
    }

    /// As [`ReflectionSimulator::simulate`].
    ///
    /// # Panics
    ///
    /// As [`ReflectionSimulator::simulate`].
    pub fn simulate<F: BorrowMut<EnergyField>>(
        &mut self,
        scene: &dyn RayTracer,
        params: &ReflectionSimulationParams<'_>,
        energy_fields: &mut [F],
        pool: &ThreadPool,
    ) {
        let num_sources = begin(
            &self.settings,
            params,
            energy_fields,
            &mut self.thread_states,
            pool,
        );

        let frame = Frame {
            samples: &self.samples,
            sources: &params.sources[..num_sources],
            directivities: &params.directivities[..num_sources],
            listener: params.listener,
            num_bounces: params.num_bounces,
            duration: params.duration,
            irradiance_min_distance: params.irradiance_min_distance,
            scalar: (4.0 * PI) / params.num_rays as f32,
        };

        let completed = run_jobs(
            params.num_rays,
            self.ray_batch_size,
            &self.thread_states,
            pool,
            |state, start, end, cancel| {
                batched_simulate_job(scene, &frame, state, start, end, cancel)
            },
        );

        if completed {
            finalize(energy_fields, &mut self.thread_states, num_sources);
        }
    }
}

fn batched_simulate_job(
    scene: &dyn RayTracer,
    frame: &Frame<'_>,
    state: &mut BatchedThreadState,
    start: usize,
    end: usize,
    cancel: &AtomicBool,
) {
    let cancelled = || cancel.load(Ordering::Relaxed);
    let n = end - start;

    // Upstream `reset`, over the whole batch arrays.
    state.accum_energy.fill([1.0; NUM_BANDS]);
    state.accum_distance.fill(0.0);
    state.escaped.fill(false);

    for (ray, &direction) in state.rays[..n]
        .iter_mut()
        .zip(&frame.samples.listener_samples[start..end])
    {
        *ray = Ray::new(frame.listener.origin, direction);
    }

    for i in 0..frame.num_bounces {
        if !batched_trace(scene, frame, state, i, n) {
            break;
        }

        if cancelled() {
            return;
        }

        for j in 0..frame.sources.len() {
            batched_shade(scene, frame, state, j, n);

            if cancelled() {
                return;
            }

            for k in 0..n {
                if state.occluded[k] {
                    continue;
                }
                frame.accumulate(
                    &mut state.energy_fields[j],
                    start + k,
                    &state.energy[k],
                    state.delay[k],
                );
            }

            if cancelled() {
                return;
            }
        }

        if i + 1 < frame.num_bounces {
            for k in 0..n {
                if state.escaped[k] {
                    continue;
                }
                frame.bounce(
                    &state.hits[k],
                    state.hit_points[k],
                    &mut state.rng,
                    &mut state.rays[k],
                    &mut state.accum_energy[k],
                    &mut state.accum_distance[k],
                );
            }

            if cancelled() {
                return;
            }
        }
    }
}

/// Traces the first `n` rays of the batch and marks those that end; `false` if they all do. A ray that has ended
/// stays marked, but is traced again at every bounce, as upstream.
fn batched_trace(
    scene: &dyn RayTracer,
    frame: &Frame<'_>,
    state: &mut BatchedThreadState,
    bounce: usize,
    n: usize,
) -> bool {
    state.min_distances[..n].fill(0.0);
    state.max_distances[..n].fill(f32::INFINITY);

    scene.closest_hits(
        &state.rays[..n],
        &state.min_distances[..n],
        &state.max_distances[..n],
        &mut state.hits[..n],
    );

    let mut num_escaped = 0;

    for k in 0..n {
        let ray = &state.rays[k];
        let hit = &mut state.hits[k];
        if frame.escaped(bounce, ray, hit, state.accum_distance[k]) {
            state.escaped[k] = true;
            num_escaped += 1;
        } else {
            if hit.normal.dot(ray.direction) > 0.0 {
                hit.normal *= -1.0;
            }

            state.hit_points[k] =
                ray.point_at_distance(hit.distance) + HIT_SURFACE_OFFSET * hit.normal;
        }
    }

    num_escaped < n
}

/// Shadow rays from the first `n` hits towards source `source_index`, and the energy and delay of those that reach it
/// (`occluded` is set for the others, including rays that have ended).
fn batched_shade(
    scene: &dyn RayTracer,
    frame: &Frame<'_>,
    state: &mut BatchedThreadState,
    source_index: usize,
    n: usize,
) {
    let source = frame.sources[source_index].origin;

    for k in 0..n {
        let hit_to_source = source - state.hit_points[k];
        let hit_to_source_distance = hit_to_source.length();

        if state.escaped[k]
            || hit_to_source_distance <= frame.irradiance_min_distance
            || state.hits[k].normal.dot(hit_to_source) < 0.0
        {
            // Never traced: `any_hits` reports a negative maximum distance as occluded.
            state.shadow_rays[k] = Ray::new(Vector3::ZERO, Vector3::X_AXIS);
            state.shadow_ray_min_distances[k] = 0.0;
            state.shadow_ray_max_distances[k] = -1.0;
        } else {
            state.shadow_rays[k] =
                Ray::new(state.hit_points[k], hit_to_source / hit_to_source_distance);
            state.shadow_ray_min_distances[k] = 0.0;
            state.shadow_ray_max_distances[k] = hit_to_source_distance;
        }
    }

    scene.any_hits(
        &state.shadow_rays[..n],
        &state.shadow_ray_min_distances[..n],
        &state.shadow_ray_max_distances[..n],
        &mut state.occluded[..n],
    );

    for k in 0..n {
        if state.occluded[k] {
            continue;
        }

        let (energy, delay) = frame.shade(
            source_index,
            &state.rays[k],
            &state.hits[k],
            state.hit_points[k],
            state.shadow_rays[k].direction,
            state.shadow_ray_max_distances[k],
            &state.accum_energy[k],
            state.accum_distance[k],
        );
        state.energy[k] = energy;
        state.delay[k] = delay;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// AnyReflectionSimulator
// ---------------------------------------------------------------------------------------------------------------------

/// Either reflection simulator (upstream `IReflectionSimulator`, as created by `ReflectionSimulatorFactory`):
/// [`ReflectionSimulator`] for upstream's own ray tracer, [`BatchedReflectionSimulator`] for custom ray tracers.
#[derive(Debug)]
pub enum AnyReflectionSimulator {
    Default(ReflectionSimulator),
    Batched(BatchedReflectionSimulator),
}

impl AnyReflectionSimulator {
    pub fn settings(&self) -> &ReflectionSimulatorSettings {
        match self {
            Self::Default(simulator) => simulator.settings(),
            Self::Batched(simulator) => simulator.settings(),
        }
    }

    /// As [`ReflectionSimulator::simulate`].
    ///
    /// # Panics
    ///
    /// As [`ReflectionSimulator::simulate`].
    pub fn simulate<F: BorrowMut<EnergyField>>(
        &mut self,
        scene: &dyn RayTracer,
        params: &ReflectionSimulationParams<'_>,
        energy_fields: &mut [F],
        pool: &ThreadPool,
    ) {
        match self {
            Self::Default(simulator) => simulator.simulate(scene, params, energy_fields, pool),
            Self::Batched(simulator) => simulator.simulate(scene, params, energy_fields, pool),
        }
    }
}

impl From<ReflectionSimulator> for AnyReflectionSimulator {
    fn from(simulator: ReflectionSimulator) -> Self {
        Self::Default(simulator)
    }
}

impl From<BatchedReflectionSimulator> for AnyReflectionSimulator {
    fn from(simulator: BatchedReflectionSimulator) -> Self {
        Self::Batched(simulator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins() {
        assert_eq!(bin_index(0.0, 10), Some(0));
        assert_eq!(bin_index(0.0999, 10), Some(9));
        assert_eq!(bin_index(0.1, 10), None);
        assert_eq!(bin_index(-1e-9, 10), None);
        assert_eq!(bin_index(f32::NAN, 10), None);
        assert_eq!(bin_index(f32::INFINITY, 10), None);
    }
}
