//! Runs the simulations of a set of sources against one listener (upstream `simulation_manager.{h,cpp}`,
//! `SimulationManager`): direct sound, reflections simulated in real time or looked up from baked probes, turned into
//! impulse responses, parametric reverb or both, and paths around obstacles through baked probe paths.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::baking::{BakedDataType, BakedDataVariation};
use crate::bands::NUM_BANDS;
use crate::directivity::Directivity;
use crate::distance_attenuation::DistanceAttenuationModel;
use crate::effects::OverlapSavePartitioner;
use crate::math::{CoordinateSpace3, Vector3};
use crate::pathing::{PathSimulationParams, PathSimulator};
use crate::probes::{ProbeBatch, ProbeManager, ProbeNeighborhood};
use crate::random::CRand;
use crate::scene::{RayTracer, Scene};
use crate::sh;
use crate::thread_pool::ThreadPool;

use super::baked_reflection;
use super::data::ReflectionSimulationState;
use super::reverb_estimator;
use super::{
    AnyReflectionSimulator, BatchedReflectionSimulator, DirectSimulationParams, DirectSimulator,
    EnergyField, HybridReverbEstimator, IndirectEffectType, ReconstructionParams,
    ReconstructionType, Reconstructor, ReflectionSimulationParams, ReflectionSimulator,
    ReflectionSimulatorSettings, SimulationData,
};

/// Which ray tracer the scene uses (upstream `SceneType`; Embree and Radeon Rays are out of scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SceneType {
    /// The crate's own [`Scene`]. Reflections are traced one ray at a time, and a scene whose version has not changed
    /// lets energy fields accumulate over frames.
    Default,
    /// A user's [`RayTracer`]. Reflections are traced in batches, and the scene always counts as changed.
    Custom,
}

/// The scene a [`SimulationManager`] simulates in, of its [`SceneType`].
pub enum SimulationScene {
    Default(Scene),
    Custom(Box<dyn RayTracer>),
}

impl SimulationScene {
    pub fn scene_type(&self) -> SceneType {
        match self {
            Self::Default(_) => SceneType::Default,
            Self::Custom(_) => SceneType::Custom,
        }
    }

    pub fn ray_tracer(&self) -> &dyn RayTracer {
        match self {
            Self::Default(scene) => scene,
            Self::Custom(ray_tracer) => ray_tracer.as_ref(),
        }
    }
}

impl std::fmt::Debug for SimulationScene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Default(scene) => f.debug_tuple("Default").field(scene).finish(),
            Self::Custom(_) => f.debug_tuple("Custom").finish_non_exhaustive(),
        }
    }
}

/// How a [`SimulationManager`] and its sources are built (the constructor arguments of upstream `SimulationManager`
/// and `SimulationData`, without OpenCL, Radeon Rays and TrueAudioNext).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimulationSettings {
    pub enable_direct: bool,
    pub enable_indirect: bool,
    pub enable_pathing: bool,
    pub scene_type: SceneType,
    pub indirect_type: IndirectEffectType,
    pub max_num_occlusion_samples: usize,
    pub max_num_rays: usize,
    pub num_diffuse_samples: usize,
    /// Longest energy field and impulse response, in seconds.
    pub max_duration: f32,
    pub max_order: usize,
    /// Most sources whose reflections are simulated in real time at once; more are ignored.
    pub max_num_sources: usize,
    pub num_threads: usize,
    /// Rays per job for a [`SceneType::Custom`] scene.
    pub ray_batch_size: usize,
    /// Pathing: points per probe for visibility tests (see
    /// [`ProbeVisibilityTester::new`](crate::pathing::ProbeVisibilityTester::new)).
    pub num_vis_samples: usize,
    /// Pathing: measure visibility ranges perpendicular to `down`.
    pub asymmetric_vis_range: bool,
    pub down: Vector3,
    pub sampling_rate: u32,
    pub frame_size: usize,
}

/// Inputs of direct simulation shared by every source (upstream `SharedDirectSimulationInputs`).
#[derive(Debug, Clone, Copy, Default)]
pub struct SharedDirectSimulationInputs {
    pub listener: CoordinateSpace3,
}

/// Inputs of reflection simulation shared by every source (upstream `SharedReflectionSimulationInputs`).
#[derive(Debug, Clone, Copy)]
pub struct SharedReflectionSimulationInputs {
    pub listener: CoordinateSpace3,
    pub num_rays: usize,
    pub num_bounces: usize,
    /// Seconds of reflections simulated and reconstructed, at most the maximum duration.
    pub duration: f32,
    /// Ambisonics order of the impulse responses, at most the maximum order.
    pub order: usize,
    pub irradiance_min_distance: f32,
    pub reconstruction_type: ReconstructionType,
}

/// Called with the two ends of each ray traced to validate a baked path, and whether it was occluded (upstream
/// `ValidationRayVisualizationCallback` with its user data).
pub type ValidationRayVisualizationCallback = Arc<dyn Fn(Vector3, Vector3, bool) + Send + Sync>;

/// Inputs of pathing simulation shared by every source (upstream `SharedPathingSimulationInputs`).
#[derive(Clone, Default)]
pub struct SharedPathingSimulationInputs {
    pub listener: CoordinateSpace3,
    pub vis_callback: Option<ValidationRayVisualizationCallback>,
}

impl std::fmt::Debug for SharedPathingSimulationInputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedPathingSimulationInputs")
            .field("listener", &self.listener)
            .field("vis_callback", &self.vis_callback.as_ref().map(Arc::as_ptr))
            .finish()
    }
}

/// A shared, lockable source, as added to a [`SimulationManager`].
pub type SharedSimulationData = Arc<Mutex<SimulationData>>;

fn lock(source: &SharedSimulationData) -> MutexGuard<'_, SimulationData> {
    source.lock().unwrap_or_else(|e| e.into_inner())
}

/// Simulates sources against a listener (upstream `SimulationManager`).
///
/// Sources and probe batches added or removed take effect at the next [`commit`](Self::commit). Sources are shared
/// with the code that sets their inputs and reads their outputs, as `Arc<Mutex<SimulationData>>`; a simulation locks
/// every committed source until it returns.
#[derive(Debug)]
pub struct SimulationManager {
    settings: SimulationSettings,
    scene: Option<SimulationScene>,
    probe_manager: ProbeManager,
    direct_simulator: Option<DirectSimulator>,
    reflection_simulator: Option<AnyReflectionSimulator>,
    reconstructor: Option<Reconstructor>,
    hybrid_reverb_estimator: Option<HybridReverbEstimator>,
    partitioner: Option<OverlapSavePartitioner>,
    thread_pool: Option<ThreadPool>,
    shared_direct: SharedDirectSimulationInputs,
    shared_reflection: SharedReflectionSimulationInputs,
    shared_pathing: SharedPathingSimulationInputs,
    prev_listener: CoordinateSpace3,
    pending_sources: Vec<SharedSimulationData>,
    sources: Vec<SharedSimulationData>,
    /// A path simulator per probe batch (compared by identity), with pathing enabled.
    pending_path_simulators: Vec<(Arc<ProbeBatch>, Arc<PathSimulator>)>,
    path_simulators: Vec<(Arc<ProbeBatch>, Arc<PathSimulator>)>,
    enable_probe_caching_for_missing_probes: bool,
    enable_paths_from_all_source_probes: bool,
    /// Version of the scene when real-time reflections were last simulated.
    scene_version: u32,

    // Scratch space.
    real_time_sources: Vec<CoordinateSpace3>,
    real_time_directivities: Vec<Directivity>,
    listener_probes: ProbeNeighborhood,
    source_probes: ProbeNeighborhood,
    unique_batches: Vec<usize>,
    pathing_source_probes: ProbeNeighborhood,
    pathing_listener_probes: ProbeNeighborhood,
}

impl SimulationManager {
    /// A manager with `settings`. The reflection simulator's random number generators start from `seed` (upstream
    /// seeds them with the clock, risk R2), and a hybrid reverb estimator draws its reverb's delay lengths from
    /// `rand` (see [`ReverbEffect::new`](crate::effects::ReverbEffect::new)).
    ///
    /// # Panics
    ///
    /// If indirect simulation is enabled with no threads, or a custom scene with a ray batch size of 0.
    pub fn new(settings: &SimulationSettings, seed: u32, rand: &mut CRand) -> Self {
        let s = settings;
        let mut manager = Self {
            settings: *s,
            scene: None,
            probe_manager: ProbeManager::new(),
            direct_simulator: s
                .enable_direct
                .then(|| DirectSimulator::new(s.max_num_occlusion_samples)),
            reflection_simulator: None,
            reconstructor: None,
            hybrid_reverb_estimator: None,
            partitioner: None,
            thread_pool: None,
            shared_direct: SharedDirectSimulationInputs::default(),
            shared_reflection: SharedReflectionSimulationInputs {
                listener: CoordinateSpace3::default(),
                num_rays: s.max_num_rays,
                num_bounces: 0,
                duration: s.max_duration,
                order: s.max_order,
                irradiance_min_distance: 1.0,
                reconstruction_type: ReconstructionType::Linear,
            },
            shared_pathing: SharedPathingSimulationInputs::default(),
            prev_listener: CoordinateSpace3::default(),
            pending_sources: Vec::new(),
            sources: Vec::new(),
            pending_path_simulators: Vec::new(),
            path_simulators: Vec::new(),
            enable_probe_caching_for_missing_probes: false,
            enable_paths_from_all_source_probes: false,
            scene_version: 0,
            real_time_sources: Vec::new(),
            real_time_directivities: Vec::new(),
            listener_probes: ProbeNeighborhood::default(),
            source_probes: ProbeNeighborhood::default(),
            unique_batches: Vec::new(),
            pathing_source_probes: ProbeNeighborhood::default(),
            pathing_listener_probes: ProbeNeighborhood::default(),
        };

        if s.enable_indirect {
            let simulator_settings = ReflectionSimulatorSettings {
                max_num_rays: s.max_num_rays,
                num_diffuse_samples: s.num_diffuse_samples,
                max_duration: s.max_duration,
                max_order: s.max_order,
                max_num_sources: s.max_num_sources,
                num_threads: s.num_threads,
            };
            manager.reflection_simulator = Some(match s.scene_type {
                SceneType::Default => ReflectionSimulator::new(&simulator_settings, seed).into(),
                SceneType::Custom => {
                    BatchedReflectionSimulator::new(&simulator_settings, s.ray_batch_size, seed)
                        .into()
                }
            });

            if s.indirect_type != IndirectEffectType::Parametric {
                manager.reconstructor = Some(Reconstructor::new(s.max_duration, s.sampling_rate));
            }

            if s.indirect_type == IndirectEffectType::Hybrid {
                manager.hybrid_reverb_estimator = Some(HybridReverbEstimator::new(
                    s.max_duration,
                    s.sampling_rate,
                    s.frame_size,
                    rand,
                ));
            }

            if s.indirect_type != IndirectEffectType::Parametric {
                manager.partitioner = Some(OverlapSavePartitioner::new(s.frame_size));
            }

            manager.thread_pool = Some(ThreadPool::new(s.num_threads));
        }

        manager
    }

    pub fn settings(&self) -> &SimulationSettings {
        &self.settings
    }

    pub fn scene(&self) -> Option<&SimulationScene> {
        self.scene.as_ref()
    }

    /// The scene, to change and commit it between simulations.
    pub fn scene_mut(&mut self) -> Option<&mut SimulationScene> {
        self.scene.as_mut()
    }

    /// Sets the scene to simulate in (upstream `scene() = ...`).
    ///
    /// # Panics
    ///
    /// If the scene is not of the manager's [`SceneType`].
    pub fn set_scene(&mut self, scene: Option<SimulationScene>) {
        if let Some(scene) = &scene {
            assert_eq!(
                scene.scene_type(),
                self.settings.scene_type,
                "scene of another type than the manager's"
            );
        }
        self.scene = scene;
    }

    /// Upstream's global `SimulationManager::sEnableProbeCachingForMissingProbes` (off by default): when set, a
    /// baked source with no valid probe around it keeps its previous outputs instead of being looked up.
    pub fn set_enable_probe_caching_for_missing_probes(&mut self, enable: bool) {
        self.enable_probe_caching_for_missing_probes = enable;
    }

    /// Upstream's global `PathSimulator::sEnablePathsFromAllSourceProbes` (off by default): see
    /// [`PathSimulationParams::paths_from_all_source_probes`].
    pub fn set_enable_paths_from_all_source_probes(&mut self, enable: bool) {
        self.enable_paths_from_all_source_probes = enable;
    }

    pub fn shared_direct_inputs(&self) -> &SharedDirectSimulationInputs {
        &self.shared_direct
    }

    pub fn set_shared_direct_inputs(&mut self, inputs: SharedDirectSimulationInputs) {
        self.shared_direct = inputs;
    }

    pub fn shared_reflection_inputs(&self) -> &SharedReflectionSimulationInputs {
        &self.shared_reflection
    }

    pub fn set_shared_reflection_inputs(&mut self, inputs: SharedReflectionSimulationInputs) {
        self.shared_reflection = inputs;
    }

    pub fn shared_pathing_inputs(&self) -> &SharedPathingSimulationInputs {
        &self.shared_pathing
    }

    pub fn set_shared_pathing_inputs(&mut self, inputs: SharedPathingSimulationInputs) {
        self.shared_pathing = inputs;
    }

    /// The committed probe batches, which baked sources are looked up in.
    pub fn probe_manager(&self) -> &ProbeManager {
        &self.probe_manager
    }

    /// Adds a batch at the next commit. With pathing enabled, the batch gets a new path simulator for its probes as
    /// they are now (replacing any it had).
    pub fn add_probe_batch(&mut self, batch: Arc<ProbeBatch>) {
        if self.settings.enable_pathing {
            let s = &self.settings;
            let simulator = Arc::new(PathSimulator::new(
                &batch,
                s.num_vis_samples,
                s.asymmetric_vis_range,
                s.down,
            ));
            match self
                .pending_path_simulators
                .iter_mut()
                .find(|(b, _)| Arc::ptr_eq(b, &batch))
            {
                Some(entry) => entry.1 = simulator,
                None => self
                    .pending_path_simulators
                    .push((Arc::clone(&batch), simulator)),
            }
        }

        self.probe_manager.add_probe_batch(batch);
    }

    pub fn remove_probe_batch(&mut self, batch: &Arc<ProbeBatch>) {
        self.probe_manager.remove_probe_batch(batch);
        self.pending_path_simulators
            .retain(|(b, _)| !Arc::ptr_eq(b, batch));
    }

    /// The committed sources, in the order they were added.
    pub fn sources(&self) -> &[SharedSimulationData] {
        &self.sources
    }

    /// Adds a source at the next commit. Adding a source that is already pending does nothing (upstream simulates
    /// it twice; here that would lock it twice).
    pub fn add_source(&mut self, source: SharedSimulationData) {
        if !self.pending_sources.iter().any(|s| Arc::ptr_eq(s, &source)) {
            self.pending_sources.push(source);
        }
    }

    /// Removes a source (compared by identity) at the next commit.
    pub fn remove_source(&mut self, source: &SharedSimulationData) {
        self.pending_sources.retain(|s| !Arc::ptr_eq(s, source));
    }

    /// Applies the probe batches and sources added and removed since the last commit.
    pub fn commit(&mut self) {
        self.probe_manager.commit();
        self.path_simulators
            .clone_from(&self.pending_path_simulators);
        self.sources.clone_from(&self.pending_sources);
    }

    /// Simulates the direct sound of every committed source (upstream `simulateDirect()`).
    ///
    /// # Panics
    ///
    /// If direct simulation is not enabled, or a source is locked by the calling thread.
    pub fn simulate_direct(&mut self) {
        for i in 0..self.sources.len() {
            let source = Arc::clone(&self.sources[i]);
            self.simulate_direct_source(&mut lock(&source));
        }
    }

    /// Simulates the direct sound of one source, which need not be committed (upstream `simulateDirect(source)`).
    ///
    /// # Panics
    ///
    /// If direct simulation is not enabled.
    pub fn simulate_direct_source(&self, source: &mut SimulationData) {
        let simulator = self
            .direct_simulator
            .as_ref()
            .expect("direct simulation is enabled");
        let inputs = &source.direct_inputs;
        source.direct_path = simulator.simulate(
            self.scene.as_ref().map(SimulationScene::ray_tracer),
            &DirectSimulationParams {
                flags: inputs.flags,
                source: inputs.source,
                listener: self.shared_direct.listener,
                distance_attenuation_model: &inputs.distance_attenuation_model,
                air_absorption_model: &inputs.air_absorption_model,
                directivity: &inputs.directivity,
                occlusion_type: inputs.occlusion_type,
                occlusion_radius: inputs.occlusion_radius,
                num_occlusion_samples: inputs.num_occlusion_samples,
                num_transmission_rays: inputs.num_transmission_rays,
            },
        );
    }

    /// Simulates the reflections of every committed source with reflections enabled (upstream `simulateIndirect`):
    /// real-time sources are simulated (averaging the energy over frames while the source, listener and scene stay
    /// the same), baked sources are looked up; then, depending on the indirect effect type, impulse responses are
    /// reconstructed and published to each source's triple buffer, reverbs are estimated, and hybrid reverb is
    /// prepared.
    ///
    /// # Panics
    ///
    /// If indirect simulation is not enabled, there is no scene, a source with reflections enabled was created
    /// without indirect simulation, a source is locked by the calling thread, or a simulator panics.
    pub fn simulate_indirect(&mut self) {
        assert!(
            self.reflection_simulator.is_some(),
            "indirect simulation is enabled"
        );
        assert!(self.scene.is_some(), "a scene to simulate reflections in");

        let order = self.shared_reflection.order;
        let num_channels = sh::num_coeffs_for_order(order);
        let num_samples =
            (self.shared_reflection.duration * self.settings.sampling_rate as f32).ceil() as usize;

        let sources = self.sources.clone();
        let mut guards: Vec<MutexGuard<'_, SimulationData>> = sources.iter().map(lock).collect();

        for source in &mut guards {
            if let Some(state) = source.reflection_state.as_mut() {
                state.valid_simulation_data = true;
            }
        }

        self.simulate_real_time_reflections(&mut guards);
        self.lookup_baked_reflections(&mut guards);

        let indirect_type = self.settings.indirect_type;
        if indirect_type != IndirectEffectType::Parametric {
            self.reconstruct_impulse_responses(&mut guards, num_samples);
        }

        if matches!(
            indirect_type,
            IndirectEffectType::Parametric | IndirectEffectType::Hybrid
        ) {
            estimate_reverb(&mut guards);
        }

        if indirect_type == IndirectEffectType::Hybrid {
            self.estimate_hybrid_reverb(&mut guards);
        }

        self.partition_impulse_responses(&mut guards, num_channels, num_samples);
    }

    /// Simulates the paths of every committed source with pathing enabled (upstream `simulatePathing()`), from the
    /// probes of its batch around it to those around the listener. The listener's probes are looked up again only
    /// when a source uses another batch than the source before it.
    ///
    /// # Panics
    ///
    /// If there is no scene, a source with pathing enabled has no probe batch or one that was not added to the
    /// manager with pathing enabled, a source's order exceeds the maximum order (or it was created without pathing),
    /// a batch has not been committed, or a source is locked by the calling thread.
    pub fn simulate_pathing(&mut self) {
        let scene = self
            .scene
            .as_ref()
            .expect("a scene to simulate paths in")
            .ray_tracer();
        let listener = self.shared_pathing.listener.origin;

        let source_probes = &mut self.pathing_source_probes;
        let listener_probes = &mut self.pathing_listener_probes;
        source_probes.resize(ProbeNeighborhood::MAX_PROBES_PER_BATCH);
        listener_probes.resize(ProbeNeighborhood::MAX_PROBES_PER_BATCH);

        let mut prev_listener_batch: Option<*const ProbeBatch> = None;

        for source in &self.sources {
            let mut source = lock(source);
            if !source.pathing_inputs.enabled {
                continue;
            }

            let (batch, simulator) = path_simulator(&self.path_simulators, &source.pathing_inputs);
            let origin = source.pathing_inputs.source.origin;

            source_probes.reset();
            find_probes(scene, batch, origin, source_probes);

            if prev_listener_batch != Some(Arc::as_ptr(batch)) {
                prev_listener_batch = Some(Arc::as_ptr(batch));

                listener_probes.reset();
                find_probes(scene, batch, listener, listener_probes);
            }

            simulate_source_pathing(
                scene,
                batch,
                simulator,
                source_probes,
                listener_probes,
                &self.shared_pathing,
                self.enable_paths_from_all_source_probes,
                &mut source,
            );
        }
    }

    /// Simulates the paths of one source, which need not be committed (upstream `simulatePathing(source)`).
    ///
    /// # Panics
    ///
    /// As [`simulate_pathing`](Self::simulate_pathing).
    pub fn simulate_pathing_source(&mut self, source: &mut SimulationData) {
        let source_probes = &mut self.pathing_source_probes;
        let listener_probes = &mut self.pathing_listener_probes;
        for probes in [&mut *source_probes, &mut *listener_probes] {
            if probes.num_probes() != ProbeNeighborhood::MAX_PROBES_PER_BATCH {
                probes.resize(ProbeNeighborhood::MAX_PROBES_PER_BATCH);
            } else {
                probes.reset();
            }
        }

        if !source.pathing_inputs.enabled {
            return;
        }

        let scene = self
            .scene
            .as_ref()
            .expect("a scene to simulate paths in")
            .ray_tracer();
        let (batch, simulator) = path_simulator(&self.path_simulators, &source.pathing_inputs);

        find_probes(
            scene,
            batch,
            source.pathing_inputs.source.origin,
            source_probes,
        );
        find_probes(
            scene,
            batch,
            self.shared_pathing.listener.origin,
            listener_probes,
        );

        simulate_source_pathing(
            scene,
            batch,
            simulator,
            source_probes,
            listener_probes,
            &self.shared_pathing,
            self.enable_paths_from_all_source_probes,
            source,
        );
    }

    fn has_listener_changed(&self) -> bool {
        (self.shared_reflection.listener.origin - self.prev_listener.origin).length() > 1e-4
    }

    /// Whether the scene changed since real-time reflections were last simulated. A custom scene always counts as
    /// changed (upstream cannot ask a user's ray tracer).
    fn has_scene_changed(&self) -> bool {
        match self.scene.as_ref() {
            Some(SimulationScene::Default(scene)) => scene.version() != self.scene_version,
            _ => true,
        }
    }

    fn reset_scene_changed(&mut self) {
        if let Some(SimulationScene::Default(scene)) = self.scene.as_ref() {
            self.scene_version = scene.version();
        }
    }

    fn simulate_real_time_reflections(&mut self, sources: &mut [MutexGuard<'_, SimulationData>]) {
        self.real_time_sources.clear();
        self.real_time_directivities.clear();

        let listener_changed = self.has_listener_changed();
        let scene_changed = self.has_scene_changed();

        // Whether each source's simulation goes straight into its accumulated field (`None`: not simulated here).
        let mut into_accum: Vec<Option<bool>> = Vec::with_capacity(sources.len());
        for source in sources.iter_mut() {
            let inputs = &source.reflection_inputs;
            if !inputs.enabled || inputs.baked {
                into_accum.push(None);
                continue;
            }

            self.real_time_sources.push(inputs.source);
            self.real_time_directivities
                .push(inputs.directivity.clone());

            let changed = listener_changed || source.has_source_changed() || scene_changed;
            if changed {
                state_of(&mut source.reflection_state).num_frames_accumulated = 0;
            }
            into_accum.push(Some(changed));
        }

        if self.real_time_sources.is_empty() {
            return;
        }

        let mut fields: Vec<&mut EnergyField> = sources
            .iter_mut()
            .zip(&into_accum)
            .filter_map(|(source, into_accum)| {
                let state = source.reflection_state.as_mut()?;
                match into_accum {
                    Some(true) => Some(&mut state.accum_energy_field),
                    Some(false) => Some(&mut state.energy_field),
                    None => None,
                }
            })
            .collect();

        let shared = &self.shared_reflection;
        let scene = self
            .scene
            .as_ref()
            .expect("a scene to simulate reflections in")
            .ray_tracer();
        self.reflection_simulator
            .as_mut()
            .expect("indirect simulation is enabled")
            .simulate(
                scene,
                &ReflectionSimulationParams {
                    sources: &self.real_time_sources,
                    directivities: &self.real_time_directivities,
                    listener: shared.listener,
                    num_rays: shared.num_rays,
                    num_bounces: shared.num_bounces,
                    duration: shared.duration,
                    order: shared.order,
                    irradiance_min_distance: shared.irradiance_min_distance,
                },
                &mut fields,
                self.thread_pool
                    .as_ref()
                    .expect("indirect simulation is enabled"),
            );
        drop(fields);

        self.accumulate_energy_fields(sources);
    }

    fn accumulate_energy_fields(&mut self, sources: &mut [MutexGuard<'_, SimulationData>]) {
        for source in sources.iter_mut() {
            if !source.reflection_inputs.enabled || source.reflection_inputs.baked {
                continue;
            }

            let source = &mut **source;
            let state = state_of(&mut source.reflection_state);
            let n = state.num_frames_accumulated;
            if n > 0 {
                let accum = &mut state.accum_energy_field;
                accum.scale_in_place(n as f32);
                accum.add_assign(&state.energy_field);
                accum.scale_in_place(1.0 / (1.0 + n as f32));
            }

            state.num_frames_accumulated += 1;

            state.prev_source = source.reflection_inputs.source;
            state.prev_directivity = source.reflection_inputs.directivity.clone();
        }

        self.prev_listener = self.shared_reflection.listener;

        self.reset_scene_changed();
    }

    fn lookup_baked_reflections(&mut self, sources: &mut [MutexGuard<'_, SimulationData>]) {
        let batches = self.probe_manager.probe_batches();
        let scene = self
            .scene
            .as_ref()
            .expect("a scene to simulate reflections in")
            .ray_tracer();
        let listener = self.shared_reflection.listener.origin;

        self.probe_manager
            .influencing_probes(listener, &mut self.listener_probes);
        self.listener_probes
            .check_occlusion(scene, listener, batches);
        self.listener_probes.calc_weights(listener, batches);

        // Upstream points at the listener's neighbourhood, and from the first static-listener source on at the
        // source's: later sources of other variations keep looking up the last static-listener source's probes.
        let mut use_source_probes = false;

        for source in sources.iter_mut() {
            let inputs = &source.reflection_inputs;
            if !inputs.enabled {
                continue;
            }
            if !inputs.baked || inputs.baked_data_identifier.data_type != BakedDataType::Reflections
            {
                continue;
            }

            let identifier = inputs.baked_data_identifier;
            if identifier.variation == BakedDataVariation::StaticListener {
                let origin = inputs.source.origin;
                self.probe_manager
                    .influencing_probes(origin, &mut self.source_probes);
                self.source_probes.check_occlusion(scene, origin, batches);
                self.source_probes.calc_weights(origin, batches);
                use_source_probes = true;
            }

            let probes = if use_source_probes {
                &self.source_probes
            } else {
                &self.listener_probes
            };

            let source = &mut **source;
            let state = state_of(&mut source.reflection_state);
            state.valid_simulation_data = if self.enable_probe_caching_for_missing_probes {
                probes.has_valid_probes()
            } else {
                true
            };
            if !state.valid_simulation_data {
                continue;
            }

            baked_reflection::find_unique_probe_batches(probes, batches, &mut self.unique_batches);

            let indirect_type = self.settings.indirect_type;
            if indirect_type != IndirectEffectType::Parametric {
                baked_reflection::lookup_energy_field(
                    &identifier,
                    probes,
                    batches,
                    &self.unique_batches,
                    &mut state.accum_energy_field,
                );
            }

            if matches!(
                indirect_type,
                IndirectEffectType::Parametric | IndirectEffectType::Hybrid
            ) {
                baked_reflection::lookup_reverb(
                    &identifier,
                    probes,
                    batches,
                    &self.unique_batches,
                    &mut source.reflection_outputs.reverb,
                );
            }
        }
    }

    /// Makes the distance attenuation correction curves that changed (upstream `generateDistanceCorrectionCurves`),
    /// then reconstructs every enabled source's impulse response (upstream `reconstructImpulseResponses`).
    fn reconstruct_impulse_responses(
        &mut self,
        sources: &mut [MutexGuard<'_, SimulationData>],
        num_samples: usize,
    ) {
        let default_model = DistanceAttenuationModel::default();
        let shared = self.shared_reflection;
        let reconstructor = self
            .reconstructor
            .as_mut()
            .expect("a reconstructor for impulse responses");

        for source in sources.iter_mut() {
            if !source.reflection_inputs.enabled {
                continue;
            }

            let source = &mut **source;
            let inputs = &mut source.reflection_inputs;
            let state = state_of(&mut source.reflection_state);

            if state.prev_distance_attenuation_model != inputs.distance_attenuation_model
                || inputs.distance_attenuation_model_dirty
            {
                DistanceAttenuationModel::correction_curve(
                    &default_model,
                    &inputs.distance_attenuation_model,
                    self.settings.sampling_rate,
                    &mut state.distance_attenuation_correction_curve[..num_samples],
                );

                inputs.distance_attenuation_model_dirty = false;
                state.prev_distance_attenuation_model = inputs.distance_attenuation_model.clone();

                // From now on the curve is always applied to this source.
                state.apply_distance_attenuation_correction_curve = true;
            }

            let correction = (state.apply_distance_attenuation_correction_curve
                && state.valid_simulation_data)
                .then_some(state.distance_attenuation_correction_curve.as_slice());

            reconstructor.reconstruct(
                &state.accum_energy_field,
                state
                    .impulse_response
                    .as_mut()
                    .expect("an impulse response for convolution"),
                &ReconstructionParams {
                    kind: shared.reconstruction_type,
                    duration: shared.duration,
                    order: shared.order,
                    air_absorption: &inputs.air_absorption_model,
                    distance_attenuation_correction: correction,
                },
            );
        }
    }

    fn estimate_hybrid_reverb(&mut self, sources: &mut [MutexGuard<'_, SimulationData>]) {
        let order = self.shared_reflection.order;
        let estimator = self
            .hybrid_reverb_estimator
            .as_mut()
            .expect("a hybrid reverb estimator");

        for source in sources.iter_mut() {
            if !source.reflection_inputs.enabled {
                continue;
            }

            let source = &mut **source;
            let state = state_of(&mut source.reflection_state);
            if !state.valid_simulation_data {
                continue;
            }

            let estimate = estimator.estimate(
                Some(&state.accum_energy_field),
                &source.reflection_outputs.reverb,
                state
                    .impulse_response
                    .as_mut()
                    .expect("an impulse response for hybrid reverb"),
                source.reflection_inputs.transition_time,
                source.reflection_inputs.overlap_fraction,
                order,
            );
            source.reflection_outputs.hybrid_eq = estimate.eq_coeffs;
            source.reflection_outputs.hybrid_delay = estimate.delay;
        }
    }

    fn partition_impulse_responses(
        &mut self,
        sources: &mut [MutexGuard<'_, SimulationData>],
        num_channels: usize,
        num_samples: usize,
    ) {
        let Some(partitioner) = self.partitioner.as_mut() else {
            return;
        };

        for source in sources.iter_mut() {
            if !source.reflection_inputs.enabled {
                continue;
            }

            let source = &mut **source;
            let state = state_of(&mut source.reflection_state);
            if !state.valid_simulation_data {
                continue;
            }

            let writer = state
                .overlap_save_fir
                .as_mut()
                .expect("a triple buffer for convolution");
            partitioner.partition(
                state
                    .impulse_response
                    .as_ref()
                    .expect("an impulse response for convolution"),
                num_channels,
                num_samples,
                &mut writer.write_buffer(),
            );
            writer.commit_write_buffer();

            source.reflection_outputs.num_channels = num_channels;
            source.reflection_outputs.num_samples = num_samples;
        }
    }
}

/// The batch of a source's pathing inputs and its path simulator.
fn path_simulator<'a>(
    simulators: &'a [(Arc<ProbeBatch>, Arc<PathSimulator>)],
    inputs: &super::PathingSimulationInputs,
) -> (&'a Arc<ProbeBatch>, &'a PathSimulator) {
    let batch = inputs.probes.as_ref().expect("a probe batch for pathing");
    let (batch, simulator) = simulators
        .iter()
        .find(|(b, _)| Arc::ptr_eq(b, batch))
        .expect("the source's probe batch was added to a manager with pathing enabled");
    (batch, simulator)
}

/// Fills `probes` (of [`MAX_PROBES_PER_BATCH`](ProbeNeighborhood::MAX_PROBES_PER_BATCH) entries) with the probes of
/// `batch` around `point` that it sees, weighted. The entries refer to the batch as index 0.
fn find_probes(
    scene: &dyn RayTracer,
    batch: &ProbeBatch,
    point: Vector3,
    probes: &mut ProbeNeighborhood,
) {
    batch.influencing_probes(point, probes, 0, 0);
    probes.check_occlusion(scene, point, &[batch]);
    probes.calc_weights(point, &[batch]);
}

/// Upstream's `findPaths` call and the copy of its results into the source's outputs.
#[allow(clippy::too_many_arguments)]
fn simulate_source_pathing(
    scene: &dyn RayTracer,
    batch: &ProbeBatch,
    simulator: &PathSimulator,
    source_probes: &ProbeNeighborhood,
    listener_probes: &ProbeNeighborhood,
    shared: &SharedPathingSimulationInputs,
    paths_from_all_source_probes: bool,
    source: &mut SimulationData,
) {
    let inputs = &source.pathing_inputs;
    let params = PathSimulationParams {
        source: inputs.source.origin,
        listener: shared.listener.origin,
        radius: inputs.vis_radius,
        threshold: inputs.vis_threshold,
        vis_range: inputs.vis_range,
        order: inputs.order,
        enable_validation: inputs.enable_validation,
        find_alternate_paths: inputs.find_alternate_paths,
        simplify_paths: inputs.simplify_paths,
        real_time_vis: inputs.real_time_vis,
        distance_attenuation_model: &inputs.distance_attenuation_model,
        deviation_model: &inputs.deviation_model,
        paths_from_all_source_probes,
        force_direct_occlusion: false,
    };

    let mut callback = shared
        .vis_callback
        .as_ref()
        .map(|callback| move |from, to, occluded| callback(from, to, occluded));

    let outputs = &mut source.pathing_outputs;
    let result = simulator.find_paths(
        scene,
        &[batch],
        0,
        source_probes,
        listener_probes,
        &params,
        &mut outputs.sh,
        callback
            .as_mut()
            .map(|c| c as &mut dyn FnMut(Vector3, Vector3, bool)),
    );

    if let Some(result) = result {
        outputs.eq = result.eq_gains;
        outputs.direction = result.direction;
        outputs.distance_ratio = result.distance_ratio;
        outputs.total_deviation = result.total_deviation;
    }
}

fn state_of(state: &mut Option<ReflectionSimulationState>) -> &mut ReflectionSimulationState {
    state
        .as_mut()
        .expect("a source with reflections enabled was created with indirect simulation")
}

/// Estimates the reverb of real-time sources and applies the reverb scale (upstream `estimateReverb`).
fn estimate_reverb(sources: &mut [MutexGuard<'_, SimulationData>]) {
    for source in sources.iter_mut() {
        if !source.reflection_inputs.enabled {
            continue;
        }

        let source = &mut **source;
        let inputs = &source.reflection_inputs;
        let state = state_of(&mut source.reflection_state);

        if !inputs.baked {
            source.reflection_outputs.reverb =
                reverb_estimator::estimate(&state.accum_energy_field, &inputs.air_absorption_model);
        }

        let should_apply_reverb_scale = inputs.reverb_scale.iter().any(|&s| s != 1.0);

        if state.valid_simulation_data && should_apply_reverb_scale {
            reverb_estimator::apply_reverb_scale(
                &inputs.reverb_scale,
                &mut state.accum_energy_field,
            );

            for i in 0..NUM_BANDS {
                source.reflection_outputs.reverb.reverb_times[i] *= inputs.reverb_scale[i];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn types_are_send() {
        fn is_send<T: Send>() {}
        is_send::<SimulationManager>();
        is_send::<SimulationData>();
    }
}
