//! Simulation of sound propagation through a scene. So far: the direct path (upstream `direct_simulator`), the energy
//! fields and impulse responses of reflections with their reconstruction (upstream `energy_field`,
//! `impulse_response`, `reconstructor`), reverb estimation (upstream `reverb_estimator`,
//! `hybrid_reverb_estimator`), the reflection simulators that produce energy fields (upstream
//! `reflection_simulator`), lookups of baked reflections at probes (upstream `baked_reflection_simulator`), and the
//! [`SimulationManager`] that runs them, and pathing (see [`crate::pathing`]), for a set of sources (upstream
//! `simulation_manager`, `simulation_data`).

pub mod baked_reflection;
pub mod data;
pub mod direct;
pub mod energy_field;
pub mod hybrid_reverb_estimator;
pub mod impulse_response;
pub mod manager;
pub mod reconstructor;
pub mod reflection;
pub mod reverb_estimator;

pub use data::{
    DirectSimulationInputs, IndirectEffectType, PathingSimulationInputs, PathingSimulationOutputs,
    ReflectionSimulationInputs, ReflectionSimulationOutputs, SimulationData,
};
pub use direct::{
    DirectSimulationFlags, DirectSimulationParams, DirectSimulator, DirectSoundPath, OcclusionType,
};
pub use energy_field::EnergyField;
pub use hybrid_reverb_estimator::{HybridReverbEstimate, HybridReverbEstimator};
pub use impulse_response::ImpulseResponse;
pub use manager::{
    SceneType, SharedDirectSimulationInputs, SharedPathingSimulationInputs,
    SharedReflectionSimulationInputs, SharedSimulationData, SimulationManager, SimulationScene,
    SimulationSettings, ValidationRayVisualizationCallback,
};
pub use reconstructor::{ReconstructionParams, ReconstructionType, Reconstructor};
pub use reflection::{
    AnyReflectionSimulator, BatchedReflectionSimulator, ReflectionSimulationParams,
    ReflectionSimulator, ReflectionSimulatorSettings,
};
pub use reverb_estimator::Reverb;
