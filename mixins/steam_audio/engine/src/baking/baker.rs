//! Baking reflections at probes (upstream `reflection_baker.{h,cpp}`, `ReflectionBaker`).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::air_absorption::AirAbsorptionModel;
use crate::directivity::Directivity;
use crate::math::CoordinateSpace3;
use crate::probes::ProbeBatch;
use crate::scene::RayTracer;
use crate::simulation::reverb_estimator;
use crate::simulation::{AnyReflectionSimulator, EnergyField, ReflectionSimulationParams};
use crate::thread_pool::ThreadPool;

use super::{
    BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation, BakedReflectionsData,
};

/// What to bake (the arguments of upstream `ReflectionBaker::bake` besides the scene, simulator and batch).
#[derive(Debug, Clone, Copy)]
pub struct ReflectionBakeParams {
    /// The layer to bake: reflections, of any variation but [`BakedDataVariation::Dynamic`].
    pub identifier: BakedDataIdentifier,
    /// Store an energy field per probe.
    pub bake_convolution: bool,
    /// Store a reverb per probe.
    pub bake_parametric: bool,
    pub num_rays: usize,
    pub num_bounces: usize,
    /// Length of the simulated energy fields, in seconds.
    pub sim_duration: f32,
    /// Length of the stored energy fields, in seconds: the simulated ones are cut or padded with silence.
    pub bake_duration: f32,
    pub order: usize,
    pub irradiance_min_distance: f32,
}

/// Bakes reflections into probe batches (upstream `ReflectionBaker`).
///
/// Upstream's baker is a class of static functions with a global cancel flag; here the flag belongs to a baker value,
/// so another thread holding a reference can [`cancel`](Self::cancel) its bake.
#[derive(Debug, Default)]
pub struct ReflectionBaker {
    cancel: AtomicBool,
    bake_in_progress: AtomicBool,
}

impl ReflectionBaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Simulates reflections at every probe of `batch` that the layer applies to, one probe at a time, and stores
    /// the results in the layer `params.identifier` of the batch, which is created if needed (upstream `bake`).
    ///
    /// For [`BakedDataVariation::Reverb`] every probe is both source and listener; for
    /// [`StaticSource`](BakedDataVariation::StaticSource) (or [`StaticListener`](BakedDataVariation::StaticListener))
    /// the endpoint's center is the source (or listener) and only the probes inside its influence are baked, as
    /// listeners (or sources). The reverb is estimated with the default air absorption model.
    ///
    /// After each simulation, and also after the last probe when it was not baked, `progress` gets the fraction of
    /// probes done; the bake then stops if it was cancelled.
    ///
    /// # Panics
    ///
    /// If neither convolution nor parametric data is baked, the layer is not for reflections or is dynamic, or the
    /// simulator panics (see [`ReflectionSimulator::simulate`](crate::simulation::ReflectionSimulator::simulate)).
    pub fn bake(
        &self,
        scene: &dyn RayTracer,
        simulator: &mut AnyReflectionSimulator,
        params: &ReflectionBakeParams,
        pool: &ThreadPool,
        batch: &mut ProbeBatch,
        mut progress: impl FnMut(f32),
    ) {
        let identifier = &params.identifier;
        assert!(
            params.bake_convolution || params.bake_parametric,
            "nothing to bake"
        );
        assert_eq!(identifier.data_type, BakedDataType::Reflections);
        assert_ne!(identifier.variation, BakedDataVariation::Dynamic);

        self.bake_in_progress.store(true, Ordering::SeqCst);

        // A layer of other data under this identifier is replaced (upstream would treat it as reflections).
        if !matches!(batch.data(identifier), Some(BakedData::Reflections(_))) {
            batch.add_data(
                *identifier,
                BakedData::Reflections(BakedReflectionsData::new(
                    *identifier,
                    batch.num_probes(),
                    params.bake_convolution,
                    params.bake_parametric,
                )),
            );
        }
        {
            let data = reflections_data(batch, identifier);
            data.set_has_convolution(params.bake_convolution);
            data.set_has_parametric(params.bake_parametric);
        }

        let air_absorption = AirAbsorptionModel::default();
        let directivities = [Directivity::default()];
        let num_probes = batch.num_probes();
        // Upstream's listener array keeps the last probe's listener (initially the canonical space at the origin),
        // which a simulation without a source still traces rays from.
        let mut listener = CoordinateSpace3::default();

        for i in 0..num_probes {
            let center = batch.probes()[i].influence.center;
            let endpoint = identifier.endpoint_influence;
            let endpoints = match identifier.variation {
                BakedDataVariation::Reverb => Some((center, center)),
                BakedDataVariation::StaticSource => endpoint
                    .contains(center)
                    .then_some((endpoint.center, center)),
                BakedDataVariation::StaticListener => endpoint
                    .contains(center)
                    .then_some((center, endpoint.center)),
                BakedDataVariation::Dynamic => unreachable!(),
            };

            // Upstream bakes one probe per batch of sources (for its CPU ray tracers), and simulates when the batch
            // is full or the last probe is reached: the last probe is simulated with no source when it is not baked.
            if endpoints.is_some() || i + 1 == num_probes {
                let sources: &[CoordinateSpace3] = match &endpoints {
                    Some((source, _)) => &[CoordinateSpace3::canonical_at(*source)],
                    None => &[],
                };
                if let Some((_, position)) = endpoints {
                    listener = CoordinateSpace3::canonical_at(position);
                }

                let mut energy_fields = [EnergyField::new(params.sim_duration, params.order)];
                let num_fields = sources.len();
                simulator.simulate(
                    scene,
                    &ReflectionSimulationParams {
                        sources,
                        directivities: &directivities[..num_fields],
                        listener,
                        num_rays: params.num_rays,
                        num_bounces: params.num_bounces,
                        duration: params.sim_duration,
                        order: params.order,
                        irradiance_min_distance: params.irradiance_min_distance,
                    },
                    &mut energy_fields[..num_fields],
                    pool,
                );

                if endpoints.is_some() {
                    let [energy_field] = energy_fields;
                    let data = reflections_data(batch, identifier);

                    if params.bake_parametric {
                        data.set_reverb(
                            i,
                            reverb_estimator::estimate(&energy_field, &air_absorption),
                        );
                    }

                    if params.bake_convolution {
                        let energy_field = if params.sim_duration == params.bake_duration {
                            energy_field
                        } else {
                            let mut baked = EnergyField::new(params.bake_duration, params.order);
                            baked.copy_from(&energy_field);
                            baked
                        };
                        data.set_energy_field(i, Some(energy_field));
                    }
                }

                progress((i as f32 + 1.0) / num_probes as f32);

                if self.cancel.swap(false, Ordering::SeqCst) {
                    break;
                }
            }
        }

        self.bake_in_progress.store(false, Ordering::SeqCst);
    }

    /// Stops a bake in progress after its current probe (upstream `cancel`); does nothing when no bake is running.
    pub fn cancel(&self) {
        if self.bake_in_progress.load(Ordering::SeqCst) {
            self.cancel.store(true, Ordering::SeqCst);
        }
    }
}

fn reflections_data<'a>(
    batch: &'a mut ProbeBatch,
    identifier: &BakedDataIdentifier,
) -> &'a mut BakedReflectionsData {
    match batch
        .data_mut(identifier)
        .expect("the layer was added before baking")
    {
        BakedData::Reflections(data) => data,
        BakedData::Pathing(_) => unreachable!("the layer was added as reflections"),
    }
}
