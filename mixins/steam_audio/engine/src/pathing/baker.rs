//! Baking paths into probe batches (upstream `PathBaker` in `path_data.{h,cpp}`).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::baking::{BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation};
use crate::probes::ProbeBatch;
use crate::scene::RayTracer;
use crate::thread_pool::ThreadPool;

use super::{BakedPathData, PathBakeParams};

/// Bakes paths into probe batches (upstream `PathBaker`).
///
/// Upstream's baker is a class of static functions with a global cancel flag; here the flag belongs to a baker value,
/// so another thread holding a reference can [`cancel`](Self::cancel) its bake.
#[derive(Debug, Default)]
pub struct PathBaker {
    cancel: AtomicBool,
    bake_in_progress: AtomicBool,
}

impl PathBaker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the layer `identifier` of `batch` with paths baked between its probes on the threads of `pool`
    /// (upstream `bake`; see [`BakedPathData`]). `progress` gets the fraction done of each phase.
    ///
    /// A cancelled bake leaves the batch without the layer (upstream adds a half-built one).
    ///
    /// # Panics
    ///
    /// If the layer is not a [`BakedDataType::Pathing`] layer of variation [`BakedDataVariation::Dynamic`].
    pub fn bake(
        &self,
        scene: &dyn RayTracer,
        identifier: &BakedDataIdentifier,
        params: &PathBakeParams,
        pool: &ThreadPool,
        batch: &mut ProbeBatch,
        mut progress: impl FnMut(f32),
    ) {
        assert_eq!(identifier.data_type, BakedDataType::Pathing);
        assert_eq!(identifier.variation, BakedDataVariation::Dynamic);

        self.bake_in_progress.store(true, Ordering::SeqCst);
        self.cancel.store(false, Ordering::SeqCst);

        batch.remove_data(identifier);

        let data = BakedPathData::bake(
            scene,
            batch.probes(),
            params,
            pool,
            &self.cancel,
            &mut progress,
        );
        match data {
            Some(data) => batch.add_data(*identifier, BakedData::Pathing(data)),
            // Upstream clears the flag when it stops.
            None => self.cancel.store(false, Ordering::SeqCst),
        }

        self.bake_in_progress.store(false, Ordering::SeqCst);
    }

    /// Stops a bake in progress at its next check (upstream `cancel`); does nothing when no bake is running.
    pub fn cancel(&self) {
        if self.bake_in_progress.load(Ordering::SeqCst) {
            self.cancel.store(true, Ordering::SeqCst);
        }
    }
}
