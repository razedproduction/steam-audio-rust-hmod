//! Probe managers (upstream `probe_manager.{h,cpp}`, `ProbeManager`).

use std::sync::Arc;

use crate::math::Vector3;

use super::{ProbeBatch, ProbeNeighborhood};

/// The probe batches a simulation looks up probes in (upstream `ProbeManager`). Added and removed batches take effect
/// at the next [`commit`](Self::commit).
#[derive(Debug, Clone, Default)]
pub struct ProbeManager {
    committed: Vec<Arc<ProbeBatch>>,
    pending: Vec<Arc<ProbeBatch>>,
}

impl ProbeManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of committed batches.
    pub fn num_probe_batches(&self) -> usize {
        self.committed.len()
    }

    /// The committed batches, in the order they were added. A [`ProbeNeighborhood`] filled by
    /// [`influencing_probes`](Self::influencing_probes) refers to batches by their index here.
    pub fn probe_batches(&self) -> &[Arc<ProbeBatch>] {
        &self.committed
    }

    /// Adds a batch at the next commit. A batch added twice is looked up twice.
    pub fn add_probe_batch(&mut self, batch: Arc<ProbeBatch>) {
        self.pending.push(batch);
    }

    /// Removes every copy of a batch (compared by identity) at the next commit.
    pub fn remove_probe_batch(&mut self, batch: &Arc<ProbeBatch>) {
        self.pending.retain(|b| !Arc::ptr_eq(b, batch));
    }

    pub fn commit(&mut self) {
        self.committed.clone_from(&self.pending);
    }

    /// Resizes `neighborhood` to [`MAX_PROBES_PER_BATCH`](ProbeNeighborhood::MAX_PROBES_PER_BATCH) entries per
    /// committed batch (or resets it if it has that size) and fills each batch's entries with the probes whose
    /// influence contains `point`. Weights are left at 0.
    ///
    /// # Panics
    ///
    /// If a batch has not been committed.
    pub fn influencing_probes(&self, point: Vector3, neighborhood: &mut ProbeNeighborhood) {
        let num_probes = self.committed.len() * ProbeNeighborhood::MAX_PROBES_PER_BATCH;
        if neighborhood.num_probes() != num_probes {
            neighborhood.resize(num_probes);
        } else {
            neighborhood.reset();
        }

        for (i, batch) in self.committed.iter().enumerate() {
            batch.influencing_probes(
                point,
                neighborhood,
                i * ProbeNeighborhood::MAX_PROBES_PER_BATCH,
                i,
            );
        }
    }
}
