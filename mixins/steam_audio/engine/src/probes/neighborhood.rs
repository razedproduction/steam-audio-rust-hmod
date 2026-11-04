//! Probe neighbourhoods (upstream `ProbeNeighborhood`, in `probe_batch.h` and `probe_manager.cpp`).

use std::borrow::Borrow;

use crate::geometry::Ray;
use crate::math::Vector3;
use crate::scene::RayTracer;

use super::ProbeBatch;

/// The probes around a point: up to [`MAX_PROBES_PER_BATCH`](Self::MAX_PROBES_PER_BATCH) entries per probe batch,
/// each a probe of a batch and its weight (upstream `ProbeNeighborhood`).
///
/// Upstream points at the batches; here an entry holds the batch's index in the list of batches the neighbourhood
/// was filled from ([`ProbeManager::probe_batches`](super::ProbeManager::probe_batches)), and the methods that look at
/// probes take that list. An entry is valid when it has both a batch and a probe.
#[derive(Debug, Clone, Default)]
pub struct ProbeNeighborhood {
    /// Index of each entry's batch (upstream `batches`, `nullptr` → `None`).
    pub batches: Vec<Option<usize>>,
    /// Each entry's probe in its batch (upstream `probeIndices`, `-1` → `None`).
    pub probe_indices: Vec<Option<usize>>,
    pub weights: Vec<f32>,

    // Scratch space for occlusion checks.
    rays: Vec<Ray>,
    min_distances: Vec<f32>,
    max_distances: Vec<f32>,
    ray_mapping: Vec<usize>,
    occluded: Vec<bool>,
}

impl ProbeNeighborhood {
    /// Entries per batch (upstream `kMaxProbesPerBatch`).
    pub const MAX_PROBES_PER_BATCH: usize = 8;

    /// Number of entries, valid or not.
    pub fn num_probes(&self) -> usize {
        self.weights.len()
    }

    /// The batch and probe of entry `i`, if it is valid.
    fn entry(&self, i: usize) -> Option<(usize, usize)> {
        Some((self.batches[i]?, self.probe_indices[i]?))
    }

    pub fn num_valid_probes(&self) -> usize {
        (0..self.num_probes())
            .filter(|&i| self.entry(i).is_some())
            .count()
    }

    pub fn has_valid_probes(&self) -> bool {
        (0..self.num_probes()).any(|i| self.entry(i).is_some())
    }

    /// Makes room for `max_probes` entries and resets them.
    pub fn resize(&mut self, max_probes: usize) {
        self.batches.resize(max_probes, None);
        self.probe_indices.resize(max_probes, None);
        self.weights.resize(max_probes, 0.0);

        self.rays.resize(max_probes, Ray::default());
        self.min_distances.resize(max_probes, 0.0);
        self.max_distances.resize(max_probes, 0.0);
        self.ray_mapping.resize(max_probes, 0);
        self.occluded.resize(max_probes, false);

        self.reset();
    }

    /// Clears every entry: no batch, no probe, weight 0.
    pub fn reset(&mut self) {
        self.batches.fill(None);
        self.probe_indices.fill(None);
        self.weights.fill(0.0);
    }

    /// Invalidates the entries whose probe `point` cannot see: one any-hit query per valid entry, from `point`
    /// to the probe's center.
    ///
    /// # Panics
    ///
    /// If an entry refers to a batch or probe that does not exist.
    pub fn check_occlusion<B: Borrow<ProbeBatch>>(
        &mut self,
        scene: &(impl RayTracer + ?Sized),
        point: Vector3,
        batches: &[B],
    ) {
        let mut num_rays = 0;
        for i in 0..self.num_probes() {
            if let Some((batch, probe)) = self.entry(i) {
                let dir = batches[batch].borrow().probes()[probe].influence.center - point;
                self.rays[num_rays] = Ray::new(point, dir.normalize_or_zero());
                self.min_distances[num_rays] = 0.0;
                self.max_distances[num_rays] = dir.length();
                self.ray_mapping[num_rays] = i;
                num_rays += 1;
            }
        }

        scene.any_hits(
            &self.rays[..num_rays],
            &self.min_distances[..num_rays],
            &self.max_distances[..num_rays],
            &mut self.occluded[..num_rays],
        );

        for j in 0..num_rays {
            if self.occluded[j] {
                self.batches[self.ray_mapping[j]] = None;
                self.probe_indices[self.ray_mapping[j]] = None;
            }
        }
    }

    /// The valid entry whose probe is nearest to `point` (the first of equally near ones), or 0 if none is valid.
    ///
    /// # Panics
    ///
    /// If an entry refers to a batch or probe that does not exist.
    pub fn find_nearest<B: Borrow<ProbeBatch>>(&self, point: Vector3, batches: &[B]) -> usize {
        let mut min_distance = f32::INFINITY;
        let mut min_index = 0;

        for i in 0..self.num_probes() {
            if let Some((batch, probe)) = self.entry(i) {
                let position = batches[batch].borrow().probes()[probe].influence.center;
                let distance = (point - position).length_squared();
                if distance < min_distance {
                    min_distance = distance;
                    min_index = i;
                }
            }
        }

        min_index
    }

    /// The probe and weight of entry `i`, or `None` if there is no such entry (upstream `getProbe`).
    pub fn probe(&self, i: usize) -> Option<(Option<usize>, f32)> {
        (i < self.num_probes()).then(|| (self.probe_indices[i], self.weights[i]))
    }

    /// Weights the valid entries by inverse distance to `point` (plus 1e-4), normalized to sum to 1. The weights of
    /// invalid entries are left as they are.
    ///
    /// # Panics
    ///
    /// If an entry refers to a batch or probe that does not exist.
    pub fn calc_weights<B: Borrow<ProbeBatch>>(&mut self, point: Vector3, batches: &[B]) {
        let mut total_weight = 0.0f32;
        for i in 0..self.num_probes() {
            if let Some((batch, probe)) = self.entry(i) {
                let center = batches[batch].borrow().probes()[probe].influence.center;
                self.weights[i] = 1.0 / ((point - center).length() + 1e-4);
                total_weight += self.weights[i];
            }
        }

        for i in 0..self.num_probes() {
            if self.entry(i).is_some() {
                self.weights[i] /= total_weight;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    /// Upstream `ProbeTree.test.cpp`: "Weight function sums to 1".
    #[test]
    fn weights_sum_to_one() {
        let mut batch = ProbeBatch::new();
        for x in [5.0, 0.1, 200.0, 0.9, 20.0] {
            batch.add_probe(Sphere::new(Vector3::new(x, 0.0, 0.0), f32::MAX));
        }

        let mut neighborhood = ProbeNeighborhood::default();
        neighborhood.resize(batch.num_probes());
        for i in 0..batch.num_probes() {
            neighborhood.batches[i] = Some(0);
            neighborhood.probe_indices[i] = Some(i);
        }

        neighborhood.calc_weights(Vector3::ZERO, &[&batch]);

        let sum: f32 = neighborhood.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "{sum}");
    }
}
