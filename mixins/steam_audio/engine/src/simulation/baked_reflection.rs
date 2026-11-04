//! Lookups of baked reflections (upstream `baked_reflection_simulator.{h,cpp}`, namespace
//! `BakedReflectionSimulator`): the energy field or reverb at a point, interpolated from the probes around it.

use std::borrow::Borrow;

use crate::baking::{BakedData, BakedDataIdentifier, BakedDataType};
use crate::probes::{ProbeBatch, ProbeNeighborhood};
use crate::simulation::{EnergyField, Reverb};

/// Writes into `unique_batches` the distinct batches the neighbourhood's entries refer to (upstream
/// `findUniqueProbeBatches`), as indices into the batch list, compared by identity (a batch added to a
/// [`ProbeManager`](crate::probes::ProbeManager) twice is one batch).
///
/// They are in the order of their first entry. Upstream collects them in an `unordered_set` of pointers, whose order
/// depends on the batches' addresses; the lookups add the batches' contributions in this order.
pub fn find_unique_probe_batches<B: Borrow<ProbeBatch>>(
    neighborhood: &ProbeNeighborhood,
    batches: &[B],
    unique_batches: &mut Vec<usize>,
) {
    unique_batches.clear();
    for &batch in neighborhood.batches.iter().flatten() {
        let is_new = unique_batches
            .iter()
            .all(|&b| !std::ptr::eq(batches[b].borrow(), batches[batch].borrow()));
        if is_new {
            unique_batches.push(batch);
        }
    }
}

/// The data layer of a unique batch to evaluate, if any.
fn reflections_data<'a, B: Borrow<ProbeBatch>>(
    identifier: &BakedDataIdentifier,
    batches: &'a [B],
    batch: usize,
) -> Option<&'a crate::baking::BakedReflectionsData> {
    match batches[batch].borrow().data(identifier)? {
        BakedData::Reflections(data) => Some(data),
        BakedData::Pathing(_) => None,
    }
}

/// Interpolates the baked energy fields with `identifier` at the neighbourhood's probes into `energy_field`, weighted
/// by the neighbourhood's weights (upstream `lookupEnergyField`). If the neighbourhood has no valid probe,
/// `energy_field` is left as it is; otherwise it is zeroed first, then each batch of `unique_batches` (indices into
/// `batches`, from [`find_unique_probe_batches`]) adds its probes' fields in turn.
///
/// # Panics
///
/// If `identifier` is not for reflections, or an index is out of range.
pub fn lookup_energy_field<B: Borrow<ProbeBatch>>(
    identifier: &BakedDataIdentifier,
    neighborhood: &ProbeNeighborhood,
    batches: &[B],
    unique_batches: &[usize],
    energy_field: &mut EnergyField,
) {
    assert_eq!(identifier.data_type, BakedDataType::Reflections);

    if neighborhood.has_valid_probes() {
        energy_field.reset();
        for &batch in unique_batches {
            if let Some(data) = reflections_data(identifier, batches, batch) {
                data.evaluate_energy_field(neighborhood, batches, energy_field);
            }
        }
    }
}

/// Interpolates the baked reverbs with `identifier` at the neighbourhood's probes into `reverb` (upstream
/// `lookupReverb`), as [`lookup_energy_field`] does for energy fields.
///
/// # Panics
///
/// If `identifier` is not for reflections, or an index is out of range.
pub fn lookup_reverb<B: Borrow<ProbeBatch>>(
    identifier: &BakedDataIdentifier,
    neighborhood: &ProbeNeighborhood,
    batches: &[B],
    unique_batches: &[usize],
    reverb: &mut Reverb,
) {
    assert_eq!(identifier.data_type, BakedDataType::Reflections);

    if neighborhood.has_valid_probes() {
        *reverb = Reverb::default();
        for &batch in unique_batches {
            if let Some(data) = reflections_data(identifier, batches, batch) {
                data.evaluate_reverb(neighborhood, batches, reverb);
            }
        }
    }
}
