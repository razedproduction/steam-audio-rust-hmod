//! Data baked at probes (upstream `probe_data.h`, `baked_reflection_data`): what a [`ProbeBatch`] stores per data
//! layer, keyed by a [`BakedDataIdentifier`], and the [`ReflectionBaker`] that computes reflections layers (upstream
//! `reflection_baker`).
//!
//! [`ProbeBatch`]: crate::probes::ProbeBatch

mod baker;
mod reflections;

use std::cmp::Ordering;

pub use baker::{ReflectionBakeParams, ReflectionBaker};
pub use reflections::BakedReflectionsData;

use crate::geometry::Sphere;
use crate::math::Vector3;
use crate::pathing::BakedPathData;
use crate::probes::Probe;

/// What a layer of baked data describes (upstream `BakedDataVariation`, serialized as a byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BakedDataVariation {
    /// Reverb at the listener.
    Reverb = 0,
    /// Reflections from a static source to anywhere.
    StaticSource = 1,
    /// Reflections from anywhere to a static listener.
    StaticListener = 2,
    Dynamic = 3,
}

/// The kind of data in a layer (upstream `BakedDataType`, serialized as a byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BakedDataType {
    Reflections = 0,
    Pathing = 1,
}

impl BakedDataVariation {
    pub(crate) fn from_i8(value: i8) -> Option<Self> {
        [
            Self::Reverb,
            Self::StaticSource,
            Self::StaticListener,
            Self::Dynamic,
        ]
        .get(usize::try_from(value).ok()?)
        .copied()
    }
}

impl BakedDataType {
    pub(crate) fn from_i8(value: i8) -> Option<Self> {
        [Self::Reflections, Self::Pathing]
            .get(usize::try_from(value).ok()?)
            .copied()
    }
}

/// Identifies a layer of baked data in a probe batch (upstream `BakedDataIdentifier`).
///
/// Identifiers are ordered as upstream's `operator<` orders them (variation, type, then the endpoint's center and
/// radius compared with `<`), which decides the order in which layers are serialized. Equality is that ordering's
/// equivalence, as for upstream's `std::map` keys; it differs from upstream's `operator==` only for NaN coordinates.
#[derive(Debug, Clone, Copy)]
pub struct BakedDataIdentifier {
    pub variation: BakedDataVariation,
    /// Upstream `type`.
    pub data_type: BakedDataType,
    /// The static source or listener whose data this is (unused for [`BakedDataVariation::Reverb`]).
    pub endpoint_influence: Sphere,
}

impl BakedDataIdentifier {
    /// Upstream `operator==`: floats compared with `==`.
    pub(crate) fn same_as(&self, other: &Self) -> bool {
        self.variation == other.variation
            && self.data_type == other.data_type
            && self.endpoint_influence.center == other.endpoint_influence.center
            && self.endpoint_influence.radius == other.endpoint_influence.radius
    }

    /// Upstream `operator<`.
    fn less(&self, other: &Self) -> bool {
        if self.variation != other.variation {
            return self.variation < other.variation;
        }
        if self.data_type != other.data_type {
            return self.data_type < other.data_type;
        }
        let (a, b) = (&self.endpoint_influence, &other.endpoint_influence);
        for axis in 0..3 {
            if a.center[axis] != b.center[axis] {
                return a.center[axis] < b.center[axis];
            }
        }
        if a.radius != b.radius {
            return a.radius < b.radius;
        }
        false
    }
}

impl Ord for BakedDataIdentifier {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.less(other) {
            Ordering::Less
        } else if other.less(self) {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for BakedDataIdentifier {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for BakedDataIdentifier {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for BakedDataIdentifier {}

/// A layer of baked data (upstream `IBakedData`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BakedData {
    Reflections(BakedReflectionsData),
    /// Paths between probes. Any change to the probes makes it need baking again
    /// ([`BakedPathData::needs_update`]).
    Pathing(BakedPathData),
}

impl BakedData {
    /// Upstream `updateProbePosition`: the probe's data must be baked again.
    pub(crate) fn update_probe_position(&mut self, index: usize, position: Vector3) {
        match self {
            Self::Reflections(data) => data.update_probe_position(index, position),
            Self::Pathing(data) => data.mark_needs_update(),
        }
    }

    pub(crate) fn add_probe(&mut self, influence: &Sphere) {
        match self {
            Self::Reflections(data) => data.add_probe(influence),
            Self::Pathing(data) => data.mark_needs_update(),
        }
    }

    pub(crate) fn remove_probe(&mut self, index: usize) {
        match self {
            Self::Reflections(data) => data.remove_probe(index),
            Self::Pathing(data) => data.mark_needs_update(),
        }
    }

    pub(crate) fn update_endpoint(
        &mut self,
        identifier: &BakedDataIdentifier,
        probes: &[Probe],
        endpoint_influence: &Sphere,
    ) {
        match self {
            Self::Reflections(data) => data.update_endpoint(identifier, probes, endpoint_influence),
            Self::Pathing(_) => {}
        }
    }

    /// Upstream `serializedSize`: an estimate of the serialized size, without the FlatBuffers overhead.
    pub fn serialized_size(&self) -> u64 {
        match self {
            Self::Reflections(data) => data.serialized_size(),
            Self::Pathing(data) => data.serialized_size(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identifier(variation: BakedDataVariation, x: f32, radius: f32) -> BakedDataIdentifier {
        BakedDataIdentifier {
            variation,
            data_type: BakedDataType::Reflections,
            endpoint_influence: Sphere::new(Vector3::new(x, 0.0, 0.0), radius),
        }
    }

    #[test]
    fn identifiers_order_like_upstream() {
        use BakedDataVariation::*;
        let a = identifier(StaticSource, 5.0, 1.0);
        assert!(identifier(Reverb, 9.0, 9.0) < a);
        assert!(identifier(StaticSource, 4.0, 9.0) < a);
        assert!(identifier(StaticSource, 5.0, 0.5) < a);
        assert_eq!(identifier(StaticSource, 5.0, 1.0), a);
        assert_eq!(
            identifier(StaticSource, -0.0, 1.0),
            identifier(StaticSource, 0.0, 1.0)
        );

        // NaN is equivalent to anything at its level, as in upstream's map, but not `==` in upstream's sense.
        let nan = identifier(StaticSource, f32::NAN, 1.0);
        assert_eq!(nan, a);
        assert!(!nan.same_as(&nan));
    }
}
