//! Probes: points where simulation results are baked, and the structures that find the probes around a point
//! (upstream `probe`, `probe_generator`, `probe_tree`, `probe_batch`, `probe_manager`).
//!
//! A [`ProbeBatch`] holds probes and the data baked at them; a [`ProbeManager`] holds the batches a simulation uses
//! and fills a [`ProbeNeighborhood`] with the probes whose influence contains a point, weighted by distance.

mod batch;
mod generator;
mod manager;
mod neighborhood;
mod tree;

pub use batch::ProbeBatch;
pub use generator::{ProbeGeneration, generate_probes};
pub use manager::ProbeManager;
pub use neighborhood::ProbeNeighborhood;
pub use tree::{ProbeTree, ProbeTreeNode};

use crate::geometry::Sphere;

/// A probe (upstream `Probe`): a point and the sphere around it in which its baked data applies.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Probe {
    pub influence: Sphere,
}

impl Probe {
    pub const fn new(influence: Sphere) -> Self {
        Self { influence }
    }

    /// The 16 bytes of `ipl.Serialized.Sphere`: center, then radius.
    pub(crate) fn sphere_bytes(sphere: &Sphere) -> [u8; 16] {
        let mut bytes = [0; 16];
        let values = [
            sphere.center.x,
            sphere.center.y,
            sphere.center.z,
            sphere.radius,
        ];
        for (out, x) in bytes.chunks_exact_mut(4).zip(values) {
            out.copy_from_slice(&x.to_le_bytes());
        }
        bytes
    }

    /// Reads an `ipl.Serialized.Sphere`. Panics unless `bytes` has 16 bytes.
    pub(crate) fn sphere_from_bytes(bytes: &[u8]) -> Sphere {
        let value =
            |i: usize| f32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        Sphere::new(
            crate::math::Vector3::new(value(0), value(1), value(2)),
            value(3),
        )
    }
}
