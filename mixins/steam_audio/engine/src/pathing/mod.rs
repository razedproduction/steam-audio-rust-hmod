//! Pathing: sound paths around obstacles through probes that see each other (upstream `path_visibility`,
//! `path_finder`, `path_data`).
//!
//! A [`PathBaker`] builds a [`ProbeVisibilityGraph`] over the probes of a batch, finds the shortest path between every
//! pair of probes with a [`PathFinder`], and stores the distinct ones as [`SoundPath`]s in a [`BakedPathData`] layer of
//! the batch. At run time a [PathSimulator] combines the baked paths from the probes around a source to those around
//! the listener into a sound field, which [PathEffect](crate::effects::PathEffect) renders.

mod baker;
mod data;
mod finder;
mod simulator;
mod visibility;

pub use baker::PathBaker;
pub use data::{BakedPathData, PathBakeParams, SoundPath};
pub use finder::{PathFinder, ProbePath, ShortestPathParams};
pub use simulator::{
    PathSimulationParams, PathSimulationResult, PathSimulator, ValidationRayCallback,
};
pub use visibility::{
    AdjacencyEntry, ProbeVisibilityGraph, ProbeVisibilityTester, VisibilityParams,
};
