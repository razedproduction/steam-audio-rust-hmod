//! Rust reimplementation of [Steam Audio](https://github.com/ValveSoftware/steam-audio), vendored from
//! [steam-audio-rust](https://github.com/razedproduction/steam-audio-rust) for a Horizon native mod.
//!
//! Changes from upstream steam-audio-rust (see `README.md` of the mod):
//!
//! - [`Complex`] is this crate's own type ([`complex`]) instead of `num_complex::Complex<f32>`;
//! - [`fft`] is a self-contained radix-2 FFT instead of `realfft`/`rustfft`;
//! - the built-in HRTF is read from `data/cipic_124.hrtf` (generated once from upstream's `cipic_124.inl`)
//!   instead of being produced by a build script;
//! - SOFA support is not compiled in;
//! - [`horizon`], [`acoustics`], [`reflections`], [`paths`], [`settings`], [`materials`] and [`debug`] are new: the
//!   binaural renderer behind the mod's sound output, the world's occlusion and transmission, the reflections and
//!   reverb simulated from the blocks, the sound paths around them, its settings page in Options > Mods, the
//!   materials and blocks it tunes, and what its visual debug mode shows.

pub mod acoustics;
pub mod air_absorption;
pub mod array_math;
pub mod audio_buffer;
pub mod baking;
pub mod bands;
pub mod complex;
pub mod debug;
pub mod delay;
pub mod deviation;
pub mod directivity;
pub mod distance_attenuation;
pub mod effects;
pub mod error;
pub mod fft;
pub mod geometry;
pub mod horizon;
pub mod hrtf;
pub mod iir;
pub mod job_graph;
pub mod materials;
pub mod math;
pub mod pathing;
pub mod paths;
pub mod probes;
pub mod propagation_medium;
pub mod random;
pub mod reflections;
pub mod sampling;
pub mod scene;
mod serialization;
pub mod settings;
pub mod sh;
pub mod simulation;
pub mod speaker_layout;
pub mod thread_pool;
pub mod triple_buffer;
pub mod window;

#[cfg(feature = "sofa")]
pub use error::SofaError;
pub use error::{Error, Result};

/// Single-precision complex number (upstream `complex_t`).
pub type Complex = complex::Complex32;
