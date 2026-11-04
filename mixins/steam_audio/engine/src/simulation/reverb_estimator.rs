//! Parametric reverb estimation from an energy field (upstream `reverb_estimator.{h,cpp}`).

use crate::air_absorption::AirAbsorptionModel;
use crate::array_math;
use crate::bands::NUM_BANDS;
use crate::math::cpp_max;
use crate::simulation::EnergyField;

/// Lowest and highest log-energy (`log10` of the normalized energy decay curve) used for the line fit.
const MIN_ENERGY_FOR_LINE_FIT: f32 = -2.5;
const MAX_ENERGY_FOR_LINE_FIT: f32 = -0.5;

/// Below this total energy (weighted by air absorption) the reverb time is 0.
const MIN_TOTAL_ENERGY: f32 = 1e-4;

/// Smallest reverb time [`estimate`] returns.
const MIN_REVERB_TIME: f32 = 0.1;

/// A parametric reverb (upstream `Reverb`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Reverb {
    /// Time taken for the energy to decay by 60 dB, per band, in seconds.
    pub reverb_times: [f32; NUM_BANDS],
}

/// Estimates the reverb time of each band from channel 0 of `energy_field`, at least 0.1 s.
///
/// Each band's energy decay curve, weighted by air absorption, is fitted with a line between −5 dB and −25 dB.
pub fn estimate(energy_field: &EnergyField, air_absorption: &AirAbsorptionModel) -> Reverb {
    Reverb {
        reverb_times: std::array::from_fn(|i| {
            cpp_max(
                reverb_time(energy_field.band(0, i), air_absorption, i),
                MIN_REVERB_TIME,
            )
        }),
    }
}

/// Stretches each band's decay after its peak (in channel 0) by `reverb_scale[band]`, keeping the band's total energy.
///
/// Every channel is scaled by the factors computed from channel 0.
pub fn apply_reverb_scale(reverb_scale: &[f32; NUM_BANDS], energy_field: &mut EnergyField) {
    let num_channels = energy_field.num_channels();

    for (i, &reverb_time_ratio) in reverb_scale.iter().enumerate() {
        let (peak_energy, peak_bin) = array_math::max_index(energy_field.band(0, i));

        let old_total_energy = sum(energy_field.band(0, i));

        for j in peak_bin..energy_field.num_bins() {
            let old_energy = energy_field.band(0, i)[j];
            if old_energy <= 0.0 {
                continue;
            }

            let new_energy = peak_energy * (old_energy / peak_energy).powf(1.0 / reverb_time_ratio);
            let scalar = new_energy / old_energy;

            for k in 0..num_channels {
                energy_field.band_mut(k, i)[j] *= scalar;
            }
        }

        let new_total_energy = sum(energy_field.band(0, i));

        let energy_scalar = if new_total_energy > 0.0 {
            old_total_energy / new_total_energy
        } else {
            1.0
        };
        for k in 0..num_channels {
            for e in energy_field.band_mut(k, i) {
                *e *= energy_scalar;
            }
        }
    }
}

/// Sums in order, like upstream's loops.
fn sum(values: &[f32]) -> f32 {
    values.iter().fold(0.0, |s, &x| s + x)
}

/// Reverb time of one band's histogram, from a least-squares line fit to its log energy decay curve.
///
/// As upstream, the backward pass starts at the time the forward pass ended (the end of the last bin, accumulated in
/// `f32`), not at the last bin's start.
fn reverb_time(histogram: &[f32], air_absorption: &AirAbsorptionModel, band: usize) -> f32 {
    let mut total_energy = 0.0f32;
    let mut x = 0.0f32;
    for &h in histogram {
        total_energy += h * air_absorption.evaluate(x, band);
        x += EnergyField::BIN_DURATION;
    }

    if total_energy < MIN_TOTAL_ENERGY {
        return 0.0;
    }

    let mut energy = 0.0f32;
    let mut sum_x = 0.0f32;
    let mut sum_y = 0.0f32;
    let mut sum_xx = 0.0f32;
    let mut sum_xy = 0.0f32;
    let mut n = 0i32;

    for &h in histogram.iter().rev() {
        energy += h * air_absorption.evaluate(x, band);
        let y = (energy / total_energy).log10();

        if (MIN_ENERGY_FOR_LINE_FIT..=MAX_ENERGY_FOR_LINE_FIT).contains(&y) {
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            n += 1;
        }

        x -= EnergyField::BIN_DURATION;
    }

    let numerator = (n as f32 * sum_xy) - (sum_x * sum_y);
    let denominator = (n as f32 * sum_xx) - (sum_x * sum_x);

    if numerator.abs() > f32::MIN_POSITIVE {
        cpp_max(0.0, -6.0 * (denominator / numerator))
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_decay() {
        // Energy decaying by 60 dB per second: a reverb time of 1 s.
        let mut field = EnergyField::new(2.0, 0);
        for b in 0..NUM_BANDS {
            for (k, e) in field.band_mut(0, b).iter_mut().enumerate() {
                *e = 10f32.powf(-6.0 * k as f32 * EnergyField::BIN_DURATION);
            }
        }
        let air = AirAbsorptionModel::Exponential {
            coefficients: [0.0; NUM_BANDS],
        };
        for t in estimate(&field, &air).reverb_times {
            assert!((t - 1.0).abs() < 0.01, "{t}");
        }
    }

    #[test]
    fn silence_gives_the_minimum() {
        let field = EnergyField::new(1.0, 1);
        let reverb = estimate(&field, &AirAbsorptionModel::default());
        assert_eq!(reverb.reverb_times, [MIN_REVERB_TIME; NUM_BANDS]);
    }
}
