//! Prepares an impulse response for hybrid reverb (upstream `hybrid_reverb_estimator.{h,cpp}`): estimates the level
//! of the parametric part and crossfades the IR into it.

use crate::audio_buffer::{AudioBuffer, AudioSettings};
use crate::bands::NUM_BANDS;
use crate::effects::{EqEffect, ReverbEffect};
use crate::iir::{Iir, IirFilterer};
use crate::math::{PI, Vector3, msvc_f32_to_i32};
use crate::random::CRand;
use crate::sh;
use crate::simulation::{EnergyField, ImpulseResponse, Reverb};

/// Relative energy of white noise in each band after the band filters (upstream `whiteNoiseNorm`).
const WHITE_NOISE_NORM: [f32; NUM_BANDS] = [0.984652, 0.996133, 1.0];

/// Gain of the parametric reverb before EQ normalization.
const PARAMETRIC_GAIN: f32 = 16.0;

/// What [`HybridReverbEstimator::estimate`] returns besides the modified IR.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HybridReverbEstimate {
    /// Level of the parametric part per band, for [`HybridReverbParams::eq_coeffs`](crate::effects::HybridReverbParams).
    pub eq_coeffs: [f32; NUM_BANDS],
    /// Delay of the parametric part in samples, for [`HybridReverbParams::delay`](crate::effects::HybridReverbParams).
    pub delay: usize,
}

/// Estimates hybrid reverb parameters and shapes the IR's transition into the parametric reverb (upstream
/// `HybridReverbEstimator`).
#[derive(Debug, Clone)]
pub struct HybridReverbEstimator {
    sampling_rate: u32,
    frame_size: usize,
    eq: EqEffect,
    reverb: ReverbEffect,
    temp0: AudioBuffer,
    temp1: AudioBuffer,
    /// The parametric reverb's impulse response, or one band of the IR when estimating the EQ from it.
    reverb_ir: Vec<f32>,
    bandpass_filters: [IirFilterer; NUM_BANDS],
}

impl HybridReverbEstimator {
    /// An estimator for IRs of up to `max_duration` seconds. Its reverb draws its delay lengths from `rand` (see
    /// [`ReverbEffect::new`]).
    pub fn new(max_duration: f32, sampling_rate: u32, frame_size: usize, rand: &mut CRand) -> Self {
        let audio_settings = AudioSettings::new(sampling_rate, frame_size);
        let reverb_ir_size = (max_duration * sampling_rate as f32).ceil() as usize;
        Self {
            sampling_rate,
            frame_size,
            eq: EqEffect::new(&audio_settings),
            reverb: ReverbEffect::new(&audio_settings, rand),
            temp0: AudioBuffer::new(1, frame_size),
            temp1: AudioBuffer::new(1, frame_size),
            reverb_ir: vec![0.0; reverb_ir_size],
            bandpass_filters: Iir::band_filters(sampling_rate).map(IirFilterer::new),
        }
    }

    /// Crossfades the first `numCoeffsForOrder(order)` channels of `ir` into a parametric reverb with `reverb`'s
    /// reverb times over the end of the transition (the last `overlap_fraction` of `transition_time` seconds), and
    /// clears them after it.
    ///
    /// The parametric part's level is taken from `energy_field` at the start of the crossfade if given, otherwise
    /// from the IR's own energy decay in each band.
    ///
    /// # Panics
    ///
    /// If `ir` has no samples or fewer channels than `order` needs, the IR, or the crossfade, is longer than the
    /// maximum duration, or the energy field ends before the crossfade starts.
    pub fn estimate(
        &mut self,
        energy_field: Option<&EnergyField>,
        reverb: &Reverb,
        ir: &mut ImpulseResponse,
        transition_time: f32,
        overlap_fraction: f32,
        order: usize,
    ) -> HybridReverbEstimate {
        let num_channels = sh::num_coeffs_for_order(order);
        let rate = self.sampling_rate as f32;
        assert!(ir.num_samples() > 0, "empty impulse response");
        assert!(ir.num_channels() >= num_channels, "IR channels");

        // If the transition time is past the end of the IR, clamp it.
        let mut transition_time = transition_time;
        let mut transition_samples = msvc_f32_to_i32((transition_time * rate).ceil()) as i64;
        if transition_samples >= ir.num_samples() as i64 {
            transition_samples = ir.num_samples() as i64 - 1;
            transition_time = transition_samples as f32 / rate;
        }

        let eq_coeffs = match energy_field {
            Some(field) => {
                let cutoff_bin = (((1.0 - overlap_fraction) * transition_time)
                    / EnergyField::BIN_DURATION)
                    .ceil() as usize;
                std::array::from_fn(|i| (4.0 * PI * field.band(0, i)[cutoff_bin]).sqrt())
            }
            None => self.estimate_eq_from_ir(ir, transition_time, overlap_fraction),
        };

        let transition_samples = msvc_f32_to_i32((transition_time * rate).ceil());
        let ramp_start = ((1.0 - overlap_fraction) * transition_samples as f32) as i32;
        let ramp_end = transition_samples;
        assert!(0 <= ramp_start && ramp_start <= ramp_end, "transition");
        let (ramp_start, ramp_end) = (ramp_start as usize, ramp_end as usize);

        self.calc_reverb_ir(ramp_end - ramp_start, &eq_coeffs, &reverb.reverb_times);

        let ramp_length = (ramp_end - ramp_start) as f32;
        for i in ramp_start..ramp_end {
            let alpha = (ramp_end - i) as f32 / ramp_length;
            for j in 0..num_channels {
                ir.channel_mut(j)[i] *= alpha.sqrt();
            }
        }

        for i in ramp_start..ramp_end {
            let alpha = (ramp_end - i) as f32 / ramp_length;
            ir.channel_mut(0)[i] -= (1.0 - (1.0 - alpha).sqrt()) * self.reverb_ir[i - ramp_start];
        }

        for j in 0..num_channels {
            ir.channel_mut(j)[ramp_end..].fill(0.0);
        }

        let delay = ((1.0 - overlap_fraction) * transition_time * rate).floor();
        HybridReverbEstimate {
            eq_coeffs,
            delay: msvc_f32_to_i32(delay).max(0) as usize,
        }
    }

    /// The parametric part's level per band from the IR's channel 0: the square root of each band's energy after the
    /// start of the crossfade, relative to the band's total.
    fn estimate_eq_from_ir(
        &mut self,
        ir: &ImpulseResponse,
        transition_time: f32,
        overlap_fraction: f32,
    ) -> [f32; NUM_BANDS] {
        let rate = self.sampling_rate as f32;
        let cutoff_sample =
            msvc_f32_to_i32(((1.0 - overlap_fraction) * transition_time * rate).floor());

        let n = ir.num_samples();
        let band_ir = &mut self.reverb_ir[..n];
        let mut eq: [f32; NUM_BANDS] = std::array::from_fn(|i| {
            let filter = &mut self.bandpass_filters[i];
            filter.reset();
            filter.apply(ir.channel(0), band_ir);
            relative_edc(band_ir, cutoff_sample)
        });

        for (e, norm) in eq.iter_mut().zip(WHITE_NOISE_NORM) {
            *e *= 1.0 / norm;
        }
        for e in &mut eq {
            *e = e.sqrt();
        }
        eq
    }

    /// The impulse response of the EQ and the parametric reverb, `num_samples` long, into `reverb_ir`.
    ///
    /// Like upstream, the EQ's and the reverb's filter states persist between calls (their `reset` keeps them).
    fn calc_reverb_ir(
        &mut self,
        num_samples: usize,
        eq_coeffs: &[f32; NUM_BANDS],
        reverb_times: &[f32; NUM_BANDS],
    ) {
        self.reverb_ir.fill(0.0);

        self.eq.reset();
        self.reverb.reset();

        let num_frames = num_samples.div_ceil(self.frame_size);

        let mut eq_coeffs = *eq_coeffs;
        let mut overall_gain = PARAMETRIC_GAIN * sh::evaluate(0, 0, Vector3::ZERO);
        EqEffect::normalize_gains(&mut eq_coeffs, &mut overall_gain);

        let reverb = Reverb {
            reverb_times: *reverb_times,
        };

        let mut num_samples_left = num_samples;
        for i in 0..num_frames {
            self.temp0.make_silent();
            if i == 0 {
                self.temp0.channel_mut(0)[0] = 1.0;
            }

            self.eq.apply(&eq_coeffs, &self.temp0, &mut self.temp1);

            for x in self.temp1.channel_mut(0) {
                *x *= overall_gain;
            }

            self.reverb.apply(&reverb, &self.temp1, &mut self.temp0);

            let n = self.frame_size.min(num_samples_left);
            num_samples_left -= n;

            let start = i * self.frame_size;
            self.reverb_ir[start..start + n].copy_from_slice(&self.temp0.channel(0)[..n]);
        }
    }
}

/// Energy from `cutoff_sample` to the end of `ir`, relative to its total energy (0 for silence, or when the cutoff is
/// outside the IR).
fn relative_edc(ir: &[f32], cutoff_sample: i32) -> f32 {
    let mut sum = 0.0f32;
    let mut cutoff_sum = 0.0f32;
    for (i, &x) in ir.iter().enumerate().rev() {
        sum += x * x;
        if i as i64 == i64::from(cutoff_sample) {
            cutoff_sum = sum;
        }
    }

    if sum == 0.0 { 0.0 } else { cutoff_sum / sum }
}
