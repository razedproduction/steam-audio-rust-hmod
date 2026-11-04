//! Rendering of the sound field a [`PathSimulator`](crate::pathing::PathSimulator) finds (upstream `PathEffect`): the
//! dry audio is equalized, then scaled by each SH coefficient into an Ambisonics buffer, or spatialized directly.

use super::overlap_add::OverlapAddConvolutionSettings;
use super::{
    AmbisonicsPanningEffect, AmbisonicsRotateEffect, EqEffect, GainEffect,
    OverlapAddConvolutionEffect,
};
use crate::Complex;
use crate::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use crate::bands::NUM_BANDS;
use crate::hrtf::{HrtfDatabase, NUM_EARS};
use crate::math::{CoordinateSpace3, DEGREES_TO_RADIANS};
use crate::sh;
use crate::speaker_layout::SpeakerLayout;

/// What a spatializing [`PathEffect`] renders to (upstream `PathEffectSettings::speakerLayout` and `hrtf`).
#[derive(Debug, Clone, Copy)]
pub struct PathSpatialization<'a> {
    /// The speakers panned to when not binaural.
    pub speaker_layout: &'a SpeakerLayout,
    /// Sets the length of the HRTFs convolved with when binaural.
    pub hrtf: &'a HrtfDatabase,
}

/// How a [`PathEffect`] is built (upstream `PathEffectSettings`).
#[derive(Debug, Clone, Copy)]
pub struct PathEffectSettings<'a> {
    pub max_order: usize,
    /// `None`: output Ambisonics of `max_order`. `Some`: spatialize to speakers or binaurally.
    pub spatialization: Option<PathSpatialization<'a>>,
}

/// The sound field to render in one frame (upstream `PathEffectParams`).
#[derive(Debug, Clone, Copy)]
pub struct PathEffectParams<'a> {
    pub eq_coeffs: &'a [f32; NUM_BANDS],
    /// At least `num_coeffs_for_order(order)` coefficients.
    pub sh_coeffs: &'a [f32],
    pub order: usize,
    /// Spatializing only: convolve with an HRTF (needs `hrtf`) instead of panning to the speaker layout.
    pub binaural: bool,
    /// Spatializing binaurally only: the HRTF whose Ambisonics HRTFs are blended. It must have the length of the one
    /// the effect was built with.
    pub hrtf: Option<&'a HrtfDatabase>,
    /// Spatializing only: the listener's orientation, which the sound field is rotated into.
    pub listener: CoordinateSpace3,
    /// Spatializing only: normalize the EQ gains to a maximum of 1 first (see [`EqEffect::normalize_gains`]).
    pub normalize_eq: bool,
}

/// The state of a spatializing effect.
#[derive(Debug, Clone)]
struct Spatializer {
    /// Built for one-sample frames, like upstream, so each frame is rotated by the previous frame's orientation.
    rotate: AmbisonicsRotateEffect,
    panning: AmbisonicsPanningEffect,
    overlap_add: OverlapAddConvolutionEffect,
    /// One sample per coefficient of the maximum order. Coefficients above a frame's order keep earlier values.
    ambisonics: AudioBuffer,
    rotated: AudioBuffer,
    speaker_gains: Vec<f32>,
    /// The blended HRTF of each ear.
    hrtf: [Vec<Complex>; NUM_EARS],
}

/// Renders the EQ and SH coefficients of a path simulation (upstream `PathEffect`).
#[derive(Debug, Clone)]
pub struct PathEffect {
    max_order: usize,
    frame_size: usize,
    eq_buffer: AudioBuffer,
    eq: EqEffect,
    /// One per SH coefficient, or one per speaker when spatializing.
    gains: Vec<GainEffect>,
    spatializer: Option<Spatializer>,
    prev_binaural: bool,
}

impl PathEffect {
    pub fn new(audio_settings: &AudioSettings, settings: &PathEffectSettings<'_>) -> Self {
        let max_order = settings.max_order;
        let num_coeffs = sh::num_coeffs_for_order(max_order);

        let spatializer = settings.spatialization.map(|s| {
            let num_speakers = s.speaker_layout.num_speakers();
            let overlap_add = OverlapAddConvolutionEffect::new(
                audio_settings,
                &OverlapAddConvolutionSettings {
                    num_channels: NUM_EARS,
                    ir_size: s.hrtf.num_samples(),
                },
            );
            let num_bins = s.hrtf.num_spectrum_samples();
            Spatializer {
                rotate: AmbisonicsRotateEffect::new(
                    &AudioSettings::new(audio_settings.sampling_rate, 1),
                    max_order,
                ),
                panning: AmbisonicsPanningEffect::new(audio_settings, s.speaker_layout, max_order),
                overlap_add,
                ambisonics: AudioBuffer::new(num_coeffs, 1),
                rotated: AudioBuffer::new(num_coeffs, 1),
                speaker_gains: vec![0.0; num_speakers],
                hrtf: std::array::from_fn(|_| vec![Complex::default(); num_bins]),
            }
        });

        let num_gains = settings
            .spatialization
            .map_or(num_coeffs, |s| s.speaker_layout.num_speakers());

        Self {
            max_order,
            frame_size: audio_settings.frame_size,
            eq_buffer: AudioBuffer::new(1, audio_settings.frame_size),
            eq: EqEffect::new(audio_settings),
            gains: (0..num_gains)
                .map(|_| GainEffect::new(audio_settings))
                .collect(),
            spatializer,
            prev_binaural: false,
        }
    }

    /// Resets the EQ, gains, rotation and convolution. Like upstream, the Ambisonics coefficients kept from earlier
    /// frames stay.
    pub fn reset(&mut self) {
        self.eq.reset();

        if let Some(s) = &mut self.spatializer {
            s.rotate.reset();
            s.panning.reset();
        }

        for gain in &mut self.gains {
            gain.reset();
        }

        if let Some(s) = &mut self.spatializer {
            s.overlap_add.reset();
        }

        self.prev_binaural = false;
    }

    /// Renders one frame of the mono `input` into `out`: `num_coeffs_for_order(max_order)` Ambisonics channels (of
    /// which those above `params.order` are silent), or when spatializing, two channels (binaural) or one per speaker.
    ///
    /// Spatializing, the EQ is normalized only if `params.normalize_eq`, and the overall gain normalization takes out
    /// is dropped; the sound field is rotated by the previous frame's listener (the identity at first, or after a
    /// reset); binaurally, only the first half of the bins of each blended HRTF is filled (upstream accumulates
    /// `numSpectrumSamples` floats of complex spectra), the rest stays zero.
    ///
    /// # Panics
    ///
    /// If `params.order` exceeds the maximum order, a buffer does not hold one frame, the input is not mono, the output
    /// has the wrong number of channels, there are too few SH coefficients, or a binaural frame has no HRTF, one of
    /// another length than the effect's, or an order above 3.
    pub fn apply(
        &mut self,
        params: &PathEffectParams<'_>,
        input: &AudioBuffer<impl AsRef<[f32]>>,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(input.num_channels(), 1, "input must be mono");
        assert_eq!(input.num_samples(), self.frame_size, "frame size");
        assert_eq!(out.num_samples(), self.frame_size, "frame size");
        assert!(
            params.order <= self.max_order,
            "order exceeds the maximum order"
        );
        let num_coeffs = sh::num_coeffs_for_order(params.order);
        assert!(
            params.sh_coeffs.len() >= num_coeffs,
            "one SH coefficient per channel of the order"
        );

        out.make_silent();

        let Some(s) = &mut self.spatializer else {
            assert_eq!(
                out.num_channels(),
                sh::num_coeffs_for_order(self.max_order),
                "one output channel per SH coefficient of the maximum order"
            );

            self.eq.apply(params.eq_coeffs, input, &mut self.eq_buffer);

            for (i, gain) in self.gains.iter_mut().enumerate().take(num_coeffs) {
                gain.apply(
                    params.sh_coeffs[i],
                    &self.eq_buffer,
                    &mut out.channel_view_mut(i),
                );
            }

            self.prev_binaural = false;
            return AudioEffectState::TailComplete;
        };

        let mut eq_gains = *params.eq_coeffs;
        if params.normalize_eq {
            let mut overall_gain = 1.0;
            EqEffect::normalize_gains(&mut eq_gains, &mut overall_gain);
        }
        self.eq.apply(&eq_gains, input, &mut self.eq_buffer);

        for i in 0..num_coeffs {
            s.ambisonics.channel_mut(i)[0] = params.sh_coeffs[i];
        }

        // Upstream rotates the buffer in place.
        s.rotate.process(
            &params.listener,
            params.order,
            &s.ambisonics,
            &mut s.rotated,
        );
        for i in 0..num_coeffs {
            s.ambisonics.channel_mut(i)[0] = s.rotated.channel(i)[0];
        }

        if params.binaural {
            let hrtf = params.hrtf.expect("an HRTF to render binaurally");
            let num_bins = s.hrtf[0].len();
            assert_eq!(
                hrtf.num_spectrum_samples(),
                num_bins,
                "an HRTF of the effect's length"
            );

            for ear in &mut s.hrtf {
                ear.fill(Complex::default());
            }

            let cosine = ((137.9 * DEGREES_TO_RADIANS) / (params.order as f32 + 1.51)).cos();
            let mut i = 0;
            for l in 0..=params.order {
                let scalar = sh::legendre(l, cosine);
                for _ in 0..2 * l + 1 {
                    let hrtf_for_channel = hrtf.ambisonics_hrtf(i);
                    let weight = scalar * s.ambisonics.channel(i)[0];
                    for (ear, input) in s.hrtf.iter_mut().zip(hrtf_for_channel) {
                        scale_accumulate_floats(num_bins, input, weight, ear);
                    }
                    i += 1;
                }
            }

            self.prev_binaural = true;

            let [left, right] = &s.hrtf;
            s.overlap_add
                .apply(&[left, right], false, &self.eq_buffer, out)
        } else {
            assert_eq!(
                out.num_channels(),
                s.speaker_gains.len(),
                "one output channel per speaker"
            );

            s.panning
                .pan_sample(s.ambisonics.as_slice(), &mut s.speaker_gains);

            for (i, gain) in self.gains.iter_mut().enumerate() {
                gain.apply(
                    s.speaker_gains[i],
                    &self.eq_buffer,
                    &mut out.channel_view_mut(i),
                );
            }

            self.prev_binaural = false;
            AudioEffectState::TailComplete
        }
    }

    /// Outputs the next frame of the binaural convolution's tail, if the last frame was binaural; silence otherwise.
    pub fn tail(
        &mut self,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        out.make_silent();

        match &mut self.spatializer {
            Some(s) if self.prev_binaural => s.overlap_add.tail(out),
            _ => AudioEffectState::TailComplete,
        }
    }

    pub fn num_tail_samples_remaining(&self) -> usize {
        match &self.spatializer {
            Some(s) if self.prev_binaural => s.overlap_add.num_tail_samples_remaining(),
            _ => 0,
        }
    }
}

/// Upstream's `ArrayMath::scaleAccumulate` of `num_floats` floats over complex spectra: `out += scalar * input` for the
/// real and imaginary parts in order, stopping after `num_floats` of them.
fn scale_accumulate_floats(num_floats: usize, input: &[Complex], scalar: f32, out: &mut [Complex]) {
    for k in 0..num_floats {
        let (x, o) = (&input[k / 2], &mut out[k / 2]);
        if k % 2 == 0 {
            o.re += scalar * x.re;
        } else {
            o.im += scalar * x.im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambisonics_output_scales_the_equalized_input_by_each_coefficient() {
        let audio = AudioSettings::new(48000, 64);
        let mut effect = PathEffect::new(
            &audio,
            &PathEffectSettings {
                max_order: 1,
                spatialization: None,
            },
        );
        let input = AudioBuffer::from_planar(1, 64, vec![0.5; 64]);
        let mut out = AudioBuffer::new(4, 64);
        let sh_coeffs = [1.0, 0.5, -0.25, 0.0];
        let params = PathEffectParams {
            eq_coeffs: &[1.0; NUM_BANDS],
            sh_coeffs: &sh_coeffs,
            order: 1,
            binaural: false,
            hrtf: None,
            listener: CoordinateSpace3::default(),
            normalize_eq: false,
        };
        assert_eq!(
            effect.apply(&params, &input, &mut out),
            AudioEffectState::TailComplete
        );
        for (i, &c) in sh_coeffs.iter().enumerate() {
            let x = out.channel(i)[63];
            assert!((x - 0.5 * c).abs() < 1e-3, "channel {i}: {x}");
        }
        assert_eq!(effect.num_tail_samples_remaining(), 0);
    }
}
