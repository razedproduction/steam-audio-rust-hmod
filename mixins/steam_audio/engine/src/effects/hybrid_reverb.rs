//! Hybrid reverb: convolution with the early part of an IR plus a parametric reverb for the late part (upstream
//! `HybridReverbEffect`).

use crate::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use crate::bands::NUM_BANDS;
use crate::effects::{
    DelayEffect, EqEffect, GainEffect, OverlapSaveConvolutionEffect,
    OverlapSaveConvolutionSettings, OverlapSaveFir, ReverbEffect, tail_state,
};
use crate::math::Vector3;
use crate::random::CRand;
use crate::sh;
use crate::simulation::Reverb;

/// Gain of the parametric part before EQ normalization.
const PARAMETRIC_GAIN: f32 = 16.0;

/// Which IR the convolution part uses for a frame (upstream's `fftIR` pointer and `fftIRUpdated` flag).
#[derive(Debug)]
pub enum HybridReverbIr<'a> {
    /// No IR (upstream's null `fftIR`): the convolution part is skipped and outputs silence.
    None,
    /// The IR the effect already holds.
    Current,
    /// A new IR, which becomes the current one; the caller's FIR receives the previous IR in exchange.
    Updated(&'a mut OverlapSaveFir),
}

/// Parameters of one [`HybridReverbEffect::apply`] (upstream `HybridReverbEffectParams`).
#[derive(Debug)]
pub struct HybridReverbParams<'a> {
    pub ir: HybridReverbIr<'a>,
    /// IR channels to convolve with.
    pub num_channels: usize,
    /// Reverb times of the parametric part.
    pub reverb: &'a Reverb,
    /// Level of the parametric part per band (normalized like [`EqEffect::normalize_gains`]).
    pub eq_coeffs: &'a [f32; NUM_BANDS],
    /// Delay of the parametric part, in samples: where the convolution part hands over.
    pub delay: usize,
}

/// Convolves a mono input with an Ambisonics IR (overlap-save) and adds, to channel 0, a delayed, equalized parametric
/// reverb of the same input.
#[derive(Debug, Clone)]
pub struct HybridReverbEffect {
    frame_size: usize,
    convolution: OverlapSaveConvolutionEffect,
    parametric: ReverbEffect,
    eq: EqEffect,
    gain: GainEffect,
    delay: DelayEffect,
    delay_temp: AudioBuffer,
    eq_temp: AudioBuffer,
    gain_temp: AudioBuffer,
    reverb_temp: AudioBuffer,
    convolution_state: AudioEffectState,
    parametric_state: AudioEffectState,
    eq_state: AudioEffectState,
    gain_state: AudioEffectState,
    delay_state: AudioEffectState,
}

impl HybridReverbEffect {
    /// The parametric part's delay line holds up to twice the IR size. Its reverb draws its delay lengths from `rand`
    /// (see [`ReverbEffect::new`]).
    pub fn new(
        audio_settings: &AudioSettings,
        settings: &OverlapSaveConvolutionSettings,
        rand: &mut CRand,
    ) -> Self {
        let frame_size = audio_settings.frame_size;
        Self {
            frame_size,
            convolution: OverlapSaveConvolutionEffect::new(audio_settings, settings),
            parametric: ReverbEffect::new(audio_settings, rand),
            eq: EqEffect::new(audio_settings),
            gain: GainEffect::new(audio_settings),
            delay: DelayEffect::new(audio_settings, 2 * settings.ir_size),
            delay_temp: AudioBuffer::new(1, frame_size),
            eq_temp: AudioBuffer::new(1, frame_size),
            gain_temp: AudioBuffer::new(1, frame_size),
            reverb_temp: AudioBuffer::new(1, frame_size),
            convolution_state: AudioEffectState::TailComplete,
            parametric_state: AudioEffectState::TailComplete,
            eq_state: AudioEffectState::TailComplete,
            gain_state: AudioEffectState::TailComplete,
            delay_state: AudioEffectState::TailComplete,
        }
    }

    /// Resets every part (with their quirks: the convolution's IR is silenced, the EQ and reverb filters keep their
    /// state).
    pub fn reset(&mut self) {
        self.convolution.reset();
        self.parametric.reset();
        self.eq.reset();
        self.gain.reset();
        self.delay.reset();

        self.convolution_state = AudioEffectState::TailComplete;
        self.parametric_state = AudioEffectState::TailComplete;
        self.eq_state = AudioEffectState::TailComplete;
        self.gain_state = AudioEffectState::TailComplete;
        self.delay_state = AudioEffectState::TailComplete;
    }

    /// Processes one frame of the mono `input` into `out`, which has the convolution's channel count.
    ///
    /// # Panics
    ///
    /// If the input is not mono, either buffer does not hold one frame, or as
    /// [`OverlapSaveConvolutionEffect::apply`].
    pub fn apply(
        &mut self,
        params: HybridReverbParams<'_>,
        input: &AudioBuffer<impl AsRef<[f32]>>,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(input.num_channels(), 1, "input must be mono");
        assert_eq!(input.num_samples(), self.frame_size, "frame size");
        assert_eq!(out.num_samples(), self.frame_size, "frame size");

        self.convolution_state = match params.ir {
            HybridReverbIr::None => {
                out.make_silent();
                AudioEffectState::TailComplete
            }
            HybridReverbIr::Current => {
                self.convolution
                    .apply(None, params.num_channels, input, out)
            }
            HybridReverbIr::Updated(fir) => {
                self.convolution
                    .apply(Some(fir), params.num_channels, input, out)
            }
        };

        let mut eq_coeffs = *params.eq_coeffs;
        let mut gain = PARAMETRIC_GAIN;
        EqEffect::normalize_gains(&mut eq_coeffs, &mut gain);

        self.delay_state = self.delay.apply(params.delay, input, &mut self.delay_temp);
        self.eq_state = self
            .eq
            .apply(&eq_coeffs, &self.delay_temp, &mut self.eq_temp);
        self.gain_state = self.gain.apply(gain, &self.eq_temp, &mut self.gain_temp);
        self.parametric_state =
            self.parametric
                .apply(params.reverb, &self.gain_temp, &mut self.reverb_temp);

        self.add_parametric(out);

        self.state()
    }

    /// Outputs the next frame of the tail into `out`: the convolution's tail, plus the parametric chain's (the first
    /// part with a tail left outputs it, and the later parts process it).
    ///
    /// # Panics
    ///
    /// If `out` does not hold one frame, or as [`OverlapSaveConvolutionEffect::tail`].
    pub fn tail(
        &mut self,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(out.num_samples(), self.frame_size, "frame size");

        out.make_silent();

        if self.convolution_state == AudioEffectState::TailRemaining {
            self.convolution_state = self.convolution.tail(out);
        }

        let remaining = AudioEffectState::TailRemaining;
        if self.parametric_state == remaining
            || self.eq_state == remaining
            || self.gain_state == remaining
            || self.delay_state == remaining
        {
            if self.delay_state == remaining {
                self.delay_state = self.delay.tail(&mut self.delay_temp);
                self.eq_state = self.eq.tail_apply(&self.delay_temp, &mut self.eq_temp);
                self.gain_state = self.gain.tail_apply(&self.eq_temp, &mut self.gain_temp);
                self.parametric_state = self
                    .parametric
                    .tail_apply(&self.gain_temp, &mut self.reverb_temp);
            } else if self.eq_state == remaining {
                self.eq_state = self.eq.tail(&mut self.eq_temp);
                self.gain_state = self.gain.tail_apply(&self.eq_temp, &mut self.gain_temp);
                self.parametric_state = self
                    .parametric
                    .tail_apply(&self.gain_temp, &mut self.reverb_temp);
            } else if self.gain_state == remaining {
                self.gain_state = self.gain.tail(&mut self.gain_temp);
                self.parametric_state = self
                    .parametric
                    .tail_apply(&self.gain_temp, &mut self.reverb_temp);
            } else {
                self.parametric_state = self.parametric.tail(&mut self.reverb_temp);
            }

            self.add_parametric(out);
        }

        self.state()
    }

    /// Samples of tail left in the longest of the parts.
    pub fn num_tail_samples_remaining(&self) -> usize {
        [
            self.convolution.num_tail_samples_remaining(),
            self.parametric.num_tail_samples_remaining(),
            self.delay.num_tail_samples_remaining(),
            self.eq.num_tail_samples_remaining(),
            self.gain.num_tail_samples_remaining(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }

    /// Scales the parametric reverb by the order-0 spherical harmonic and adds it to channel 0 of `out`.
    fn add_parametric(&mut self, out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>) {
        let scalar = sh::evaluate(0, 0, Vector3::ZERO);
        let reverb = self.reverb_temp.channel_mut(0);
        for r in reverb.iter_mut() {
            *r *= scalar;
        }
        for (o, &r) in out.channel_mut(0).iter_mut().zip(&*reverb) {
            *o += r;
        }
    }

    fn state(&self) -> AudioEffectState {
        let remaining = AudioEffectState::TailRemaining;
        tail_state(
            self.convolution_state == remaining
                || self.parametric_state == remaining
                || self.eq_state == remaining
                || self.gain_state == remaining
                || self.delay_state == remaining,
        )
    }
}
