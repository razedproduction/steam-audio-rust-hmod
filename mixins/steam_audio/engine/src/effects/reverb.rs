//! Parametric reverb (upstream `ReverbEffect`: `reverb_effect.cpp`, `float8_reverb_effect.cpp`).

use crate::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use crate::bands::NUM_BANDS;
use crate::delay::{Allpass, Delay};
use crate::effects::eq::eq_filters;
use crate::effects::tail_state;
use crate::iir::IirFilterer;
use crate::math::{cpp_max, msvc_f32_to_i32};
use crate::random::CRand;
use crate::simulation::Reverb;

/// Number of delay lines in the feedback delay network.
const NUM_DELAYS: usize = 16;

/// Lengths of the allpass filters after the network, in samples.
const ALLPASS_DELAYS: [usize; 4] = [225, 341, 441, 556];

/// Gain the allpasses are created with. Their `reset` zeroes it (R9), so they act as plain delays.
const ALLPASS_GAIN: f32 = 0.5;

/// Delay line `i` is a power of `PRIMES[i]`, so that the lengths are mutually prime.
const PRIMES: [i32; NUM_DELAYS] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];

/// Reverb time the delay lengths are chosen for, in seconds.
const DESIGN_REVERB_TIME: f32 = 10.0;

/// Reverb times are clamped to at least this for filtering (not for the tail length).
const MIN_REVERB_TIME: f32 = 0.1;

/// Samples per step of the allpass chain (upstream's AVX path, R8).
const BLOCK: usize = 8;

/// Floor of the per-delay-line absorption gains, which keeps the absorption filters stable (upstream's
/// `ReverbEffect::sMinAbsorptiveGain`, a mutable global nothing changes).
pub const MIN_ABSORPTIVE_GAIN: f32 = 1e-3;

/// Applies a parametric reverb to a mono signal: a feedback delay network of 16 delay lines mixed by a Hadamard matrix,
/// with per-band absorption in each line, followed by four allpass delays and a tone correction EQ.
///
/// Like upstream on any CPU with AVX, buffers are processed 8 samples per step (R8). Unlike upstream, which overruns
/// its buffers in that case, any frame size works.
#[derive(Debug, Clone)]
pub struct ReverbEffect {
    sampling_rate: u32,
    frame_size: usize,
    delay_values: [usize; NUM_DELAYS],
    delay_lines: [Delay; NUM_DELAYS],
    allpasses: [Allpass; ALLPASS_DELAYS.len()],
    absorptive: Box<[[IirFilterer; NUM_BANDS]; NUM_DELAYS]>,
    tone_correction: [IirFilterer; NUM_BANDS],
    /// One frame per delay line, row by row: what came out of each line, then (summed into row 0) the output.
    x_old: Vec<f32>,
    /// One frame per delay line: what goes into each line.
    x_new: Vec<f32>,
    prev_reverb: Reverb,
    /// Signed like upstream: every [`tail`](Self::tail) decrements it, also past zero.
    num_tail_frames_remaining: i32,
}

impl ReverbEffect {
    /// Draws the 16 delay lengths from `rand` (16 calls to `rand()`, as upstream's constructor makes).
    pub fn new(audio_settings: &AudioSettings, rand: &mut CRand) -> Self {
        let sampling_rate = audio_settings.sampling_rate;
        let frame_size = audio_settings.frame_size;
        let delay_values = delays_for_reverb_time(DESIGN_REVERB_TIME, sampling_rate, rand);

        let mut effect = Self {
            sampling_rate,
            frame_size,
            delay_values,
            delay_lines: delay_values.map(|delay| Delay::new(delay, frame_size)),
            allpasses: ALLPASS_DELAYS.map(|delay| Allpass::new(delay, ALLPASS_GAIN, 0)),
            absorptive: Box::default(),
            tone_correction: Default::default(),
            x_old: vec![0.0; NUM_DELAYS * frame_size],
            x_new: vec![0.0; NUM_DELAYS * frame_size],
            prev_reverb: Reverb::default(),
            num_tail_frames_remaining: 0,
        };
        effect.reset();
        effect
    }

    /// Lengths of the 16 delay lines, in samples.
    pub fn delay_lengths(&self) -> &[usize; NUM_DELAYS] {
        &self.delay_values
    }

    /// Silences the delay lines and allpasses, forgets the previous reverb and the tail.
    ///
    /// Like upstream, the absorption and tone correction filters keep their state.
    pub fn reset(&mut self) {
        for delay in &mut self.delay_lines {
            delay.reset();
        }
        for allpass in &mut self.allpasses {
            allpass.reset();
        }
        self.prev_reverb = Reverb {
            reverb_times: [MIN_REVERB_TIME; NUM_BANDS],
        };
        self.num_tail_frames_remaining = 0;
    }

    /// Reverberates one frame of the mono `input` into channel 0 of `out` (other channels are silenced).
    ///
    /// The tail lasts twice the longest reverb time.
    ///
    /// # Panics
    ///
    /// If the input is not mono, or either buffer does not hold one frame.
    pub fn apply(
        &mut self,
        reverb: &Reverb,
        input: &AudioBuffer<impl AsRef<[f32]>>,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(input.num_channels(), 1, "input must be mono");
        assert_eq!(input.num_samples(), self.frame_size, "frame size");
        assert_eq!(out.num_samples(), self.frame_size, "frame size");

        out.make_silent();

        self.set_filters(&reverb.reverb_times);
        self.process(Some(input.channel(0)), out.channel_mut(0));

        self.prev_reverb = *reverb;

        let max_reverb_time = reverb.reverb_times.iter().fold(0.0, |m, &t| cpp_max(m, t));
        let num_frames =
            ((max_reverb_time * self.sampling_rate as f32) / self.frame_size as f32).ceil();
        // Upstream's `2 * static_cast<int>(...)`: an infinite reverb time gives `INT_MIN`, doubled to 0.
        self.num_tail_frames_remaining = msvc_f32_to_i32(num_frames).wrapping_mul(2);

        tail_state(self.num_tail_frames_remaining > 0)
    }

    /// Applies the last reverb again (upstream `tailApply`).
    pub fn tail_apply(
        &mut self,
        input: &AudioBuffer<impl AsRef<[f32]>>,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        let reverb = self.prev_reverb;
        self.apply(&reverb, input, out)
    }

    /// Outputs the next frame of the tail, as if the input were silent, to channel 0 of `out`.
    ///
    /// # Panics
    ///
    /// If `out` does not hold one frame.
    pub fn tail(
        &mut self,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(out.num_samples(), self.frame_size, "frame size");

        out.make_silent();
        self.process(None, out.channel_mut(0));

        self.num_tail_frames_remaining = self.num_tail_frames_remaining.wrapping_sub(1);
        tail_state(self.num_tail_frames_remaining > 0)
    }

    /// Samples of tail left to output (0 once [`tail`](Self::tail) has been called past the end of the tail).
    pub fn num_tail_samples_remaining(&self) -> usize {
        self.num_tail_frames_remaining.max(0) as usize * self.frame_size
    }

    /// Designs the absorption filters of every delay line and the tone correction for `reverb_times`.
    fn set_filters(&mut self, reverb_times: &[f32; NUM_BANDS]) {
        let clamped = reverb_times.map(|t| cpp_max(MIN_REVERB_TIME, t));

        for (filters, &delay) in self.absorptive.iter_mut().zip(&self.delay_values) {
            let gains = absorptive_gains(&clamped, delay, self.sampling_rate);
            for (filter, iir) in filters
                .iter_mut()
                .zip(eq_filters(&gains, self.sampling_rate))
            {
                filter.set_filter(iir);
            }
        }

        let gains = tone_correction_gains(&clamped);
        for (filter, iir) in self
            .tone_correction
            .iter_mut()
            .zip(eq_filters(&gains, self.sampling_rate))
        {
            filter.set_filter(iir);
        }
    }

    /// Runs the network for one frame, feeding `input` (if any) into every delay line, and writes the output to `out`.
    fn process(&mut self, input: Option<&[f32]>, out: &mut [f32]) {
        let f = self.frame_size;

        for ((row, delay), filters) in self
            .x_old
            .chunks_exact_mut(f)
            .zip(&mut self.delay_lines)
            .zip(self.absorptive.iter_mut())
        {
            delay.get(row);
            for filter in filters {
                filter.apply_in_place(row);
            }
        }

        for s in 0..f {
            let x: [f32; NUM_DELAYS] = std::array::from_fn(|j| self.x_old[j * f + s]);
            for (r, row) in self.x_new.chunks_exact_mut(f).enumerate() {
                row[s] = hadamard_row(r, &x) * 0.25;
            }
        }

        for (row, delay) in self.x_new.chunks_exact_mut(f).zip(&mut self.delay_lines) {
            if let Some(input) = input {
                for (x, &v) in row.iter_mut().zip(input) {
                    *x += v;
                }
            }
            delay.put(row);
        }

        let (sum, rows) = self.x_old.split_at_mut(f);
        for row in rows.chunks_exact(f) {
            for (s, &x) in sum.iter_mut().zip(row) {
                *s += x;
            }
        }
        for s in sum.iter_mut() {
            *s *= 1.0 / NUM_DELAYS as f32;
        }

        let simd_size = f / BLOCK * BLOCK;
        for (o, x) in out[..simd_size]
            .chunks_exact_mut(BLOCK)
            .zip(sum[..simd_size].chunks_exact(BLOCK))
        {
            let mut y: [f32; BLOCK] = x.try_into().expect("block");
            for allpass in &mut self.allpasses {
                y = allpass.apply_block(y);
            }
            o.copy_from_slice(&y);
        }
        // Not reached by upstream, which overruns its buffers instead. The allpasses have zero gain, so samples can go
        // through them one at a time.
        for (o, &x) in out[simd_size..].iter_mut().zip(&sum[simd_size..]) {
            let mut y = x;
            for allpass in &mut self.allpasses {
                y = allpass.apply(y);
            }
            *o = y;
        }

        for filter in &mut self.tone_correction {
            filter.apply_in_place(out);
        }
    }
}

/// Row `r` of the 16 × 16 Sylvester-Hadamard matrix times `x`, summed left to right as upstream writes it out:
/// the sign of `x[k]` is `(-1)^popcount(r & k)`.
fn hadamard_row(r: usize, x: &[f32; NUM_DELAYS]) -> f32 {
    let mut acc = x[0];
    for (k, &v) in x.iter().enumerate().skip(1) {
        if (r & k).count_ones() % 2 == 0 {
            acc += v;
        } else {
            acc -= v;
        }
    }
    acc
}

/// Delay lengths for a reverb time: each line gets a power of its prime near `0.15 × reverb time / 16` seconds plus a
/// random offset of 0–100 samples.
fn delays_for_reverb_time(
    reverb_time: f32,
    sampling_rate: u32,
    rand: &mut CRand,
) -> [usize; NUM_DELAYS] {
    let delay_sum = 0.15 * reverb_time * sampling_rate as f32;
    let delay_min = (delay_sum / NUM_DELAYS as f32) as i32;

    std::array::from_fn(|i| {
        let random_offset = rand.rand() % 101;
        next_power_of_prime(delay_min + random_offset, PRIMES[i])
    })
}

/// The power of `p` nearest to `x` on a log scale.
fn next_power_of_prime(x: i32, p: i32) -> usize {
    let m = ((x as f32).ln() / (p as f32).ln()).round();
    // Upstream's `pow(int, int)` is the `double` overload.
    f64::from(p).powf(f64::from(msvc_f32_to_i32(m))) as usize
}

/// Gain per band for one pass through a delay line of `delay` samples: −60 dB per reverb time.
fn absorptive_gains(
    reverb_times: &[f32; NUM_BANDS],
    delay: usize,
    sampling_rate: u32,
) -> [f32; NUM_BANDS] {
    reverb_times.map(|t| {
        cpp_max(
            (-(6.91 * delay as f32) / (t * sampling_rate as f32)).exp(),
            MIN_ABSORPTIVE_GAIN,
        )
    })
}

/// Gains that flatten the network's spectrum: `1 / sqrt(reverb time)` per band, normalized to a maximum of 1.
fn tone_correction_gains(reverb_times: &[f32; NUM_BANDS]) -> [f32; NUM_BANDS] {
    let mut gains = reverb_times.map(|t| (1.0 / t).sqrt());
    let max_gain = gains.iter().fold(0.0, |m, &g| cpp_max(m, g));
    for g in &mut gains {
        *g /= max_gain;
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hadamard_rows_are_orthogonal() {
        for a in 0..NUM_DELAYS {
            for b in 0..NUM_DELAYS {
                let dot: f32 = (0..NUM_DELAYS)
                    .map(|k| {
                        let mut x = [0.0; NUM_DELAYS];
                        x[k] = 1.0;
                        hadamard_row(a, &x) * hadamard_row(b, &x)
                    })
                    .sum();
                assert_eq!(dot, if a == b { 16.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn delays_are_prime_powers() {
        let effect = ReverbEffect::new(&AudioSettings::new(48000, 256), &mut CRand::default());
        for (&d, &p) in effect.delay_lengths().iter().zip(&PRIMES) {
            let mut x = d;
            while x % p as usize == 0 {
                x /= p as usize;
            }
            assert_eq!(x, 1, "{d} is not a power of {p}");
        }
    }

    /// Frame sizes that are not multiples of 8 (upstream overruns its buffers) work: an impulse comes out after the
    /// shortest delay line (a ring of delay + frame samples) and the allpasses.
    #[test]
    fn frame_sizes_that_are_not_multiples_of_8() {
        let reverb = Reverb {
            reverb_times: [1.0, 0.8, 0.5],
        };
        let frame = 61;
        let mut effect =
            ReverbEffect::new(&AudioSettings::new(24000, frame), &mut CRand::default());
        let first = effect.delay_lengths().iter().min().unwrap()
            + frame
            + ALLPASS_DELAYS.iter().sum::<usize>();

        let mut input = AudioBuffer::new(1, frame);
        input.channel_mut(0)[0] = 1.0;
        let mut y = AudioBuffer::new(1, frame);
        let mut out = Vec::new();
        for i in 0..(first + 10 * frame) / frame {
            effect.apply(&reverb, &input, &mut y);
            out.extend_from_slice(y.channel(0));
            if i == 0 {
                input.make_silent();
            }
        }
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out[..first].iter().all(|x| x.abs() < 1e-6));
        assert!(out[first..].iter().any(|x| x.abs() > 1e-3));
    }
}
