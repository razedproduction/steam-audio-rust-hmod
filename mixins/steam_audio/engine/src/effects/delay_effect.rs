//! Delay of a mono signal by a whole number of samples (upstream `DelayEffect`).

use crate::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use crate::effects::tail_state;

/// Delays a mono signal through a ring buffer of `max_delay_in_samples` samples.
///
/// Upstream sets up a linear crossfade between the previous and the new delay, but never clears its first-frame flag,
/// so every frame uses the new delay from its first sample; the port does the same (R24).
#[derive(Debug, Clone)]
pub struct DelayEffect {
    frame_size: usize,
    ring: Vec<f32>,
    write_pos: usize,
    num_tail_samples_remaining: usize,
}

impl DelayEffect {
    pub fn new(audio_settings: &AudioSettings, max_delay_in_samples: usize) -> Self {
        Self {
            frame_size: audio_settings.frame_size,
            ring: vec![0.0; max_delay_in_samples],
            write_pos: 0,
            num_tail_samples_remaining: 0,
        }
    }

    /// Silences the ring buffer and forgets the tail.
    pub fn reset(&mut self) {
        self.ring.fill(0.0);
        self.write_pos = 0;
        self.num_tail_samples_remaining = 0;
    }

    /// Delays one frame of the mono `input` by `delay` samples into `out`.
    ///
    /// A delay of at least the maximum outputs silence and leaves the effect unchanged (the input is dropped).
    ///
    /// # Panics
    ///
    /// If either buffer is not mono or does not hold one frame.
    pub fn apply(
        &mut self,
        delay: usize,
        input: &AudioBuffer<impl AsRef<[f32]>>,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(input.num_channels(), 1, "input must be mono");
        assert_eq!(input.num_samples(), self.frame_size, "frame size");
        assert_eq!(out.num_channels(), 1, "output must be mono");
        assert_eq!(out.num_samples(), self.frame_size, "frame size");

        let n = self.ring.len();
        if delay >= n {
            out.make_silent();
            return AudioEffectState::TailComplete;
        }

        // Upstream's interpolation between two samples, with weights 1 and 0 for a whole number of samples (the
        // second term still matters for signed zeros and non-finite samples).
        let delay_f = delay as f32;
        let weight1 = delay_f.ceil() - delay_f;
        let weight0 = 1.0 - weight1;

        for (o, &x) in out.channel_mut(0).iter_mut().zip(input.channel(0)) {
            self.ring[self.write_pos] = x;

            let index0 = (self.write_pos as f32 - delay_f).floor() as isize;
            let i0 = wrap(index0, n);
            let i1 = wrap(index0 + 1, n);

            *o = weight0 * self.ring[i0] + weight1 * self.ring[i1];

            self.write_pos = (self.write_pos + 1) % n;
        }

        self.num_tail_samples_remaining = delay.saturating_sub(self.frame_size);
        tail_state(self.num_tail_samples_remaining > 0)
    }

    /// Outputs the next frame of the tail into `out`.
    ///
    /// Like upstream, the write position does not advance: the frame holds the delayed samples that were still due.
    ///
    /// # Panics
    ///
    /// If `out` is not mono or does not hold one frame.
    pub fn tail(
        &mut self,
        out: &mut AudioBuffer<impl AsRef<[f32]> + AsMut<[f32]>>,
    ) -> AudioEffectState {
        assert_eq!(out.num_channels(), 1, "output must be mono");
        assert_eq!(out.num_samples(), self.frame_size, "frame size");

        let n = self.ring.len();
        if self.num_tail_samples_remaining >= n {
            out.make_silent();
            return AudioEffectState::TailComplete;
        }

        for o in out.channel_mut(0) {
            let index = wrap(
                self.write_pos as isize - self.num_tail_samples_remaining as isize,
                n,
            );
            *o = self.ring[index];
            self.num_tail_samples_remaining = self.num_tail_samples_remaining.saturating_sub(1);
        }

        tail_state(self.num_tail_samples_remaining > 0)
    }

    pub fn num_tail_samples_remaining(&self) -> usize {
        self.num_tail_samples_remaining
    }
}

/// Brings an index at most one ring length outside `0..n` back into it.
fn wrap(index: isize, n: usize) -> usize {
    let n = n as isize;
    let i = if index < 0 {
        index + n
    } else if index >= n {
        index - n
    } else {
        index
    };
    i as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delays_by_whole_samples_and_outputs_the_tail() {
        let mut effect = DelayEffect::new(&AudioSettings::new(48000, 4), 8);
        let input = AudioBuffer::from_planar(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = AudioBuffer::new(1, 4);

        assert_eq!(
            effect.apply(6, &input, &mut out),
            AudioEffectState::TailRemaining
        );
        assert_eq!(out.channel(0), &[0.0; 4]);
        assert_eq!(effect.num_tail_samples_remaining(), 2);

        assert_eq!(effect.tail(&mut out), AudioEffectState::TailComplete);
        assert_eq!(out.channel(0), &[3.0, 4.0, 0.0, 0.0]);

        assert_eq!(
            effect.apply(8, &input, &mut out),
            AudioEffectState::TailComplete
        );
        assert_eq!(out.channel(0), &[0.0; 4]);

        assert_eq!(
            effect.apply(0, &input, &mut out),
            AudioEffectState::TailComplete
        );
        assert_eq!(out.channel(0), &[1.0, 2.0, 3.0, 4.0]);
    }
}
