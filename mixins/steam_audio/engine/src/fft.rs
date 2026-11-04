//! Discrete Fourier transforms (upstream `FFT`, PFFFT backend `pffft_fft.cpp`).
//!
//! Sizes are rounded up to the next power of two; callers zero-pad their signals to the rounded size. Forward
//! transforms are unscaled and inverse transforms are scaled by `1 / n`, so a forward transform followed by an
//! inverse transform reproduces the input.
//!
//! steam-audio-rust computes these with `realfft`/`rustfft`, which are not in Horizon's `Cargo.lock` (a native mod's
//! dependency allow list). This copy has its own iterative radix-2 transform with twiddles computed in `f64`; a real
//! transform of `n` samples runs as a complex transform of `n / 2`. Results differ from `rustfft` only in rounding,
//! as `rustfft`'s already differ from PFFFT's (risk R1 in steam-audio-rust's `docs/PORTING_PLAN.md`).
//!
//! Like upstream, each transform owns its scratch buffers, so applying it takes `&mut self` and never allocates.

use std::fmt;

use crate::Complex;
use crate::math::next_pow2;

/// An in-place complex radix-2 transform of a power-of-two size.
#[derive(Clone)]
struct Radix2 {
    n: usize,
    /// `e^(-2πik/n)` for `k` in `0..n / 2`.
    twiddles: Vec<Complex>,
    /// Bit-reversed index of each index.
    reversed: Vec<u32>,
}

impl Radix2 {
    fn new(n: usize) -> Self {
        debug_assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let twiddles = (0..n / 2)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / n as f64;
                Complex::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect();
        let reversed = (0..n as u32)
            .map(|i| {
                if bits == 0 {
                    0
                } else {
                    i.reverse_bits() >> (32 - bits)
                }
            })
            .collect();
        Self {
            n,
            twiddles,
            reversed,
        }
    }

    /// Unscaled forward (`e^-`) or inverse (`e^+`) transform of `data` in place.
    fn process(&self, data: &mut [Complex], inverse: bool) {
        let n = self.n;
        debug_assert_eq!(data.len(), n);
        for (i, &j) in self.reversed.iter().enumerate() {
            let j = j as usize;
            if i < j {
                data.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = n / len;
            for block in data.chunks_exact_mut(len) {
                let (lo, hi) = block.split_at_mut(half);
                for (k, (a, b)) in lo.iter_mut().zip(hi.iter_mut()).enumerate() {
                    let w = self.twiddles[k * stride];
                    let w = if inverse { w.conj() } else { w };
                    let t = *b * w;
                    *b = *a - t;
                    *a += t;
                }
            }
            len <<= 1;
        }
    }
}

/// Fourier transform of real-valued signals (upstream `FFT(n, FFTDomain::Real)`).
///
/// The spectrum has `n / 2 + 1` bins (conjugate-even form): bin 0 is DC and the last bin is Nyquist, both with a
/// zero imaginary part.
#[derive(Clone)]
pub struct RealFft {
    num_real_samples: usize,
    /// The complex transform of `n / 2` (absent for `n == 1`).
    half: Option<Radix2>,
    /// `e^(-2πik/n)` for `k` in `0..n / 2`.
    twiddles: Vec<Complex>,
    work: Vec<Complex>,
}

impl RealFft {
    /// A transform for signals of `size` samples, rounded up to the next power of two. Panics if `size` is 0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "FFT size must be positive");
        let num_real_samples = next_pow2(size);
        let m = num_real_samples / 2;
        let twiddles = (0..m)
            .map(|k| {
                let angle = -2.0 * std::f64::consts::PI * k as f64 / num_real_samples as f64;
                Complex::new(angle.cos() as f32, angle.sin() as f32)
            })
            .collect();
        Self {
            num_real_samples,
            half: (m > 0).then(|| Radix2::new(m)),
            twiddles,
            work: vec![Complex::default(); m],
        }
    }

    /// Size of the time-domain signal: the requested size rounded up to a power of two.
    pub fn num_real_samples(&self) -> usize {
        self.num_real_samples
    }

    /// Number of spectrum bins: `num_real_samples / 2 + 1`.
    pub fn num_complex_samples(&self) -> usize {
        self.num_real_samples / 2 + 1
    }

    /// `e^(-2πik/n)` for `k` in `0..=n / 2`.
    fn twiddle(&self, k: usize) -> Complex {
        self.twiddles
            .get(k)
            .copied()
            .unwrap_or(Complex::new(-1.0, 0.0))
    }

    /// Transforms a real signal of [`num_real_samples`](Self::num_real_samples) samples into its spectrum of
    /// [`num_complex_samples`](Self::num_complex_samples) bins. Panics on a length mismatch.
    pub fn forward(&mut self, signal: &[f32], spectrum: &mut [Complex]) {
        assert_eq!(signal.len(), self.num_real_samples, "signal length");
        assert_eq!(
            spectrum.len(),
            self.num_complex_samples(),
            "spectrum length"
        );
        let Some(half) = &self.half else {
            spectrum[0] = Complex::new(signal[0], 0.0);
            return;
        };
        let m = half.n;
        for (z, pair) in self.work.iter_mut().zip(signal.chunks_exact(2)) {
            *z = Complex::new(pair[0], pair[1]);
        }
        half.process(&mut self.work, false);
        // Even and odd halves: E = (Z[k] + Z*[m-k]) / 2, O = (Z[k] - Z*[m-k]) / 2i; X[k] = E + W^k O.
        let minus_half_i = Complex::new(0.0, -0.5);
        for (k, x) in spectrum.iter_mut().enumerate() {
            let zk = self.work[k % m];
            let zmk = self.work[(m - k) % m].conj();
            let even = (zk + zmk) * 0.5;
            let odd = (zk - zmk) * minus_half_i;
            *x = even + self.twiddle(k) * odd;
        }
        spectrum[0].im = 0.0;
        spectrum[m].im = 0.0;
    }

    /// Transforms a spectrum of [`num_complex_samples`](Self::num_complex_samples) bins back into a real signal,
    /// scaled by `1 / num_real_samples`. The imaginary parts of the DC and Nyquist bins are ignored, as upstream.
    /// Panics on a length mismatch.
    pub fn inverse(&mut self, spectrum: &[Complex], signal: &mut [f32]) {
        assert_eq!(
            spectrum.len(),
            self.num_complex_samples(),
            "spectrum length"
        );
        assert_eq!(signal.len(), self.num_real_samples, "signal length");
        let Some(half) = &self.half else {
            signal[0] = spectrum[0].re;
            return;
        };
        let m = half.n;
        let bin = |k: usize| {
            let mut x = spectrum[k];
            if k == 0 || k == m {
                x.im = 0.0;
            }
            x
        };
        // Z[k] = (X[k] + X*[m-k]) + i W^-k (X[k] - X*[m-k]) = 2 (E[k] + i O[k]).
        let i = Complex::i();
        for k in 0..m {
            let xk = bin(k);
            let xmk = bin(m - k).conj();
            self.work[k] = (xk + xmk) + i * self.twiddle(k).conj() * (xk - xmk);
        }
        half.process(&mut self.work, true);
        let scale = 1.0 / self.num_real_samples as f32;
        for (pair, z) in signal.chunks_exact_mut(2).zip(&self.work) {
            pair[0] = z.re * scale;
            pair[1] = z.im * scale;
        }
    }
}

impl fmt::Debug for RealFft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealFft")
            .field("num_real_samples", &self.num_real_samples)
            .finish_non_exhaustive()
    }
}

/// Fourier transform of complex-valued signals (upstream `FFT(n, FFTDomain::Complex)`).
///
/// Signal and spectrum both have `n` samples.
#[derive(Clone)]
pub struct ComplexFft {
    radix2: Radix2,
}

impl ComplexFft {
    /// A transform for signals of `size` samples, rounded up to the next power of two. Panics if `size` is 0.
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "FFT size must be positive");
        Self {
            radix2: Radix2::new(next_pow2(size)),
        }
    }

    /// Size of the signal and of the spectrum: the requested size rounded up to a power of two (upstream
    /// `numRealSamples` and `numComplexSamples`, which are equal in the complex domain).
    pub fn num_samples(&self) -> usize {
        self.radix2.n
    }

    /// Transforms a complex signal into its spectrum. Panics on a length mismatch.
    pub fn forward(&mut self, signal: &[Complex], spectrum: &mut [Complex]) {
        assert_eq!(signal.len(), self.radix2.n, "signal length");
        spectrum.copy_from_slice(signal);
        self.radix2.process(spectrum, false);
    }

    /// Transforms a spectrum back into a complex signal, scaled by `1 / num_samples`. Panics on a length mismatch.
    pub fn inverse(&mut self, spectrum: &[Complex], signal: &mut [Complex]) {
        assert_eq!(spectrum.len(), self.radix2.n, "spectrum length");
        signal.copy_from_slice(spectrum);
        self.radix2.process(signal, true);

        let scale = 1.0 / self.radix2.n as f32;
        for x in signal {
            x.re *= scale;
            x.im *= scale;
        }
    }
}

impl fmt::Debug for ComplexFft {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ComplexFft")
            .field("num_samples", &self.radix2.n)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send<T: Send>() {}

    #[test]
    fn transforms_are_send() {
        assert_send::<RealFft>();
        assert_send::<ComplexFft>();
    }

    #[test]
    fn sizes_round_up_to_powers_of_two() {
        let real = RealFft::new(20);
        assert_eq!(real.num_real_samples(), 32);
        assert_eq!(real.num_complex_samples(), 17);
        assert_eq!(ComplexFft::new(100).num_samples(), 128);
    }

    /// The textbook DFT, in f64.
    fn dft(signal: &[(f64, f64)]) -> Vec<(f64, f64)> {
        let n = signal.len();
        (0..n)
            .map(|k| {
                signal
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(re, im), (t, &(a, b))| {
                        let angle = -2.0 * std::f64::consts::PI * (k * t) as f64 / n as f64;
                        let (c, s) = (angle.cos(), angle.sin());
                        (re + a * c - b * s, im + a * s + b * c)
                    })
            })
            .collect()
    }

    fn signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 7919 % 113) as f32 / 113.0 - 0.5) * if i % 3 == 0 { 2.0 } else { 1.0 })
            .collect()
    }

    #[test]
    fn real_transform_matches_the_dft() {
        for n in [1, 2, 4, 8, 16, 64, 512, 1024] {
            let x = signal(n);
            let mut fft = RealFft::new(n);
            let mut spectrum = vec![Complex::default(); fft.num_complex_samples()];
            fft.forward(&x, &mut spectrum);
            let expected = dft(&x.iter().map(|&v| (f64::from(v), 0.0)).collect::<Vec<_>>());
            let peak = expected
                .iter()
                .map(|(a, b)| a.hypot(*b))
                .fold(1e-9, f64::max);
            for (k, s) in spectrum.iter().enumerate() {
                let (re, im) = expected[k];
                let err = (f64::from(s.re) - re).hypot(f64::from(s.im) - im);
                assert!(err <= 1e-5 * peak, "n {n} bin {k}: {s} vs {re}+{im}i");
            }
        }
    }

    #[test]
    fn complex_transform_matches_the_dft() {
        for n in [1, 2, 8, 128] {
            let x: Vec<Complex> = signal(2 * n)
                .chunks_exact(2)
                .map(|p| Complex::new(p[0], p[1]))
                .collect();
            let mut fft = ComplexFft::new(n);
            let mut spectrum = vec![Complex::default(); n];
            fft.forward(&x, &mut spectrum);
            let expected = dft(&x
                .iter()
                .map(|c| (f64::from(c.re), f64::from(c.im)))
                .collect::<Vec<_>>());
            for (k, s) in spectrum.iter().enumerate() {
                let (re, im) = expected[k];
                let err = (f64::from(s.re) - re).hypot(f64::from(s.im) - im);
                assert!(err <= 1e-4, "n {n} bin {k}: {s} vs {re}+{im}i");
            }
        }
    }

    #[test]
    fn transforms_round_trip() {
        for size in [1, 2, 4, 8, 16, 1024] {
            let mut real = RealFft::new(size);
            let signal: Vec<f32> = (0..size).map(|i| i as f32 + 1.0).collect();
            let mut spectrum = vec![Complex::default(); real.num_complex_samples()];
            let mut round_trip = vec![0.0; size];
            real.forward(&signal, &mut spectrum);
            real.inverse(&spectrum, &mut round_trip);
            for (a, e) in round_trip.iter().zip(&signal) {
                assert!((a - e).abs() <= 1e-4 * e.abs(), "real {size}: {a} vs {e}");
            }

            let mut complex = ComplexFft::new(size);
            let signal: Vec<Complex> = (0..size)
                .map(|i| Complex::new(i as f32 + 1.0, -(i as f32)))
                .collect();
            let mut spectrum = vec![Complex::default(); size];
            let mut round_trip = vec![Complex::default(); size];
            complex.forward(&signal, &mut spectrum);
            complex.inverse(&spectrum, &mut round_trip);
            for (a, e) in round_trip.iter().zip(&signal) {
                assert!(
                    (*a - *e).norm() <= 1e-4 * e.norm(),
                    "complex {size}: {a} vs {e}"
                );
            }
        }
    }
}
