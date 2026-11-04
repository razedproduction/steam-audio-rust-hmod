//! Deviation models: the attenuation of sound that bends around obstacles (upstream `deviation.{h,cpp}`).

use std::fmt;
use std::sync::Arc;

use crate::Complex;
use crate::bands::{HIGH_CUTOFF_FREQUENCIES, LOW_CUTOFF_FREQUENCIES};
use crate::math::PI;
use crate::propagation_medium::SPEED_OF_SOUND;

/// Gain in band `band` of sound whose path turns by `angle` radians in total, for a
/// [`DeviationModel::Callback`].
pub type DeviationCallback = Arc<dyn Fn(f32, usize) -> f32 + Send + Sync>;

/// How sound is attenuated per band as a function of the total angle its path turns by (upstream `DeviationModel`).
///
/// Upstream stores an optional callback (with a user data pointer) and uses the UTD model without one; here they are
/// the two variants. Two `Callback` models are equal when they share the same `Arc`.
#[derive(Clone, Default)]
pub enum DeviationModel {
    /// Diffraction by a wedge, from the Uniform Theory of Diffraction.
    #[default]
    Utd,
    /// A user-defined function of angle and band.
    Callback(DeviationCallback),
}

impl DeviationModel {
    /// Whether this is the default (UTD) model.
    pub fn is_default(&self) -> bool {
        matches!(self, Self::Utd)
    }

    /// Gain in band `band` for a total deviation of `angle` radians.
    pub fn evaluate(&self, angle: f32, band: usize) -> f32 {
        match self {
            Self::Utd => utd_deviation(angle, band),
            Self::Callback(callback) => callback(angle, band),
        }
    }
}

impl PartialEq for DeviationModel {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Utd, Self::Utd) => true,
            (Self::Callback(a), Self::Callback(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl fmt::Debug for DeviationModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utd => f.write_str("Utd"),
            Self::Callback(callback) => f
                .debug_tuple("Callback")
                .field(&Arc::as_ptr(callback))
                .finish(),
        }
    }
}

/// `std::polar(1.0f, theta)`.
fn polar(theta: f32) -> Complex {
    Complex::new(theta.cos(), theta.sin())
}

/// `float - std::complex<float>` (MSVC's operator subtracts from `(left, 0)`).
fn sub_from(left: f32, right: Complex) -> Complex {
    Complex::new(left - right.re, 0.0 - right.im)
}

/// UTD diffraction by a wedge of exterior angle `n π` with `n = 2` (a half-plane), seen from a distance of 5 cm, at the
/// center frequency of `band` (upstream `utdDeviation`; see <http://www-sop.inria.fr/reves/Nicolas.Tsingos/publis/sig2001.pdf>).
///
/// Panics if `band` is out of range.
fn utd_deviation(angle: f32, band: usize) -> f32 {
    let cot = |theta: f32| 1.0 / theta.tan();

    let n_plus = |n: f32, x: f32| if x <= PI * (n - 1.0) { 0.0 } else { 1.0 };

    let n_minus = |n: f32, x: f32| {
        if x < PI * (1.0 - n) {
            -1.0
        } else if PI * (1.0 - n) <= x && x <= PI * (1.0 + n) {
            0.0
        } else {
            1.0
        }
    };

    let a = |n: f32, beta: f32, big_n: f32| {
        let cosine = ((PI * n * big_n) - (0.5 * beta)).cos();
        2.0 * cosine * cosine
    };

    let fresnel = |x: f32| {
        let e = polar(0.25 * PI * (x / (x + 1.4)).sqrt());
        if x < 0.8 {
            let term1 = (PI * x).sqrt();
            let term2 = 1.0 - (x.sqrt() / (0.7 * x.sqrt() + 1.2));
            term1 * term2 * e
        } else {
            let term1 = 1.0 - (0.8 / ((x + 1.25) * (x + 1.25)));
            term1 * e
        }
    };

    let sign = |x: f32| if x > 0.0 { 1.0 } else { -1.0 };

    let n = 2.0f32;
    let alpha_i = 0.0f32;
    let alpha_d = alpha_i + PI + angle;
    let l_edge = 0.05f32;

    let c = SPEED_OF_SOUND;
    let f = (LOW_CUTOFF_FREQUENCIES[band] + HIGH_CUTOFF_FREQUENCIES[band]) / 2.0;
    let wavelength = c / f;
    let k = (2.0 * PI) / wavelength;

    let e = polar(-0.25 * PI);

    let d0 = e / (2.0 * n * (2.0 * PI * k).sqrt());

    let beta1 = alpha_d - alpha_i;
    let beta2 = alpha_d - alpha_i;
    let beta3 = alpha_d + alpha_i;
    let beta4 = alpha_d + alpha_i;

    let t = [
        cot((PI + beta1) / (2.0 * n)),
        cot((PI - beta2) / (2.0 * n)),
        cot((PI + beta3) / (2.0 * n)),
        cot((PI - beta4) / (2.0 * n)),
    ];

    let big_n = [
        n_plus(n, beta1),
        n_minus(n, beta2),
        n_plus(n, beta3),
        n_minus(n, beta4),
    ];

    let betas = [beta1, beta2, beta3, beta4];

    let epsilon = [
        beta1 - (2.0 * PI * n * big_n[0]) + PI,
        -(beta2 - (2.0 * PI * n * big_n[1]) - PI),
        beta3 - (2.0 * PI * n * big_n[2]) + PI,
        -(beta4 - (2.0 * PI * n * big_n[3]) - PI),
    ];

    let d: [Complex; 4] = std::array::from_fn(|i| {
        if t[i].is_finite() {
            let x = k * l_edge * a(n, betas[i], big_n[i]);
            t[i] * fresnel(x)
        } else {
            let lhs = (2.0 * PI * k * l_edge).sqrt() * sign(epsilon[i]);
            n * e * sub_from(lhs, 2.0 * k * l_edge * epsilon[i] * e)
        }
    });

    let total = d0 * (d[0] + d[1] + d[2] + d[3]);
    total.re.hypot(total.im)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn callbacks_compare_by_identity() {
        let callback: DeviationCallback = Arc::new(|angle, band| angle + band as f32);
        let a = DeviationModel::Callback(callback.clone());
        let b = DeviationModel::Callback(callback);
        assert_eq!(a, b);
        assert_ne!(a, DeviationModel::Callback(Arc::new(|_, _| 0.0)));
        assert!(!a.is_default());
        assert!(DeviationModel::default().is_default());
        assert_eq!(a.evaluate(1.0, 2), 3.0);
    }

    #[test]
    fn utd_attenuates_more_at_high_frequencies() {
        let model = DeviationModel::default();
        let gains: Vec<f32> = (0..3).map(|band| model.evaluate(1.0, band)).collect();
        assert!(gains[0] > gains[1] && gains[1] > gains[2], "{gains:?}");
    }
}
