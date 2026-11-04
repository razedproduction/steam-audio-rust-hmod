//! A single-precision complex number with the subset of `num_complex::Complex<f32>`'s API the port uses.
//!
//! steam-audio-rust uses `num-complex`, which is not in Horizon's `Cargo.lock`, the allow list of a native mod's
//! dependencies. The layout (`re`, then `im`) and the arithmetic follow `num_complex` so results are unchanged.

use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A complex number in Cartesian form.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[repr(C)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    #[inline]
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// `i`.
    #[inline]
    pub const fn i() -> Self {
        Self::new(0.0, 1.0)
    }

    /// `r · e^(iθ)`.
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }

    /// `|z|²`.
    #[inline]
    pub fn norm_sqr(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// `|z|`, computed with `hypot` as `num_complex` does.
    #[inline]
    pub fn norm(&self) -> f32 {
        self.re.hypot(self.im)
    }

    /// The argument in `(-π, π]`.
    #[inline]
    pub fn arg(&self) -> f32 {
        self.im.atan2(self.re)
    }

    /// `(|z|, arg z)`.
    #[inline]
    pub fn to_polar(&self) -> (f32, f32) {
        (self.norm(), self.arg())
    }

    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Multiplies both parts by `t`.
    #[inline]
    pub fn scale(&self, t: f32) -> Self {
        Self::new(self.re * t, self.im * t)
    }

    /// Divides both parts by `t`.
    #[inline]
    pub fn unscale(&self, t: f32) -> Self {
        Self::new(self.re / t, self.im / t)
    }

    /// `1 / z`.
    #[inline]
    pub fn inv(&self) -> Self {
        let n = self.norm_sqr();
        Self::new(self.re / n, -self.im / n)
    }

    /// `e^z`.
    #[inline]
    pub fn exp(&self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    /// The principal natural logarithm.
    #[inline]
    pub fn ln(&self) -> Self {
        let (r, theta) = self.to_polar();
        Self::new(r.ln(), theta)
    }

    /// The principal square root (as `num_complex::Complex::sqrt`).
    pub fn sqrt(&self) -> Self {
        if self.im == 0.0 {
            if self.re >= 0.0 {
                Self::new(self.re.sqrt(), self.im)
            } else {
                let re = 0.0f32;
                let im = (-self.re).sqrt();
                if self.im.is_sign_positive() {
                    Self::new(re, im)
                } else {
                    Self::new(re, -im)
                }
            }
        } else if self.re == 0.0 {
            let x = (self.im.abs() / 2.0).sqrt();
            if self.im.is_sign_positive() {
                Self::new(x, x)
            } else {
                Self::new(x, -x)
            }
        } else {
            let (r, theta) = self.to_polar();
            Self::from_polar(r.sqrt(), theta / 2.0)
        }
    }

    /// `z^e` for a real exponent.
    pub fn powf(&self, e: f32) -> Self {
        if e == 0.0 {
            return Self::new(1.0, 0.0);
        }
        let (r, theta) = self.to_polar();
        Self::from_polar(r.powf(e), theta * e)
    }

    #[inline]
    pub fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline]
    pub fn is_nan(&self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }
}

impl From<f32> for Complex32 {
    #[inline]
    fn from(re: f32) -> Self {
        Self::new(re, 0.0)
    }
}

impl fmt::Display for Complex32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im < 0.0 || (self.im == 0.0 && self.im.is_sign_negative()) {
            write!(f, "{}-{}i", self.re, -self.im)
        } else {
            write!(f, "{}+{}i", self.re, self.im)
        }
    }
}

impl Add for Complex32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl Sub for Complex32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl Mul for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl Div for Complex32 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        let norm_sqr = rhs.norm_sqr();
        let re = self.re * rhs.re + self.im * rhs.im;
        let im = self.im * rhs.re - self.re * rhs.im;
        Self::new(re / norm_sqr, im / norm_sqr)
    }
}

impl Neg for Complex32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl Add<f32> for Complex32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: f32) -> Self {
        Self::new(self.re + rhs, self.im)
    }
}

impl Sub<f32> for Complex32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: f32) -> Self {
        Self::new(self.re - rhs, self.im)
    }
}

impl Mul<f32> for Complex32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Div<f32> for Complex32 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: f32) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl Mul<Complex32> for f32 {
    type Output = Complex32;
    #[inline]
    fn mul(self, rhs: Complex32) -> Complex32 {
        Complex32::new(self * rhs.re, self * rhs.im)
    }
}

impl Add<Complex32> for f32 {
    type Output = Complex32;
    #[inline]
    fn add(self, rhs: Complex32) -> Complex32 {
        Complex32::new(self + rhs.re, rhs.im)
    }
}

impl Sub<Complex32> for f32 {
    type Output = Complex32;
    #[inline]
    fn sub(self, rhs: Complex32) -> Complex32 {
        Complex32::new(self - rhs.re, -rhs.im)
    }
}

macro_rules! assign_ops {
    ($($trait:ident $method:ident $op:tt $rhs:ty;)+) => {$(
        impl $trait<$rhs> for Complex32 {
            #[inline]
            fn $method(&mut self, rhs: $rhs) {
                *self = *self $op rhs;
            }
        }
    )+};
}

assign_ops! {
    AddAssign add_assign + Complex32;
    SubAssign sub_assign - Complex32;
    MulAssign mul_assign * Complex32;
    DivAssign div_assign / Complex32;
    AddAssign add_assign + f32;
    SubAssign sub_assign - f32;
    MulAssign mul_assign * f32;
    DivAssign div_assign / f32;
}

impl std::iter::Sum for Complex32 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| a + b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_matches_the_textbook_formulas() {
        let a = Complex32::new(1.0, 2.0);
        let b = Complex32::new(3.0, -1.0);
        assert_eq!(a + b, Complex32::new(4.0, 1.0));
        assert_eq!(a - b, Complex32::new(-2.0, 3.0));
        assert_eq!(a * b, Complex32::new(5.0, 5.0));
        let q = (a * b) / b;
        assert!((q - a).norm() < 1e-6);
        assert!((Complex32::new(-4.0, 0.0).sqrt() - Complex32::new(0.0, 2.0)).norm() < 1e-6);
        assert!((Complex32::new(0.0, std::f32::consts::PI).exp() + 1.0).norm() < 1e-6);
    }
}
