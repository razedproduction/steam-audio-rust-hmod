//! Ports of the MSVC C and C++ library random number generators that upstream relies on (risk R2 in
//! `docs/PORTING_PLAN.md`): the C library's `rand()` ([`CRand`]), and, crate-private, `std::default_random_engine`,
//! which is `std::mt19937` on MSVC, with `std::uniform_real_distribution<float>` and
//! `std::uniform_int_distribution<int>` as implemented by MSVC's `<random>` (VS 2022 17.14).

/// The state of the C library's `rand()` as MSVC implements it: a 32-bit LCG returning bits 16–30 of each new state.
///
/// Upstream draws `ReverbEffect`'s delay lengths from `rand()`, whose state MSVC keeps per thread, starting at 1 (as if
/// `srand(1)` had been called). The port has no global state: whatever constructs reverb effects passes a `CRand`, and
/// effects built one after another from the same `CRand` get the delays upstream's effects get when built one after
/// another on one thread. [`CRand::default`] is a fresh thread's state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CRand {
    state: u32,
}

impl CRand {
    /// Largest value [`rand`](Self::rand) returns (`RAND_MAX`).
    pub const RAND_MAX: i32 = 0x7fff;

    /// The state after `srand(seed)`.
    pub const fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    /// The next value in `0..=RAND_MAX` (`rand()`).
    pub fn rand(&mut self) -> i32 {
        self.state = self.state.wrapping_mul(214_013).wrapping_add(2_531_011);
        ((self.state >> 16) & 0x7fff) as i32
    }
}

impl Default for CRand {
    fn default() -> Self {
        Self::new(1)
    }
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// `std::mt19937`. The algorithm is fixed by the C++ standard; [`Mt19937::default`] is a default-constructed engine
/// (seed 5489).
#[derive(Debug, Clone)]
pub(crate) struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    pub(crate) const DEFAULT_SEED: u32 = 5489;

    pub(crate) fn new(seed: u32) -> Self {
        let mut state = [0; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: N }
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `std::uniform_real_distribution<float>(min, max)` applied to this engine. MSVC builds the canonical value
    /// from the top 24 bits of one draw (`generate_canonical<float, 24>`), then scales it as
    /// `u * (max - min) + min`.
    pub(crate) fn uniform_f32(&mut self, min: f32, max: f32) -> f32 {
        let u = (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0);
        u * (max - min) + min
    }

    /// `std::uniform_int_distribution<int>(0, INT_MAX)` applied to this engine. MSVC multiplies one draw by the range
    /// size, 2^31, and keeps the high 32 bits (Lemire's method; nothing is ever rejected for this range), which is the
    /// draw shifted right by one.
    pub(crate) fn uniform_int(&mut self) -> i32 {
        (self.next_u32() >> 1) as i32
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }
}

impl Default for Mt19937 {
    fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MSVC's `rand()` on a fresh thread.
    #[test]
    fn c_rand_sequence() {
        let mut r = CRand::default();
        let values: Vec<i32> = (0..10).map(|_| r.rand()).collect();
        assert_eq!(
            values,
            [
                41, 18467, 6334, 26500, 19169, 15724, 11478, 29358, 26962, 24464
            ]
        );
    }

    /// The C++ standard requires the 10000th draw of a default-constructed `mt19937` to be 4123659995.
    #[test]
    fn ten_thousandth_draw() {
        let mut mt = Mt19937::default();
        let x = (0..10_000).map(|_| mt.next_u32()).last();
        assert_eq!(x, Some(4_123_659_995));
    }

    #[test]
    fn uniform_range() {
        let mut mt = Mt19937::default();
        assert!(
            (0..10_000)
                .map(|_| mt.uniform_f32(-1.0, 1.0))
                .all(|x| (-1.0..1.0).contains(&x))
        );
    }
}
