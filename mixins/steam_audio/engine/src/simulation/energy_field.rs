//! Energy fields (upstream `energy_field.{h,cpp}`): the energy arriving at a listener over time, per Ambisonics channel
//! and frequency band, in bins of [`EnergyField::BIN_DURATION`]. Reflection simulation produces them and the
//! [`Reconstructor`](super::Reconstructor) turns them into impulse responses.

use crate::array_math;
use crate::bands::NUM_BANDS;
use crate::error::{Error, Result};
use crate::serialization::{Builder, Offset, Table};
use crate::sh::num_coeffs_for_order;

// Vtable slots of `ipl.Serialized.EnergyField`.
const VT_NUM_CHANNELS: u16 = 4;
const VT_NUM_BINS: u16 = 6;
const VT_DATA: u16 = 8;

/// A `channels × bands × bins` array of energies (upstream `EnergyField`). The default value is empty.
///
/// Upstream's `swap` is [`std::mem::swap`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EnergyField {
    num_channels: usize,
    num_bins: usize,
    data: Vec<f32>,
}

impl EnergyField {
    /// Duration of one bin, in seconds.
    pub const BIN_DURATION: f32 = 1e-2;

    /// A silent field of `ceil(duration / BIN_DURATION)` bins for an Ambisonics `order`.
    pub fn new(duration: f32, order: usize) -> Self {
        let num_bins = (duration / Self::BIN_DURATION).ceil() as usize;
        Self::with_size(num_coeffs_for_order(order), num_bins)
    }

    /// A silent field with the given dimensions.
    pub fn with_size(num_channels: usize, num_bins: usize) -> Self {
        Self {
            num_channels,
            num_bins,
            data: vec![0.0; num_channels * NUM_BANDS * num_bins],
        }
    }

    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    pub fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// All energies, channel-major, then band, then bin (upstream `flatData`).
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// All energies, channel-major, then band, then bin (upstream `flatData`).
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// The bins of `band` in `channel` (upstream `field[channel][band]`). Panics if either is out of range.
    pub fn band(&self, channel: usize, band: usize) -> &[f32] {
        let start = self.offset(channel, band);
        &self.data[start..start + self.num_bins]
    }

    /// The bins of `band` in `channel` (upstream `field[channel][band]`). Panics if either is out of range.
    pub fn band_mut(&mut self, channel: usize, band: usize) -> &mut [f32] {
        let start = self.offset(channel, band);
        &mut self.data[start..start + self.num_bins]
    }

    /// Zeroes every energy.
    pub fn reset(&mut self) {
        self.data.fill(0.0);
    }

    /// Copies as many channels and bins of `other` as both fields have; the rest is left as it is.
    pub fn copy_from(&mut self, other: &Self) {
        let n = self.num_bins.min(other.num_bins);
        for i in 0..self.num_channels.min(other.num_channels) {
            for j in 0..NUM_BANDS {
                self.band_mut(i, j)[..n].copy_from_slice(&other.band(i, j)[..n]);
            }
        }
    }

    /// `out = in1 + in2` over the channels and bins all three have; the rest of `out` is left as it is.
    pub fn add(in1: &Self, in2: &Self, out: &mut Self) {
        let num_channels = in1.num_channels.min(in2.num_channels).min(out.num_channels);
        let n = in1.num_bins.min(in2.num_bins).min(out.num_bins);
        for i in 0..num_channels {
            for j in 0..NUM_BANDS {
                array_math::add(
                    &in1.band(i, j)[..n],
                    &in2.band(i, j)[..n],
                    &mut out.band_mut(i, j)[..n],
                );
            }
        }
    }

    /// `self += other` over the channels and bins both have: upstream's `add(self, other, self)`.
    pub(crate) fn add_assign(&mut self, other: &Self) {
        let (num_channels, n) = common_size(self, other);
        for i in 0..num_channels {
            for j in 0..NUM_BANDS {
                for (out, x) in self.band_mut(i, j)[..n]
                    .iter_mut()
                    .zip(&other.band(i, j)[..n])
                {
                    *out += x;
                }
            }
        }
    }

    /// `out = input * scalar` over the channels and bins both have; the rest of `out` is left as it is.
    pub fn scale(input: &Self, scalar: f32, out: &mut Self) {
        let (num_channels, n) = common_size(input, out);
        for i in 0..num_channels {
            for j in 0..NUM_BANDS {
                array_math::scale(&input.band(i, j)[..n], scalar, &mut out.band_mut(i, j)[..n]);
            }
        }
    }

    /// `self = self * scalar`: upstream's `scale(self, scalar, self)`.
    pub(crate) fn scale_in_place(&mut self, scalar: f32) {
        for x in &mut self.data {
            *x *= scalar;
        }
    }

    /// `out += input * scalar` over the channels and bins both have.
    pub fn scale_accumulate(input: &Self, scalar: f32, out: &mut Self) {
        let (num_channels, n) = common_size(input, out);
        for i in 0..num_channels {
            for j in 0..NUM_BANDS {
                array_math::scale_accumulate(
                    &input.band(i, j)[..n],
                    scalar,
                    &mut out.band_mut(i, j)[..n],
                );
            }
        }
    }

    /// Upstream `serializedSize`: the bytes of the two dimensions and the energies, an estimate that leaves out the
    /// FlatBuffers overhead.
    pub fn serialized_size(&self) -> u64 {
        (2 * size_of::<i32>() + self.data.len() * size_of::<f32>()) as u64
    }

    /// Serializes the field as the root of its own FlatBuffer (schema `ipl.Serialized.EnergyField`). Upstream only
    /// embeds energy fields in baked data; this writes the same table upstream's `serialize` adds there.
    pub fn serialize(&self) -> Vec<u8> {
        let mut builder = Builder::new();
        let root = self.serialize_into(&mut builder);
        builder.finish(root)
    }

    /// Reads a field written by [`serialize`](Self::serialize).
    ///
    /// # Errors
    ///
    /// [`Error::MalformedSerializedData`](crate::Error::MalformedSerializedData) if the data is not a valid
    /// serialized field, including one with no channels or no bins (which upstream asserts against).
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Self::from_table(&Table::root(data)?)
    }

    /// Upstream `serialize`.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let data = builder.create_vector(self.data.len(), 4, |bytes| {
            for (out, x) in bytes.chunks_exact_mut(4).zip(&self.data) {
                out.copy_from_slice(&x.to_le_bytes());
            }
        });
        let num_channels = i32::try_from(self.num_channels).expect("channel count fits in i32");
        let num_bins = i32::try_from(self.num_bins).expect("bin count fits in i32");

        let start = builder.start_table();
        builder.add_offset(VT_DATA, Some(data));
        builder.add_i32(VT_NUM_BINS, num_bins, 0);
        builder.add_i32(VT_NUM_CHANNELS, num_channels, 0);
        builder.end_table(start)
    }

    /// Upstream `EnergyField(const Serialized::EnergyField*)`. Energies past the field's size are ignored.
    pub(crate) fn from_table(table: &Table<'_>) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;
        let num_channels = usize::try_from(table.i32(VT_NUM_CHANNELS, 0)?)
            .ok()
            .filter(|&n| n > 0)
            .ok_or(malformed("energy field without channels"))?;
        let num_bins = usize::try_from(table.i32(VT_NUM_BINS, 0)?)
            .ok()
            .filter(|&n| n > 0)
            .ok_or(malformed("energy field without bins"))?;
        let data = table
            .vector(VT_DATA, 4)?
            .ok_or(malformed("energy field without data"))?;

        let total = num_channels
            .checked_mul(NUM_BANDS * num_bins)
            .filter(|&n| n <= data.len())
            .ok_or(malformed("energy field data too short"))?;
        Ok(Self {
            num_channels,
            num_bins,
            data: data.f32s().take(total).collect(),
        })
    }

    fn offset(&self, channel: usize, band: usize) -> usize {
        assert!(channel < self.num_channels, "channel index out of range");
        assert!(band < NUM_BANDS, "band index out of range");
        (channel * NUM_BANDS + band) * self.num_bins
    }
}

fn common_size(a: &EnergyField, b: &EnergyField) -> (usize, usize) {
    (
        a.num_channels.min(b.num_channels),
        a.num_bins.min(b.num_bins),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        let field = EnergyField::new(0.0123, 2);
        assert_eq!((field.num_channels(), field.num_bins()), (9, 2));
        assert_eq!(field.data().len(), 9 * NUM_BANDS * 2);
        assert_eq!(EnergyField::new(0.0, 0).num_bins(), 0);
    }

    #[test]
    fn operations_use_the_common_size() {
        let mut a = EnergyField::with_size(2, 3);
        a.data_mut().fill(1.0);
        let mut out = EnergyField::with_size(1, 4);
        out.data_mut().fill(5.0);
        EnergyField::scale_accumulate(&a, 2.0, &mut out);
        assert_eq!(out.band(0, 1), [7.0, 7.0, 7.0, 5.0]);
    }
}
