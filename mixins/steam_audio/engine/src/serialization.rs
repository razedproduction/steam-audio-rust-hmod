//! A FlatBuffers writer and reader for the schemas Steam Audio serializes (upstream `serialized_object` and the
//! `.fbs` files in `core/src/core`), without `unsafe` (ADR 0002).
//!
//! [`Builder`] reproduces upstream's `flatbuffers::FlatBufferBuilder` 1.12 operation for operation (back-to-front
//! layout, alignment padding, defaults left out, vtable deduplication), so the port writes the same bytes as upstream.
//! [`Table`] and [`Vector`] read any valid FlatBuffer; every access is bounds-checked and malformed data is an
//! [`Error::MalformedSerializedData`].

use crate::error::{Error, Result};

/// The position of a finished object: its distance from the end of the buffer (upstream `Offset<T>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Offset(u32);

/// Builds a FlatBuffer back to front (upstream `FlatBufferBuilder` with default options: vtables deduplicated,
/// fields equal to their default left out).
#[derive(Debug)]
pub(crate) struct Builder {
    /// The bytes written so far are `buf[head..]`.
    buf: Vec<u8>,
    head: usize,
    minalign: usize,
    /// Offsets of the vtables written so far, oldest first.
    vtables: Vec<u32>,
    /// `(offset, vtable slot)` of each field of the table being built.
    fields: Vec<(u32, u16)>,
    max_voffset: u16,
    nested: bool,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    /// Upstream's `SerializedObject` starts its builder with 1024 bytes.
    const INITIAL_SIZE: usize = 1024;

    pub(crate) fn new() -> Self {
        Self {
            buf: vec![0; Self::INITIAL_SIZE],
            head: Self::INITIAL_SIZE,
            minalign: 1,
            vtables: Vec::new(),
            fields: Vec::new(),
            max_voffset: 0,
            nested: false,
        }
    }

    /// Number of bytes written (upstream `GetSize`).
    fn size(&self) -> u32 {
        u32::try_from(self.buf.len() - self.head).expect("FlatBuffers are limited to 2 GiB")
    }

    /// Index into `buf` of the byte at `offset` from the end (upstream `data_at`).
    fn at(&self, offset: u32) -> usize {
        self.buf.len() - offset as usize
    }

    /// Makes room for `n` more bytes in front of the data and returns them (upstream `make_space`).
    fn make_space(&mut self, n: usize) -> &mut [u8] {
        if self.head < n {
            let used = self.buf.len() - self.head;
            let new_len = (self.buf.len() * 2).max(used + n);
            let mut buf = vec![0; new_len];
            buf[new_len - used..].copy_from_slice(&self.buf[self.head..]);
            self.head = new_len - used;
            self.buf = buf;
        }
        self.head -= n;
        &mut self.buf[self.head..self.head + n]
    }

    fn push_bytes(&mut self, bytes: &[u8]) {
        self.make_space(bytes.len()).copy_from_slice(bytes);
    }

    fn fill(&mut self, n: usize) {
        self.make_space(n).fill(0);
    }

    fn track_min_align(&mut self, alignment: usize) {
        self.minalign = self.minalign.max(alignment);
    }

    /// Upstream `PaddingBytes`. `alignment` need not be a power of two: upstream masks with `alignment - 1` anyway.
    fn padding(size: usize, alignment: usize) -> usize {
        (!size).wrapping_add(1) & (alignment - 1)
    }

    fn align(&mut self, alignment: usize) {
        self.track_min_align(alignment);
        self.fill(Self::padding(self.size() as usize, alignment));
    }

    /// Pads so that an object of `alignment` can follow `len` more bytes without padding (upstream `PreAlign`).
    fn pre_align(&mut self, len: usize, alignment: usize) {
        self.track_min_align(alignment);
        self.fill(Self::padding(self.size() as usize + len, alignment));
    }

    /// Upstream `PushElement`: aligns, writes, returns the size.
    fn push_u32(&mut self, value: u32) -> u32 {
        self.align(4);
        self.push_bytes(&value.to_le_bytes());
        self.size()
    }

    /// Converts an offset from the end into one relative to where it is about to be stored (upstream `ReferTo`).
    fn refer_to(&mut self, offset: Offset) -> u32 {
        self.align(4);
        debug_assert!(offset.0 != 0 && offset.0 <= self.size());
        self.size() - offset.0 + 4
    }

    fn start_vector(&mut self, len: usize, elem_size: usize) {
        assert!(!self.nested, "objects cannot be nested");
        self.nested = true;
        self.pre_align(len * elem_size, 4);
        self.pre_align(len * elem_size, elem_size);
    }

    fn end_vector(&mut self, len: usize) -> Offset {
        self.nested = false;
        Offset(self.push_u32(u32::try_from(len).expect("vector too long")))
    }

    /// A vector of `len` scalars of `elem_size` bytes, written into the bytes `write` is given (upstream
    /// `CreateVector(const T*, size_t)`).
    pub(crate) fn create_vector(
        &mut self,
        len: usize,
        elem_size: usize,
        write: impl FnOnce(&mut [u8]),
    ) -> Offset {
        self.start_vector(len, elem_size);
        write(self.make_space(len * elem_size));
        self.end_vector(len)
    }

    /// A vector of `len` structs of `struct_size` bytes aligned to 4 (upstream `CreateVectorOfStructs`, which
    /// starts the vector with elements of the struct's alignment).
    pub(crate) fn create_vector_of_structs(
        &mut self,
        len: usize,
        struct_size: usize,
        write: impl FnOnce(&mut [u8]),
    ) -> Offset {
        const ALIGNMENT: usize = 4;
        self.start_vector(len * struct_size / ALIGNMENT, ALIGNMENT);
        write(self.make_space(len * struct_size));
        self.end_vector(len)
    }

    /// A vector of `len` structs of `struct_size` bytes (upstream `CreateUninitializedVectorOfStructs`, which starts
    /// the vector with elements of the struct's *size*: a 12-byte struct pads to a "multiple" of 12 under upstream's
    /// power-of-two mask and raises the buffer's minimum alignment to 12).
    pub(crate) fn create_uninitialized_vector_of_structs(
        &mut self,
        len: usize,
        struct_size: usize,
        write: impl FnOnce(&mut [u8]),
    ) -> Offset {
        self.create_vector(len, struct_size, write)
    }

    /// A vector of offsets to tables (upstream `CreateVector(const Offset<T>*, size_t)`).
    pub(crate) fn create_vector_of_offsets(&mut self, offsets: &[Offset]) -> Offset {
        self.start_vector(offsets.len(), 4);
        for &offset in offsets.iter().rev() {
            let relative = self.refer_to(offset);
            self.push_u32(relative);
        }
        self.end_vector(offsets.len())
    }

    /// A vector of offsets to tables where `None` is upstream's null offset. Upstream's release builds write a null
    /// offset as `ReferTo(0)` (its debug builds assert), which points at the end of the buffer; so does this.
    pub(crate) fn create_vector_of_optional_offsets(
        &mut self,
        offsets: &[Option<Offset>],
    ) -> Offset {
        self.start_vector(offsets.len(), 4);
        for &offset in offsets.iter().rev() {
            let relative = match offset {
                Some(offset) => self.refer_to(offset),
                None => {
                    self.align(4);
                    self.size() + 4
                }
            };
            self.push_u32(relative);
        }
        self.end_vector(offsets.len())
    }

    /// Upstream `StartTable`.
    pub(crate) fn start_table(&mut self) -> u32 {
        assert!(
            !self.nested && self.fields.is_empty(),
            "objects cannot be nested"
        );
        self.nested = true;
        self.size()
    }

    fn track_field(&mut self, slot: u16, offset: u32) {
        self.fields.push((offset, slot));
        self.max_voffset = self.max_voffset.max(slot);
    }

    /// Adds a 32-bit integer field unless it equals its default (upstream `AddElement<int32_t>`).
    pub(crate) fn add_i32(&mut self, slot: u16, value: i32, default: i32) {
        if value != default {
            let offset = self.push_u32(value as u32);
            self.track_field(slot, offset);
        }
    }

    /// Adds a 32-bit float field unless it equals its default as compared with `==`, so `-0.0` is left out like
    /// `0.0` and NaN is always written (upstream `AddElement<float>`).
    pub(crate) fn add_f32(&mut self, slot: u16, value: f32, default: f32) {
        if value != default {
            let offset = self.push_u32(value.to_bits());
            self.track_field(slot, offset);
        }
    }

    /// Adds a 16-bit integer field unless it equals its default (upstream `AddElement<int16_t>`).
    pub(crate) fn add_i16(&mut self, slot: u16, value: i16, default: i16) {
        if value != default {
            self.align(2);
            self.push_bytes(&value.to_le_bytes());
            let offset = self.size();
            self.track_field(slot, offset);
        }
    }

    /// Adds an 8-bit integer field unless it equals its default (upstream `AddElement<int8_t>`).
    pub(crate) fn add_i8(&mut self, slot: u16, value: i8, default: i8) {
        if value != default {
            self.align(1);
            self.push_bytes(&value.to_le_bytes());
            let offset = self.size();
            self.track_field(slot, offset);
        }
    }

    /// Adds a struct field stored inline, aligned to 4 (upstream `AddStruct`).
    pub(crate) fn add_struct(&mut self, slot: u16, bytes: &[u8]) {
        self.align(4);
        self.push_bytes(bytes);
        let offset = self.size();
        self.track_field(slot, offset);
    }

    /// Adds a field referring to a finished object; `None` leaves it out (upstream `AddOffset`).
    pub(crate) fn add_offset(&mut self, slot: u16, offset: Option<Offset>) {
        if let Some(offset) = offset {
            let relative = self.refer_to(offset);
            let offset = self.push_u32(relative);
            self.track_field(slot, offset);
        }
    }

    /// Writes the table's vtable, or points it at an identical earlier one (upstream `EndTable`).
    pub(crate) fn end_table(&mut self, start: u32) -> Offset {
        assert!(self.nested, "end_table without start_table");
        let vtable_offset_loc = self.push_u32(0);

        self.max_voffset = (self.max_voffset + 2).max(4);
        let vtable_size = self.max_voffset;
        self.fill(usize::from(vtable_size));
        let object_size =
            u16::try_from(vtable_offset_loc - start).expect("tables are limited to 64 KiB");
        let head = self.head;
        self.buf[head + 2..head + 4].copy_from_slice(&object_size.to_le_bytes());
        self.buf[head..head + 2].copy_from_slice(&vtable_size.to_le_bytes());
        for &(offset, slot) in &self.fields {
            let position = (vtable_offset_loc - offset) as u16;
            let i = head + usize::from(slot);
            self.buf[i..i + 2].copy_from_slice(&position.to_le_bytes());
        }
        self.fields.clear();
        self.max_voffset = 0;

        let mut vtable_use = self.size();
        let vtable = head..head + usize::from(vtable_size);
        if let Some(&existing) = self.vtables.iter().find(|&&existing| {
            let i = self.at(existing);
            self.buf[i..i + 2] == self.buf[vtable.start..vtable.start + 2]
                && self.buf[i..i + usize::from(vtable_size)] == self.buf[vtable.clone()]
        }) {
            vtable_use = existing;
            self.head += (self.size() - vtable_offset_loc) as usize;
        }
        if vtable_use == self.size() {
            self.vtables.push(vtable_use);
        }

        let i = self.at(vtable_offset_loc);
        let relative = vtable_use as i32 - vtable_offset_loc as i32;
        self.buf[i..i + 4].copy_from_slice(&relative.to_le_bytes());

        self.nested = false;
        Offset(vtable_offset_loc)
    }

    /// Writes the root offset and returns the finished buffer (upstream `Finish` without a file identifier, then
    /// `SerializedObject::commit`).
    pub(crate) fn finish(mut self, root: Offset) -> Vec<u8> {
        assert!(
            !self.nested && self.fields.is_empty(),
            "finish inside an object"
        );
        self.pre_align(4, self.minalign);
        let relative = self.refer_to(root);
        self.push_u32(relative);
        self.buf.split_off(self.head)
    }
}

fn malformed(what: &'static str) -> Error {
    Error::MalformedSerializedData(what)
}

fn read<const N: usize>(buf: &[u8], pos: usize) -> Result<[u8; N]> {
    pos.checked_add(N)
        .and_then(|end| buf.get(pos..end))
        .map(|bytes| bytes.try_into().expect("slice has N bytes"))
        .ok_or_else(|| malformed("offset out of bounds"))
}

fn read_u16(buf: &[u8], pos: usize) -> Result<u16> {
    read(buf, pos).map(u16::from_le_bytes)
}

fn read_u32(buf: &[u8], pos: usize) -> Result<u32> {
    read(buf, pos).map(u32::from_le_bytes)
}

/// Follows the unsigned offset stored at `pos`.
fn follow(buf: &[u8], pos: usize) -> Result<usize> {
    pos.checked_add(read_u32(buf, pos)? as usize)
        .ok_or_else(|| malformed("offset out of bounds"))
}

/// A table in a FlatBuffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Table<'a> {
    buf: &'a [u8],
    pos: usize,
    vtable: usize,
    vtable_size: usize,
}

impl<'a> Table<'a> {
    /// The root table of a finished buffer (upstream `GetRoot`).
    pub(crate) fn root(buf: &'a [u8]) -> Result<Self> {
        Self::at(buf, follow(buf, 0)?)
    }

    fn at(buf: &'a [u8], pos: usize) -> Result<Self> {
        let relative = i32::from_le_bytes(read(buf, pos)?);
        let vtable = (pos as i64 - i64::from(relative))
            .try_into()
            .map_err(|_| malformed("vtable out of bounds"))?;
        let vtable_size = usize::from(read_u16(buf, vtable)?);
        if vtable_size < 4 || vtable_size % 2 != 0 || vtable + vtable_size > buf.len() {
            return Err(malformed("invalid vtable"));
        }
        let object_size = usize::from(read_u16(buf, vtable + 2)?);
        if object_size < 4 || pos + object_size > buf.len() {
            return Err(malformed("table out of bounds"));
        }
        Ok(Self {
            buf,
            pos,
            vtable,
            vtable_size,
        })
    }

    /// Position of the field in vtable `slot`, if present.
    fn field(&self, slot: u16) -> Result<Option<usize>> {
        let slot = usize::from(slot);
        if slot + 2 > self.vtable_size {
            return Ok(None);
        }
        match read_u16(self.buf, self.vtable + slot)? {
            0 => Ok(None),
            offset => Ok(Some(self.pos + usize::from(offset))),
        }
    }

    pub(crate) fn i32(&self, slot: u16, default: i32) -> Result<i32> {
        match self.field(slot)? {
            Some(pos) => read(self.buf, pos).map(i32::from_le_bytes),
            None => Ok(default),
        }
    }

    pub(crate) fn f32(&self, slot: u16, default: f32) -> Result<f32> {
        match self.field(slot)? {
            Some(pos) => read(self.buf, pos).map(f32::from_le_bytes),
            None => Ok(default),
        }
    }

    pub(crate) fn i16(&self, slot: u16, default: i16) -> Result<i16> {
        match self.field(slot)? {
            Some(pos) => read(self.buf, pos).map(i16::from_le_bytes),
            None => Ok(default),
        }
    }

    pub(crate) fn i8(&self, slot: u16, default: i8) -> Result<i8> {
        match self.field(slot)? {
            Some(pos) => read(self.buf, pos).map(i8::from_le_bytes),
            None => Ok(default),
        }
    }

    /// The bytes of a struct field of `size` bytes, if present.
    pub(crate) fn struct_bytes(&self, slot: u16, size: usize) -> Result<Option<&'a [u8]>> {
        let Some(pos) = self.field(slot)? else {
            return Ok(None);
        };
        pos.checked_add(size)
            .and_then(|end| self.buf.get(pos..end))
            .map(Some)
            .ok_or_else(|| malformed("struct out of bounds"))
    }

    pub(crate) fn table(&self, slot: u16) -> Result<Option<Table<'a>>> {
        self.field(slot)?
            .map(|pos| Table::at(self.buf, follow(self.buf, pos)?))
            .transpose()
    }

    /// A vector of elements of `elem_size` bytes.
    pub(crate) fn vector(&self, slot: u16, elem_size: usize) -> Result<Option<Vector<'a>>> {
        let Some(pos) = self.field(slot)? else {
            return Ok(None);
        };
        let start = follow(self.buf, pos)?;
        let len = read_u32(self.buf, start)? as usize;
        let data = start + 4;
        let bytes = len
            .checked_mul(elem_size)
            .and_then(|n| data.checked_add(n))
            .and_then(|end| self.buf.get(data..end))
            .ok_or_else(|| malformed("vector out of bounds"))?;
        Ok(Some(Vector {
            buf: self.buf,
            data,
            bytes,
            len,
            elem_size,
        }))
    }
}

/// A vector in a FlatBuffer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Vector<'a> {
    buf: &'a [u8],
    data: usize,
    bytes: &'a [u8],
    len: usize,
    elem_size: usize,
}

impl<'a> Vector<'a> {
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The bytes of element `i`. Panics if `i` is out of range.
    pub(crate) fn element(&self, i: usize) -> &'a [u8] {
        &self.bytes[i * self.elem_size..(i + 1) * self.elem_size]
    }

    /// The 4-byte little-endian words of element `i`, in order.
    pub(crate) fn words(&self, i: usize) -> impl Iterator<Item = [u8; 4]> + 'a {
        self.element(i)
            .chunks_exact(4)
            .map(|word| word.try_into().expect("4 bytes"))
    }

    /// The `f32`s of every element, in order.
    pub(crate) fn f32s(&self) -> impl Iterator<Item = f32> + 'a {
        self.bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("4 bytes")))
    }

    /// The `i32`s of every element, in order.
    pub(crate) fn i32s(&self) -> impl Iterator<Item = i32> + 'a {
        self.bytes
            .chunks_exact(4)
            .map(|word| i32::from_le_bytes(word.try_into().expect("4 bytes")))
    }

    /// Element `i` of a vector of tables. Panics if `i` is out of range.
    pub(crate) fn table(&self, i: usize) -> Result<Table<'a>> {
        assert!(i < self.len, "vector index out of range");
        Table::at(self.buf, follow(self.buf, self.data + i * 4)?)
    }

    /// Element `i` of a vector of tables that may hold null offsets (see
    /// [`Builder::create_vector_of_optional_offsets`]): `None` for one pointing at the end of the buffer. Panics if `i`
    /// is out of range.
    pub(crate) fn optional_table(&self, i: usize) -> Result<Option<Table<'a>>> {
        assert!(i < self.len, "vector index out of range");
        let pos = follow(self.buf, self.data + i * 4)?;
        if pos == self.buf.len() {
            return Ok(None);
        }
        Table::at(self.buf, pos).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_with_shared_vtables() {
        let mut builder = Builder::new();
        let mut tables = Vec::new();
        for i in 0..3 {
            let values = builder.create_vector(i, 4, |bytes| {
                for (k, word) in bytes.chunks_exact_mut(4).enumerate() {
                    word.copy_from_slice(&(k as f32).to_le_bytes());
                }
            });
            let start = builder.start_table();
            builder.add_offset(8, Some(values));
            builder.add_i32(6, i as i32, 0);
            builder.add_i32(4, 7, 0);
            tables.push(builder.end_table(start));
        }
        let list = builder.create_vector_of_offsets(&tables);
        let start = builder.start_table();
        builder.add_offset(4, Some(list));
        let root = builder.end_table(start);
        let buf = builder.finish(root);

        let root = Table::root(&buf).unwrap();
        let list = root.vector(4, 4).unwrap().unwrap();
        assert_eq!(list.len(), 3);
        for i in 0..3 {
            let table = list.table(i).unwrap();
            assert_eq!(table.i32(4, 0).unwrap(), 7);
            assert_eq!(table.i32(6, 0).unwrap(), i as i32);
            let values = table.vector(8, 4).unwrap().unwrap();
            assert_eq!(
                values.f32s().collect::<Vec<_>>(),
                (0..i).map(|k| k as f32).collect::<Vec<_>>()
            );
            assert!(table.table(10).unwrap().is_none());
        }
    }

    #[test]
    fn truncated_buffers_are_errors() {
        let mut builder = Builder::new();
        let start = builder.start_table();
        builder.add_i32(4, 1, 0);
        let root = builder.end_table(start);
        let buf = builder.finish(root);
        for len in 0..buf.len() {
            let result = Table::root(&buf[..len]).and_then(|t| t.i32(4, 0));
            assert!(result.is_err(), "length {len}");
        }
    }
}
