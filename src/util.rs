// Copyright (c) 2025 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

//! Common types used in multiple modules.

use std::io::{Read, Seek, SeekFrom, Write};

use crate::pdb::string::StringError;
use binrw::{binrw, file_ptr::IntoSeekFrom, BinRead, BinResult, BinWrite, Endian};
use thiserror::Error;

/// Enumerates errors returned by this library.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum RekordcrateError {
    /// Represents a failure to decode a DeviceSQL string.
    #[error(transparent)]
    StringError(#[from] StringError),

    /// Represents a failure to parse input.
    #[error(transparent)]
    ParseError(#[from] binrw::Error),

    /// Represents a failure to validate a constraint.
    #[error("failed integrity constraint: {0}")]
    IntegrityError(&'static str),

    /// Represents an `std::io::Error`.
    #[error(transparent)]
    IOError(#[from] std::io::Error),

    /// Represents an `std::io::Error`.
    #[error("component not loaded")]
    NotLoadedError,
}

/// Type alias for results where the error is a `RekordcrateError`.
pub type RekordcrateResult<T> = std::result::Result<T, RekordcrateError>;

/// Indexed Color identifiers used for memory cues and tracks.
#[binrw]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ColorIndex {
    /// No color.
    #[brw(magic = 0u8)]
    None,
    /// Pink color.
    #[brw(magic = 1u8)]
    Pink,
    /// Red color.
    #[brw(magic = 2u8)]
    Red,
    /// Orange color.
    #[brw(magic = 3u8)]
    Orange,
    /// Yellow color.
    #[brw(magic = 4u8)]
    Yellow,
    /// Green color.
    #[brw(magic = 5u8)]
    Green,
    /// Aqua color.
    #[brw(magic = 6u8)]
    Aqua,
    /// Blue color.
    #[brw(magic = 7u8)]
    Blue,
    /// Purple color.
    #[brw(magic = 8u8)]
    Purple,
}

/// Track file type.
#[binrw]
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum FileType {
    /// Unknown file type.
    #[brw(magic = 0x0u16)]
    Unknown,
    /// MP3.
    #[brw(magic = 0x1u16)]
    Mp3,
    /// M4A.
    #[brw(magic = 0x4u16)]
    M4a,
    /// FLAC.
    #[brw(magic = 0x5u16)]
    Flac,
    /// WAV.
    #[brw(magic = 0xbu16)]
    Wav,
    /// AIFF.
    #[brw(magic = 0xcu16)]
    Aiff,
    /// Value that we haven't seen before.
    Other(u16),
}

/// Parses a sequence of values with `BinRead` from the offsets provided by the iterator.
pub(crate) fn parse_at_offsets<Offset, Value, T, Args, It, Reader>(
    it: It,
) -> impl FnOnce(&mut Reader, Endian, Args) -> BinResult<T>
where
    Offset: IntoSeekFrom,
    Value: for<'a> BinRead<Args<'a> = Args>,
    T: FromIterator<(Offset, Value)>,
    Args: Clone,
    It: IntoIterator<Item = Offset>,
    Reader: Read + Seek,
{
    move |reader, endian, args| {
        let base = reader.stream_position()?;
        it.into_iter()
            .map(|offset| {
                reader.seek(offset.into_seek_from())?;
                let v = Value::read_options(reader, endian, args.clone())?;
                // Restore position after each item.
                reader.seek(SeekFrom::Start(base))?;
                Ok((offset, v))
            })
            .collect()
    }
}

/// Writes a sequence of values with `BinWrite` at the offsets provided by the iterator.
pub(crate) fn write_at_offsets<Offset, Value, T, Args, Writer>(
    t: &T,
    writer: &mut Writer,
    endian: Endian,
    args: Args,
) -> BinResult<()>
where
    Offset: IntoSeekFrom,
    Value: for<'a> BinWrite<Args<'a> = Args>,
    for<'a> &'a T: IntoIterator<Item = (&'a Offset, &'a Value)>,
    Args: Clone,
    Writer: Write + Seek,
{
    let base = writer.stream_position()?;
    t.into_iter().try_for_each(|(offset, v)| {
        writer.seek(offset.into_seek_from())?;
        v.write_options(writer, endian, args.clone())?;
        // Restore position after each item.
        writer.seek(SeekFrom::Start(base))?;
        Ok(())
    })
}

/// align given value to the alignment requirements by the given type.
#[must_use]
pub const fn align_by(alignment: u64, mut offset: u64) -> u64 {
    // This is technically dependent on the compile time ABI
    // but for x86 (which this is likely compiled on), we should be able
    // to assume that
    // the alignment of a type is just the size of its largest
    // member. That likely matches the assumptions made for the 32-bit
    // MCU (Renesas R8A77240D500BG) built into the different CDJ-2000 variants.
    // In either way, its better to overshoot the alignment
    // than to undershoot it. For CDJ-3000s, this assumption
    // is likely also correct since they use a 64-bit ARM CPU (Renesas R8A774C0HA01BG)
    if !offset.is_multiple_of(alignment) {
        offset += alignment - (offset % alignment);
    }
    offset
}

/// A reusable scratch buffer for measuring serialized sizes of values.
///
/// This avoids the need for separate `serialized_size()` implementations that could
/// drift out of sync with the actual `BinWrite` implementation. Instead, we simply
/// write to a temporary buffer and measure the result.
///
/// # Example
///
/// ```
/// use rekordcrate::util::ScratchWriter;
/// use binrw::BinWrite;
///
/// let mut scratch = ScratchWriter::new();
///
/// // Measure the size of a u32
/// let size = scratch.measure_size(&42u32).unwrap();
/// assert_eq!(size, 4);
///
/// // Buffer is reused for subsequent measurements
/// let size2 = scratch.measure_size(&0u64).unwrap();
/// assert_eq!(size2, 8);
/// ```
#[derive(Debug)]
pub struct ScratchWriter {
    buffer: Vec<u8>,
    endian: Endian,
}

impl Default for ScratchWriter {
    fn default() -> Self {
        Self {
            buffer: Vec::new(),
            endian: Endian::Little,
        }
    }
}

impl ScratchWriter {
    /// Create a new `ScratchWriter` with an empty buffer and little-endian byte order.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new `ScratchWriter` with a pre-allocated buffer capacity and little-endian byte order.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            endian: Endian::Little,
        }
    }

    /// Create a new `ScratchWriter` with the specified endianness.
    #[must_use]
    pub fn with_endian(endian: Endian) -> Self {
        Self {
            buffer: Vec::new(),
            endian,
        }
    }

    /// Set the endianness for this `ScratchWriter`.
    pub fn set_endian(&mut self, endian: Endian) {
        self.endian = endian;
    }

    /// Get the current endianness.
    #[must_use]
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// Measure the serialized size of a value by writing it to the scratch buffer.
    ///
    /// The buffer is cleared before each measurement, but its capacity is retained
    /// to avoid repeated allocations.
    pub fn measure_size<T>(&mut self, value: &T) -> BinResult<usize>
    where
        T: for<'a> BinWrite<Args<'a> = ()>,
    {
        self.measure_size_with_endian(value, self.endian)
    }

    /// Measure the serialized size of a value with a specific endianness.
    ///
    /// The buffer is cleared before each measurement, but its capacity is retained
    /// to avoid repeated allocations.
    pub fn measure_size_with_endian<T>(&mut self, value: &T, endian: Endian) -> BinResult<usize>
    where
        T: for<'a> BinWrite<Args<'a> = ()>,
    {
        let (size, _) = self.write_and_measure_with_endian(value, endian)?;
        Ok(size)
    }

    /// Measure the serialized size of a value that requires write arguments.
    ///
    /// The buffer is cleared before each measurement, but its capacity is retained
    /// to avoid repeated allocations.
    pub fn measure_size_with_args<'a, T>(
        &mut self,
        value: &T,
        args: T::Args<'a>,
    ) -> BinResult<usize>
    where
        T: BinWrite,
    {
        self.measure_size_with_args_and_endian(value, args, self.endian)
    }

    /// Measure the serialized size of a value that requires write arguments with a specific endianness.
    ///
    /// The buffer is cleared before each measurement, but its capacity is retained
    /// to avoid repeated allocations.
    pub fn measure_size_with_args_and_endian<'a, T>(
        &mut self,
        value: &T,
        args: T::Args<'a>,
        endian: Endian,
    ) -> BinResult<usize>
    where
        T: BinWrite,
    {
        let (size, _) = self.write_and_measure_with_args_and_endian(value, args, endian)?;
        Ok(size)
    }

    /// Write a value to the scratch buffer and return both the size and the bytes.
    ///
    /// This is useful when you need to both measure and then copy the serialized data.
    /// The returned slice is valid until the next call to any method on this `ScratchWriter`.
    pub fn write_and_measure<T>(&mut self, value: &T) -> BinResult<(usize, &[u8])>
    where
        T: for<'a> BinWrite<Args<'a> = ()>,
    {
        self.write_and_measure_with_endian(value, self.endian)
    }

    /// Write a value to the scratch buffer with a specific endianness and return both the size and the bytes.
    ///
    /// This is useful when you need to both measure and then copy the serialized data.
    /// The returned slice is valid until the next call to any method on this `ScratchWriter`.
    pub fn write_and_measure_with_endian<T>(
        &mut self,
        value: &T,
        endian: Endian,
    ) -> BinResult<(usize, &[u8])>
    where
        T: for<'a> BinWrite<Args<'a> = ()>,
    {
        self.buffer.clear();
        let mut cursor = std::io::Cursor::new(&mut self.buffer);
        value.write_options(&mut cursor, endian, ())?;
        Ok((self.buffer.len(), &self.buffer))
    }

    /// Write a value with arguments to the scratch buffer and return both the size and the bytes.
    ///
    /// This is useful when you need to both measure and then copy the serialized data.
    /// The returned slice is valid until the next call to any method on this `ScratchWriter`.
    pub fn write_and_measure_with_args<'a, T>(
        &mut self,
        value: &T,
        args: T::Args<'a>,
    ) -> BinResult<(usize, &[u8])>
    where
        T: BinWrite,
    {
        self.write_and_measure_with_args_and_endian(value, args, self.endian)
    }

    /// Write a value with arguments to the scratch buffer with a specific endianness and return both the size and the bytes.
    ///
    /// This is useful when you need to both measure and then copy the serialized data.
    /// The returned slice is valid until the next call to any method on this `ScratchWriter`.
    pub fn write_and_measure_with_args_and_endian<'a, T>(
        &mut self,
        value: &T,
        args: T::Args<'a>,
        endian: Endian,
    ) -> BinResult<(usize, &[u8])>
    where
        T: BinWrite,
    {
        self.buffer.clear();
        let mut cursor = std::io::Cursor::new(&mut self.buffer);
        value.write_options(&mut cursor, endian, args)?;
        Ok((self.buffer.len(), &self.buffer))
    }

    /// Get the current buffer contents without clearing.
    ///
    /// Useful if you need to inspect the last serialized value.
    #[must_use]
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use binrw::{
        meta::{ReadEndian, WriteEndian},
        prelude::*,
        Endian,
    };
    use pretty_assertions::assert_eq;
    use pretty_hex::pretty_hex;

    macro_rules! assert_eq_hex {
        ($cond:expr, $expected:expr) => {
            assert_eq!(pretty_hex($cond), pretty_hex($expected));
        };
    }

    pub fn test_roundtrip_with_args<'a, T>(
        bin: &[u8],
        obj: T,
        read_args: <T as binrw::BinRead>::Args<'a>,
        write_args: <T as binrw::BinWrite>::Args<'a>,
    ) where
        <T as binrw::BinRead>::Args<'a>: Clone,
        <T as binrw::BinWrite>::Args<'a>: Clone,
        T: BinRead + BinWrite + PartialEq + core::fmt::Debug + ReadEndian + WriteEndian,
    {
        let endian = Endian::NATIVE;
        // T->binary
        let mut writer = binrw::io::Cursor::new(Vec::with_capacity(bin.len()));
        obj.write_options(&mut writer, endian, write_args.clone())
            .unwrap();
        assert_eq_hex!(&writer.get_ref(), &bin);
        // T->binary->T
        writer.set_position(0);
        let parsed = T::read_options(&mut writer, endian, read_args.clone()).unwrap();
        assert_eq!(parsed, obj);
        // binary->T
        let mut cursor = binrw::io::Cursor::new(bin);
        let parsed = T::read_options(&mut cursor, endian, read_args.clone()).unwrap();
        assert_eq!(parsed, obj);
        // binary->T->binary
        writer.set_position(0);
        parsed
            .write_options(&mut writer, endian, write_args.clone())
            .unwrap();
        assert_eq_hex!(&bin, &writer.get_ref());
    }

    pub fn test_roundtrip<'a, T>(bin: &[u8], obj: T)
    where
        <T as binrw::BinRead>::Args<'a>: Default + Clone,
        <T as binrw::BinWrite>::Args<'a>: Default + Clone,
        T: BinRead + BinWrite + PartialEq + core::fmt::Debug + ReadEndian + WriteEndian,
    {
        test_roundtrip_with_args(
            bin,
            obj,
            <T as binrw::BinRead>::Args::default(),
            <T as binrw::BinWrite>::Args::default(),
        );
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{collections::BTreeMap, io::Cursor};

    #[test]
    fn test_parse_at_offsets() {
        // Little-endian u16 values at offsets 0, 2, 5, 7 with garbage 4th byte.
        const DATA: &[u8] = &[0x01, 0x00, 0x02, 0x00, 0xFF, 0x03, 0x00, 0x04, 0x00];
        const OFFSETS: &[u64] = &[0, 7, 2, 5];
        let mut cursor = Cursor::new(DATA);
        let serialized: BTreeMap<u64, u16> =
            parse_at_offsets(OFFSETS.iter().copied())(&mut cursor, Endian::Little, ()).unwrap();
        let expected = BTreeMap::from([(0u64, 1u16), (7, 4), (2, 2), (5, 3)]);
        assert_eq!(serialized, expected);

        // We also expect the cursor position to be restored.
        assert_eq!(cursor.stream_position().unwrap(), 0);
    }

    #[test]
    fn test_write_at_offsets() {
        // Little-endian u16 values at offsets 0, 2, 5, 7 with skipped 4th byte.
        const EXPECTED: &[u8] = &[0x01, 0x00, 0x02, 0x00, 0x00, 0x03, 0x00, 0x04, 0x00];
        let serialized: BTreeMap<u64, u16> =
            vec![(0, 1), (7, 4), (2, 2), (5, 3)].into_iter().collect();
        let mut cursor = Cursor::new(Vec::with_capacity(EXPECTED.len()));
        write_at_offsets(&serialized, &mut cursor, Endian::Little, ()).unwrap();
        assert_eq!(&cursor.get_ref()[..], EXPECTED);

        // We also expect the cursor position to be restored.
        assert_eq!(cursor.stream_position().unwrap(), 0);
    }

    #[test]
    fn test_scratch_writer_measure_size() {
        let mut scratch = ScratchWriter::new();

        // Measure primitive types
        assert_eq!(scratch.measure_size(&42u8).unwrap(), 1);
        assert_eq!(scratch.measure_size(&42u16).unwrap(), 2);
        assert_eq!(scratch.measure_size(&42u32).unwrap(), 4);
        assert_eq!(scratch.measure_size(&42u64).unwrap(), 8);

        // Buffer capacity should be retained across calls
        let cap_after = scratch.buffer.capacity();
        assert!(cap_after >= 8);
    }

    #[test]
    fn test_scratch_writer_write_and_measure() {
        let mut scratch = ScratchWriter::new();

        let (size, bytes) = scratch.write_and_measure(&0x1234u16).unwrap();
        assert_eq!(size, 2);
        assert_eq!(bytes, &[0x34, 0x12]); // Little-endian (default)

        // Test with explicit big-endian
        let (size, bytes) = scratch
            .write_and_measure_with_endian(&0x1234u16, Endian::Big)
            .unwrap();
        assert_eq!(size, 2);
        assert_eq!(bytes, &[0x12, 0x34]); // Big-endian
    }

    #[test]
    fn test_scratch_writer_with_endian() {
        let mut scratch = ScratchWriter::with_endian(Endian::Big);
        assert_eq!(scratch.endian(), Endian::Big);

        let (size, bytes) = scratch.write_and_measure(&0x1234u16).unwrap();
        assert_eq!(size, 2);
        assert_eq!(bytes, &[0x12, 0x34]); // Big-endian

        // Test set_endian
        scratch.set_endian(Endian::Little);
        assert_eq!(scratch.endian(), Endian::Little);

        let (size, bytes) = scratch.write_and_measure(&0x1234u16).unwrap();
        assert_eq!(size, 2);
        assert_eq!(bytes, &[0x34, 0x12]); // Little-endian
    }

    #[test]
    fn test_scratch_writer_with_capacity() {
        let scratch = ScratchWriter::with_capacity(1024);
        assert!(scratch.buffer.capacity() >= 1024);
    }
}
