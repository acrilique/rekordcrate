// Copyright (c) 2026 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

//! High-level API for Rekordbox device exports.

pub mod layout;
pub mod reader;
pub mod writer;

/// ANLZ column construction shared by `analysis` (FFT) and `cpp` (caller-supplied) paths. Owned
/// here so format knowledge lives in one place regardless of how the bands arrive.
#[cfg(any(feature = "analysis", feature = "cpp"))]
pub mod anlz_build;

#[cfg(feature = "analysis")]
pub mod analysis;

pub use crate::device::reader::DeviceExportReader;
pub use crate::device::writer::{
    AddTrackOutcome, AnlzInput, DeviceExportWriter, TagCategoryId, Track,
};

pub use crate::device::reader::{get_playlists, Playlist, PlaylistFolder, PlaylistNode};
