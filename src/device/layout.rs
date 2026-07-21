// Copyright (c) 2026 Jan Holthaus <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

//! On-disk layout of a Rekordbox device export: where `export.pdb`, `exportExt.pdb`, the
//! `SETTING.DAT` files and the `PIONEER`/`USBANLZ`/`Contents` directories live relative
//! to the device root. Shared by [`crate::device::DeviceExportReader`] and
//! [`crate::device::DeviceExportWriter`].

use crate::setting::SettingType;
use std::path::{Path, PathBuf};

/// The `*SETTING.DAT` files in a device export, in the order Rekordbox writes them.
pub const DAT_FILES: &[(&str, SettingType)] = &[
    ("DEVSETTING.DAT", SettingType::DevSetting),
    ("DJMMYSETTING.DAT", SettingType::DJMMySetting),
    ("MYSETTING.DAT", SettingType::MySetting),
    ("MYSETTING2.DAT", SettingType::MySetting2),
];

/// On-disk layout of a device export rooted at `root`. Derives all paths from it on demand.
///
/// Exposed so expert callers can locate [`Self::export_pdb`] / [`Self::export_ext_pdb`] and the
/// surrounding `PIONEER` / `Contents` directories directly, for manual inspection or modification
/// outside the high-level reader/writer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    root: PathBuf,
}

impl Layout {
    /// Wrap a device-export root directory.
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// The device root directory this layout was built from.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The `PIONEER` directory.
    #[must_use]
    pub fn pioneer_dir(&self) -> PathBuf {
        self.root.join("PIONEER")
    }

    /// The `PIONEER/rekordbox` directory holding the PDB files.
    #[must_use]
    pub fn rekordbox_dir(&self) -> PathBuf {
        self.pioneer_dir().join("rekordbox")
    }

    /// Path to `export.pdb`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rekordcrate::device::layout::Layout;
    /// let layout = Layout::new("/srv/export".into());
    /// assert!(layout.export_pdb().ends_with("export.pdb"));
    /// ```
    #[must_use]
    pub fn export_pdb(&self) -> PathBuf {
        self.rekordbox_dir().join("export.pdb")
    }

    /// Path to `exportExt.pdb`.
    #[must_use]
    pub fn export_ext_pdb(&self) -> PathBuf {
        self.rekordbox_dir().join("exportExt.pdb")
    }

    /// The `PIONEER/USBANLZ` directory holding per-track analysis files.
    #[must_use]
    pub fn usbanlz_dir(&self) -> PathBuf {
        self.pioneer_dir().join("USBANLZ")
    }

    /// The `Contents` directory holding audio files.
    #[must_use]
    pub fn contents_dir(&self) -> PathBuf {
        self.root.join("Contents")
    }

    /// Path to a `*SETTING.DAT` file by name, under `PIONEER`.
    #[must_use]
    pub fn dat_path(&self, filename: &str) -> PathBuf {
        self.pioneer_dir().join(filename)
    }

    /// The `PIONEER/Artwork` directory (under the `artwork` feature).
    #[cfg(feature = "artwork")]
    #[must_use]
    pub fn artwork_dir(&self) -> PathBuf {
        self.pioneer_dir().join("Artwork")
    }

    /// Path to the 80×80 thumbnail `a{id}.jpg` (under the `artwork` feature).
    #[cfg(feature = "artwork")]
    #[must_use]
    pub fn artwork_file(&self, id: u32) -> PathBuf {
        self.artwork_dir()
            .join(artwork_folder(id))
            .join(format!("a{id}.jpg"))
    }

    /// Path to the 240×240 `a{id}_m.jpg` (under the `artwork` feature).
    #[cfg(feature = "artwork")]
    #[must_use]
    pub fn artwork_m_file(&self, id: u32) -> PathBuf {
        self.artwork_dir()
            .join(artwork_folder(id))
            .join(format!("a{id}_m.jpg"))
    }

    /// Per-track analysis directory `PIONEER/USBANLZ/P{XXX}/{HHHHHHHH}`, keyed by the audio file's
    /// device-relative path. Pioneer hardware ignores the PDB `analyze_path` and recomputes this
    /// directory from the on-drive path, so it must match the algorithm in [`path_hash`].
    ///
    /// Ungated so callers can inspect an existing export's analysis files without enabling ANLZ
    /// generation (mirrors `artwork_folder`).
    ///
    /// `audio_path` is the path relative to the drive root, starting with a leading slash (e.g.
    /// `/Contents/Artist/Album/01 Title.mp3`).
    #[must_use]
    pub fn anlz_dir(&self, audio_path: &str) -> PathBuf {
        let (p_value, hash) = path_hash(audio_path);
        self.usbanlz_dir()
            .join(format!("P{p_value:03X}"))
            .join(format!("{hash:08X}"))
    }

    /// Host path to a track's `ANLZ0000.DAT` (base sections: beatgrid, cues, mono waveforms).
    #[must_use]
    pub fn anlz_dat_file(&self, audio_path: &str) -> PathBuf {
        self.anlz_dir(audio_path).join("ANLZ0000.DAT")
    }

    /// Host path to a track's `ANLZ0000.EXT` (Nexus-era colored waveforms + song structure).
    #[must_use]
    pub fn anlz_ext_file(&self, audio_path: &str) -> PathBuf {
        self.anlz_dir(audio_path).join("ANLZ0000.EXT")
    }

    /// Host path to a track's `ANLZ0000.2EX` (newest 3-band waveforms + per-band calibration).
    #[must_use]
    pub fn anlz_2ex_file(&self, audio_path: &str) -> PathBuf {
        self.anlz_dir(audio_path).join("ANLZ0000.2EX")
    }
}

/// Five-digit shard folder name for artwork `id`: `id/20 + 1`, zero-padded. This is compiled even
/// without the `artwork` feature, as the only thing the feature enables is converting and copying
/// artwork files to their destination.
#[must_use]
pub fn artwork_folder(id: u32) -> String {
    format!("{:05}", id / 20 + 1)
}

/// Device-relative path stored in the PDB `analyze_path` column: the `.DAT` the player loads
/// first; sibling `.EXT`/`.2EX` are found by extension substitution on the same stem. The path
/// is keyed by the audio file's device-relative path (with a leading slash), matching what Pioneer
/// hardware recomputes — see [`path_hash`].
#[must_use]
pub fn anlz_device_path(audio_path: &str) -> String {
    let (p_value, hash) = path_hash(audio_path);
    format!("/PIONEER/USBANLZ/P{p_value:03X}/{hash:08X}/ANLZ0000.DAT")
}

/// Compute the Pioneer `(p_value, hash)` pair for an audio file's device-relative path.
///
/// `audio_path` is the path relative to the drive root, starting with a leading slash (e.g.
/// `/Contents/Artist/Album/01 Title.mp3`). Pioneer CDJ/XDJ hardware ignores the PDB `analyze_path`
/// and recomputes the analysis directory from this hash, so the on-disk layout must match it for
/// waveforms to display. Algorithm reverse-engineered from rekordbox's `CreateAnlzFileFolderPath`:
/// the path is hashed as UTF-16 code units with a custom rolling hash, reduced modulo 200003
/// (prime), and a 7-bit P value is then extracted from scattered bits of the result. The same
/// path always yields the same directory, so re-syncs address the same analysis folder.
///
/// # Panics
/// Never. Characters outside the BMP contribute only their low 16 bits instead of a proper
/// surrogate pair this matches observed rekordbox behavior for the sanitized paths it produces.
#[must_use]
pub fn path_hash(audio_path: &str) -> (u16, u32) {
    let mut hash: u32 = 0;

    for c in audio_path.chars() {
        // Each char is masked to a single UTF-16 code unit instead of being encoded as
        // a surrogate pair. Non-BMP paths therefore collide with unrelated BMP ones. Ceiling:
        // correct for BMP only. Upgrade path: `audio_path.encode_utf16()` and iterate the u16s.
        let code_unit = (c as u32) & 0xFFFF;
        let temp = hash.wrapping_mul(0x5BC9).wrapping_add(code_unit);
        hash = temp.wrapping_mul(0x93B5).wrapping_add(code_unit);
    }

    let hash_result = hash % 0x30D43; // modulo 200003 (prime)

    // The P value is assembled from non-contiguous bits of the hash, in the exact bit order the
    // rekordbox disassembly extracts them.
    let mut p_value: u16 = 0;
    p_value |= (hash_result & 1) as u16; // bit 0  -> bit 0
    p_value |= ((hash_result >> 1) & 2) as u16; // bit 2  -> bit 1
    p_value |= ((hash_result >> 4) & 4) as u16; // bit 6  -> bit 2
    p_value |= ((hash_result >> 4) & 8) as u16; // bit 7  -> bit 3
    p_value |= ((hash_result >> 5) & 0x10) as u16; // bit 9  -> bit 4
    p_value |= ((hash_result >> 8) & 0x20) as u16; // bit 13 -> bit 5
    p_value |= ((hash_result >> 10) & 0x40) as u16; // bit 16 -> bit 6

    (p_value, hash_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real rekordbox exports in `pdb/test_roundtrip.rs` store both the audio `file_path` and the
    /// resulting `analyze_path`. Pioneer hardware recomputes the latter from the former, so
    /// `path_hash` must reproduce it exactly. Each case is `(file_path, P-folder, hash-leaf)`.
    #[test]
    fn path_hash_matches_real_rekordbox_exports() {
        let cases: &[(&str, &str, &str)] = &[
            (
                "/Contents/Loopmasters/UnknownAlbum/Demo Track 1.mp3",
                "P016",
                "0000875E",
            ),
            (
                "/Contents/UnknownArtist/UnknownAlbum/NOISE.wav",
                "P019",
                "00020AA9",
            ),
            (
                "/Contents/UnknownArtist/UnknownAlbum/SINEWAVE.wav",
                "P043",
                "00011517",
            ),
            (
                "/Contents/UnknownArtist/UnknownAlbum/SIREN.wav",
                "P017",
                "00009B77",
            ),
            (
                "/Contents/UnknownArtist/UnknownAlbum/HORN.wav",
                "P021",
                "00006D2B",
            ),
        ];
        for (audio_path, want_folder, want_leaf) in cases {
            let (p, h) = path_hash(audio_path);
            assert_eq!(format!("P{p:03X}"), *want_folder, "folder for {audio_path}");
            assert_eq!(format!("{h:08X}"), *want_leaf, "leaf for {audio_path}");
        }
    }
}
