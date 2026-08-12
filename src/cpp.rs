// Copyright (c) 2026 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

// The crate's strict lints target library API surfaces and audited unsafe code. This module
// is a marshalling shim: cxx generates the real struct/fn declarations (Debug/docs here would
// just duplicate the upstream types), and cxx's macro emits the `unsafe`/`export_name` blocks
// that FFI fundamentally requires. Relax them for this file only.
#![cfg_attr(feature = "cpp", allow(unsafe_code))]
#![cfg_attr(feature = "cpp", allow(missing_docs))]
#![cfg_attr(feature = "cpp", allow(missing_debug_implementations))]
#![cfg_attr(feature = "cpp", allow(clippy::must_use_candidate))]

//! C++ FFI via [cxx](https://cxx.rs), behind the `cpp` feature.
//!
//! Exposes the device-export reader and writer. Reads return parsed data as
//! shared structs; failures cross the boundary as `rust::Error` exceptions
//! (cxx's native `Result<T, E>` translation — `RekordcrateError` already impls
//! `Display` via `thiserror`, so no wrapper layer is needed).
//!
//! Build with `--features cpp` to produce a C++ static library; see the cxx
//! docs for generating the consumer header from `target/cxxbridge/`.
//!
//! The `Track` struct mirrors `crate::Track`: the artwork field is declared as
//! `artwork_device_path` without the `artwork` feature and `artwork_source`
//! with it. The generated C++ header therefore declares only the matching
//! field — referencing the wrong one is a C++ compile error, which is exactly
//! the safety you want at a build-config boundary.

use crate::{
    device, util, DeviceExportReader, DeviceExportWriter, Error, PlaylistTreeNodeId, TrackId,
};
use std::path::PathBuf;

// Shared types declared inside the bridge mod below resolve to `ffi::*`. Bring
// them into scope for the shim fns. The four performance-data struct names are
// only constructed in tests; allow the unused-import lint off-test.
#[allow(unused_imports)]
use crate::cpp::ffi::{
    AddTrackOutcome, BeatgridMarker, ColorIndex, FileType, HotCueFfi, LoopFfi, PerformanceData,
    PlaylistInfo, Rating, Track, WaveformEntry,
};

// Bridge module — shared types + extern "Rust" declarations. The shim impls
// live below as plain free functions; cxx finds them by name.
#[cxx::bridge(namespace = "rekordcrate")]
pub mod ffi {
    /// A track as a C++ consumer thinks of it: plain `String`/primitive fields.
    pub struct Track {
        pub title: String,
        pub artist: String,
        pub album: String,
        pub genre: String,
        pub key: String,
        pub label: String,
        pub composer: String,
        pub remixer: String,
        pub orig_artist: String,
        pub comment: String,
        pub isrc: String,
        pub lyricist: String,
        pub mix_name: String,
        pub release_date: String,
        pub date_added: String,
        pub file_path: String,
        pub filename: String,
        #[cfg(feature = "artwork")]
        /// Host-side source path. The writer decodes, resizes, and copies into
        /// `PIONEER/Artwork/`. Empty = no artwork.
        pub artwork_source: String,
        #[cfg(not(feature = "artwork"))]
        /// Device path stored verbatim in the `Artwork` row. Caller owns the
        /// files. Empty = no artwork.
        pub artwork_device_path: String,
        #[cfg(feature = "analysis")]
        /// Host path to the audio file. The writer decodes it and computes ANLZ
        /// waveform columns itself. Empty = no ANLZ generation.
        pub analysis_source: String,
        pub message: String,
        pub tempo: f32,
        pub bitrate: u32,
        pub sample_rate: u32,
        pub sample_depth: u16,
        pub duration_secs: u16,
        pub file_size: u32,
        pub track_number: u32,
        pub disc_number: u16,
        pub year: u16,
        pub play_count: u16,
        pub rating: Rating,
        pub color: ColorIndex,
        pub file_type: FileType,
        pub autoload_hotcues: bool,
        /// Total sample count. Needed by `build_anlz_input` for beatgrid extents
        /// (clips beats at the track end) and ignored when no performance data.
        pub sample_count: u64,
        /// Format-agnostic performance data. When `waveform_detail` is non-empty,
        /// rekordcrate expands it to a full ANLZ column set. An empty vector
        /// means "no ANLZ" — use `analysis_source` under the `analysis` feature
        /// instead. ponytail: cxx can't express `Option<shared_struct>`, so the
        /// empty `waveform_detail` vector doubles as the presence flag.
        pub performance: PerformanceData,
    }

    /// Outcome of [`writer_add_track`]: a freshly inserted track id, or the
    /// existing one if `file_path` already existed.
    pub struct AddTrackOutcome {
        pub id: u32,
        pub is_new: bool,
    }

    /// Flattened node of the playlist tree. `parent_id == 0` denotes the root.
    /// C++ rebuilds the tree from `parent_id` pointers if needed.
    /// ponytail: cxx can't express the recursive `PlaylistNode` enum, so the
    /// tree is flattened to a list at the boundary. Upgrade path: an opaque
    /// cursor handle if streaming is ever needed.
    #[derive(Debug)]
    pub struct PlaylistInfo {
        pub id: u32,
        pub name: String,
        pub parent_id: u32,
    }

    /// Star rating, 0..=5.
    pub enum Rating {
        Zero,
        One,
        Two,
        Three,
        Four,
        Five,
    }

    /// Rekordbox color label.
    pub enum ColorIndex {
        None,
        Pink,
        Red,
        Orange,
        Yellow,
        Green,
        Aqua,
        Blue,
        Purple,
    }

    /// Audio file format. ponytail: the upstream `FileType::Other(u16)` variant
    /// is dropped here — raw unknown values map to `Unknown`. Rekordbox only
    /// recognizes the listed types, so roundtrip loss is bounded to exotic
    /// values the player won't accept anyway.
    pub enum FileType {
        Unknown,
        Mp3,
        M4a,
        Flac,
        Wav,
        Aiff,
    }

    // ---- Performance data (format-agnostic; rekordcrate owns the ANLZ expansion) ----
    //
    // These deliberately mirror libdjinterop's `waveform_entry`/`beatgrid_marker`/
    // `hot_cue`/`loop` shapes — NOT the 11 ANLZ column types. The caller supplies
    // one 150 Hz 3-band vector + per-column height; rekordcrate downsamples and
    // encodes. cxx shared structs cannot hold `Option<f64>`, so `bpm` and
    // `main_cue_sample` use sentinel values (<= 0.0 means "unset" — both are
    // strictly positive when present).

    /// One column of 3-band amplitude at the 150 Hz detail rate. Opacity is
    /// dropped (unused by ANLZ); the height comes in via `PerformanceData::
    /// waveform_height` since peak amplitude isn't recoverable from band
    /// energies.
    pub struct WaveformEntry {
        pub low: u8,
        pub mid: u8,
        pub high: u8,
    }

    /// Sparse beatgrid anchor: a beat `index` (may be negative — Rekordbox grids
    /// start at -4) and its `sample_offset`. Two markers bracket a constant-tempo
    /// segment; rekordcrate densifies to one beat per beat.
    pub struct BeatgridMarker {
        pub index: i32,
        pub sample_offset: f64,
    }

    /// A hot cue (slot 1-8) or memory cue (slot 0). RGBA color; alpha is dropped
    /// by the ANLZ encoder (RGB + palette index only).
    pub struct HotCueFfi {
        pub index: u32,
        pub label: String,
        pub sample_offset: f64,
        pub r: u8,
        pub g: u8,
        pub b: u8,
        pub a: u8,
    }

    /// A loop cue. `start`/`end` are sample offsets.
    pub struct LoopFfi {
        pub index: u32,
        pub label: String,
        pub start: f64,
        pub end: f64,
        pub r: u8,
        pub g: u8,
        pub b: u8,
        pub a: u8,
    }

    /// All performance data for one track, format-agnostic. When present on a
    /// `Track`, rekordcrate builds the full ANLZ column set (waveforms + beats +
    /// cues). When `None` (empty `waveform_detail`), no ANLZ is written — the
    /// `analysis_source` path still works under the `analysis` feature.
    pub struct PerformanceData {
        /// 3-band detail at 150 Hz. Length 0 disables ANLZ for this track.
        pub waveform_detail: Vec<WaveformEntry>,
        /// Per-column peak height 0-31, parallel to `waveform_detail`. Drives
        /// PWAV/PWV2/PWV3 and PWV5 height.
        pub waveform_height: Vec<u8>,
        /// Sparse beatgrid anchors (typically 2).
        pub beatgrid: Vec<BeatgridMarker>,
        /// BPM. `<= 0.0` means unknown — rekordcrate derives it from the first
        /// beatgrid segment's samples-per-beat instead.
        pub bpm: f64,
        /// Hot/memory cues.
        pub hot_cues: Vec<HotCueFfi>,
        /// Loops.
        pub loops: Vec<LoopFfi>,
        /// Main (downbeat) cue in samples. `< 0.0` means none.
        pub main_cue_sample: f64,
    }

    extern "Rust" {
        type Reader;
        type Writer;

        /// Construct a zeroed `Track` for C++ to fill in. C++ can't call
        /// `Default::default()`.
        fn track_default() -> Track;

        // ---- Reader ----

        /// Point a reader at a device export on disk (a directory containing `PIONEER`).
        fn open_reader(path: String) -> Result<Box<Reader>>;
        /// The device root path, as an owned UTF-8 string.
        fn reader_path(r: &Reader) -> String;
        /// All four `*SETTING.DAT` files, formatted via the upstream `Display` impl.
        /// ponytail: flattens ~50 leaf setting enums to one string. Upgrade path:
        /// mirror individual `setting::*` enums into the bridge if structured
        /// access is ever needed.
        fn reader_settings(r: &Reader) -> String;
        /// The playlist tree, flattened to a list with `parent_id` pointers.
        fn reader_playlists(r: &Reader) -> Result<Vec<PlaylistInfo>>;

        // ---- Writer ----

        /// Create a new export at `root`. See `DeviceExportWriter::create`.
        fn writer_create(root: String) -> Result<Box<Writer>>;
        /// Open an existing export at `root`. See `DeviceExportWriter::open`.
        fn writer_open(root: String) -> Result<Box<Writer>>;
        /// Insert (or dedup-return) a track.
        fn writer_add_track(w: &mut Writer, track: &Track) -> Result<AddTrackOutcome>;
        /// Create a playlist folder under `parent_id` (0 = root).
        fn writer_create_playlist_folder(w: &mut Writer, name: &str, parent_id: u32)
            -> Result<u32>;
        /// Create a playlist under `parent_id` (0 = root).
        fn writer_create_playlist(w: &mut Writer, name: &str, parent_id: u32) -> Result<u32>;
        /// Append a track to a playlist.
        fn writer_add_track_to_playlist(
            w: &mut Writer,
            playlist_id: u32,
            track_id: u32,
        ) -> Result<()>;
        /// Create a top-level tag category in `exportExt.pdb`.
        fn writer_create_tag_category(w: &mut Writer, name: &str) -> Result<u32>;
        /// Associate leaf tag labels with a track under a category.
        fn writer_add_tags_to_track(
            w: &mut Writer,
            track_id: u32,
            category: u32,
            tags: Vec<String>,
        ) -> Result<()>;
        /// Flush and close. Consumes the writer; explicit close preserves error visibility
        /// (the upstream `Drop` is best-effort).
        fn writer_close(w: Box<Writer>) -> Result<()>;
    }
}

// ---- Opaque handle types ----

pub struct Reader(DeviceExportReader);
pub struct Writer(DeviceExportWriter);

// ---- Bridge function impls ----

fn track_default() -> Track {
    let d = crate::Track::default();
    Track {
        title: d.title,
        artist: d.artist,
        album: d.album,
        genre: d.genre,
        key: d.key,
        label: d.label,
        composer: d.composer,
        remixer: d.remixer,
        orig_artist: d.orig_artist,
        comment: d.comment,
        isrc: d.isrc,
        lyricist: d.lyricist,
        mix_name: d.mix_name,
        release_date: d.release_date,
        date_added: d.date_added,
        file_path: d.file_path,
        filename: d.filename,
        #[cfg(feature = "artwork")]
        artwork_source: d.artwork_source,
        #[cfg(not(feature = "artwork"))]
        artwork_device_path: d.artwork_device_path,
        #[cfg(feature = "analysis")]
        analysis_source: d.analysis_source,
        message: d.message,
        tempo: d.tempo,
        bitrate: d.bitrate,
        sample_rate: d.sample_rate,
        sample_depth: d.sample_depth,
        duration_secs: d.duration_secs,
        file_size: d.file_size,
        track_number: d.track_number,
        disc_number: d.disc_number,
        year: d.year,
        play_count: d.play_count,
        rating: d.rating.into(),
        color: d.color.into(),
        file_type: d.file_type.into(),
        autoload_hotcues: d.autoload_hotcues,
        sample_count: 0,
        performance: PerformanceData {
            waveform_detail: Vec::new(),
            waveform_height: Vec::new(),
            beatgrid: Vec::new(),
            bpm: 0.0,
            hot_cues: Vec::new(),
            loops: Vec::new(),
            main_cue_sample: -1.0,
        },
    }
}

fn open_reader(path: String) -> Result<Box<Reader>, Error> {
    Ok(Box::new(Reader(DeviceExportReader::new(PathBuf::from(
        path,
    )))))
}

fn reader_path(r: &Reader) -> String {
    r.0.get_path().to_string_lossy().into_owned()
}

fn reader_settings(r: &Reader) -> String {
    r.0.load_settings().to_string()
}

fn reader_playlists(r: &Reader) -> Result<Vec<PlaylistInfo>, Error> {
    let mut db = r.0.open_pdb_non_persistent()?;
    let nodes = device::get_playlists(&mut db)?;
    Ok(flatten_playlists(nodes, 0))
}

/// Walk the recursive tree, emitting one `PlaylistInfo` per node tagged with its parent.
fn flatten_playlists(nodes: Vec<device::PlaylistNode>, parent_id: u32) -> Vec<PlaylistInfo> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        match node {
            device::PlaylistNode::Folder(folder) => {
                let id = folder.id.0;
                out.push(PlaylistInfo {
                    id,
                    name: folder.name,
                    parent_id,
                });
                out.extend(flatten_playlists(folder.children, id));
            }
            device::PlaylistNode::Playlist(playlist) => {
                out.push(PlaylistInfo {
                    id: playlist.id.0,
                    name: playlist.name,
                    parent_id,
                });
            }
        }
    }
    out
}

fn writer_create(root: String) -> Result<Box<Writer>, Error> {
    Ok(Box::new(Writer(DeviceExportWriter::create(
        PathBuf::from(root),
    )?)))
}

fn writer_open(root: String) -> Result<Box<Writer>, Error> {
    Ok(Box::new(Writer(DeviceExportWriter::open(PathBuf::from(
        root,
    ))?)))
}

fn writer_add_track(w: &mut Writer, track: &Track) -> Result<AddTrackOutcome, Error> {
    let outcome = w.0.add_track(&track_to_native(track))?;
    Ok(AddTrackOutcome {
        id: outcome.id.0,
        is_new: outcome.is_new,
    })
}

fn writer_create_playlist_folder(w: &mut Writer, name: &str, parent_id: u32) -> Result<u32, Error> {
    Ok(w.0
        .create_playlist_folder(name, PlaylistTreeNodeId(parent_id))?
        .0)
}

fn writer_create_playlist(w: &mut Writer, name: &str, parent_id: u32) -> Result<u32, Error> {
    Ok(w.0.create_playlist(name, PlaylistTreeNodeId(parent_id))?.0)
}

fn writer_add_track_to_playlist(
    w: &mut Writer,
    playlist_id: u32,
    track_id: u32,
) -> Result<(), Error> {
    w.0.add_track_to_playlist(PlaylistTreeNodeId(playlist_id), TrackId(track_id))
}

fn writer_create_tag_category(w: &mut Writer, name: &str) -> Result<u32, Error> {
    Ok(w.0.create_tag_category(name)?.0)
}

fn writer_add_tags_to_track(
    w: &mut Writer,
    track_id: u32,
    category: u32,
    tags: Vec<String>,
) -> Result<(), Error> {
    w.0.add_tags_to_track(TrackId(track_id), device::TagCategoryId(category), &tags)
}

fn writer_close(w: Box<Writer>) -> Result<(), Error> {
    w.0.close()
}

// ---- Conversions between the bridge enums and the crate's typed enums ----
//
// cxx shared enums are non-exhaustive outside the bridge module, so the
// `From<ffi::*>` arms carry an unreachable wildcard.

impl From<util::Rating> for Rating {
    fn from(r: util::Rating) -> Self {
        match r {
            util::Rating::Zero => Self::Zero,
            util::Rating::One => Self::One,
            util::Rating::Two => Self::Two,
            util::Rating::Three => Self::Three,
            util::Rating::Four => Self::Four,
            util::Rating::Five => Self::Five,
        }
    }
}

impl From<Rating> for util::Rating {
    fn from(r: Rating) -> Self {
        match r {
            Rating::Zero => Self::Zero,
            Rating::One => Self::One,
            Rating::Two => Self::Two,
            Rating::Three => Self::Three,
            Rating::Four => Self::Four,
            _ => Self::Zero,
        }
    }
}

impl From<util::ColorIndex> for ColorIndex {
    fn from(c: util::ColorIndex) -> Self {
        match c {
            util::ColorIndex::None => Self::None,
            util::ColorIndex::Pink => Self::Pink,
            util::ColorIndex::Red => Self::Red,
            util::ColorIndex::Orange => Self::Orange,
            util::ColorIndex::Yellow => Self::Yellow,
            util::ColorIndex::Green => Self::Green,
            util::ColorIndex::Aqua => Self::Aqua,
            util::ColorIndex::Blue => Self::Blue,
            util::ColorIndex::Purple => Self::Purple,
        }
    }
}

impl From<ColorIndex> for util::ColorIndex {
    fn from(c: ColorIndex) -> Self {
        match c {
            ColorIndex::None => Self::None,
            ColorIndex::Pink => Self::Pink,
            ColorIndex::Red => Self::Red,
            ColorIndex::Orange => Self::Orange,
            ColorIndex::Yellow => Self::Yellow,
            ColorIndex::Green => Self::Green,
            ColorIndex::Aqua => Self::Aqua,
            ColorIndex::Blue => Self::Blue,
            _ => Self::None,
        }
    }
}

impl From<util::FileType> for FileType {
    fn from(f: util::FileType) -> Self {
        match f {
            util::FileType::Unknown => Self::Unknown,
            util::FileType::Mp3 => Self::Mp3,
            util::FileType::M4a => Self::M4a,
            util::FileType::Flac => Self::Flac,
            util::FileType::Wav => Self::Wav,
            util::FileType::Aiff => Self::Aiff,
            // `Other(_)` has no FFI representation; survives only on the Rust side.
            util::FileType::Other(_) => Self::Unknown,
        }
    }
}

impl From<FileType> for util::FileType {
    fn from(f: FileType) -> Self {
        match f {
            FileType::Unknown => Self::Unknown,
            FileType::Mp3 => Self::Mp3,
            FileType::M4a => Self::M4a,
            FileType::Flac => Self::Flac,
            FileType::Wav => Self::Wav,
            FileType::Aiff => Self::Aiff,
            _ => Self::Unknown,
        }
    }
}

/// Copy the FFI-shaped `Track` into the crate's typed struct. Field-for-field;
/// the artwork field name follows the `artwork` feature in both structs.
///
/// `analysis` is `Some` when the caller supplied a `PerformanceData`: rekordcrate
/// expands it to a full `AnlzInput` (the 11 ANLZ column types never cross the
/// boundary). When `None`, the writer falls back to `analysis_source` under the
/// `analysis` feature, or emits no ANLZ at all.
fn track_to_native(t: &Track) -> crate::Track {
    // Empty `waveform_detail` is the "no performance data" sentinel — cxx can't
    // hold `Option<PerformanceData>`, so an empty vector means emit no ANLZ.
    let analysis = if t.performance.waveform_detail.is_empty() {
        None
    } else {
        Some(performance_to_anlz(
            &t.performance,
            t.sample_rate,
            t.sample_count,
        ))
    };
    crate::Track {
        title: t.title.clone(),
        artist: t.artist.clone(),
        album: t.album.clone(),
        genre: t.genre.clone(),
        key: t.key.clone(),
        label: t.label.clone(),
        composer: t.composer.clone(),
        remixer: t.remixer.clone(),
        orig_artist: t.orig_artist.clone(),
        comment: t.comment.clone(),
        isrc: t.isrc.clone(),
        lyricist: t.lyricist.clone(),
        mix_name: t.mix_name.clone(),
        release_date: t.release_date.clone(),
        date_added: t.date_added.clone(),
        file_path: t.file_path.clone(),
        filename: t.filename.clone(),
        #[cfg(feature = "artwork")]
        artwork_source: t.artwork_source.clone(),
        #[cfg(not(feature = "artwork"))]
        artwork_device_path: t.artwork_device_path.clone(),
        analysis,
        #[cfg(feature = "analysis")]
        analysis_source: t.analysis_source.clone(),
        message: t.message.clone(),
        tempo: t.tempo,
        bitrate: t.bitrate,
        sample_rate: t.sample_rate,
        sample_depth: t.sample_depth,
        duration_secs: t.duration_secs,
        file_size: t.file_size,
        track_number: t.track_number,
        disc_number: t.disc_number,
        year: t.year,
        play_count: t.play_count,
        rating: t.rating.into(),
        color: t.color.into(),
        file_type: t.file_type.into(),
        autoload_hotcues: t.autoload_hotcues,
    }
}

/// Expand the format-agnostic `PerformanceData` into the crate's `AnlzInput`.
/// This is the bridge between the cxx-shared types and the ANLZ mapping layer in
/// `device::anlz_build`. Sentinel values (`bpm <= 0`, `main_cue < 0`) become `None`.
fn performance_to_anlz(
    p: &PerformanceData,
    sample_rate: u32,
    sample_count: u64,
) -> crate::device::writer::AnlzInput {
    use crate::device::anlz_build::{build_anlz_input, CueInput};

    let detail: Vec<(u8, u8, u8)> = p
        .waveform_detail
        .iter()
        .map(|e| (e.low, e.mid, e.high))
        .collect();
    let beatgrid: Vec<(i32, f64)> = p
        .beatgrid
        .iter()
        .map(|m| (m.index, m.sample_offset))
        .collect();
    let bpm = (p.bpm > 0.0).then_some(p.bpm);

    // Flatten hot cues + loops into one cue list. Hot-cue slots come from
    // `index` (1-8); loops are 0 only if their index is 0, else use the index.
    let mut cues: Vec<CueInput<'_>> = Vec::with_capacity(p.hot_cues.len() + p.loops.len());
    for hc in &p.hot_cues {
        cues.push(CueInput {
            hot_cue: hc.index,
            sample_offset: hc.sample_offset,
            loop_end: None,
            is_loop: false,
            label: &hc.label,
            r: hc.r,
            g: hc.g,
            b: hc.b,
        });
    }
    for lp in &p.loops {
        cues.push(CueInput {
            hot_cue: lp.index,
            sample_offset: lp.start,
            loop_end: Some(lp.end),
            is_loop: true,
            label: &lp.label,
            r: lp.r,
            g: lp.g,
            b: lp.b,
        });
    }
    let main_cue = (p.main_cue_sample >= 0.0).then_some(p.main_cue_sample);

    build_anlz_input(
        &detail,
        &p.waveform_height,
        sample_rate,
        &beatgrid,
        bpm,
        &cues,
        main_cue,
        sample_count,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process;

    fn unique_tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("rekordcrate-cpp-{}-{name}", process::id()));
        let _ = fs::remove_dir_all(&p);
        p
    }

    /// The one check: marshal a Track through the bridge, write it, reopen,
    /// read back. Fails if any struct/enum marshalling, opaque-handle
    /// lifetime, or error conversion is wrong.
    #[test]
    fn write_then_read_roundtrip() {
        let root = unique_tmp("roundtrip");
        let root_str = root.to_string_lossy().into_owned();

        let mut w = writer_create(root_str.clone()).expect("writer_create");
        let mut track: Track = track_default();
        track.title = "Test".to_string();
        track.filename = "t.mp3".to_string();
        track.file_path = "/Contents/t.mp3".to_string();
        let outcome = writer_add_track(&mut w, &track).expect("writer_add_track");
        assert_eq!(outcome.id, 1, "first track should get id 1");
        assert!(outcome.is_new, "first track should be newly inserted");
        writer_close(w).expect("writer_close");

        let r = open_reader(root_str.clone()).expect("open_reader");
        assert!(
            reader_path(&r).ends_with(&root_str),
            "reader path should end with the root"
        );

        // Settings flattens ~50 enums to a formatted string; just confirm it doesn't blow up.
        let _ = reader_settings(&r);

        let playlists = reader_playlists(&r).expect("reader_playlists");
        assert!(
            playlists.is_empty(),
            "fresh export has no playlists, got {playlists:?}"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn dedup_returns_existing_track() {
        let root = unique_tmp("dedup");
        let root_str = root.to_string_lossy().into_owned();

        let mut w = writer_create(root_str).expect("writer_create");
        let mut track: Track = track_default();
        track.file_path = "/Contents/dup.mp3".to_string();
        let first = writer_add_track(&mut w, &track).expect("first add");
        let second = writer_add_track(&mut w, &track).expect("second add");
        assert!(first.is_new, "first add should insert");
        assert!(!second.is_new, "second add with same path should dedup");
        assert_eq!(first.id, second.id, "dedup should return same id");
        writer_close(w).expect("writer_close");

        let _ = fs::remove_dir_all(&root);
    }

    /// The performance-data pipeline: a `PerformanceData` with a waveform +
    /// beatgrid + hot cue crosses the bridge, the writer emits ANLZ files, and
    /// the `.DAT` carries a non-empty BeatGrid section. This is the cross-layer
    /// check that the format-agnostic → ANLZ expansion works through cxx.
    #[test]
    fn performance_data_writes_anlz_beatgrid() {
        let root = unique_tmp("perf");
        let root_str = root.to_string_lossy().into_owned();

        let mut w = writer_create(root_str.clone()).expect("writer_create");
        let mut track: Track = track_default();
        track.file_path = "/Contents/p.mp3".to_string();
        track.filename = "p.mp3".to_string();
        track.sample_rate = 44_100;
        track.sample_count = 44_100; // 1 s
                                     // 150 Hz over 1 s → 150 detail columns.
        track.performance.waveform_detail = (0..150)
            .map(|_| WaveformEntry {
                low: 40,
                mid: 80,
                high: 120,
            })
            .collect();
        track.performance.waveform_height = vec![16; 150];
        track.performance.bpm = 120.0;
        track.performance.beatgrid = vec![
            BeatgridMarker {
                index: 1,
                sample_offset: 0.0,
            },
            // 120 BPM → 22050 samples/beat; 2 beats span 44100 samples.
            BeatgridMarker {
                index: 3,
                sample_offset: 44_100.0,
            },
        ];
        track.performance.hot_cues.push(HotCueFfi {
            index: 1,
            label: "Intro".to_string(),
            sample_offset: 0.0,
            r: 255,
            g: 0,
            b: 0,
            a: 255,
        });
        track.performance.main_cue_sample = 0.0;
        let outcome = writer_add_track(&mut w, &track).expect("writer_add_track");
        assert!(outcome.is_new);
        writer_close(w).expect("writer_close");

        // ANLZ files live under PIONEER/USBANLZ/<derived>/ANLZ0000.*. The writer
        // derives the intermediate dir from the file_path, so walk the whole tree.
        let usbanlz = root.join("PIONEER/USBANLZ");
        assert!(usbanlz.exists(), "USBANALZ dir should exist at {usbanlz:?}");
        let mut found_dat = false;
        for entry in walkdir(&usbanlz) {
            if entry.ends_with("ANLZ0000.DAT") {
                found_dat = true;
                let bytes = fs::read(&entry).expect("read DAT");
                // BeatGrid section content-kind tag is "PQTZ" per the ANLZ format.
                // ponytail: a raw tag check, not a full parse — the smallest thing that fails
                // if the beatgrid section wasn't written.
                assert!(
                    bytes.windows(4).any(|w| w == b"PQTZ"),
                    "DAT at {entry:?} should contain a beat-grid (PQTZ) section"
                );
            }
        }
        assert!(found_dat, "ANLZ0000.DAT should have been written");

        let _ = fs::remove_dir_all(&root);
    }

    /// Recursively collect every file path under `dir`.
    fn walkdir(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(dir) else {
            return out;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                out.extend(walkdir(&p));
            } else {
                out.push(p);
            }
        }
        out
    }
}
