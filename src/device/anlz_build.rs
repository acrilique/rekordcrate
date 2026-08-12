// Copyright (c) 2026 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

//! ANLZ column construction shared by the `analysis` (FFT from audio) and `cpp` (caller-supplied
//! 3-band) paths. This module owns the *format knowledge*: how 150 Hz detail bands expand to the
//! ANLZ column vectors, how a sparse beatgrid densifies, and how cues map. It depends only on
//! `crate::anlz`, never on audio decoding, so it compiles under either feature.

use crate::anlz::{
    Beat, Cue, CueListType, CueType, ExtendedCue, Waveform3BandDetailColumn,
    Waveform3BandPreviewColumn, WaveformColorDetailColumn, WaveformColorPreviewColumn,
};
use crate::device::writer::AnlzInput;

/// Detail section columns per second (PWV3/PWV5/PWV7). Pinned by the format: 75 frames/sec × 2.
pub const DETAIL_HZ: f64 = 150.0;

/// Preview section columns per second (PWAV/PWV2/PWV4/PWV6). Not pinned by the format; Rekordbox
/// uses ~1 column per 150 ms. Heuristic, matches observed fixtures within a few percent.
pub const PREVIEW_HZ: f64 = 6.667;

/// Integer detail columns aggregated into one preview column.
pub const DETAIL_PER_PREVIEW: usize = (DETAIL_HZ / PREVIEW_HZ).round() as usize;

/// Build the four band-derived column groups (PWV4 color preview, PWV5 color detail, PWV6/PWV7
/// 3-band) from a single 150 Hz 3-band detail vector.
///
/// `heights` is a per-column peak height 0-31 used only by PWV5 (the detail color column carries an
/// independent height). When `heights` is shorter than `bands`, missing entries default to the
/// column's `max(low, mid, high)` scaled to 0-31 — an honest fallback so a caller who didn't sample
/// peak amplitude still gets a plausible height.
///
/// Whiteness and the PWV4 bottom-half band stay 0; Rekordbox's exact derivation is proprietary and
/// undocumented. ponytail: ceiling — the bottom-half frequency band and whiteness bytes are guesses.
#[must_use]
pub fn build_band_columns(
    bands: &[(u8, u8, u8)],
    heights: &[u8],
) -> (
    Vec<WaveformColorPreviewColumn>,
    Vec<WaveformColorDetailColumn>,
    Vec<Waveform3BandPreviewColumn>,
    Vec<Waveform3BandDetailColumn>,
) {
    let detail_per_preview = DETAIL_PER_PREVIEW.max(1);

    // PWV4 color preview: integer-mean of detail bands over each preview window.
    let color_preview: Vec<WaveformColorPreviewColumn> = bands
        .chunks(detail_per_preview)
        .map(|chunk| {
            let n = chunk.len().max(1) as u32;
            let (sl, sm, sh) = chunk
                .iter()
                .fold((0u32, 0u32, 0u32), |(l, m, h), &(bl, bm, bh)| {
                    (l + u32::from(bl), m + u32::from(bm), h + u32::from(bh))
                });
            let (l, m, h) = ((sl / n) as u8, (sm / n) as u8, (sh / n) as u8);
            // `new(bottom_half, bottom_third, mid_third, top_third)`: bottom_half is an unknown
            // Rekordbox-specific value; mirror bottom_third as a harmless guess (analysis.rs did the same).
            WaveformColorPreviewColumn::new(l, l, m, h)
        })
        .collect();

    // PWV5 color detail: one column per detail slot. RGB mapping mirrors analysis.rs's guess.
    let color_detail: Vec<WaveformColorDetailColumn> = bands
        .iter()
        .enumerate()
        .map(|(i, &(l, m, h))| color_detail_column(l, m, h, heights.get(i).copied()))
        .collect();

    // PWV6 / PWV7: reuse the band energies directly (field order mid, top, bottom).
    let band3_preview: Vec<Waveform3BandPreviewColumn> = color_preview
        .iter()
        .map(|c| Waveform3BandPreviewColumn {
            energy_mid_third_freq: c.energy_mid_third_freq,
            energy_top_third_freq: c.energy_top_third_freq,
            energy_bottom_third_freq: c.energy_bottom_third_freq,
        })
        .collect();
    let band3_detail: Vec<Waveform3BandDetailColumn> = bands
        .iter()
        .map(|&(l, m, h)| Waveform3BandDetailColumn {
            energy_mid_third_freq: m,
            energy_top_third_freq: h,
            energy_bottom_third_freq: l,
        })
        .collect();

    (color_preview, color_detail, band3_preview, band3_detail)
}

/// Quantize one color-detail column. RGB picks the dominant band (high→blue, mid→green, low→red);
/// `height` falls back to the scaled band max when the caller supplied none.
fn color_detail_column(
    low: u8,
    mid: u8,
    high: u8,
    height: Option<u8>,
) -> WaveformColorDetailColumn {
    let height = height.unwrap_or_else(|| {
        ((u32::from(low).max(u32::from(mid)).max(u32::from(high)) as f32 / 255.0) * 31.0).round()
            as u8
    });
    let (r, g, b) = if high >= mid && high >= low {
        (0u8, 0, 7) // high → blue
    } else if mid >= low {
        (0, 7, 0) // mid → green
    } else {
        (7, 0, 0) // low → red
    };
    WaveformColorDetailColumn::new()
        .with_red(r)
        .with_green(g)
        .with_blue(b)
        .with_height(height)
}

/// Build the preview + tiny mono columns (PWAV / PWV2) at `PREVIEW_HZ` from a per-column peak
/// height vector. Each preview entry downsamples `detail_per_preview` detail columns by taking the
/// max height, matching how a coarser view of the same peaks looks.
#[must_use]
pub fn build_preview_columns(
    heights_0_31: &[u8],
) -> (
    Vec<crate::anlz::WaveformPreviewColumn>,
    Vec<crate::anlz::TinyWaveformPreviewColumn>,
) {
    use crate::anlz::{TinyWaveformPreviewColumn, WaveformPreviewColumn};

    let detail_per_preview = DETAIL_PER_PREVIEW.max(1);
    let mut preview = Vec::with_capacity(heights_0_31.len() / detail_per_preview + 1);
    let mut tiny = Vec::with_capacity(preview.capacity());
    for chunk in heights_0_31.chunks(detail_per_preview) {
        let peak = chunk.iter().copied().max().unwrap_or(0).min(31);
        // PWV2 carries 4-bit height (0-15); PWAV carries 5-bit (0-31).
        preview.push(
            WaveformPreviewColumn::new()
                .with_height(peak)
                .with_whiteness(0),
        );
        tiny.push(TinyWaveformPreviewColumn::new().with_height((peak / 2).min(15)));
    }
    (preview, tiny)
}

/// Expand a sparse 2-marker beatgrid into one `Beat` per beat. Each marker carries an `index`
/// (beat number, may be negative — Rekordbox grids start at -4) and a `sample_offset`.
///
/// ponytail: ceiling — assumes **constant tempo between markers** (linear interpolation of
/// sample offsets). No rubato, no tempo curves, no time-signature changes. Upgrade path: emit
/// markers directly as a dense grid when a caller can supply per-beat offsets.
///
/// `bpm` seeds the `tempo` field (centi-BPM); if `None`, it's derived from the first segment's
/// samples-per-beat. `beat_number` cycles 1-4 anchored to `marker.index` so a grid starting at
/// beat -4 still aligns bar downbeats.
///
/// `sample_count` clips the tail: no beats are emitted past the track end.
#[must_use]
pub fn expand_beatgrid(
    markers: &[(i32, f64)],
    sample_rate: u32,
    bpm: Option<f64>,
    sample_count: u64,
) -> Vec<Beat> {
    if markers.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    // Sort by sample offset so callers need not pre-sort; the format requires ascending offsets.
    let mut sorted: Vec<(i32, f64)> = markers.to_vec();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut beats = Vec::new();
    for window in sorted.windows(2) {
        let (idx_a, off_a) = (i64::from(window[0].0), window[0].1);
        let (idx_b, off_b) = (i64::from(window[1].0), window[1].1);
        let beat_span = (idx_b - idx_a).max(1) as f64;
        let samples_per_beat = (off_b - off_a) / beat_span;
        if samples_per_beat <= 0.0 {
            continue;
        }
        let local_bpm = 60.0 * f64::from(sample_rate) / samples_per_beat;
        let tempo = centi_bpm(bpm.unwrap_or(local_bpm));
        let ms_per_beat = samples_per_beat / f64::from(sample_rate) * 1000.0;
        let start_ms = off_a / f64::from(sample_rate) * 1000.0;
        for k in 0..beat_span as i64 {
            let global_beat = idx_a + k;
            let off = off_a + k as f64 * samples_per_beat;
            if off < 0.0 || off > sample_count as f64 {
                continue;
            }
            beats.push(Beat {
                beat_number: bar_position(global_beat),
                tempo,
                time: (start_ms + k as f64 * ms_per_beat).round() as u32,
            });
        }
    }
    beats
}

/// Map a (possibly negative) global beat index to its 1-4 position in the bar. Anchored so a grid
/// starting at beat index -4 still places the first emitted beat at bar position 1.
fn bar_position(global_beat: i64) -> u16 {
    let r = ((global_beat - 1).rem_euclid(4) + 1) as u16;
    r.clamp(1, 4)
}

/// Round bpm to centi-BPM (u16), saturating the format's 655.35 BPM ceiling.
fn centi_bpm(bpm: f64) -> u16 {
    if bpm.is_finite() && bpm > 0.0 {
        ((bpm * 100.0).round() as u64).min(u16::MAX as u64) as u16
    } else {
        0
    }
}

/// Convert a sample offset to milliseconds at `sample_rate`. `f64` math, rounded to u32 for ANLZ.
#[must_use]
pub fn samples_to_ms(samples: f64, sample_rate: u32) -> u32 {
    if sample_rate == 0 {
        return 0;
    }
    (samples / f64::from(sample_rate) * 1000.0).round() as u32
}

/// A cue (point or loop) in sample units, before ANLZ encoding.
#[derive(Debug, Clone, Copy)]
pub struct CueInput<'a> {
    /// Hot-cue slot, 1-based (1=A … 8=H). 0 = memory cue.
    pub hot_cue: u32,
    /// Point position in samples.
    pub sample_offset: f64,
    /// Loop end in samples; ignored when `is_loop` is false.
    pub loop_end: Option<f64>,
    /// Whether this cue is a loop (uses `loop_end`) or a point cue.
    pub is_loop: bool,
    /// Label (only stored on the extended cue).
    pub label: &'a str,
    /// RGBA color. Alpha is dropped (ANLZ stores RGB + a palette index).
    pub r: u8,
    /// Green component of the cue color.
    pub g: u8,
    /// Blue component of the cue color.
    pub b: u8,
}

/// Build the `.DAT` plain-cue list and the `.EXT` extended-cue list from one set of cues. Both
/// lists share `cue_list_type`. Memory cues (`hot_cue == 0`) are emitted as `MemoryCues`; hot cues
/// as `HotCues` — the writer applies the single shared type to both lists, so we pick `HotCues`
/// when any hot cue is present, else `MemoryCues`.
///
/// ponytail: ceiling — `hot_cue_color_index` (palette slot) defaults to 0; RGBA→palette mapping is
/// a gap. Alpha is dropped (documented). Memory-cue `color` defaults to `ColorIndex::None`.
#[must_use]
pub fn build_cues(
    cues: &[CueInput<'_>],
    sample_rate: u32,
) -> (Vec<Cue>, Vec<ExtendedCue>, CueListType) {
    let cue_list_type = if cues.iter().any(|c| c.hot_cue != 0) {
        CueListType::HotCues
    } else {
        CueListType::MemoryCues
    };

    let plain: Vec<Cue> = cues
        .iter()
        .map(|c| {
            let mut cue = Cue::default();
            cue.hot_cue = c.hot_cue;
            cue.cue_type = if c.is_loop {
                CueType::Loop
            } else {
                CueType::Point
            };
            cue.time = samples_to_ms(c.sample_offset, sample_rate);
            cue.loop_time = c
                .loop_end
                .filter(|_| c.is_loop)
                .map(|e| samples_to_ms(e, sample_rate))
                .unwrap_or(u32::MAX);
            cue
        })
        .collect();

    let extended: Vec<ExtendedCue> = cues
        .iter()
        .map(|c| {
            let mut e = ExtendedCue::default();
            e.hot_cue = c.hot_cue;
            e.cue_type = if c.is_loop {
                CueType::Loop
            } else {
                CueType::Point
            };
            e.time = samples_to_ms(c.sample_offset, sample_rate);
            e.loop_time = c
                .loop_end
                .filter(|_| c.is_loop)
                .map(|end| samples_to_ms(end, sample_rate))
                .unwrap_or(u32::MAX);
            e.comment = c.label.into();
            e.hot_cue_color_rgb = (c.r, c.g, c.b);
            e
        })
        .collect();

    (plain, extended, cue_list_type)
}

/// Convert a caller-supplied 3-band detail vector (150 Hz) plus per-column heights (0-31) into the
/// mono-detail columns (PWV3) used by `.EXT`. Each entry maps a single detail slot.
#[must_use]
pub fn build_detail_mono(heights_0_31: &[u8]) -> Vec<crate::anlz::WaveformPreviewColumn> {
    heights_0_31
        .iter()
        .map(|&h| {
            crate::anlz::WaveformPreviewColumn::new()
                .with_height(h.min(31))
                .with_whiteness(0)
        })
        .collect()
}

/// The single entry point for the `cpp` bridge: assemble a complete `AnlzInput` from
/// format-agnostic performance data.
///
/// `detail` is the 150 Hz 3-band vector; `detail_height` is the parallel per-column peak height
/// (0-31) driving PWAV/PWV2/PWV3 and PWV5 height. Both must be the same length; if `detail_height`
/// is shorter, missing heights fall back to the band max (see [`build_band_columns`]).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_anlz_input(
    detail: &[(u8, u8, u8)],
    detail_height: &[u8],
    sample_rate: u32,
    beatgrid: &[(i32, f64)],
    bpm: Option<f64>,
    cues: &[CueInput<'_>],
    main_cue_sample: Option<f64>,
    sample_count: u64,
) -> AnlzInput {
    let (color_preview, color_detail, band3_preview, band3_detail) =
        build_band_columns(detail, detail_height);
    let (preview_mono, tiny_preview) = build_preview_columns(detail_height);
    let detail_mono = build_detail_mono(detail_height);
    let beats = expand_beatgrid(beatgrid, sample_rate, bpm, sample_count);

    // A main cue (if supplied) is prepended as a memory cue; `build_cues` then sees one list and
    // picks HotCues only when a real hot cue follows.
    let main_cue = main_cue_sample.map(|mc| CueInput {
        hot_cue: 0,
        sample_offset: mc,
        loop_end: None,
        is_loop: false,
        label: "",
        r: 0,
        g: 0,
        b: 0,
    });
    let combined: Vec<CueInput<'_>> = main_cue.into_iter().chain(cues.iter().copied()).collect();
    let (cues_plain, cues_extended, cue_list_type) = build_cues(&combined, sample_rate);

    AnlzInput {
        beats,
        cues: cues_plain,
        cues_extended,
        cue_list_type,
        preview_mono,
        tiny_preview,
        detail_mono: Some(detail_mono),
        color_preview: Some(color_preview),
        color_detail: Some(color_detail),
        band3_preview: Some(band3_preview),
        band3_detail: Some(band3_detail),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Self-check: a 2-marker beatgrid at 120 BPM over 1 s densifies to ~2 beats with cycling bar
    /// positions and ms offsets within ~1 sample. Fails if the constant-tempo interpolation breaks.
    #[test]
    fn beatgrid_expansion_is_constant_tempo() {
        let sr = 44_100u32;
        // 120 BPM → 22050 samples/beat. Two beats span 44100 samples (1 s).
        let samples_per_beat = 60.0 / 120.0 * sr as f64; // 22050
        let m0 = (1, 0.0);
        let m1 = (3, 2.0 * samples_per_beat); // beat 1 → beat 3
        let beats = expand_beatgrid(&[m0, m1], sr, Some(120.0), u64::MAX);
        assert_eq!(beats.len(), 2, "two beats between markers 1 and 3");
        assert_eq!(
            beats.iter().map(|b| b.beat_number).collect::<Vec<_>>(),
            vec![1, 2],
            "bar positions cycle from the marker index"
        );
        assert_eq!(beats[0].tempo, 12_000, "120.00 BPM → 12000 centi-BPM");
        assert_eq!(beats[0].time, 0);
        assert!((beats[1].time as i64 - 500).abs() <= 1, "beat 2 at ~500 ms");

        // Total count from duration: at 120 BPM, 1 s holds 2 beats.
        let from_dur = expand_beatgrid(&[m0, m1], sr, Some(120.0), sr as u64);
        assert_eq!(from_dur.len(), 2);
    }

    /// Bar positions cycle 1-4 with downbeats at global index ≡ 1 (mod 4). So a grid spanning
    /// beats -4..-1 yields [4,1,2,3] — the downbeat at -3 lands on position 1, and -4 is the
    /// beat just before it (position 4). This is the correct rekordbox bar alignment, not [1,2,3,4].
    #[test]
    fn beatgrid_negative_index_aligns_bar() {
        let sr = 44_100u32;
        let spb = 60.0 / 120.0 * sr as f64;
        let beats = expand_beatgrid(&[(-4, 0.0), (0, 4.0 * spb)], sr, Some(120.0), u64::MAX);
        let positions: Vec<u16> = beats.iter().map(|b| b.beat_number).collect::<Vec<_>>();
        assert_eq!(positions, vec![4, 1, 2, 3]);
    }

    /// 150-entry detail → ~7 preview columns at 6.667 Hz (DETAIL_PER_PREVIEW = 22).
    #[test]
    fn preview_downsample_count() {
        let detail: Vec<(u8, u8, u8)> = (0..150).map(|_| (10, 20, 30)).collect();
        let heights: Vec<u8> = vec![16; 150];
        let (cp, cd, bp, bd) = build_band_columns(&detail, &heights);
        // 150 / 22 = 6.8 → 7 windows.
        assert_eq!(cp.len(), 7, "color preview windows");
        assert_eq!(bp.len(), 7, "band3 preview windows");
        assert_eq!(cd.len(), 150, "color detail is 1:1 with detail");
        assert_eq!(bd.len(), 150, "band3 detail is 1:1 with detail");
        let (preview, tiny) = build_preview_columns(&heights);
        assert_eq!(preview.len(), 7);
        assert_eq!(tiny.len(), 7);
        assert!(preview.iter().all(|c| c.height() <= 31));
        assert!(tiny.iter().all(|c| c.height() <= 15));
    }

    /// Cues encode point vs loop and pick HotCues type when a hot cue is present.
    #[test]
    fn cue_building_point_and_loop() {
        let sr = 44_100u32;
        let cues = [
            CueInput {
                hot_cue: 1,
                sample_offset: 44_100.0, // 1 s
                loop_end: None,
                is_loop: false,
                label: "Intro",
                r: 255,
                g: 0,
                b: 0,
            },
            CueInput {
                hot_cue: 2,
                sample_offset: 88_200.0,
                loop_end: Some(132_300.0),
                is_loop: true,
                label: "Loop",
                r: 0,
                g: 255,
                b: 0,
            },
        ];
        let (plain, ext, t) = build_cues(&cues, sr);
        assert_eq!(t, CueListType::HotCues, "hot cue present → HotCues");
        assert_eq!(plain.len(), 2);
        assert_eq!(plain[0].time, 1000, "44100 samples @ 44100 Hz = 1000 ms");
        assert_eq!(plain[0].cue_type, CueType::Point);
        assert_eq!(plain[1].cue_type, CueType::Loop);
        assert_eq!(plain[1].loop_time, 3000, "loop end at 3 s");
        assert_eq!(ext.len(), 2);
        assert_eq!(ext[0].hot_cue_color_rgb, (255, 0, 0), "RGBA drops alpha");
    }

    /// End-to-end assembly: a populated `build_anlz_input` yields a non-empty `AnlzInput` whose
    /// waveform sections are internally consistent (detail == height count, preview is ~1/22).
    #[test]
    fn build_anlz_input_assembles_consistently() {
        let sr = 44_100u32;
        let detail: Vec<(u8, u8, u8)> = (0..150).map(|_| (50, 100, 150)).collect();
        let heights: Vec<u8> = vec![20; 150];
        let input = build_anlz_input(
            &detail,
            &heights,
            sr,
            &[(1, 0.0), (5, 60.0 / 120.0 * sr as f64 * 4.0)],
            Some(120.0),
            &[CueInput {
                hot_cue: 1,
                sample_offset: sr as f64,
                loop_end: None,
                is_loop: false,
                label: "x",
                r: 0,
                g: 0,
                b: 0,
            }],
            Some(0.0),
            sr as u64,
        );
        assert!(!input.beats.is_empty());
        assert!(!input.cues.is_empty());
        assert_eq!(input.cues.len(), 2, "main cue + hot cue");
        assert_eq!(input.cue_list_type, CueListType::HotCues);
        assert_eq!(input.detail_mono.as_ref().unwrap().len(), 150);
        assert_eq!(input.color_detail.as_ref().unwrap().len(), 150);
        assert_eq!(input.band3_detail.as_ref().unwrap().len(), 150);
        assert_eq!(input.preview_mono.len(), 7);
    }
}
