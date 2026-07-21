// Copyright (c) 2026 Jan Holthuis <jan.holthuis@rub.de>
//
// This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy
// of the MPL was not distributed with this file, You can obtain one at
// http://mozilla.org/MPL/2.0/.
//
// SPDX-License-Identifier: MPL-2.0

//! Audio analysis: decodes an audio file and produces ANLZ waveform column vectors.
//!
//! Compiled only under the `analysis` feature. Rekordbox's exact analysis is proprietary and
//! not reproduced here — band boundaries, "whiteness" bytes, and RGB mapping are guesses from
//! the deepsymmetry docs. Output resembles but does not match a Rekordbox analysis and is
//! unvalidated on hardware.

use crate::anlz::{
    TinyWaveformPreviewColumn, Waveform3BandDetailColumn, Waveform3BandPreviewColumn,
    WaveformColorDetailColumn, WaveformColorPreviewColumn, WaveformPreviewColumn,
};
use crate::device::writer::{AnlzInput, Track};
use crate::{Error, Result};
use realfft::RealFftPlanner;
use std::path::Path;

/// Detail section columns per second (PWV3/PWV5/PWV7). Pinned by the format: 75 frames/sec × 2.
const DETAIL_HZ: f64 = 150.0;

/// Preview section columns per second (PWAV/PWV2/PWV4/PWV6). Not pinned by the format; Rekordbox
/// uses ~1 column per 150 ms. Heuristic, matches observed fixtures within a few percent.
const PREVIEW_HZ: f64 = 6.667;

/// FFT window size in samples. Must be a power of two for `realfft`.
const FFT_SIZE: usize = 1024;

/// Output of analyzing an audio file. Beats and cues are not computed here.
#[derive(Debug, Clone)]
pub struct ComputedAnlz {
    /// Fixed-width mono preview (PWAV).
    pub preview_mono: Vec<WaveformPreviewColumn>,
    /// Fixed-width tiny mono preview (PWV2).
    pub tiny_preview: Vec<TinyWaveformPreviewColumn>,
    /// Variable-width mono detail at 150 Hz (PWV3).
    pub detail_mono: Vec<WaveformPreviewColumn>,
    /// Fixed-width color preview (PWV4).
    pub color_preview: Vec<WaveformColorPreviewColumn>,
    /// Variable-width color detail at 150 Hz (PWV5).
    pub color_detail: Vec<WaveformColorDetailColumn>,
    /// Fixed-width 3-band preview (PWV6).
    pub band3_preview: Vec<Waveform3BandPreviewColumn>,
    /// Variable-width 3-band detail at 150 Hz (PWV7).
    pub band3_detail: Vec<Waveform3BandDetailColumn>,
}

impl ComputedAnlz {
    #[must_use]
    /// Wrap these columns in an [`AnlzInput`]. Beats and cues are left empty.
    pub fn into_input(self, _track: &Track) -> AnlzInput {
        AnlzInput {
            beats: Vec::new(),
            cues: Vec::new(),
            cues_extended: Vec::new(),
            cue_list_type: crate::anlz::CueListType::MemoryCues,
            preview_mono: self.preview_mono,
            tiny_preview: self.tiny_preview,
            detail_mono: Some(self.detail_mono),
            color_preview: Some(self.color_preview),
            color_detail: Some(self.color_detail),
            band3_preview: Some(self.band3_preview),
            band3_detail: Some(self.band3_detail),
        }
    }
}

/// Decode `path` to a mono f32 sample buffer and its sample rate.
fn decode_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::{probe::Hint, FormatOptions};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    let file = std::fs::File::open(path).map_err(|e| Error::AnlzError {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| Error::AnlzError {
            path: path.to_path_buf(),
            message: format!("probe failed: {e}"),
        })?;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.as_ref().is_some_and(|c| c.is_audio()))
        .cloned()
        .ok_or_else(|| Error::AnlzError {
            path: path.to_path_buf(),
            message: "no decodable audio track found".into(),
        })?;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|c| c.audio())
        .ok_or_else(|| Error::AnlzError {
            path: path.to_path_buf(),
            message: "track has no audio codec parameters".into(),
        })?;
    let sample_rate = audio_params.sample_rate.ok_or_else(|| Error::AnlzError {
        path: path.to_path_buf(),
        message: "unknown sample rate".into(),
    })?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|e| Error::AnlzError {
            path: path.to_path_buf(),
            message: format!("decoder init failed: {e}"),
        })?;
    let track_id = track.id;

    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                return Err(Error::AnlzError {
                    path: path.to_path_buf(),
                    message: format!("read failed: {e}"),
                })
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet).map_err(|e| Error::AnlzError {
            path: path.to_path_buf(),
            message: format!("decode failed: {e}"),
        })?;
        let channels = decoded.spec().channels().count();
        let frames = decoded.frames();
        if channels == 0 || frames == 0 {
            continue;
        }
        // Interleave to f32, then downmix by averaging channels per frame.
        let mut interleave = vec![0.0f32; frames * channels];
        decoded.copy_to_slice_interleaved(&mut interleave);
        for frame in 0..frames {
            let mut sum = 0.0f32;
            for ch in 0..channels {
                sum += interleave[frame * channels + ch];
            }
            mono.push(sum / channels as f32);
        }
    }
    Ok((mono, sample_rate))
}

/// Peak amplitude in `[0, 1]` over `samples`.
fn peak_amplitude(samples: &[f32]) -> f32 {
    samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, |a, b| a.max(b))
        .clamp(0.0, 1.0)
}

/// Mono preview + tiny preview columns at `PREVIEW_HZ`.
fn build_mono_previews(
    mono: &[f32],
    sample_rate: u32,
) -> (Vec<WaveformPreviewColumn>, Vec<TinyWaveformPreviewColumn>) {
    let samples_per_col = (f64::from(sample_rate) / PREVIEW_HZ).round() as usize;
    let mut preview = Vec::new();
    let mut tiny = Vec::new();
    let mut i = 0;
    while i < mono.len() {
        let chunk = &mono[i..mono.len().min(i + samples_per_col)];
        let peak = peak_amplitude(chunk);
        // 5-bit height (0-31) for PWAV, 4-bit (0-15) for PWV2.
        let h5 = (peak * 31.0).round() as u8;
        let h4 = (peak * 15.0).round() as u8;
        // Whiteness is unknown; leave at 0.
        preview.push(
            WaveformPreviewColumn::new()
                .with_height(h5)
                .with_whiteness(0),
        );
        tiny.push(TinyWaveformPreviewColumn::new().with_height(h4));
        i += samples_per_col;
    }
    (preview, tiny)
}

/// Mono detail columns at `DETAIL_HZ`.
fn build_mono_detail(mono: &[f32], sample_rate: u32) -> Vec<WaveformPreviewColumn> {
    let samples_per_col = (f64::from(sample_rate) / DETAIL_HZ).round() as usize;
    mono.chunks(samples_per_col)
        .map(|chunk| {
            let peak = peak_amplitude(chunk);
            let h = (peak * 31.0).round() as u8;
            WaveformPreviewColumn::new()
                .with_height(h)
                .with_whiteness(0)
        })
        .collect()
}

/// Per-column band energies (low, mid, high) via windowed FFT at `DETAIL_HZ`. Band boundaries
/// are naive Nyquist thirds, not Rekordbox's proprietary split.
fn build_band_energies(mono: &[f32], sample_rate: u32) -> Vec<(f32, f32, f32)> {
    let samples_per_col = (f64::from(sample_rate) / DETAIL_HZ).round() as usize;
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(FFT_SIZE);
    let mut spectrum = r2c.make_output_vec();
    let nyquist = f64::from(sample_rate) / 2.0;
    let low_cutoff = nyquist / 3.0;
    let mid_cutoff = 2.0 * nyquist / 3.0;
    let bin_hz = f64::from(sample_rate) / FFT_SIZE as f64;
    let low_bins = (low_cutoff / bin_hz).round() as usize;
    let mid_bins = (mid_cutoff / bin_hz).round() as usize;

    let mut out = Vec::new();
    let mut i = 0;
    while i < mono.len() {
        let end = mono.len().min(i + samples_per_col);
        let chunk = &mono[i..end];
        let mut frame = vec![0.0f32; FFT_SIZE];
        let copy = chunk.len().min(FFT_SIZE);
        frame[..copy].copy_from_slice(&chunk[..copy]);
        // Hann window to reduce spectral leakage.
        for (j, s) in frame.iter_mut().enumerate() {
            let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * j as f32 / FFT_SIZE as f32).cos());
            *s *= w;
        }
        let mut input = frame;
        r2c.process(&mut input, &mut spectrum)
            .map_err(|_| ())
            .expect("fft sizes match");
        let mut low = 0.0f64;
        let mut mid = 0.0f64;
        let mut high = 0.0f64;
        for (bin, c) in spectrum.iter().enumerate() {
            let mag = (c.re * c.re + c.im * c.im).sqrt() as f64;
            if bin < low_bins {
                low += mag;
            } else if bin < mid_bins {
                mid += mag;
            } else {
                high += mag;
            }
        }
        out.push((low as f32, mid as f32, high as f32));
        i += samples_per_col;
    }
    out
}

/// Quantize band energies to `u8` by normalizing against the global max.
fn quantize_bands(bands: &[(f32, f32, f32)]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let max = bands
        .iter()
        .flat_map(|(l, m, h)| [l, m, h])
        .copied()
        .fold(0.0f32, |a, b| a.max(b))
        .max(1e-9);
    let q = |v: f32| ((v / max) * 255.0).round().clamp(0.0, 255.0) as u8;
    let low: Vec<u8> = bands.iter().map(|(l, _, _)| q(*l)).collect();
    let mid: Vec<u8> = bands.iter().map(|(_, m, _)| q(*m)).collect();
    let high: Vec<u8> = bands.iter().map(|(_, _, h)| q(*h)).collect();
    (low, mid, high)
}

/// Map band energies to a packed color-detail column. RGB mapping is a guess.
fn color_detail_column(low: u8, mid: u8, high: u8, peak: f32) -> WaveformColorDetailColumn {
    let height = (peak * 31.0).round().clamp(0.0, 31.0) as u8;
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

/// Analyze `path` and produce all waveform column vectors.
///
/// # Errors
///
/// Returns [`Error::AnlzError`] if the file cannot be opened, probed, or decoded.
pub fn analyze(path: &Path) -> Result<ComputedAnlz> {
    let (mono, sample_rate) = decode_mono(path)?;

    let (preview_mono, tiny_preview) = build_mono_previews(&mono, sample_rate);
    let detail_mono = build_mono_detail(&mono, sample_rate);

    let bands = build_band_energies(&mono, sample_rate);
    let (low, mid, high) = quantize_bands(&bands);

    // Color preview (PWV4): average band energies across the detail columns in each slot.
    let samples_per_preview = (f64::from(sample_rate) / PREVIEW_HZ).round() as usize;
    let mut color_preview = Vec::with_capacity(preview_mono.len());
    let mut detail_idx = 0;
    let detail_per_preview = (DETAIL_HZ / PREVIEW_HZ).round() as usize;
    let mut preview_start = 0;
    while preview_start < mono.len() {
        let preview_end = mono.len().min(preview_start + samples_per_preview);
        let n = detail_per_preview.max(1);
        let band_range = detail_idx..(detail_idx + n).min(bands.len());
        let (l, m, h) = if band_range.is_empty() {
            (0u8, 0, 0)
        } else {
            let count = band_range.len() as u32;
            let bl = band_range.clone().map(|i| u32::from(low[i])).sum::<u32>() / count;
            let bm = band_range.clone().map(|i| u32::from(mid[i])).sum::<u32>() / count;
            let bh = band_range.clone().map(|i| u32::from(high[i])).sum::<u32>() / count;
            (bl as u8, bm as u8, bh as u8)
        };
        color_preview.push(WaveformColorPreviewColumn::new(l, l, m, h));
        detail_idx += n;
        preview_start = preview_end;
    }

    // Color detail (PWV5): one column per detail slot.
    let color_detail: Vec<WaveformColorDetailColumn> = low
        .iter()
        .zip(mid.iter().zip(high.iter()))
        .enumerate()
        .map(|(i, (&l, (&m, &h)))| {
            let col_start = i * (f64::from(sample_rate) / DETAIL_HZ).round() as usize;
            let col_end = mono
                .len()
                .min(col_start + (f64::from(sample_rate) / DETAIL_HZ).round() as usize);
            let peak = peak_amplitude(&mono[col_start..col_end]);
            color_detail_column(l, m, h, peak)
        })
        .collect();

    // 3-band preview/detail (PWV6/PWV7): reuse the band energies directly.
    let band3_preview: Vec<Waveform3BandPreviewColumn> = color_preview
        .iter()
        .map(|c| Waveform3BandPreviewColumn {
            energy_mid_third_freq: c.energy_mid_third_freq,
            energy_top_third_freq: c.energy_top_third_freq,
            energy_bottom_third_freq: c.energy_bottom_third_freq,
        })
        .collect();
    let band3_detail: Vec<Waveform3BandDetailColumn> = low
        .iter()
        .zip(mid.iter().zip(high.iter()))
        .map(|(&l, (&m, &h))| Waveform3BandDetailColumn {
            energy_mid_third_freq: m,
            energy_top_third_freq: h,
            energy_bottom_third_freq: l,
        })
        .collect();

    Ok(ComputedAnlz {
        preview_mono,
        tiny_preview,
        detail_mono,
        color_preview,
        color_detail,
        band3_preview,
        band3_detail,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Self-check: synthesized mono produces column vectors with expected counts and valid ranges.
    #[test]
    fn synthesized_signal_produces_valid_columns() {
        // 1 second of 440 Hz sine at 44100 Hz, amplitude 0.5.
        let sample_rate = 44_100u32;
        let n = sample_rate as usize;
        let mono: Vec<f32> = (0..n)
            .map(|i| {
                0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect();

        let (preview, tiny) = build_mono_previews(&mono, sample_rate);
        let detail = build_mono_detail(&mono, sample_rate);

        // 150 Hz over 1 second → ~150 columns.
        assert!(
            (140..=160).contains(&detail.len()),
            "detail column count {} should be ~150",
            detail.len()
        );
        // ~6.67 Hz → ~7 columns.
        assert!(
            (5..=9).contains(&preview.len()),
            "preview column count {} should be ~7",
            preview.len()
        );
        assert_eq!(preview.len(), tiny.len());

        assert!(preview.iter().all(|c| c.height() <= 31));
        assert!(tiny.iter().all(|c| c.height() <= 15));
        assert!(preview.iter().any(|c| c.height() > 0));

        let bands = build_band_energies(&mono, sample_rate);
        let (low, _mid, _high) = quantize_bands(&bands);
        assert_eq!(low.len(), detail.len());
    }

    /// End-to-end: decode a real MP3 fixture and verify the full `analyze` path returns non-empty
    /// vectors at the expected rates. Skipped if the fixture audio isn't present.
    #[test]
    fn analyze_real_mp3_fixture() {
        // TODO: export_with_anlz stuff isn't committed, add another file somewhere in data/ and amend this
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("export_with_anlz/Contents/Reboot/www.electronicfresh.com/01. Reboot - Bako (Original Mix).mp3");
        if !path.exists() {
            eprintln!("skipping: fixture audio not present at {path:?}");
            return;
        }
        let computed = analyze(&path).expect("analyze should succeed on the MP3 fixture");
        assert!(!computed.preview_mono.is_empty());
        assert!(!computed.tiny_preview.is_empty());
        assert!(!computed.detail_mono.is_empty());
        assert!(!computed.color_preview.is_empty());
        assert!(!computed.color_detail.is_empty());
        assert!(!computed.band3_preview.is_empty());
        assert!(!computed.band3_detail.is_empty());
        // Detail (150 Hz) should vastly outnumber preview (~6.67 Hz).
        assert!(
            computed.detail_mono.len() > computed.preview_mono.len() * 10,
            "detail {} should be >>10x preview {}",
            computed.detail_mono.len(),
            computed.preview_mono.len()
        );
    }
}
