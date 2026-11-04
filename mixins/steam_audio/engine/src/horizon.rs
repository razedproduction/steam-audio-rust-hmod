//! The renderer behind the mod's sound output (`crate::audio::steam_audio` in `horizon-client-render`).
//!
//! Horizon mixes every playing source into the output callback's buffer with OpenAL Soft's stereo panning. With the
//! mod, its output callback hands the sources to [`Renderer`] instead: each positioned source is filtered by Steam Audio's air
//! absorption model ([`EqEffect`]) and rendered binaurally with the built-in CIPIC HRTF ([`BinauralEffect`]); sources
//! without a direction (music, the UI, sounds at the listener) and those beyond the voice budget keep Horizon's own
//! stereo gains. Distance attenuation stays Horizon's (vanilla's linear model), so what can be heard and how loud
//! does not change, only how it sounds.
//!
//! Steam Audio's effects work on fixed frames, the output callback asks for any number of frames: the renderer mixes
//! whole frames of [`FRAME_SIZE`] into a buffer and serves the callback from it, at most one frame late.
//!
//! The HRTF takes a moment to load; it loads on a thread of its own the first time the mixer asks, and until it is
//! ready (or when it can't be: a sampling rate the HRTF has no measurements for) the output leaves the mixing to
//! Horizon's own `mix`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::air_absorption::AirAbsorptionModel;
use crate::audio_buffer::{AudioBuffer, AudioSettings};
use crate::bands::NUM_BANDS;
use crate::effects::{BinauralEffect, BinauralParams, EqEffect};
use crate::hrtf::{
    HrtfDatabase, HrtfInterpolation, HrtfNormalization, HrtfPhaseType, HrtfSettings, HrtfSource,
};
use crate::math::Vector3;

/// Samples per Steam Audio frame (about 10.7 ms at 48 kHz).
pub const FRAME_SIZE: usize = 512;

/// How the mod renders, from the environment (read once).
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// `HORIZON_STEAM_AUDIO=0` turns the mod off: Horizon mixes as without it.
    pub enabled: bool,
    /// `HORIZON_STEAM_AUDIO_AIR=0` turns air absorption off.
    pub air_absorption: bool,
    /// `HORIZON_STEAM_AUDIO_INTERPOLATION=bilinear` blends the four nearest HRTF measurements instead of taking the
    /// nearest one (smoother movement, more CPU).
    pub bilinear: bool,
    /// `HORIZON_STEAM_AUDIO_VOICES=N`: how many of the loudest sources are rendered binaurally (default 32); the
    /// rest are panned as Horizon pans them.
    pub max_binaural_voices: usize,
    /// `HORIZON_STEAM_AUDIO_VOLUME_DB=x`: HRTF volume adjustment in dB (default 0).
    pub volume_db: f32,
    /// `HORIZON_STEAM_AUDIO_HRTF=path`: an HRTF in Steam Audio's `.hrtf` format instead of the built-in one.
    pub hrtf_path: Option<std::path::PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: true,
            air_absorption: true,
            bilinear: false,
            max_binaural_voices: 32,
            volume_db: 0.0,
            hrtf_path: None,
        }
    }
}

impl Config {
    /// The configuration `var` describes (`std::env::var` in the game).
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Self {
        let off = |name: &str| {
            var(name).is_some_and(|v| {
                matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "0" | "off" | "false" | "no"
                )
            })
        };
        let default = Self::default();
        Self {
            enabled: !off("HORIZON_STEAM_AUDIO"),
            air_absorption: !off("HORIZON_STEAM_AUDIO_AIR"),
            bilinear: var("HORIZON_STEAM_AUDIO_INTERPOLATION")
                .is_some_and(|v| v.trim().eq_ignore_ascii_case("bilinear")),
            max_binaural_voices: var("HORIZON_STEAM_AUDIO_VOICES")
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(default.max_binaural_voices),
            volume_db: var("HORIZON_STEAM_AUDIO_VOLUME_DB")
                .and_then(|v| v.trim().parse::<f32>().ok())
                .filter(|v| v.is_finite())
                .map_or(default.volume_db, |v| v.clamp(-24.0, 24.0)),
            hrtf_path: var("HORIZON_STEAM_AUDIO_HRTF")
                .filter(|v| !v.trim().is_empty())
                .map(Into::into),
        }
    }

    /// The game's configuration, from its environment.
    pub fn get() -> &'static Config {
        static CONFIG: OnceLock<Config> = OnceLock::new();
        CONFIG.get_or_init(|| Self::from_vars(|name| std::env::var(name).ok()))
    }
}

/// One playing source for one output callback.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Voice {
    /// The mixer's channel id.
    pub id: u64,
    /// The source's gain with distance attenuation (0 when no camera hears it: it still advances).
    pub gain: f32,
    /// Horizon's stereo gains for the source's direction (without `gain`), used when it is not rendered binaurally.
    pub pan: [f32; 2],
    /// Direction to the source in listener space (+x right, +y up, -z ahead), or `None` to pan it as Horizon does.
    pub direction: Option<[f32; 3]>,
    /// Distance to the source in blocks (metres), for air absorption.
    pub distance: f32,
    /// Source samples per output sample (sampling rate ratio times pitch).
    pub step: f64,
}

/// What a source's samples are, for [`Renderer::render`]: fills `out` with the source's mono samples resampled at
/// `step` source samples per output sample, advancing its position, and returns whether it still plays (a stopped
/// source fills the rest with silence).
pub type ReadSource<'a> = dyn FnMut(u64, f64, &mut [f32]) -> bool + 'a;

/// A source rendered binaurally: its effects' state between frames.
#[derive(Debug)]
struct VoiceState {
    binaural: BinauralEffect,
    eq: EqEffect,
    /// The gain the last frame ended at, ramped from to avoid zipper noise.
    gain: f32,
    /// The frame it was last rendered in ([`Renderer::frame`]).
    frame: u64,
}

/// Binaural mixing of Horizon's sources at one sampling rate.
#[derive(Debug)]
pub struct Renderer {
    rate: u32,
    config: Config,
    settings: AudioSettings,
    hrtf: HrtfDatabase,
    air: AirAbsorptionModel,
    states: HashMap<u64, VoiceState>,
    /// The sources of the current callback, whether each is rendered binaurally, and whether it still plays.
    voices: Vec<Voice>,
    binaural: Vec<bool>,
    playing: Vec<bool>,
    order: Vec<usize>,
    mono: AudioBuffer,
    stereo: AudioBuffer,
    /// The current frame, interleaved stereo, and how many of its frames the callbacks took.
    block: Vec<f32>,
    taken: usize,
    /// Frames rendered so far.
    frame: u64,
    /// The mixer being served (its address): another one starts over.
    owner: usize,
}

impl Renderer {
    /// Loads the HRTF for `rate`. Fails for a sampling rate the HRTF has no measurements for (the built-in one has
    /// 44100, 48000 and 24000 Hz) or an HRTF file that can't be read.
    pub fn new(rate: u32, config: Config) -> Result<Self, String> {
        let custom = match &config.hrtf_path {
            Some(path) => Some(
                std::fs::read(path)
                    .map_err(|e| format!("cannot read the HRTF {}: {e}", path.display()))?,
            ),
            None => None,
        };
        let settings = HrtfSettings {
            source: custom
                .as_deref()
                .map_or(HrtfSource::BuiltIn, HrtfSource::Data),
            volume: config.volume_db,
            normalization: HrtfNormalization::None,
        };
        let hrtf = HrtfDatabase::new(&settings, rate, FRAME_SIZE)
            .map_err(|e| format!("cannot load the HRTF at {rate} Hz: {e}"))?;
        Ok(Self {
            rate,
            settings: AudioSettings::new(rate, FRAME_SIZE),
            hrtf,
            air: AirAbsorptionModel::default(),
            states: HashMap::new(),
            voices: Vec::new(),
            binaural: Vec::new(),
            playing: Vec::new(),
            order: Vec::new(),
            mono: AudioBuffer::new(1, FRAME_SIZE),
            stereo: AudioBuffer::new(2, FRAME_SIZE),
            block: vec![0.0; 2 * FRAME_SIZE],
            taken: FRAME_SIZE,
            frame: 0,
            owner: 0,
            config,
        })
    }

    /// The sampling rate it renders at.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Starts a callback for the mixer at `owner` (any value identifying it): forgets the last callback's sources.
    pub fn begin(&mut self, owner: usize) {
        if owner != self.owner {
            self.owner = owner;
            self.states.clear();
            self.taken = FRAME_SIZE;
        }
        self.voices.clear();
    }

    /// Adds a playing source to this callback.
    pub fn push(&mut self, voice: Voice) {
        self.voices.push(voice);
    }

    /// Mixes this callback's sources into `out` (interleaved, `channels` of at least 2; channels after the second
    /// stay silent), reading their samples through `read`.
    pub fn render(&mut self, out: &mut [f32], channels: usize, read: &mut ReadSource<'_>) {
        assert!(channels >= 2, "binaural output needs two channels");
        out.fill(0.0);
        self.choose_binaural();
        self.playing.clear();
        self.playing.resize(self.voices.len(), true);
        for frame in out.chunks_exact_mut(channels) {
            if self.taken == FRAME_SIZE {
                self.render_frame(read);
                self.taken = 0;
            }
            frame[0] = self.block[2 * self.taken];
            frame[1] = self.block[2 * self.taken + 1];
            self.taken += 1;
        }
    }

    /// The loudest sources with a direction, up to the budget, are rendered binaurally.
    fn choose_binaural(&mut self) {
        let voices = &self.voices;
        self.order.clear();
        self.order.extend(
            (0..voices.len()).filter(|&i| voices[i].direction.is_some() && voices[i].gain > 0.0),
        );
        if self.order.len() > self.config.max_binaural_voices {
            self.order
                .sort_by(|&a, &b| voices[b].gain.total_cmp(&voices[a].gain));
            self.order.truncate(self.config.max_binaural_voices);
        }
        self.binaural.clear();
        self.binaural.resize(voices.len(), false);
        for &i in &self.order {
            self.binaural[i] = true;
        }
    }

    fn render_frame(&mut self, read: &mut ReadSource<'_>) {
        self.frame += 1;
        let frame = self.frame;
        self.block.fill(0.0);
        let Self {
            config,
            settings,
            hrtf,
            air,
            states,
            voices,
            binaural,
            playing,
            mono,
            stereo,
            block,
            ..
        } = self;
        for (i, voice) in voices.iter().enumerate() {
            if !playing[i] {
                continue;
            }
            playing[i] = read(voice.id, voice.step, mono.channel_mut(0));
            match (binaural[i], voice.direction) {
                (true, Some(direction)) => {
                    let state = states.entry(voice.id).or_insert_with(|| VoiceState {
                        binaural: BinauralEffect::new(settings, hrtf),
                        eq: EqEffect::new(settings),
                        gain: voice.gain,
                        frame,
                    });
                    state.frame = frame;
                    if config.air_absorption {
                        let distance = voice.distance.max(0.0);
                        let gains: [f32; NUM_BANDS] =
                            std::array::from_fn(|band| air.evaluate(distance, band));
                        state.eq.apply_in_place(&gains, mono);
                    }
                    ramp(mono.channel_mut(0), state.gain, voice.gain);
                    state.gain = voice.gain;
                    let mut params = BinauralParams::new(Vector3::from_array(direction), hrtf);
                    if config.bilinear {
                        params.interpolation = HrtfInterpolation::Bilinear;
                    }
                    params.phase_type = HrtfPhaseType::None;
                    state.binaural.apply(params, mono, stereo);
                    for (j, out) in block.chunks_exact_mut(2).enumerate() {
                        out[0] += stereo.channel(0)[j];
                        out[1] += stereo.channel(1)[j];
                    }
                }
                _ => {
                    let [left, right] = voice.pan.map(|pan| pan * voice.gain);
                    if left == 0.0 && right == 0.0 {
                        continue;
                    }
                    for (out, &s) in block.chunks_exact_mut(2).zip(mono.channel(0)) {
                        out[0] += s * left;
                        out[1] += s * right;
                    }
                }
            }
        }
        // Sources that stopped, or left the binaural set, drop their state.
        states.retain(|_, state| state.frame == frame);
    }
}

/// Scales `samples` by a gain ramping linearly from `from` to `to`.
fn ramp(samples: &mut [f32], from: f32, to: f32) {
    if from == to {
        if to != 1.0 {
            for s in samples {
                *s *= to;
            }
        }
        return;
    }
    let n = samples.len() as f32;
    for (i, s) in samples.iter_mut().enumerate() {
        *s *= from + (to - from) * (i as f32 / n);
    }
}

/// Reads a mono source at `step` samples per output sample from `position`, as Horizon's mixer does (linear
/// interpolation; a looping source wraps, another stops at its end). Returns the new position and whether it still
/// plays; after it stops, `out` is silent.
pub fn read_linear(
    samples: &[f32],
    mut position: f64,
    step: f64,
    looping: bool,
    out: &mut [f32],
) -> (f64, bool) {
    for (k, o) in out.iter_mut().enumerate() {
        let mut i = position as usize;
        if i + 1 >= samples.len() {
            if looping && samples.len() > 1 {
                position = 0.0;
                i = 0;
            } else {
                out[k..].fill(0.0);
                return (position, false);
            }
        }
        let t = (position - i as f64) as f32;
        *o = samples[i] * (1.0 - t) + samples[i + 1] * t;
        position += step;
    }
    (position, true)
}

enum Slot {
    Empty,
    Loading(u32),
    Ready(Box<Renderer>),
    Failed(u32),
}

static SLOT: Mutex<Slot> = Mutex::new(Slot::Empty);

/// Runs `f` with the renderer for `rate`, when the mod is on and the HRTF for that rate is loaded. The first call
/// for a rate starts loading it on a thread of its own and returns `None`, as every call does until it is ready.
pub fn with_renderer<R>(rate: u32, f: impl FnOnce(&mut Renderer) -> R) -> Option<R> {
    let config = Config::get();
    if !config.enabled {
        return None;
    }
    let mut slot = SLOT.lock().ok()?;
    match &mut *slot {
        Slot::Ready(renderer) if renderer.rate == rate => return Some(f(renderer)),
        Slot::Loading(r) | Slot::Failed(r) if *r == rate => return None,
        _ => {}
    }
    *slot = Slot::Loading(rate);
    drop(slot);
    let config = config.clone();
    let spawned = std::thread::Builder::new()
        .name("steam-audio-hrtf".into())
        .spawn(move || {
            let started = std::time::Instant::now();
            let result = Renderer::new(rate, config);
            let Ok(mut slot) = SLOT.lock() else {
                return;
            };
            if !matches!(*slot, Slot::Loading(r) if r == rate) {
                return;
            }
            *slot = match result {
                Ok(renderer) => {
                    log::info!(
                        "steam_audio: HRTF loaded at {rate} Hz in {} ms; binaural rendering on",
                        started.elapsed().as_millis()
                    );
                    Slot::Ready(Box::new(renderer))
                }
                Err(e) => {
                    log::warn!("steam_audio: {e}; Horizon's own stereo mixing stays on");
                    Slot::Failed(rate)
                }
            };
        });
    if spawned.is_err()
        && let Ok(mut slot) = SLOT.lock()
    {
        *slot = Slot::Failed(rate);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_reads_the_environment() {
        let config = Config::from_vars(|name| {
            match name {
                "HORIZON_STEAM_AUDIO_AIR" => Some("off"),
                "HORIZON_STEAM_AUDIO_VOICES" => Some("8"),
                "HORIZON_STEAM_AUDIO_INTERPOLATION" => Some("Bilinear"),
                "HORIZON_STEAM_AUDIO_VOLUME_DB" => Some("-90"),
                _ => None,
            }
            .map(Into::into)
        });
        assert!(config.enabled);
        assert!(!config.air_absorption);
        assert!(config.bilinear);
        assert_eq!(config.max_binaural_voices, 8);
        assert_eq!(config.volume_db, -24.0);
        assert!(!Config::from_vars(|n| (n == "HORIZON_STEAM_AUDIO").then(|| "0".into())).enabled);
        assert_eq!(Config::from_vars(|_| None), Config::default());
    }

    #[test]
    fn reading_interpolates_loops_and_stops() {
        let samples = [0.0, 1.0, 2.0, 3.0];
        let mut out = [9.0; 4];
        let (position, playing) = read_linear(&samples, 0.0, 0.5, false, &mut out);
        assert_eq!(out, [0.0, 0.5, 1.0, 1.5]);
        assert_eq!((position, playing), (2.0, true));
        let (_, playing) = read_linear(&samples, 2.0, 1.0, false, &mut out);
        assert!(!playing);
        assert_eq!(out, [2.0, 0.0, 0.0, 0.0]);
        let (_, playing) = read_linear(&samples, 2.0, 1.0, true, &mut out);
        assert!(playing);
        assert_eq!(out, [2.0, 0.0, 1.0, 2.0]);
    }

    /// Renders broadband noise from `direction` and returns the energy of each ear.
    fn ear_energy(renderer: &mut Renderer, direction: Option<[f32; 3]>) -> [f32; 2] {
        let mut seed = 0x2545_f491_u32;
        let tone: Vec<f32> = (0..48_000)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as f32 / u32::MAX as f32 - 0.5
            })
            .collect();
        let mut position = 0.0;
        let mut out = vec![0.0; 2 * 4800];
        renderer.begin(1);
        renderer.push(Voice {
            id: 7,
            gain: 0.5,
            pan: [0.596, 0.596],
            direction,
            distance: 4.0,
            step: 1.0,
        });
        renderer.render(&mut out, 2, &mut |id, step, buf| {
            assert_eq!(id, 7);
            let (p, playing) = read_linear(&tone, position, step, true, buf);
            position = p;
            playing
        });
        let mut energy = [0.0; 2];
        for frame in out.chunks_exact(2).skip(FRAME_SIZE * 2) {
            energy[0] += frame[0] * frame[0];
            energy[1] += frame[1] * frame[1];
        }
        energy
    }

    #[test]
    fn a_source_on_the_right_is_louder_in_the_right_ear() {
        let mut renderer =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        let [left, right] = ear_energy(&mut renderer, Some([1.0, 0.0, 0.0]));
        assert!(right > 2.0 * left, "left {left}, right {right}");
        let [left, right] = ear_energy(&mut renderer, Some([-1.0, 0.0, 0.0]));
        assert!(left > 2.0 * right, "left {left}, right {right}");
        let [left, right] = ear_energy(&mut renderer, Some([0.0, 0.0, -1.0]));
        assert!(
            (left / right - 1.0).abs() < 0.5,
            "left {left}, right {right}"
        );
        assert!(left > 0.0);
        assert!(renderer.states.contains_key(&7));
    }

    #[test]
    fn sources_without_a_direction_keep_horizons_panning() {
        let mut renderer =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        let [left, right] = ear_energy(&mut renderer, None);
        assert!(
            (left - right).abs() <= 1e-3 * left,
            "left {left}, right {right}"
        );
        assert!(renderer.states.is_empty());
    }

    #[test]
    fn unsupported_rates_fail_to_load() {
        assert!(Renderer::new(96_000, Config::default()).is_err());
    }
}
