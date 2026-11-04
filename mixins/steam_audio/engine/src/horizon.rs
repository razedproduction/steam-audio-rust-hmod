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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use crate::acoustics::{AcousticsSettings, Snapshot};
use crate::air_absorption::AirAbsorptionModel;
use crate::audio_buffer::{AudioBuffer, AudioEffectState, AudioSettings};
use crate::bands::NUM_BANDS;
use crate::effects::{
    AmbisonicsDecodeEffect, AmbisonicsDecodeParams, BinauralEffect, BinauralParams, EqEffect,
    HybridReverbEffect, HybridReverbIr, HybridReverbParams, OverlapSaveConvolutionSettings,
};
use crate::hrtf::{
    HrtfDatabase, HrtfInterpolation, HrtfNormalization, HrtfPhaseType, HrtfSettings, HrtfSource,
};
use crate::math::{CoordinateSpace3, Vector3};
use crate::random::CRand;
use crate::reflections::{self, Reflections};
use crate::simulation::Reverb;
use crate::speaker_layout::{SpeakerLayout, SpeakerLayoutType};

/// Samples per Steam Audio frame (about 10.7 ms at 48 kHz).
pub const FRAME_SIZE: usize = 512;

/// The simulated reflections are physical: energy coming back to the listener from a source at the listener, relative
/// to the direct sound 1 m from it. Horizon's sources are quieter with distance than that (linear, not 1/r), so the
/// reverb is brought up by this gain: at a reverb level of 1, a 9 Г— 5 Г— 9 stone cave's reverb is about as loud as the
/// sound sent to it (measured by `measure_reverb_level`).
const REFLECTIONS_GAIN: f32 = 1.4;

/// How the mod renders: the defaults, then what the player chose on the mod's settings page (Options > Mods,
/// [`crate::settings`]), then the environment variables below, which win. It can change while the game runs
/// ([`Config::set`]); the renderer picks a change up at its next output callback.
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
    /// `HORIZON_STEAM_AUDIO_OCCLUSION=0` turns occlusion and transmission through blocks off.
    pub occlusion: bool,
    /// `HORIZON_STEAM_AUDIO_WALLS=x` (0 to 1, default 0.35): how much walls muffle; 1 is Steam Audio's physical
    /// transmission (a stone wall lets through about в€’36 dB), 0 lets everything through.
    pub walls: f32,
    /// `HORIZON_STEAM_AUDIO_REVERB=0` turns the reverb of the surroundings off.
    pub reverb: bool,
    /// `HORIZON_STEAM_AUDIO_REVERB_LEVEL=x` (0 to 2, default 0.5): reverb level in a fully enclosed place.
    pub reverb_level: f32,
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
            occlusion: true,
            walls: 0.35,
            reverb: true,
            reverb_level: 0.5,
        }
    }
}

/// A boolean setting's value: `0`, `off`, `false`, `no` are off, `1`, `on`, `true`, `yes` on.
pub(crate) fn parse_switch(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "false" | "no" => Some(false),
        "1" | "on" | "true" | "yes" => Some(true),
        _ => None,
    }
}

/// A number setting's value, finite and clamped into `range`.
pub(crate) fn parse_number(value: &str, range: std::ops::RangeInclusive<f32>) -> Option<f32> {
    value
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(*range.start(), *range.end()))
}

impl Config {
    /// The configuration `var` describes over the defaults (`std::env::var` in the game).
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Self {
        let mut config = Self::default();
        config.apply_vars(var);
        config
    }

    /// Applies the environment variables `var` has; the others leave their setting as it is.
    pub fn apply_vars(&mut self, var: impl Fn(&str) -> Option<String>) {
        let switch = |name: &str, value: &mut bool| {
            if let Some(v) = var(name).as_deref().and_then(parse_switch) {
                *value = v;
            }
        };
        switch("HORIZON_STEAM_AUDIO", &mut self.enabled);
        switch("HORIZON_STEAM_AUDIO_AIR", &mut self.air_absorption);
        switch("HORIZON_STEAM_AUDIO_OCCLUSION", &mut self.occlusion);
        switch("HORIZON_STEAM_AUDIO_REVERB", &mut self.reverb);
        let number = |name: &str, range: std::ops::RangeInclusive<f32>, value: &mut f32| {
            if let Some(v) = var(name).and_then(|v| parse_number(&v, range)) {
                *value = v;
            }
        };
        number("HORIZON_STEAM_AUDIO_WALLS", 0.0..=1.0, &mut self.walls);
        number(
            "HORIZON_STEAM_AUDIO_REVERB_LEVEL",
            0.0..=2.0,
            &mut self.reverb_level,
        );
        number(
            "HORIZON_STEAM_AUDIO_VOLUME_DB",
            -24.0..=24.0,
            &mut self.volume_db,
        );
        if let Some(v) = var("HORIZON_STEAM_AUDIO_INTERPOLATION") {
            self.bilinear = v.trim().eq_ignore_ascii_case("bilinear");
        }
        if let Some(v) = var("HORIZON_STEAM_AUDIO_VOICES").and_then(|v| v.trim().parse().ok()) {
            self.max_binaural_voices = v;
        }
        if let Some(v) = var("HORIZON_STEAM_AUDIO_HRTF").filter(|v| !v.trim().is_empty()) {
            self.hrtf_path = Some(v.into());
        }
    }

    /// The world acoustics' tuning for this configuration.
    pub fn acoustics(&self) -> AcousticsSettings {
        AcousticsSettings {
            transmission_exponent: self.walls,
            ..AcousticsSettings::default()
        }
    }

    /// The game's current configuration (until the settings are loaded, the defaults and the environment).
    pub fn get() -> Config {
        live()
            .read()
            .map_or_else(|e| e.into_inner().clone(), |c| c.clone())
    }

    /// Whether the mod is on, without copying the configuration (the audio thread asks every callback).
    pub fn enabled() -> bool {
        live();
        ENABLED.load(Ordering::Relaxed)
    }

    /// Makes `config` the game's configuration, from now on.
    pub fn set(config: Config) {
        let lock = live();
        let mut current = lock.write().unwrap_or_else(|e| e.into_inner());
        ENABLED.store(config.enabled, Ordering::Relaxed);
        *current = config;
        VERSION.fetch_add(1, Ordering::Release);
    }

    /// Bumped by each [`Config::set`].
    pub fn version() -> u64 {
        VERSION.load(Ordering::Acquire)
    }
}

static ENABLED: AtomicBool = AtomicBool::new(true);
static VERSION: AtomicU64 = AtomicU64::new(0);

fn live() -> &'static RwLock<Config> {
    static LIVE: OnceLock<RwLock<Config>> = OnceLock::new();
    LIVE.get_or_init(|| {
        let config = Config::from_vars(|name| std::env::var(name).ok());
        ENABLED.store(config.enabled, Ordering::Relaxed);
        RwLock::new(config)
    })
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
    /// The world acoustics published by the game thread.
    acoustics: Snapshot,
    /// The reverb bus: what the sources send it (faded to the reverb level the last frame ended at), Steam Audio's
    /// hybrid reverb of the simulated reflections into order-1 Ambisonics, and their binaural decoding in the
    /// listener's orientation.
    reverb_in: AudioBuffer,
    ambisonics: AudioBuffer,
    reverb_out: AudioBuffer,
    hybrid: HybridReverbEffect,
    decode: AmbisonicsDecodeEffect,
    reverb_state: AudioEffectState,
    wet: f32,
    /// The reflections being played (reverb times, parametric level per band, delay) and new ones not played yet.
    reflections: Option<(Reverb, [f32; NUM_BANDS], usize)>,
    pending: Option<Box<Reflections>>,
    /// The listener's orientation in the world.
    orientation: CoordinateSpace3,
    /// The current frame, interleaved stereo, and how many of its frames the callbacks took.
    block: Vec<f32>,
    taken: usize,
    /// Frames rendered so far.
    frame: u64,
    /// The mixer being served (its address): another one starts over.
    owner: usize,
    /// The [`Config::version`] its configuration is from.
    config_version: u64,
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
        let audio_settings = AudioSettings::new(rate, FRAME_SIZE);
        let hybrid = HybridReverbEffect::new(
            &audio_settings,
            &OverlapSaveConvolutionSettings {
                num_channels: reflections::NUM_CHANNELS,
                ir_size: reflections::ir_size(rate),
            },
            &mut CRand::default(),
        );
        let decode = AmbisonicsDecodeEffect::new(
            &audio_settings,
            &SpeakerLayout::new(SpeakerLayoutType::Stereo),
            reflections::ORDER,
            Some(&hrtf),
        );
        reflections::set_rate(rate, FRAME_SIZE);
        Ok(Self {
            rate,
            settings: audio_settings,
            hrtf,
            acoustics: Snapshot::default(),
            reverb_in: AudioBuffer::new(1, FRAME_SIZE),
            ambisonics: AudioBuffer::new(reflections::NUM_CHANNELS, FRAME_SIZE),
            reverb_out: AudioBuffer::new(2, FRAME_SIZE),
            hybrid,
            decode,
            reverb_state: AudioEffectState::TailComplete,
            wet: 0.0,
            reflections: None,
            pending: None,
            orientation: CoordinateSpace3::default(),
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
            config_version: Config::version(),
            config,
        })
    }

    /// The sampling rate it renders at.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Replaces the world acoustics it applies (normally what the game thread publishes).
    pub fn set_acoustics(&mut self, acoustics: Snapshot) {
        self.acoustics = acoustics;
    }

    /// Plays `reflections` from the next frame on (normally what the reflection simulation publishes).
    pub fn set_reflections(&mut self, reflections: Box<Reflections>) {
        self.pending = Some(reflections);
    }

    /// The listener's orientation: where it looks and its up, in the world (unit vectors, perpendicular). The
    /// reverb's reflections turn with it.
    pub fn set_orientation(&mut self, ahead: [f32; 3], up: [f32; 3]) {
        let (ahead, up) = (Vector3::from_array(ahead), Vector3::from_array(up));
        if ahead.length() > 0.5 && up.length() > 0.5 {
            self.orientation = CoordinateSpace3::from_ahead_up(ahead, up, Vector3::ZERO);
        }
    }

    /// Starts a callback for the mixer at `owner` (any value identifying it): forgets the last callback's sources.
    pub fn begin(&mut self, owner: usize) {
        if owner != self.owner {
            self.owner = owner;
            self.states.clear();
            self.taken = FRAME_SIZE;
        }
        self.voices.clear();
        // The settings page changed something: take it, but keep the HRTF it was loaded with (its file and volume
        // come from the environment only).
        let version = Config::version();
        if version != self.config_version {
            self.config_version = version;
            let mut config = Config::get();
            config.hrtf_path = self.config.hrtf_path.take();
            config.volume_db = self.config.volume_db;
            self.config = config;
        }
    }

    /// The configuration it renders with.
    pub fn config(&self) -> &Config {
        &self.config
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
        self.acoustics.update();
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
            acoustics,
            reverb_in,
            ambisonics,
            reverb_out,
            hybrid,
            decode,
            reverb_state,
            wet,
            reflections: playing_reflections,
            pending,
            orientation,
            rate,
            block,
            ..
        } = self;
        reverb_in.make_silent();
        for (i, voice) in voices.iter().enumerate() {
            if !playing[i] {
                continue;
            }
            playing[i] = read(voice.id, voice.step, mono.channel_mut(0));
            if voice.direction.is_none() || voice.gain == 0.0 {
                // Music, the UI, sounds at the listener: as Horizon pans them, dry.
                let [left, right] = voice.pan.map(|pan| pan * voice.gain);
                if left != 0.0 || right != 0.0 {
                    for (out, &s) in block.chunks_exact_mut(2).zip(mono.channel(0)) {
                        out[0] += s * left;
                        out[1] += s * right;
                    }
                }
                continue;
            }
            // Air absorption, then what the walls let through: band gains for the direct path.
            let distance = voice.distance.max(0.0);
            let air_gains: [f32; NUM_BANDS] = if config.air_absorption {
                std::array::from_fn(|band| air.evaluate(distance, band))
            } else {
                [1.0; NUM_BANDS]
            };
            let direct = if config.occlusion {
                acoustics.direct_gains(voice.id)
            } else {
                [1.0; NUM_BANDS]
            };
            // The reverb hears the source whether or not a wall hides it (sound goes round), through the air.
            if config.reverb {
                let send = voice.gain * air_gains[1];
                for (r, &s) in reverb_in.channel_mut(0).iter_mut().zip(mono.channel(0)) {
                    *r += s * send;
                }
            }
            let mut eq_gains: [f32; NUM_BANDS] = std::array::from_fn(|b| air_gains[b] * direct[b]);
            let mut overall = 1.0;
            EqEffect::normalize_gains(&mut eq_gains, &mut overall);
            match (binaural[i], voice.direction) {
                (true, Some(direction)) => {
                    let state = states.entry(voice.id).or_insert_with(|| VoiceState {
                        binaural: BinauralEffect::new(settings, hrtf),
                        eq: EqEffect::new(settings),
                        gain: voice.gain * overall,
                        frame,
                    });
                    state.frame = frame;
                    state.eq.apply_in_place(&eq_gains, mono);
                    let gain = voice.gain * overall;
                    ramp(mono.channel_mut(0), state.gain, gain);
                    state.gain = gain;
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
                    // Beyond the binaural budget: Horizon's panning, with the walls' and the air's mean loss.
                    let loss = eq_gains.iter().sum::<f32>() / NUM_BANDS as f32 * overall;
                    let [left, right] = voice.pan.map(|pan| pan * voice.gain * loss);
                    for (out, &s) in block.chunks_exact_mut(2).zip(mono.channel(0)) {
                        out[0] += s * left;
                        out[1] += s * right;
                    }
                }
            }
        }
        // Sources that stopped, or left the binaural set, drop their state.
        states.retain(|_, state| state.frame == frame);

        // The reverb bus: the simulated reflections of the surroundings (convolved, turning with the head) and their
        // parametric tail. The level fades what goes in, so a change leaves the tail already ringing as it is.
        let target = if config.reverb {
            config.reverb_level * REFLECTIONS_GAIN
        } else {
            0.0
        };
        if pending.is_none()
            && let Some(new) = reflections::take()
        {
            if new.rate == *rate {
                *pending = Some(new);
            } else {
                reflections::recycle(new.fir);
            }
        }
        if target > 0.0 || *wet > 0.0 || *reverb_state == AudioEffectState::TailRemaining {
            ramp(reverb_in.channel_mut(0), *wet, target);
            let mut update = pending.take();
            if let Some(new) = &update {
                *playing_reflections = Some((new.reverb, new.eq, new.delay));
            }
            if let Some((reverb, eq, delay)) = playing_reflections.as_ref() {
                let ir = match &mut update {
                    Some(new) => HybridReverbIr::Updated(&mut new.fir),
                    None => HybridReverbIr::Current,
                };
                *reverb_state = hybrid.apply(
                    HybridReverbParams {
                        ir,
                        num_channels: reflections::NUM_CHANNELS,
                        reverb,
                        eq_coeffs: eq,
                        delay: *delay,
                    },
                    reverb_in,
                    ambisonics,
                );
                decode.apply(
                    AmbisonicsDecodeParams {
                        order: reflections::ORDER,
                        orientation: *orientation,
                        binaural: true,
                        hrtf: Some(&*hrtf),
                    },
                    ambisonics,
                    reverb_out,
                );
                for (j, out) in block.chunks_exact_mut(2).enumerate() {
                    out[0] += reverb_out.channel(0)[j];
                    out[1] += reverb_out.channel(1)[j];
                }
            }
            // The effect handed back the impulse response it played until now.
            if let Some(old) = update {
                reflections::recycle(old.fir);
            }
        }
        *wet = target;
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
    if !Config::enabled() {
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
    let config = Config::get();
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
        let mut out = vec![0.0; 2 * 36_000];
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
    fn the_renderer_takes_settings_changes_at_its_next_callback() {
        let mut renderer =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        // A change other tests' renderers may also take: harmless to them.
        Config::set(Config {
            max_binaural_voices: 31,
            ..Config::default()
        });
        assert_eq!(renderer.config().max_binaural_voices, 32);
        renderer.begin(1);
        assert_eq!(renderer.config().max_binaural_voices, 31);
        assert!(Config::enabled());
    }

    #[test]
    fn unsupported_rates_fail_to_load() {
        assert!(Renderer::new(96_000, Config::default()).is_err());
    }

    use crate::acoustics::{SourceAcoustics, Surface};
    use crate::effects::{OverlapSaveFir, OverlapSavePartitioner};
    use crate::reflections::{GRID_SIZE, GridBuilder, NUM_CHANNELS, Simulation, Voxels};
    use crate::simulation::ImpulseResponse;
    use std::sync::Arc;

    fn renderer_with(sources: &[(u64, SourceAcoustics)]) -> Renderer {
        let mut renderer =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        renderer.set_acoustics(Snapshot::fixed(sources.iter().copied().collect()));
        renderer
    }

    /// The simulated reflections of a stone cave of 9 × 5 × 9 blocks around the listener.
    fn cave() -> Box<Reflections> {
        let listener = [0.5, 65.6, 0.5];
        let grid = Arc::new(Voxels::from_fn(
            GridBuilder::origin_for(listener),
            GRID_SIZE,
            |c| {
                let inside = c[0].abs() <= 4 && (64..69).contains(&c[1]) && c[2].abs() <= 4;
                (!inside).then_some(Surface::Rock)
            },
        ));
        let mut simulation = Simulation::new(48_000, FRAME_SIZE);
        let mut last = None;
        for _ in 0..4 {
            last = simulation.run(&grid, listener, None);
        }
        Box::new(last.expect("the listener is inside the grid"))
    }

    /// Reflections arriving from `direction` (world) only, for a quarter of a second, without a parametric tail.
    fn reflections_from(direction: [f32; 3]) -> Box<Reflections> {
        let size = reflections::ir_size(48_000);
        let c = reflections::coefficients(Vector3::from_array(direction));
        let mut ir = ImpulseResponse::with_size(NUM_CHANNELS, size);
        let mut seed = 0x1234_5678_u32;
        for k in 0..size / 4 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let n = (seed as f32 / u32::MAX as f32 - 0.5) * (-(k as f32) / 3000.0).exp();
            for (ch, &coefficient) in c.iter().enumerate() {
                ir.channel_mut(ch)[k] = coefficient * n;
            }
        }
        let mut fir = OverlapSaveFir::new(NUM_CHANNELS, size, FRAME_SIZE);
        OverlapSavePartitioner::new(FRAME_SIZE).partition(&ir, NUM_CHANNELS, size, &mut fir);
        Box::new(Reflections {
            rate: 48_000,
            fir,
            reverb: Reverb {
                reverb_times: [0.1; NUM_BANDS],
            },
            eq: [0.0; NUM_BANDS],
            delay: size,
        })
    }

    /// Renders 0.2 s with no source playing and returns the energy of each ear over the last `from` frames on.
    fn tail(renderer: &mut Renderer, from: usize) -> [f32; 2] {
        let mut out = vec![0.0; 2 * 9600];
        renderer.begin(1);
        renderer.render(&mut out, 2, &mut |_, _, _| false);
        let mut energy = [0.0; 2];
        for frame in out.chunks_exact(2).skip(from) {
            energy[0] += frame[0] * frame[0];
            energy[1] += frame[1] * frame[1];
        }
        energy
    }

    #[test]
    fn a_source_behind_a_wall_is_quieter() {
        let walled = SourceAcoustics {
            occlusion: 0.0,
            transmission: [0.23, 0.11, 0.09],
        };
        let mut open = renderer_with(&[]);
        let mut hidden = renderer_with(&[(7, walled)]);
        let [l0, r0] = ear_energy(&mut open, Some([1.0, 0.0, 0.0]));
        let [l1, r1] = ear_energy(&mut hidden, Some([1.0, 0.0, 0.0]));
        let ratio = (l1 + r1) / (l0 + r0);
        assert!(ratio < 0.1, "behind a wall: {ratio} of the energy");
        assert!(r1 > l1, "still on the right: left {l1}, right {r1}");
    }

    #[test]
    fn an_enclosed_place_leaves_a_reverb_tail() {
        let mut dry = renderer_with(&[]);
        let mut wet = renderer_with(&[]);
        wet.set_reflections(cave());
        let direct = ear_energy(&mut dry, Some([0.0, 0.0, -1.0]));
        let with_reverb = ear_energy(&mut wet, Some([0.0, 0.0, -1.0]));
        let added = (with_reverb[0] + with_reverb[1]) / (direct[0] + direct[1]);
        // The reverb adds to the direct sound without drowning it.
        assert!(
            added > 1.05 && added < 3.0,
            "with reverb: {added} times the energy"
        );
        // After the source stops, the tail goes on.
        let [dl, dr] = tail(&mut dry, 4800);
        let [wl, wr] = tail(&mut wet, 4800);
        assert!(wl + wr > 1e-3, "wet tail {}", wl + wr);
        assert!(dl + dr < 1e-9, "dry tail {}", dl + dr);
    }

    #[test]
    fn reflections_turn_with_the_head() {
        // A wall to the east (+x) reflects: on the right looking north (-z), on the left looking south.
        let mut north = renderer_with(&[]);
        north.set_reflections(reflections_from([1.0, 0.0, 0.0]));
        let mut south = renderer_with(&[]);
        south.set_reflections(reflections_from([1.0, 0.0, 0.0]));
        south.set_orientation([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]);
        ear_energy(&mut north, Some([0.0, 0.0, -1.0]));
        ear_energy(&mut south, Some([0.0, 0.0, -1.0]));
        let [l, r] = tail(&mut north, 0);
        assert!(r > 2.0 * l, "looking north: left {l}, right {r}");
        let [l, r] = tail(&mut south, 0);
        assert!(l > 2.0 * r, "looking south: left {l}, right {r}");
    }

    #[test]
    fn turning_the_reverb_off_lets_the_tail_ring_out() {
        let mut renderer = renderer_with(&[]);
        renderer.set_reflections(cave());
        ear_energy(&mut renderer, Some([0.0, 0.0, -1.0]));
        renderer.config.reverb = false;
        renderer.config_version = Config::version();
        let [l, r] = tail(&mut renderer, 0);
        assert!(l + r > 1e-3, "the tail rings out: {}", l + r);
        // New sounds are dry.
        let direct = ear_energy(&mut renderer_with(&[]), Some([0.0, 0.0, -1.0]));
        let later = ear_energy(&mut renderer, Some([0.0, 0.0, -1.0]));
        let ratio = (later[0] + later[1]) / (direct[0] + direct[1]);
        assert!((ratio - 1.0).abs() < 0.05, "{ratio}");
    }

    #[test]
    #[ignore = "measurement"]
    fn measure_reverb_level() {
        let mut dry = renderer_with(&[]);
        let mut wet = renderer_with(&[]);
        wet.config.reverb_level = 1.0;
        wet.set_reflections(cave());
        let direct = ear_energy(&mut dry, Some([0.0, 0.0, -1.0]));
        let with_reverb = ear_energy(&mut wet, Some([0.0, 0.0, -1.0]));
        let added = (with_reverb[0] + with_reverb[1]) / (direct[0] + direct[1]) - 1.0;
        eprintln!(
            "cave at level 1: reverb {added:.4} of the direct energy; gain for unity {:.3}",
            REFLECTIONS_GAIN / added.sqrt()
        );
    }
}
