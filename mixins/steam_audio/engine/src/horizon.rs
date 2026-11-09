//! The renderer behind the mod's sound output (`crate::audio::steam_audio` in `horizon-client-render`).
//!
//! Horizon mixes every playing source into the output callback's buffer with OpenAL Soft's stereo panning. With the
//! mod, its output callback hands the sources to [`Renderer`] instead: each positioned source is filtered by Steam Audio's air
//! absorption model ([`EqEffect`]) and rendered binaurally with the built-in CIPIC HRTF ([`BinauralEffect`]); sources
//! without a direction (music, the UI, sounds at the listener) and those beyond the voice budget keep Horizon's own
//! stereo gains. Distance attenuation stays Horizon's (vanilla's linear model), so what can be heard and how loud
//! does not change, only how it sounds. A source hidden by walls is also heard the way round them, from where its
//! paths arrive ([`crate::paths`]), with a second binaural voice.
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

use crate::acoustics::{AcousticsSettings, Snapshot, SurfaceTuning};
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
use crate::reflections::{self, Reflections, ReverbIr};
use crate::simulation::Reverb;
use crate::speaker_layout::{SpeakerLayout, SpeakerLayoutType};

/// Samples per Steam Audio frame (about 10.7 ms at 48 kHz).
pub const FRAME_SIZE: usize = 512;

/// The simulated reflections are physical: energy coming back to the listener from a source at the listener, relative
/// to the direct sound 1 m from it. Horizon's sources are quieter with distance than that (linear, not 1/r), so the
/// reverb is brought up by this gain: at a reverb level of 1, a 9 × 5 × 9 stone cave's reverb is about as loud as the
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
    /// transmission (a stone wall lets through about −36 dB), 0 lets everything through.
    pub walls: f32,
    /// `HORIZON_STEAM_AUDIO_PATHING=0` turns the sound paths around walls off (with occlusion on, a hidden source
    /// is then heard through the wall only).
    pub pathing: bool,
    /// `HORIZON_STEAM_AUDIO_REVERB=0` turns the reverb of the surroundings off.
    pub reverb: bool,
    /// `HORIZON_STEAM_AUDIO_REVERB_LEVEL=x` (0 to 2, default 0.5): reverb level in a fully enclosed place.
    pub reverb_level: f32,
    /// `HORIZON_STEAM_AUDIO_UNDERWATER=0`: the world sounds the same with the head under water (by default it is
    /// muffled, and water no longer counts as a wall around the listener).
    pub underwater: bool,
    /// `HORIZON_STEAM_AUDIO_DISTANCE=vanilla`: Horizon's linear distance attenuation, silent at the sound's range
    /// (16 blocks for a mob). By default it is physical ([`distance_gain`]): a mob is heard across a large cave.
    pub physical_distance: bool,
    /// `HORIZON_STEAM_AUDIO_RANGE=x` (2 to 8, default [`PHYSICAL_RANGE`]): with physical attenuation, how many of
    /// Horizon's ranges a source is heard across.
    pub hearing_range: f32,
    /// `HORIZON_STEAM_AUDIO_SOURCE_VOLUME=x` (0 to 2, default 1): volume of the sounds of the world (those with a
    /// place; music and the UI keep theirs).
    pub source_volume: f32,
    /// `HORIZON_STEAM_AUDIO_ABSORPTION=x` (0 to 4, default 1): factor on how much the surfaces absorb what reaches
    /// them, for the reflections and the reverb ([`SurfaceTuning`]).
    pub absorption: f32,
    /// `HORIZON_STEAM_AUDIO_SCATTERING=x` (0 to 1, default 0.05, Steam Audio's presets): how much of what the
    /// surfaces reflect goes off in every direction rather than mirror-like ([`SurfaceTuning`]).
    pub scattering: f32,
    /// The settings page's materials and blocks ([`crate::materials`]): kept by the page only.
    pub tuning: std::sync::Arc<crate::materials::Tuning>,
    /// The visual debug layers ([`crate::debug`]), each its own button on the settings page and off by default:
    /// the sources with their occlusion rays and walls, the paths round the walls and their probes, the reflection
    /// grid with sample reflection rays and the sources with reflections of their own, and the statistics panel.
    pub debug_sources: bool,
    pub debug_paths: bool,
    pub debug_reflections: bool,
    pub debug_stats: bool,
}

impl Config {
    /// Whether any visual debug layer is on.
    pub fn debugging(&self) -> bool {
        self.debug_sources || self.debug_paths || self.debug_reflections || self.debug_stats
    }

    /// How far sources reach, in Horizon's ranges, for [`distance_gain`] and [`audible_range`]: `None` with
    /// Horizon's attenuation.
    pub fn reach(&self) -> Option<f32> {
        self.physical_distance.then_some(self.hearing_range)
    }

    /// The reflecting surfaces' tuning.
    pub fn surfaces(&self) -> SurfaceTuning {
        SurfaceTuning {
            absorption: self.absorption,
            scattering: self.scattering,
        }
    }
}

/// How far a sound with Horizon's range `linear` is heard with physical attenuation by default: four times its range
/// ([`Config::hearing_range`]).
pub const PHYSICAL_RANGE: f32 = 4.0;

/// The distance attenuation of a source with Horizon's `linear` range (`None`: not attenuated) at `distance`, with
/// `reach` ([`Config::reach`]).
///
/// Horizon's (`reach == None`) is OpenAL's linear model, `1 − d / range`, silent at the range. The physical one
/// is Steam Audio's inverse distance, `r₀ / d`, with the reference distance `r₀` a quarter of the range: full
/// volume up close, the same as Horizon's at half the range, a quarter (−12 dB) at the range instead of silence,
/// and fading out smoothly over the last quarter before `reach` ranges ([`PHYSICAL_RANGE`] by default).
pub fn distance_gain(linear: Option<f32>, distance: f32, reach: Option<f32>) -> f32 {
    let Some(range) = linear else {
        return 1.0;
    };
    if range <= 0.0 {
        return 0.0;
    }
    let Some(reach) = reach else {
        return (1.0 - distance.min(range) / range).clamp(0.0, 1.0);
    };
    let reference = (range / 4.0).max(1.0);
    let inverse = (reference / distance.max(1e-3)).min(1.0);
    let end = range * reach.max(1.0);
    let fade_from = end * 0.75;
    let fade = if distance <= fade_from {
        1.0
    } else {
        let t = ((distance - fade_from) / (end - fade_from)).clamp(0.0, 1.0);
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    inverse * fade
}

/// How far a source with Horizon's range `linear` can be heard (`None`: everywhere).
pub fn audible_range(linear: Option<f32>, reach: Option<f32>) -> Option<f32> {
    linear.map(|range| reach.map_or(range, |reach| range * reach.max(1.0)))
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
            pathing: true,
            reverb: true,
            reverb_level: 0.5,
            underwater: true,
            physical_distance: true,
            hearing_range: PHYSICAL_RANGE,
            source_volume: 1.0,
            absorption: 1.0,
            scattering: crate::acoustics::DEFAULT_SCATTERING,
            tuning: std::sync::Arc::default(),
            debug_sources: false,
            debug_paths: false,
            debug_reflections: false,
            debug_stats: false,
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
        switch("HORIZON_STEAM_AUDIO_PATHING", &mut self.pathing);
        switch("HORIZON_STEAM_AUDIO_REVERB", &mut self.reverb);
        switch("HORIZON_STEAM_AUDIO_UNDERWATER", &mut self.underwater);
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
            "HORIZON_STEAM_AUDIO_RANGE",
            2.0..=8.0,
            &mut self.hearing_range,
        );
        number(
            "HORIZON_STEAM_AUDIO_SOURCE_VOLUME",
            0.0..=2.0,
            &mut self.source_volume,
        );
        number(
            "HORIZON_STEAM_AUDIO_ABSORPTION",
            0.0..=4.0,
            &mut self.absorption,
        );
        number(
            "HORIZON_STEAM_AUDIO_SCATTERING",
            0.0..=1.0,
            &mut self.scattering,
        );
        number(
            "HORIZON_STEAM_AUDIO_VOLUME_DB",
            -24.0..=24.0,
            &mut self.volume_db,
        );
        if let Some(v) = var("HORIZON_STEAM_AUDIO_DISTANCE") {
            self.physical_distance = !v.trim().eq_ignore_ascii_case("vanilla");
        }
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
        crate::materials::publish(&config);
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
        crate::materials::publish(&config);
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
    /// The source's gain without distance attenuation (0 when no camera hears it): what it sends to reflections
    /// simulated from where it is, which attenuate with distance themselves.
    pub source_gain: f32,
    /// A two-channel sound (music...): Horizon plays it as it is, left and right, unattenuated and unpanned
    /// (`direction` is `None`, `pan` unused). Its samples are read interleaved.
    pub stereo: bool,
    /// Where it is in the world, for a source with a place (`None`: relative to the listener, or not known). Sources
    /// near one another (the splashes of a waterfall, rain on the ground nearby) are rendered as one binaural voice:
    /// [`group_radius`].
    pub at: Option<[f64; 3]>,
}

/// How near a source must be to another, `distance` blocks from the listener, to be heard with it as one binaural
/// voice: a quarter of the distance (from there, a few degrees apart at most), never more than [`GROUP_RADIUS`]. Next
/// to the listener, sources stay apart. The world's acoustics take the same groups' sources as one too.
pub fn group_radius(distance: f64) -> f64 {
    (GROUP_SPREAD * distance).min(GROUP_RADIUS)
}

/// See [`group_radius`].
pub const GROUP_RADIUS: f64 = 2.5;
const GROUP_SPREAD: f64 = 0.25;
/// A source stays in its group while it is within this many times the group radius of where the group is: a source
/// leaving its group starts a binaural voice of its own, which can be heard as a click.
const GROUP_STAY: f64 = 2.0;

/// No group ([`Renderer::voice_group`]).
const NO_GROUP: usize = usize::MAX;

/// Sources near one another rendered as one binaural voice ([`group_radius`]): one EQ, one HRTF convolution and one way
/// round the walls for all of them, from their loudness-weighted direction. The state between frames is the group's,
/// under its key: the id of the source that started it, which it keeps after that source has stopped.
#[derive(Debug)]
struct Group {
    key: u64,
    /// The members' gains this callback (it is chosen for the binaural budget by them) and whether one of them is
    /// heard round the walls (a second binaural voice).
    weight: f32,
    path: bool,
    /// Whether it is rendered binaurally this callback.
    chosen: bool,
    /// What the members bring to this frame: their samples, each at its gain; their gains; their direct band gains,
    /// directions, and ways round (band gains, the loudest one's gain and direction), each weighted by its gain.
    mix: AudioBuffer,
    gain: f32,
    direct: [f32; NUM_BANDS],
    direction: [f32; 3],
    round: [f32; NUM_BANDS],
    round_loudest: f32,
    round_direction: Option<[f32; 3]>,
}

impl Group {
    fn new() -> Self {
        Self {
            key: 0,
            weight: 0.0,
            path: false,
            chosen: false,
            mix: AudioBuffer::new(1, FRAME_SIZE),
            gain: 0.0,
            direct: [0.0; NUM_BANDS],
            direction: [0.0; 3],
            round: [0.0; NUM_BANDS],
            round_loudest: 0.0,
            round_direction: None,
        }
    }

    /// Nothing brought to the frame yet.
    fn clear_frame(&mut self) {
        self.mix.make_silent();
        self.gain = 0.0;
        self.direct = [0.0; NUM_BANDS];
        self.direction = [0.0; 3];
        self.round = [0.0; NUM_BANDS];
        self.round_loudest = 0.0;
        self.round_direction = None;
    }
}

/// What a source's samples are, for [`Renderer::render`]: fills `out` with the source's samples resampled at `step`
/// source samples per output sample, advancing its position, and returns whether it still plays (a stopped source
/// fills the rest with silence). `out` is one frame of mono samples, or of interleaved stereo ones for a
/// [`Voice::stereo`] source.
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
    /// The way round the walls hiding it, while there is one or its sound fades out.
    path: Option<PathVoice>,
}

/// A source with reflections of its own ([`reflections::MAX_SOURCES`] of the nearest): Steam Audio's hybrid reverb of
/// them, fed with the source alone.
#[derive(Debug)]
struct SourceReverb {
    effect: HybridReverbEffect,
    input: AudioBuffer,
    /// The reflections being played (reverb times, parametric level per band, delay) and new ones not played yet.
    playing: Option<(Reverb, [f32; NUM_BANDS], usize)>,
    pending: Option<ReverbIr>,
    state: AudioEffectState,
    /// Whether the latest reflections have it (it sends to them, not to the listener's).
    own: bool,
    /// Whether it sent something this frame.
    sent: bool,
}

impl SourceReverb {
    /// Back to as new, for another source (its impulse responses to `reflections::recycle`).
    fn reset(&mut self) {
        self.effect.reset();
        self.input.make_silent();
        self.playing = None;
        if let Some(old) = self.pending.take() {
            reflections::recycle(old.fir);
        }
        self.state = AudioEffectState::TailComplete;
        self.own = false;
        self.sent = false;
    }

    fn new(settings: &AudioSettings, rate: u32) -> Self {
        Self {
            effect: HybridReverbEffect::new(
                settings,
                &OverlapSaveConvolutionSettings {
                    num_channels: reflections::NUM_CHANNELS,
                    ir_size: reflections::ir_size(rate),
                },
                &mut CRand::default(),
            ),
            input: AudioBuffer::new(1, FRAME_SIZE),
            playing: None,
            pending: None,
            state: AudioEffectState::TailComplete,
            own: false,
            sent: false,
        }
    }
}

/// The audio thread's load (rendering time over playing time of a frame) above which fewer sources are rendered
/// binaurally, and below which more are again; the frames between two changes, and the fewest binaural voices.
const HIGH_LOAD: f32 = 0.6;
const LOW_LOAD: f32 = 0.35;
const LOAD_SETTLE_FRAMES: u32 = 20;
const MIN_BINAURAL_VOICES: usize = 8;

/// Most source reverbs at once: those with reflections of their own and as many ringing out.
const MAX_SOURCE_REVERBS: usize = 2 * reflections::MAX_SOURCES;

/// The sound of a source coming round what hides it ([`crate::paths`]): a second binaural voice, from where the
/// paths arrive, with what their bends and length take off each band.
#[derive(Debug)]
struct PathVoice {
    binaural: BinauralEffect,
    eq: EqEffect,
    /// Band gains (normalized), the gain the last frame ended at and the direction (world) last played.
    eq_gains: [f32; NUM_BANDS],
    gain: f32,
    direction: [f32; 3],
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
    /// Each source's group this callback ([`NO_GROUP`] for none), the groups (the first `group_count`; the rest are
    /// kept for later callbacks, with their buffers), and the groups in order of loudness.
    voice_group: Vec<usize>,
    groups: Vec<Group>,
    group_count: usize,
    group_order: Vec<usize>,
    /// Across callbacks: each source's group key, and where each group that sources can join is (where the source
    /// that started it is, or was last). The `next_` ones are filled for this callback, then swapped in.
    memberships: HashMap<u64, u64>,
    anchors: HashMap<u64, [f64; 3]>,
    next_memberships: HashMap<u64, u64>,
    next_anchors: HashMap<u64, (usize, Option<[f64; 3]>)>,
    mono: AudioBuffer,
    /// A source's samples before the direct path's EQ, for its path round the walls.
    dry: AudioBuffer,
    stereo: AudioBuffer,
    /// A two-channel source's frame, interleaved ([`Voice::stereo`]).
    interleaved: Vec<f32>,
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
    /// The sources with reflections of their own (by mixer channel id), and where their reverbs are summed.
    source_reverbs: HashMap<u64, SourceReverb>,
    /// Source reverbs made beforehand and ones that rang out, for the next sources: making one (a second of
    /// four-channel convolution) takes over a millisecond, too long for the audio thread.
    spare_reverbs: Vec<SourceReverb>,
    /// How much of a frame's time rendering it takes (smoothed), the binaural budget it allows and the frames since that
    /// budget changed.
    load: f32,
    voice_limit: usize,
    since_change: u32,
    source_ambisonics: AudioBuffer,
    wet: f32,
    /// The reflections being played (reverb times, parametric level per band, delay) and new ones not played yet.
    reflections: Option<(Reverb, [f32; NUM_BANDS], usize)>,
    pending: Option<Box<Reflections>>,
    /// Reflections given with [`Renderer::set_reflections`], taken at the next frame as published ones are.
    incoming: Option<Box<Reflections>>,
    /// The listener's orientation in the world.
    orientation: CoordinateSpace3,
    /// How far under water the listener sounds (0 dry, 1 under), moving towards what is published over a few frames.
    underwater: f32,
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
            source_reverbs: HashMap::with_capacity(MAX_SOURCE_REVERBS),
            load: 0.0,
            voice_limit: usize::MAX,
            since_change: 0,
            spare_reverbs: (0..MAX_SOURCE_REVERBS)
                .map(|_| SourceReverb::new(&audio_settings, rate))
                .collect(),
            source_ambisonics: AudioBuffer::new(reflections::NUM_CHANNELS, FRAME_SIZE),
            wet: 0.0,
            reflections: None,
            pending: None,
            incoming: None,
            orientation: CoordinateSpace3::default(),
            underwater: 0.0,
            air: AirAbsorptionModel::default(),
            states: HashMap::new(),
            voices: Vec::new(),
            binaural: Vec::new(),
            playing: Vec::new(),
            order: Vec::new(),
            voice_group: Vec::new(),
            groups: Vec::new(),
            group_count: 0,
            group_order: Vec::new(),
            memberships: HashMap::new(),
            anchors: HashMap::new(),
            next_memberships: HashMap::new(),
            next_anchors: HashMap::new(),
            mono: AudioBuffer::new(1, FRAME_SIZE),
            dry: AudioBuffer::new(1, FRAME_SIZE),
            stereo: AudioBuffer::new(2, FRAME_SIZE),
            interleaved: vec![0.0; 2 * FRAME_SIZE],
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
        self.incoming = Some(reflections);
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
                let start = std::time::Instant::now();
                self.render_frame(read);
                self.measure_load(start.elapsed());
                self.taken = 0;
            }
            frame[0] = self.block[2 * self.taken];
            frame[1] = self.block[2 * self.taken + 1];
            self.taken += 1;
        }
        if self.config.debugging() {
            self.publish_debug();
        }
    }

    /// What this callback rendered, for the debug mode's statistics and labels ([`crate::debug`]).
    fn publish_debug(&self) {
        crate::debug::publish_renderer(|s| {
            s.rate = self.rate;
            s.voices = self.voices.len();
            s.positional = self
                .voices
                .iter()
                .filter(|v| v.direction.is_some() && v.gain > 0.0)
                .count();
            let chosen = || self.groups[..self.group_count].iter().filter(|g| g.chosen);
            s.binaural = chosen().count();
            s.grouped = self.order.len() - s.binaural.min(self.order.len());
            s.paths = chosen()
                .filter(|g| {
                    self.states
                        .get(&g.key)
                        .is_some_and(|state| state.path.is_some())
                })
                .count();
            s.budget = self.config.max_binaural_voices;
            s.limit = self.config.max_binaural_voices.min(self.voice_limit);
            s.load = self.load;
            s.source_reverbs = self.source_reverbs.len();
            s.wet = self.wet;
            s.underwater = self.underwater;
            s.binaural_ids.clear();
            s.binaural_ids
                .extend(self.order.iter().map(|&i| self.voices[i].id));
        });
    }

    /// Follows how long frames take to render, relative to how long they play: when the audio thread gets near its
    /// deadline, fewer sources are rendered binaurally ([`Renderer::choose_binaural`]); with room again, more.
    fn measure_load(&mut self, took: std::time::Duration) {
        let frame = FRAME_SIZE as f32 / self.rate as f32;
        let load = took.as_secs_f32() / frame;
        self.load += (load - self.load) * 0.1;
        self.since_change += 1;
        if self.since_change < LOAD_SETTLE_FRAMES {
            return;
        }
        let max = self.config.max_binaural_voices.max(MIN_BINAURAL_VOICES);
        // Starts unlimited (`usize::MAX`): bring it to the configuration's first, so `* 3` can't overflow.
        self.voice_limit = self.voice_limit.min(max);
        if self.load > HIGH_LOAD && self.voice_limit > MIN_BINAURAL_VOICES {
            self.voice_limit = (self.voice_limit * 3 / 4).max(MIN_BINAURAL_VOICES);
            self.since_change = 0;
            log::debug!(
                "steam_audio: rendering takes {:.0}% of the time; {} binaural voices",
                self.load * 100.0,
                self.voice_limit
            );
        } else if self.load < LOW_LOAD && self.voice_limit < max {
            self.voice_limit = (self.voice_limit + 2).min(max);
            self.since_change = 0;
        }
        self.voice_limit = self.voice_limit.min(max);
    }

    /// The sources rendered binaurally: the loudest with a direction, within the budget, where a source with a way
    /// round the walls counts twice (its path is a second binaural voice). The budget is the configuration's, lowered
    /// while the audio thread is busy ([`Renderer::measure_load`]).
    ///
    /// Sources near one another go as one ([`group_radius`]): they count once, by their gains together.
    fn choose_binaural(&mut self) {
        self.group_sources();
        let voices = &self.voices;
        let groups = &mut self.groups[..self.group_count];
        let budget = self.config.max_binaural_voices.min(self.voice_limit);
        self.group_order.clear();
        self.group_order.extend(0..groups.len());
        self.group_order
            .sort_unstable_by(|&a, &b| groups[b].weight.total_cmp(&groups[a].weight));
        let mut used = 0;
        for &g in &self.group_order {
            let c = 1 + usize::from(groups[g].path);
            groups[g].chosen = used + c <= budget;
            if groups[g].chosen {
                used += c;
            }
        }
        self.order.clear();
        self.binaural.clear();
        self.binaural.resize(voices.len(), false);
        for i in 0..voices.len() {
            let g = self.voice_group[i];
            if g != NO_GROUP && groups[g].chosen {
                self.binaural[i] = true;
                self.order.push(i);
            }
        }
        self.order
            .sort_unstable_by(|&a, &b| voices[b].gain.total_cmp(&voices[a].gain));
    }

    /// Puts this callback's sources with a direction in groups ([`Renderer::voice_group`]). A source keeps its group
    /// while it stays near where the group is; a new source joins the nearest group within [`group_radius`] of it, or
    /// starts one of its own. A source that already plays on its own never joins another group: its binaural voice
    /// would start over, which can be heard.
    fn group_sources(&mut self) {
        let paths = self.config.occlusion && self.config.pathing;
        let Self {
            voices,
            acoustics,
            voice_group,
            groups,
            group_count,
            group_order,
            memberships,
            anchors,
            next_memberships,
            next_anchors,
            ..
        } = self;
        voice_group.clear();
        voice_group.resize(voices.len(), NO_GROUP);
        next_memberships.clear();
        next_anchors.clear();
        *group_count = 0;
        // The oldest first (ids grow): groups are started by the sources that came first.
        group_order.clear();
        group_order.extend(
            (0..voices.len()).filter(|&i| voices[i].direction.is_some() && voices[i].gain > 0.0),
        );
        group_order.sort_unstable_by_key(|&i| voices[i].id);
        let near = |a: [f64; 3], b: [f64; 3], r: f64| {
            (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>() <= r * r
        };
        for &i in group_order.iter() {
            let v = &voices[i];
            let radius = group_radius(f64::from(v.distance));
            let previous = memberships.get(&v.id).copied();
            let key = match v.at {
                None => v.id,
                Some(at) => match previous {
                    // Its own group, or one it is still near.
                    Some(key)
                        if key == v.id
                            || next_anchors
                                .get(&key)
                                .and_then(|(_, a)| *a)
                                .or_else(|| anchors.get(&key).copied())
                                .is_some_and(|a| near(at, a, GROUP_STAY * radius)) =>
                    {
                        key
                    }
                    // It left its group: a voice of its own.
                    Some(_) => v.id,
                    // New: the nearest group it can join, or its own.
                    None => next_anchors
                        .iter()
                        .filter_map(|(&key, &(_, a))| {
                            let a = a?;
                            let d2 = (0..3).map(|k| (at[k] - a[k]).powi(2)).sum::<f64>();
                            (d2 <= radius * radius).then_some((key, d2))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map_or(v.id, |(key, _)| key),
                },
            };
            let g = match next_anchors.get(&key) {
                Some(&(g, _)) => g,
                None => {
                    let g = *group_count;
                    *group_count += 1;
                    if groups.len() < *group_count {
                        groups.push(Group::new());
                    }
                    let group = &mut groups[g];
                    group.key = key;
                    group.weight = 0.0;
                    group.path = false;
                    group.chosen = false;
                    // Where the group is: where its first source is, or was last; none for a source without a place.
                    let anchor = v.at.and(if key == v.id {
                        v.at
                    } else {
                        anchors.get(&key).copied().or(v.at)
                    });
                    next_anchors.insert(key, (g, anchor));
                    g
                }
            };
            // The source that started the group moves it with it.
            if key == v.id
                && let Some(entry) = next_anchors.get_mut(&key)
                && v.at.is_some()
            {
                entry.1 = v.at;
            }
            voice_group[i] = g;
            groups[g].weight += v.gain;
            groups[g].path |= paths && acoustics.get(v.id).is_some_and(|a| a.path_gain() > 0.0);
            next_memberships.insert(v.id, key);
        }
        std::mem::swap(memberships, next_memberships);
        anchors.clear();
        anchors.extend(
            next_anchors
                .iter()
                .filter_map(|(&key, &(_, a))| Some((key, a?))),
        );
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
            voice_group,
            groups,
            group_count,
            mono,
            dry,
            stereo,
            interleaved,
            acoustics,
            reverb_in,
            ambisonics,
            reverb_out,
            hybrid,
            decode,
            reverb_state,
            source_reverbs,
            spare_reverbs,
            source_ambisonics,
            wet,
            reflections: playing_reflections,
            pending,
            incoming,
            orientation,
            underwater,
            rate,
            block,
            ..
        } = self;
        // Under water: every sound of the world loses its highs on the way in (the direct sound, the way round, what
        // goes to the reverb). The change takes about five frames (50 ms).
        let target = if config.underwater && acoustics.underwater {
            1.0
        } else {
            0.0
        };
        *underwater += (target - *underwater) * 0.35;
        if (*underwater - target).abs() < 1e-3 {
            *underwater = target;
        }
        let medium: [f32; NUM_BANDS] = std::array::from_fn(|b| {
            1.0 - *underwater * (1.0 - crate::acoustics::UNDERWATER_GAINS[b])
        });
        let medium_send = medium.iter().sum::<f32>() / NUM_BANDS as f32;
        // The reverb level this frame ramps to; the sources send to the reverb while it or the last one is above 0
        // (not at a level of 0: their reverbs would convolve silence).
        let reverb_target = if config.reverb {
            config.reverb_level * REFLECTIONS_GAIN
        } else {
            0.0
        };
        let send_reverb = reverb_target > 0.0 || *wet > 0.0;
        reverb_in.make_silent();
        for r in source_reverbs.values_mut() {
            r.input.make_silent();
            r.sent = false;
        }
        let groups = &mut groups[..*group_count];
        for g in groups.iter_mut().filter(|g| g.chosen) {
            g.clear_frame();
        }
        for (i, voice) in voices.iter().enumerate() {
            if !playing[i] {
                continue;
            }
            if voice.stereo {
                // Two channels (music...): as Horizon plays them, left and right as they are, dry.
                playing[i] = read(voice.id, voice.step, interleaved);
                if voice.gain != 0.0 {
                    for (out, s) in block.chunks_exact_mut(2).zip(interleaved.chunks_exact(2)) {
                        out[0] += s[0] * voice.gain;
                        out[1] += s[1] * voice.gain;
                    }
                }
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
            // The reverb hears the source whether or not a wall hides it (sound goes round), through the air: its own
            // reflections if it has some (simulated from where it is, with the air and the distance), else the
            // listener's.
            if send_reverb {
                let (input, send) = match source_reverbs.get_mut(&voice.id).filter(|r| r.own) {
                    Some(r) => {
                        r.sent = true;
                        (&mut r.input, voice.source_gain * medium_send)
                    }
                    None => (&mut *reverb_in, voice.gain * air_gains[1] * medium_send),
                };
                for (r, &s) in input.channel_mut(0).iter_mut().zip(mono.channel(0)) {
                    *r += s * send;
                }
            }
            let mut eq_gains: [f32; NUM_BANDS] =
                std::array::from_fn(|b| air_gains[b] * direct[b] * medium[b]);
            let mut overall = 1.0;
            EqEffect::normalize_gains(&mut eq_gains, &mut overall);
            // The way round the walls, for the hidden part of the source: band gains (the bends, and the air along
            // the longer way), gain and direction.
            let path = if config.occlusion && config.pathing {
                acoustics.get(voice.id).and_then(|a| {
                    let p = a.path?;
                    let gain = a.path_gain();
                    (gain > 0.0).then(|| {
                        let mut eq: [f32; NUM_BANDS] = std::array::from_fn(|b| {
                            let air = if config.air_absorption {
                                air.evaluate(p.distance.max(0.0), b)
                            } else {
                                1.0
                            };
                            air * p.eq[b] * medium[b]
                        });
                        let mut overall = 1.0;
                        EqEffect::normalize_gains(&mut eq, &mut overall);
                        (eq, gain * overall, p.direction)
                    })
                })
            } else {
                None
            };
            match (binaural[i], voice.direction) {
                (true, Some(direction)) => {
                    // To its group's binaural voice, rendered once all have brought their part (below).
                    let g = &mut groups[voice_group[i]];
                    let w = voice.gain;
                    for (m, &s) in g.mix.channel_mut(0).iter_mut().zip(mono.channel(0)) {
                        *m += s * w;
                    }
                    g.gain += w;
                    let length = direction.iter().map(|c| c * c).sum::<f32>().sqrt();
                    if length > 0.0 {
                        for (d, c) in g.direction.iter_mut().zip(direction) {
                            *d += w * c / length;
                        }
                    }
                    for (d, e) in g.direct.iter_mut().zip(eq_gains) {
                        *d += w * e * overall;
                    }
                    if let Some((eq, gain, direction)) = path {
                        for (r, e) in g.round.iter_mut().zip(eq) {
                            *r += w * gain * e;
                        }
                        if w * gain > g.round_loudest {
                            g.round_loudest = w * gain;
                            g.round_direction = Some(direction);
                        }
                    }
                }
                _ => {
                    // Beyond the binaural budget: Horizon's panning, with the walls' and the air's mean loss, and
                    // the way round's mean gain.
                    let mean = |g: &[f32; NUM_BANDS]| g.iter().sum::<f32>() / NUM_BANDS as f32;
                    let loss = mean(&eq_gains) * overall
                        + path.map_or(0.0, |(eq, gain, _)| mean(&eq) * gain);
                    let [left, right] = voice.pan.map(|pan| pan * voice.gain * loss);
                    for (out, &s) in block.chunks_exact_mut(2).zip(mono.channel(0)) {
                        out[0] += s * left;
                        out[1] += s * right;
                    }
                }
            }
        }
        // The groups' binaural voices: their members' samples (at their gains, over their gains together), with their
        // weighted band gains, from their weighted direction. A group of one is its source as it is.
        for g in groups.iter_mut().filter(|g| g.chosen && g.gain > 0.0) {
            let inverse = g.gain.recip();
            for s in g.mix.channel_mut(0) {
                *s *= inverse;
            }
            let mut eq_gains: [f32; NUM_BANDS] = std::array::from_fn(|b| g.direct[b] * inverse);
            let mut overall = 1.0;
            EqEffect::normalize_gains(&mut eq_gains, &mut overall);
            let length = g.direction.iter().map(|c| c * c).sum::<f32>().sqrt();
            let direction = if length > 1e-6 {
                g.direction.map(|c| c / length)
            } else {
                [0.0, 0.0, -1.0]
            };
            let path = g.round_direction.map(|direction| {
                let mut eq: [f32; NUM_BANDS] = std::array::from_fn(|b| g.round[b] * inverse);
                let mut overall = 1.0;
                EqEffect::normalize_gains(&mut eq, &mut overall);
                (eq, overall, direction)
            });
            let voice_gain = g.gain;
            let mono = &mut g.mix;
            {
                {
                    let state = states.entry(g.key).or_insert_with(|| VoiceState {
                        binaural: BinauralEffect::new(settings, hrtf),
                        eq: EqEffect::new(settings),
                        gain: voice_gain * overall,
                        frame,
                        path: None,
                    });
                    state.frame = frame;
                    if path.is_some() || state.path.is_some() {
                        dry.channel_mut(0).copy_from_slice(mono.channel(0));
                    }
                    state.eq.apply_in_place(&eq_gains, mono);
                    let gain = voice_gain * overall;
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
                    // The way round: from where the paths arrive, turning with the head. When it goes, its last
                    // sound fades out over a frame.
                    let target = match path {
                        Some((eq, gain, direction)) => {
                            let p = state.path.get_or_insert_with(|| PathVoice {
                                binaural: BinauralEffect::new(settings, hrtf),
                                eq: EqEffect::new(settings),
                                eq_gains: eq,
                                gain: 0.0,
                                direction,
                            });
                            p.eq_gains = eq;
                            p.direction = direction;
                            voice_gain * gain
                        }
                        None => 0.0,
                    };
                    if let Some(p) = &mut state.path {
                        p.eq.apply_in_place(&p.eq_gains, dry);
                        ramp(dry.channel_mut(0), p.gain, target);
                        let local =
                            orientation.direction_world_to_local(Vector3::from_array(p.direction));
                        let mut params = BinauralParams::new(local, hrtf);
                        if config.bilinear {
                            params.interpolation = HrtfInterpolation::Bilinear;
                        }
                        params.phase_type = HrtfPhaseType::None;
                        p.binaural.apply(params, dry, stereo);
                        for (j, out) in block.chunks_exact_mut(2).enumerate() {
                            out[0] += stereo.channel(0)[j];
                            out[1] += stereo.channel(1)[j];
                        }
                        let silent = p.gain == 0.0 && target == 0.0;
                        p.gain = target;
                        if silent {
                            state.path = None;
                        }
                    }
                }
            }
        }
        // Sources that stopped, or left the binaural set, drop their state.
        states.retain(|_, state| state.frame == frame);

        // The reverb bus: the simulated reflections of the surroundings (convolved, turning with the head) and their
        // parametric tail. The level fades what goes in, so a change leaves the tail already ringing as it is.
        let target = reverb_target;
        if pending.is_none()
            && let Some(mut new) = incoming.take().or_else(reflections::take)
        {
            if new.rate == *rate {
                // The sources' reflections go to their own reverbs; the others' are left to ring out.
                for r in source_reverbs.values_mut() {
                    r.own = false;
                }
                // First a reverb for each of them; then their impulse responses.
                for k in 0..new.sources.len() {
                    let id = new.sources[k].0;
                    if source_reverbs.contains_key(&id) {
                        continue;
                    }
                    // At most `MAX_SOURCE_REVERBS`: a new source's takes the place of one ringing out, never of
                    // one of these sources (their `own` is only set below).
                    let sources = &new.sources;
                    if source_reverbs.len() >= MAX_SOURCE_REVERBS
                        && let Some(&gone) = source_reverbs
                            .iter()
                            .find(|(rid, r)| !r.own && !sources.iter().any(|(s, _)| s == *rid))
                            .map(|(id, _)| id)
                        && let Some(mut r) = source_reverbs.remove(&gone)
                    {
                        r.reset();
                        spare_reverbs.push(r);
                    }
                    let r = spare_reverbs
                        .pop()
                        .unwrap_or_else(|| SourceReverb::new(settings, *rate));
                    source_reverbs.insert(id, r);
                }
                for (id, ir) in new.sources.drain(..) {
                    let Some(r) = source_reverbs.get_mut(&id) else {
                        continue;
                    };
                    r.own = true;
                    if let Some(old) = r.pending.replace(ir) {
                        reflections::recycle(old.fir);
                    }
                }
                *pending = Some(new);
            } else {
                for fir in new.into_firs() {
                    reflections::recycle(fir);
                }
            }
        }
        let sources_ring = source_reverbs
            .values()
            .any(|r| r.sent || r.state == AudioEffectState::TailRemaining);
        if target > 0.0
            || *wet > 0.0
            || *reverb_state == AudioEffectState::TailRemaining
            || sources_ring
        {
            ramp(reverb_in.channel_mut(0), *wet, target);
            ambisonics.make_silent();
            let mut any = false;
            // At most one new impulse response a frame: each crossfades, convolving with both for that frame.
            let mut crossfades = 1usize;
            // With nothing going in, new reflections wait: taken now, the convolution would never get their impulse
            // response while the parametric tail played their reverb times.
            let silent = target == 0.0 && *wet == 0.0;
            let mut update = if silent { None } else { pending.take() };
            crossfades -= usize::from(update.is_some());
            if let Some(new) = &update {
                *playing_reflections = Some((new.reverb, new.eq, new.delay));
            }
            if silent {
                // The reverb is off and nothing goes in: what rings still rings out, then the bus stops.
                if *reverb_state == AudioEffectState::TailRemaining {
                    *reverb_state = hybrid.tail(ambisonics);
                    any = true;
                }
            } else if let Some((reverb, eq, delay)) = playing_reflections.as_ref() {
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
                any = true;
            }
            // The effect handed back the impulse response it played until now.
            if let Some(old) = update {
                reflections::recycle(old.fir);
            }
            // The sources' own reverbs, summed into the same Ambisonics.
            for r in source_reverbs.values_mut() {
                if !(r.sent || r.state == AudioEffectState::TailRemaining) {
                    continue;
                }
                if !r.sent {
                    // Nothing goes in any more: its tail rings out, and ends (`apply` would keep it going for ever,
                    // starting the tail over at every frame).
                    if let Some(old) = r.pending.take() {
                        reflections::recycle(old.fir);
                    }
                    r.state = r.effect.tail(source_ambisonics);
                    add_ambisonics(ambisonics, source_ambisonics);
                    any = true;
                    continue;
                }
                ramp(r.input.channel_mut(0), *wet, target);
                let mut update = if crossfades > 0 {
                    r.pending.take()
                } else {
                    None
                };
                crossfades -= usize::from(update.is_some());
                if let Some(new) = &update {
                    r.playing = Some((new.reverb, new.eq, new.delay));
                }
                if let Some((reverb, eq, delay)) = r.playing.as_ref() {
                    let ir = match &mut update {
                        Some(new) => HybridReverbIr::Updated(&mut new.fir),
                        None => HybridReverbIr::Current,
                    };
                    r.state = r.effect.apply(
                        HybridReverbParams {
                            ir,
                            num_channels: reflections::NUM_CHANNELS,
                            reverb,
                            eq_coeffs: eq,
                            delay: *delay,
                        },
                        &r.input,
                        source_ambisonics,
                    );
                    add_ambisonics(ambisonics, source_ambisonics);
                    any = true;
                }
                if let Some(old) = update {
                    reflections::recycle(old.fir);
                }
            }
            if any {
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
        }
        // A source's reverb goes once it has no reflections of its own and has rung out, back to the spares.
        let gone: Option<u64> = source_reverbs
            .iter()
            .find(|(_, r)| !(r.own || r.sent || r.state == AudioEffectState::TailRemaining))
            .map(|(id, _)| *id);
        if let Some(mut r) = gone.and_then(|id| source_reverbs.remove(&id)) {
            r.reset();
            if spare_reverbs.len() < MAX_SOURCE_REVERBS {
                spare_reverbs.push(r);
            }
        }
        *wet = target;
    }
}

/// Adds the Ambisonics `from` into `into`, channel by channel.
fn add_ambisonics(into: &mut AudioBuffer, from: &AudioBuffer) {
    for ch in 0..reflections::NUM_CHANNELS {
        for (a, &s) in into.channel_mut(ch).iter_mut().zip(from.channel(ch)) {
            *a += s;
        }
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
    position: f64,
    step: f64,
    looping: bool,
    out: &mut [f32],
) -> (f64, bool) {
    read_frames(samples, 1, position, step, looping, out)
}

/// [`read_linear`] for a source of `channels` interleaved channels (`position` and `step` in frames): fills `out`,
/// interleaved too, a frame at a time.
pub fn read_frames(
    samples: &[f32],
    channels: usize,
    mut position: f64,
    step: f64,
    looping: bool,
    out: &mut [f32],
) -> (f64, bool) {
    let channels = channels.max(1);
    let frames = samples.len() / channels;
    for (k, o) in out.chunks_exact_mut(channels).enumerate() {
        let mut i = position as usize;
        if i + 1 >= frames {
            if looping && frames > 1 {
                position = 0.0;
                i = 0;
            } else {
                out[k * channels..].fill(0.0);
                return (position, false);
            }
        }
        let t = (position - i as f64) as f32;
        for (c, o) in o.iter_mut().enumerate() {
            *o = samples[i * channels + c] * (1.0 - t) + samples[(i + 1) * channels + c] * t;
        }
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

        // Two interleaved channels: frames (0, 10), (1, 11), (2, 12).
        let stereo = [0.0, 10.0, 1.0, 11.0, 2.0, 12.0];
        let mut out = [9.0; 4];
        let (position, playing) = read_frames(&stereo, 2, 0.0, 0.5, false, &mut out);
        assert_eq!(out, [0.0, 10.0, 0.5, 10.5]);
        assert_eq!((position, playing), (1.0, true));
        let (_, playing) = read_frames(&stereo, 2, 1.0, 1.0, false, &mut out);
        assert!(!playing);
        assert_eq!(out, [1.0, 11.0, 0.0, 0.0]);
    }

    #[test]
    fn stereo_sources_play_as_they_are() {
        let mut renderer =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        // Left channel all 0.25, right all -0.5: they come out as they are, at the gain.
        let music: Vec<f32> = (0..2 * 48_000)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.5 })
            .collect();
        let mut position = 0.0;
        let mut out = vec![0.0; 2 * 4 * FRAME_SIZE];
        renderer.begin(1);
        renderer.push(Voice {
            id: 3,
            gain: 0.8,
            pan: [0.0, 0.0],
            direction: None,
            distance: 0.0,
            step: 1.0,
            source_gain: 0.8,
            stereo: true,
            at: None,
        });
        renderer.render(&mut out, 2, &mut |_, step, buf| {
            assert_eq!(buf.len(), 2 * FRAME_SIZE);
            let (p, playing) = read_frames(&music, 2, position, step, true, buf);
            position = p;
            playing
        });
        let last = &out[out.len() - 2..];
        assert!((last[0] - 0.2).abs() < 1e-6, "{last:?}");
        assert!((last[1] + 0.4).abs() < 1e-6, "{last:?}");
        assert!(renderer.states.is_empty());
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
            source_gain: 0.5,
            stereo: false,
            at: None,
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
    use crate::paths::PathAcoustics;
    use crate::reflections::{GRID_SIZE, GridBuilder, NUM_CHANNELS, ReverbIr, Simulation, Voxels};
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
            last = simulation.run(&grid, listener, &[], &mut Vec::new());
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
            sources: Vec::new(),
        })
    }

    #[test]
    fn a_source_with_reflections_of_its_own_is_heard_in_them() {
        // The listener's reflections come from the west, source 7's own from the east (looking north).
        let mut renderer = renderer_with(&[]);
        let mut both = reflections_from([-1.0, 0.0, 0.0]);
        let own = reflections_from([1.0, 0.0, 0.0]);
        both.sources.push((
            7,
            ReverbIr {
                fir: own.fir,
                reverb: own.reverb,
                eq: own.eq,
                delay: own.delay,
            },
        ));
        renderer.set_reflections(both);
        // The reflections are taken at the first frame; the source sends to its own from then on.
        ear_energy(&mut renderer, Some([0.0, 0.0, -1.0]));
        assert!(renderer.source_reverbs.get(&7).is_some_and(|r| r.own));
        let [l, r] = tail(&mut renderer, 0);
        assert!(r > 2.0 * l, "from the east: left {l}, right {r}");
        // Without them, the same source is heard in the listener's (from the west).
        let mut shared = renderer_with(&[]);
        shared.set_reflections(reflections_from([-1.0, 0.0, 0.0]));
        ear_energy(&mut shared, Some([0.0, 0.0, -1.0]));
        let [l, r] = tail(&mut shared, 0);
        assert!(l > 2.0 * r, "from the west: left {l}, right {r}");
    }

    #[test]
    fn physical_distance_attenuation_reaches_past_the_range() {
        let mob = Some(16.0);
        // Horizon's: silent at the range.
        assert_eq!(distance_gain(mob, 16.0, None), 0.0);
        assert!((distance_gain(mob, 8.0, None) - 0.5).abs() < 1e-6);
        // Physical: full up close, the same at half the range, then a long tail fading out at four ranges.
        assert_eq!(distance_gain(mob, 2.0, Some(PHYSICAL_RANGE)), 1.0);
        assert!((distance_gain(mob, 8.0, Some(PHYSICAL_RANGE)) - 0.5).abs() < 1e-6);
        assert!((distance_gain(mob, 16.0, Some(PHYSICAL_RANGE)) - 0.25).abs() < 1e-6);
        assert!((distance_gain(mob, 32.0, Some(PHYSICAL_RANGE)) - 0.125).abs() < 1e-6);
        let tail: Vec<f32> = (40..=64)
            .map(|d| distance_gain(mob, d as f32, Some(PHYSICAL_RANGE)))
            .collect();
        assert!(tail.windows(2).all(|w| w[1] <= w[0]), "{tail:?}");
        assert_eq!(distance_gain(mob, 64.0, Some(PHYSICAL_RANGE)), 0.0);
        assert_eq!(distance_gain(None, 100.0, Some(PHYSICAL_RANGE)), 1.0);
        assert_eq!(audible_range(mob, Some(PHYSICAL_RANGE)), Some(64.0));
        assert_eq!(audible_range(mob, None), Some(16.0));
        // A shorter or longer reach on the settings page.
        assert_eq!(audible_range(mob, Some(2.0)), Some(32.0));
        assert_eq!(distance_gain(mob, 32.0, Some(2.0)), 0.0);
        assert!(distance_gain(mob, 64.0, Some(8.0)) > 0.0);
    }

    /// The audio thread's load: 32 voices for a second of sound, with the listener's reverb only and with
    /// `own` sources having reflections of their own. Prints the share of real time the rendering takes.
    fn audio_load(own: usize) -> f64 {
        let mut renderer = renderer_with(&[]);
        let mut reflections = reflections_from([-1.0, 0.0, 0.0]);
        for id in 0..own as u64 {
            let r = reflections_from([1.0, 0.0, 0.0]);
            reflections.sources.push((
                id,
                ReverbIr {
                    fir: r.fir,
                    reverb: r.reverb,
                    eq: r.eq,
                    delay: r.delay,
                },
            ));
        }
        renderer.set_reflections(reflections);
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
        let mut positions = [0.0f64; 32];
        let mut out = vec![0.0; 2 * 480];
        let push = |renderer: &mut Renderer| {
            renderer.begin(1);
            for id in 0..32u64 {
                let a = id as f32 * 0.4;
                renderer.push(Voice {
                    id,
                    gain: 0.3,
                    pan: [0.5, 0.5],
                    direction: Some([a.sin(), 0.0, -a.cos()]),
                    distance: 4.0 + id as f32,
                    step: 1.0,
                    source_gain: 0.3,
                    stereo: false,
                    at: None,
                });
            }
        };
        // Warm up (the reflections are taken at the first frame), then a second of 10 ms callbacks.
        for _ in 0..10 {
            push(&mut renderer);
            renderer.render(&mut out, 2, &mut |id, step, buf| {
                let p = &mut positions[id as usize];
                *p = read_linear(&tone, *p, step, true, buf).0;
                true
            });
        }
        let start = std::time::Instant::now();
        for _ in 0..100 {
            push(&mut renderer);
            renderer.render(&mut out, 2, &mut |id, step, buf| {
                let p = &mut positions[id as usize];
                *p = read_linear(&tone, *p, step, true, buf).0;
                true
            });
        }
        start.elapsed().as_secs_f64()
    }

    /// A busy cave: `voices` sources, every one hidden with a way round (two binaural voices each), `own` with
    /// reflections of their own, new reflections every 250 ms and a new sound (channel) every 50 ms. Returns the mean
    /// and the longest 10 ms callback.
    fn busy_audio_load(voices: u64, own: u64) -> (std::time::Duration, std::time::Duration) {
        let hidden = |id: u64| {
            (
                id,
                SourceAcoustics {
                    occlusion: 0.3,
                    transmission: [0.2, 0.1, 0.05],
                    through: [1.0; NUM_BANDS],
                    path: Some(PathAcoustics {
                        direction: [1.0, 0.0, 0.0],
                        gain: 0.7,
                        eq: [0.8, 0.5, 0.3],
                        distance: 12.0,
                    }),
                },
            )
        };
        let reflections = |first: u64| {
            let mut r = reflections_from([-1.0, 0.0, 0.0]);
            for id in first..first + own {
                let s = reflections_from([1.0, 0.0, 0.0]);
                r.sources.push((
                    id,
                    ReverbIr {
                        fir: s.fir,
                        reverb: s.reverb,
                        eq: s.eq,
                        delay: s.delay,
                    },
                ));
            }
            r
        };
        let tone: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
        let mut out = vec![0.0; 2 * 480];
        let mut renderer = renderer_with(&[]);
        let (mut total, mut longest) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        let mut base = 0u64;
        for call in 0..310u64 {
            // A new sound every 5 callbacks: the oldest goes, a new channel id comes.
            if call % 5 == 0 {
                base += 1;
                renderer
                    .set_acoustics(Snapshot::fixed((base..base + voices).map(hidden).collect()));
            }
            if call % 25 == 0 {
                renderer.set_reflections(reflections(base));
            }
            renderer.begin(1);
            for id in base..base + voices {
                let a = id as f32 * 0.4;
                renderer.push(Voice {
                    id,
                    gain: 0.3,
                    pan: [0.5, 0.5],
                    direction: Some([a.sin(), 0.0, -a.cos()]),
                    distance: 4.0 + (id % 32) as f32,
                    step: 1.0,
                    source_gain: 0.3,
                    stereo: false,
                    at: None,
                });
            }
            let t = std::time::Instant::now();
            renderer.render(&mut out, 2, &mut |_, step, buf| {
                read_linear(&tone, 0.0, step, true, buf);
                true
            });
            let took = t.elapsed();
            assert!(renderer.source_reverbs.len() <= MAX_SOURCE_REVERBS);
            if call >= 10 {
                total += took;
                longest = longest.max(took);
            }
        }
        (total / 300, longest)
    }

    #[test]
    fn source_reverbs_ring_out_and_are_reused() {
        let mut renderer = renderer_with(&[]);
        let spares = renderer.spare_reverbs.len();
        assert_eq!(spares, MAX_SOURCE_REVERBS);
        // Sources 0..3 in turn get reflections of their own, then none does.
        for first in 0..4u64 {
            let mut r = reflections_from([-1.0, 0.0, 0.0]);
            for id in first..first + 2 {
                let own = reflections_from([1.0, 0.0, 0.0]);
                r.sources.push((
                    id,
                    ReverbIr {
                        fir: own.fir,
                        reverb: own.reverb,
                        eq: own.eq,
                        delay: own.delay,
                    },
                ));
            }
            renderer.set_reflections(r);
            for id in first..first + 2 {
                ear_energy_of(&mut renderer, id);
            }
            assert!(renderer.source_reverbs.len() <= MAX_SOURCE_REVERBS);
        }
        renderer.set_reflections(reflections_from([-1.0, 0.0, 0.0]));
        // Two seconds of silence: every source's reverb rings out (its tail ends) and goes back to the spares.
        for _ in 0..10 {
            tail(&mut renderer, 0);
        }
        assert!(
            renderer.source_reverbs.is_empty(),
            "{}",
            renderer.source_reverbs.len()
        );
        assert_eq!(renderer.spare_reverbs.len(), spares);
    }

    /// Renders a short burst of source `id` straight ahead.
    fn ear_energy_of(renderer: &mut Renderer, id: u64) {
        let tone: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.05).sin() * 0.1).collect();
        let mut out = vec![0.0; 2 * 4800];
        renderer.begin(1);
        renderer.push(Voice {
            id,
            gain: 0.5,
            pan: [0.5, 0.5],
            direction: Some([0.0, 0.0, -1.0]),
            distance: 4.0,
            step: 1.0,
            source_gain: 0.5,
            stereo: false,
            at: None,
        });
        renderer.render(&mut out, 2, &mut |_, step, buf| {
            read_linear(&tone, 0.0, step, true, buf);
            true
        });
    }

    #[test]
    fn a_busy_audio_thread_renders_fewer_voices_binaurally() {
        let mut renderer = renderer_with(&[]);
        let frame = FRAME_SIZE as f64 / 48_000.0;
        // Frames taking 90% of their time: the budget comes down, not below the minimum.
        for _ in 0..400 {
            renderer.measure_load(std::time::Duration::from_secs_f64(frame * 0.9));
        }
        assert_eq!(renderer.voice_limit, MIN_BINAURAL_VOICES);
        // Room again: back up to the configuration's.
        for _ in 0..2000 {
            renderer.measure_load(std::time::Duration::from_secs_f64(frame * 0.1));
        }
        assert_eq!(renderer.voice_limit, renderer.config.max_binaural_voices);
    }

    #[test]
    fn a_way_round_counts_as_a_second_binaural_voice() {
        let hidden: Vec<(u64, SourceAcoustics)> = (0..32)
            .map(|id| {
                (
                    id,
                    SourceAcoustics {
                        occlusion: 0.3,
                        transmission: [0.2; NUM_BANDS],
                        through: [1.0; NUM_BANDS],
                        path: Some(PathAcoustics {
                            direction: [1.0, 0.0, 0.0],
                            gain: 0.7,
                            eq: [0.8, 0.5, 0.3],
                            distance: 12.0,
                        }),
                    },
                )
            })
            .collect();
        let mut renderer = renderer_with(&hidden);
        renderer.begin(1);
        for id in 0..32u64 {
            renderer.push(Voice {
                id,
                gain: 0.5 - id as f32 * 0.01,
                pan: [0.5, 0.5],
                direction: Some([0.0, 0.0, -1.0]),
                distance: 4.0,
                step: 1.0,
                source_gain: 0.5,
                stereo: false,
                at: None,
            });
        }
        renderer.acoustics.update();
        renderer.choose_binaural();
        let chosen: Vec<usize> = (0..32).filter(|&i| renderer.binaural[i]).collect();
        // 32 voices' budget: 16 sources with their ways round, the loudest.
        assert_eq!(chosen, (0..16).collect::<Vec<_>>());
    }

    /// A source `distance` blocks ahead of a listener at the origin looking down -Z, `side` blocks to its right.
    fn placed(id: u64, distance: f32, side: f32) -> Voice {
        let d = (distance * distance + side * side).sqrt();
        Voice {
            id,
            gain: 0.5,
            pan: [0.5, 0.5],
            direction: Some([side / d, 0.0, -distance / d]),
            distance: d,
            step: 1.0,
            source_gain: 0.5,
            stereo: false,
            at: Some([f64::from(side), 0.0, -f64::from(distance)]),
        }
    }

    /// Renders `voices` for `frames` frames of noise (each its own) and returns the output.
    fn render_noise(renderer: &mut Renderer, voices: &[Voice], frames: usize) -> Vec<f32> {
        let mut out = vec![0.0; 2 * frames * FRAME_SIZE];
        let mut seeds: HashMap<u64, u32> = voices
            .iter()
            .map(|v| (v.id, 0x9e37_79b9 ^ v.id as u32))
            .collect();
        renderer.begin(1);
        for v in voices {
            renderer.push(*v);
        }
        renderer.render(&mut out, 2, &mut |id, _, buf| {
            let seed = seeds.get_mut(&id).expect("a pushed source");
            for s in buf.iter_mut() {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 17;
                *seed ^= *seed << 5;
                *s = *seed as f32 / u32::MAX as f32 - 0.5;
            }
            true
        });
        out
    }

    #[test]
    fn sources_near_one_another_are_one_binaural_voice() {
        let mut renderer =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        // A waterfall 12 blocks ahead: splashes within a block of one another; another 8 blocks to the right of it;
        // and two sounds at the listener's feet, half a block apart (too near the ears to go together).
        let mut voices: Vec<Voice> = (0..10).map(|i| placed(i, 12.0, i as f32 * 0.1)).collect();
        voices.push(placed(20, 12.0, 8.0));
        voices.push(placed(30, 1.0, 0.0));
        voices.push(placed(31, 1.0, 0.5));
        render_noise(&mut renderer, &voices, 2);
        assert_eq!(renderer.states.len(), 4, "{:?}", renderer.states.keys());
        for key in [0, 20, 30, 31] {
            assert!(renderer.states.contains_key(&key), "group {key}");
        }
        assert_eq!(renderer.order.len(), voices.len());

        // The source that started the waterfall's group stops: the group goes on as it was.
        render_noise(&mut renderer, &voices[1..], 2);
        assert!(renderer.states.contains_key(&0));
        assert!(!renderer.states.contains_key(&1));

        // A new splash there joins it; one that has played on its own does not.
        let mut more = voices[1..].to_vec();
        more.push(placed(40, 12.0, 0.5));
        render_noise(&mut renderer, &more, 1);
        assert_eq!(renderer.memberships.get(&40), Some(&0));
        assert_eq!(renderer.memberships.get(&20), Some(&20));
        more.iter_mut().find(|v| v.id == 20).expect("source 20").at = Some([0.4, 0.0, -12.0]);
        render_noise(&mut renderer, &more, 1);
        assert_eq!(renderer.memberships.get(&20), Some(&20));

        // A splash carried away from its group (farther than it may stay) gets a voice of its own.
        more.iter_mut().find(|v| v.id == 40).expect("source 40").at = Some([7.0, 0.0, -12.0]);
        render_noise(&mut renderer, &more, 1);
        assert_eq!(renderer.memberships.get(&40), Some(&40));
    }

    #[test]
    fn a_group_of_one_sounds_as_its_source_alone() {
        let voice = placed(5, 6.0, 2.0);
        let mut alone = Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        let mut grouped =
            Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
        let a = render_noise(&mut alone, &[Voice { at: None, ..voice }], 4);
        let b = render_noise(&mut grouped, &[voice], 4);
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-5, "{worst}");
    }

    #[test]
    #[ignore = "measurement"]
    fn measure_waterfall_load() {
        // 24 splashes of a waterfall 10 blocks ahead, within two blocks: apart, then as the renderer groups them.
        let splashes: Vec<Voice> = (0..24)
            .map(|i| placed(i, 10.0 + (i % 4) as f32 * 0.3, (i / 4) as f32 * 0.3))
            .collect();
        for grouping in [false, true] {
            let voices: Vec<Voice> = splashes
                .iter()
                .map(|v| Voice {
                    at: if grouping { v.at } else { None },
                    ..*v
                })
                .collect();
            let mut renderer =
                Renderer::new(48_000, Config::default()).expect("built-in HRTF at 48 kHz");
            render_noise(&mut renderer, &voices, 10);
            let start = std::time::Instant::now();
            render_noise(&mut renderer, &voices, 200);
            let per_frame = start.elapsed() / 200;
            let frame = std::time::Duration::from_secs_f64(FRAME_SIZE as f64 / 48_000.0);
            println!(
                "grouping {grouping}: {per_frame:?} per frame, {:.1}% of a core, {} binaural voices",
                per_frame.as_secs_f64() / frame.as_secs_f64() * 100.0,
                renderer.states.len()
            );
        }
    }

    #[test]
    #[ignore = "measurement"]
    fn measure_busy_audio_load() {
        for (voices, own) in [(16, 0), (32, 0), (32, 2)] {
            let (mean, longest) = busy_audio_load(voices, own);
            println!(
                "{voices} hidden sources with paths, {own} own reverbs: mean {mean:?}, longest {longest:?} per 10 ms callback"
            );
        }
    }

    #[test]
    #[ignore = "measurement"]
    fn measure_audio_load() {
        for own in [0, 1, 2] {
            let load = audio_load(own);
            println!(
                "32 voices, {own} sources with their own reflections: {:.1}% of a core",
                load * 100.0
            );
        }
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
            through: [1.0; NUM_BANDS],
            path: None,
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
    fn under_water_the_world_is_muffled_unless_turned_off() {
        let mut dry = renderer_with(&[]);
        let under = |config: Config| {
            let mut renderer = renderer_with(&[]);
            renderer.config = config;
            renderer.config_version = Config::version();
            let mut snapshot = Snapshot::fixed(HashMap::new());
            snapshot.underwater = true;
            renderer.set_acoustics(snapshot);
            renderer
        };
        let mut wet = under(Config::default());
        let mut off = under(Config {
            underwater: false,
            ..Config::default()
        });
        let [l0, r0] = ear_energy(&mut dry, Some([1.0, 0.0, 0.0]));
        let [l1, r1] = ear_energy(&mut wet, Some([1.0, 0.0, 0.0]));
        let [l2, r2] = ear_energy(&mut off, Some([1.0, 0.0, 0.0]));
        let ratio = (l1 + r1) / (l0 + r0);
        assert!(
            ratio < 0.5 && ratio > 0.01,
            "under water: {ratio} of the energy"
        );
        assert!(r1 > l1, "still on the right: left {l1}, right {r1}");
        assert!(((l2 + r2) / (l0 + r0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn a_hidden_source_comes_round_from_where_its_path_arrives() {
        // The source is to the right behind a wall; the doorway is to the left (the listener looks north, -z).
        let walled = SourceAcoustics {
            occlusion: 0.0,
            transmission: [0.23, 0.11, 0.09],
            through: [1.0; NUM_BANDS],
            path: Some(PathAcoustics {
                direction: [-1.0, 0.0, 0.0],
                gain: 0.8,
                eq: [0.9, 0.6, 0.4],
                distance: 9.0,
            }),
        };
        let mut open = renderer_with(&[]);
        let mut round = renderer_with(&[(7, walled)]);
        let mut through = renderer_with(&[(7, walled)]);
        through.config.pathing = false;
        through.config_version = Config::version();
        let [l0, r0] = ear_energy(&mut open, Some([1.0, 0.0, 0.0]));
        let [l1, r1] = ear_energy(&mut round, Some([1.0, 0.0, 0.0]));
        let [l2, r2] = ear_energy(&mut through, Some([1.0, 0.0, 0.0]));
        assert!(round.states[&7].path.is_some(), "a path voice");
        assert!(through.states[&7].path.is_none());
        assert!(
            l1 > 2.0 * r1,
            "from the doorway, on the left: left {l1}, right {r1}"
        );
        let (open, round, through) = (l0 + r0, l1 + r1, l2 + r2);
        assert!(
            round > 3.0 * through,
            "round {round}, through the wall {through}"
        );
        assert!(round < open, "round {round}, open {open}");
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
