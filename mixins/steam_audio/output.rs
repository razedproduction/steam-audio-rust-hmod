// The steam_audio mod's support module. horizon-mixins emits it as `crate::audio::steam_audio`, a child of Horizon's
// `audio` module, so it reaches the mixer's private state (`Mixer`, `Channel`, `mix`, `stereo_gains`...).
//
// Horizon plays sound through one cpal output stream whose callback calls `audio::mix`, which pans every source with
// OpenAL Soft's stereo gains. `mix` can't be hooked (horizon-mixins refuses its `cfg_attr`, and its caller is
// `cfg`-gated), so the mod takes the output over instead: the first time the camera moves (`Audio::set_listener`)
// it opens its own stream on the same device and pauses Horizon's (paused, not closed: Horizon still sees an open
// output and keeps following device changes). The mod's callback locks the same mixer and renders it with Steam
// Audio (`render`), or calls Horizon's own `mix` while the HRTF loads, on a sampling rate it has no measurements
// for, on a mono output and in splitscreen.
//
// Horizon's `mix` fills `Mixer::cameras` and the mod's output leaves it empty, so a filled one means Horizon's
// stream mixed again: it was reopened (another device, a lost one). The mod then reopens its own on Horizon's
// device and pauses the new one.
//
// The world's acoustics (occlusion, transmission, paths around walls, reverb) need the blocks, which live on the game thread: the mod's
// `frame` hook in horizon-client calls `update_acoustics` with the world, and the output applies what it publishes.
// The module is `pub` for that hook, and for the settings page's hooks (`mods`, `settings`, `settings_click`), which
// hand over to `steam_audio_hmod::settings`.

use super::{Audio, Mixer, State, mix, stereo_gains};
use steam_audio_hmod::horizon::{Voice, audible_range, distance_gain, read_linear, with_renderer};

/// Takes Horizon's sound output over, once per output Horizon opens. Called on the game thread at every camera
/// update; without the `audio` feature there is no output and it does nothing.
pub(crate) fn take_over(audio: &mut Audio) {
    #[cfg(feature = "audio")]
    output::take_over(audio);
    #[cfg(not(feature = "audio"))]
    let _ = audio;
}

/// The mod's id: its row in Options > Mods and its settings folder.
pub const MOD_ID: &str = steam_audio_hmod::settings::MOD_ID;

/// The mod's row in Options > Mods: id, title and description in the game's `language` (`ru_ru`...). For the `mods`
/// hook.
pub fn mod_row(language: &str) -> (String, String, String) {
    use steam_audio_hmod::settings;
    (
        MOD_ID.to_owned(),
        settings::title(),
        settings::description(language),
    )
}

/// The labels of the mod's settings page, one button per setting with its value. For the `settings` hook.
pub fn settings_page(language: &str, buttons: &mut Vec<String>) {
    let config = steam_audio_hmod::horizon::Config::get();
    buttons.extend(steam_audio_hmod::settings::labels(&config, language));
}

/// A click on button `button` of the page: the setting moves to its next value, at once, and is kept through
/// `store` (Horizon's `native_set_setting`). For the `settings_click` hook.
pub fn settings_click(button: usize, store: &mut dyn FnMut(&str, &str)) {
    steam_audio_hmod::settings::click_and_store(button, store);
}

/// Reads the kept settings through `read` (Horizon's `native_setting`), once per game. For the `frame` hook.
pub fn load_settings(read: &dyn Fn(&str) -> Option<String>) {
    steam_audio_hmod::settings::load(read);
}

/// The world's acoustics for the sources playing now, from the blocks around the camera: how much each source is
/// occluded, what its walls let through and the way round them, and the blocks the reverb and the paths are
/// simulated in. Called by the mod's `frame` hook in horizon-client, once a frame on the game thread (where the world
/// is). The sources are looked at 20 times a second at most, and their results published for the output to apply;
/// the blocks around the camera are copied a slice per frame and handed to the reflection simulation's thread and to
/// the pathing bake's.
pub fn update_acoustics(
    audio: &Audio,
    world: &crate::world::World,
    facts: &horizon_items::blocks::BlockFacts,
) {
    use std::cell::RefCell;
    use std::time::{Duration, Instant};
    use std::sync::Arc;
    use steam_audio_hmod::acoustics::{self, Surface};
    use steam_audio_hmod::paths;
    use steam_audio_hmod::reflections::{self, GridBuilder};

    // A source's distance attenuation (Horizon's, with its own range), for the longer way round a wall.
    type Attenuation = Box<dyn Fn(f32) -> f32>;

    // Sources are looked at this often.
    const PERIOD: Duration = Duration::from_millis(50);
    // At most this many sources, the nearest, and none farther than this.
    const MAX_SOURCES: usize = 32;
    const MAX_DISTANCE: f64 = 64.0;
    // Sources this near get reflections of their own (the nearest `reflections::MAX_SOURCES`), once they have played
    // this long: a footstep or a hit is over before its reflections could be simulated, and new sounds coming all the
    // time would keep the simulation from ever settling.
    const OWN_REFLECTIONS_DISTANCE: f64 = 32.0;
    const OWN_REFLECTIONS_AGE: Duration = Duration::from_secs(1);
    // Blocks copied for the reverb and the paths per frame (a whole grid is 589 824), in slices, for at most
    // `GRID_TIME`: a grid around a new place in 24 frames or more. Once the grid around the camera is there, it is
    // copied again only to see blocks change, `GRID_REFRESH` a frame (about 1.5 s for all of it at 100 fps).
    const GRID_BUDGET: usize = 24_576;
    const GRID_SLICE: usize = 2_048;
    const GRID_TIME: Duration = Duration::from_micros(1_500);
    const GRID_REFRESH: usize = 4_096;

    // What a block state is to the sound: its wall surface, if any; its shape when it is a wall but not a full cube
    // (a slab, a door: rays pass beside it), and the axes along which that shape closes its cell; whether it is a full
    // wall; whether it holds water.
    #[derive(Clone)]
    struct Block {
        surface: Option<Surface>,
        shape: Option<acoustics::Shape>,
        seals: u8,
        full: bool,
        water: bool,
    }

    // The game thread's state: when it last looked, what it found then (the settings, the camera), what each block
    // state is (`None` until asked) for the block facts it was made from, and the grid being copied.
    struct Cache {
        last: Option<Instant>,
        occlusion: bool,
        pathing: bool,
        reverb: bool,
        underwater: bool,
        listener: Option<[f64; 3]>,
        facts: usize,
        blocks: Vec<Option<Block>>,
        sources: Vec<(u64, [f64; 3], Attenuation)>,
        results: Vec<(u64, acoustics::SourceAcoustics)>,
        // Since when each source has been playing (by mixer channel id).
        playing_since: Vec<(u64, Instant)>,
        grid: GridBuilder,
        // Where the last complete grid was (its corner).
        grid_origin: Option<[i32; 3]>,
    }
    thread_local! {
        static CACHE: RefCell<Cache> = const {
            RefCell::new(Cache {
                last: None,
                occlusion: false,
                pathing: false,
                reverb: false,
                underwater: false,
                listener: None,
                facts: 0,
                blocks: Vec::new(),
                sources: Vec::new(),
                results: Vec::new(),
                playing_since: Vec::new(),
                grid: GridBuilder::new(),
                grid_origin: None,
            })
        };
    }

    if !steam_audio_hmod::horizon::Config::enabled() || audio.backend.owner != 0 {
        return;
    }
    CACHE.with_borrow_mut(|state| {
        let due = state.last.is_none_or(|t| t.elapsed() >= PERIOD);
        let mut settings = None;
        if due {
            state.last = Some(Instant::now());
            let config = steam_audio_hmod::horizon::Config::get();
            state.occlusion = config.occlusion;
            state.pathing = config.occlusion && config.pathing;
            state.reverb = config.reverb;
            state.underwater = config.underwater;
            settings = Some(config.acoustics());
            state.sources.clear();
            // The camera and the sources it can hear, the mixer locked only to copy them.
            if config.occlusion || config.reverb || config.underwater {
                let Ok(m) = audio.backend.mixer.lock() else {
                    return;
                };
                let here = m.worlds.get(&0).copied().unwrap_or_default();
                let listener = m.listener;
                state.listener = Some(listener);
                if config.occlusion || config.reverb {
                    for (&id, c) in &m.channels {
                        if c.state != State::Playing || c.relative || c.world != here {
                            continue;
                        }
                        let at = [f64::from(c.at[0]), f64::from(c.at[1]), f64::from(c.at[2])];
                        let d2 = (0..3).map(|i| (at[i] - listener[i]).powi(2)).sum::<f64>();
                        let range = audible_range(c.linear, config.physical_distance)
                            .map_or(MAX_DISTANCE, |max| f64::from(max).min(MAX_DISTANCE));
                        if d2 <= range * range {
                            let (linear, physical) = (c.linear, config.physical_distance);
                            let attenuation: Attenuation =
                                Box::new(move |d| distance_gain(linear, d, physical));
                            state.sources.push((id, at, attenuation));
                        }
                    }
                }
            }
            let now = Instant::now();
            let sources = &state.sources;
            state
                .playing_since
                .retain(|(id, _)| sources.iter().any(|s| s.0 == *id));
            for (id, _, _) in sources {
                if !state.playing_since.iter().any(|p| p.0 == *id) {
                    state.playing_since.push((*id, now));
                }
            }
        }
        let Some(listener) = state.listener else {
            return;
        };

        // Walls: full collision cubes, of the surface their block and sound type say; and what is not a full cube
        // but counts (liquids, slabs and stairs, doors, leaves, fences and panes: `Surface::classify_partial`), with
        // the shape of its state.
        let key = facts as *const horizon_items::blocks::BlockFacts as usize;
        if state.facts != key || state.blocks.len() != facts.states.len() {
            state.facts = key;
            state.blocks.clear();
            state.blocks.resize(facts.states.len(), None);
        }
        let blocks = RefCell::new(std::mem::take(&mut state.blocks));
        let block = |c: [i32; 3]| -> Option<Block> {
            let id = world.block(c[0], c[1], c[2]);
            let mut blocks = blocks.borrow_mut();
            let slot = blocks.get_mut(id.0 as usize)?;
            if slot.is_none() {
                let f = facts.get(id);
                let water = f.fluid == horizon_items::blocks::Fluid::Water;
                let surface = if f.full_collision {
                    Some(Surface::classify(&f.name, &f.sounds.break_sound))
                } else {
                    // Kelp, seagrass and the like are water to the sound.
                    Surface::classify_partial(&f.name, &f.sounds.break_sound)
                        .or(water.then_some(Surface::Water))
                };
                let cube = f.outline.len() == 1 && f.outline[0] == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
                let shape = (!f.full_collision
                    && surface.is_some_and(|s| s.is_solid() && s != Surface::Water)
                    && !f.outline.is_empty()
                    && !cube)
                    .then(|| f.outline.clone());
                *slot = Some(Block {
                    surface,
                    seals: shape.as_deref().map_or(0, acoustics::shape_seals),
                    full: surface.is_some_and(Surface::is_solid) && shape.is_none(),
                    shape,
                    water,
                });
            }
            slot.clone()
        };
        // The head under water: the world is muffled, and the water around is the medium, not walls.
        let underwater = state.underwater
            && block(listener.map(|v| v.floor() as i32)).is_some_and(|b| b.water);
        acoustics::publish_underwater(underwater);
        let medium = |b: &Block| underwater && b.surface == Some(Surface::Water);
        // For the rays: the surface and the shape.
        let mut wall = |c: [i32; 3]| -> Option<(Surface, Option<acoustics::Shape>)> {
            let b = block(c)?;
            if medium(&b) {
                return None;
            }
            Some((b.surface?, b.shape))
        };
        // For the reflection and pathing grids, whole cells: a shaped block counts if it closes the cell between
        // two open ones (a closed door, a slab floor over a void), not otherwise (an open door, a slab on the ground).
        let mut cell = |c: [i32; 3]| -> Option<Surface> {
            let b = block(c)?;
            if medium(&b) {
                return None;
            }
            let surface = b.surface?;
            if b.shape.is_some()
                && !acoustics::shape_blocks_cell(b.seals, |o| {
                    block([c[0] + o[0], c[1] + o[1], c[2] + o[2]]).is_some_and(|n| n.full && !medium(&n))
                })
            {
                return None;
            }
            Some(surface)
        };
        if !(state.occlusion || state.reverb) {
            state.blocks = blocks.into_inner();
            return;
        }
        let d2 = |at: [f64; 3]| (0..3).map(|i| (at[i] - listener[i]).powi(2)).sum::<f64>();
        if state.sources.len() > MAX_SOURCES {
            state.sources.sort_by(|a, b| d2(a.1).total_cmp(&d2(b.1)));
            state.sources.truncate(MAX_SOURCES);
        }

        if let Some(settings) = settings {
            state.results.clear();
            // The nearest sources get reflections simulated from where they are.
            if state.reverb {
                let mut nearest: Vec<(u64, [f64; 3])> = state
                    .sources
                    .iter()
                    .filter(|(id, at, _)| {
                        d2(*at) <= OWN_REFLECTIONS_DISTANCE.powi(2)
                            && state
                                .playing_since
                                .iter()
                                .any(|(p, t)| p == id && t.elapsed() >= OWN_REFLECTIONS_AGE)
                    })
                    .map(|(id, at, _)| (*id, *at))
                    .collect();
                nearest.sort_by(|a, b| d2(a.1).total_cmp(&d2(b.1)).then(a.0.cmp(&b.0)));
                nearest.truncate(reflections::MAX_SOURCES);
                reflections::submit_sources(&nearest);
            }
            // The paths baked in the blocks around the camera, for the sources a wall hides.
            let baked = if state.pathing { paths::latest() } else { None };
            let occluding = if state.occlusion { &state.sources[..] } else { &[] };
            for (id, at, attenuation) in occluding {
                let mut result =
                    acoustics::source_acoustics_shaped(&mut wall, listener, *at, &settings);
                if result.occlusion < 1.0
                    && let Some(baked) = &baked
                    && let Some(mut path) = baked.find(listener, *at)
                {
                    // The way round is longer: Horizon's attenuation over its length, relative to the straight one.
                    let straight = attenuation(d2(*at).sqrt() as f32);
                    let round = attenuation(path.distance);
                    path.gain *= if straight > 0.0 {
                        (round / straight).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    result.path = (path.gain > 0.0).then_some(path);
                }
                state.results.push((*id, result));
            }
            acoustics::publish(&state.results);
            if state.reverb {
                reflections::submit_listener(listener);
            }
        }
        // The blocks around the camera, for the reverb and the paths: a slice of the grid each frame, the grid to
        // the simulation and the pathing bake when complete.
        if state.reverb || state.pathing {
            let refresh = state.grid_origin == Some(GridBuilder::origin_for(listener));
            let (budget, start) = (if refresh { GRID_REFRESH } else { GRID_BUDGET }, Instant::now());
            let mut copied = 0;
            while copied < budget && start.elapsed() < GRID_TIME {
                let slice = GRID_SLICE.min(budget - copied);
                copied += slice;
                if let Some(grid) = state.grid.step(listener, slice, &mut cell) {
                    state.grid_origin = Some(grid.origin());
                    let grid = Arc::new(grid);
                    if state.reverb {
                        reflections::submit_grid(&grid);
                    }
                    if state.pathing {
                        paths::submit_grid(&grid);
                    }
                    break;
                }
            }
        }
        state.blocks = blocks.into_inner();
    });
}

/// Mixes one output buffer (interleaved, `channels` per frame): with Steam Audio when it can, else as Horizon does.
#[cfg_attr(not(feature = "audio"), allow(dead_code))]
pub(super) fn render(mixer: &mut Mixer, data: &mut [f32], channels: usize, rate: u32) {
    if !render_binaural(mixer, data, channels, rate) {
        mix(mixer, data, channels, rate);
    }
    // Only Horizon's stream may leave `cameras` filled (see the top of this module).
    mixer.cameras.clear();
}

/// The vanilla mixer's geometry (camera basis, world check, distance gain) with the samples rendered by Steam Audio.
/// Returns false, having done nothing, when Horizon should mix.
#[cfg_attr(not(feature = "audio"), allow(dead_code))]
fn render_binaural(mixer: &mut Mixer, data: &mut [f32], channels: usize, rate: u32) -> bool {
    // Mono output: nothing to spatialize. Splitscreen: the speakers are nobody's ears.
    if channels < 2 || !mixer.listeners.is_empty() {
        return false;
    }
    let owner = &*mixer as *const Mixer as usize;
    let rendered = with_renderer(rate, |renderer| {
        // The camera's forward and up vectors and OpenAL's right = forward x up, as Horizon's mixer has them.
        let (yaw, pitch) = (
            f64::from(mixer.listener_yaw.to_radians()),
            f64::from(mixer.listener_pitch.to_radians()),
        );
        let forward = [
            -yaw.sin() * pitch.cos(),
            -pitch.sin(),
            yaw.cos() * pitch.cos(),
        ];
        let up = [
            -yaw.sin() * pitch.sin(),
            pitch.cos(),
            yaw.cos() * pitch.sin(),
        ];
        let right = [
            forward[1] * up[2] - forward[2] * up[1],
            forward[2] * up[0] - forward[0] * up[2],
            forward[0] * up[1] - forward[1] * up[0],
        ];
        let dot = |a: [f64; 3], b: [f64; 3]| (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) as f32;
        let listener = mixer.listener;
        let world = mixer.worlds.get(&0).copied().unwrap_or_default();
        renderer.begin(owner);
        let physical = renderer.config().physical_distance;
        renderer.set_orientation(forward.map(|v| v as f32), up.map(|v| v as f32));
        for (&id, c) in &mixer.channels {
            let Some((_, source_rate)) = &c.samples else {
                continue;
            };
            if c.state != State::Playing {
                continue;
            }
            let at = [f64::from(c.at[0]), f64::from(c.at[1]), f64::from(c.at[2])];
            let d = if c.relative {
                at
            } else {
                [
                    at[0] - listener[0],
                    at[1] - listener[1],
                    at[2] - listener[2],
                ]
            };
            let distance = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() as f32;
            // A source in another world is unheard but still advances, as in Horizon's mixer.
            let gain = if c.relative || c.world == world {
                c.gain.clamp(0.0, 1.0) * distance_gain(c.linear, distance, physical)
            } else {
                0.0
            };
            // Listener space: right, up, ahead (relative sources are already in it, with -Z ahead).
            let (x, y, ahead) = if c.relative {
                (d[0] as f32, d[1] as f32, -d[2] as f32)
            } else {
                (dot(d, right), dot(d, up), dot(d, forward))
            };
            renderer.push(Voice {
                id,
                gain,
                pan: stereo_gains(x, y, ahead),
                // Steam Audio's listener space has -Z ahead. A source at the listener (music, the UI) has no
                // direction and is panned as Horizon pans it.
                direction: (distance > 0.05).then_some([x, y, -ahead]),
                distance,
                step: f64::from(*source_rate) / f64::from(rate) * f64::from(c.pitch.max(0.0)),
                source_gain: if c.relative || c.world == world {
                    c.gain.clamp(0.0, 1.0)
                } else {
                    0.0
                },
            });
        }
        let sources = &mut mixer.channels;
        renderer.render(data, channels, &mut |id, step, out| {
            let Some(c) = sources.get_mut(&id) else {
                out.fill(0.0);
                return false;
            };
            let Some((samples, _)) = &c.samples else {
                out.fill(0.0);
                return false;
            };
            let (position, playing) = read_linear(samples, c.position, step, c.looping, out);
            c.position = position;
            if !playing {
                c.state = State::Stopped;
            }
            playing
        });
    });
    if rendered.is_none() {
        return false;
    }
    // Horizon's output limiter: the whole buffer together, released over 200 ms.
    let peak = data.iter().fold(1.0f32, |peak, s| peak.max(s.abs()));
    mixer.limiter_peak = mixer.limiter_peak.max(peak);
    let release = (-1.0 / (0.2 * rate as f32)).exp();
    for frame in data.chunks_mut(channels) {
        let gain = mixer.limiter_peak.recip();
        for s in frame {
            *s *= gain;
        }
        mixer.limiter_peak = (mixer.limiter_peak * release).max(peak);
    }
    true
}

#[cfg(feature = "audio")]
mod output {
    use super::super::{Audio, Mixer};
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex, Weak};
    use std::time::{Duration, Instant};
    use steam_audio_hmod::horizon::Config;

    /// How long after a takeover a filled `cameras` is put down to Horizon's last callbacks, not a new stream.
    const SETTLE: Duration = Duration::from_millis(500);

    /// The output the mod took over: Horizon's mixer, the mod's stream on it (`None`: it could not take this one
    /// over and leaves it to Horizon) and when.
    struct Taken {
        mixer: Weak<Mutex<Mixer>>,
        stream: Option<cpal::Stream>,
        at: Instant,
    }

    thread_local! {
        // cpal streams are not `Send` on every platform: this lives on the game thread, where the hook runs.
        static TAKEN: RefCell<Option<Taken>> = const { RefCell::new(None) };
    }

    pub(super) fn take_over(audio: &mut Audio) {
        // Splitscreen players share the first player's output; no output (none opened, or turned off): nothing.
        if audio.backend.owner != 0 || audio._stream.is_none() || !Config::enabled() {
            return;
        }
        let mixer = audio.backend.mixer.clone();
        let horizon_mixed = match mixer.lock() {
            Ok(mut m) => {
                let mixed = !m.cameras.is_empty();
                m.cameras.clear();
                mixed
            }
            Err(_) => return,
        };
        TAKEN.with_borrow_mut(|taken| {
            if let Some(t) = taken
                && std::ptr::eq(t.mixer.as_ptr(), Arc::as_ptr(&mixer))
                && (t.stream.is_none() || !horizon_mixed || t.at.elapsed() < SETTLE)
            {
                return;
            }
            // A first output, another mixer or Horizon's stream reopened: the mod's old stream goes first.
            *taken = None;
            let stream = match open(&audio.device_name, &mixer) {
                Ok(stream) => stream,
                Err(e) => {
                    log::warn!(
                        "steam_audio: could not open an output ({e}); Horizon keeps its own"
                    );
                    *taken = Some(Taken {
                        mixer: Arc::downgrade(&mixer),
                        stream: None,
                        at: Instant::now(),
                    });
                    return;
                }
            };
            let paused = match &audio._stream {
                Some(horizon) => horizon.pause().map_err(|e| e.to_string()),
                None => Ok(()),
            };
            let stream = match paused {
                Ok(()) => Some(stream),
                Err(e) => {
                    log::warn!(
                        "steam_audio: could not pause Horizon's output ({e}); Horizon keeps it"
                    );
                    None
                }
            };
            *taken = Some(Taken {
                mixer: Arc::downgrade(&mixer),
                stream,
                at: Instant::now(),
            });
        });
    }

    /// A stream on the device Horizon plays on: the one named (`Audio::device_name`) when it is there, else the
    /// default, in its default configuration, rendering `mixer`.
    fn open(name: &str, mixer: &Arc<Mutex<Mixer>>) -> Result<cpal::Stream, String> {
        use cpal::SampleFormat as F;
        let host = cpal::default_host();
        // Listing every device probes each one (slow on Linux), so only for a named device, as Horizon does.
        let named = (!name.is_empty())
            .then(|| host.output_devices().ok())
            .flatten()
            .and_then(|mut devices| devices.find(|d| d.name().is_ok_and(|n| n == name)));
        let device = named
            .or_else(|| host.default_output_device())
            .ok_or("no output device")?;
        let config = device.default_output_config().map_err(|e| e.to_string())?;
        let weak = Arc::downgrade(mixer);
        let stream = match config.sample_format() {
            F::F32 => build::<f32>(&device, &config, weak),
            F::F64 => build::<f64>(&device, &config, weak),
            F::I16 => build::<i16>(&device, &config, weak),
            F::I32 => build::<i32>(&device, &config, weak),
            F::I8 => build::<i8>(&device, &config, weak),
            F::U16 => build::<u16>(&device, &config, weak),
            F::U32 => build::<u32>(&device, &config, weak),
            F::U8 => build::<u8>(&device, &config, weak),
            other => Err(format!("unsupported sample format {other}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        let device_name = device.name().unwrap_or_default();
        log::info!(
            "steam_audio: output {device_name} ({} Hz, {} channels, {}) rendered with Steam Audio",
            config.sample_rate().0,
            config.channels(),
            config.sample_format()
        );
        if let Ok(mut m) = mixer.lock() {
            m.device_name = device_name;
            m.device_lost = false;
        }
        Ok(stream)
    }

    /// The output callback: Horizon's own (`build_stream`), rendering through `super::render`.
    fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
        mixer: Weak<Mutex<Mixer>>,
    ) -> Result<cpal::Stream, String> {
        let rate = config.sample_rate().0;
        let channels = usize::from(config.channels());
        let on_error = mixer.clone();
        let mut scratch = Vec::new();
        device
            .build_output_stream(
                &config.config(),
                move |data: &mut [T], _| {
                    scratch.resize(data.len(), 0.0);
                    match mixer.upgrade() {
                        Some(m) => match m.lock() {
                            Ok(mut m) => super::render(&mut m, &mut scratch, channels, rate),
                            Err(_) => scratch.fill(0.0),
                        },
                        // Horizon's audio is gone: silence until the stream goes too.
                        None => scratch.fill(0.0),
                    }
                    for (out, s) in data.iter_mut().zip(&scratch) {
                        *out = T::from_sample(*s);
                    }
                },
                move |e| {
                    log::warn!("steam_audio: output stream error: {e}");
                    // Horizon reopens a lost device (its next stream mixes, and the mod takes it over again).
                    if matches!(e, cpal::StreamError::DeviceNotAvailable)
                        && let Some(m) = on_error.upgrade()
                        && let Ok(mut m) = m.lock()
                    {
                        m.device_lost = true;
                    }
                },
                None,
            )
            .map_err(|e| e.to_string())
    }
}
