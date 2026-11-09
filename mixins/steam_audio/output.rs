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
use steam_audio_hmod::horizon::{Voice, audible_range, distance_gain, read_frames, with_renderer};

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
    use steam_audio_hmod::debug;
    use steam_audio_hmod::paths;
    use steam_audio_hmod::reflections::{self, GridBuilder};

    // A source's distance attenuation (Horizon's, with its own range), for the longer way round a wall.
    type Attenuation = Box<dyn Fn(f32) -> f32>;

    // Sources are looked at this often.
    const PERIOD: Duration = Duration::from_millis(50);
    // At most this many sources, the nearest, and none farther than this.
    const MAX_SOURCES: usize = 32;
    const MAX_DISTANCE: f64 = 64.0;
    // Sources whose occlusion and paths are worked out per frame (a round of 32 takes 4 frames).
    const SOURCES_PER_FRAME: usize = 8;
    // Sources this near get reflections of their own (the nearest `reflections::MAX_SOURCES`), once they have played
    // this long: a footstep or a hit is over before its reflections could be simulated, and new sounds coming all the
    // time would keep the simulation from ever settling.
    const OWN_REFLECTIONS_DISTANCE: f64 = 32.0;
    const OWN_REFLECTIONS_AGE: Duration = Duration::from_secs(1);
    // Blocks copied for the reverb and the paths per frame (a whole grid is 589 824), in slices, for at most
    // `GRID_TIME`: a grid around a new place in 24 frames or more. Once the grid around the camera is there, it is
    // copied again only to see blocks change, `GRID_REFRESH` a frame (about 3 s for all of it at 100 fps).
    const GRID_BUDGET: usize = 24_576;
    const GRID_SLICE: usize = 2_048;
    const GRID_TIME: Duration = Duration::from_micros(1_500);
    const GRID_REFRESH: usize = 2_048;

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
        // The round of sources being worked out (its settings) and the next source in it.
        round: Option<acoustics::AcousticsSettings>,
        next_source: usize,
        // Since when each source has been playing (by mixer channel id).
        playing_since: Vec<(u64, Instant)>,
        grid: GridBuilder,
        // Where the last complete grid was (its corner).
        grid_origin: Option<[i32; 3]>,
        // The visual debug mode: its layers this round, what the round found for them, the sources given reflections
        // of their own, the last complete grid, and whether a scene was published (to forget it once they are off).
        layers: debug::Layers,
        debug_round: Vec<debug::DebugSource>,
        own_reflections: Vec<u64>,
        last_grid: Option<Arc<reflections::Voxels>>,
        debugging: bool,
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
                round: None,
                next_source: 0,
                playing_since: Vec::new(),
                grid: GridBuilder::new(),
                grid_origin: None,
                layers: debug::Layers {
                    sources: false,
                    paths: false,
                    reflections: false,
                    stats: false,
                },
                debug_round: Vec::new(),
                own_reflections: Vec::new(),
                last_grid: None,
                debugging: false,
            })
        };
    }

    // Forgets the published scene, so that its labels and lines (of a world left, or of a mod turned off) are not
    // drawn again before the next complete round.
    fn forget_debug(state: &mut Cache) {
        if state.debugging {
            state.debugging = false;
            state.debug_round.clear();
            debug::clear_scene();
        }
    }

    if !steam_audio_hmod::horizon::Config::enabled() || audio.backend.owner != 0 {
        CACHE.with_borrow_mut(forget_debug);
        return;
    }
    CACHE.with_borrow_mut(|state| {
        // A new look once the last round of sources is complete (at a low frame rate, a little later than `PERIOD`).
        let due = state.round.is_none() && state.last.is_none_or(|t| t.elapsed() >= PERIOD);
        let mut settings = None;
        if due {
            state.last = Some(Instant::now());
            let config = steam_audio_hmod::horizon::Config::get();
            state.occlusion = config.occlusion;
            state.pathing = config.occlusion && config.pathing;
            state.reverb = config.reverb;
            state.underwater = config.underwater;
            state.layers = debug::Layers::of(&config);
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
                        let range = audible_range(c.linear, config.reach())
                            .map_or(MAX_DISTANCE, |max| f64::from(max).min(MAX_DISTANCE));
                        if d2 <= range * range {
                            let (linear, reach) = (c.linear, config.reach());
                            let attenuation: Attenuation =
                                Box::new(move |d| distance_gain(linear, d, reach));
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
        // No camera yet, or the debug layers turned off: what they showed goes.
        let Some(listener) = state.listener else {
            forget_debug(state);
            return;
        };
        if !state.layers.any() {
            forget_debug(state);
        }

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
            if state.debugging {
                state.debugging = false;
                debug::clear_scene();
            }
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
                state.own_reflections.clear();
                state.own_reflections.extend(nearest.iter().map(|(id, _)| *id));
            } else {
                state.own_reflections.clear();
            }
            state.round = Some(settings);
            state.next_source = 0;
            state.debug_round.clear();
        }
        // The sources' occlusion and paths, `SOURCES_PER_FRAME` a frame: a round over the frames of a period, not
        // all in one frame. Published when the round is complete.
        if let Some(settings) = state.round {
            // The paths baked in the blocks around the camera, for the sources a wall hides.
            let baked = if state.pathing { paths::latest() } else { None };
            let occluding = if state.occlusion { &state.sources[..] } else { &[] };
            let from = state.next_source.min(occluding.len());
            let to = (from + SOURCES_PER_FRAME).min(occluding.len());
            state.next_source = to;
            let layers = state.layers;
            for (id, at, attenuation) in &occluding[from..to] {
                // With the debug layers on, what the rays and the paths went through too.
                let mut trace = debug::SourceTrace::default();
                let mut legs = Vec::new();
                let mut result = acoustics::source_acoustics_traced(
                    &mut wall,
                    listener,
                    *at,
                    &settings,
                    layers.sources.then_some(&mut trace),
                );
                if result.occlusion < 1.0
                    && let Some(baked) = &baked
                    && let Some(mut path) =
                        baked.find_traced(listener, *at, layers.paths.then_some(&mut legs))
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
                if layers.any() {
                    if result.path.is_none() {
                        legs.clear();
                    }
                    state.debug_round.push(debug::DebugSource {
                        id: *id,
                        at: *at,
                        acoustics: result,
                        trace,
                        path_legs: legs,
                        own_reflections: false,
                    });
                }
            }
            if to == occluding.len() {
                state.round = None;
                acoustics::publish(&state.results);
                if state.reverb {
                    reflections::submit_listener(listener);
                }
                if layers.any() {
                    let mut sources = std::mem::take(&mut state.debug_round);
                    // Sources not looked at (occlusion off) are still shown where they are.
                    for (id, at, _) in &state.sources {
                        if !sources.iter().any(|s| s.id == *id) {
                            sources.push(debug::DebugSource {
                                id: *id,
                                at: *at,
                                acoustics: acoustics::SourceAcoustics::OPEN,
                                trace: debug::SourceTrace::default(),
                                path_legs: Vec::new(),
                                own_reflections: false,
                            });
                        }
                    }
                    for s in &mut sources {
                        s.own_reflections = state.own_reflections.contains(&s.id);
                    }
                    let grid = state.last_grid.as_deref().filter(|_| layers.reflections);
                    debug::publish_scene(debug::Scene {
                        listener: Some(listener),
                        underwater,
                        sources,
                        probes: match &baked {
                            Some(baked) if layers.paths => baked.probe_positions(),
                            _ => Vec::new(),
                        },
                        grid: grid.map(|g| (g.origin(), g.size())),
                        reflection_rays: grid.map_or_else(Vec::new, |g| {
                            debug::reflection_rays(g, listener, 48, 3)
                        }),
                    });
                    state.debugging = true;
                }
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
                    state.last_grid = Some(Arc::clone(&grid));
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
///
/// Steam Audio's rendering takes milliseconds, more with every source, and the game thread locks the same mixer to
/// play, move and stop each sound. So the mixer is locked only to take what the rendering needs (each source's
/// parameters, its samples by `Arc`, its position) and, after, to put the sources' new positions back; the rendering
/// itself runs unlocked. Horizon's own `mix`, cheap, runs locked as in Horizon.
#[cfg_attr(not(feature = "audio"), allow(dead_code))]
pub(super) fn render(mixer: &std::sync::Mutex<Mixer>, data: &mut [f32], channels: usize, rate: u32) {
    if render_binaural(mixer, data, channels, rate) {
        return;
    }
    match mixer.lock() {
        Ok(mut m) => {
            mix(&mut m, data, channels, rate);
            // Only Horizon's stream may leave `cameras` filled (see the top of this module).
            m.cameras.clear();
        }
        Err(_) => data.fill(0.0),
    }
}

/// A source as the output took it from the mixer: its samples, read without the mixer's lock, where it was then and
/// where the rendering has left it.
#[cfg_attr(not(feature = "audio"), allow(dead_code))]
struct TakenSource {
    id: u64,
    samples: std::sync::Arc<Vec<f32>>,
    /// Its channels, interleaved in `samples` (1, or 2 for music and the like).
    channels: usize,
    looping: bool,
    start: f64,
    position: f64,
    playing: bool,
}

/// The vanilla mixer's geometry (camera basis, world check, distance gain) with the samples rendered by Steam Audio.
/// Returns false, having done nothing, when Horizon should mix.
#[cfg_attr(not(feature = "audio"), allow(dead_code))]
fn render_binaural(
    mixer: &std::sync::Mutex<Mixer>,
    data: &mut [f32],
    channels: usize,
    rate: u32,
) -> bool {
    use std::cell::RefCell;
    thread_local! {
        // The audio thread's sources of the current callback, kept to be refilled without allocating.
        static SOURCES: RefCell<Vec<TakenSource>> = const { RefCell::new(Vec::new()) };
    }
    let owner = mixer as *const std::sync::Mutex<Mixer> as usize;
    let rendered = with_renderer(rate, |renderer| {
        SOURCES.with_borrow_mut(|sources| {
            sources.clear();
            {
                let Ok(m) = mixer.lock() else {
                    return false;
                };
                // Mono output: nothing to spatialize. Splitscreen: the speakers are nobody's ears.
                if channels < 2 || !m.listeners.is_empty() {
                    return false;
                }
                take_voices(&m, renderer, owner, rate, sources);
            }
            sources.sort_unstable_by_key(|s| s.id);
            renderer.render(data, channels, &mut |id, step, out| {
                let Ok(i) = sources.binary_search_by_key(&id, |s| s.id) else {
                    out.fill(0.0);
                    return false;
                };
                let s = &mut sources[i];
                let (position, playing) =
                    read_frames(&s.samples, s.channels, s.position, step, s.looping, out);
                s.position = position;
                s.playing &= playing;
                s.playing
            });
            let Ok(mut m) = mixer.lock() else {
                return true;
            };
            // Where the sources got to, unless the game moved, restarted or stopped one meanwhile.
            for s in sources.iter() {
                if let Some(c) = m.channels.get_mut(&s.id)
                    && c.state == State::Playing
                    && c.position == s.start
                    && c.samples
                        .as_ref()
                        .is_some_and(|(samples, _, _)| std::sync::Arc::ptr_eq(samples, &s.samples))
                {
                    c.position = s.position;
                    if !s.playing {
                        c.state = State::Stopped;
                    }
                }
            }
            // Horizon's output limiter: the whole buffer together, released over 200 ms.
            let peak = data.iter().fold(1.0f32, |peak, s| peak.max(s.abs()));
            m.limiter_peak = m.limiter_peak.max(peak);
            let release = (-1.0 / (0.2 * rate as f32)).exp();
            for frame in data.chunks_mut(channels) {
                let gain = m.limiter_peak.recip();
                for s in frame {
                    *s *= gain;
                }
                m.limiter_peak = (m.limiter_peak * release).max(peak);
            }
            // Only Horizon's stream may leave `cameras` filled (see the top of this module).
            m.cameras.clear();
            sources.clear();
            true
        })
    });
    rendered == Some(true)
}

/// Gives `renderer` this callback's voices from the locked mixer `m` (the camera, each playing source's gain,
/// direction and step), and `sources` their samples.
#[cfg_attr(not(feature = "audio"), allow(dead_code))]
fn take_voices(
    m: &Mixer,
    renderer: &mut steam_audio_hmod::horizon::Renderer,
    owner: usize,
    rate: u32,
    sources: &mut Vec<TakenSource>,
) {
    let mixer = m;
    {
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
        let reach = renderer.config().reach();
        let source_volume = renderer.config().source_volume.clamp(0.0, 2.0);
        renderer.set_orientation(forward.map(|v| v as f32), up.map(|v| v as f32));
        for (&id, c) in &mixer.channels {
            let Some((_, source_rate, source_channels)) = &c.samples else {
                continue;
            };
            // Two channels (music and the like): Horizon plays them as they are, unattenuated and unpanned.
            let stereo = *source_channels > 1;
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
            // A source in another world is unheard but still advances, as in Horizon's mixer. The sounds of the world
            // (those with a place) get the settings page's source volume; music and the UI keep theirs.
            let placed = if !stereo && distance > 0.05 {
                source_volume
            } else {
                1.0
            };
            let source_gain = if c.relative || c.world == world {
                c.gain.clamp(0.0, 1.0) * placed
            } else {
                0.0
            };
            let gain = if stereo {
                source_gain
            } else {
                source_gain * distance_gain(c.linear, distance, reach)
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
                direction: (!stereo && distance > 0.05).then_some([x, y, -ahead]),
                distance,
                step: f64::from(*source_rate) / f64::from(rate) * f64::from(c.pitch.max(0.0)),
                source_gain,
                stereo,
            });
            if let Some((samples, _, source_channels)) = &c.samples {
                sources.push(TakenSource {
                    id,
                    samples: samples.clone(),
                    channels: usize::from(*source_channels).max(1),
                    looping: c.looping,
                    start: c.position,
                    position: c.position,
                    playing: true,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The visual debug mode (`steam_audio_hmod::debug`): its world layers, drawn by the mod's `world_drawn` hook, and its
// labels and statistics panel, drawn by its `hud` hook.
// ---------------------------------------------------------------------------------------------------------------------

/// The camera the world was last drawn with (camera-relative view-projection, column-major, and its position), for
/// the HUD to put the labels where the sources are.
static DEBUG_VIEW: std::sync::Mutex<Option<([f32; 16], [f64; 3])>> = std::sync::Mutex::new(None);

/// Draws the debug mode's world layers over the world (through the walls: what they show is mostly behind them).
/// For the mod's `world_drawn` hook, on the render thread; does nothing while they are off.
pub fn debug_draw(world: &mut crate::gfx::native_render::WorldDrawn<'_>) {
    use steam_audio_hmod::debug;
    let config = steam_audio_hmod::horizon::Config::get();
    let layers = debug::Layers::of(&config);
    let on = config.enabled && layers.any();
    if let Ok(mut view) = DEBUG_VIEW.lock() {
        *view = on.then(|| (world.view_proj.to_cols_array(), world.camera.to_array()));
    }
    if !on || !layers.world() {
        return;
    }
    let Some(lines) = debug::with_scene(|scene| debug::lines(scene, layers)) else {
        return;
    };
    if !lines.is_empty() {
        debug_gpu::draw(world, &lines);
    }
}

/// One thing the HUD hook draws: a rectangle (`[x, y, w, h]` in GUI pixels, ARGB) or a line of text (at, RGB).
pub enum HudDraw {
    Fill([f32; 4], u32),
    Text(String, [f32; 2], u32),
}

/// What the debug mode draws on the HUD: the statistics panel (its size, and its draws from its own top-left corner)
/// for the hook to place in a free corner, and the sources' labels where they are on the screen.
pub struct DebugHud {
    pub panel: Option<([f32; 2], Vec<HudDraw>)>,
    pub labels: Vec<HudDraw>,
}

/// The debug mode's HUD for a `width` × `height` GUI in the game's `language`, `text_width` measuring text in GUI
/// pixels. For the mod's `hud` hook; `None` while the debug layers are off.
pub fn debug_hud(
    language: &str,
    width: f32,
    height: f32,
    text_width: &dyn Fn(&str) -> f32,
) -> Option<DebugHud> {
    use steam_audio_hmod::debug;
    const LINE: f32 = 10.0;
    const PAD: f32 = 2.0;
    const BACKGROUND: u32 = 0x9000_0000;
    let config = steam_audio_hmod::horizon::Config::get();
    let layers = debug::Layers::of(&config);
    if !config.enabled || !layers.any() {
        return None;
    }
    let renderer = debug::renderer();
    let reflections = debug::reflections();
    let mut hud = DebugHud {
        panel: None,
        labels: Vec::new(),
    };
    // A box of lines of text with its top-left corner at `at`.
    let boxed = |out: &mut Vec<HudDraw>, at: [f32; 2], lines: &[(String, u32)], w: f32| {
        let h = lines.len() as f32 * LINE + PAD;
        out.push(HudDraw::Fill([at[0], at[1], w, h], BACKGROUND));
        for (i, (text, color)) in lines.iter().enumerate() {
            out.push(HudDraw::Text(
                text.clone(),
                [at[0] + PAD, at[1] + PAD + i as f32 * LINE],
                *color,
            ));
        }
    };
    let widest = |lines: &[(String, u32)]| {
        lines
            .iter()
            .map(|(t, _)| text_width(t))
            .fold(0.0f32, f32::max)
            + 2.0 * PAD
    };
    if layers.world()
        && let Ok(view) = DEBUG_VIEW.lock()
        && let Some((view_proj, camera)) = *view
    {
        let labels = debug::with_scene(|scene| {
            debug::labels(
                scene,
                layers,
                language,
                renderer.as_ref().map(|r| r.binaural_ids.as_slice()),
                reflections.as_ref(),
            )
        })
        .unwrap_or_default();
        for label in labels {
            let Some([x, y]) = debug::project(&view_proj, camera, label.at, width, height) else {
                continue;
            };
            let w = widest(&label.lines);
            let h = label.lines.len() as f32 * LINE + PAD;
            // Centred over the source, kept on the screen.
            let at = [
                (x - w / 2.0).clamp(0.0, (width - w).max(0.0)),
                (y - h).clamp(0.0, (height - h).max(0.0)),
            ];
            boxed(&mut hud.labels, at, &label.lines, w);
        }
    }
    if layers.stats {
        let lines = debug::with_scene(|scene| {
            debug::panel(
                Some(scene),
                renderer.as_ref(),
                reflections.as_ref(),
                &debug::paths(),
                language,
            )
        })
        .unwrap_or_else(|| {
            debug::panel(
                None,
                renderer.as_ref(),
                reflections.as_ref(),
                &debug::paths(),
                language,
            )
        });
        let w = widest(&lines);
        let h = lines.len() as f32 * LINE + PAD;
        let mut draws = Vec::new();
        boxed(&mut draws, [0.0, 0.0], &lines, w);
        hud.panel = Some(([w, h], draws));
    }
    Some(hud)
}

/// The world layers' lines on the GPU: each line a screen-facing quad of its width in pixels, drawn over the frame
/// with alpha blending and no depth test.
mod debug_gpu {
    use crate::gfx::native_render::{WorldDrawn, wgpu};
    use std::sync::Mutex;
    use steam_audio_hmod::debug::Line;

    const SHADER: &str = r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct Out {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(
    @builtin(vertex_index) corner: u32,
    @location(0) a: vec3<f32>,
    @location(1) b: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) width: f32,
) -> Out {
    var out: Out;
    var ca = u.view_proj * vec4<f32>(a, 1.0);
    var cb = u.view_proj * vec4<f32>(b, 1.0);
    // Cut the line where it goes behind the camera.
    let near = 0.01;
    if ca.w < near && cb.w < near {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec4<f32>(0.0);
        return out;
    }
    if ca.w < near {
        ca = mix(ca, cb, (near - ca.w) / (cb.w - ca.w));
    }
    if cb.w < near {
        cb = mix(cb, ca, (near - cb.w) / (ca.w - cb.w));
    }
    let pixels = u.viewport.xy;
    var along = (cb.xy / cb.w - ca.xy / ca.w) * pixels;
    if dot(along, along) < 1e-8 {
        along = vec2<f32>(1.0, 0.0);
    }
    along = normalize(along);
    let across = vec2<f32>(-along.y, along.x);
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0),
    );
    let c = corners[corner];
    let end = select(ca, cb, c.x > 0.5);
    // Half the width each side, in clip space at the end's depth.
    let offset = across * c.y * width / pixels * end.w;
    // Mid-depth: never clipped by the near or far plane, and there is no depth test.
    out.position = vec4<f32>(end.xy + offset, 0.5 * end.w, end.w);
    out.color = color;
    return out;
}

@fragment
fn fs(in: Out) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

    /// Bytes of one line: two ends, a colour and a width.
    const STRIDE: u64 = (3 + 3 + 4 + 1) * 4;

    struct Gpu {
        device: wgpu::Device,
        format: wgpu::TextureFormat,
        pipeline: wgpu::RenderPipeline,
        uniforms: wgpu::Buffer,
        group: wgpu::BindGroup,
        lines: wgpu::Buffer,
        capacity: u64,
    }

    static GPU: Mutex<Option<Gpu>> = Mutex::new(None);

    fn create(device: &wgpu::Device, format: wgpu::TextureFormat) -> Gpu {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("steam_audio debug lines"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("steam_audio debug lines"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("steam_audio debug lines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x3,
            1 => Float32x3,
            2 => Float32x4,
            3 => Float32,
        ];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("steam_audio debug lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: STRIDE,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &attributes,
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("steam_audio debug lines"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("steam_audio debug lines"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let capacity = 1024;
        Gpu {
            device: device.clone(),
            format,
            pipeline,
            uniforms,
            group,
            lines: lines_buffer(device, capacity),
            capacity,
        }
    }

    fn lines_buffer(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("steam_audio debug lines"),
            size: capacity * STRIDE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Draws `lines` over `world`'s frame.
    pub(super) fn draw(world: &mut WorldDrawn<'_>, lines: &[Line]) {
        let Ok(mut gpu) = GPU.lock() else {
            return;
        };
        // A new device (the renderer restarted) or target format: everything made again.
        if gpu
            .as_ref()
            .is_none_or(|g| g.device != *world.device || g.format != world.color_format)
        {
            *gpu = Some(create(world.device, world.color_format));
        }
        let Some(gpu) = gpu.as_mut() else {
            return;
        };
        let count = lines.len() as u64;
        if count > gpu.capacity {
            gpu.capacity = count.next_power_of_two();
            gpu.lines = lines_buffer(world.device, gpu.capacity);
        }
        let camera = world.camera.to_array();
        let mut bytes = Vec::with_capacity((count * STRIDE) as usize);
        let mut put = |v: f32| bytes.extend_from_slice(&v.to_le_bytes());
        for line in lines {
            for p in [line.a, line.b] {
                for a in 0..3 {
                    put((p[a] - camera[a]) as f32);
                }
            }
            for c in line.color {
                put(c);
            }
            put(line.width);
        }
        let mut uniforms = Vec::with_capacity(80);
        for v in world.view_proj.to_cols_array() {
            uniforms.extend_from_slice(&v.to_le_bytes());
        }
        for v in [world.size[0] as f32, world.size[1] as f32, 0.0, 0.0] {
            uniforms.extend_from_slice(&v.to_le_bytes());
        }
        world.queue.write_buffer(&gpu.uniforms, 0, &uniforms);
        world.queue.write_buffer(&gpu.lines, 0, &bytes);
        let mut pass = world
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("steam_audio debug lines"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: world.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &gpu.group, &[]);
        pass.set_vertex_buffer(0, gpu.lines.slice(..count * STRIDE));
        pass.draw(0..6, 0..count as u32);
    }
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
        stream: Held,
        at: Instant,
    }

    /// The mod's stream, closed only on purpose ([`Held::close`]). Left to itself, it is dropped by the thread-local's
    /// destructor, which on Windows runs while the process exits, after the system has ended every other thread:
    /// cpal's drop then joins the stream's thread, finds it gone and panics (a crash report on quitting). Then it is
    /// left as it is; the process is going anyway.
    struct Held(Option<cpal::Stream>);

    impl Held {
        fn is_none(&self) -> bool {
            self.0.is_none()
        }

        fn close(&mut self) {
            drop(self.0.take());
        }
    }

    impl Drop for Held {
        fn drop(&mut self) {
            if let Some(stream) = self.0.take() {
                std::mem::forget(stream);
            }
        }
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
            if let Some(mut old) = taken.take() {
                old.stream.close();
            }
            let stream = match open(&audio.device_name, &mixer) {
                Ok(stream) => stream,
                Err(e) => {
                    log::warn!(
                        "steam_audio: could not open an output ({e}); Horizon keeps its own"
                    );
                    *taken = Some(Taken {
                        mixer: Arc::downgrade(&mixer),
                        stream: Held(None),
                        at: Instant::now(),
                    });
                    return;
                }
            };
            let paused = match &audio._stream {
                Some(horizon) => horizon.pause().map_err(|e| e.to_string()),
                None => Ok(()),
            };
            let mut stream = Held(Some(stream));
            if let Err(e) = paused {
                log::warn!("steam_audio: could not pause Horizon's output ({e}); Horizon keeps it");
                stream.close();
            }
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
                        // Locks the mixer itself, only as long as it must.
                        Some(m) => super::render(&m, &mut scratch, channels, rate),
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
