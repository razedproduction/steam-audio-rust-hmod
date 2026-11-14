# steam_audio — Steam Audio for Horizon

[English](README.md) · [Русский](README.ru.md)

A native mod (`.hnmod`) for [Horizon](https://github.com/horizon-rust/horizon) that replaces Horizon's stereo panning
(a copy of OpenAL Soft) with the **binaural rendering of Steam Audio** from
[steam-audio-rust](https://github.com/razedproduction/steam-audio-rust), a Rust port of Valve's Steam Audio.

- **HRTF**: every sound in the world is convolved with Steam Audio's built-in HRTF (CIPIC, subject 124,
  1250 directions). In headphones you hear where a source is — behind, above or below you — not just left or right.
- **Air absorption**: Steam Audio's default three-band model (`AirAbsorptionModel`): distant sounds lose their high
  frequencies.
- **Occlusion and transmission** *(0.2.0)*: a sound behind a wall is duller and darker. Rays are cast through the
  blocks, like Steam Audio's volumetric occlusion and frequency-dependent transmission. The wall's material (rock,
  wood, wool, glass, metal...) comes from Steam Audio's presets, picked by the block's sound type. Around a corner the
  sound fades out smoothly.
- **Paths around walls** *(0.5.0)*: Steam Audio's pathing over blocks. A source behind a wall is also heard around
  it — through a doorway, from around a corner, along a corridor — from the direction the path arrives from, with
  the high frequencies lost at the bends (the UTD deviation model) and attenuated by the path's length. Probes are
  placed on floors and in openings, the paths between them are baked in the background (`PathBaker`), and for each
  hidden source `PathSimulator` looks them up. Since 0.6.0 this also works through windows and one-block-high
  crawlspaces, within ±48 blocks.
- **Fluids and partial blocks** *(0.6.0)*: water reflects almost everything and lets almost nothing through; slabs,
  stairs, chests and the like are walls according to their material; foliage, fences, glass panes, bars and railings
  do not block sound, but muffle it (foliage more so at high frequencies).
- **Block shape and state** *(0.7.0)*: a partial block gets in the sound's way where it actually is, not across its
  whole cell — by the shape of its state. A closed door or trapdoor muffles, an open one doesn't; above a bottom slab
  sound passes freely.
- **Under water** *(0.7.0)*: while your head is in water, the whole world sounds muffled (lows pass, highs almost
  vanish), and the water around is a medium, not a wall: a source in the same body of water is heard directly.
- **Reverb from reflections** *(0.4.0)*: a real Steam Audio reflections simulation over the blocks around the
  listener. Rays (`ReflectionSimulator`, 2048 rays × 16 bounces) bounce off walls with their materials, an impulse
  response (first-order ambisonics) is reconstructed from the energy field, and Steam Audio's hybrid reverb
  (`HybridReverbEffect`) plays the first second by convolution and the rest as a parametric tail. Early reflections
  arrive from where the walls are and turn with your head (binaural ambisonics decoding). A cave sounds booming, a stone
  hall has a long tail, a room made of wool sounds dead, a field sounds dry. Since 0.6.0 the grid is 96 × 64 × 96:
  large caves sound along their whole length.
- **Per-source reflections** *(0.6.0, 0.7.0)*: the two nearest sources (within 32 blocks) that have been playing for
  more than a second get their own reflections, computed from where they sound: a sound in the next hall hums with its
  echo and arrives from where it reflects, not from under the listener's feet. The other sources share the listener's
  reverb.
- **Physical attenuation** *(0.7.0)*: instead of Horizon's linear attenuation, which cuts a sound to zero at its range
  (16 blocks for a mob), Steam Audio's inverse distance `r₀ / d` with `r₀` a quarter of the range: full volume up
  close, the same as in Horizon at half the range, −12 dB instead of silence at the full range, and beyond it the sound
  keeps playing quietly and fades smoothly out toward four ranges (64 blocks for a mob). In a big cave you can hear a
  mob at the other end of it. The vanilla curve is available with a button or `HORIZON_STEAM_AUDIO_DISTANCE=vanilla`.
- Walls muffle sound but don't make it inaudible: by default transmission is softened (`HORIZON_STEAM_AUDIO_WALLS`).
- Music, the interface and "at the listener" sounds without a direction are panned as in Horizon and play without
  reverb. Two-channel sounds (Horizon has stored stereo since Nightly 2026-11-08) *(0.8.0)* play as they are — left and
  right channel, without attenuation or panning, as in Horizon.
- **Advanced settings** *(0.8.0)*: source volume, surface absorption and scattering for reflections, hearing range —
  a separate section of the settings page (see [Settings](#settings)).
- **Materials and blocks** *(0.9.0)*: each of the 14 materials has its own absorption, sound isolation and scattering;
  each block of the game has its own material and its own values on top of the material. Blocks are paged, searched by
  id (typed right on the page) and filtered by tag (`#minecraft:logs`...); a whole tag can be tuned at once.
- **Visual debug mode** *(0.8.0)*: four layers, each with its own button in the settings — sources with occlusion
  rays and walls, paths around walls with probes, the reflections grid and rays, a statistics panel. You can see what
  the mod "thinks" about each sound (see [below](#visual-debug-mode-080)).

## Compatibility

| | |
| --- | --- |
| Horizon | Nightly 2026-11-08, `fdc87aecdbd116bf1e08adcd062fcbaea8c2e2ca` (`nightly-source/2026-11-08-fdc87ae`) |
| horizon-mixins | `5a64af938ec86b00d53d16a408aabc0038fb3e57` (the launcher's revision at this commit) |
| Minecraft | 26.3 |
| Output sample rate | 44.1, 48 or 24 kHz (the rates of the built-in HRTF); at other rates the sound stays vanilla |

The mod is tied to an exact Horizon commit: it touches the internals of `horizon-client-render`, not only the stable
`native_hooks` points, so on a different Nightly the launcher will warn about a version mismatch.

## Installation

From a GitHub release:

```bash
horizon-launcher mods install-github https://github.com/razedproduction/steam-audio-rust-hmod --trust-native-code
```

Or build the file yourself (needs Python 3.11+):

```bash
python tools/package.py
```

This produces `dist/steam_audio-<version>.hmod`: `horizon-native.json` and `mixins/steam_audio/`. Horizon's registry
currently publishes native mods as `.hnmod` (the same format); `--hnmod` puts a copy with that extension next to it.
To build the game with the mod through the launcher, point it at the bundle folder (or an unpacked `.hmod`):

```bash
horizon-launcher native-build mixins/steam_audio --trust-native-code
```

Native mods are code with the user's privileges; the launcher compiles it into the game. Read the source before
installing (all of it is in `mixins/steam_audio/`).

## Settings

### In game *(0.3.0)*

**Options → Mods → Steam Audio.** Each button is one setting with its value; a click switches to the next value and
applies immediately, without a restart (the audio picks the change up within ~10 ms). The page is split into three
sections *(0.8.0)*: the main one, "Advanced settings..." and "Debug..."; the "< Back" button returns to the main one.

**Main section**

| Button | Values |
| --- | --- |
| Steam Audio | On / Off (off — sound as in Horizon) |
| Source volume | 25, 50, 75, **100**, 110, 125, 150% — sounds of the world (those that have a position); music and interface are unchanged *(0.8.0)* |
| Walls muffle sound | On / Off |
| Wall strength | 0, 15, 25, **35**, 50, 75, 100% |
| Sound goes round walls | **On** / Off (works while walls are on) |
| Reverb | On / Off |
| Reverb level | 0, 25, **50**, 75, 100, 150, 200% |
| Muffled under water | **On** / Off |
| Distance attenuation | **Physical** / Horizon's |
| Advanced settings... | opens the section below |
| Materials... | materials *(0.9.0)*; "(changed: N)" if any are tuned |
| Blocks... | the game's blocks *(0.9.0)*; "(changed: N)" if any are tuned |
| Debug... | opens the debug section; "(on)" if at least one layer is enabled |
| Reset to defaults | everything back to defaults (debug is turned off); materials and blocks are reset by their own buttons |

**Advanced settings** *(0.8.0)*

| Button | Values |
| --- | --- |
| Air absorption | **On** / Off |
| Surface absorption | 25, 50, 75, **100**, 150, 200, 300% — a multiplier on the absorption of Steam Audio's materials for reflections: less — a longer, brighter echo, more — duller and shorter |
| Reflection scattering | **5** (Steam Audio's presets), 15, 30, 50, 75, 100% — the share of reflected sound that goes in all directions instead of mirror-like |
| Hearing range | ×2, ×3, **×4**, ×6, ×8 of the sound's Horizon range (physical attenuation only) |
| HRTF interpolation | **Nearest** / Bilinear |
| Binaural voices | 8, 16, **32**, 48, 64 |

After the absorption or scattering changes, the reverb is recomputed (about a second until it is back to its previous
accuracy).

**Materials** *(0.9.0)*

A list of 14 materials (Steam Audio's presets and three of the mod's own: water, foliage, thin — fences, panes, bars);
each has its own values, with `*` on the changed ones. A material's page:

| Button | Values |
| --- | --- |
| Absorption | 0, 25, 50, 75, **100**, 150, 200, 300, 400% of the preset's absorption: less — a brighter, longer echo, more — duller (100% absorption is the limit) |
| Isolation | 0, 25, 50, 75, **100**, 150, 200, 300%: how much a wall made of it muffles sound passing through it (0% — lets everything through, 200% — lets through the square of what it let through) |
| Scattering | **global** (from the advanced settings), 0, 5, 15, 30, 50, 75, 100% |
| Reset the material | back to the preset's values |

**Blocks** *(0.9.0)*

All the game's blocks in alphabetical order, 15 per page: `oak_planks: Auto (Wood)`, with `*` on the tuned ones. At the
top:

- **Search** — click the button and type right on the page: the list is filtered as you type. Latin letters, digits,
  `_` (a space or `-`), `:` (`;`), Backspace; Enter or Esc — done. The Russian keyboard layout doesn't type (the mod
  receives physical keys), but block ids are Latin. "Clear the search" resets it.
- **Tag** — the list of the game's block tags (`#minecraft:logs (52 blocks)`, searchable with the same search); a
  selected tag leaves only its own blocks, and "Tune the whole tag..." appears — a page where every change applies to
  all the tag's blocks at once.
- **Changed only**, **Reset all blocks (N)**, pages forward and back.

A block's page: **material** (Auto — by the block's sound, as before; None — not a wall, sound passes through; or any
of the 14), **absorption**, **isolation** and **scattering** ("the material's" or your own: a multiplier on top of the
material, and for scattering — its own value) and "Reset the block". Changes apply immediately: the grid of blocks
around you is copied again (about a second) and the reverb is recomputed. There can be any number of blocks with their
own values (not only a material), but at most 240 distinct sets of values: blocks with identical values share one.

**Debug** *(0.8.0)*

| Button | Values |
| --- | --- |
| Debug: sources and occlusion | On / **Off** |
| Debug: paths round walls | On / **Off** |
| Debug: reflections and reverb | On / **Off** |
| Debug: statistics panel | On / **Off** |

Labels are in Russian when the game's language is Russian (as well as Ukrainian and Belarusian), otherwise in English.
The choice is stored by Horizon's means (`native_set_setting`, the mod's data folder `steam_audio`) and read at startup.

### Environment variables

If a variable is set, it **takes precedence** over the menu choice (at startup). Two settings exist only here, because
they change how the HRTF is loaded: `VOLUME_DB` and `HRTF`. The menu saves only its own values *(0.7.3)*: before 0.7.3
any click also wrote to the settings whatever a variable had set, and it stayed after the variable was removed.

| Variable | What it does |
| --- | --- |
| `HORIZON_STEAM_AUDIO=0` | turn the mod off: sound as in Horizon |
| `HORIZON_STEAM_AUDIO_AIR=0` | turn air absorption off |
| `HORIZON_STEAM_AUDIO_OCCLUSION=0` | turn occlusion and transmission through blocks off |
| `HORIZON_STEAM_AUDIO_WALLS=x` | how much walls muffle, 0…1 (default 0.35; 1 — Steam Audio's physical values, a stone wall ≈ −36 dB) |
| `HORIZON_STEAM_AUDIO_PATHING=0` | turn paths around walls off (a hidden source is heard only through the wall) |
| `HORIZON_STEAM_AUDIO_REVERB=0` | turn the environment reverb off |
| `HORIZON_STEAM_AUDIO_REVERB_LEVEL=x` | reverb level in a fully enclosed place, 0…2 (default 0.5) |
| `HORIZON_STEAM_AUDIO_UNDERWATER=0` | under water, sound is as in air |
| `HORIZON_STEAM_AUDIO_DISTANCE=vanilla` | Horizon's linear attenuation (silence at the sound's range) instead of the physical one |
| `HORIZON_STEAM_AUDIO_RANGE=x` | hearing range with physical attenuation, in Horizon ranges, 2…8 (default 4) |
| `HORIZON_STEAM_AUDIO_SOURCE_VOLUME=x` | volume of world sounds, 0…2 (default 1) |
| `HORIZON_STEAM_AUDIO_ABSORPTION=x` | multiplier of surface absorption for reflections, 0…4 (default 1) |
| `HORIZON_STEAM_AUDIO_SCATTERING=x` | surface scattering, 0…1 (default 0.05) |
| `HORIZON_STEAM_AUDIO_INTERPOLATION=bilinear` | bilinear HRTF interpolation (smoother when moving, costlier) |
| `HORIZON_STEAM_AUDIO_VOICES=N` | how many of the loudest sources to render binaurally (default 32; the rest — vanilla panning) |
| `HORIZON_STEAM_AUDIO_VOLUME_DB=x` | HRTF loudness correction in dB (−24…24) |
| `HORIZON_STEAM_AUDIO_HRTF=path` | your own HRTF in Steam Audio's `.hrtf` format instead of the built-in one |

Binaural sound is designed for headphones.

## How it works

```
mixins/steam_audio/
  horizon-mixin.toml   manifest: module, seven hooks, crate
  listener.rs          HEAD hook on Audio::set_listener: takes the output over
  frame.rs             the frame point: loads the settings (once) and the world's acoustics
  mods.rs              the mods point: the mod's row in Options → Mods
  settings.rs          the settings point: labels of the settings page's buttons
  settings_click.rs    the settings_click point: a click on a button
  world_drawn.rs       the world_drawn point: the debug mode's lines over the world
  hud.rs               the hud point: labels over sources and the statistics panel
  output.rs            pub support module horizon_client_render::audio::steam_audio (inside audio.rs)
  engine/              the steam-audio-hmod crate: vendored steam-audio-rust + renderer + acoustics + paths +
                       reflections + settings + debug mode
    data/cipic_124.hrtf the built-in HRTF (tools/gen_hrtf.py from core/src/core/cipic_124.inl)
```

### The settings page

The page of a native mod is drawn by Horizon itself: the mod adds its row to the list (`mods`), supplies the labels of
the buttons (`settings`, asked for when the page opens and after every click) and hears the clicks (`settings_click`) —
all three points are stable (`native-hooks.json`, version 1). The page's model is `steam_audio_hmod::settings`:
ru/en labels, the cycle of values, saving by key (`enabled`, `walls`, `reverb_level`...). The configuration is live:
`Config::set` changes it under an `RwLock` and bumps a version counter, the audio thread checks the counter at the
start of its callback, and reads whether the mod is enabled from an atomic flag, copying nothing.

### Sound output

Horizon plays sound through a single cpal stream whose callback calls `audio::mix`. horizon-mixins doesn't let you
intercept `mix` itself (`E_UNSUPPORTED_CFG` because of the `#[cfg_attr(...)]` on the function, and `build_stream`,
which calls it, is entirely under `#[cfg(feature = "audio")]`). So the mod **takes the output over**:

1. On a camera update (`Audio::set_listener`) the hook calls `steam_audio::take_over`. It opens its own stream on the
   same device and **pauses** Horizon's stream (it doesn't close it: Horizon still sees an open output and keeps
   watching for device changes). The mod's stream is closed only explicitly (when reopening); on exit from the game it
   is left as it is *(0.7.2)*: on Windows the destructor of the thread-local it lives in runs after the system has
   already terminated the other threads, and cpal panicked waiting for its own (the crash report on exit in 0.7.1 and
   earlier).
2. The mod's callback renders the same `Mixer` through `steam_audio_hmod::horizon::Renderer` in frames of 512 samples:
   the same geometry as the vanilla mixer (the camera basis, the world check, attenuation), then the direct path's EQ
   (air × occlusion/transmission), `BinauralEffect`, the reverb bus (`HybridReverbEffect` → `AmbisonicsDecodeEffect`
   in the camera's orientation) and the vanilla limiter. Rendering takes milliseconds and grows with the number of
   sources, while the game thread locks the same mixer mutex on every `play`, `set_position`, `stop`. So *(0.7.1)* the
   mutex is held only to take the voices' parameters (the samples go by `Arc`) and later to put the positions back;
   the rendering itself runs without it. A position isn't put back if the game has restarted, stopped or replaced the
   channel in the meantime. Before 0.7.1 the mutex was held for the whole render, and with many sources the game
   thread waited on it for a long time on every sound.
3. While the HRTF is loading (in a separate thread, a fraction of a second), at an unsupported sample rate, on a mono
   output and in splitscreen, the callback calls the vanilla `mix` — the sound doesn't drop out for a moment.
4. The vanilla `mix` fills `Mixer::cameras`, and the mod's output leaves it empty. If it is filled, Horizon's stream is
   playing again (the launcher reopened the device): the mod reopens its own and pauses Horizon's again.

### World acoustics

Blocks live on the game thread, so a hook on the stable `frame` point (once per frame) calls
`steam_audio::update_acoustics(&game.audio, &game.world, game.res.facts())`. No more than 20 times a second it:

- takes from the mixer up to 32 nearest playing sources in the listener's world and processes them 8 per frame
  *(0.7.1)*: a round of 32 takes 4 frames instead of one, without a spike of several milliseconds;
- for each casts rays (an Amanatides–Woo grid traversal) to the source's center and to five points around and above
  it: the share of free rays is the occlusion; along the central ray each wall multiplies the per-band transmission
  by its material's coefficients (to the power of `WALLS`). A wall is a block with full collision, and among partial
  ones, fluids, slabs, stairs, doors, trapdoors and the like (`Surface::classify_partial`); a partial wall holds a ray
  back only if it passes through the boxes of its shape (the state's `outline` in `BlockFacts`: an open door is a
  panel along the wall, a closed one across the opening); the source's own block (a note block, a furnace) doesn't
  get in the way. Foliage and thin blocks (fences, panes, bars, railings) give no occlusion, but each one on the
  central ray multiplies the bands by its transmission (`through`; up to 12 blocks);
- publishes the results; the audio thread takes them without waiting (`try_lock`), changes are smoothed;
- checks whether the camera is in water (the `fluid` of its cell: water, kelp, seagrass, a waterlogged block). Then
  water cells are void for rays and grids, and within ~50 ms the renderer moves all of the world's sounds (direct,
  path, the send to reverb) to the `UNDERWATER_GAINS` bands (0.9 / 0.3 / 0.06).

The reflection and path grids consist of whole cells, so a partial wall gets into them only if it closes a cell between
two open ones: its shape covers the whole face along some axis (`shape_seals`), and both neighbors along that axis are
not solid walls (`shape_blocks_cell`). A closed door in a doorway and a floor of slabs over a void are walls; an open
door (its panel lies along the wall) and a slab on the ground are not.

In the renderer the direct path's per-band gains are `occlusion + (1 − occlusion) · transmission` (the formula of
Steam Audio's `DirectEffect`), multiplied by `through` and normalized by `EqEffect::normalize_gains`.

### Paths around walls *(0.5.0)*

Steam Audio's pathing (`steam_audio_hmod::paths`) works in the same 96 × 64 × 96 grid as reflections:

1. **Probes.** Every 4 blocks in x and z there is a probe 1.5 blocks above each floor that has two free blocks above it
   (Steam Audio's uniform floor placement, but by cells rather than by rays downward), or in the middle of a
   one-block-high gap (a crawlspace, a window in a wall). A one-block-wide door between such probes is easy to miss,
   so one more probe is placed in each opening: a floor between two walls, open along the other axis (in corridors —
   no more often than every 2.5 blocks). No more than 768 probes, those nearest to the center of the grid.
2. **Baking.** The `steam-audio-pathing` thread, on each new grid, uses `PathBaker` to build the probes' visibility
   graph (rays through cells; probes further than 16 blocks apart are not connected) and the shortest paths between
   all pairs: ~110–230 ms in release on 580–770 probes. Between bakes — at least 500 ms and at least three durations
   of the last one: the thread is busy no more than a quarter of the time.
3. **Search.** Every 50 ms, for each source that walls cover at least partially, the game thread takes the probes
   around the source and the listener (those visible to them, weighted by distance) and asks `PathSimulator::find_paths`
   about the baked paths between them (from all the source's probes, as if the direct path were blocked). From the
   result come the direction the paths arrive from, their EQ by the UTD deviation model, the path's length (via
   `distance_ratio`) and the share of probes the path connected. A sound inside its own block (a note block) comes out
   of the neighboring free cell. For 32 sources — up to 2 ms.
4. **Sound.** A path is the source's second binaural voice: the dry signal, EQ (bends × air over the path's length),
   the gain `(1 − occlusion) · share · g(path length) / g(direct distance)`, where `g` is this source's distance
   attenuation, and the HRTF for the path's direction in the camera's orientation. Transmission through the wall is
   still heard, and a visible source needs no path. A path's appearance and disappearance are smoothed, and a vanished
   path fades out within a frame. Voices over the binaural budget get the path's average loudness in vanilla panning.

Steam Audio's `PathEffect` (the paths' ambisonics with a binaural HRTF) is not used: in binaural mode, like the
original, it fills only the first half of the bins of the mixed HRTF, that is, it cuts everything above a quarter of
the sample rate. A separate `BinauralEffect` along the paths' average direction (`calcAverageDirectionForPaths`)
sounds full-band.

### Reflections and reverb *(0.4.0, 0.6.0)*

This is Steam Audio's "listener-centered reverb": a source is placed at the listener's point, and the reflections
simulator computes how much energy returns to it, by time, band and direction. The `steam_audio_hmod::reflections`
module:

1. **The block grid.** The game thread copies the blocks from a 96 × 64 × 96 cube around the camera (`GridBuilder`): a
   grid at a new place — up to 24,576 blocks per frame and no longer than 1.5 ms (the whole thing in ~24 frames), and
   when a grid around the camera already exists, only to notice block changes — 2,048 per frame (*0.7.0, 0.7.1*; in
   0.6.0 the copying ran at the full budget without stopping). Foliage and thin blocks don't get into the grid: rays
   pass through them. The grid's corner is snapped to a step of 8 blocks, so while the player walks nearby the grid is
   the same and the simulation keeps refining. A finished grid goes to the `steam-audio-reflections` thread.
2. **Tracing.** `Voxels` implements Steam Audio's `RayTracer` directly on the grid (a cell traversal, the normal is the
   block's face, the material is a Steam Audio preset by sound type), without triangles or a BVH.
   `ReflectionSimulator`: 2048 rays, 16 bounces, 2 s, ambisonics order 1, two threads.
3. **Sources.** Along with the listener, up to 2 sources (`MAX_SOURCES`) are simulated, chosen by the game thread: the
   nearest within 32 blocks of those that have been playing for more than a second. Short sounds (steps, hits) will
   end before their reflections arrive, and new channels every fraction of a second would not let the simulation
   settle: in 0.6.0 that is why it never stopped. A source inside its own block sounds from the neighboring free cell.
4. **Accumulation.** While the listener (to 0.25 block) and the blocks are the same, the energy fields are averaged
   over runs, as in `SimulationManager` (up to 16 runs, then the simulation waits for changes); a source's field — while
   it has moved less than a block. A run takes 22–44 ms with only the listener and 63–106 ms with two sources
   (release); between runs at least 250 ms and at least three durations of the last one.
5. **IR.** From each field: the reverb time by band (`reverb_estimator`), the impulse response of the first second
   (`Reconstructor`), its transition into the parametric tail over the last quarter (`HybridReverbEstimator`) and the
   partitioning for convolution (`OverlapSavePartitioner`). The audio thread takes the result without waiting; an IR
   change is crossfaded within one frame.

In the renderer all positional sources send their signal (with distance attenuation and air — the reverb hears a
source behind a wall too, sound goes around obstacles) to the reverb bus. `HybridReverbEffect` gives 4 ambisonics
channels in world coordinates, `AmbisonicsDecodeEffect` rotates them into the camera's orientation and decodes them
binaurally with the same HRTF. The level changes what goes into the bus, so a disabled reverb lets an already started
tail ring out. Reflections are physical (relative to the direct sound at 1 m), while Horizon's attenuation is linear,
so the bus is boosted 1.4 times: at level 1 the reverb of a 9 × 5 × 9 stone cave equals the direct sound in energy
(the `measure_reverb_level` measurement).

A source with its own reflections sends its signal not to the shared bus but to its own `HybridReverbEffect` — without
Horizon's attenuation and without air: both are already in its reflections. The outputs of all reverbs are summed into
one set of ambisonics and decoded together. When a source loses its own reflections (it moved away, it was displaced),
it sounds in the shared bus again, and its reverb lets the tail ring out and is removed. *(0.7.2)* The ring-out goes
through `tail()`: `apply()` with silence on the input, as in Steam Audio, starts the tail anew every frame, and before
0.7.2 such reverbs were never removed — with every change of sources there were more of them, until the audio thread
stopped keeping up (after a while the sound started to lag and crackle). There are no more than 4 source reverbs (2
with their own reflections and 2 ringing out); they are created in advance and reused, because creating a new one (a
convolution of a second into 4 channels) takes more than a millisecond in the audio thread. A new impulse response
(a crossfade, twice as costly) — no more than one per frame.

### Audio thread budget *(0.7.2)*

The loudest sources within the budget (`Binaural voices`, 32 by default) are rendered binaurally, and a source with a
path around walls counts as two voices (the path is a second `BinauralEffect`). The rest go through vanilla panning
with the same losses at walls and in air. The renderer measures what share of the frame time its rendering takes: above
60% the budget is reduced by a quarter (not below 8 voices), below 35% it grows back, no more often than once every 20
frames. On a weak CPU distant quiet sounds lose their HRTF, but the sound doesn't crackle. *(0.7.3)* In 0.7.2 the very
first budget reduction overflowed the counter (it starts at "unlimited"): in a debug build — a panic of the audio
thread and binaural sound off until the end of the game, in release — instead of decreasing, the budget jumped to the
maximum.

*(0.7.3)* At reverb level 0 sources no longer send a signal to the reverbs (the audio thread used to convolve
silence), and new reflections wait until the reverb sounds again: before, the parametric tail took their reverb times
while the convolution stayed with the old impulse response. A source's reverb that has new reflections can no longer be
displaced for the sake of another source from the same batch (the tail was cut off, a click).

### Visual debug mode *(0.8.0)*

Four layers, each its own button in **Options → Mods → Steam Audio**, all off by default and stored like the other
settings. Lines are drawn over the world, through walls (`world_drawn`: its own wgpu pipeline, each line a quad of a
set thickness in pixels, without a depth test); labels and the panel are drawn on the HUD (`hud`; not visible with
F1). It's most convenient to look from third person (F5): from first person the rays converge on a point under the
crosshair.

| Layer | What you see |
| --- | --- |
| Sources and occlusion | a frame at each source (green — in view, yellow — half hidden, red — blocked); six occlusion rays from the listener (green passed, red hit something); orange frames — walls on the line of sight, light-green — foliage and fences on it. The label: id, HRTF or vanilla panning (outside the budget), visible share, number of walls, transmission by band, losses in foliage |
| Paths around walls | cyan crosses — pathing probes; yellow segments — the paths found from probe to probe; a magenta line from the listener — where the path arrives from. The label: the path's loudness, its length, the bends' EQ |
| Reflections and reverb | a blue frame — the 96 × 64 × 96 grid; 48 rays from the listener with three mirror bounces (lighter — nearer the start); purple frames — sources with their own reflections, with their RT60 by band and the number of runs in the label |
| Statistics panel | voices (playing, in the world, binaural, the budget and the current limit, paths), the audio thread's load (green < 35%, yellow, red > 60%), the reverb level and the sources' reverbs, reflection runs and the listener's RT60, probes and the path baking time, how many sources are blocked and with a path, whether the listener is under water. It takes the first free corner (F3 occupies the top ones) |

The data is collected only while at least one layer is on (`steam_audio_hmod::debug`): the game thread, at the end of
each round of sources (once in ~50 ms), puts down the scene — rays and walls from the same tracing as the occlusion
(`source_acoustics_traced`), path segments from the `find_paths` callback (without validation it merely reports each
segment), the probes of the last bake, the reflection rays over the last grid; the audio thread, after its callback,
its statistics (through `try_lock`, it doesn't wait); the reflection and path threads — the run and bake time. With the
layers off the mod works as in 0.7.3.

### Differences in the vendored steam-audio-rust

The mod's crate can depend only on crates already recorded in Horizon's `Cargo.lock`. `rustfft`, `realfft` and
`num-complex` aren't there, so:

- `src/fft.rs` is our own FFT (iterative radix-2, real-to-complex through a half-size complex FFT, twiddle factors in
  `f64`) with the same API;
- `src/complex.rs` is our own `Complex32` with the needed subset of the `num_complex` API;
- the built-in HRTF is read from `engine/data/` instead of by `build.rs`; SOFA support isn't compiled;
- `src/horizon.rs`, `src/acoustics.rs`, `src/paths.rs`, `src/reflections.rs` and `src/debug.rs` are new: the renderer
  for Horizon's mixer, occlusion, paths around walls, block-based reflections and the debug mode.

The rest of steam-audio-rust's code is unchanged.

## Verification

```bash
cargo test -p steam-audio-hmod
```

182 tests (and 5 measurements under `--ignored`): the whole unit test suite of steam-audio-rust plus tests of the FFT
(compared against a DFT in `f64`), the renderer (sides, a wall muffles, a hidden source is heard from the side of the
opening, the reverb tail, reflections turn with the head, a disabled reverb rings out, a source sounds in its own
reflections, settings change on the fly, muffled under water, physical attenuation beyond the sound's range, the
sources' reverbs ring out and return to the pool, the voice budget under overload, a path is a second voice), acoustics
(grid traversal, a wall and a corner, materials, fluids and partial blocks, block shapes: a closed door muffles, an open
one doesn't, a slab and stairs close a cell along their axes, foliage muffles without occlusion, smoothing, a path's
appearance and disappearance), paths (probes on the floor, a path through a door between two rooms and its absence
without the door, a path through a window and a crawlspace, a sound from its own block, the baking thread), reflections
(tracing over voxels, the grid in parts, foliage is not a wall, a cave / hall / field / wool room, a cave wider than 64
blocks, the side of a reflection off a wall, accumulation and stopping, sources' reflections), the settings page
(ru/en labels, the cycle of values, reset, saving, the environment's priority) and the debug mode (the rays and walls
tracing matches the occlusion, path segments through a door, reflection rays bounce inside a room, the lines of each
layer, labels and the panel in ru/en, projection onto the screen).
In addition, during development:

- the golden tests of the original steam-audio-rust (a comparison against reference data from the C++ Steam Audio) were
  run with this FFT and `Complex32` — all pass;
- `output.rs`, `listener.rs` and `frame.rs` were compiled and tested in mock crates with exact copies of the needed
  definitions of the pinned Nightly (`Mixer`, `Channel`, `Audio`, `mix`, `World::block`, `BlockFacts`, `Game`), with
  cpal 0.16.0 and without the `audio` feature: sides and rotation, the fallback to `mix`, stopping sounds, the
  end-to-end path `update_acoustics` → renderer (a wall in a stone room muffles a source behind it, reverb appears) and
  the settings page through all four points (`frame`, `mods`, `settings`, `settings_click`) with real
  `native_setting`/`native_set_setting` modeled on Horizon's;
- for 0.4.0, `output.rs` was compiled and run in a simplified mock crate (the fields of `Mixer`/`Channel`/`Audio` and
  `World::block`/`BlockFacts` as the mod uses them, without the `audio` feature): the end-to-end path stone cave →
  grid → simulation thread → renderer, and after a source stops the reverb tail remains;
- for 0.5.0, `output.rs` was likewise compiled in a simplified mock crate and the end-to-end path run: two stone rooms
  with a door → `update_acoustics` → grid → baking thread → the path of a source in the neighboring room arrives from
  the side of the door;
- for 0.6.0, the same mock crate: the nearest source gets its own reflections from the simulation thread, a path
  through a door is still found;
- for 0.7.0, `output.rs` was compiled in a mock crate with exact copies of the pinned Nightly's definitions (`Mixer`,
  `Channel`, `Audio`, `World::block`, `BlockFacts`/`StateFacts` with `outline` and `fluid`, without the `audio` feature)
  and the end-to-end path `update_acoustics` → published acoustics was run: behind a closed oak door in a stone wall the
  highs are ≈ 0.16, behind an open one — as behind an empty opening (≈ 1.0); with the head in water the "under water"
  flag is raised, and a source in the same body of water is not blocked by the water;
- for 0.7.1, in the same mock crate, 64 looped sources: the output callback renders in up to ~3 ms, while the "game
  thread", moving a sound in parallel, waited for the mixer mutex for no more than ~40 µs (before 0.7.1 — the whole
  render time); in a stone cave with a wall and 64 sources, `update_acoustics` after the grid is built — a median of
  ~0.5 ms per frame, a maximum of ~1 ms (while the grid is being built at a new place — up to ~1.5 ms per frame);
- load (the `measure_audio_load` measurement, release, this machine): the audio thread with 32 binaural voices and
  the listener's reverb — about 25% of one core, plus about 9% for each of (at most 2) sources with their own
  reflections — in 0.6.0 there could be 4 of them, which gave up to 64%; plus a path voice for each hidden source.
  Background threads (reflections — two threads, path baking — two) are busy no more than a quarter of the time and
  stop when idle;
- a loaded cave (the `measure_busy_audio_load` measurement: 32 hidden sources, each with a path, 2 own reverbs, new
  reflections every 250 ms and a new sound every 50 ms): a callback for 10 ms of sound takes 5.2 ms on average, 6.7 ms
  at most. In 0.7.1 — 15.6 ms on average and up to 29 ms, and it grew over time (accumulated reverbs).
- for 0.8.0, `gfx::native_render::WorldDrawn` (the pinned Nightly's fields, wgpu 30.0.1, glam 0.30.10) and the `hud`
  point with `Game`, `HudBatch`, `Corners`, `Corner` were added to the mock crate; `world_drawn.rs` and `hud.rs` were
  compiled in their places, `output.rs` — also with the `audio` feature. The end-to-end path: a stone wall →
  `update_acoustics` → the debug scene (six blocked rays, the wall `[3, 64, 0]`, the grid and 48 reflection rays; with
  the layers off the scene disappears). Lines were drawn with a real GPU (Vulkan) into a 64 × 64 texture: a source's
  frame in the center, a ray under it, the corners empty; the HUD puts the panel in the free bottom-left corner and
  the label above the source; with the layers off the HUD draws nothing.

- for 0.8.0 on Nightly 2026-11-08 (`fdc87ae`), the launcher (0.2.21) applied the mixins and compiled `steam-audio-hmod`
  and `horizon-client-render` without errors; the last step (writing the `.rlib`) failed on a Windows file lock
  (`os error 32`) in the build environment, so a full game build did not finish here.

- for 0.9.0, the model of materials and blocks (the palette, material variants, storage) and the settings page
  (sections, pages, keyboard search, tags, tuning a whole tag) are covered by unit tests; the code on Horizon's side
  (`output.rs`, `frame.rs`, `key.rs`) was checked against the `fdc87ae` sources for field names and visibility, but
  building the game with the launcher in the development environment failed (the same file lock, `os error 32`).

Version 0.1.0 was verified in the game. For 0.2.0–0.9.0, building through the launcher and verifying in the game are
left to the user.

## What's missing (for now)

Block states and under-water sound were done in 0.7.0. The remaining limitations are sizes and approximations:

- Everything is computed in a 96 × 64 × 96 grid around the listener: a cave larger than it sounds shorter than it is,
  and a source further than ±48 blocks has neither a path nor its own reflections (it is heard through walls and in
  the shared reverb).
- Own reflections go to the 2 nearest long-playing sources; the rest share the listener's reverb.
- A sound the game didn't create the mod won't play: if the server (in a network game) doesn't send sounds beyond
  their range, physical attenuation will only extend the ones that arrived.
- The shape of a partial block is exact for direct sound, but in the reflection and path grids a block is either a
  whole wall or an empty cell (see "World acoustics").
- Under water the whole world is muffled equally: the boundary between water and air between the listener and a source
  is not counted separately, and a body of water's reverb is the same simulation over its walls as in air.

## License

Apache-2.0, like steam-audio-rust and Steam Audio. Steam Audio is © Valve Corporation; steam-audio-rust is ©
razed.production (see `mixins/steam_audio/engine/NOTICE`). The mod is not affiliated with Valve and not endorsed by it.
