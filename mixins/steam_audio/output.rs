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

use super::{Audio, Mixer, State, distance_gain, mix, stereo_gains};
use steam_audio_hmod::horizon::{Voice, read_linear, with_renderer};

/// Takes Horizon's sound output over, once per output Horizon opens. Called on the game thread at every camera
/// update; without the `audio` feature there is no output and it does nothing.
pub(crate) fn take_over(audio: &mut Audio) {
    #[cfg(feature = "audio")]
    output::take_over(audio);
    #[cfg(not(feature = "audio"))]
    let _ = audio;
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
                c.gain.clamp(0.0, 1.0) * distance_gain(c.linear, distance)
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
        if audio.backend.owner != 0 || audio._stream.is_none() || !Config::get().enabled {
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
