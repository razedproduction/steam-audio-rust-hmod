//! The mod's settings page in Options > Mods.
//!
//! Horizon draws a native mod's page itself: the mod gives the labels of its buttons (`native_settings`, asked again
//! after each click) and hears which one was clicked (`native_settings_click`). Each button here is one setting
//! showing its value; a click moves it to the next value. The page keeps its values with Horizon's
//! `native_set_setting` (one small file per key in the mod's data folder) and reads them back at start.
//!
//! What the page sets is the base; environment variables (`HORIZON_STEAM_AUDIO_*`) win over it while they are set.

use std::sync::atomic::{AtomicBool, Ordering};

use crate::horizon::{Config, parse_number, parse_switch};

/// The mod's id: its row in Options > Mods and its settings folder.
pub const MOD_ID: &str = "steam_audio";

/// The mod's row in Options > Mods: title and description, in the game's language.
pub fn title() -> String {
    format!("Steam Audio {}", env!("CARGO_PKG_VERSION"))
}

pub fn description(language: &str) -> String {
    if russian(language) {
        "Бинауральный звук Steam Audio: HRTF, поглощение в воздухе, стены и реверберация. Для наушников.".into()
    } else {
        "Steam Audio's binaural sound: HRTF, air absorption, walls and reverb. Made for headphones."
            .into()
    }
}

/// One button of the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Enabled,
    AirAbsorption,
    Occlusion,
    Walls,
    Reverb,
    ReverbLevel,
    Interpolation,
    Voices,
    Reset,
}

/// The page's buttons, in order.
pub const ITEMS: [Item; 9] = [
    Item::Enabled,
    Item::AirAbsorption,
    Item::Occlusion,
    Item::Walls,
    Item::Reverb,
    Item::ReverbLevel,
    Item::Interpolation,
    Item::Voices,
    Item::Reset,
];

/// The values a click goes through.
const WALLS: [f32; 7] = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0];
const REVERB_LEVELS: [f32; 7] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
const VOICES: [usize; 5] = [8, 16, 32, 48, 64];

fn russian(language: &str) -> bool {
    let language = language.to_ascii_lowercase();
    language.starts_with("ru") || language.starts_with("be") || language.starts_with("uk")
}

fn percent(value: f32) -> String {
    format!("{}%", (value * 100.0).round() as i32)
}

/// The value after `current` among `values` (the first after the last, or after a value that isn't there).
fn next<T: Copy + PartialOrd>(values: &[T], current: T) -> T {
    values
        .iter()
        .position(|&v| v > current)
        .map_or(values[0], |i| values[i])
}

/// The labels of the page's buttons for `config`, in the game's `language` (`ru_ru`, `en_us`...).
pub fn labels(config: &Config, language: &str) -> Vec<String> {
    let ru = russian(language);
    let on_off = |on: bool| match (on, ru) {
        (true, true) => "Вкл",
        (false, true) => "Выкл",
        (true, false) => "On",
        (false, false) => "Off",
    };
    ITEMS
        .iter()
        .map(|item| match (item, ru) {
            (Item::Enabled, _) => format!("Steam Audio: {}", on_off(config.enabled)),
            (Item::AirAbsorption, true) => {
                format!("Поглощение в воздухе: {}", on_off(config.air_absorption))
            }
            (Item::AirAbsorption, false) => {
                format!("Air absorption: {}", on_off(config.air_absorption))
            }
            (Item::Occlusion, true) => format!("Стены глушат звук: {}", on_off(config.occlusion)),
            (Item::Occlusion, false) => format!("Walls muffle sound: {}", on_off(config.occlusion)),
            (Item::Walls, true) => format!("Сила стен: {}", percent(config.walls)),
            (Item::Walls, false) => format!("Wall strength: {}", percent(config.walls)),
            (Item::Reverb, true) => format!("Реверберация: {}", on_off(config.reverb)),
            (Item::Reverb, false) => format!("Reverb: {}", on_off(config.reverb)),
            (Item::ReverbLevel, true) => {
                format!("Уровень реверберации: {}", percent(config.reverb_level))
            }
            (Item::ReverbLevel, false) => format!("Reverb level: {}", percent(config.reverb_level)),
            (Item::Interpolation, true) => format!(
                "Интерполяция HRTF: {}",
                if config.bilinear {
                    "Билинейная"
                } else {
                    "Ближайшая"
                }
            ),
            (Item::Interpolation, false) => format!(
                "HRTF interpolation: {}",
                if config.bilinear {
                    "Bilinear"
                } else {
                    "Nearest"
                }
            ),
            (Item::Voices, true) => {
                format!("Бинауральных голосов: {}", config.max_binaural_voices)
            }
            (Item::Voices, false) => format!("Binaural voices: {}", config.max_binaural_voices),
            (Item::Reset, true) => "Сбросить настройки".into(),
            (Item::Reset, false) => "Reset to defaults".into(),
        })
        .collect()
}

/// What a click on button `index` does to `config`. Returns whether anything changed.
pub fn click(config: &mut Config, index: usize) -> bool {
    let Some(item) = ITEMS.get(index) else {
        return false;
    };
    let before = config.clone();
    match item {
        Item::Enabled => config.enabled = !config.enabled,
        Item::AirAbsorption => config.air_absorption = !config.air_absorption,
        Item::Occlusion => config.occlusion = !config.occlusion,
        Item::Walls => config.walls = next(&WALLS, config.walls + 1e-4),
        Item::Reverb => config.reverb = !config.reverb,
        Item::ReverbLevel => config.reverb_level = next(&REVERB_LEVELS, config.reverb_level + 1e-4),
        Item::Interpolation => config.bilinear = !config.bilinear,
        Item::Voices => config.max_binaural_voices = next(&VOICES, config.max_binaural_voices),
        Item::Reset => {
            let defaults = Config::default();
            *config = Config {
                hrtf_path: config.hrtf_path.take(),
                volume_db: config.volume_db,
                ..defaults
            };
        }
    }
    *config != before
}

/// The keys the page keeps, and their values in `config`.
pub fn values(config: &Config) -> [(&'static str, String); 8] {
    let switch = |on: bool| if on { "on" } else { "off" }.to_owned();
    [
        ("enabled", switch(config.enabled)),
        ("air_absorption", switch(config.air_absorption)),
        ("occlusion", switch(config.occlusion)),
        ("walls", config.walls.to_string()),
        ("reverb", switch(config.reverb)),
        ("reverb_level", config.reverb_level.to_string()),
        (
            "interpolation",
            if config.bilinear {
                "bilinear"
            } else {
                "nearest"
            }
            .to_owned(),
        ),
        ("voices", config.max_binaural_voices.to_string()),
    ]
}

/// Applies the kept values `read` finds; a missing or unreadable one leaves its setting as it is.
pub fn apply_saved(config: &mut Config, read: &dyn Fn(&str) -> Option<String>) {
    let switch = |key: &str, value: &mut bool| {
        if let Some(v) = read(key).as_deref().and_then(parse_switch) {
            *value = v;
        }
    };
    switch("enabled", &mut config.enabled);
    switch("air_absorption", &mut config.air_absorption);
    switch("occlusion", &mut config.occlusion);
    switch("reverb", &mut config.reverb);
    if let Some(v) = read("walls").and_then(|v| parse_number(&v, 0.0..=1.0)) {
        config.walls = v;
    }
    if let Some(v) = read("reverb_level").and_then(|v| parse_number(&v, 0.0..=2.0)) {
        config.reverb_level = v;
    }
    if let Some(v) = read("interpolation") {
        config.bilinear = v.trim().eq_ignore_ascii_case("bilinear");
    }
    if let Some(v) = read("voices").and_then(|v| v.trim().parse::<usize>().ok()) {
        config.max_binaural_voices = v.clamp(1, 256);
    }
}

static LOADED: AtomicBool = AtomicBool::new(false);

/// Loads the kept settings into the game's configuration, once: the defaults, what `read` has kept, then the
/// environment. Later calls do nothing.
pub fn load(read: &dyn Fn(&str) -> Option<String>) {
    if LOADED.swap(true, Ordering::AcqRel) {
        return;
    }
    let mut config = Config::default();
    apply_saved(&mut config, read);
    config.apply_vars(|name| std::env::var(name).ok());
    Config::set(config);
}

/// A click on button `index` of the page: changes the game's configuration and keeps it through `store`.
pub fn click_and_store(index: usize, store: &mut dyn FnMut(&str, &str)) {
    let mut config = Config::get();
    if click(&mut config, index) {
        for (key, value) in values(&config) {
            store(key, &value);
        }
        Config::set(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn labels_follow_the_language_and_the_values() {
        let config = Config::default();
        let en = labels(&config, "en_us");
        assert_eq!(en.len(), ITEMS.len());
        assert_eq!(en[0], "Steam Audio: On");
        assert_eq!(en[3], "Wall strength: 35%");
        assert_eq!(en[5], "Reverb level: 50%");
        let ru = labels(&config, "ru_ru");
        assert_eq!(ru[2], "Стены глушат звук: Вкл");
        assert_eq!(ru[7], "Бинауральных голосов: 32");
        assert!(description("ru_ru").contains("наушников"));
    }

    #[test]
    fn clicks_cycle_through_the_values() {
        let mut config = Config::default();
        assert!(click(&mut config, 0));
        assert!(!config.enabled);
        let walls: Vec<f32> = (0..7)
            .map(|_| {
                click(&mut config, 3);
                config.walls
            })
            .collect();
        assert_eq!(walls, [0.5, 0.75, 1.0, 0.0, 0.15, 0.25, 0.35]);
        click(&mut config, 7);
        assert_eq!(config.max_binaural_voices, 48);
        config.max_binaural_voices = 20;
        click(&mut config, 7);
        assert_eq!(config.max_binaural_voices, 32);
        config.max_binaural_voices = 100;
        click(&mut config, 7);
        assert_eq!(config.max_binaural_voices, 8);
        assert!(!click(&mut config, 99));
    }

    #[test]
    fn reset_keeps_the_hrtf_from_the_environment() {
        let mut config = Config {
            enabled: false,
            walls: 1.0,
            hrtf_path: Some("custom.hrtf".into()),
            volume_db: -3.0,
            ..Config::default()
        };
        assert!(click(&mut config, ITEMS.len() - 1));
        assert!(config.enabled);
        assert_eq!(config.walls, 0.35);
        assert_eq!(config.hrtf_path, Some("custom.hrtf".into()));
        assert_eq!(config.volume_db, -3.0);
    }

    #[test]
    fn kept_values_round_trip_and_bad_ones_are_ignored() {
        let mut changed = Config::default();
        click(&mut changed, 2);
        click(&mut changed, 5);
        click(&mut changed, 6);
        let kept: HashMap<&str, String> = values(&changed).into_iter().collect();
        let mut loaded = Config::default();
        apply_saved(&mut loaded, &|key| kept.get(key).cloned());
        assert_eq!(loaded, changed);

        let mut config = Config::default();
        apply_saved(&mut config, &|key| match key {
            "walls" => Some("lots".into()),
            "reverb_level" => Some("9".into()),
            "enabled" => Some("maybe".into()),
            _ => None,
        });
        assert_eq!(config.walls, 0.35);
        assert_eq!(config.reverb_level, 2.0);
        assert!(config.enabled);
    }

    #[test]
    fn environment_variables_win_over_the_page() {
        let mut config = Config::default();
        apply_saved(&mut config, &|key| (key == "reverb").then(|| "off".into()));
        config.apply_vars(|name| (name == "HORIZON_STEAM_AUDIO_REVERB").then(|| "1".into()));
        assert!(config.reverb);
    }
}
