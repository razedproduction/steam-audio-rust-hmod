//! The mod's settings page in Options > Mods.
//!
//! Horizon draws a native mod's page itself: the mod gives the labels of its buttons (`native_settings`, asked again
//! after each click) and hears which one was clicked (`native_settings_click`). Each button here is one setting
//! showing its value; a click moves it to the next value. The page keeps its values with Horizon's
//! `native_set_setting` (one small file per key in the mod's data folder) and reads them back at start.
//!
//! The page has three [`Section`]s: the main settings, the advanced ones (the sound's physics: the surfaces'
//! absorption and scattering, how far sounds carry...) and the visual debug mode. A button of the main section opens
//! each of the others, and their first button goes back; the labels Horizon asks for after a click are the new
//! section's.
//!
//! What the page sets is the base; environment variables (`HORIZON_STEAM_AUDIO_*`) win over it while they are set.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use crate::horizon::{Config, parse_number, parse_switch};

/// The mod's id: its row in Options > Mods and its settings folder.
pub const MOD_ID: &str = "steam_audio";

/// The mod's row in Options > Mods: title and description, in the game's language.
pub fn title() -> String {
    format!("Steam Audio {}", env!("CARGO_PKG_VERSION"))
}

pub fn description(language: &str) -> String {
    if russian(language) {
        "Бинауральный звук Steam Audio: HRTF, поглощение в воздухе, стены, пути вокруг них и реверберация. Для наушников."
            .into()
    } else {
        "Steam Audio's binaural sound: HRTF, air absorption, walls, paths around them and reverb. Made for headphones."
            .into()
    }
}

/// A part of the page: the buttons shown together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Section {
    Main,
    Advanced,
    Debug,
}

impl Section {
    /// Its buttons, in order.
    pub fn items(self) -> &'static [Item] {
        match self {
            Self::Main => &MAIN,
            Self::Advanced => &ADVANCED,
            Self::Debug => &DEBUG,
        }
    }
}

/// One button of the page.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Item {
    Enabled,
    SourceVolume,
    Occlusion,
    Walls,
    Pathing,
    Reverb,
    ReverbLevel,
    Underwater,
    Distance,
    OpenAdvanced,
    OpenDebug,
    Reset,
    Back,
    AirAbsorption,
    Absorption,
    Scattering,
    HearingRange,
    Interpolation,
    Voices,
    DebugSources,
    DebugPaths,
    DebugReflections,
    DebugStats,
}

const MAIN: [Item; 12] = [
    Item::Enabled,
    Item::SourceVolume,
    Item::Occlusion,
    Item::Walls,
    Item::Pathing,
    Item::Reverb,
    Item::ReverbLevel,
    Item::Underwater,
    Item::Distance,
    Item::OpenAdvanced,
    Item::OpenDebug,
    Item::Reset,
];

const ADVANCED: [Item; 7] = [
    Item::Back,
    Item::AirAbsorption,
    Item::Absorption,
    Item::Scattering,
    Item::HearingRange,
    Item::Interpolation,
    Item::Voices,
];

const DEBUG: [Item; 5] = [
    Item::Back,
    Item::DebugSources,
    Item::DebugPaths,
    Item::DebugReflections,
    Item::DebugStats,
];

/// The values a click goes through.
const WALLS: [f32; 7] = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0];
const REVERB_LEVELS: [f32; 7] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
const VOICES: [usize; 5] = [8, 16, 32, 48, 64];
/// Volume of the sounds of the world (Horizon's output limiter keeps loud places from clipping).
const SOURCE_VOLUMES: [f32; 7] = [0.25, 0.5, 0.75, 1.0, 1.1, 1.25, 1.5];
/// Factor on the surfaces' absorption: less rings longer and brighter, more is duller and shorter.
const ABSORPTIONS: [f32; 7] = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
/// The surfaces' scattering (Steam Audio's presets: 5%).
const SCATTERINGS: [f32; 6] = [0.05, 0.15, 0.3, 0.5, 0.75, 1.0];
/// How many of Horizon's ranges a sound carries with physical attenuation.
const HEARING_RANGES: [f32; 5] = [2.0, 3.0, 4.0, 6.0, 8.0];

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

/// The section shown now (not kept: a new game opens the page on the main one).
static SECTION: AtomicU8 = AtomicU8::new(Section::Main as u8);

/// The section shown now.
pub fn section() -> Section {
    match SECTION.load(Ordering::Relaxed) {
        1 => Section::Advanced,
        2 => Section::Debug,
        _ => Section::Main,
    }
}

fn set_section(section: Section) {
    SECTION.store(section as u8, Ordering::Relaxed);
}

/// The labels of the buttons of the section shown now for `config`, in the game's `language`.
pub fn labels(config: &Config, language: &str) -> Vec<String> {
    section_labels(config, section(), language)
}

/// The labels of the buttons of `section` for `config`, in the game's `language` (`ru_ru`, `en_us`...).
pub fn section_labels(config: &Config, section: Section, language: &str) -> Vec<String> {
    let ru = russian(language);
    section
        .items()
        .iter()
        .map(|&item| label(config, item, ru))
        .collect()
}

fn label(config: &Config, item: Item, ru: bool) -> String {
    let on_off = |on: bool| {
        match (on, ru) {
            (true, true) => "Вкл",
            (false, true) => "Выкл",
            (true, false) => "On",
            (false, false) => "Off",
        }
        .to_owned()
    };
    let pick = |ru_text: &str, en_text: &str| if ru { ru_text } else { en_text }.to_owned();
    let named = |ru_name: &str, en_name: &str, value: String| {
        format!("{}: {value}", if ru { ru_name } else { en_name })
    };
    match item {
        Item::Enabled => format!("Steam Audio: {}", on_off(config.enabled)),
        Item::SourceVolume => named(
            "Громкость источников",
            "Source volume",
            percent(config.source_volume),
        ),
        Item::AirAbsorption => named(
            "Поглощение в воздухе",
            "Air absorption",
            on_off(config.air_absorption),
        ),
        Item::Occlusion => named(
            "Стены глушат звук",
            "Walls muffle sound",
            on_off(config.occlusion),
        ),
        Item::Walls => named("Сила стен", "Wall strength", percent(config.walls)),
        Item::Pathing => named(
            "Звук огибает стены",
            "Sound goes round walls",
            on_off(config.pathing),
        ),
        Item::Reverb => named("Реверберация", "Reverb", on_off(config.reverb)),
        Item::ReverbLevel => named(
            "Уровень реверберации",
            "Reverb level",
            percent(config.reverb_level),
        ),
        Item::Underwater => named(
            "Глухо под водой",
            "Muffled under water",
            on_off(config.underwater),
        ),
        Item::Distance => named(
            "Затухание с расстоянием",
            "Distance attenuation",
            if config.physical_distance {
                pick("Физическое", "Physical")
            } else {
                pick("Как в Horizon", "Horizon's")
            },
        ),
        Item::Absorption => named(
            "Поглощение поверхностями",
            "Surface absorption",
            percent(config.absorption),
        ),
        Item::Scattering => named(
            "Рассеивание отражений",
            "Reflection scattering",
            percent(config.scattering),
        ),
        Item::HearingRange => {
            let value = format!("x{}", config.hearing_range);
            let value = if config.physical_distance {
                value
            } else {
                format!("{value} ({})", pick("только физическое", "physical only"))
            };
            named("Дальность слышимости", "Hearing range", value)
        }
        Item::Interpolation => named(
            "Интерполяция HRTF",
            "HRTF interpolation",
            if config.bilinear {
                pick("Билинейная", "Bilinear")
            } else {
                pick("Ближайшая", "Nearest")
            },
        ),
        Item::Voices => named(
            "Бинауральных голосов",
            "Binaural voices",
            config.max_binaural_voices.to_string(),
        ),
        Item::DebugSources => named(
            "Отладка: источники и окклюзия",
            "Debug: sources and occlusion",
            on_off(config.debug_sources),
        ),
        Item::DebugPaths => named(
            "Отладка: пути вокруг стен",
            "Debug: paths round walls",
            on_off(config.debug_paths),
        ),
        Item::DebugReflections => named(
            "Отладка: отражения и реверберация",
            "Debug: reflections and reverb",
            on_off(config.debug_reflections),
        ),
        Item::DebugStats => named(
            "Отладка: панель статистики",
            "Debug: statistics panel",
            on_off(config.debug_stats),
        ),
        Item::OpenAdvanced => pick("Расширенные настройки...", "Advanced settings..."),
        Item::OpenDebug => {
            let state = if config.debugging() {
                pick(" (вкл)", " (on)")
            } else {
                String::new()
            };
            format!("{}{state}", pick("Отладка...", "Debug..."))
        }
        Item::Back => pick("< Назад", "< Back"),
        Item::Reset => pick("Сбросить настройки", "Reset to defaults"),
    }
}

/// What a click on button `index` of `section` does: to `config` (returns whether it changed) and to the section
/// shown (`section` becomes the one to show next).
pub fn click(config: &mut Config, section: &mut Section, index: usize) -> bool {
    let Some(&item) = section.items().get(index) else {
        return false;
    };
    let before = config.clone();
    match item {
        Item::Enabled => config.enabled = !config.enabled,
        Item::SourceVolume => {
            config.source_volume = next(&SOURCE_VOLUMES, config.source_volume + 1e-4);
        }
        Item::AirAbsorption => config.air_absorption = !config.air_absorption,
        Item::Occlusion => config.occlusion = !config.occlusion,
        Item::Walls => config.walls = next(&WALLS, config.walls + 1e-4),
        Item::Pathing => config.pathing = !config.pathing,
        Item::Reverb => config.reverb = !config.reverb,
        Item::ReverbLevel => config.reverb_level = next(&REVERB_LEVELS, config.reverb_level + 1e-4),
        Item::Underwater => config.underwater = !config.underwater,
        Item::Distance => config.physical_distance = !config.physical_distance,
        Item::Absorption => config.absorption = next(&ABSORPTIONS, config.absorption + 1e-4),
        Item::Scattering => config.scattering = next(&SCATTERINGS, config.scattering + 1e-4),
        Item::HearingRange => {
            config.hearing_range = next(&HEARING_RANGES, config.hearing_range + 1e-4);
        }
        Item::Interpolation => config.bilinear = !config.bilinear,
        Item::Voices => config.max_binaural_voices = next(&VOICES, config.max_binaural_voices),
        Item::DebugSources => config.debug_sources = !config.debug_sources,
        Item::DebugPaths => config.debug_paths = !config.debug_paths,
        Item::DebugReflections => config.debug_reflections = !config.debug_reflections,
        Item::DebugStats => config.debug_stats = !config.debug_stats,
        Item::OpenAdvanced => *section = Section::Advanced,
        Item::OpenDebug => *section = Section::Debug,
        Item::Back => *section = Section::Main,
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
pub fn values(config: &Config) -> [(&'static str, String); 19] {
    let switch = |on: bool| if on { "on" } else { "off" }.to_owned();
    [
        ("enabled", switch(config.enabled)),
        ("source_volume", config.source_volume.to_string()),
        ("air_absorption", switch(config.air_absorption)),
        ("occlusion", switch(config.occlusion)),
        ("walls", config.walls.to_string()),
        ("pathing", switch(config.pathing)),
        ("reverb", switch(config.reverb)),
        ("reverb_level", config.reverb_level.to_string()),
        ("underwater", switch(config.underwater)),
        (
            "distance",
            if config.physical_distance {
                "physical"
            } else {
                "vanilla"
            }
            .to_owned(),
        ),
        ("absorption", config.absorption.to_string()),
        ("scattering", config.scattering.to_string()),
        ("hearing_range", config.hearing_range.to_string()),
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
        ("debug_sources", switch(config.debug_sources)),
        ("debug_paths", switch(config.debug_paths)),
        ("debug_reflections", switch(config.debug_reflections)),
        ("debug_stats", switch(config.debug_stats)),
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
    switch("pathing", &mut config.pathing);
    switch("reverb", &mut config.reverb);
    switch("underwater", &mut config.underwater);
    switch("debug_sources", &mut config.debug_sources);
    switch("debug_paths", &mut config.debug_paths);
    switch("debug_reflections", &mut config.debug_reflections);
    switch("debug_stats", &mut config.debug_stats);
    let number = |key: &str, range: std::ops::RangeInclusive<f32>, value: &mut f32| {
        if let Some(v) = read(key).and_then(|v| parse_number(&v, range)) {
            *value = v;
        }
    };
    number("walls", 0.0..=1.0, &mut config.walls);
    number("reverb_level", 0.0..=2.0, &mut config.reverb_level);
    number("source_volume", 0.0..=2.0, &mut config.source_volume);
    number("absorption", 0.0..=4.0, &mut config.absorption);
    number("scattering", 0.0..=1.0, &mut config.scattering);
    number("hearing_range", 2.0..=8.0, &mut config.hearing_range);
    if let Some(v) = read("distance") {
        config.physical_distance = !v.trim().eq_ignore_ascii_case("vanilla");
    }
    if let Some(v) = read("interpolation") {
        config.bilinear = v.trim().eq_ignore_ascii_case("bilinear");
    }
    if let Some(v) = read("voices").and_then(|v| v.trim().parse::<usize>().ok()) {
        config.max_binaural_voices = v.clamp(1, 256);
    }
}

static LOADED: AtomicBool = AtomicBool::new(false);

/// What the page itself has set (the defaults and the kept values, without the environment): what a click changes
/// and what is kept, so an environment variable set for one game is never kept as the page's value.
static PAGE: Mutex<Option<Config>> = Mutex::new(None);

/// The page's own configuration (before the settings are loaded, the defaults).
fn page() -> Config {
    PAGE.lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Makes `page` the page's configuration, and it with the environment over it the game's.
fn set_page(page: Config) {
    let mut config = page.clone();
    *PAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some(page);
    config.apply_vars(|name| std::env::var(name).ok());
    Config::set(config);
}

/// Loads the kept settings into the game's configuration, once: the defaults, what `read` has kept, then the
/// environment. Later calls do nothing.
pub fn load(read: &dyn Fn(&str) -> Option<String>) {
    if LOADED.swap(true, Ordering::AcqRel) {
        return;
    }
    let mut config = Config::default();
    apply_saved(&mut config, read);
    set_page(config);
}

/// A click on button `index` of the section shown: changes the page's configuration, keeps it through `store` and
/// makes it, with the environment over it, the game's; or opens another section.
pub fn click_and_store(index: usize, store: &mut dyn FnMut(&str, &str)) {
    let mut config = page();
    let mut shown = section();
    let changed = click(&mut config, &mut shown, index);
    set_section(shown);
    if changed {
        for (key, value) in values(&config) {
            store(key, &value);
        }
        set_page(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const SECTIONS: [Section; 3] = [Section::Main, Section::Advanced, Section::Debug];

    /// The index of `item` in `section`.
    fn at(section: Section, item: Item) -> usize {
        section.items().iter().position(|&i| i == item).unwrap()
    }

    /// Clicks `item`, in the section that has it.
    fn press(config: &mut Config, item: Item) -> bool {
        let section = SECTIONS
            .into_iter()
            .find(|s| s.items().contains(&item))
            .unwrap();
        let mut shown = section;
        click(config, &mut shown, at(section, item))
    }

    #[test]
    fn every_item_is_on_the_page_once() {
        let all: Vec<Item> = SECTIONS
            .iter()
            .flat_map(|s| s.items().iter().copied())
            .filter(|&i| i != Item::Back)
            .collect();
        for (i, item) in all.iter().enumerate() {
            assert!(!all[i + 1..].contains(item), "{item:?} twice");
        }
        assert_eq!(all.len(), 22);
        assert_eq!(Section::Advanced.items()[0], Item::Back);
        assert_eq!(Section::Debug.items()[0], Item::Back);
    }

    #[test]
    fn labels_follow_the_language_and_the_values() {
        let config = Config::default();
        let en = section_labels(&config, Section::Main, "en_us");
        assert_eq!(en.len(), Section::Main.items().len());
        assert_eq!(en[0], "Steam Audio: On");
        assert_eq!(en[1], "Source volume: 100%");
        assert_eq!(en[3], "Wall strength: 35%");
        assert_eq!(en[4], "Sound goes round walls: On");
        assert_eq!(en[6], "Reverb level: 50%");
        assert_eq!(en[7], "Muffled under water: On");
        assert_eq!(en[9], "Advanced settings...");
        assert_eq!(en[10], "Debug...");
        assert_eq!(en[11], "Reset to defaults");
        let ru = section_labels(&config, Section::Main, "ru_ru");
        assert_eq!(ru[1], "Громкость источников: 100%");
        assert_eq!(ru[2], "Стены глушат звук: Вкл");
        assert_eq!(ru[7], "Глухо под водой: Вкл");
        assert_eq!(ru[8], "Затухание с расстоянием: Физическое");

        let ru = section_labels(&config, Section::Advanced, "ru_ru");
        assert_eq!(ru[0], "< Назад");
        assert_eq!(ru[1], "Поглощение в воздухе: Вкл");
        assert_eq!(ru[2], "Поглощение поверхностями: 100%");
        assert_eq!(ru[3], "Рассеивание отражений: 5%");
        assert_eq!(ru[4], "Дальность слышимости: x4");
        assert_eq!(ru[6], "Бинауральных голосов: 32");
        let en = section_labels(&config, Section::Advanced, "en_us");
        assert_eq!(en[3], "Reflection scattering: 5%");
        let vanilla = Config {
            physical_distance: false,
            ..Config::default()
        };
        assert_eq!(
            section_labels(&vanilla, Section::Advanced, "en_us")[4],
            "Hearing range: x4 (physical only)"
        );

        let ru = section_labels(&config, Section::Debug, "ru_ru");
        assert_eq!(ru[1], "Отладка: источники и окклюзия: Выкл");
        let en = section_labels(&config, Section::Debug, "en_us");
        assert_eq!(en[4], "Debug: statistics panel: Off");
        let debugging = Config {
            debug_stats: true,
            ..Config::default()
        };
        assert_eq!(
            section_labels(&debugging, Section::Main, "en_us")[10],
            "Debug... (on)"
        );
        assert!(description("ru_ru").contains("наушников"));
    }

    #[test]
    fn sections_open_and_go_back() {
        let mut config = Config::default();
        let mut shown = Section::Main;
        assert!(!click(
            &mut config,
            &mut shown,
            at(Section::Main, Item::OpenAdvanced)
        ));
        assert_eq!(shown, Section::Advanced);
        assert!(!click(&mut config, &mut shown, 0));
        assert_eq!(shown, Section::Main);
        click(&mut config, &mut shown, at(Section::Main, Item::OpenDebug));
        assert_eq!(shown, Section::Debug);
        assert!(click(&mut config, &mut shown, 1));
        assert!(config.debug_sources);
        assert_eq!(shown, Section::Debug);
        assert!(!click(&mut config, &mut shown, 99));
        let expected = Config {
            debug_sources: true,
            ..Config::default()
        };
        assert_eq!(config, expected);
    }

    #[test]
    fn clicks_cycle_through_the_values() {
        let mut config = Config::default();
        assert!(press(&mut config, Item::Enabled));
        assert!(!config.enabled);
        let walls: Vec<f32> = (0..7)
            .map(|_| {
                press(&mut config, Item::Walls);
                config.walls
            })
            .collect();
        assert_eq!(walls, [0.5, 0.75, 1.0, 0.0, 0.15, 0.25, 0.35]);
        let volumes: Vec<f32> = (0..7)
            .map(|_| {
                press(&mut config, Item::SourceVolume);
                config.source_volume
            })
            .collect();
        assert_eq!(volumes, [1.1, 1.25, 1.5, 0.25, 0.5, 0.75, 1.0]);
        press(&mut config, Item::Absorption);
        assert_eq!(config.absorption, 1.5);
        press(&mut config, Item::Scattering);
        assert_eq!(config.scattering, 0.15);
        press(&mut config, Item::HearingRange);
        assert_eq!(config.hearing_range, 6.0);
        assert_eq!(config.reach(), Some(6.0));
        press(&mut config, Item::Voices);
        assert_eq!(config.max_binaural_voices, 48);
        config.max_binaural_voices = 20;
        press(&mut config, Item::Voices);
        assert_eq!(config.max_binaural_voices, 32);
        config.max_binaural_voices = 100;
        press(&mut config, Item::Voices);
        assert_eq!(config.max_binaural_voices, 8);
    }

    #[test]
    fn reset_keeps_the_hrtf_from_the_environment() {
        let mut config = Config {
            enabled: false,
            walls: 1.0,
            absorption: 3.0,
            scattering: 1.0,
            source_volume: 0.5,
            hrtf_path: Some("custom.hrtf".into()),
            volume_db: -3.0,
            ..Config::default()
        };
        config.debug_stats = true;
        assert!(press(&mut config, Item::Reset));
        assert!(config.enabled);
        assert!(!config.debugging());
        assert_eq!(config.walls, 0.35);
        assert_eq!(
            config.surfaces(),
            crate::acoustics::SurfaceTuning::default()
        );
        assert_eq!(config.source_volume, 1.0);
        assert_eq!(config.hrtf_path, Some("custom.hrtf".into()));
        assert_eq!(config.volume_db, -3.0);
    }

    #[test]
    fn kept_values_round_trip_and_bad_ones_are_ignored() {
        let mut changed = Config::default();
        for item in [
            Item::SourceVolume,
            Item::Occlusion,
            Item::Pathing,
            Item::ReverbLevel,
            Item::Underwater,
            Item::Distance,
            Item::Absorption,
            Item::Scattering,
            Item::HearingRange,
            Item::Interpolation,
            Item::DebugSources,
            Item::DebugReflections,
        ] {
            press(&mut changed, item);
        }
        assert!(changed.debug_sources && changed.debug_reflections && !changed.debug_paths);
        assert!(changed.debugging());
        let kept: HashMap<&str, String> = values(&changed).into_iter().collect();
        let mut loaded = Config::default();
        apply_saved(&mut loaded, &|key| kept.get(key).cloned());
        assert_eq!(loaded, changed);

        let mut config = Config::default();
        apply_saved(&mut config, &|key| match key {
            "walls" => Some("lots".into()),
            "reverb_level" => Some("9".into()),
            "enabled" => Some("maybe".into()),
            "scattering" => Some("-1".into()),
            "hearing_range" => Some("NaN".into()),
            _ => None,
        });
        assert_eq!(config.walls, 0.35);
        assert_eq!(config.reverb_level, 2.0);
        assert_eq!(config.scattering, 0.0);
        assert_eq!(config.hearing_range, 4.0);
        assert!(config.enabled);
    }

    #[test]
    fn environment_variables_win_over_the_page() {
        let mut config = Config::default();
        apply_saved(&mut config, &|key| (key == "reverb").then(|| "off".into()));
        config.apply_vars(|name| (name == "HORIZON_STEAM_AUDIO_REVERB").then(|| "1".into()));
        assert!(config.reverb);
        config.apply_vars(|name| (name == "HORIZON_STEAM_AUDIO_ABSORPTION").then(|| "2".into()));
        assert_eq!(config.absorption, 2.0);
    }
}
