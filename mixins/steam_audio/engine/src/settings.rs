//! The mod's settings page in Options > Mods.
//!
//! Horizon draws a native mod's page itself: the mod gives the labels of its buttons (`native_settings`, asked again
//! after each click) and hears which one was clicked (`native_settings_click`). Each button here is one setting
//! showing its value; a click moves it to the next value. The page keeps its values with Horizon's
//! `native_set_setting` (one small file per key in the mod's data folder) and reads them back at start.
//!
//! The page has [`Section`]s: the main settings, the advanced ones (the sound's physics: the surfaces' absorption and
//! scattering, how far sounds carry...), the materials, the blocks and the visual debug mode. A button of the main
//! section opens each of the others, and their first button goes back; the labels Horizon asks for after a click are
//! the new section's.
//!
//! The blocks section lists every block of the game ([`crate::materials::Catalog`]), a page at a time, filtered by a
//! search, a block tag and whether a block is tuned. The search is typed on the page itself: the mod hears the keys
//! ([`key`]) and has Horizon ask for the labels again.
//!
//! What the page sets is the base; environment variables (`HORIZON_STEAM_AUDIO_*`) win over it while they are set.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use crate::acoustics::Surface;
use crate::horizon::{Config, parse_number, parse_switch};
use crate::materials::{Assign, BlockTuning, Catalog, MaterialTuning};

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

/// What a block's page tunes: one block, or every block of a tag at once (by their indices in the catalog).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Block(usize),
    Tag(usize),
}

/// A part of the page: the buttons shown together.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Section {
    #[default]
    Main,
    Advanced,
    Debug,
    Materials,
    /// One preset (by its code).
    Material(u8),
    Blocks,
    Tags,
    Block(Target),
}

/// Where the page is: its section, and the blocks section's filter and pages.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct View {
    pub section: Section,
    /// The blocks' page, and the tags'.
    pub page: usize,
    pub tag_page: usize,
    /// What the block (or tag) ids must contain, and whether keys go to it now.
    pub query: String,
    pub typing: bool,
    /// The tag the blocks are of (its index in the catalog's tags).
    pub tag: Option<usize>,
    /// Only the blocks tuned on their own.
    pub changed_only: bool,
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
    OpenMaterials,
    OpenBlocks,
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
    // The materials.
    MaterialEntry(u8),
    ResetMaterials,
    MaterialTitle,
    MaterialAbsorption,
    MaterialIsolation,
    MaterialScattering,
    ResetMaterial,
    // The blocks.
    Search,
    ClearSearch,
    TagFilter,
    EditTag,
    ChangedOnly,
    ResetBlocks,
    PreviousPage,
    NextPage,
    BlockEntry(usize),
    NoBlocks,
    // The tags.
    AllBlocks,
    TagEntry(usize),
    // A block's (or a tag's) page.
    BlockTitle,
    BlockAssign,
    BlockAbsorption,
    BlockIsolation,
    BlockScattering,
    ResetBlock,
}

const MAIN: [Item; 14] = [
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
    Item::OpenMaterials,
    Item::OpenBlocks,
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

/// Blocks (or tags) per page.
pub const PAGE: usize = 15;

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
/// A material's or a block's absorption and isolation factors, and scattering.
const MATERIAL_ABSORPTIONS: [f32; 9] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0];
const ISOLATIONS: [f32; 8] = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0];
const MATERIAL_SCATTERINGS: [f32; 7] = [0.0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0];

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

/// The value after `current` among `None` then `values`: `None` (the material's, the global one) after the last.
fn next_optional(values: &[f32], current: Option<f32>) -> Option<f32> {
    match current {
        None => Some(values[0]),
        Some(v) => values.iter().copied().find(|&x| x > v + 1e-4),
    }
}

/// A preset's name.
pub fn surface_name(surface: Surface, ru: bool) -> &'static str {
    const RU: [&str; 14] = [
        "Обычный",
        "Кирпич",
        "Бетон",
        "Керамика",
        "Гравий",
        "Ковёр, шерсть",
        "Стекло",
        "Штукатурка",
        "Дерево",
        "Металл",
        "Камень",
        "Вода",
        "Листва",
        "Тонкое (забор, панель)",
    ];
    const EN: [&str; 14] = [
        "Generic",
        "Brick",
        "Concrete",
        "Ceramic",
        "Gravel",
        "Carpet, wool",
        "Glass",
        "Plaster",
        "Wood",
        "Metal",
        "Rock",
        "Water",
        "Foliage",
        "Thin (fence, pane)",
    ];
    let i = usize::from(surface.base().code()).min(13);
    if ru { RU[i] } else { EN[i] }
}

/// A block's id without `minecraft:`.
fn short(name: &str) -> &str {
    name.strip_prefix("minecraft:").unwrap_or(name)
}

/// Whether block or tag id `name` matches the search `query` (a space or `-` typed is `_`).
fn matches(name: &str, query: &str) -> bool {
    query.is_empty() || name.contains(query)
}

/// The blocks the blocks section lists for `view`: indices in the catalog.
pub fn listed_blocks(config: &Config, view: &View, catalog: &Catalog) -> Vec<usize> {
    let in_tag = view.tag.and_then(|t| catalog.tags.get(t)).map(|(_, b)| b);
    (0..catalog.blocks.len())
        .filter(|&i| in_tag.is_none_or(|b| b.binary_search(&i).is_ok()))
        .filter(|&i| matches(&catalog.blocks[i].name, &view.query))
        .filter(|&i| {
            !view.changed_only || config.tuning.blocks.contains_key(&catalog.blocks[i].name)
        })
        .collect()
}

/// The tags the tags section lists for `view`: indices in the catalog's tags.
pub fn listed_tags(view: &View, catalog: &Catalog) -> Vec<usize> {
    (0..catalog.tags.len())
        .filter(|&t| matches(&catalog.tags[t].0, &view.query))
        .collect()
}

fn pages(count: usize) -> usize {
    count.div_ceil(PAGE).max(1)
}

/// The blocks of `target`.
fn target_blocks(target: Target, catalog: &Catalog) -> Vec<usize> {
    match target {
        Target::Block(i) => (i < catalog.blocks.len())
            .then_some(i)
            .into_iter()
            .collect(),
        Target::Tag(t) => catalog
            .tags
            .get(t)
            .map(|(_, b)| b.clone())
            .unwrap_or_default(),
    }
}

/// The buttons of the section `view` is at.
pub fn items(config: &Config, view: &View, catalog: &Catalog) -> Vec<Item> {
    match view.section {
        Section::Main => MAIN.to_vec(),
        Section::Advanced => ADVANCED.to_vec(),
        Section::Debug => DEBUG.to_vec(),
        Section::Materials => std::iter::once(Item::Back)
            .chain(Surface::ALL.iter().map(|s| Item::MaterialEntry(s.code())))
            .chain(std::iter::once(Item::ResetMaterials))
            .collect(),
        Section::Material(_) => vec![
            Item::Back,
            Item::MaterialTitle,
            Item::MaterialAbsorption,
            Item::MaterialIsolation,
            Item::MaterialScattering,
            Item::ResetMaterial,
        ],
        Section::Blocks => {
            let mut items = vec![Item::Back, Item::Search];
            if !view.query.is_empty() {
                items.push(Item::ClearSearch);
            }
            items.push(Item::TagFilter);
            if view.tag.is_some() {
                items.push(Item::EditTag);
            }
            items.push(Item::ChangedOnly);
            if !config.tuning.blocks.is_empty() {
                items.push(Item::ResetBlocks);
            }
            let listed = listed_blocks(config, view, catalog);
            if listed.len() > PAGE {
                items.push(Item::PreviousPage);
                items.push(Item::NextPage);
            }
            let page = view.page.min(pages(listed.len()) - 1);
            let shown: Vec<Item> = listed
                .iter()
                .skip(page * PAGE)
                .take(PAGE)
                .map(|&i| Item::BlockEntry(i))
                .collect();
            if shown.is_empty() {
                items.push(Item::NoBlocks);
            }
            items.extend(shown);
            items
        }
        Section::Tags => {
            let mut items = vec![Item::Back, Item::Search];
            if !view.query.is_empty() {
                items.push(Item::ClearSearch);
            }
            items.push(Item::AllBlocks);
            let listed = listed_tags(view, catalog);
            if listed.len() > PAGE {
                items.push(Item::PreviousPage);
                items.push(Item::NextPage);
            }
            let page = view.tag_page.min(pages(listed.len()) - 1);
            items.extend(
                listed
                    .iter()
                    .skip(page * PAGE)
                    .take(PAGE)
                    .map(|&t| Item::TagEntry(t)),
            );
            items
        }
        Section::Block(_) => vec![
            Item::Back,
            Item::BlockTitle,
            Item::BlockAssign,
            Item::BlockAbsorption,
            Item::BlockIsolation,
            Item::BlockScattering,
            Item::ResetBlock,
        ],
    }
}

/// The page's state now: where it is (not kept: a new game opens it on the main section).
static VIEW: Mutex<Option<View>> = Mutex::new(None);

/// Where the page is now.
pub fn view() -> View {
    VIEW.lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

fn set_view(view: View) {
    *VIEW.lock().unwrap_or_else(|e| e.into_inner()) = Some(view);
}

/// The labels of the buttons of the section shown now for `config`, in the game's `language`.
pub fn labels(config: &Config, language: &str) -> Vec<String> {
    view_labels(config, &view(), &crate::materials::catalog(), language)
}

/// The labels of the buttons of `view`'s section for `config`, in the game's `language` (`ru_ru`, `en_us`...).
pub fn view_labels(config: &Config, view: &View, catalog: &Catalog, language: &str) -> Vec<String> {
    let ru = russian(language);
    items(config, view, catalog)
        .into_iter()
        .map(|item| label(config, view, catalog, item, ru))
        .collect()
}

/// A block's material as the page shows it: what it is assigned, or what it gets by itself.
fn assign_label(b: &BlockTuning, auto: Option<Surface>, ru: bool) -> String {
    let open = if ru { "не стена" } else { "not a wall" };
    match b.assign {
        Assign::Auto => {
            let what = auto.map_or(open, |s| surface_name(s, ru));
            if ru {
                format!("Авто ({what})")
            } else {
                format!("Auto ({what})")
            }
        }
        Assign::Open => {
            if ru {
                "Нет (не стена)".into()
            } else {
                "None (not a wall)".into()
            }
        }
        Assign::As(s) => surface_name(s, ru).into(),
    }
}

fn label(config: &Config, view: &View, catalog: &Catalog, item: Item, ru: bool) -> String {
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
    let optional = |value: Option<f32>, none_ru: &str, none_en: &str| {
        value.map_or_else(|| pick(none_ru, none_en), percent)
    };
    // The page of a block (or a tag): its first block's tuning stands for all.
    let target = match view.section {
        Section::Block(t) => Some(t),
        _ => None,
    };
    let first = target
        .and_then(|t| target_blocks(t, catalog).first().copied())
        .map(|i| &catalog.blocks[i]);
    let block = first.map_or_else(BlockTuning::default, |b| config.tuning.block(&b.name));
    let material = match view.section {
        Section::Material(code) => Surface::from_code(code),
        _ => Surface::Generic,
    };
    let mt = config.tuning.material(material);
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
        Item::OpenMaterials => {
            let tuned = config
                .tuning
                .materials
                .iter()
                .filter(|m| **m != MaterialTuning::default())
                .count();
            let base = pick("Материалы...", "Materials...");
            if tuned > 0 {
                format!("{base} ({}: {tuned})", pick("изменено", "changed"))
            } else {
                base
            }
        }
        Item::OpenBlocks => {
            let tuned = config.tuning.blocks.len();
            let base = pick("Блоки...", "Blocks...");
            if tuned > 0 {
                format!("{base} ({}: {tuned})", pick("изменено", "changed"))
            } else {
                base
            }
        }
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

        Item::MaterialEntry(code) => {
            let s = Surface::from_code(code);
            let m = config.tuning.material(s);
            let star = if m == MaterialTuning::default() {
                ""
            } else {
                " *"
            };
            let scattering = optional(m.scattering, "общее", "global");
            if ru {
                format!(
                    "{}: поглощение {}, изоляция {}, рассеивание {scattering}{star}",
                    surface_name(s, ru),
                    percent(m.absorption),
                    percent(m.isolation)
                )
            } else {
                format!(
                    "{}: absorption {}, isolation {}, scattering {scattering}{star}",
                    surface_name(s, ru),
                    percent(m.absorption),
                    percent(m.isolation)
                )
            }
        }
        Item::ResetMaterials => pick("Сбросить все материалы", "Reset all materials"),
        Item::MaterialTitle => named("Материал", "Material", surface_name(material, ru).into()),
        Item::MaterialAbsorption => named(
            "Поглощение (меньше - звонче эхо)",
            "Absorption (less: brighter echo)",
            percent(mt.absorption),
        ),
        Item::MaterialIsolation => named(
            "Звукоизоляция (меньше - пропускает)",
            "Isolation (less: lets through)",
            percent(mt.isolation),
        ),
        Item::MaterialScattering => named(
            "Рассеивание",
            "Scattering",
            optional(
                mt.scattering,
                "общее (расширенные настройки)",
                "global (advanced settings)",
            ),
        ),
        Item::ResetMaterial => pick("Сбросить материал", "Reset the material"),

        Item::Search => {
            let query = if view.typing {
                format!("{}_", view.query)
            } else if view.query.is_empty() {
                pick("(нажмите и печатайте)", "(click and type)")
            } else {
                view.query.clone()
            };
            let hint = if view.typing {
                pick(" - Enter: готово", " - Enter: done")
            } else {
                String::new()
            };
            format!("{}: {query}{hint}", pick("Поиск", "Search"))
        }
        Item::ClearSearch => pick("Очистить поиск", "Clear the search"),
        Item::TagFilter => {
            let tag = view
                .tag
                .and_then(|t| catalog.tags.get(t))
                .map_or_else(|| pick("все блоки", "all blocks"), |(t, _)| format!("#{t}"));
            named("Тег", "Tag", format!("{tag}..."))
        }
        Item::EditTag => {
            let (tag, count) = view
                .tag
                .and_then(|t| catalog.tags.get(t))
                .map_or(("", 0), |(t, b)| (short(t), b.len()));
            if ru {
                format!("Настроить весь тег #{tag} ({count} бл.)...")
            } else {
                format!("Tune the whole tag #{tag} ({count} blocks)...")
            }
        }
        Item::ChangedOnly => named(
            "Только изменённые",
            "Changed only",
            on_off(view.changed_only),
        ),
        Item::ResetBlocks => {
            let count = config.tuning.blocks.len();
            if ru {
                format!("Сбросить все блоки ({count})")
            } else {
                format!("Reset all blocks ({count})")
            }
        }
        Item::PreviousPage | Item::NextPage => {
            let (page, count) = match view.section {
                Section::Tags => (view.tag_page, listed_tags(view, catalog).len()),
                _ => (view.page, listed_blocks(config, view, catalog).len()),
            };
            let total = pages(count);
            let page = page.min(total - 1) + 1;
            if item == Item::PreviousPage {
                format!(
                    "< {} ({page}/{total})",
                    pick("Предыдущая страница", "Previous page")
                )
            } else {
                format!(
                    "{} ({page}/{total}) >",
                    pick("Следующая страница", "Next page")
                )
            }
        }
        Item::NoBlocks => {
            if catalog.blocks.is_empty() {
                pick(
                    "(блоки появятся, когда загрузится мир)",
                    "(the blocks show once a world is loaded)",
                )
            } else {
                pick("(ничего не найдено)", "(nothing found)")
            }
        }
        Item::BlockEntry(i) => {
            let Some(b) = catalog.blocks.get(i) else {
                return String::new();
            };
            let t = config.tuning.block(&b.name);
            let star = if t.is_default() { "" } else { " *" };
            format!("{}: {}{star}", short(&b.name), assign_label(&t, b.auto, ru))
        }
        Item::AllBlocks => pick("Все блоки (без тега)", "All blocks (no tag)"),
        Item::TagEntry(t) => catalog.tags.get(t).map_or_else(String::new, |(tag, b)| {
            if ru {
                format!("#{tag} ({} бл.)", b.len())
            } else {
                format!("#{tag} ({} blocks)", b.len())
            }
        }),
        Item::BlockTitle => match target {
            Some(Target::Tag(t)) => catalog.tags.get(t).map_or_else(String::new, |(tag, b)| {
                if ru {
                    format!("Тег #{tag}: {} бл. (меняются все сразу)", b.len())
                } else {
                    format!("Tag #{tag}: {} blocks (all change at once)", b.len())
                }
            }),
            _ => first.map_or_else(String::new, |b| named("Блок", "Block", b.name.clone())),
        },
        Item::BlockAssign => named(
            "Материал",
            "Material",
            assign_label(&block, first.and_then(|b| b.auto), ru),
        ),
        Item::BlockAbsorption => named(
            "Поглощение",
            "Absorption",
            optional(block.absorption, "как у материала", "the material's"),
        ),
        Item::BlockIsolation => named(
            "Звукоизоляция",
            "Isolation",
            optional(block.isolation, "как у материала", "the material's"),
        ),
        Item::BlockScattering => named(
            "Рассеивание",
            "Scattering",
            optional(block.scattering, "как у материала", "the material's"),
        ),
        Item::ResetBlock => match target {
            Some(Target::Tag(_)) => pick("Сбросить блоки тега", "Reset the tag's blocks"),
            _ => pick("Сбросить блок", "Reset the block"),
        },
    }
}

/// The material after `current` a block's page goes to: Auto, None, then the presets.
fn next_assign(current: Assign) -> Assign {
    match current {
        Assign::Auto => Assign::Open,
        Assign::Open => Assign::As(Surface::ALL[0]),
        Assign::As(s) => {
            let i = usize::from(s.base().code()) + 1;
            Surface::ALL.get(i).map_or(Assign::Auto, |&s| Assign::As(s))
        }
    }
}

/// What a click on button `index` of `view`'s section does: to `config` (returns whether it changed) and to `view`
/// (the section and filter to show next).
pub fn click(config: &mut Config, view: &mut View, catalog: &Catalog, index: usize) -> bool {
    let Some(item) = items(config, view, catalog).get(index).copied() else {
        return false;
    };
    let before = config.clone();
    if item != Item::Search {
        view.typing = false;
    }
    let target = match view.section {
        Section::Block(t) => Some(t),
        _ => None,
    };
    // A change to a block's tuning, for every block of the page's target, from its first block's.
    let tune_blocks = |config: &mut Config, change: &dyn Fn(BlockTuning) -> BlockTuning| {
        let Some(target) = target else {
            return;
        };
        let blocks = target_blocks(target, catalog);
        let Some(&first) = blocks.first() else {
            return;
        };
        let new = change(config.tuning.block(&catalog.blocks[first].name));
        let tuning = Arc::make_mut(&mut config.tuning);
        for i in blocks {
            let name = &catalog.blocks[i].name;
            let mut b = tuning.block(name);
            b.assign = new.assign;
            b.absorption = new.absorption;
            b.isolation = new.isolation;
            b.scattering = new.scattering;
            tuning.set_block(name, b);
        }
    };
    let material = match view.section {
        Section::Material(code) => Some(Surface::from_code(code)),
        _ => None,
    };
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
        Item::OpenAdvanced => view.section = Section::Advanced,
        Item::OpenMaterials => view.section = Section::Materials,
        Item::OpenBlocks => view.section = Section::Blocks,
        Item::OpenDebug => view.section = Section::Debug,
        Item::Back => {
            view.section = match view.section {
                Section::Material(_) => Section::Materials,
                Section::Tags | Section::Block(_) => Section::Blocks,
                _ => Section::Main,
            }
        }
        Item::Reset => {
            // Everything but the HRTF from the environment and the materials and blocks (they have resets of their
            // own).
            let defaults = Config::default();
            *config = Config {
                hrtf_path: config.hrtf_path.take(),
                volume_db: config.volume_db,
                tuning: Arc::clone(&config.tuning),
                ..defaults
            };
        }

        Item::MaterialEntry(code) => view.section = Section::Material(code),
        Item::ResetMaterials => {
            Arc::make_mut(&mut config.tuning).materials = Default::default();
        }
        Item::MaterialTitle | Item::BlockTitle | Item::NoBlocks => {}
        Item::MaterialAbsorption
        | Item::MaterialIsolation
        | Item::MaterialScattering
        | Item::ResetMaterial => {
            if let Some(s) = material {
                let m = Arc::make_mut(&mut config.tuning).material_mut(s);
                match item {
                    Item::MaterialAbsorption => {
                        m.absorption = next(&MATERIAL_ABSORPTIONS, m.absorption + 1e-4);
                    }
                    Item::MaterialIsolation => m.isolation = next(&ISOLATIONS, m.isolation + 1e-4),
                    Item::MaterialScattering => {
                        m.scattering = next_optional(&MATERIAL_SCATTERINGS, m.scattering);
                    }
                    _ => *m = MaterialTuning::default(),
                }
            }
        }

        Item::Search => {
            view.typing = !view.typing;
        }
        Item::ClearSearch => {
            view.query.clear();
            view.page = 0;
            view.tag_page = 0;
        }
        Item::TagFilter => view.section = Section::Tags,
        Item::EditTag => {
            if let Some(t) = view.tag {
                view.section = Section::Block(Target::Tag(t));
            }
        }
        Item::ChangedOnly => {
            view.changed_only = !view.changed_only;
            view.page = 0;
        }
        Item::ResetBlocks => Arc::make_mut(&mut config.tuning).blocks.clear(),
        Item::PreviousPage | Item::NextPage => {
            let tags = view.section == Section::Tags;
            let count = if tags {
                listed_tags(view, catalog).len()
            } else {
                listed_blocks(config, view, catalog).len()
            };
            let page = if tags {
                &mut view.tag_page
            } else {
                &mut view.page
            };
            let total = pages(count);
            let current = (*page).min(total - 1);
            *page = if item == Item::NextPage {
                (current + 1) % total
            } else {
                (current + total - 1) % total
            };
        }
        Item::BlockEntry(i) => view.section = Section::Block(Target::Block(i)),
        Item::AllBlocks => {
            view.tag = None;
            view.page = 0;
            view.query.clear();
            view.section = Section::Blocks;
        }
        Item::TagEntry(t) => {
            view.tag = Some(t);
            view.page = 0;
            view.query.clear();
            view.section = Section::Blocks;
        }
        Item::BlockAssign => tune_blocks(config, &|b| BlockTuning {
            assign: next_assign(b.assign),
            ..b
        }),
        Item::BlockAbsorption => tune_blocks(config, &|b| BlockTuning {
            absorption: next_optional(&MATERIAL_ABSORPTIONS, b.absorption),
            ..b
        }),
        Item::BlockIsolation => tune_blocks(config, &|b| BlockTuning {
            isolation: next_optional(&ISOLATIONS, b.isolation),
            ..b
        }),
        Item::BlockScattering => tune_blocks(config, &|b| BlockTuning {
            scattering: next_optional(&MATERIAL_SCATTERINGS, b.scattering),
            ..b
        }),
        Item::ResetBlock => tune_blocks(config, &|_| BlockTuning::default()),
    }
    *config != before
}

/// A key the player pressed (`key.keyboard.a`...) while `view` searches: a letter, digit, `_` (space or `-`) or
/// `:` (`;`) goes into the search, Backspace takes one out, Enter or Escape ends it. Returns whether `view` changed.
pub fn type_key(view: &mut View, key: &str) -> bool {
    if !view.typing {
        return false;
    }
    let Some(name) = key.strip_prefix("key.keyboard.") else {
        return false;
    };
    match name {
        "enter" | "keypad.enter" | "escape" => view.typing = false,
        "backspace" => {
            if view.query.pop().is_none() {
                return false;
            }
        }
        "space" | "minus" => view.query.push('_'),
        "semicolon" => view.query.push(':'),
        _ => {
            let mut chars = name.chars();
            match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphanumeric() => {
                    view.query.push(c.to_ascii_lowercase());
                }
                _ => return false,
            }
        }
    }
    view.page = 0;
    view.tag_page = 0;
    true
}

/// A key the player pressed: into the page's search when it is being typed. Returns whether the page's labels
/// changed (Horizon should ask for them again).
pub fn key(name: &str) -> bool {
    let mut view = view();
    let changed = type_key(&mut view, name);
    if changed {
        set_view(view);
    }
    changed
}

/// Whether the page's search is being typed.
pub fn typing() -> bool {
    view().typing
}

/// Ends the search being typed (the player went back to the game).
pub fn stop_typing() {
    let mut view = view();
    if view.typing {
        view.typing = false;
        set_view(view);
    }
}

/// The keys the page keeps, and their values in `config`.
pub fn values(config: &Config) -> [(&'static str, String); 21] {
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
        ("materials", config.tuning.format_materials()),
        ("blocks", config.tuning.format_blocks()),
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
    let materials = read("materials");
    let blocks = read("blocks");
    if materials.is_some() || blocks.is_some() {
        let tuning = Arc::make_mut(&mut config.tuning);
        if let Some(v) = materials {
            tuning.parse_materials(&v);
        }
        if let Some(v) = blocks {
            tuning.parse_blocks(&v);
        }
    }
}

static LOADED: AtomicBool = AtomicBool::new(false);

/// What the page itself has set (the defaults and the kept values, without the environment): what a click changes
/// and what is kept, so an environment variable set for one game is never kept as the page's value.
static PAGE_CONFIG: Mutex<Option<Config>> = Mutex::new(None);

/// The page's own configuration (before the settings are loaded, the defaults).
fn page() -> Config {
    PAGE_CONFIG
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .unwrap_or_default()
}

/// Makes `page` the page's configuration, and it with the environment over it the game's.
fn set_page(page: Config) {
    let mut config = page.clone();
    *PAGE_CONFIG.lock().unwrap_or_else(|e| e.into_inner()) = Some(page);
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
    let mut shown = view();
    let changed = click(&mut config, &mut shown, &crate::materials::catalog(), index);
    set_view(shown);
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
    use crate::materials::CatalogBlock;
    use std::collections::HashMap;

    fn catalog() -> Catalog {
        let blocks = (0..40)
            .map(|i| CatalogBlock {
                name: format!("minecraft:stone_{i:02}"),
                auto: Some(Surface::Rock),
            })
            .chain([
                CatalogBlock {
                    name: "minecraft:oak_planks".into(),
                    auto: Some(Surface::Wood),
                },
                CatalogBlock {
                    name: "minecraft:oak_log".into(),
                    auto: Some(Surface::Wood),
                },
                CatalogBlock {
                    name: "minecraft:poppy".into(),
                    auto: None,
                },
            ]);
        Catalog::new(
            blocks,
            [
                (
                    "minecraft:logs".to_owned(),
                    vec!["minecraft:oak_log".to_owned()],
                ),
                (
                    "minecraft:planks_and_logs".to_owned(),
                    vec!["oak_planks".to_owned(), "oak_log".to_owned()],
                ),
            ],
        )
    }

    /// The view at `section`.
    fn at(section: Section) -> View {
        View {
            section,
            ..View::default()
        }
    }

    /// The index of `item` in `view`'s section.
    fn index(config: &Config, view: &View, catalog: &Catalog, item: Item) -> usize {
        items(config, view, catalog)
            .iter()
            .position(|&i| i == item)
            .unwrap_or_else(|| panic!("{item:?} not in {:?}", view.section))
    }

    /// Clicks `item` of `view`'s section.
    fn press_in(config: &mut Config, view: &mut View, catalog: &Catalog, item: Item) -> bool {
        let i = index(config, view, catalog, item);
        click(config, view, catalog, i)
    }

    /// Clicks `item`, in the plain section that has it.
    fn press(config: &mut Config, item: Item) -> bool {
        let section = [Section::Main, Section::Advanced, Section::Debug]
            .into_iter()
            .find(|&s| items(config, &at(s), &Catalog::default()).contains(&item))
            .unwrap();
        press_in(config, &mut at(section), &Catalog::default(), item)
    }

    fn labels_of(config: &Config, view: &View, catalog: &Catalog, language: &str) -> Vec<String> {
        view_labels(config, view, catalog, language)
    }

    #[test]
    fn labels_follow_the_language_and_the_values() {
        let config = Config::default();
        let c = Catalog::default();
        let en = labels_of(&config, &at(Section::Main), &c, "en_us");
        assert_eq!(en.len(), MAIN.len());
        assert_eq!(en[0], "Steam Audio: On");
        assert_eq!(en[1], "Source volume: 100%");
        assert_eq!(en[3], "Wall strength: 35%");
        assert_eq!(en[4], "Sound goes round walls: On");
        assert_eq!(en[6], "Reverb level: 50%");
        assert_eq!(en[7], "Muffled under water: On");
        assert_eq!(en[9], "Advanced settings...");
        assert_eq!(en[10], "Materials...");
        assert_eq!(en[11], "Blocks...");
        assert_eq!(en[12], "Debug...");
        assert_eq!(en[13], "Reset to defaults");
        let ru = labels_of(&config, &at(Section::Main), &c, "ru_ru");
        assert_eq!(ru[1], "Громкость источников: 100%");
        assert_eq!(ru[2], "Стены глушат звук: Вкл");
        assert_eq!(ru[8], "Затухание с расстоянием: Физическое");
        assert_eq!(ru[10], "Материалы...");
        assert_eq!(ru[11], "Блоки...");

        let ru = labels_of(&config, &at(Section::Advanced), &c, "ru_ru");
        assert_eq!(ru[0], "< Назад");
        assert_eq!(ru[2], "Поглощение поверхностями: 100%");
        assert_eq!(ru[3], "Рассеивание отражений: 5%");
        assert_eq!(ru[4], "Дальность слышимости: x4");
        let vanilla = Config {
            physical_distance: false,
            ..Config::default()
        };
        assert_eq!(
            labels_of(&vanilla, &at(Section::Advanced), &c, "en_us")[4],
            "Hearing range: x4 (physical only)"
        );
        let en = labels_of(&config, &at(Section::Debug), &c, "en_us");
        assert_eq!(en[4], "Debug: statistics panel: Off");
        let debugging = Config {
            debug_stats: true,
            ..Config::default()
        };
        assert_eq!(
            labels_of(&debugging, &at(Section::Main), &c, "en_us")[12],
            "Debug... (on)"
        );
        assert!(description("ru_ru").contains("наушников"));
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
        assert_eq!(config.reach(), Some(6.0));
        press(&mut config, Item::Voices);
        assert_eq!(config.max_binaural_voices, 48);
        config.max_binaural_voices = 100;
        press(&mut config, Item::Voices);
        assert_eq!(config.max_binaural_voices, 8);
    }

    #[test]
    fn sections_open_and_go_back() {
        let c = catalog();
        let mut config = Config::default();
        let mut view = View::default();
        for (open, section) in [
            (Item::OpenAdvanced, Section::Advanced),
            (Item::OpenMaterials, Section::Materials),
            (Item::OpenBlocks, Section::Blocks),
            (Item::OpenDebug, Section::Debug),
        ] {
            assert!(!press_in(&mut config, &mut view, &c, open));
            assert_eq!(view.section, section);
            assert!(!press_in(&mut config, &mut view, &c, Item::Back));
            assert_eq!(view.section, Section::Main);
        }
        // Material page > Back: the materials; block page > Back: the blocks.
        view.section = Section::Materials;
        press_in(
            &mut config,
            &mut view,
            &c,
            Item::MaterialEntry(Surface::Rock.code()),
        );
        assert_eq!(view.section, Section::Material(Surface::Rock.code()));
        press_in(&mut config, &mut view, &c, Item::Back);
        assert_eq!(view.section, Section::Materials);
        view.section = Section::Blocks;
        press_in(&mut config, &mut view, &c, Item::BlockEntry(0));
        assert_eq!(view.section, Section::Block(Target::Block(0)));
        press_in(&mut config, &mut view, &c, Item::Back);
        assert_eq!(view.section, Section::Blocks);
        assert!(!click(&mut config, &mut view, &c, 99));
        assert_eq!(config, Config::default());
    }

    #[test]
    fn materials_are_tuned_on_their_pages() {
        let c = catalog();
        let mut config = Config::default();
        let mut view = at(Section::Material(Surface::Rock.code()));
        let labels = labels_of(&config, &view, &c, "ru_ru");
        assert_eq!(labels[1], "Материал: Камень");
        assert_eq!(labels[2], "Поглощение (меньше - звонче эхо): 100%");
        assert_eq!(labels[4], "Рассеивание: общее (расширенные настройки)");
        assert!(press_in(
            &mut config,
            &mut view,
            &c,
            Item::MaterialAbsorption
        ));
        assert!(press_in(
            &mut config,
            &mut view,
            &c,
            Item::MaterialIsolation
        ));
        assert!(press_in(
            &mut config,
            &mut view,
            &c,
            Item::MaterialScattering
        ));
        let rock = config.tuning.material(Surface::Rock);
        assert_eq!(
            (rock.absorption, rock.isolation, rock.scattering),
            (1.5, 1.5, Some(0.0))
        );
        // Scattering: the global one after the last.
        for _ in 0..6 {
            press_in(&mut config, &mut view, &c, Item::MaterialScattering);
        }
        assert_eq!(config.tuning.material(Surface::Rock).scattering, Some(1.0));
        press_in(&mut config, &mut view, &c, Item::MaterialScattering);
        assert_eq!(config.tuning.material(Surface::Rock).scattering, None);
        let materials = labels_of(&config, &at(Section::Materials), &c, "en_us");
        assert_eq!(materials.len(), 16);
        assert_eq!(
            materials[1 + usize::from(Surface::Rock.code())],
            "Rock: absorption 150%, isolation 150%, scattering global *"
        );
        assert_eq!(
            labels_of(&config, &at(Section::Main), &c, "en_us")[10],
            "Materials... (changed: 1)"
        );
        assert!(press_in(&mut config, &mut view, &c, Item::ResetMaterial));
        assert_eq!(
            config.tuning.material(Surface::Rock),
            MaterialTuning::default()
        );
    }

    #[test]
    fn blocks_are_listed_a_page_at_a_time_and_filtered() {
        let c = catalog();
        let mut config = Config::default();
        let mut view = at(Section::Blocks);
        let en = labels_of(&config, &view, &c, "en_us");
        assert_eq!(en[0], "< Back");
        assert_eq!(en[1], "Search: (click and type)");
        assert_eq!(en[2], "Tag: all blocks...");
        assert_eq!(en[3], "Changed only: Off");
        assert_eq!(en[4], "< Previous page (1/3)");
        assert_eq!(en[5], "Next page (1/3) >");
        assert_eq!(en[6], "oak_log: Auto (Wood)");
        assert_eq!(en.len(), 6 + PAGE);
        press_in(&mut config, &mut view, &c, Item::NextPage);
        press_in(&mut config, &mut view, &c, Item::NextPage);
        assert_eq!(view.page, 2);
        let last = labels_of(&config, &view, &c, "en_us");
        assert_eq!(last.len(), 6 + 13);
        press_in(&mut config, &mut view, &c, Item::NextPage);
        assert_eq!(view.page, 0);
        press_in(&mut config, &mut view, &c, Item::PreviousPage);
        assert_eq!(view.page, 2);

        // Typing a search.
        assert!(!press_in(&mut config, &mut view, &c, Item::Search));
        assert!(view.typing);
        for key in [
            "key.keyboard.o",
            "key.keyboard.a",
            "key.keyboard.k",
            "key.keyboard.minus",
        ] {
            assert!(type_key(&mut view, key));
        }
        assert_eq!(view.query, "oak_");
        assert_eq!(view.page, 0);
        assert!(!type_key(&mut view, "key.keyboard.left.shift"));
        assert!(!type_key(&mut view, "key.mouse.left"));
        let en = labels_of(&config, &view, &c, "en_us");
        assert_eq!(en[1], "Search: oak__ - Enter: done");
        assert_eq!(en[2], "Clear the search");
        assert_eq!(
            &en[5..],
            ["oak_log: Auto (Wood)", "oak_planks: Auto (Wood)"]
        );
        assert!(type_key(&mut view, "key.keyboard.backspace"));
        assert!(type_key(&mut view, "key.keyboard.enter"));
        assert!(!view.typing);
        assert!(!type_key(&mut view, "key.keyboard.x"));
        let ru = labels_of(&config, &view, &c, "ru_ru");
        assert_eq!(ru[1], "Поиск: oak");
        view.query = "nothing".into();
        assert_eq!(
            labels_of(&config, &view, &c, "ru_ru").last().unwrap(),
            "(ничего не найдено)"
        );
        press_in(&mut config, &mut view, &c, Item::ClearSearch);
        assert!(view.query.is_empty());

        // A tag.
        press_in(&mut config, &mut view, &c, Item::TagFilter);
        assert_eq!(view.section, Section::Tags);
        let tags = labels_of(&config, &view, &c, "en_us");
        assert_eq!(
            &tags[2..],
            [
                "All blocks (no tag)",
                "#minecraft:logs (1 blocks)",
                "#minecraft:planks_and_logs (2 blocks)"
            ]
        );
        press_in(&mut config, &mut view, &c, Item::TagEntry(1));
        assert_eq!(view.section, Section::Blocks);
        let en = labels_of(&config, &view, &c, "en_us");
        assert_eq!(en[2], "Tag: #minecraft:planks_and_logs...");
        assert_eq!(en[3], "Tune the whole tag #planks_and_logs (2 blocks)...");
        assert_eq!(
            &en[5..],
            ["oak_log: Auto (Wood)", "oak_planks: Auto (Wood)"]
        );

        // Changed only.
        let poppy = c.find("minecraft:poppy").unwrap();
        Arc::make_mut(&mut config.tuning).set_block(
            "minecraft:poppy",
            BlockTuning {
                assign: Assign::As(Surface::Carpet),
                ..BlockTuning::default()
            },
        );
        view.tag = None;
        press_in(&mut config, &mut view, &c, Item::ChangedOnly);
        assert_eq!(listed_blocks(&config, &view, &c), [poppy]);
        let en = labels_of(&config, &view, &c, "en_us");
        assert_eq!(en.last().unwrap(), "poppy: Carpet, wool *");
        assert_eq!(en[4], "Reset all blocks (1)");
        assert!(press_in(&mut config, &mut view, &c, Item::ResetBlocks));
        assert!(config.tuning.blocks.is_empty());
        assert_eq!(
            labels_of(
                &Config::default(),
                &at(Section::Blocks),
                &Catalog::default(),
                "en_us"
            )
            .last()
            .unwrap(),
            "(the blocks show once a world is loaded)"
        );
    }

    #[test]
    fn a_block_and_a_whole_tag_are_tuned_on_their_page() {
        let c = catalog();
        let planks = c.find("minecraft:oak_planks").unwrap();
        let mut config = Config::default();
        let mut view = at(Section::Block(Target::Block(planks)));
        let ru = labels_of(&config, &view, &c, "ru_ru");
        assert_eq!(ru[1], "Блок: minecraft:oak_planks");
        assert_eq!(ru[2], "Материал: Авто (Дерево)");
        assert_eq!(ru[3], "Поглощение: как у материала");
        // Material: Auto > None > Generic > Brick...
        assert!(press_in(&mut config, &mut view, &c, Item::BlockAssign));
        assert_eq!(
            config.tuning.block("minecraft:oak_planks").assign,
            Assign::Open
        );
        press_in(&mut config, &mut view, &c, Item::BlockAssign);
        press_in(&mut config, &mut view, &c, Item::BlockAssign);
        assert_eq!(
            config.tuning.block("minecraft:oak_planks").assign,
            Assign::As(Surface::Brick)
        );
        press_in(&mut config, &mut view, &c, Item::BlockAbsorption);
        press_in(&mut config, &mut view, &c, Item::BlockIsolation);
        press_in(&mut config, &mut view, &c, Item::BlockScattering);
        let b = config.tuning.block("minecraft:oak_planks");
        assert_eq!(
            (b.absorption, b.isolation, b.scattering),
            (Some(0.0), Some(0.0), Some(0.0))
        );
        let en = labels_of(&config, &view, &c, "en_us");
        assert_eq!(en[2], "Material: Brick");
        assert_eq!(en[3], "Absorption: 0%");
        assert!(press_in(&mut config, &mut view, &c, Item::ResetBlock));
        assert!(config.tuning.blocks.is_empty());
        // Past the last preset, Auto again.
        for _ in 0..16 {
            press_in(&mut config, &mut view, &c, Item::BlockAssign);
        }
        assert!(config.tuning.blocks.is_empty());

        // The whole tag at once.
        let tag = c
            .tags
            .iter()
            .position(|(t, _)| t == "minecraft:planks_and_logs")
            .unwrap();
        let mut view = at(Section::Block(Target::Tag(tag)));
        assert_eq!(
            labels_of(&config, &view, &c, "en_us")[1],
            "Tag #minecraft:planks_and_logs: 2 blocks (all change at once)"
        );
        press_in(&mut config, &mut view, &c, Item::BlockIsolation);
        press_in(&mut config, &mut view, &c, Item::BlockIsolation);
        assert_eq!(
            config.tuning.block("minecraft:oak_log").isolation,
            Some(0.25)
        );
        assert_eq!(
            config.tuning.block("minecraft:oak_planks").isolation,
            Some(0.25)
        );
        assert_eq!(config.tuning.blocks.len(), 2);
        press_in(&mut config, &mut view, &c, Item::ResetBlock);
        assert!(config.tuning.blocks.is_empty());
    }

    #[test]
    fn reset_keeps_the_hrtf_and_the_materials_and_blocks() {
        let mut config = Config {
            enabled: false,
            walls: 1.0,
            absorption: 3.0,
            source_volume: 0.5,
            hrtf_path: Some("custom.hrtf".into()),
            volume_db: -3.0,
            debug_stats: true,
            ..Config::default()
        };
        Arc::make_mut(&mut config.tuning)
            .material_mut(Surface::Wood)
            .absorption = 2.0;
        assert!(press(&mut config, Item::Reset));
        assert!(config.enabled);
        assert!(!config.debugging());
        assert_eq!(config.walls, 0.35);
        assert_eq!(
            config.surfaces(),
            crate::acoustics::SurfaceTuning::default()
        );
        assert_eq!(config.hrtf_path, Some("custom.hrtf".into()));
        assert_eq!(config.volume_db, -3.0);
        assert_eq!(config.tuning.material(Surface::Wood).absorption, 2.0);
    }

    #[test]
    fn kept_values_round_trip_and_bad_ones_are_ignored() {
        let mut changed = Config::default();
        for item in [
            Item::SourceVolume,
            Item::Occlusion,
            Item::Pathing,
            Item::ReverbLevel,
            Item::Distance,
            Item::Absorption,
            Item::Scattering,
            Item::HearingRange,
            Item::Interpolation,
            Item::DebugSources,
        ] {
            press(&mut changed, item);
        }
        {
            let tuning = Arc::make_mut(&mut changed.tuning);
            tuning.material_mut(Surface::Glass).isolation = 3.0;
            tuning.set_block(
                "minecraft:white_wool",
                BlockTuning {
                    assign: Assign::As(Surface::Carpet),
                    scattering: Some(0.5),
                    ..BlockTuning::default()
                },
            );
        }
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
            "blocks" => Some("garbage".into()),
            _ => None,
        });
        assert_eq!(config.walls, 0.35);
        assert_eq!(config.reverb_level, 2.0);
        assert_eq!(config.scattering, 0.0);
        assert_eq!(config.hearing_range, 4.0);
        assert!(config.enabled);
        assert!(config.tuning.blocks.is_empty());
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
