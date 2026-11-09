//! What the world's blocks are made of, as the settings page tunes it: each material preset's absorption,
//! isolation and scattering, and each block's own (its material, and its values over the material's).
//!
//! The rays and the reflections know a block by its [`Surface`]: one of the 14 presets, or a *variant* of one this
//! module makes for a block tuned on its own. The [`Palette`] holds the resolved material of every surface; it is
//! built from the configuration ([`publish`], at every [`crate::horizon::Config::set`]) and from the game's blocks
//! ([`set_catalog`]), and read lock-free on the hot paths ([`material_of`], [`base_of`]).
//!
//! The settings are kept as two strings ([`Tuning::format_materials`], [`Tuning::format_blocks`]).

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use crate::acoustics::{Surface, SurfaceTuning};
use crate::scene::Material;

/// Most surfaces a palette holds (a cell of the reflection grid is `1 + code` in a byte).
pub const MAX_SURFACES: usize = 255;

/// A preset's tuning: factors over Steam Audio's values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialTuning {
    /// Factor on its absorption (1: the preset's; more: duller, shorter reverb). At most everything is absorbed.
    pub absorption: f32,
    /// How strongly it keeps sound from passing through: the exponent on its transmission (1: the preset's; 0: lets
    /// everything through; 2: lets through the square of it).
    pub isolation: f32,
    /// Its scattering, or the advanced page's for every surface.
    pub scattering: Option<f32>,
}

impl Default for MaterialTuning {
    fn default() -> Self {
        Self {
            absorption: 1.0,
            isolation: 1.0,
            scattering: None,
        }
    }
}

/// What a block is to the sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assign {
    /// The material its id and its break sound say ([`Surface::classify`]).
    Auto,
    /// Not a wall: sound goes through it.
    Open,
    /// This preset.
    As(Surface),
}

/// A block's own tuning: its material, and its values over the material's (`None`: the material's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockTuning {
    pub assign: Assign,
    pub absorption: Option<f32>,
    pub isolation: Option<f32>,
    pub scattering: Option<f32>,
}

impl Default for BlockTuning {
    fn default() -> Self {
        Self {
            assign: Assign::Auto,
            absorption: None,
            isolation: None,
            scattering: None,
        }
    }
}

impl BlockTuning {
    /// Whether it changes nothing.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }

    /// Whether it has values of its own (beyond its material).
    fn has_values(&self) -> bool {
        self.absorption.is_some() || self.isolation.is_some() || self.scattering.is_some()
    }
}

/// The settings page's materials and blocks.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Tuning {
    /// Per preset, in [`Surface::ALL`]'s order.
    pub materials: [MaterialTuning; Surface::ALL.len()],
    /// Per block id (`minecraft:oak_planks`); only blocks that change something.
    pub blocks: BTreeMap<String, BlockTuning>,
}

/// The lowercase name of a preset, as the kept settings write it.
pub fn preset_key(surface: Surface) -> &'static str {
    const KEYS: [&str; 14] = [
        "generic", "brick", "concrete", "ceramic", "gravel", "carpet", "glass", "plaster", "wood",
        "metal", "rock", "water", "foliage", "thin",
    ];
    KEYS[usize::from(surface.base().code()).min(KEYS.len() - 1)]
}

fn preset_from_key(key: &str) -> Option<Surface> {
    Surface::ALL
        .into_iter()
        .find(|&s| preset_key(s) == key.trim())
}

/// A kept number: finite, in `range`.
fn number(text: &str, range: std::ops::RangeInclusive<f32>) -> Option<f32> {
    text.trim()
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(*range.start(), *range.end()))
}

/// Ranges the kept values are clamped into.
const ABSORPTION_RANGE: std::ops::RangeInclusive<f32> = 0.0..=8.0;
const ISOLATION_RANGE: std::ops::RangeInclusive<f32> = 0.0..=8.0;
const SCATTERING_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;

impl Tuning {
    /// A preset's tuning.
    pub fn material(&self, surface: Surface) -> MaterialTuning {
        self.materials[usize::from(surface.base().code())]
    }

    pub fn material_mut(&mut self, surface: Surface) -> &mut MaterialTuning {
        &mut self.materials[usize::from(surface.base().code())]
    }

    /// A block's tuning (the default when it has none).
    pub fn block(&self, name: &str) -> BlockTuning {
        self.blocks.get(name).copied().unwrap_or_default()
    }

    /// Sets a block's tuning; a default one is forgotten.
    pub fn set_block(&mut self, name: &str, tuning: BlockTuning) {
        if tuning.is_default() {
            self.blocks.remove(name);
        } else {
            self.blocks.insert(name.to_owned(), tuning);
        }
    }

    /// The materials as kept: `rock=a1.5,i2,s0.3;wood=a0.5` (only the tuned ones).
    pub fn format_materials(&self) -> String {
        let mut out = Vec::new();
        for surface in Surface::ALL {
            let m = self.material(surface);
            if m == MaterialTuning::default() {
                continue;
            }
            let mut fields = Vec::new();
            if m.absorption != 1.0 {
                fields.push(format!("a{}", m.absorption));
            }
            if m.isolation != 1.0 {
                fields.push(format!("i{}", m.isolation));
            }
            if let Some(s) = m.scattering {
                fields.push(format!("s{s}"));
            }
            out.push(format!("{}={}", preset_key(surface), fields.join(",")));
        }
        out.join(";")
    }

    /// Reads the materials as kept; what it cannot read is left as it is.
    pub fn parse_materials(&mut self, text: &str) {
        for entry in text.split(';') {
            let Some((key, fields)) = entry.split_once('=') else {
                continue;
            };
            let Some(surface) = preset_from_key(key) else {
                continue;
            };
            let m = self.material_mut(surface);
            for field in fields.split(',') {
                let field = field.trim();
                if let Some(v) = field
                    .strip_prefix('a')
                    .and_then(|v| number(v, ABSORPTION_RANGE))
                {
                    m.absorption = v;
                } else if let Some(v) = field
                    .strip_prefix('i')
                    .and_then(|v| number(v, ISOLATION_RANGE))
                {
                    m.isolation = v;
                } else if let Some(v) = field
                    .strip_prefix('s')
                    .and_then(|v| number(v, SCATTERING_RANGE))
                {
                    m.scattering = Some(v);
                }
            }
        }
    }

    /// The blocks as kept: `minecraft:oak_planks=m:glass,a0.5,i2,s0.3;minecraft:dirt=m:open`.
    pub fn format_blocks(&self) -> String {
        let mut out = Vec::new();
        for (name, b) in &self.blocks {
            if b.is_default() {
                continue;
            }
            let mut fields = Vec::new();
            match b.assign {
                Assign::Auto => {}
                Assign::Open => fields.push("m:open".to_owned()),
                Assign::As(s) => fields.push(format!("m:{}", preset_key(s))),
            }
            if let Some(v) = b.absorption {
                fields.push(format!("a{v}"));
            }
            if let Some(v) = b.isolation {
                fields.push(format!("i{v}"));
            }
            if let Some(v) = b.scattering {
                fields.push(format!("s{v}"));
            }
            out.push(format!("{name}={}", fields.join(",")));
        }
        out.join(";")
    }

    /// Reads the blocks as kept (replacing the ones it had); what it cannot read is left out.
    pub fn parse_blocks(&mut self, text: &str) {
        self.blocks.clear();
        for entry in text.split(';') {
            let Some((name, fields)) = entry.rsplit_once('=') else {
                continue;
            };
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let mut b = BlockTuning::default();
            for field in fields.split(',') {
                let field = field.trim();
                if let Some(m) = field.strip_prefix("m:") {
                    if m == "open" {
                        b.assign = Assign::Open;
                    } else if let Some(s) = preset_from_key(m) {
                        b.assign = Assign::As(s);
                    }
                } else if let Some(v) = field
                    .strip_prefix('a')
                    .and_then(|v| number(v, ABSORPTION_RANGE))
                {
                    b.absorption = Some(v);
                } else if let Some(v) = field
                    .strip_prefix('i')
                    .and_then(|v| number(v, ISOLATION_RANGE))
                {
                    b.isolation = Some(v);
                } else if let Some(v) = field
                    .strip_prefix('s')
                    .and_then(|v| number(v, SCATTERING_RANGE))
                {
                    b.scattering = Some(v);
                }
            }
            self.set_block(name, b);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The game's blocks and tags.
// ---------------------------------------------------------------------------------------------------------------------

/// One of the game's blocks: its id, and the material its id and break sound give it (`None`: not a wall).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogBlock {
    pub name: String,
    pub auto: Option<Surface>,
}

/// The game's blocks (sorted by id) and the block tags that hold some of them (sorted, with the blocks' indices).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Catalog {
    pub blocks: Vec<CatalogBlock>,
    pub tags: Vec<(String, Vec<usize>)>,
}

impl Catalog {
    /// The catalog of `blocks` (id and material, in any order, repeats left out) and the block `tags` (tag id and
    /// the ids of its blocks, `minecraft:` or not).
    pub fn new(
        blocks: impl IntoIterator<Item = CatalogBlock>,
        tags: impl IntoIterator<Item = (String, Vec<String>)>,
    ) -> Self {
        let mut list: Vec<CatalogBlock> = blocks.into_iter().collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list.dedup_by(|a, b| a.name == b.name);
        let index = |name: &str| {
            let full = if name.contains(':') {
                name.to_owned()
            } else {
                format!("minecraft:{name}")
            };
            list.binary_search_by(|b| b.name.as_str().cmp(&full)).ok()
        };
        let mut tags: Vec<(String, Vec<usize>)> = tags
            .into_iter()
            .filter_map(|(tag, names)| {
                let mut blocks: Vec<usize> = names.iter().filter_map(|n| index(n)).collect();
                blocks.sort_unstable();
                blocks.dedup();
                (!blocks.is_empty()).then_some((tag, blocks))
            })
            .collect();
        tags.sort_by(|a, b| a.0.cmp(&b.0));
        Self { blocks: list, tags }
    }

    /// The index of block `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.blocks
            .binary_search_by(|b| b.name.as_str().cmp(name))
            .ok()
    }

    /// The blocks of tag `tag`.
    pub fn tag(&self, tag: &str) -> &[usize] {
        self.tags
            .iter()
            .find(|(t, _)| t == tag)
            .map_or(&[], |(_, b)| b.as_slice())
    }
}

static CATALOG: LazyLock<RwLock<Arc<Catalog>>> =
    LazyLock::new(|| RwLock::new(Arc::new(Catalog::default())));

/// The game's blocks and tags (empty until the game gives them).
pub fn catalog() -> Arc<Catalog> {
    Arc::clone(&CATALOG.read().unwrap_or_else(|e| e.into_inner()))
}

/// Gives the game's blocks and tags; a catalog like the one there changes nothing. The palette is made again for the
/// blocks tuned on their own (their material is known now).
pub fn set_catalog(catalog: Catalog) {
    {
        let mut current = CATALOG.write().unwrap_or_else(|e| e.into_inner());
        if **current == catalog {
            return;
        }
        *current = Arc::new(catalog);
    }
    publish(&crate::horizon::Config::get());
}

// ---------------------------------------------------------------------------------------------------------------------
// The palette.
// ---------------------------------------------------------------------------------------------------------------------

/// How a tuned block resolves: not a wall, or its surface for the material it has.
#[derive(Clone, Debug, PartialEq)]
enum Resolved {
    Open,
    /// Its assigned preset (`None`: the one the block gives) and its variants: per preset, the surface with the
    /// block's own values.
    Tuned {
        assign: Option<Surface>,
        variants: Vec<(Surface, Surface)>,
    },
}

/// The resolved material of every surface (the presets first, then the variants of tuned blocks), and how each tuned
/// block resolves.
#[derive(Clone, Debug, PartialEq)]
pub struct Palette {
    /// Per surface code: its preset and its material.
    surfaces: Vec<(Surface, Material)>,
    blocks: HashMap<String, Resolved>,
}

/// The material of preset `base` with `material`'s and `block`'s tuning over it and `global` for every surface.
fn resolve_material(
    base: Surface,
    global: SurfaceTuning,
    material: MaterialTuning,
    block: Option<&BlockTuning>,
) -> Material {
    let mut m = base.preset();
    let absorption =
        global.absorption * material.absorption * block.and_then(|b| b.absorption).unwrap_or(1.0);
    for a in &mut m.absorption {
        *a = (*a * absorption).clamp(0.0, 1.0);
    }
    let isolation = material.isolation * block.and_then(|b| b.isolation).unwrap_or(1.0);
    if isolation != 1.0 {
        for t in &mut m.transmission {
            *t = t.max(1e-4).powf(isolation.max(0.0)).clamp(0.0, 1.0);
        }
    }
    m.scattering = block
        .and_then(|b| b.scattering)
        .or(material.scattering)
        .unwrap_or(global.scattering)
        .clamp(0.0, 1.0);
    m
}

impl Palette {
    /// The palette of `tuning` with `global` over every surface, for the blocks of `catalog` (a block tuned on its
    /// own whose material comes from the game gets the variant of the one the catalog gives it). Past
    /// [`MAX_SURFACES`], a block keeps its material without its own values.
    pub fn build(global: SurfaceTuning, tuning: &Tuning, catalog: &Catalog) -> Self {
        let mut surfaces: Vec<(Surface, Material)> = Surface::ALL
            .iter()
            .map(|&s| (s, resolve_material(s, global, tuning.material(s), None)))
            .collect();
        let mut blocks = HashMap::new();
        for (name, b) in &tuning.blocks {
            let assign = match b.assign {
                Assign::Open => {
                    blocks.insert(name.clone(), Resolved::Open);
                    continue;
                }
                Assign::Auto => None,
                Assign::As(s) => Some(s.base()),
            };
            let mut variants = Vec::new();
            if b.has_values() {
                let base = assign.or_else(|| {
                    catalog
                        .find(name)
                        .and_then(|i| catalog.blocks[i].auto)
                        .map(Surface::base)
                });
                if let Some(base) = base {
                    let material = resolve_material(base, global, tuning.material(base), Some(b));
                    let code = surfaces
                        .iter()
                        .position(|&(s, m)| s == base && m == material)
                        .or_else(|| {
                            (surfaces.len() < MAX_SURFACES).then(|| {
                                surfaces.push((base, material));
                                surfaces.len() - 1
                            })
                        });
                    if let Some(code) = code {
                        variants.push((base, Surface::from_code(code as u8)));
                    }
                }
            }
            blocks.insert(name.clone(), Resolved::Tuned { assign, variants });
        }
        Self { surfaces, blocks }
    }

    /// The palette of the defaults.
    pub fn defaults() -> Self {
        Self::build(
            SurfaceTuning::default(),
            &Tuning::default(),
            &Catalog::default(),
        )
    }

    /// The surface of block `name`, which gives it `auto` ([`Surface::classify`]): `None` when it is not a wall.
    pub fn resolve(&self, name: &str, auto: Option<Surface>) -> Option<Surface> {
        match self.blocks.get(name) {
            None => auto,
            Some(Resolved::Open) => None,
            Some(Resolved::Tuned { assign, variants }) => {
                let base = assign.or(auto)?.base();
                Some(
                    variants
                        .iter()
                        .find(|(b, _)| *b == base)
                        .map_or(base, |&(_, v)| v),
                )
            }
        }
    }

    /// The material of `surface` (a preset's when the palette has no such code).
    pub fn material(&self, surface: Surface) -> Material {
        self.surfaces
            .get(usize::from(surface.code()))
            .map_or_else(|| Surface::Generic.preset(), |&(_, m)| m)
    }

    /// The preset `surface` is a variant of.
    pub fn base(&self, surface: Surface) -> Surface {
        self.surfaces
            .get(usize::from(surface.code()))
            .map_or(Surface::Generic, |&(b, _)| b)
    }

    /// How many surfaces it has.
    pub fn len(&self) -> usize {
        self.surfaces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.surfaces.is_empty()
    }

    /// The material of every surface code, for the reflections (codes it has not: Generic's preset).
    pub fn materials(&self) -> Vec<Material> {
        (0..MAX_SURFACES)
            .map(|c| self.material(Surface::from_code(c as u8)))
            .collect()
    }
}

static PALETTE: LazyLock<RwLock<Arc<Palette>>> =
    LazyLock::new(|| RwLock::new(Arc::new(Palette::defaults())));
static VERSION: AtomicU64 = AtomicU64::new(0);
/// Each surface code's preset, readable without the lock ([`base_of`]); the presets are their own.
static BASES: [AtomicU8; 256] = {
    let mut bases = [const { AtomicU8::new(0) }; 256];
    let mut i = 0;
    while i < Surface::ALL.len() {
        bases[i] = AtomicU8::new(i as u8);
        i += 1;
    }
    bases
};

/// Makes the palette of `config`'s tuning (and the game's blocks) the one the rays and the reflections use, unless
/// it is the same.
pub fn publish(config: &crate::horizon::Config) {
    let palette = Palette::build(config.surfaces(), &config.tuning, &catalog());
    let mut current = PALETTE.write().unwrap_or_else(|e| e.into_inner());
    if **current == palette {
        return;
    }
    for (code, &(base, _)) in palette.surfaces.iter().enumerate() {
        BASES[code].store(base.code(), Ordering::Relaxed);
    }
    *current = Arc::new(palette);
    VERSION.fetch_add(1, Ordering::Release);
}

/// Bumped by each new palette.
pub fn version() -> u64 {
    VERSION.load(Ordering::Acquire)
}

/// The palette now.
pub fn palette() -> Arc<Palette> {
    thread_local! {
        static CACHED: RefCell<Option<(u64, Arc<Palette>)>> = const { RefCell::new(None) };
    }
    let version = version();
    CACHED.with_borrow_mut(|cached| {
        if let Some((v, p)) = cached.as_ref()
            && *v == version
        {
            return Arc::clone(p);
        }
        let p = Arc::clone(&PALETTE.read().unwrap_or_else(|e| e.into_inner()));
        *cached = Some((version, Arc::clone(&p)));
        p
    })
}

/// The material of `surface` in the palette now.
pub fn material_of(surface: Surface) -> Material {
    palette().material(surface)
}

/// The preset of a variant `surface` (a preset is its own).
pub fn base_of(surface: Surface) -> Surface {
    Surface::from_code(BASES[usize::from(surface.code())].load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalog() -> Catalog {
        Catalog::new(
            [
                CatalogBlock {
                    name: "minecraft:oak_planks".into(),
                    auto: Some(Surface::Wood),
                },
                CatalogBlock {
                    name: "minecraft:stone".into(),
                    auto: Some(Surface::Rock),
                },
                CatalogBlock {
                    name: "minecraft:oak_log".into(),
                    auto: Some(Surface::Wood),
                },
                CatalogBlock {
                    name: "minecraft:stone".into(),
                    auto: Some(Surface::Rock),
                },
                CatalogBlock {
                    name: "minecraft:poppy".into(),
                    auto: None,
                },
            ],
            [
                (
                    "minecraft:logs".into(),
                    vec!["minecraft:oak_log".into(), "nothing".into()],
                ),
                ("minecraft:planks".into(), vec!["oak_planks".into()]),
                ("minecraft:empty".into(), vec![]),
            ],
        )
    }

    #[test]
    fn the_catalog_sorts_blocks_and_keeps_tags_with_blocks() {
        let c = catalog();
        let names: Vec<&str> = c.blocks.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "minecraft:oak_log",
                "minecraft:oak_planks",
                "minecraft:poppy",
                "minecraft:stone"
            ]
        );
        assert_eq!(c.tags.len(), 2);
        assert_eq!(c.tag("minecraft:logs"), &[0]);
        assert_eq!(c.tag("minecraft:planks"), &[1]);
        assert_eq!(c.find("minecraft:stone"), Some(3));
    }

    #[test]
    fn kept_tuning_round_trips_and_bad_values_are_left_out() {
        let mut t = Tuning::default();
        t.material_mut(Surface::Rock).absorption = 1.5;
        t.material_mut(Surface::Rock).isolation = 2.0;
        t.material_mut(Surface::Wood).scattering = Some(0.3);
        t.set_block(
            "minecraft:oak_planks",
            BlockTuning {
                assign: Assign::As(Surface::Glass),
                absorption: Some(0.5),
                ..BlockTuning::default()
            },
        );
        t.set_block(
            "minecraft:dirt",
            BlockTuning {
                assign: Assign::Open,
                ..BlockTuning::default()
            },
        );
        t.set_block("minecraft:stone", BlockTuning::default());
        assert_eq!(t.blocks.len(), 2);
        let materials = t.format_materials();
        let blocks = t.format_blocks();
        assert_eq!(materials, "wood=s0.3;rock=a1.5,i2");
        assert_eq!(
            blocks,
            "minecraft:dirt=m:open;minecraft:oak_planks=m:glass,a0.5"
        );
        let mut back = Tuning::default();
        back.parse_materials(&materials);
        back.parse_blocks(&blocks);
        assert_eq!(back, t);

        let mut bad = Tuning::default();
        bad.parse_materials("lava=a2;rock=aNaN,i99,sx;=;wood");
        bad.parse_blocks("=a1;minecraft:stone=m:lava,a-3;;minecraft:glass=");
        assert_eq!(bad.material(Surface::Rock).isolation, 8.0);
        assert_eq!(bad.material(Surface::Rock).absorption, 1.0);
        assert_eq!(bad.blocks.len(), 1);
        assert_eq!(bad.block("minecraft:stone").absorption, Some(0.0));
        assert_eq!(bad.block("minecraft:stone").assign, Assign::Auto);
    }

    #[test]
    fn the_palette_resolves_tuned_blocks_to_variants() {
        let c = catalog();
        let mut t = Tuning::default();
        // Rock absorbs twice as much and isolates twice as well.
        t.material_mut(Surface::Rock).absorption = 2.0;
        t.material_mut(Surface::Rock).isolation = 2.0;
        // Planks are glass; logs are wood that scatters a lot; the poppy is a wall of brick; stone is open.
        t.set_block(
            "minecraft:oak_planks",
            BlockTuning {
                assign: Assign::As(Surface::Glass),
                ..BlockTuning::default()
            },
        );
        let scattering = BlockTuning {
            scattering: Some(0.9),
            ..BlockTuning::default()
        };
        t.set_block("minecraft:oak_log", scattering);
        t.set_block(
            "minecraft:poppy",
            BlockTuning {
                assign: Assign::As(Surface::Brick),
                ..BlockTuning::default()
            },
        );
        t.set_block(
            "minecraft:stone",
            BlockTuning {
                assign: Assign::Open,
                ..BlockTuning::default()
            },
        );
        let p = Palette::build(SurfaceTuning::default(), &t, &c);
        let rock = p.material(Surface::Rock);
        let preset = Surface::Rock.preset();
        assert!((rock.absorption[0] - 2.0 * preset.absorption[0]).abs() < 1e-6);
        assert!((rock.transmission[0] - preset.transmission[0].powi(2)).abs() < 1e-9);

        assert_eq!(
            p.resolve("minecraft:oak_planks", Some(Surface::Wood)),
            Some(Surface::Glass)
        );
        assert_eq!(p.resolve("minecraft:stone", Some(Surface::Rock)), None);
        assert_eq!(p.resolve("minecraft:poppy", None), Some(Surface::Brick));
        assert_eq!(
            p.resolve("minecraft:dirt", Some(Surface::Gravel)),
            Some(Surface::Gravel)
        );
        let log = p.resolve("minecraft:oak_log", Some(Surface::Wood)).unwrap();
        assert!(!log.is_preset());
        assert_eq!(p.base(log), Surface::Wood);
        assert_eq!(p.material(log).scattering, 0.9);
        assert_eq!(
            p.material(log).absorption,
            Surface::Wood.preset().absorption
        );
        assert_eq!(p.len(), Surface::ALL.len() + 1);
        // A state of it the catalog did not foresee keeps the plain preset.
        assert_eq!(
            p.resolve("minecraft:oak_log", Some(Surface::Thin)),
            Some(Surface::Thin)
        );

        // The same values share one variant.
        t.set_block("minecraft:oak_planks", scattering);
        let p = Palette::build(SurfaceTuning::default(), &t, &c);
        assert_eq!(p.len(), Surface::ALL.len() + 1);
        assert_eq!(
            p.resolve("minecraft:oak_planks", Some(Surface::Wood)),
            p.resolve("minecraft:oak_log", Some(Surface::Wood))
        );
    }

    #[test]
    fn the_advanced_page_tunes_every_surface_and_blocks_win() {
        let mut t = Tuning::default();
        t.material_mut(Surface::Wood).scattering = Some(0.4);
        let global = SurfaceTuning {
            absorption: 0.5,
            scattering: 0.2,
        };
        let p = Palette::build(global, &t, &Catalog::default());
        assert_eq!(p.material(Surface::Rock).scattering, 0.2);
        assert_eq!(p.material(Surface::Wood).scattering, 0.4);
        let brick = Surface::Brick.preset();
        assert!(
            (p.material(Surface::Brick).absorption[0] - 0.5 * brick.absorption[0]).abs() < 1e-6
        );
        // Isolation 0 lets everything through; absorption at most 1.
        t.material_mut(Surface::Gravel).isolation = 0.0;
        t.material_mut(Surface::Gravel).absorption = 8.0;
        let p = Palette::build(SurfaceTuning::default(), &t, &Catalog::default());
        assert_eq!(p.material(Surface::Gravel).transmission, [1.0; 3]);
        assert_eq!(p.material(Surface::Gravel).absorption, [1.0; 3]);
        assert_eq!(
            Palette::defaults().material(Surface::Wood),
            Surface::Wood.preset()
        );
    }
}
