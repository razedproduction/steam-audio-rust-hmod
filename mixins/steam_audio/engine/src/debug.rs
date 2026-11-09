//! The mod's visual debug mode: what the acoustics found, for the game to draw over the world.
//!
//! Four layers, each a button of the settings page ([`crate::settings`]) and off by default:
//!
//! - **Sources and occlusion** ([`Config::debug_sources`]): every source being looked at, its occlusion rays (clear
//!   or blocked), the walls and the leaves or fences on its line of sight, and a label with its occlusion,
//!   transmission and whether it is rendered binaurally.
//! - **Paths round walls** ([`Config::debug_paths`]): the pathing probes and, for each hidden source, the legs of the
//!   baked paths between them and where the way round arrives from.
//! - **Reflections and reverb** ([`Config::debug_reflections`]): the grid the reflections and paths are simulated in,
//!   sample reflection rays from the listener bouncing off the walls, and the sources with reflections of their own.
//! - **Statistics** ([`Config::debug_stats`]): a panel with the renderer's voices and load, the reflection
//!   simulation's runs and reverb times, and the pathing bake.
//!
//! The game thread ([`publish_scene`]), the audio thread ([`publish_renderer`], never waiting) and the simulation
//! threads ([`publish_reflections`], [`publish_paths`]) leave what they have here; the game's HUD and render hooks
//! read it. Nothing is gathered while every layer is off.
//!
//! [`Config::debug_sources`]: crate::horizon::Config::debug_sources
//! [`Config::debug_paths`]: crate::horizon::Config::debug_paths
//! [`Config::debug_reflections`]: crate::horizon::Config::debug_reflections
//! [`Config::debug_stats`]: crate::horizon::Config::debug_stats

use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::acoustics::SourceAcoustics;
use crate::bands::NUM_BANDS;
use crate::geometry::Ray;
use crate::math::Vector3;
use crate::reflections::Voxels;
use crate::scene::RayTracer;

/// What the occlusion of one source looked at: each ray from the listener (the point it aimed at, whether it got
/// through), the walls on the line of sight (cells) and the leaves and fences on it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceTrace {
    pub rays: Vec<([f64; 3], bool)>,
    pub walls: Vec<[i32; 3]>,
    pub filters: Vec<[i32; 3]>,
}

/// One source as the debug layers show it.
#[derive(Clone, Debug, PartialEq)]
pub struct DebugSource {
    /// The mixer's channel id.
    pub id: u64,
    /// Where it is (world).
    pub at: [f64; 3],
    /// What the acoustics found this round (before smoothing).
    pub acoustics: SourceAcoustics,
    pub trace: SourceTrace,
    /// The legs of its paths round the walls, probe to probe (world), when it has some.
    pub path_legs: Vec<([f64; 3], [f64; 3])>,
    /// Whether it has reflections simulated from where it is.
    pub own_reflections: bool,
}

/// What the game thread found, for the world layers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Scene {
    pub listener: Option<[f64; 3]>,
    pub underwater: bool,
    pub sources: Vec<DebugSource>,
    /// The pathing probes of the latest bake (world).
    pub probes: Vec<[f64; 3]>,
    /// The grid the reflections and paths are simulated in: its corner and size, in blocks.
    pub grid: Option<([i32; 3], [usize; 3])>,
    /// Sample reflection rays from the listener: the points where each bounces (world), the listener first.
    pub reflection_rays: Vec<Vec<[f64; 3]>>,
}

/// What the audio thread rendered in its latest callback.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RendererStats {
    pub rate: u32,
    /// Playing sources, those with a direction, those rendered binaurally, and those also heard round the walls.
    pub voices: usize,
    pub positional: usize,
    pub binaural: usize,
    pub paths: usize,
    /// Sources heard in another's binaural voice, near it (`horizon::group_radius`): the binaural voices are
    /// `binaural`, and they play `binaural + grouped` sources.
    pub grouped: usize,
    /// The binaural budget: the configuration's and what the load allows now.
    pub budget: usize,
    pub limit: usize,
    /// Rendering time over playing time (smoothed).
    pub load: f32,
    /// Source reverbs playing or ringing out.
    pub source_reverbs: usize,
    /// The reverb bus level and how far under water the listener sounds (0 to 1).
    pub wet: f32,
    pub underwater: f32,
    /// The channel ids rendered binaurally.
    pub binaural_ids: Vec<u64>,
    pub at: Option<Instant>,
}

/// The reflection simulation's latest run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReflectionStats {
    /// Runs averaged for the listener's reverb, out of the most it averages.
    pub runs: usize,
    pub max_runs: usize,
    pub last_run: Option<Duration>,
    /// The listener's reverb times (s) per band.
    pub reverb_times: Option<[f32; NUM_BANDS]>,
    /// The sources with reflections of their own: id, where, runs averaged, reverb times.
    pub sources: Vec<(u64, [f64; 3], usize, [f32; NUM_BANDS])>,
    pub at: Option<Instant>,
}

/// The pathing bake's latest result.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PathStats {
    pub probes: usize,
    pub bake: Option<Duration>,
    pub at: Option<Instant>,
}

static SCENE: Mutex<Option<Scene>> = Mutex::new(None);
static RENDERER: Mutex<Option<RendererStats>> = Mutex::new(None);
static REFLECTIONS: Mutex<Option<ReflectionStats>> = Mutex::new(None);
static PATHS: Mutex<PathStats> = Mutex::new(PathStats {
    probes: 0,
    bake: None,
    at: None,
});

/// Leaves the game thread's latest findings for the world layers.
pub fn publish_scene(scene: Scene) {
    if let Ok(mut s) = SCENE.lock() {
        *s = Some(scene);
    }
}

/// Forgets the scene (the debug layers were turned off, or there is no world).
pub fn clear_scene() {
    if let Ok(mut s) = SCENE.lock() {
        *s = None;
    }
}

/// Runs `f` with the latest scene, if there is one.
pub fn with_scene<R>(f: impl FnOnce(&Scene) -> R) -> Option<R> {
    let s = SCENE.lock().ok()?;
    s.as_ref().map(f)
}

/// Updates the audio thread's statistics in place (never waits: a busy lock skips this callback).
pub fn publish_renderer(f: impl FnOnce(&mut RendererStats)) {
    if let Ok(mut s) = RENDERER.try_lock() {
        let stats = s.get_or_insert_with(RendererStats::default);
        f(stats);
        stats.at = Some(Instant::now());
    }
}

/// The audio thread's latest statistics, if it rendered lately (within a second).
pub fn renderer() -> Option<RendererStats> {
    let s = RENDERER.lock().ok()?;
    s.as_ref()
        .filter(|s| s.at.is_some_and(|t| t.elapsed() < Duration::from_secs(1)))
        .cloned()
}

/// Leaves the reflection simulation's latest run.
pub fn publish_reflections(stats: ReflectionStats) {
    if let Ok(mut s) = REFLECTIONS.lock() {
        *s = Some(stats);
    }
}

pub fn reflections() -> Option<ReflectionStats> {
    REFLECTIONS.lock().ok()?.clone()
}

/// Leaves the pathing bake's latest result.
pub fn publish_paths(probes: usize, bake: Duration) {
    if let Ok(mut s) = PATHS.lock() {
        *s = PathStats {
            probes,
            bake: Some(bake),
            at: Some(Instant::now()),
        };
    }
}

pub fn paths() -> PathStats {
    PATHS.lock().map(|s| *s).unwrap_or_default()
}

/// Sample reflection rays from `listener` in `grid`: `count` directions spread evenly over the sphere (a Fibonacci
/// lattice, the same every time), each followed for up to `bounces` specular bounces off the walls or until it
/// leaves the grid. Each ray is the listener and the points where it bounced (world), and where it ended.
pub fn reflection_rays(
    grid: &Voxels,
    listener: [f64; 3],
    count: usize,
    bounces: usize,
) -> Vec<Vec<[f64; 3]>> {
    const MAX_LENGTH: f32 = 64.0;
    let start = grid.to_local(listener);
    if !grid.inside(start)
        || grid.is_wall_local([start.x, start.y, start.z].map(|v| v.floor() as i32))
    {
        return Vec::new();
    }
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    (0..count)
        .map(|i| {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / count as f32;
            let r = (1.0 - y * y).max(0.0).sqrt();
            let phi = golden * i as f32;
            let mut ray = Ray::new(start, Vector3::new(r * phi.cos(), y, r * phi.sin()));
            let mut points = vec![listener];
            for _ in 0..=bounces {
                let hit = grid.closest_hit(&ray, 1e-3, MAX_LENGTH);
                if !hit.distance.is_finite() {
                    points.push(
                        grid.to_world(ray.point_at_distance(MAX_LENGTH.min(exit(grid, &ray)))),
                    );
                    break;
                }
                let at = ray.point_at_distance(hit.distance);
                points.push(grid.to_world(at));
                let n = hit.normal;
                let d = ray.direction;
                let reflected = d - n * (2.0 * Vector3::dot(d, n));
                // Off the wall a little, so the next trace does not hit the same face.
                ray = Ray::new(at + n * 1e-3, reflected.normalize_or_zero());
            }
            points
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------------------------------
// What to draw
// ---------------------------------------------------------------------------------------------------------------------

/// Which layers are on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layers {
    pub sources: bool,
    pub paths: bool,
    pub reflections: bool,
    pub stats: bool,
}

impl Layers {
    pub fn of(config: &crate::horizon::Config) -> Self {
        Self {
            sources: config.debug_sources,
            paths: config.debug_paths,
            reflections: config.debug_reflections,
            stats: config.debug_stats,
        }
    }

    pub fn any(self) -> bool {
        self.sources || self.paths || self.reflections || self.stats
    }

    /// Whether a layer drawn in the world is on.
    pub fn world(self) -> bool {
        self.sources || self.paths || self.reflections
    }
}

/// A line in the world, drawn over everything: its ends (world), colour (RGBA, 0 to 1) and width in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Line {
    pub a: [f64; 3],
    pub b: [f64; 3],
    pub color: [f32; 4],
    pub width: f32,
}

/// A label in the world: where (world) and its lines of text with their colours (0xRRGGBB).
#[derive(Clone, Debug, PartialEq)]
pub struct Label {
    pub at: [f64; 3],
    pub lines: Vec<(String, u32)>,
}

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 0.9];
const CLEAR_RAY: [f32; 4] = [0.3, 1.0, 0.4, 0.55];
const BLOCKED_RAY: [f32; 4] = [1.0, 0.25, 0.2, 0.55];
const WALL: [f32; 4] = [1.0, 0.55, 0.1, 0.9];
const FILTER: [f32; 4] = [0.55, 0.9, 0.3, 0.8];
const PROBE: [f32; 4] = [0.2, 0.85, 1.0, 0.7];
const PATH: [f32; 4] = [1.0, 0.9, 0.15, 0.95];
const ARRIVAL: [f32; 4] = [1.0, 0.3, 0.9, 0.95];
const GRID: [f32; 4] = [0.35, 0.5, 1.0, 0.7];
const OWN_REFLECTIONS: [f32; 4] = [0.75, 0.4, 1.0, 0.95];

/// The colour of a source by its occlusion: red hidden, yellow half seen, green in plain sight.
pub fn occlusion_color(occlusion: f32) -> [f32; 4] {
    let o = occlusion.clamp(0.0, 1.0);
    [(2.0 * (1.0 - o)).min(1.0), (2.0 * o).min(1.0), 0.2, 1.0]
}

fn rgb(color: [f32; 4]) -> u32 {
    let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    (c(color[0]) << 16) | (c(color[1]) << 8) | c(color[2])
}

/// The twelve edges of the box from `min` to `max`.
fn push_box(lines: &mut Vec<Line>, min: [f64; 3], max: [f64; 3], color: [f32; 4], width: f32) {
    let corner = |i: usize| -> [f64; 3] {
        std::array::from_fn(|a| if i & (1 << a) != 0 { max[a] } else { min[a] })
    };
    for i in 0..8 {
        for a in 0..3 {
            let j = i | (1 << a);
            if j != i {
                lines.push(Line {
                    a: corner(i),
                    b: corner(j),
                    color,
                    width,
                });
            }
        }
    }
}

/// A box of half-size `r` around `at`.
fn push_cube(lines: &mut Vec<Line>, at: [f64; 3], r: f64, color: [f32; 4], width: f32) {
    push_box(lines, at.map(|v| v - r), at.map(|v| v + r), color, width);
}

/// The block `cell`, a little larger so its edges show on the block's own.
fn push_cell(lines: &mut Vec<Line>, cell: [i32; 3], color: [f32; 4]) {
    let min = cell.map(|v| f64::from(v) - 0.01);
    let max = cell.map(|v| f64::from(v) + 1.01);
    push_box(lines, min, max, color, 2.0);
}

/// Three short lines crossing at `at`.
fn push_cross(lines: &mut Vec<Line>, at: [f64; 3], r: f64, color: [f32; 4], width: f32) {
    for a in 0..3 {
        let (mut p, mut q) = (at, at);
        p[a] -= r;
        q[a] += r;
        lines.push(Line {
            a: p,
            b: q,
            color,
            width,
        });
    }
}

/// The lines the world layers draw for `scene`.
pub fn lines(scene: &Scene, layers: Layers) -> Vec<Line> {
    let mut lines = Vec::new();
    if let Some(listener) = scene.listener
        && layers.world()
    {
        push_cross(&mut lines, listener, 0.3, WHITE, 2.0);
    }
    if layers.reflections {
        if let Some((origin, size)) = scene.grid {
            let min = origin.map(f64::from);
            let max: [f64; 3] = std::array::from_fn(|a| min[a] + size[a] as f64);
            push_box(&mut lines, min, max, GRID, 2.0);
        }
        for ray in &scene.reflection_rays {
            for (bounce, leg) in ray.windows(2).enumerate() {
                // Brighter on the way out, fainter with each bounce.
                let fade = 0.85f32.powi(bounce as i32);
                lines.push(Line {
                    a: leg[0],
                    b: leg[1],
                    color: [
                        0.45 * fade + 0.2,
                        0.75 * fade + 0.15,
                        1.0,
                        0.25 + 0.5 * fade,
                    ],
                    width: 1.5,
                });
            }
        }
        for s in scene.sources.iter().filter(|s| s.own_reflections) {
            push_cube(&mut lines, s.at, 0.45, OWN_REFLECTIONS, 3.0);
        }
    }
    if layers.paths {
        for &probe in &scene.probes {
            push_cross(&mut lines, probe, 0.2, PROBE, 1.5);
        }
    }
    for s in &scene.sources {
        if layers.sources {
            let color = occlusion_color(s.acoustics.occlusion);
            push_cube(&mut lines, s.at, 0.25, color, 3.0);
            if let Some(listener) = scene.listener {
                for &(point, clear) in &s.trace.rays {
                    lines.push(Line {
                        a: listener,
                        b: point,
                        color: if clear { CLEAR_RAY } else { BLOCKED_RAY },
                        width: 1.5,
                    });
                }
            }
            for &cell in &s.trace.walls {
                push_cell(&mut lines, cell, WALL);
            }
            for &cell in &s.trace.filters {
                push_cell(&mut lines, cell, FILTER);
            }
        }
        if layers.paths {
            for &(a, b) in &s.path_legs {
                lines.push(Line {
                    a,
                    b,
                    color: PATH,
                    width: 3.0,
                });
            }
            if let (Some(listener), Some(path)) = (scene.listener, s.acoustics.path) {
                let d = path.direction.map(f64::from);
                lines.push(Line {
                    a: listener,
                    b: std::array::from_fn(|a| listener[a] + 3.0 * d[a]),
                    color: ARRIVAL,
                    width: 3.0,
                });
                // The source, if only the paths layer marks it.
                if !layers.sources {
                    push_cube(&mut lines, s.at, 0.25, PATH, 2.0);
                }
            }
        }
    }
    lines
}

fn russian(language: &str) -> bool {
    let language = language.to_ascii_lowercase();
    language.starts_with("ru") || language.starts_with("be") || language.starts_with("uk")
}

fn bands(values: [f32; NUM_BANDS], digits: usize) -> String {
    values
        .iter()
        .map(|v| format!("{v:.digits$}"))
        .collect::<Vec<_>>()
        .join("/")
}

fn percent(v: f32) -> String {
    format!("{}%", (v * 100.0).round() as i32)
}

/// The labels of the sources, in the game's `language`. `binaural` says which channels the renderer rendered
/// binaurally (`None`: not known), `reflections` the reflection simulation's latest run.
pub fn labels(
    scene: &Scene,
    layers: Layers,
    language: &str,
    binaural: Option<&[u64]>,
    reflections: Option<&ReflectionStats>,
) -> Vec<Label> {
    let ru = russian(language);
    let mut labels = Vec::new();
    for s in &scene.sources {
        let a = &s.acoustics;
        let mut lines = Vec::new();
        if layers.sources {
            let color = rgb(occlusion_color(a.occlusion));
            let rendered = match binaural.map(|ids| ids.contains(&s.id)) {
                Some(true) => "HRTF",
                Some(false) if ru => "ванильно",
                Some(false) => "vanilla",
                None => "",
            };
            lines.push((
                format!("#{} {rendered}", s.id).trim_end().to_owned(),
                0xFF_FF_FF,
            ));
            let (occlusion, walls) = if ru {
                ("видно", "стены")
            } else {
                ("seen", "walls")
            };
            lines.push((
                format!(
                    "{occlusion} {} · {walls} {}",
                    percent(a.occlusion),
                    s.trace.walls.len()
                ),
                color,
            ));
            if a.occlusion < 1.0 {
                let through = if ru { "сквозь" } else { "through" };
                lines.push((
                    format!("{through} {}", bands(a.transmission, 3)),
                    0xFF_AA_55,
                ));
            }
            if a.through.iter().any(|&t| t < 0.999) {
                let leaves = if ru {
                    "листва/заборы"
                } else {
                    "leaves/fences"
                };
                lines.push((format!("{leaves} {}", bands(a.through, 2)), 0x99_E6_66));
            }
        }
        if layers.paths
            && let Some(p) = a.path
        {
            let (way, blocks) = if ru {
                ("путь", "бл.")
            } else {
                ("path", "blocks")
            };
            lines.push((
                format!(
                    "{way} {} · {:.1} {blocks} · EQ {}",
                    percent(a.path_gain()),
                    p.distance,
                    bands(p.eq, 2)
                ),
                rgb(PATH),
            ));
        }
        if layers.reflections
            && s.own_reflections
            && let Some((_, _, runs, times)) =
                reflections.and_then(|r| r.sources.iter().find(|(id, ..)| *id == s.id))
        {
            let own = if ru {
                "свои отражения"
            } else {
                "own reflections"
            };
            lines.push((
                format!("{own}: RT60 {} s ({runs})", bands(*times, 2)),
                rgb(OWN_REFLECTIONS),
            ));
        }
        if !lines.is_empty() {
            labels.push(Label {
                at: [s.at[0], s.at[1] + 0.6, s.at[2]],
                lines,
            });
        }
    }
    labels
}

fn millis(d: Option<Duration>) -> String {
    d.map_or_else(
        || "—".into(),
        |d| format!("{:.0}", d.as_secs_f64() * 1000.0),
    )
}

/// The statistics panel's lines, in the game's `language`.
pub fn panel(
    scene: Option<&Scene>,
    renderer: Option<&RendererStats>,
    reflections: Option<&ReflectionStats>,
    paths: &PathStats,
    language: &str,
) -> Vec<(String, u32)> {
    const TITLE: u32 = 0xFF_E0_70;
    const TEXT: u32 = 0xE6_E6_E6;
    const DIM: u32 = 0xA0_A0_A0;
    let ru = russian(language);
    let mut lines = vec![(
        if ru {
            "Steam Audio — отладка"
        } else {
            "Steam Audio — debug"
        }
        .to_owned(),
        TITLE,
    )];
    match renderer {
        Some(r) => {
            let load_color = if r.load > 0.6 {
                0xFF_60_50
            } else if r.load > 0.35 {
                0xFF_D0_50
            } else {
                0x80_FF_80
            };
            if ru {
                lines.push((
                    format!("Источники: {} играет, {} в мире", r.voices, r.positional),
                    TEXT,
                ));
                lines.push((
                    format!(
                        "Бинаурально: {} (бюджет {}, сейчас {}), путей {}, слито {}",
                        r.binaural, r.budget, r.limit, r.paths, r.grouped
                    ),
                    TEXT,
                ));
                lines.push((
                    format!("Аудиопоток: {} кадра, {} Гц", percent(r.load), r.rate),
                    load_color,
                ));
                lines.push((
                    format!(
                        "Реверберация: уровень {:.2}, своих ревербераторов {}",
                        r.wet, r.source_reverbs
                    ),
                    TEXT,
                ));
            } else {
                lines.push((
                    format!(
                        "Sources: {} playing, {} in the world",
                        r.voices, r.positional
                    ),
                    TEXT,
                ));
                lines.push((
                    format!(
                        "Binaural: {} (budget {}, now {}), paths {}, merged {}",
                        r.binaural, r.budget, r.limit, r.paths, r.grouped
                    ),
                    TEXT,
                ));
                lines.push((
                    format!(
                        "Audio thread: {} of a frame, {} Hz",
                        percent(r.load),
                        r.rate
                    ),
                    load_color,
                ));
                lines.push((
                    format!(
                        "Reverb: level {:.2}, source reverbs {}",
                        r.wet, r.source_reverbs
                    ),
                    TEXT,
                ));
            }
        }
        None => lines.push((
            if ru {
                "Аудиопоток: не рендерит (звук Horizon или HRTF грузится)"
            } else {
                "Audio thread: not rendering (Horizon's sound, or the HRTF loading)"
            }
            .to_owned(),
            DIM,
        )),
    }
    match reflections {
        Some(r) => {
            let times = r.reverb_times.map_or_else(|| "—".into(), |t| bands(t, 2));
            lines.push((
                if ru {
                    format!(
                        "Отражения: прогонов {}/{}, прогон {} мс, RT60 {times} с",
                        r.runs,
                        r.max_runs,
                        millis(r.last_run)
                    )
                } else {
                    format!(
                        "Reflections: runs {}/{}, run {} ms, RT60 {times} s",
                        r.runs,
                        r.max_runs,
                        millis(r.last_run)
                    )
                },
                TEXT,
            ));
            lines.push((
                if ru {
                    format!("Свои отражения: {} источн.", r.sources.len())
                } else {
                    format!("Own reflections: {} sources", r.sources.len())
                },
                TEXT,
            ));
        }
        None => lines.push((
            if ru {
                "Отражения: ещё нет"
            } else {
                "Reflections: none yet"
            }
            .to_owned(),
            DIM,
        )),
    }
    lines.push((
        if ru {
            format!(
                "Пути: проб {}, запекание {} мс",
                paths.probes,
                millis(paths.bake)
            )
        } else {
            format!(
                "Paths: {} probes, bake {} ms",
                paths.probes,
                millis(paths.bake)
            )
        },
        if paths.at.is_some() { TEXT } else { DIM },
    ));
    if let Some(scene) = scene {
        let hidden = scene
            .sources
            .iter()
            .filter(|s| s.acoustics.occlusion < 1.0)
            .count();
        let with_path = scene
            .sources
            .iter()
            .filter(|s| s.acoustics.path.is_some())
            .count();
        lines.push((
            if ru {
                format!(
                    "Окклюзия: {} источн., закрыто {}, с путём {}",
                    scene.sources.len(),
                    hidden,
                    with_path
                )
            } else {
                format!(
                    "Occlusion: {} sources, {} hidden, {} with a path",
                    scene.sources.len(),
                    hidden,
                    with_path
                )
            },
            TEXT,
        ));
        if scene.underwater {
            lines.push((
                if ru {
                    "Под водой"
                } else {
                    "Under water"
                }
                .to_owned(),
                0x60_C0_FF,
            ));
        }
    }
    lines
}

/// Where a world point is on the screen, in GUI pixels of a `width` × `height` GUI, for the camera-relative
/// `view_proj` (column-major) of a camera at `camera`; `None` behind the camera or off the screen.
pub fn project(
    view_proj: &[f32; 16],
    camera: [f64; 3],
    point: [f64; 3],
    width: f32,
    height: f32,
) -> Option<[f32; 2]> {
    let p: [f32; 3] = std::array::from_fn(|a| (point[a] - camera[a]) as f32);
    let m = view_proj;
    let row = |r: usize| m[r] * p[0] + m[4 + r] * p[1] + m[8 + r] * p[2] + m[12 + r];
    let (x, y, w) = (row(0), row(1), row(3));
    if w <= 0.05 {
        return None;
    }
    let (nx, ny) = (x / w, y / w);
    if nx.abs() > 1.05 || ny.abs() > 1.05 {
        return None;
    }
    Some([(nx * 0.5 + 0.5) * width, (0.5 - ny * 0.5) * height])
}

/// How far `ray` (local, inside `grid`) goes before it leaves the grid.
fn exit(grid: &Voxels, ray: &Ray) -> f32 {
    let size = grid.size();
    (0..3)
        .map(|a| {
            let d = ray.direction[a];
            if d > 1e-6 {
                (size[a] as f32 - ray.origin[a]) / d
            } else if d < -1e-6 {
                -ray.origin[a] / d
            } else {
                f32::INFINITY
            }
        })
        .fold(f32::INFINITY, f32::min)
        .max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::Surface;
    use crate::reflections::{GRID_SIZE, GridBuilder};

    #[test]
    fn reflection_rays_bounce_off_the_walls_of_a_room() {
        let listener = [0.5, 65.5, 0.5];
        // A stone room, 9 × 5 × 9 inside.
        let grid = Voxels::from_fn(GridBuilder::origin_for(listener), GRID_SIZE, |[x, y, z]| {
            let inside = (-4..=4).contains(&x) && (64..=68).contains(&y) && (-4..=4).contains(&z);
            (!inside).then_some(Surface::Rock)
        });
        let rays = reflection_rays(&grid, listener, 16, 3);
        assert_eq!(rays.len(), 16);
        for ray in &rays {
            // The listener, then three bounces and the point after the last, all inside the room's shell.
            assert_eq!(ray.len(), 5, "{ray:?}");
            assert_eq!(ray[0], listener);
            for p in &ray[1..] {
                assert!(
                    (-4.01..=5.01).contains(&p[0])
                        && (63.99..=69.01).contains(&p[1])
                        && (-4.01..=5.01).contains(&p[2]),
                    "{p:?}"
                );
            }
        }
        // In the open, a ray leaves the grid: the listener and where it left.
        let open = Voxels::from_fn(GridBuilder::origin_for(listener), GRID_SIZE, |_| None);
        let rays = reflection_rays(&open, listener, 4, 3);
        assert!(rays.iter().all(|r| r.len() == 2), "{rays:?}");
    }

    fn hidden_source() -> DebugSource {
        DebugSource {
            id: 7,
            at: [6.5, 64.5, 0.5],
            acoustics: SourceAcoustics {
                occlusion: 0.0,
                transmission: [0.2, 0.1, 0.05],
                through: [1.0; NUM_BANDS],
                path: Some(crate::paths::PathAcoustics {
                    direction: [0.0, 0.0, -1.0],
                    gain: 0.5,
                    eq: [0.9, 0.7, 0.4],
                    distance: 9.5,
                }),
            },
            trace: SourceTrace {
                rays: vec![([6.5, 64.5, 0.5], false), ([7.1, 64.5, 0.5], true)],
                walls: vec![[3, 64, 0]],
                filters: Vec::new(),
            },
            path_legs: vec![([1.5, 65.5, -2.5], [4.5, 65.5, -2.5])],
            own_reflections: true,
        }
    }

    fn scene() -> Scene {
        Scene {
            listener: Some([0.5, 65.6, 0.5]),
            underwater: false,
            sources: vec![hidden_source()],
            probes: vec![[1.5, 65.5, -2.5], [4.5, 65.5, -2.5]],
            grid: Some(([-48, 32, -48], GRID_SIZE)),
            reflection_rays: vec![vec![[0.5, 65.6, 0.5], [0.5, 70.0, 0.5], [0.5, 64.0, 0.5]]],
        }
    }

    #[test]
    fn each_layer_draws_its_own() {
        let scene = scene();
        let none = lines(&scene, Layers::default());
        assert!(none.is_empty());
        let only = |layers: Layers| lines(&scene, layers);
        let sources = only(Layers {
            sources: true,
            ..Layers::default()
        });
        // The listener's cross, the source's box, two rays, the wall's box.
        assert_eq!(sources.len(), 3 + 12 + 2 + 12);
        assert!(sources.contains(&Line {
            a: [0.5, 65.6, 0.5],
            b: [6.5, 64.5, 0.5],
            color: BLOCKED_RAY,
            width: 1.5,
        }));
        let paths = only(Layers {
            paths: true,
            ..Layers::default()
        });
        // The cross, two probes, the leg, where the path arrives from, the source's box.
        assert_eq!(paths.len(), 3 + 2 * 3 + 1 + 1 + 12);
        let arrival = paths.iter().find(|l| l.color == ARRIVAL).expect("arrival");
        assert!((arrival.b[2] - (0.5 - 3.0)).abs() < 1e-9, "{arrival:?}");
        let reflections = only(Layers {
            reflections: true,
            ..Layers::default()
        });
        // The cross, the grid's box, two legs of the ray, the own reflections' box.
        assert_eq!(reflections.len(), 3 + 12 + 2 + 12);
        // The panel alone draws nothing in the world.
        assert!(
            only(Layers {
                stats: true,
                ..Layers::default()
            })
            .is_empty()
        );
    }

    #[test]
    fn labels_and_the_panel_speak_the_language() {
        let scene = scene();
        let all = Layers {
            sources: true,
            paths: true,
            reflections: true,
            stats: true,
        };
        let stats = ReflectionStats {
            runs: 16,
            max_runs: 16,
            last_run: Some(Duration::from_millis(42)),
            reverb_times: Some([1.2, 0.9, 0.6]),
            sources: vec![(7, [6.5, 64.5, 0.5], 5, [1.0, 0.8, 0.5])],
            at: None,
        };
        let ru = labels(&scene, all, "ru_ru", Some(&[7]), Some(&stats));
        assert_eq!(ru.len(), 1);
        let text: Vec<&str> = ru[0].lines.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            text,
            [
                "#7 HRTF",
                "видно 0% · стены 1",
                "сквозь 0.200/0.100/0.050",
                "путь 50% · 9.5 бл. · EQ 0.90/0.70/0.40",
                "свои отражения: RT60 1.00/0.80/0.50 s (5)",
            ]
        );
        assert_eq!(ru[0].lines[1].1, 0xFF_00_33);
        let en = labels(
            &scene,
            Layers {
                sources: true,
                ..Layers::default()
            },
            "en_us",
            Some(&[]),
            None,
        );
        assert_eq!(en[0].lines[0].0, "#7 vanilla");
        assert_eq!(en[0].lines.len(), 3);

        let renderer = RendererStats {
            rate: 48_000,
            voices: 12,
            positional: 10,
            binaural: 8,
            paths: 1,
            budget: 32,
            limit: 24,
            load: 0.7,
            ..RendererStats::default()
        };
        let paths = PathStats {
            probes: 640,
            bake: Some(Duration::from_millis(180)),
            at: Some(Instant::now()),
        };
        let panel_ru = panel(Some(&scene), Some(&renderer), Some(&stats), &paths, "ru_ru");
        let text: Vec<&str> = panel_ru.iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            text.contains(&"Бинаурально: 8 (бюджет 32, сейчас 24), путей 1, слито 0"),
            "{text:?}"
        );
        assert!(
            text.contains(&"Аудиопоток: 70% кадра, 48000 Гц"),
            "{text:?}"
        );
        assert!(text.contains(&"Отражения: прогонов 16/16, прогон 42 мс, RT60 1.20/0.90/0.60 с"));
        assert!(text.contains(&"Пути: проб 640, запекание 180 мс"));
        assert!(text.contains(&"Окклюзия: 1 источн., закрыто 1, с путём 1"));
        let idle = panel(None, None, None, &PathStats::default(), "en_us");
        assert_eq!(idle.len(), 4, "{idle:?}");
        assert!(idle[1].0.starts_with("Audio thread: not rendering"));
    }

    #[test]
    fn points_are_projected_onto_the_gui() {
        // A camera at the origin looking down -z, 90° field of view, square: x/(-z), y/(-z).
        #[rustfmt::skip]
        let view_proj = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -1.0,
            0.0, 0.0, 0.1, 0.0,
        ];
        let camera = [10.0, 64.0, 10.0];
        let at = |p: [f64; 3]| project(&view_proj, camera, p, 200.0, 100.0);
        assert_eq!(at([10.0, 64.0, 5.0]), Some([100.0, 50.0]));
        assert_eq!(at([12.5, 66.5, 5.0]), Some([150.0, 25.0]));
        // Behind the camera, off the side.
        assert_eq!(at([10.0, 64.0, 15.0]), None);
        assert_eq!(at([30.0, 64.0, 5.0]), None);
    }

    #[test]
    fn the_audio_thread_statistics_go_stale() {
        publish_renderer(|s| s.voices = 3);
        assert_eq!(renderer().map(|s| s.voices), Some(3));
        publish_scene(Scene {
            listener: Some([1.0, 2.0, 3.0]),
            ..Scene::default()
        });
        assert_eq!(with_scene(|s| s.listener), Some(Some([1.0, 2.0, 3.0])));
        clear_scene();
        assert_eq!(with_scene(|s| s.listener), None);
    }
}
