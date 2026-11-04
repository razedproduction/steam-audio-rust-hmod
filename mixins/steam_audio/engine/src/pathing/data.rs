//! Baked paths between probes (upstream `path_data.{h,cpp}`: `SoundPath`, `SoundPathRef`, `BakedPathData`).

use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{Error, Result};
use crate::job_graph::JobGraph;
use crate::math::Vector3;
use crate::probes::Probe;
use crate::scene::RayTracer;
use crate::serialization::{Builder, Offset, Table};
use crate::thread_pool::ThreadPool;

use super::{PathFinder, ProbePath, ProbeVisibilityGraph, ProbeVisibilityTester, VisibilityParams};

// Vtable slots of `ipl.Serialized.SoundPath`.
const VT_FIRST_PROBE: u16 = 4;
const VT_LAST_PROBE: u16 = 6;
const VT_PROBE_AFTER_FIRST: u16 = 8;
const VT_PROBE_BEFORE_LAST: u16 = 10;
const VT_DIRECT: u16 = 12;
const VT_DISTANCE_INTERNAL: u16 = 14;
const VT_DEVIATION_INTERNAL: u16 = 16;

// Vtable slots of `ipl.Serialized.BakedPathingData`.
const VT_VIS_GRAPH: u16 = 4;
const VT_UNIQUE_PATHS: u16 = 6;
const VT_PATH_INDICES: u16 = 8;
const VT_PATHS: u16 = 10;

/// `sizeof(SoundPath)` in upstream's serialized size estimate: four `int16_t`, a `bool` and two `float`s.
const SOUND_PATH_SIZE: u64 = 20;

/// What is needed to turn a path between probes into a virtual source (upstream `SoundPath`).
///
/// Probe indices are `i16` as upstream stores them; `-1` means none. A path between two probes that see each other is
/// `direct`; otherwise `first_probe` and `last_probe` are the probes after the start and before the end.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoundPath {
    pub first_probe: i16,
    pub last_probe: i16,
    /// The probe after `first_probe`, if the path has at least two probes in between.
    pub probe_after_first: i16,
    /// The probe before `last_probe`, if the path has at least two probes in between.
    pub probe_before_last: i16,
    pub direct: bool,
    /// Length of the path from `first_probe` to `last_probe`.
    pub distance_internal: f32,
    /// Sum of the angles the path turns by between `first_probe` and `last_probe`.
    pub deviation_internal: f32,
}

impl Default for SoundPath {
    /// An invalid path.
    fn default() -> Self {
        Self {
            first_probe: -1,
            last_probe: -1,
            probe_after_first: -1,
            probe_before_last: -1,
            direct: false,
            distance_internal: 0.0,
            deviation_internal: 0.0,
        }
    }
}

fn center(probes: &[Probe], index: impl TryInto<usize>) -> Vector3 {
    let index = index.try_into().ok().expect("a valid probe index");
    probes[index].influence.center
}

/// The angle a path turns by at `cur` (upstream's `angleBetween` of the two unit directions).
fn turn(prev: Vector3, cur: Vector3, next: Vector3) -> f32 {
    let prev_dir = (cur - prev).normalize_or_zero();
    let next_dir = (next - cur).normalize_or_zero();
    Vector3::angle_between(prev_dir, next_dir)
}

impl SoundPath {
    /// The sound path of a probe path (upstream's constructor from a `ProbePath`): direct if it has no nodes. Probe
    /// indices are truncated to `i16` as upstream does. Panics if a node is out of range of `probes`.
    pub fn from_probe_path(path: &ProbePath, probes: &[Probe]) -> Self {
        let mut result = Self::default();
        if !path.valid {
            return result;
        }

        let nodes = &path.nodes;
        let (Some(&first), Some(&last)) = (nodes.first(), nodes.last()) else {
            result.direct = true;
            return result;
        };

        result.first_probe = first as i16;
        result.last_probe = last as i16;
        if nodes.len() >= 2 {
            result.probe_after_first = nodes[1] as i16;
            result.probe_before_last = nodes[nodes.len() - 2] as i16;
        }

        for pair in nodes.windows(2) {
            result.distance_internal +=
                (center(probes, pair[1]) - center(probes, pair[0])).length();
        }

        for triple in nodes.windows(3) {
            result.deviation_internal += turn(
                center(probes, triple[0]),
                center(probes, triple[1]),
                center(probes, triple[2]),
            );
        }

        result
    }

    /// Upstream `isValid`.
    pub fn is_valid(&self) -> bool {
        self.direct || (self.first_probe >= 0 && self.last_probe >= 0)
    }

    /// Length of the path from probe `start` to probe `end` (upstream `distance(probes, int, int)`).
    ///
    /// Panics if the path is invalid or a probe is out of range.
    pub fn distance(&self, probes: &[Probe], start: usize, end: usize) -> f32 {
        assert!(self.is_valid(), "the path is invalid");
        if self.direct {
            (center(probes, end) - center(probes, start)).length()
        } else {
            let mut result = self.distance_internal;
            result += (center(probes, self.first_probe) - center(probes, start)).length();
            result += (center(probes, end) - center(probes, self.last_probe)).length();
            result
        }
    }

    /// Length of the path from `source` to probe `end` (upstream `distance(probes, Vector3f, int)`): for a direct
    /// path, only the internal distance (0 for baked paths), as upstream.
    ///
    /// Panics if the path is invalid or a probe is out of range.
    pub fn distance_from(&self, probes: &[Probe], source: Vector3, end: usize) -> f32 {
        assert!(self.is_valid(), "the path is invalid");
        let mut result = self.distance_internal;
        if !self.direct {
            result += (center(probes, self.first_probe) - source).length();
            result += (center(probes, end) - center(probes, self.last_probe)).length();
        }
        result
    }

    /// Total angle the path turns by from probe `start` to probe `end` (upstream `deviation`).
    ///
    /// Panics if the path is invalid or a probe is out of range.
    pub fn deviation(&self, probes: &[Probe], start: usize, end: usize) -> f32 {
        assert!(self.is_valid(), "the path is invalid");
        let mut result = self.deviation_internal;
        if self.direct {
            return result;
        }

        let first = center(probes, self.first_probe);
        let last = center(probes, self.last_probe);
        if self.probe_after_first < 0 && self.probe_before_last < 0 {
            result += turn(center(probes, start), first, center(probes, end));
        } else {
            if self.probe_after_first >= 0 {
                result += turn(
                    center(probes, start),
                    first,
                    center(probes, self.probe_after_first),
                );
            }
            if self.probe_before_last >= 0 {
                result += turn(
                    center(probes, self.probe_before_last),
                    last,
                    center(probes, end),
                );
            }
        }
        result
    }

    /// A virtual source for the path from probe `start` to probe `end`: as far from `end` as the path is long, in the
    /// direction of the probe before `end` (upstream `toVirtualSource(probes, int, int)`).
    ///
    /// Panics if the path is invalid and not direct, or a probe is out of range.
    pub fn to_virtual_source(&self, probes: &[Probe], start: usize, end: usize) -> Vector3 {
        if self.direct {
            return center(probes, start);
        }
        let total_distance = self.distance(probes, start, end);
        let direction = (center(probes, self.last_probe) - center(probes, end)).normalize_or_zero();
        center(probes, end) + total_distance * direction
    }

    /// [`to_virtual_source`](Self::to_virtual_source) from `source` instead of a probe (upstream
    /// `toVirtualSource(probes, Vector3f, int)`).
    pub fn to_virtual_source_from(&self, probes: &[Probe], source: Vector3, end: usize) -> Vector3 {
        if self.direct {
            return source;
        }
        let total_distance = self.distance_from(probes, source, end);
        let direction = (center(probes, self.last_probe) - center(probes, end)).normalize_or_zero();
        center(probes, end) + total_distance * direction
    }

    fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let start = builder.start_table();
        builder.add_f32(VT_DEVIATION_INTERNAL, self.deviation_internal, 0.0);
        builder.add_f32(VT_DISTANCE_INTERNAL, self.distance_internal, 0.0);
        builder.add_i16(VT_PROBE_BEFORE_LAST, self.probe_before_last, 0);
        builder.add_i16(VT_PROBE_AFTER_FIRST, self.probe_after_first, 0);
        builder.add_i16(VT_LAST_PROBE, self.last_probe, 0);
        builder.add_i16(VT_FIRST_PROBE, self.first_probe, 0);
        builder.add_i8(VT_DIRECT, i8::from(self.direct), 0);
        builder.end_table(start)
    }

    fn from_table(table: &Table<'_>, num_probes: usize) -> Result<Self> {
        let path = Self {
            first_probe: table.i16(VT_FIRST_PROBE, 0)?,
            last_probe: table.i16(VT_LAST_PROBE, 0)?,
            probe_after_first: table.i16(VT_PROBE_AFTER_FIRST, 0)?,
            probe_before_last: table.i16(VT_PROBE_BEFORE_LAST, 0)?,
            direct: table.i8(VT_DIRECT, 0)? != 0,
            distance_internal: table.f32(VT_DISTANCE_INTERNAL, 0.0)?,
            deviation_internal: table.f32(VT_DEVIATION_INTERNAL, 0.0)?,
        };
        let probes = [
            path.first_probe,
            path.last_probe,
            path.probe_after_first,
            path.probe_before_last,
        ];
        if probes.iter().any(|&p| i64::from(p) >= num_probes as i64) {
            return Err(Error::MalformedSerializedData(
                "sound path through a missing probe",
            ));
        }
        Ok(path)
    }
}

/// The parameters of a pathing bake (the arguments of upstream `PathBaker::bake` besides the scene, threads and
/// batch).
#[derive(Debug, Clone, Copy)]
pub struct PathBakeParams {
    /// Points per probe for visibility tests; at most 1 for point-to-point visibility.
    pub num_samples: usize,
    /// Probe radius for volumetric visibility.
    pub radius: f32,
    /// Fraction of sample pairs that must see each other.
    pub threshold: f32,
    /// Probes farther apart than this are not connected.
    pub vis_range: f32,
    /// Range the graph is pruned to for run-time searches, if `prune_vis_graph`.
    pub vis_range_real_time: f32,
    /// Paths longer than this are not baked.
    pub path_range: f32,
    /// Measure ranges perpendicular to `down`.
    pub asymmetric_vis_range: bool,
    pub down: Vector3,
    pub prune_vis_graph: bool,
}

/// Paths baked between every pair of probes of a batch (upstream `BakedPathData`): the visibility graph, the distinct
/// [`SoundPath`]s, and for every pair `(start, end)` with `end < start` the index of its path (upstream
/// `SoundPathRef`); the other half is looked up reversed.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedPathData {
    vis_graph: ProbeVisibilityGraph,
    unique_paths: Vec<SoundPath>,
    /// `num_probes * num_probes` indices into `unique_paths`, row-major by start.
    path_refs: Vec<u32>,
    num_probes: usize,
    needs_update: bool,
}

impl BakedPathData {
    /// Bakes paths (upstream's constructor): builds the visibility graph, then finds the shortest paths from every
    /// probe, both one job per probe on `pool`, then keeps the distinct paths and optionally prunes the graph.
    ///
    /// `progress` gets the fraction of jobs done in each phase, then of probes indexed, then 1. Returns `None` once
    /// `cancel` is found set (upstream returns a half-built object).
    pub(crate) fn bake(
        scene: &dyn RayTracer,
        probes: &[Probe],
        params: &PathBakeParams,
        pool: &ThreadPool,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(f32),
    ) -> Option<Self> {
        let n = probes.len();
        let cancelled = || cancel.load(Ordering::SeqCst);

        let tester = ProbeVisibilityTester::new(
            params.num_samples,
            params.asymmetric_vis_range,
            params.down,
        );
        let mut vis_graph = ProbeVisibilityGraph::build(
            scene,
            probes,
            &tester,
            VisibilityParams {
                radius: params.radius,
                threshold: params.threshold,
                vis_range: params.vis_range,
            },
            pool,
            cancel,
            progress,
        )?;
        if cancelled() {
            return None;
        }

        // Shortest paths from every probe; `probe_paths[i * n + j]` is the path from i to j.
        let finder = PathFinder::new(n, pool.num_threads());
        let mut probe_paths = vec![ProbePath::default(); n * n];
        {
            let mut jobs = JobGraph::new();
            for (i, row) in probe_paths.chunks_mut(n.max(1)).enumerate().take(n) {
                let (finder, vis_graph) = (&finder, &vis_graph);
                jobs.add_job(move |thread_index, _| {
                    if cancel.load(Ordering::SeqCst) {
                        return;
                    }
                    finder.find_all_shortest_paths(
                        vis_graph,
                        i,
                        params.path_range,
                        thread_index,
                        row,
                    );
                });
            }
            pool.process_with_progress(&mut jobs, &mut *progress);
        }

        // Paths with end > start are dropped: they are the reverses of paths kept.
        for i in 0..n {
            for j in i + 1..n {
                probe_paths[i * n + j] = ProbePath::default();
            }
        }

        if cancelled() {
            return None;
        }

        // Upstream sorts the paths themselves with `std::sort`. Equal paths have equal nodes, so the order among them
        // does not change the result, and sorting indices stably gives the same distinct paths in the same order.
        let mut order: Vec<usize> = (0..n * n).collect();
        order.sort_by(|&a, &b| {
            let (a, b) = (&probe_paths[a], &probe_paths[b]);
            if path_less(a, b) {
                std::cmp::Ordering::Less
            } else if path_less(b, a) {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });

        if cancelled() {
            return None;
        }

        let mut path_refs = vec![0; n * n];
        let mut unique_paths: Vec<SoundPath> = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let index = i * n + j;
                let path = &probe_paths[order[index]];
                if index == 0 || !paths_equal(path, &probe_paths[order[index - 1]]) {
                    unique_paths.push(SoundPath::from_probe_path(path, probes));
                }

                if unique_paths.last().is_some_and(SoundPath::is_valid) {
                    let (Some(start), Some(end)) = (path.start, path.end) else {
                        unreachable!("valid paths have endpoints");
                    };
                    path_refs[start * n + end] = (unique_paths.len() - 1) as u32;
                }
            }

            if cancelled() {
                return None;
            }

            progress(((i + 1) * n) as f32 / (n * n) as f32);
        }
        progress(1.0);

        if params.prune_vis_graph {
            vis_graph.prune(probes, &tester, params.vis_range_real_time);
        }

        if cancelled() {
            return None;
        }

        Some(Self {
            vis_graph,
            unique_paths,
            path_refs,
            num_probes: n,
            needs_update: false,
        })
    }

    pub fn num_probes(&self) -> usize {
        self.num_probes
    }

    /// The visibility graph (pruned if the bake asked for it).
    pub fn vis_graph(&self) -> &ProbeVisibilityGraph {
        &self.vis_graph
    }

    /// Whether probes changed since the bake: the data must be baked again.
    pub fn needs_update(&self) -> bool {
        self.needs_update
    }

    /// Sets the graph's costs to the distances between `probes` (upstream `updateVisGraphCosts`).
    pub fn update_vis_graph_costs(&mut self, probes: &[Probe]) {
        self.vis_graph.update_costs(probes);
    }

    pub(crate) fn mark_needs_update(&mut self) {
        self.needs_update = true;
    }

    /// The baked path from probe `start` to probe `end` (upstream `lookupShortestPath` without a probe path); invalid
    /// if there is none. Panics if either probe is out of range.
    pub fn lookup_shortest_path(&self, start: usize, end: usize) -> SoundPath {
        assert!(
            start < self.num_probes && end < self.num_probes,
            "probe index out of range"
        );
        if start < end {
            let mut path =
                self.unique_paths[self.path_refs[end * self.num_probes + start] as usize];
            std::mem::swap(&mut path.first_probe, &mut path.last_probe);
            std::mem::swap(&mut path.probe_after_first, &mut path.probe_before_last);
            path
        } else {
            self.unique_paths[self.path_refs[start * self.num_probes + end] as usize]
        }
    }

    /// [`lookup_shortest_path`](Self::lookup_shortest_path), and the probe path it stands for, rebuilt by looking up
    /// the path to each probe before `end` in turn (upstream `lookupShortestPath` with a probe path).
    ///
    /// The probe path is invalid if the sound path is, or one of the paths looked up is. Baked data never has a
    /// cycle; for malformed data that has one, the probe path is reset after as many steps as the batch has probes
    /// (upstream loops forever).
    pub fn lookup_probe_path(&self, start: usize, end: usize) -> (SoundPath, ProbePath) {
        let sound_path = self.lookup_shortest_path(start, end);
        let probe_path = self.reconstruct_probe_path(start, end, &sound_path);
        (sound_path, probe_path)
    }

    /// Upstream `reconstructProbePath`.
    fn reconstruct_probe_path(
        &self,
        start: usize,
        end: usize,
        sound_path: &SoundPath,
    ) -> ProbePath {
        let mut probe_path = ProbePath::default();
        if !sound_path.is_valid() {
            return probe_path;
        }

        probe_path.valid = true;
        probe_path.start = Some(start);
        probe_path.end = Some(end);

        let previous = |path: &SoundPath| {
            if path.direct {
                start
            } else {
                path.last_probe as usize
            }
        };

        let mut current = end;
        let mut prev = previous(sound_path);
        let mut steps = 0;
        while current != start {
            if current != end {
                probe_path.nodes.push(current);
            }

            // A path visits each probe at most once.
            steps += 1;
            if steps > self.num_probes {
                probe_path.reset();
                return probe_path;
            }

            if prev == start {
                break;
            }

            let next_path = self.lookup_shortest_path(start, prev);
            if !next_path.is_valid() {
                probe_path.reset();
                return probe_path;
            }

            current = prev;
            prev = previous(&next_path);
        }

        probe_path.nodes.reverse();
        probe_path
    }

    /// Whether the pair `index` (row-major) has a valid path: the ones serialized.
    fn has_valid_path(&self, index: usize) -> bool {
        self.unique_paths[self.path_refs[index] as usize].is_valid()
    }

    /// Upstream `serializedSize`: an estimate without the FlatBuffers overhead.
    pub fn serialized_size(&self) -> u64 {
        let mut size = 4 + self.vis_graph.serialized_size() + 4 + 4;
        size += self.unique_paths.len() as u64 * SOUND_PATH_SIZE;
        let num_valid = (0..self.path_refs.len())
            .filter(|&index| self.has_valid_path(index))
            .count();
        size += num_valid as u64 * 8;
        size
    }

    /// Upstream `serialize`.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let vis_graph = self.vis_graph.serialize_into(builder);

        let paths: Vec<Offset> = self
            .unique_paths
            .iter()
            .map(|path| path.serialize_into(builder))
            .collect();
        let paths = builder.create_vector_of_offsets(&paths);

        let valid: Vec<usize> = (0..self.path_refs.len())
            .filter(|&index| self.has_valid_path(index))
            .collect();
        let write_i32s = |values: Vec<i32>| {
            move |bytes: &mut [u8]| {
                for (out, value) in bytes.chunks_exact_mut(4).zip(values) {
                    out.copy_from_slice(&value.to_le_bytes());
                }
            }
        };
        let path_indices = builder.create_vector(
            valid.len(),
            4,
            write_i32s(valid.iter().map(|&index| index as i32).collect()),
        );
        let refs = builder.create_vector(
            valid.len(),
            4,
            write_i32s(
                valid
                    .iter()
                    .map(|&index| self.path_refs[index] as i32)
                    .collect(),
            ),
        );

        let start = builder.start_table();
        builder.add_offset(VT_PATHS, Some(refs));
        builder.add_offset(VT_PATH_INDICES, Some(path_indices));
        builder.add_offset(VT_UNIQUE_PATHS, Some(paths));
        builder.add_offset(VT_VIS_GRAPH, Some(vis_graph));
        builder.end_table(start)
    }

    /// Upstream's deserializing constructor, for a batch of `probes`, followed by upstream `ProbeBatch`'s call to
    /// [`update_vis_graph_costs`](Self::update_vis_graph_costs).
    ///
    /// Data for another number of probes, indices out of range and fewer path indices than paths are errors (upstream
    /// reads out of bounds).
    pub(crate) fn from_table(table: &Table<'_>, probes: &[Probe]) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;

        let mut vis_graph = ProbeVisibilityGraph::from_table(
            &table
                .table(VT_VIS_GRAPH)?
                .ok_or(malformed("baked paths without a visibility graph"))?,
        )?;
        let n = vis_graph.num_probes();
        if n != probes.len() {
            return Err(malformed("baked paths for another number of probes"));
        }

        let serialized_paths = table
            .vector(VT_UNIQUE_PATHS, 4)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("baked paths without paths"))?;
        let unique_paths = (0..serialized_paths.len())
            .map(|i| SoundPath::from_table(&serialized_paths.table(i)?, n))
            .collect::<Result<Vec<_>>>()?;

        let path_indices = table
            .vector(VT_PATH_INDICES, 4)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("baked paths without path indices"))?;
        let refs = table
            .vector(VT_PATHS, 4)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("baked paths without path references"))?;
        if path_indices.len() < refs.len() {
            return Err(malformed("baked paths with fewer indices than paths"));
        }

        let mut path_refs = vec![0; n * n];
        for (index, path) in path_indices.i32s().zip(refs.i32s()) {
            let index = usize::try_from(index)
                .ok()
                .filter(|&index| index < n * n)
                .ok_or(malformed("baked path for a missing pair of probes"))?;
            path_refs[index] = u32::try_from(path)
                .ok()
                .filter(|&path| (path as usize) < unique_paths.len())
                .ok_or(malformed("reference to a missing baked path"))?;
        }

        vis_graph.update_costs(probes);

        Ok(Self {
            vis_graph,
            unique_paths,
            path_refs,
            num_probes: n,
            needs_update: false,
        })
    }
}

/// Upstream's `compareProbePaths`: invalid paths first, then by nodes. Invalid paths have no nodes, so this is a strict
/// weak ordering.
fn path_less(a: &ProbePath, b: &ProbePath) -> bool {
    if !a.valid && b.valid {
        return true;
    }
    a.nodes < b.nodes
}

/// Upstream's `areProbePathsEqual`.
fn paths_equal(a: &ProbePath, b: &ProbePath) -> bool {
    a.valid == b.valid && a.nodes == b.nodes
}
