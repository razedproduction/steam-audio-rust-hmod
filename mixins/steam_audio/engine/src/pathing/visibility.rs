//! Visibility between probes (upstream `path_visibility.{h,cpp}`).

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::error::{Error, Result};
use crate::geometry::Sphere;
use crate::job_graph::JobGraph;
use crate::math::Vector3;
use crate::probes::Probe;
use crate::sampling;
use crate::scene::RayTracer;
use crate::serialization::{Builder, Offset, Table};
use crate::thread_pool::ThreadPool;

// Vtable slots of `ipl.Serialized.VisibilityGraph` and `ipl.Serialized.VisibilityList`.
const VT_NODES: u16 = 4;
const VT_EDGES: u16 = 4;

/// Tests whether two probes see each other (upstream `ProbeVisibilityTester`).
#[derive(Debug, Clone)]
pub struct ProbeVisibilityTester {
    /// Points in the unit sphere; empty for point-to-point visibility.
    samples: Vec<Vector3>,
    asymmetric_vis_range: bool,
    down: Vector3,
}

impl ProbeVisibilityTester {
    /// A tester that traces rays between `num_samples` points around each probe, or between the probes' centers when
    /// `num_samples` is at most 1. With `asymmetric_vis_range`, distances are measured perpendicular to `down`.
    pub fn new(num_samples: usize, asymmetric_vis_range: bool, down: Vector3) -> Self {
        let mut samples = Vec::new();
        if num_samples > 1 {
            samples = vec![Vector3::ZERO; num_samples];
            sampling::sphere_volume_samples(&mut samples);
        }
        Self {
            samples,
            asymmetric_vis_range,
            down,
        }
    }

    /// Whether probes `from` and `to` see each other (upstream `areProbesVisible`).
    ///
    /// Point-to-point (no samples, or `radius` not positive): one ray between the centers. Volumetric: the probes are
    /// spheres of `radius`, rays are traced between every pair of sample points visible from their own probe's center,
    /// and the probes see each other once the visible pairs reach `threshold` times the number of samples.
    ///
    /// Panics if `from` or `to` is out of range.
    pub fn are_probes_visible(
        &self,
        scene: &dyn RayTracer,
        probes: &[Probe],
        from: usize,
        to: usize,
        radius: f32,
        threshold: f32,
    ) -> bool {
        let from_probe = probes[from].influence.center;
        let to_probe = probes[to].influence.center;

        let volumetric = !self.samples.is_empty() && radius > 0.0;
        if !volumetric {
            return !scene.is_occluded(from_probe, to_probe);
        }

        let mut num_visible_samples = 0u32;
        for &from_unit in &self.samples {
            let from_sample = sampling::transform_sphere_volume_sample(
                from_unit,
                &Sphere::new(from_probe, radius),
            );
            if scene.is_occluded(from_probe, from_sample) {
                continue;
            }

            for &to_unit in &self.samples {
                let to_sample = sampling::transform_sphere_volume_sample(
                    to_unit,
                    &Sphere::new(to_probe, radius),
                );
                if scene.is_occluded(to_probe, to_sample) {
                    continue;
                }

                if !scene.is_occluded(from_sample, to_sample) {
                    num_visible_samples += 1;
                    if num_visible_samples as f32 / self.samples.len() as f32 >= threshold {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Whether probes `from` and `to` are farther apart than `vis_range` (upstream `areProbesTooFar`). Panics if
    /// either is out of range.
    pub fn are_probes_too_far(
        &self,
        probes: &[Probe],
        from: usize,
        to: usize,
        vis_range: f32,
    ) -> bool {
        let mut d = probes[from].influence.center - probes[to].influence.center;
        if self.asymmetric_vis_range {
            d = d - d.dot(self.down) * self.down;
        }
        d.length() > vis_range
    }
}

/// An edge of a [`ProbeVisibilityGraph`] (upstream `AdjacencyListEntry`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdjacencyEntry {
    /// The probe at the other end.
    pub index: usize,
    /// The distance between the probes.
    pub cost: f32,
}

/// Which probes see each other: an undirected graph over the probes of a batch, stored as adjacency lists (upstream
/// `ProbeVisibilityGraph`).
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeVisibilityGraph {
    adjacent: Vec<Vec<AdjacencyEntry>>,
}

/// The parameters of [`ProbeVisibilityGraph::new`].
#[derive(Debug, Clone, Copy)]
pub struct VisibilityParams {
    /// Probe radius for volumetric visibility (see [`ProbeVisibilityTester::are_probes_visible`]).
    pub radius: f32,
    pub threshold: f32,
    /// Probes farther apart than this are not connected.
    pub vis_range: f32,
}

impl ProbeVisibilityGraph {
    /// Builds the graph of the probes within `vis_range` of each other that see each other, one job per probe on
    /// `pool` (upstream's constructor followed by `ThreadPool::process`).
    ///
    /// Each probe's list holds the probes before it in index order, then those after it in index order.
    pub fn new(
        scene: &dyn RayTracer,
        probes: &[Probe],
        tester: &ProbeVisibilityTester,
        params: VisibilityParams,
        pool: &ThreadPool,
    ) -> Self {
        Self::build(
            scene,
            probes,
            tester,
            params,
            pool,
            &AtomicBool::new(false),
            &mut |_| {},
        )
        .expect("not cancelled")
    }

    /// [`new`](Self::new), reporting the fraction of jobs done to `progress`. Returns `None` if `cancel` is set before
    /// every job has run (upstream leaves such a graph without its back edges).
    pub(crate) fn build(
        scene: &dyn RayTracer,
        probes: &[Probe],
        tester: &ProbeVisibilityTester,
        params: VisibilityParams,
        pool: &ThreadPool,
        cancel: &AtomicBool,
        progress: &mut dyn FnMut(f32),
    ) -> Option<Self> {
        let rows: Vec<Mutex<Vec<AdjacencyEntry>>> =
            (0..probes.len()).map(|_| Mutex::new(Vec::new())).collect();
        let num_jobs_done = AtomicUsize::new(0);

        let mut graph = JobGraph::new();
        for (i, row) in rows.iter().enumerate() {
            let num_jobs_done = &num_jobs_done;
            graph.add_job(move |_, _| {
                if cancel.load(Ordering::SeqCst) {
                    return;
                }
                let mut row = row.lock().unwrap_or_else(|e| e.into_inner());
                for j in 0..i {
                    if tester.are_probes_too_far(probes, i, j, params.vis_range) {
                        continue;
                    }
                    if !tester.are_probes_visible(
                        scene,
                        probes,
                        i,
                        j,
                        params.radius,
                        params.threshold,
                    ) {
                        continue;
                    }
                    let cost = (probes[i].influence.center - probes[j].influence.center).length();
                    row.push(AdjacencyEntry { index: j, cost });
                }
                num_jobs_done.fetch_add(1, Ordering::SeqCst);
            });
        }
        pool.process_with_progress(&mut graph, progress);
        drop(graph);

        if num_jobs_done.into_inner() != probes.len() {
            return None;
        }

        let mut adjacent: Vec<Vec<AdjacencyEntry>> = rows
            .into_iter()
            .map(|row| row.into_inner().unwrap_or_else(|e| e.into_inner()))
            .collect();
        // Upstream's last job adds the back edges, row by row.
        for i in 0..adjacent.len() {
            for k in 0..adjacent[i].len() {
                let entry = adjacent[i][k];
                adjacent[entry.index].push(AdjacencyEntry {
                    index: i,
                    cost: entry.cost,
                });
            }
        }

        Some(Self { adjacent })
    }

    pub fn num_probes(&self) -> usize {
        self.adjacent.len()
    }

    /// The edges of probe `index` (upstream `mAdjacent[index]`). Panics if `index` is out of range.
    pub fn adjacent(&self, index: usize) -> &[AdjacencyEntry] {
        &self.adjacent[index]
    }

    /// Sets every edge's cost to the distance between its probes (upstream `updateCosts`). Panics if `probes` has
    /// fewer probes than the graph.
    pub fn update_costs(&mut self, probes: &[Probe]) {
        for (i, row) in self.adjacent.iter_mut().enumerate() {
            for entry in row {
                entry.cost =
                    (probes[i].influence.center - probes[entry.index].influence.center).length();
            }
        }
    }

    /// Whether the graph has an edge from `from` to `to` (upstream `hasEdge`). Panics if `from` is out of range.
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        self.adjacent[from].iter().any(|entry| entry.index == to)
    }

    /// Removes the edges between probes farther apart than `vis_range` (upstream `prune`).
    pub fn prune(&mut self, probes: &[Probe], tester: &ProbeVisibilityTester, vis_range: f32) {
        for (i, row) in self.adjacent.iter_mut().enumerate() {
            row.retain(|entry| !tester.are_probes_too_far(probes, i, entry.index, vis_range));
        }
    }

    /// Edges to probes with a lower index: the half of the graph that is serialized.
    fn lower_edges(&self, i: usize) -> impl Iterator<Item = usize> + '_ {
        self.adjacent[i]
            .iter()
            .map(|entry| entry.index)
            .filter(move |&j| j < i)
    }

    /// Upstream `serializedSize`: the probe count, then per probe its edge count and its edges to lower indices.
    pub fn serialized_size(&self) -> u64 {
        let mut size = 4;
        for i in 0..self.adjacent.len() {
            size += 4 + 4 * self.lower_edges(i).count() as u64;
        }
        size
    }

    /// Upstream `serialize`.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let lists: Vec<Offset> = (0..self.adjacent.len())
            .map(|i| {
                let edges: Vec<usize> = self.lower_edges(i).collect();
                let edges = builder.create_vector(edges.len(), 4, |bytes| {
                    for (out, &j) in bytes.chunks_exact_mut(4).zip(&edges) {
                        out.copy_from_slice(&(j as i32).to_le_bytes());
                    }
                });
                let start = builder.start_table();
                builder.add_offset(VT_EDGES, Some(edges));
                builder.end_table(start)
            })
            .collect();
        let lists = builder.create_vector_of_offsets(&lists);

        let start = builder.start_table();
        builder.add_offset(VT_NODES, Some(lists));
        builder.end_table(start)
    }

    /// Upstream's deserializing constructor. Costs are 0 until [`update_costs`](Self::update_costs). Edges to probes
    /// with a lower index get their back edges; others are kept as they are (upstream writes none).
    pub(crate) fn from_table(table: &Table<'_>) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;

        let nodes = table
            .vector(VT_NODES, 4)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("visibility graph without probes"))?;
        let num_probes = nodes.len();

        let mut adjacent = Vec::with_capacity(num_probes);
        for i in 0..num_probes {
            let edges = nodes
                .table(i)?
                .vector(VT_EDGES, 4)?
                .ok_or(malformed("visibility graph probe without edges"))?;
            let row = edges
                .i32s()
                .map(|j| {
                    usize::try_from(j)
                        .ok()
                        .filter(|&j| j < num_probes)
                        .map(|index| AdjacencyEntry { index, cost: 0.0 })
                        .ok_or(malformed("visibility graph edge to a missing probe"))
                })
                .collect::<Result<Vec<_>>>()?;
            adjacent.push(row);
        }

        for i in 0..num_probes {
            for k in 0..adjacent[i].len() {
                let j = adjacent[i][k].index;
                if j < i {
                    adjacent[j].push(AdjacencyEntry {
                        index: i,
                        cost: 0.0,
                    });
                }
            }
        }

        Ok(Self { adjacent })
    }
}
