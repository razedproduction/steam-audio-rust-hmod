//! Shortest paths through a visibility graph (upstream `path_finder.{h,cpp}`).

use std::sync::Mutex;

use crate::probes::Probe;
use crate::scene::RayTracer;
use crate::scene::msvc_sort;

use super::{ProbeVisibilityGraph, ProbeVisibilityTester};

/// A path from one probe to another as a sequence of probes (upstream `ProbePath`).
///
/// `nodes` are the probes between `start` and `end`, in order. Paths found by
/// [`PathFinder::find_shortest_path`] are an exception kept from upstream: their nodes begin with `start` itself.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProbePath {
    pub valid: bool,
    /// Upstream `-1` is `None`.
    pub start: Option<usize>,
    pub end: Option<usize>,
    pub nodes: Vec<usize>,
}

impl ProbePath {
    /// Makes the path invalid, without start, end or nodes (upstream `reset`).
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// An entry of the priority queue (upstream `PriorityQueueEntry`).
#[derive(Debug, Clone, Copy)]
struct QueueEntry {
    node: usize,
    cost: f32,
}

/// Upstream's `operator<`, which makes `std::priority_queue` pop the lowest cost first.
fn less(a: &QueueEntry, b: &QueueEntry) -> bool {
    a.cost > b.cost
}

/// Per-thread state of a search.
#[derive(Debug, Clone)]
struct Scratch {
    /// The predecessor of each probe; `-1` for none.
    parents: Vec<i32>,
    costs: Vec<f32>,
    /// A binary heap in the layout of MSVC's `std::priority_queue`, so probes of equal cost are taken in upstream's
    /// order (which decides between paths of equal length).
    queue: Vec<QueueEntry>,
}

impl Scratch {
    fn reset(&mut self, start: usize) {
        self.parents.fill(-1);
        self.costs.fill(f32::INFINITY);
        self.costs[start] = 0.0;
        self.queue.clear();
        self.push(start, 0.0);
    }

    fn push(&mut self, node: usize, cost: f32) {
        msvc_sort::push_heap(&mut self.queue, QueueEntry { node, cost }, less);
    }

    fn parent(&self, node: usize) -> Option<usize> {
        usize::try_from(self.parents[node]).ok()
    }
}

/// Finds shortest paths between probes over a [`ProbeVisibilityGraph`] (upstream `PathFinder`): from one probe to all
/// others with Dijkstra's algorithm when baking, and between two probes with A* at run time.
///
/// Each thread of a search has its own scratch space, indexed by the thread's index.
#[derive(Debug)]
pub struct PathFinder {
    scratch: Vec<Mutex<Scratch>>,
}

/// The parameters of [`PathFinder::find_shortest_path`] besides the graph, tester and endpoints.
#[derive(Debug, Clone, Copy)]
pub struct ShortestPathParams {
    /// Probe radius for volumetric visibility tests (see [`ProbeVisibilityTester::are_probes_visible`]).
    pub radius: f32,
    pub threshold: f32,
    /// Shortcut nodes: in `a, b, c`, drop `b` when `a` sees `c`.
    pub simplify_paths: bool,
    /// Test visibility with rays instead of trusting the graph.
    pub real_time_vis: bool,
}

impl PathFinder {
    /// A path finder for `num_probes` probes, searched from `num_threads` threads.
    pub fn new(num_probes: usize, num_threads: usize) -> Self {
        let scratch = Scratch {
            parents: vec![-1; num_probes],
            costs: vec![0.0; num_probes],
            queue: Vec::with_capacity(2 * num_probes),
        };
        Self {
            scratch: (0..num_threads)
                .map(|_| Mutex::new(scratch.clone()))
                .collect(),
        }
    }

    fn scratch(&self, thread_index: usize) -> std::sync::MutexGuard<'_, Scratch> {
        self.scratch[thread_index]
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    /// The shortest paths from `start` to every probe, no longer than `path_range` (upstream `findAllShortestPaths`).
    /// `paths[i]` gets start `start` and end `i`; it is valid if `i` was reached, and its nodes (which must be empty)
    /// get the probes in between. The path from `start` to itself is invalid.
    ///
    /// # Panics
    ///
    /// If `paths` or the graph has fewer entries than the finder has probes, or `thread_index` is out of range.
    pub fn find_all_shortest_paths(
        &self,
        graph: &ProbeVisibilityGraph,
        start: usize,
        path_range: f32,
        thread_index: usize,
        paths: &mut [ProbePath],
    ) {
        let mut scratch = self.scratch(thread_index);
        let scratch = &mut *scratch;
        scratch.reset(start);

        while !scratch.queue.is_empty() {
            let u = msvc_sort::pop_heap(&mut scratch.queue, less).node;

            for entry in graph.adjacent(u) {
                let v = entry.index;
                let new_cost = scratch.costs[u] + entry.cost;

                if new_cost > path_range {
                    continue;
                }

                if new_cost < scratch.costs[v] {
                    scratch.costs[v] = new_cost;
                    scratch.parents[v] = u as i32;
                    scratch.push(v, new_cost);
                }
            }
        }

        for (i, path) in paths.iter_mut().enumerate().take(scratch.parents.len()) {
            path.start = Some(start);
            path.end = Some(i);
            path.valid = scratch.parents[i] >= 0;

            if path.valid {
                let mut parent = scratch.parent(i);
                while let Some(p) = parent.filter(|&p| p != start) {
                    path.nodes.push(p);
                    parent = scratch.parent(p);
                }
                path.nodes.reverse();
            }
        }
    }

    /// The shortest path from `start` to `end` (upstream `findShortestPath`), found with A* over `graph`, optionally
    /// checking each edge with `tester` and shortening the result.
    ///
    /// Upstream's quirk is kept: the nodes of the path begin with `start`. A path from a probe to itself is invalid.
    ///
    /// # Panics
    ///
    /// If `start`, `end` or a probe of the graph is out of range of `probes` or of the finder, or `thread_index` is.
    #[allow(clippy::too_many_arguments)]
    pub fn find_shortest_path(
        &self,
        scene: &dyn RayTracer,
        probes: &[Probe],
        graph: &ProbeVisibilityGraph,
        tester: &ProbeVisibilityTester,
        start: usize,
        end: usize,
        params: &ShortestPathParams,
        thread_index: usize,
    ) -> ProbePath {
        let mut result = ProbePath {
            valid: false,
            start: Some(start),
            end: Some(end),
            nodes: Vec::new(),
        };

        let mut scratch = self.scratch(thread_index);
        let scratch = &mut *scratch;
        scratch.reset(start);

        let distance =
            |a: usize, b: usize| (probes[a].influence.center - probes[b].influence.center).length();

        while let Some(top) = scratch.queue.first() {
            let u = top.node;
            if u == end {
                break;
            }
            msvc_sort::pop_heap(&mut scratch.queue, less);

            for entry in graph.adjacent(u) {
                let v = entry.index;
                let new_cost = scratch.costs[u] + entry.cost;

                if scratch.costs[v] == f32::INFINITY || new_cost < scratch.costs[v] {
                    if params.real_time_vis
                        && !tester.are_probes_visible(
                            scene,
                            probes,
                            u,
                            v,
                            params.radius,
                            params.threshold,
                        )
                    {
                        continue;
                    }

                    scratch.costs[v] = new_cost;
                    scratch.parents[v] = u as i32;
                    scratch.push(v, new_cost + distance(v, end));
                }
            }
        }

        if scratch.parents[end] < 0 {
            return result;
        }

        if params.simplify_paths {
            simplify_path(
                scene,
                probes,
                graph,
                tester,
                start,
                end,
                params,
                &mut scratch.parents,
            );
        }

        let mut parent = scratch.parent(end);
        while let Some(p) = parent {
            result.nodes.push(p);
            parent = scratch.parent(p);
        }
        result.nodes.reverse();

        result.valid = true;
        result
    }
}

/// Upstream `simplifyPath`: walking back from `end`, while a probe sees its grandparent, the grandparent becomes its
/// parent.
#[allow(clippy::too_many_arguments)]
fn simplify_path(
    scene: &dyn RayTracer,
    probes: &[Probe],
    graph: &ProbeVisibilityGraph,
    tester: &ProbeVisibilityTester,
    start: usize,
    end: usize,
    params: &ShortestPathParams,
    parents: &mut [i32],
) {
    let parent = |parents: &[i32], node: usize| usize::try_from(parents[node]).ok();

    let mut current = Some(end);
    while let Some(c) = current.filter(|&c| c != start) {
        while let Some(p) = parent(parents, c) {
            let Some(grandparent) = parent(parents, p) else {
                break;
            };

            let visible = if params.real_time_vis {
                tester.are_probes_visible(
                    scene,
                    probes,
                    c,
                    grandparent,
                    params.radius,
                    params.threshold,
                )
            } else {
                graph.has_edge(c, grandparent)
            };
            if !visible {
                break;
            }

            parents[c] = grandparent as i32;
        }

        current = parent(parents, c);
    }
}
