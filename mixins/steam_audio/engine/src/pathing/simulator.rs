//! Paths from a source to the listener through baked probe paths (upstream `path_simulator.{h,cpp}`).

use std::borrow::Borrow;

use crate::baking::{BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation};
use crate::bands::NUM_BANDS;
use crate::deviation::DeviationModel;
use crate::distance_attenuation::DistanceAttenuationModel;
use crate::effects::EqEffect;
use crate::geometry::Sphere;
use crate::math::{Vector3, cpp_max};
use crate::probes::{Probe, ProbeBatch, ProbeNeighborhood};
use crate::scene::RayTracer;
use crate::sh;

use super::{BakedPathData, PathFinder, ProbeVisibilityTester, ShortestPathParams, SoundPath};

/// Most paths a simulation combines (upstream `kMaxPaths`): one per pair of source and listener probes.
const MAX_PATHS: usize =
    ProbeNeighborhood::MAX_PROBES_PER_BATCH * ProbeNeighborhood::MAX_PROBES_PER_BATCH;

/// Called with the two ends of each ray traced to validate a baked path, and whether it was occluded (upstream
/// `ValidationRayVisualizationCallback`).
pub type ValidationRayCallback<'a> = &'a mut dyn FnMut(Vector3, Vector3, bool);

/// The inputs of [`PathSimulator::find_paths`] besides the scene, probes and neighbourhoods.
#[derive(Debug, Clone, Copy)]
pub struct PathSimulationParams<'a> {
    pub source: Vector3,
    pub listener: Vector3,
    /// Probe radius for volumetric visibility tests (see [`ProbeVisibilityTester::are_probes_visible`]).
    pub radius: f32,
    pub threshold: f32,
    /// Unused, as upstream's run-time path search ignores it.
    pub vis_range: f32,
    /// Ambisonics order of the SH coefficients.
    pub order: usize,
    /// Test each leg of a baked path with rays, and drop the path if one is occluded.
    pub enable_validation: bool,
    /// Search the visibility graph again for a path that validation dropped.
    pub find_alternate_paths: bool,
    /// See [`ShortestPathParams::simplify_paths`].
    pub simplify_paths: bool,
    /// See [`ShortestPathParams::real_time_vis`].
    pub real_time_vis: bool,
    pub distance_attenuation_model: &'a DistanceAttenuationModel,
    pub deviation_model: &'a DeviationModel,
    /// Combine the paths from every probe around the source, weighted, instead of only from the nearest one
    /// (upstream's global `PathSimulator::sEnablePathsFromAllSourceProbes`).
    pub paths_from_all_source_probes: bool,
    /// Look for paths even if the source and listener see each other.
    pub force_direct_occlusion: bool,
}

/// The sound field of the paths found, besides its SH coefficients (upstream `findPaths`' outputs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathSimulationResult {
    /// The paths' EQ, weighted: from the deviation model, normalized per path, or 1 for a direct path. 1 if no path
    /// was found.
    pub eq_gains: [f32; NUM_BANDS],
    /// Direction from the listener to the paths' virtual sources, weighted by inverse distance (zero if no path
    /// was found).
    pub direction: Vector3,
    /// Weighted ratio of the direct distance from the source to each path's distance (1 for direct paths and for
    /// distances up to 1 m; 0 if no path was found).
    pub distance_ratio: f32,
    /// Weighted total deviation of the paths through probes.
    pub total_deviation: f32,
}

/// A path found, its weight, and its start and end probes (`None` for the direct path).
#[derive(Debug, Clone, Copy, Default)]
struct Entry {
    path: SoundPath,
    weight: f32,
    probes: Option<(usize, usize)>,
}

/// The paths of one simulation, in a fixed-size array (upstream's stack arrays).
struct Paths {
    entries: [Entry; MAX_PATHS],
    len: usize,
}

impl Paths {
    fn push(&mut self, entry: Entry) {
        self.entries[self.len] = entry;
        self.len += 1;
    }

    fn valid(&self) -> impl Iterator<Item = &Entry> {
        self.entries[..self.len]
            .iter()
            .filter(|e| e.path.is_valid())
    }
}

/// What a search for paths reads (the arguments upstream passes along).
struct Search<'a> {
    scene: &'a dyn RayTracer,
    probes: &'a [Probe],
    data: &'a BakedPathData,
    params: &'a PathSimulationParams<'a>,
    callback: &'a mut dyn FnMut(Vector3, Vector3, bool),
    has_callback: bool,
}

/// Finds paths from a source to the listener with the paths baked between the probes of a batch, validating them and
/// searching for others at run time when needed (upstream `PathSimulator`).
#[derive(Debug)]
pub struct PathSimulator {
    vis_tester: ProbeVisibilityTester,
    path_finder: PathFinder,
}

/// The identifier of the pathing layer a simulation uses.
fn pathing_identifier() -> BakedDataIdentifier {
    BakedDataIdentifier {
        variation: BakedDataVariation::Dynamic,
        data_type: BakedDataType::Pathing,
        endpoint_influence: Sphere::default(),
    }
}

fn center(probes: &[Probe], index: usize) -> Vector3 {
    probes[index].influence.center
}

/// The probe before the end of a path that ends at a probe of `start`'s path tree.
fn previous(path: &SoundPath, start: usize) -> usize {
    if path.direct {
        start
    } else {
        usize::try_from(path.last_probe).expect("a valid path through probes")
    }
}

impl PathSimulator {
    /// A simulator for the probes of `batch`, testing visibility with `num_samples` points per probe (see
    /// [`ProbeVisibilityTester::new`]).
    pub fn new(
        batch: &ProbeBatch,
        num_samples: usize,
        asymmetric_vis_range: bool,
        down: Vector3,
    ) -> Self {
        Self {
            vis_tester: ProbeVisibilityTester::new(num_samples, asymmetric_vis_range, down),
            path_finder: PathFinder::new(batch.num_probes(), 1),
        }
    }

    /// Finds the paths from the source to the listener (upstream `findPaths`) and returns the sound field they make:
    /// SH coefficients of `params.order` in `sh_coeffs`, and the rest in the result.
    ///
    /// If the listener sees the source (unless `force_direct_occlusion`), there is one direct path. Otherwise, from
    /// the probe of `source_probes` nearest to the source (or from each, weighted), a path to each probe of
    /// `listener_probes` is looked up in the batch's baked pathing layer (none without one), optionally validated
    /// with rays and replaced by a run-time search, and weighted by the listener probe's weight. Only entries of the
    /// batch `batches[batch_index]` count; the neighbourhoods' batch indices refer to `batches`.
    ///
    /// Returns `None`, leaving `sh_coeffs` alone, if the source is occluded and either neighbourhood has no valid
    /// probe. `validation` is called for each validation ray.
    ///
    /// # Panics
    ///
    /// If `sh_coeffs` has fewer than `num_coeffs_for_order(order)` coefficients, a neighbourhood refers to a batch or
    /// probe that does not exist, or the batch has more probes than when the simulator was made.
    #[allow(clippy::too_many_arguments)]
    pub fn find_paths<B: Borrow<ProbeBatch>>(
        &self,
        scene: &dyn RayTracer,
        batches: &[B],
        batch_index: usize,
        source_probes: &ProbeNeighborhood,
        listener_probes: &ProbeNeighborhood,
        params: &PathSimulationParams<'_>,
        sh_coeffs: &mut [f32],
        validation: Option<ValidationRayCallback<'_>>,
    ) -> Option<PathSimulationResult> {
        let batch = batches[batch_index].borrow();
        let probes = batch.probes();

        let mut paths = Paths {
            entries: [Entry::default(); MAX_PATHS],
            len: 0,
        };

        if scene.is_occluded(params.listener, params.source) || params.force_direct_occlusion {
            if !(source_probes.has_valid_probes() && listener_probes.has_valid_probes()) {
                return None;
            }

            if let Some(BakedData::Pathing(data)) = batch.data(&pathing_identifier()) {
                let mut noop = |_: Vector3, _: Vector3, _: bool| {};
                let has_callback = validation.is_some();
                let mut search = Search {
                    scene,
                    probes,
                    data,
                    params,
                    callback: match validation {
                        Some(callback) => callback,
                        None => &mut noop,
                    },
                    has_callback,
                };

                if params.paths_from_all_source_probes {
                    for i in 0..source_probes.num_probes() {
                        self.find_paths_from_source_probe(
                            &mut search,
                            source_probes,
                            listener_probes,
                            batch_index,
                            i,
                            source_probes.weights[i],
                            &mut paths,
                        );
                    }
                } else {
                    let nearest = source_probes.find_nearest(params.source, batches);
                    self.find_paths_from_source_probe(
                        &mut search,
                        source_probes,
                        listener_probes,
                        batch_index,
                        nearest,
                        1.0,
                        &mut paths,
                    );
                }
            }
        } else {
            paths.push(Entry {
                path: SoundPath {
                    direct: true,
                    ..SoundPath::default()
                },
                weight: 1.0,
                probes: None,
            });
        }

        calc_ambisonics_coeffs(params, probes, &paths, sh_coeffs);
        let (eq_gains, total_deviation) = calc_eq(params.deviation_model, probes, &paths);
        Some(PathSimulationResult {
            eq_gains,
            direction: calc_average_direction(params, probes, &paths),
            distance_ratio: calc_distance_ratio(params.source, probes, &paths),
            total_deviation,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn find_paths_from_source_probe(
        &self,
        search: &mut Search<'_>,
        source_probes: &ProbeNeighborhood,
        listener_probes: &ProbeNeighborhood,
        batch_index: usize,
        source_entry: usize,
        source_weight: f32,
        paths: &mut Paths,
    ) {
        let (Some(batch), Some(source_probe)) = (
            source_probes.batches[source_entry],
            source_probes.probe_indices[source_entry],
        ) else {
            return;
        };
        if batch != batch_index {
            return;
        }

        for i in 0..listener_probes.num_probes() {
            let (Some(batch), Some(listener_probe)) =
                (listener_probes.batches[i], listener_probes.probe_indices[i])
            else {
                continue;
            };
            if batch != batch_index {
                continue;
            }

            let mut sound_path = search
                .data
                .lookup_shortest_path(source_probe, listener_probe);
            let mut try_real_time = false;
            if sound_path.is_valid()
                && self.is_path_occluded(search, &sound_path, source_probe, listener_probe)
                && search.params.find_alternate_paths
            {
                try_real_time = true;
            }

            if try_real_time {
                let params = search.params;
                let probe_path = self.path_finder.find_shortest_path(
                    search.scene,
                    search.probes,
                    search.data.vis_graph(),
                    &self.vis_tester,
                    source_probe,
                    listener_probe,
                    &ShortestPathParams {
                        radius: params.radius,
                        threshold: params.threshold,
                        simplify_paths: params.simplify_paths,
                        real_time_vis: params.real_time_vis,
                    },
                    0,
                );
                sound_path = SoundPath::from_probe_path(&probe_path, search.probes);
            }

            if sound_path.is_valid() {
                paths.push(Entry {
                    path: sound_path,
                    weight: source_weight * listener_probes.weights[i],
                    probes: Some((source_probe, listener_probe)),
                });
            }
        }
    }

    /// Whether a leg of the baked path from `start` to `end` is occluded (upstream `isPathOccluded`): walking back from
    /// `end`, each probe is tested against the one before it, which is found by looking up the path from `start` to
    /// it. Without validation the rays are only reported, as visible.
    fn is_path_occluded(
        &self,
        search: &mut Search<'_>,
        path: &SoundPath,
        start: usize,
        end: usize,
    ) -> bool {
        let params = search.params;
        if !(params.enable_validation || search.has_callback) {
            return false;
        }

        let mut current = end;
        let mut prev = previous(path, start);
        // A baked path visits each probe at most once; malformed data could loop (upstream would forever).
        let mut steps = 0;

        while current != start {
            let visible = !params.enable_validation
                || self.vis_tester.are_probes_visible(
                    search.scene,
                    search.probes,
                    current,
                    prev,
                    params.radius,
                    params.threshold,
                );
            if search.has_callback {
                (search.callback)(
                    center(search.probes, prev),
                    center(search.probes, current),
                    !visible,
                );
            }
            if !visible {
                return true;
            }

            if prev == start {
                break;
            }

            steps += 1;
            let next_path = search.data.lookup_shortest_path(start, prev);
            if !next_path.is_valid() || steps > search.data.num_probes() {
                return true;
            }

            current = prev;
            prev = previous(&next_path, start);
        }

        false
    }
}

/// The virtual source of a path and its distance from where it is heard: the end probe, or the listener for the
/// direct path.
fn virtual_source(
    entry: &Entry,
    probes: &[Probe],
    source: Vector3,
    listener: Vector3,
) -> (Vector3, f32) {
    match entry.probes {
        Some((start, end)) => {
            let virtual_source = entry.path.to_virtual_source(probes, start, end);
            (
                virtual_source,
                (virtual_source - center(probes, end)).length(),
            )
        }
        None => (source, (source - listener).length()),
    }
}

/// Upstream `calcAmbisonicsCoeffsForPaths`: each path's virtual source projected into SH, attenuated with distance and
/// weighted.
fn calc_ambisonics_coeffs(
    params: &PathSimulationParams<'_>,
    probes: &[Probe],
    paths: &Paths,
    coeffs: &mut [f32],
) {
    let num_coeffs = sh::num_coeffs_for_order(params.order);
    coeffs[..num_coeffs].fill(0.0);

    for entry in paths.valid() {
        let (virtual_source, distance) =
            virtual_source(entry, probes, params.source, params.listener);

        let distance_attenuation = params.distance_attenuation_model.evaluate(distance);
        let gain = entry.weight * distance_attenuation;

        let direction = (virtual_source - params.listener).normalize_or_zero();
        sh::project_single_point_and_update(direction, params.order, gain, coeffs);
    }
}

/// Upstream `calcDeviationTerm`.
fn deviation_term(deviation: f32, model: &DeviationModel) -> [f32; NUM_BANDS] {
    std::array::from_fn(|band| model.evaluate(deviation, band))
}

/// Upstream `calcEQForPaths`: the weighted EQ, and the weighted total deviation.
fn calc_eq(model: &DeviationModel, probes: &[Probe], paths: &Paths) -> ([f32; NUM_BANDS], f32) {
    let mut eq_gains = [0.0; NUM_BANDS];
    let mut num_valid_paths = 0;

    for entry in paths.valid() {
        if let Some((start, end)) = entry.probes {
            let mut term = deviation_term(
                cpp_max(1e-8, entry.path.deviation(probes, start, end)),
                model,
            );

            let reference = deviation_term(1e-8, model);
            for (t, r) in term.iter_mut().zip(reference) {
                *t /= r;
            }

            let mut overall_gain = 1.0;
            EqEffect::normalize_gains(&mut term, &mut overall_gain);

            for (g, t) in eq_gains.iter_mut().zip(term) {
                *g += entry.weight * overall_gain * t;
            }
        } else {
            for g in &mut eq_gains {
                *g += entry.weight;
            }
        }

        num_valid_paths += 1;
    }

    if num_valid_paths == 0 {
        eq_gains = [1.0; NUM_BANDS];
    }

    let mut total_deviation = 0.0;
    for entry in paths.valid() {
        if let Some((start, end)) = entry.probes {
            total_deviation += entry.weight * entry.path.deviation(probes, start, end);
        }
    }

    (eq_gains, total_deviation)
}

/// Upstream `calcAverageDirectionForPaths`.
fn calc_average_direction(
    params: &PathSimulationParams<'_>,
    probes: &[Probe],
    paths: &Paths,
) -> Vector3 {
    let mut direction = Vector3::ZERO;
    for entry in paths.valid() {
        let (virtual_source, distance) =
            virtual_source(entry, probes, params.source, params.listener);

        let distance_attenuation = 1.0 / cpp_max(distance, 1.0);
        let gain = entry.weight * distance_attenuation;
        let path_direction = (virtual_source - params.listener).normalize_or_zero();

        direction += path_direction * gain;
    }

    direction.normalize_or_zero()
}

/// Upstream `calcDistanceRatioForPaths`.
fn calc_distance_ratio(source: Vector3, probes: &[Probe], paths: &Paths) -> f32 {
    let mut ratio = 0.0;
    for entry in paths.valid() {
        let mut path_ratio = 1.0;

        if let Some((_, end)) = entry.probes {
            let virtual_source = entry.path.to_virtual_source_from(probes, source, end);
            let path_distance = (virtual_source - center(probes, end)).length();
            let direct_distance = (source - center(probes, end)).length();
            path_ratio = if path_distance > 1.0 && direct_distance > 1.0 {
                direct_distance / path_distance
            } else {
                1.0
            };
        }

        ratio += entry.weight * path_ratio;
    }

    ratio
}
