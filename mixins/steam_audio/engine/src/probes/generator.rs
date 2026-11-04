//! Probe generation (upstream `probe_generator.{h,cpp}`, `ProbeGenerator`).

use crate::geometry::{Ray, Sphere};
use crate::math::{Matrix4, Vector3, Vector4, cpp_min, msvc_f32_to_i32};
use crate::scene::RayTracer;

use super::Probe;

/// How far below a floor hit the search for the next floor starts (upstream `kDownwardOffset`).
const DOWNWARD_OFFSET: f32 = 0.01;

/// How to place probes in an oriented box (upstream `ProbeGenerationType` and its parameters). Upstream's `Octree`
/// type is not implemented there either (it throws).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbeGeneration {
    /// One probe at the box's center, with a radius of half its smallest side.
    Centroid,
    /// A grid of `spacing` over the box's horizontal extent, seen from above: a probe `height` above every floor
    /// a downward ray finds (several per column for stacked floors), each of radius `spacing`.
    UniformFloor { spacing: f32, height: f32 },
}

/// Places probes in the oriented box `obb_transform` maps the unit cube `[-0.5, 0.5]³` onto (upstream
/// `ProbeGenerator::generateProbes`). The box's local y axis is up.
///
/// `probes` is replaced with the generated probes, except when a uniform floor box has a side of zero length (below
/// `f32::MIN_POSITIVE`): then it is left as it is, as upstream leaves its probe array.
pub fn generate_probes(
    scene: &(impl RayTracer + ?Sized),
    obb_transform: &Matrix4,
    generation: ProbeGeneration,
    probes: &mut Vec<Probe>,
) {
    match generation {
        ProbeGeneration::Centroid => generate_centroid_probe(obb_transform, probes),
        ProbeGeneration::UniformFloor { spacing, height } => {
            generate_uniform_floor_probes(scene, obb_transform, spacing, height, probes);
        }
    }
}

/// Length of column `j` of the transform's upper 3×3 part: the box's side along local axis `j`.
fn side_length(transform: &Matrix4, j: usize) -> f32 {
    Vector3::new(transform[(0, j)], transform[(1, j)], transform[(2, j)]).length()
}

fn generate_centroid_probe(transform: &Matrix4, probes: &mut Vec<Probe>) {
    let center = Vector3::new(transform[(0, 3)], transform[(1, 3)], transform[(2, 3)]);

    let sx = side_length(transform, 0);
    let sy = side_length(transform, 1);
    let sz = side_length(transform, 2);
    let radius = cpp_min(cpp_min(sx / 2.0, sy / 2.0), sz / 2.0);

    probes.clear();
    probes.push(Probe::new(Sphere::new(center, radius)));
}

fn generate_uniform_floor_probes(
    scene: &(impl RayTracer + ?Sized),
    transform: &Matrix4,
    spacing: f32,
    height: f32,
    probes: &mut Vec<Probe>,
) {
    let sx = side_length(transform, 0);
    let sy = side_length(transform, 1);
    let sz = side_length(transform, 2);

    if sx < f32::MIN_POSITIVE || sy < f32::MIN_POSITIVE || sz < f32::MIN_POSITIVE {
        return;
    }

    let num_probes_x = msvc_f32_to_i32((sx / spacing).floor()).wrapping_add(1);
    let num_probes_z = msvc_f32_to_i32((sz / spacing).floor()).wrapping_add(1);
    let residual_x = (sx - (num_probes_x - 1) as f32 * spacing) / 2.0;
    let residual_z = (sz - (num_probes_z - 1) as f32 * spacing) / 2.0;

    let down = *transform * Vector4::new(0.0, -1.0, 0.0, 0.0);
    let down = Vector3::new(down.x, down.y, down.z).normalize_or_zero();

    probes.clear();
    for i in 0..num_probes_x {
        for j in 0..num_probes_z {
            let x = -0.5 + (i as f32 * spacing + residual_x) / sx;
            let y = 0.5;
            let z = -0.5 + (j as f32 * spacing + residual_z) / sz;

            let point = *transform * Vector4::new(x, y, z, 1.0);
            let point = Vector3::new(point.x, point.y, point.z);

            floor_probes_below(scene, point, down, sy, spacing, height, probes);
        }
    }
}

/// Upstream `computeFloorProbesBelow`: follows a downward ray through the box's height `sy`, adding a probe `height`
/// above each floor it hits.
fn floor_probes_below(
    scene: &(impl RayTracer + ?Sized),
    origin: Vector3,
    down: Vector3,
    sy: f32,
    spacing: f32,
    height: f32,
    probes: &mut Vec<Probe>,
) {
    let mut origin = origin;
    let mut distance_from_floor = sy;

    while distance_from_floor > 0.0 {
        let ray = Ray::new(origin, down);
        let hit = scene.closest_hit(&ray, height, distance_from_floor + height);
        if !hit.is_valid() {
            break;
        }

        let center = origin + down * (hit.distance - height);
        probes.push(Probe::new(Sphere::new(center, spacing)));

        origin += down * (hit.distance + DOWNWARD_OFFSET);
        distance_from_floor -= hit.distance + DOWNWARD_OFFSET;
    }
}
