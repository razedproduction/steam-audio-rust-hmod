//! Probe trees (upstream `probe_tree.{h,cpp}`): a bounding volume hierarchy over the probes' influence spheres, used to
//! find the probes whose influence contains a point.

use crate::geometry::Aabb;
use crate::math::Vector3;
use crate::scene::bvh::{Centroid, GrowableBox};
use crate::scene::msvc_sort;

use super::Probe;

/// Depth of the lookup stack (upstream `kProbeLookupStackSize`). Trees are split at the median, so their depth is
/// about `log2(number of probes)`.
const LOOKUP_STACK_SIZE: usize = 128;

/// A node of a [`ProbeTree`] (upstream `ProbeTreeNode`).
///
/// Upstream keeps the packed data in the padding lane of the box's minimum and the split coordinate in that of its
/// maximum; here they are fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbeTreeNode {
    bounds: Aabb,
    data: i32,
    split_coordinate: f32,
}

impl Default for ProbeTreeNode {
    fn default() -> Self {
        Self {
            bounds: Aabb::EMPTY,
            data: 0,
            split_coordinate: 0.0,
        }
    }
}

impl ProbeTreeNode {
    /// The box around the influence spheres of the node's probes.
    pub fn bounds(&self) -> &Aabb {
        &self.bounds
    }

    /// Upstream's packed data: `probe << 2 | 3` for a leaf, `left child offset << 2 | split axis` for an internal node.
    pub fn packed_data(&self) -> i32 {
        self.data
    }

    /// The coordinate along the split axis that separates the children (0 for a leaf).
    pub fn split_coordinate(&self) -> f32 {
        self.split_coordinate
    }

    pub fn is_leaf(&self) -> bool {
        self.data & 3 == 3
    }

    fn split_axis(&self) -> usize {
        (self.data & 3) as usize
    }

    /// Probe index (leaf) or left child offset (internal).
    fn payload(&self) -> usize {
        (self.data >> 2) as usize
    }
}

/// Leaves `[start, end]` (inclusive) of the node at `node`, whose children go to `left_child` and `left_child + 1`
/// (upstream `ProbeTreeConstructionTask`).
#[derive(Debug, Clone, Copy)]
struct ConstructionTask {
    node: usize,
    start: usize,
    end: usize,
    left_child: usize,
}

/// A hierarchy of probe influence spheres (upstream `ProbeTree`), split at the median along the longest axis of each
/// node's box. A tree of `n` probes has `2n - 1` nodes, laid out as upstream lays them out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProbeTree {
    nodes: Vec<ProbeTreeNode>,
}

impl ProbeTree {
    /// Builds the tree of `probes`. Without probes the tree is empty.
    pub fn new(probes: &[Probe]) -> Self {
        let num_probes = probes.len();
        if num_probes == 0 {
            return Self::default();
        }
        assert!(
            i32::try_from(2 * num_probes).is_ok(),
            "too many probes for a probe tree"
        );

        let mut nodes = vec![ProbeTreeNode::default(); 2 * num_probes - 1];

        let mut leaf_indices: Vec<u32> = (0..num_probes as u32).collect();
        let leaf_bounds: Vec<Aabb> = probes
            .iter()
            .map(|probe| {
                let sphere = &probe.influence;
                let delta = sphere.radius * Vector3::new(1.0, 1.0, 1.0);
                Aabb::new(sphere.center - delta, sphere.center + delta)
            })
            .collect();

        let mut centroids = vec![
            Centroid {
                coordinate: 0.0,
                leaf: 0,
            };
            3 * num_probes
        ];

        let mut stack = Vec::new();
        let mut task = ConstructionTask {
            node: 0,
            start: 0,
            end: num_probes - 1,
            left_child: 1,
        };

        loop {
            if task.start == task.end {
                let leaf = leaf_indices[task.start];
                let node = &mut nodes[task.node];
                node.bounds = leaf_bounds[leaf as usize];
                node.data = ((leaf as i32) << 2) | 3;

                match stack.pop() {
                    Some(next) => task = next,
                    None => break,
                }
            } else {
                let mut node_bounds = GrowableBox::EMPTY;
                for &leaf in &leaf_indices[task.start..=task.end] {
                    node_bounds
                        .grow_to_contain(&GrowableBox::from_aabb(&leaf_bounds[leaf as usize]));
                }
                nodes[task.node].bounds = node_bounds.to_aabb();

                for i in task.start..=task.end {
                    let leaf = leaf_indices[i];
                    let center = probes[leaf as usize].influence.center;
                    for axis in 0..3 {
                        centroids[axis * num_probes + i] = Centroid {
                            coordinate: center[axis],
                            leaf,
                        };
                    }
                }

                let split_index = (task.end - task.start).div_ceil(2);
                let split_axis = nodes[task.node].bounds.extents().index_of_max_component();

                let row = &mut centroids[split_axis * num_probes..(split_axis + 1) * num_probes];
                msvc_sort::sort(&mut row[task.start..=task.end], |a, b| {
                    a.coordinate < b.coordinate
                });

                for i in task.start..=task.end {
                    leaf_indices[i] = row[i].leaf;
                }

                let node = &mut nodes[task.node];
                node.data = (((task.left_child - task.node) as i32) << 2) | split_axis as i32;
                node.split_coordinate = row[task.start + split_index].coordinate;

                stack.push(ConstructionTask {
                    node: task.left_child + 1,
                    start: task.start + split_index,
                    end: task.end,
                    left_child: task.left_child + 2 * split_index,
                });
                task = ConstructionTask {
                    node: task.left_child,
                    start: task.start,
                    end: task.start + split_index - 1,
                    left_child: task.left_child + 2,
                };
            }
        }

        Self { nodes }
    }

    pub fn nodes(&self) -> &[ProbeTreeNode] {
        &self.nodes
    }

    /// Writes into `probe_indices` the probes (of `probes`, the ones the tree was built from) whose influence contains
    /// `point`, in the order the depth-first search finds them, stopping when `probe_indices` is full; the remaining
    /// entries are `None` (upstream `getInfluencingProbes`, with `maxInfluencingProbes = probe_indices.len()`).
    ///
    /// # Panics
    ///
    /// If the tree refers to probes past the end of `probes`.
    pub fn influencing_probes(
        &self,
        point: Vector3,
        probes: &[Probe],
        probe_indices: &mut [Option<usize>],
    ) {
        probe_indices.fill(None);
        if self.nodes.is_empty() || probe_indices.is_empty() {
            return;
        }

        let mut num_influencing_probes = 0;
        let mut stack = [0usize; LOOKUP_STACK_SIZE];
        let mut stack_size = 0;
        let mut node_index = 0;

        loop {
            let node = &self.nodes[node_index];
            if node.bounds.contains(point) {
                if node.is_leaf() {
                    let probe = node.payload();
                    if probes[probe].influence.contains(point) {
                        probe_indices[num_influencing_probes] = Some(probe);
                        num_influencing_probes += 1;
                        if num_influencing_probes >= probe_indices.len() {
                            break;
                        }
                    }
                } else {
                    let mut near_child = node_index + node.payload();
                    let mut far_child = near_child + 1;
                    if point[node.split_axis()] > node.split_coordinate {
                        std::mem::swap(&mut near_child, &mut far_child);
                    }

                    stack[stack_size] = far_child;
                    stack_size += 1;
                    node_index = near_child;
                    continue;
                }
            }

            if stack_size == 0 {
                break;
            }
            stack_size -= 1;
            node_index = stack[stack_size];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Sphere;

    /// Upstream `ProbeTree.test.cpp`: "ProbeTree::getSamplesThatInfluence returns correct number of probes".
    #[test]
    fn influencing_probes_upstream_test() {
        let probes = [
            Probe::new(Sphere::new(Vector3::new(-8.0, -8.0, -8.0), 4.0)),
            Probe::new(Sphere::new(Vector3::new(-6.0, -6.0, -6.0), 4.0)),
            Probe::new(Sphere::new(Vector3::new(4.0, 4.0, 4.0), 4.0)),
            Probe::new(Sphere::new(Vector3::new(32.0, 32.0, 32.0), 8.0)),
        ];
        let tree = ProbeTree::new(&probes);
        assert_eq!(tree.nodes().len(), 7);

        let mut indices = [None; 4];
        tree.influencing_probes(Vector3::new(-7.0, -7.0, -7.0), &probes, &mut indices);
        assert_eq!(indices.iter().flatten().count(), 2);
    }

    #[test]
    fn empty_tree_finds_nothing() {
        let tree = ProbeTree::new(&[]);
        let mut indices = [Some(3); 2];
        tree.influencing_probes(Vector3::ZERO, &[], &mut indices);
        assert_eq!(indices, [None, None]);
    }
}
