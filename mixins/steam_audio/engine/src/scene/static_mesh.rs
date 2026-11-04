use crate::error::{Error, Result};
use crate::geometry::{Aabb, Ray};
use crate::math::Vector3;
use crate::serialization::{Builder, Offset, Table};

use super::bvh::Bvh;
use super::mesh::{Hit, Material, Mesh, Triangle};

// Vtable slots of `ipl.Serialized.StaticMesh`.
const VT_MESH: u16 = 4;
const VT_MATERIAL_INDICES: u16 = 6;
const VT_MATERIALS: u16 = 8;
const VT_MATERIALS2: u16 = 10;

/// A triangle mesh whose geometry never changes, with a material per triangle (upstream `StaticMesh`, the
/// built-in ray tracer's `IStaticMesh`).
#[derive(Debug, Clone)]
pub struct StaticMesh {
    mesh: Mesh,
    bvh: Bvh,
    material_indices: Vec<u32>,
    materials: Vec<Material>,
    /// Materials set by [`Scene::set_static_mesh_material`](super::Scene::set_static_mesh_material), applied by
    /// the next commit (upstream `materialsToUpdate`).
    pub(crate) pending_materials: Vec<Material>,
    pub(crate) materials_changed: bool,
}

impl StaticMesh {
    /// Builds the mesh and its BVH (upstream `StaticMesh(...)`, `IScene::createStaticMesh`).
    ///
    /// # Panics
    ///
    /// If there are no triangles, a triangle refers to a missing vertex, `material_indices` does not have one entry
    /// per triangle, or a material index is out of range.
    pub fn new(
        vertices: &[Vector3],
        triangles: &[Triangle],
        material_indices: &[u32],
        materials: &[Material],
    ) -> Self {
        assert_eq!(
            material_indices.len(),
            triangles.len(),
            "expected one material index per triangle"
        );
        if let Some(&index) = material_indices
            .iter()
            .find(|&&i| i as usize >= materials.len())
        {
            panic!(
                "material index {index} out of range ({} materials)",
                materials.len()
            );
        }

        let mesh = Mesh::new(vertices, triangles);
        let bvh = Bvh::new(&mesh);
        Self {
            mesh,
            bvh,
            material_indices: material_indices.to_vec(),
            materials: materials.to_vec(),
            pending_materials: materials.to_vec(),
            materials_changed: false,
        }
    }

    pub fn num_vertices(&self) -> usize {
        self.mesh.num_vertices()
    }

    pub fn num_triangles(&self) -> usize {
        self.mesh.num_triangles()
    }

    pub fn num_materials(&self) -> usize {
        self.materials.len()
    }

    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }

    pub fn bounds(&self) -> Aabb {
        self.bvh.bounds()
    }

    pub fn material_indices(&self) -> &[u32] {
        &self.material_indices
    }

    /// The materials rays see (set changes become visible here at the next commit).
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    /// The closest hit, with normal and material filled in (upstream `closestHit`).
    pub fn closest_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> Hit {
        let mut hit = self
            .bvh
            .closest_hit(ray, &self.mesh, min_distance, max_distance);

        if let (true, Some(triangle)) = (hit.is_valid(), hit.triangle_index) {
            let material_index = self.material_indices[triangle] as usize;
            hit.normal = self.mesh.normal(triangle);
            hit.material_index = Some(material_index);
            hit.material = Some(self.materials[material_index]);
        }

        hit
    }

    /// Upstream `anyHit`.
    pub fn any_hit(&self, ray: &Ray, min_distance: f32, max_distance: f32) -> bool {
        self.bvh
            .any_hit(ray, &self.mesh, min_distance, max_distance)
    }

    /// `true` if any triangle intersects `aabb` (upstream `intersectsBox`).
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        self.bvh.intersects_aabb(aabb, &self.mesh)
    }

    /// Serializes the mesh as the root of its own FlatBuffer (schema `ipl.Serialized.StaticMesh`; upstream
    /// `serializeAsRoot`). Materials go into the 3-band `materials` array, as upstream builds without
    /// `IPL_ENABLE_OCTAVE_BANDS` write them; `materials2` is left out.
    pub fn serialize(&self) -> Vec<u8> {
        let mut builder = Builder::new();
        let root = self.serialize_into(&mut builder);
        builder.finish(root)
    }

    /// Reads a mesh written by [`serialize`](Self::serialize) and builds its BVH (upstream
    /// `StaticMesh(SerializedObject&)`, `IScene::createStaticMesh(SerializedObject&)`). Only the 3-band `materials`
    /// are read; `materials2` is ignored.
    ///
    /// # Errors
    ///
    /// [`Error::MalformedSerializedData`](crate::Error::MalformedSerializedData) if the data is not a valid
    /// serialized static mesh: missing or empty arrays (which upstream asserts against), fewer material indices than
    /// triangles, or vertex or material indices out of range (which upstream does not check).
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Self::from_table(&Table::root(data)?)
    }

    /// Upstream `serialize`. The material pending for the next commit is not serialized.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let mesh = self.mesh.serialize_into(builder);
        let material_indices = builder.create_vector(self.material_indices.len(), 4, |bytes| {
            for (out, index) in bytes.chunks_exact_mut(4).zip(&self.material_indices) {
                out.copy_from_slice(&index.to_le_bytes());
            }
        });
        let materials = builder.create_vector_of_structs(self.materials.len(), 28, |bytes| {
            for (out, m) in bytes.chunks_exact_mut(28).zip(&self.materials) {
                let values = m
                    .absorption
                    .iter()
                    .chain([&m.scattering])
                    .chain(&m.transmission);
                for (word, x) in out.chunks_exact_mut(4).zip(values) {
                    word.copy_from_slice(&x.to_le_bytes());
                }
            }
        });

        let start = builder.start_table();
        builder.add_offset(VT_MATERIALS2, None);
        builder.add_offset(VT_MATERIALS, Some(materials));
        builder.add_offset(VT_MATERIAL_INDICES, Some(material_indices));
        builder.add_offset(VT_MESH, Some(mesh));
        builder.end_table(start)
    }

    /// Upstream `StaticMesh(const Serialized::StaticMesh*)`. Material indices past the triangle count are ignored.
    pub(crate) fn from_table(table: &Table<'_>) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;
        let mesh = Mesh::from_table(
            &table
                .table(VT_MESH)?
                .ok_or(malformed("static mesh without a mesh"))?,
        )?;
        let material_indices = table
            .vector(VT_MATERIAL_INDICES, 4)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("static mesh without material indices"))?;
        let materials = table
            .vector(VT_MATERIALS, 28)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("static mesh without materials"))?;

        if material_indices.len() < mesh.num_triangles() {
            return Err(malformed("fewer material indices than triangles"));
        }
        let material_indices = material_indices
            .i32s()
            .take(mesh.num_triangles())
            .map(|index| {
                u32::try_from(index)
                    .ok()
                    .filter(|&index| (index as usize) < materials.len())
                    .ok_or(malformed("material index out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let materials: Vec<Material> = (0..materials.len())
            .map(|i| {
                let mut values = materials.words(i).map(f32::from_le_bytes);
                let mut next = || values.next().expect("28-byte material");
                Material {
                    absorption: [next(), next(), next()],
                    scattering: next(),
                    transmission: [next(), next(), next()],
                }
            })
            .collect();

        let bvh = Bvh::new(&mesh);
        Ok(Self {
            mesh,
            bvh,
            material_indices,
            pending_materials: materials.clone(),
            materials,
            materials_changed: false,
        })
    }

    pub(crate) fn apply_pending_materials(&mut self) {
        if self.materials_changed {
            self.materials.copy_from_slice(&self.pending_materials);
            self.materials_changed = false;
        }
    }
}
