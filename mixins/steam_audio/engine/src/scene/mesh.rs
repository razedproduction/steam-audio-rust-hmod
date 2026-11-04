use crate::bands::NUM_BANDS;
use crate::error::{Error, Result};
use crate::geometry::Ray;
use crate::math::Vector3;
use crate::serialization::{Builder, Offset, Table};

// Vtable slots of `ipl.Serialized.Mesh`.
const VT_VERTICES: u16 = 4;
const VT_TRIANGLES: u16 = 6;

/// An acoustic material (upstream `Material`): per-band absorption and transmission coefficients and one
/// random-incidence scattering coefficient.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Material {
    pub absorption: [f32; NUM_BANDS],
    pub scattering: f32,
    pub transmission: [f32; NUM_BANDS],
}

/// An indexed triangle (upstream `Triangle`): three indices into a mesh's vertices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Triangle {
    pub indices: [u32; 3],
}

impl Triangle {
    pub const fn new(i0: u32, i1: u32, i2: u32) -> Self {
        Self {
            indices: [i0, i1, i2],
        }
    }
}

/// The result of tracing a ray (upstream `Hit`).
///
/// Upstream marks missing indices with `-1` and points at the material; here they are `None`, and the material is
/// copied into the hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    /// Distance along the ray, `+∞` if nothing was hit.
    pub distance: f32,
    pub triangle_index: Option<usize>,
    /// Never set by [`Scene`](super::Scene); available to custom ray tracers.
    pub object_index: Option<usize>,
    pub material_index: Option<usize>,
    /// Surface normal at the hit point (zero if nothing was hit).
    pub normal: Vector3,
    pub material: Option<Material>,
}

impl Default for Hit {
    fn default() -> Self {
        Self {
            distance: f32::INFINITY,
            triangle_index: None,
            object_index: None,
            material_index: None,
            normal: Vector3::ZERO,
            material: None,
        }
    }
}

impl Hit {
    /// `true` if something was hit (`distance < ∞`; upstream `isValid`).
    pub fn is_valid(&self) -> bool {
        self.distance < f32::INFINITY
    }
}

/// An indexed triangle mesh with one normal per triangle (upstream `Mesh`).
#[derive(Debug, Clone, PartialEq)]
pub struct Mesh {
    vertices: Vec<Vector3>,
    triangles: Vec<Triangle>,
    normals: Vec<Vector3>,
}

impl Mesh {
    /// Copies the geometry and computes the normals, `normalize((v1 - v0) × (v2 - v0))` (zero for degenerate
    /// triangles).
    ///
    /// # Panics
    ///
    /// If a triangle refers to a vertex that does not exist.
    pub fn new(vertices: &[Vector3], triangles: &[Triangle]) -> Self {
        for (i, triangle) in triangles.iter().enumerate() {
            for &index in &triangle.indices {
                assert!(
                    (index as usize) < vertices.len(),
                    "triangle {i} refers to vertex {index}, but the mesh has {} vertices",
                    vertices.len()
                );
            }
        }

        let mut mesh = Self {
            vertices: vertices.to_vec(),
            triangles: triangles.to_vec(),
            normals: Vec::with_capacity(triangles.len()),
        };
        for i in 0..mesh.triangles.len() {
            let [v0, v1, v2] = mesh.triangle_vertices(i);
            mesh.normals
                .push((v1 - v0).cross(v2 - v0).normalize_or_zero());
        }
        mesh
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_triangles(&self) -> usize {
        self.triangles.len()
    }

    pub fn vertices(&self) -> &[Vector3] {
        &self.vertices
    }

    pub fn triangles(&self) -> &[Triangle] {
        &self.triangles
    }

    pub fn vertex(&self, i: usize) -> Vector3 {
        self.vertices[i]
    }

    pub fn triangle(&self, i: usize) -> Triangle {
        self.triangles[i]
    }

    /// The three vertices of triangle `i` (upstream `triangleVertex(i, 0..3)`).
    pub fn triangle_vertices(&self, i: usize) -> [Vector3; 3] {
        self.triangles[i]
            .indices
            .map(|index| self.vertices[index as usize])
    }

    pub fn normal(&self, i: usize) -> Vector3 {
        self.normals[i]
    }

    /// Distance along `ray` to triangle `i`, or `+∞` (upstream `Ray::intersect(mesh, i)`).
    pub fn intersect(&self, ray: &Ray, i: usize) -> f32 {
        let [v0, v1, v2] = self.triangle_vertices(i);
        ray.intersect_triangle(v0, v1, v2)
    }

    /// Upstream `serialize`. The schema declares a triangle's indices as floats, but upstream copies its `int`
    /// indices into them bit for bit, so the bytes are the indices'.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let vertices =
            builder.create_uninitialized_vector_of_structs(self.vertices.len(), 12, |bytes| {
                for (out, v) in bytes.chunks_exact_mut(12).zip(&self.vertices) {
                    out[0..4].copy_from_slice(&v.x.to_le_bytes());
                    out[4..8].copy_from_slice(&v.y.to_le_bytes());
                    out[8..12].copy_from_slice(&v.z.to_le_bytes());
                }
            });
        let triangles = builder.create_vector_of_structs(self.triangles.len(), 12, |bytes| {
            for (out, t) in bytes.chunks_exact_mut(12).zip(&self.triangles) {
                for (word, index) in out.chunks_exact_mut(4).zip(t.indices) {
                    word.copy_from_slice(&index.to_le_bytes());
                }
            }
        });

        let start = builder.start_table();
        builder.add_offset(VT_TRIANGLES, Some(triangles));
        builder.add_offset(VT_VERTICES, Some(vertices));
        builder.end_table(start)
    }

    /// Upstream `Mesh(const Serialized::Mesh*)`: copies the geometry and computes the normals.
    pub(crate) fn from_table(table: &Table<'_>) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;
        let vertices = table
            .vector(VT_VERTICES, 12)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("mesh without vertices"))?;
        let triangles = table
            .vector(VT_TRIANGLES, 12)?
            .filter(|t| t.len() > 0)
            .ok_or(malformed("mesh without triangles"))?;

        let vertices: Vec<Vector3> = (0..vertices.len())
            .map(|i| {
                let mut xyz = vertices.words(i).map(f32::from_le_bytes);
                let mut next = || xyz.next().expect("12-byte vertex");
                Vector3::new(next(), next(), next())
            })
            .collect();
        let triangles = (0..triangles.len())
            .map(|i| {
                let mut indices = [0; 3];
                for (index, word) in indices.iter_mut().zip(triangles.words(i)) {
                    *index = u32::try_from(i32::from_le_bytes(word))
                        .ok()
                        .filter(|&index| (index as usize) < vertices.len())
                        .ok_or(malformed("triangle vertex index out of range"))?;
                }
                Ok(Triangle { indices })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self::new(&vertices, &triangles))
    }
}
