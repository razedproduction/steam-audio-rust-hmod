//! Probe batches (upstream `probe_batch.{h,cpp}`, `ProbeBatch`).

use std::collections::BTreeMap;

use crate::baking::{
    BakedData, BakedDataIdentifier, BakedDataType, BakedDataVariation, BakedReflectionsData,
};
use crate::error::{Error, Result};
use crate::geometry::Sphere;
use crate::math::Vector3;
use crate::pathing::BakedPathData;
use crate::serialization::{Builder, Offset, Table};

use super::{Probe, ProbeNeighborhood, ProbeTree};

// Vtable slots of `ipl.Serialized.ProbeBatch`.
const VT_PROBES: u16 = 4;
const VT_DATA_LAYERS: u16 = 6;

// Vtable slots of `ipl.Serialized.BakedDataLayer`.
const VT_IDENTIFIER: u16 = 4;
const VT_REFLECTIONS_DATA: u16 = 6;
const VT_PATHING_DATA: u16 = 8;

// Vtable slots of `ipl.Serialized.BakedDataIdentifier`.
const VT_VARIATION: u16 = 4;
const VT_TYPE: u16 = 6;
const VT_INFLUENCE: u16 = 8;

/// Size of `ipl.Serialized.Sphere`.
const SPHERE_SIZE: usize = 16;

/// A set of probes and the data baked at them, in layers keyed by [`BakedDataIdentifier`] (upstream `ProbeBatch`).
///
/// Changes to the probes take effect for lookups at the next [`commit`](Self::commit), which builds the batch's
/// [`ProbeTree`].
#[derive(Debug, Clone, Default)]
pub struct ProbeBatch {
    probes: Vec<Probe>,
    tree: Option<ProbeTree>,
    data: BTreeMap<BakedDataIdentifier, BakedData>,
}

impl ProbeBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_probes(&self) -> usize {
        self.probes.len()
    }

    /// The probes (upstream `probes()`, `operator[](int)`, `toProbeArray`).
    pub fn probes(&self) -> &[Probe] {
        &self.probes
    }

    /// The probe tree built by the last [`commit`](Self::commit), if any.
    pub fn tree(&self) -> Option<&ProbeTree> {
        self.tree.as_ref()
    }

    /// The data layers, in identifier order (upstream `getData`).
    pub fn data_layers(&self) -> impl Iterator<Item = (&BakedDataIdentifier, &BakedData)> {
        self.data.iter()
    }

    pub fn has_data(&self, identifier: &BakedDataIdentifier) -> bool {
        self.data.contains_key(identifier)
    }

    /// The layer with `identifier` (upstream `operator[](const BakedDataIdentifier&)`).
    pub fn data(&self, identifier: &BakedDataIdentifier) -> Option<&BakedData> {
        self.data.get(identifier)
    }

    pub fn data_mut(&mut self, identifier: &BakedDataIdentifier) -> Option<&mut BakedData> {
        self.data.get_mut(identifier)
    }

    /// Panics if `index` is out of range.
    pub fn update_probe_radius(&mut self, index: usize, radius: f32) {
        self.probes[index].influence.radius = radius;
    }

    /// Moves probe `index`; its baked data must be baked again. Panics if `index` is out of range.
    pub fn update_probe_position(&mut self, index: usize, position: Vector3) {
        self.probes[index].influence.center = position;
        for data in self.data.values_mut() {
            data.update_probe_position(index, position);
        }
    }

    /// Adds a probe, which needs baking in every layer.
    pub fn add_probe(&mut self, influence: Sphere) {
        self.probes.push(Probe::new(influence));
        for data in self.data.values_mut() {
            data.add_probe(&influence);
        }
    }

    /// Adds probes in order (upstream `addProbeArray`).
    pub fn add_probes(&mut self, probes: &[Probe]) {
        for probe in probes {
            self.add_probe(probe.influence);
        }
    }

    /// Removes probe `index` and its baked data. Panics if `index` is out of range.
    pub fn remove_probe(&mut self, index: usize) {
        self.probes.remove(index);
        for data in self.data.values_mut() {
            data.remove_probe(index);
        }
    }

    /// Tells the layer with `identifier` that its endpoint now has the influence `endpoint_influence` (upstream
    /// `updateEndpoint`; see [`BakedReflectionsData`] for what changes).
    pub fn update_endpoint(
        &mut self,
        identifier: &BakedDataIdentifier,
        endpoint_influence: &Sphere,
    ) {
        for data in self.data.values_mut() {
            data.update_endpoint(identifier, &self.probes, endpoint_influence);
        }
    }

    /// Builds the probe tree used by [`influencing_probes`](Self::influencing_probes).
    pub fn commit(&mut self) {
        self.tree = Some(ProbeTree::new(&self.probes));
    }

    /// Adds a layer, replacing any layer with the same identifier.
    pub fn add_data(&mut self, identifier: BakedDataIdentifier, data: BakedData) {
        self.data.insert(identifier, data);
    }

    pub fn remove_data(&mut self, identifier: &BakedDataIdentifier) {
        self.data.remove(identifier);
    }

    /// Fills the [`MAX_PROBES_PER_BATCH`](ProbeNeighborhood::MAX_PROBES_PER_BATCH) entries of `neighborhood` from
    /// `offset` on with the probes whose influence contains `point`, in the order the tree finds them, and sets their
    /// batch to `batch_index` (also for the entries left without a probe). Weights are left as they are (upstream
    /// `getInfluencingProbes`).
    ///
    /// # Panics
    ///
    /// If the batch has not been committed, or the neighbourhood has fewer than `offset + MAX_PROBES_PER_BATCH`
    /// entries.
    pub fn influencing_probes(
        &self,
        point: Vector3,
        neighborhood: &mut ProbeNeighborhood,
        offset: usize,
        batch_index: usize,
    ) {
        let tree = self
            .tree
            .as_ref()
            .expect("a probe batch must be committed before probes are looked up");
        let range = offset..offset + ProbeNeighborhood::MAX_PROBES_PER_BATCH;
        tree.influencing_probes(
            point,
            &self.probes,
            &mut neighborhood.probe_indices[range.clone()],
        );
        neighborhood.batches[range].fill(Some(batch_index));
    }

    /// Serializes the probes and data layers as the root of a FlatBuffer (upstream `serializeAsRoot`, schema
    /// `ipl.Serialized.ProbeBatch`). The probe tree is not serialized.
    pub fn serialize(&self) -> Vec<u8> {
        let mut builder = Builder::new();
        let root = self.serialize_into(&mut builder);
        builder.finish(root)
    }

    /// Reads a batch written by [`serialize`](Self::serialize) or by upstream. It must be committed before probes
    /// are looked up.
    ///
    /// # Errors
    ///
    /// [`Error::MalformedSerializedData`](crate::Error::MalformedSerializedData) if the data is not a valid
    /// serialized batch, including one without probes.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Self::from_table(&Table::root(data)?)
    }

    /// Upstream `serialize`.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let probes = builder.create_vector_of_structs(self.probes.len(), SPHERE_SIZE, |bytes| {
            for (out, probe) in bytes.chunks_exact_mut(SPHERE_SIZE).zip(&self.probes) {
                out.copy_from_slice(&Probe::sphere_bytes(&probe.influence));
            }
        });

        let layers: Vec<Offset> = self
            .data
            .iter()
            .map(|(identifier, data)| {
                let start = builder.start_table();
                builder.add_struct(
                    VT_INFLUENCE,
                    &Probe::sphere_bytes(&identifier.endpoint_influence),
                );
                builder.add_i8(VT_TYPE, identifier.data_type as i8, 0);
                builder.add_i8(VT_VARIATION, identifier.variation as i8, 0);
                let identifier = builder.end_table(start);

                let (reflections, pathing) = match data {
                    BakedData::Reflections(data) => (Some(data.serialize_into(builder)), None),
                    BakedData::Pathing(data) => (None, Some(data.serialize_into(builder))),
                };

                let start = builder.start_table();
                builder.add_offset(VT_PATHING_DATA, pathing);
                builder.add_offset(VT_REFLECTIONS_DATA, reflections);
                builder.add_offset(VT_IDENTIFIER, Some(identifier));
                builder.end_table(start)
            })
            .collect();
        let layers = builder.create_vector_of_offsets(&layers);

        let start = builder.start_table();
        builder.add_offset(VT_DATA_LAYERS, Some(layers));
        builder.add_offset(VT_PROBES, Some(probes));
        builder.end_table(start)
    }

    /// Upstream's deserializing constructor.
    pub(crate) fn from_table(table: &Table<'_>) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;

        let serialized_probes = table
            .vector(VT_PROBES, SPHERE_SIZE)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("probe batch without probes"))?;
        let probes: Vec<Probe> = (0..serialized_probes.len())
            .map(|i| Probe::new(Probe::sphere_from_bytes(serialized_probes.element(i))))
            .collect();

        let mut batch = Self {
            probes,
            tree: None,
            data: BTreeMap::new(),
        };

        if let Some(layers) = table.vector(VT_DATA_LAYERS, 4)? {
            for i in 0..layers.len() {
                let layer = layers.table(i)?;
                let serialized_identifier = layer
                    .table(VT_IDENTIFIER)?
                    .ok_or(malformed("baked data layer without identifier"))?;
                let identifier = BakedDataIdentifier {
                    variation: BakedDataVariation::from_i8(
                        serialized_identifier.i8(VT_VARIATION, 0)?,
                    )
                    .ok_or(malformed("unknown baked data variation"))?,
                    data_type: BakedDataType::from_i8(serialized_identifier.i8(VT_TYPE, 0)?)
                        .ok_or(malformed("unknown baked data type"))?,
                    endpoint_influence: Probe::sphere_from_bytes(
                        serialized_identifier
                            .struct_bytes(VT_INFLUENCE, SPHERE_SIZE)?
                            .ok_or(malformed("baked data identifier without influence"))?,
                    ),
                };

                let data = match identifier.data_type {
                    BakedDataType::Reflections => {
                        let reflections = layer
                            .table(VT_REFLECTIONS_DATA)?
                            .ok_or(malformed("baked reflections layer without data"))?;
                        BakedData::Reflections(BakedReflectionsData::from_table(
                            identifier,
                            batch.probes.len(),
                            &reflections,
                        )?)
                    }
                    BakedDataType::Pathing => {
                        let pathing = layer
                            .table(VT_PATHING_DATA)?
                            .ok_or(malformed("baked pathing layer without data"))?;
                        BakedData::Pathing(BakedPathData::from_table(&pathing, &batch.probes)?)
                    }
                };
                batch.add_data(identifier, data);
            }
        }

        Ok(batch)
    }
}
