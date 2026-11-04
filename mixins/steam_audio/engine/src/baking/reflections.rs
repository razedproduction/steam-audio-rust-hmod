//! Baked reflections (upstream `baked_reflection_data.{h,cpp}`, `BakedReflectionsData`): an energy field
//! (convolution) and/or a reverb (parametric) per probe.

use std::borrow::Borrow;

use crate::bands::NUM_BANDS;
use crate::error::{Error, Result};
use crate::geometry::Sphere;
use crate::math::Vector3;
use crate::probes::{Probe, ProbeBatch, ProbeNeighborhood};
use crate::serialization::{Builder, Offset, Table};
use crate::simulation::{EnergyField, Reverb};

use super::{BakedData, BakedDataIdentifier};

// Vtable slots of `ipl.Serialized.BakedReflectionsData`.
const VT_ENERGY_FIELDS: u16 = 4;
const VT_REVERBS: u16 = 6;
const VT_NEEDS_UPDATE: u16 = 8;

/// Size of `ipl.Serialized.Reverb`.
const REVERB_SIZE: usize = 4 * NUM_BANDS;

/// Reflections baked at the probes of a batch (upstream `BakedReflectionsData`).
///
/// Each probe has a flag saying whether it must be baked (again); a new probe, or one that moved, needs baking. Its
/// energy field ([`has_convolution`](Self::has_convolution)) may be missing even when it is baked: the bake left it
/// out, or the endpoint moved away from the probe.
#[derive(Debug, Clone, PartialEq)]
pub struct BakedReflectionsData {
    identifier: BakedDataIdentifier,
    has_convolution: bool,
    has_parametric: bool,
    energy_fields: Vec<Option<EnergyField>>,
    reverbs: Vec<Reverb>,
    /// Upstream keeps bytes; a deserialized value other than 0 or 1 is kept and written back as it is.
    needs_update: Vec<u8>,
}

impl BakedReflectionsData {
    /// Data for `num_probes` probes, none of them baked.
    pub fn new(
        identifier: BakedDataIdentifier,
        num_probes: usize,
        has_convolution: bool,
        has_parametric: bool,
    ) -> Self {
        Self {
            identifier,
            has_convolution,
            has_parametric,
            energy_fields: if has_convolution {
                vec![None; num_probes]
            } else {
                Vec::new()
            },
            reverbs: if has_parametric {
                vec![Reverb::default(); num_probes]
            } else {
                Vec::new()
            },
            needs_update: vec![1; num_probes],
        }
    }

    pub fn identifier(&self) -> &BakedDataIdentifier {
        &self.identifier
    }

    pub fn num_probes(&self) -> usize {
        self.needs_update.len()
    }

    pub fn has_convolution(&self) -> bool {
        self.has_convolution
    }

    pub fn has_parametric(&self) -> bool {
        self.has_parametric
    }

    /// Turning convolution on makes every probe need baking. Turning it off keeps the energy fields, unused.
    pub fn set_has_convolution(&mut self, has_convolution: bool) {
        if !self.has_convolution && has_convolution {
            self.energy_fields.resize(self.needs_update.len(), None);
            self.needs_update.fill(1);
        }
        self.has_convolution = has_convolution;
    }

    /// Turning parametric reverb on makes every probe need baking. Turning it off keeps the reverbs, unused.
    pub fn set_has_parametric(&mut self, has_parametric: bool) {
        if !self.has_parametric && has_parametric {
            self.reverbs
                .resize(self.needs_update.len(), Reverb::default());
            self.needs_update.fill(1);
        }
        self.has_parametric = has_parametric;
    }

    /// Whether probe `index` must be baked. Panics if `index` is out of range.
    pub fn needs_update(&self, index: usize) -> bool {
        self.needs_update[index] != 0
    }

    /// Stores the energy field baked at probe `index` (`None`: none) and marks the probe baked (upstream
    /// `set(index, unique_ptr<EnergyField>)`).
    ///
    /// # Panics
    ///
    /// If `index` is out of range or the data has no energy fields.
    pub fn set_energy_field(&mut self, index: usize, value: Option<EnergyField>) {
        self.energy_fields[index] = value;
        self.needs_update[index] = 0;
    }

    /// Stores the reverb baked at probe `index` and marks the probe baked (upstream `set(index, const Reverb&)`).
    ///
    /// # Panics
    ///
    /// If `index` is out of range or the data has no reverbs.
    pub fn set_reverb(&mut self, index: usize, value: Reverb) {
        self.reverbs[index] = value;
        self.needs_update[index] = 0;
    }

    /// The energy field of probe `index`, if the data has convolution and the probe has one (upstream
    /// `lookupEnergyField`; an index out of range also gives `None`).
    pub fn energy_field(&self, index: usize) -> Option<&EnergyField> {
        if self.has_convolution {
            self.energy_fields.get(index)?.as_ref()
        } else {
            None
        }
    }

    /// The reverb of probe `index`, if the data is parametric (upstream `lookupReverb`; an index out of range also
    /// gives `None`).
    pub fn reverb(&self, index: usize) -> Option<&Reverb> {
        if self.has_parametric {
            self.reverbs.get(index)
        } else {
            None
        }
    }

    /// Every probe's energy field, used or not (upstream `getEnergyFields`).
    pub fn energy_fields(&self) -> &[Option<EnergyField>] {
        &self.energy_fields
    }

    /// Every probe's reverb, used or not (upstream `getReverbs`).
    pub fn reverbs(&self) -> &[Reverb] {
        &self.reverbs
    }

    pub(crate) fn update_probe_position(&mut self, index: usize, _position: Vector3) {
        self.needs_update[index] = 1;
    }

    pub(crate) fn add_probe(&mut self, _influence: &Sphere) {
        self.needs_update.push(1);
        if self.has_convolution {
            self.energy_fields.push(None);
        }
        if self.has_parametric {
            self.reverbs.push(Reverb::default());
        }
    }

    pub(crate) fn remove_probe(&mut self, index: usize) {
        self.needs_update.remove(index);
        if self.has_convolution {
            self.energy_fields.remove(index);
        }
        if self.has_parametric {
            self.reverbs.remove(index);
        }
    }

    /// Upstream `updateEndpoint`, for data whose identifier is `identifier` (otherwise nothing happens). If the
    /// endpoint's center moved, every probe needs baking. Otherwise probes now inside the endpoint's influence need
    /// baking and those no longer inside lose their data. Only the first `energy_fields().len()` probes are looked at
    /// (none without convolution), and the data keeps its identifier.
    pub(crate) fn update_endpoint(
        &mut self,
        identifier: &BakedDataIdentifier,
        probes: &[Probe],
        endpoint_influence: &Sphere,
    ) {
        if !identifier.same_as(&self.identifier) {
            return;
        }

        let previous = self.identifier.endpoint_influence;
        if previous.center != endpoint_influence.center {
            for i in 0..self.energy_fields.len() {
                self.needs_update[i] = 1;
            }
        } else {
            for (i, probe) in probes.iter().enumerate().take(self.energy_fields.len()) {
                let center = probe.influence.center;
                let is_inside = endpoint_influence.contains(center);
                let was_inside = previous.contains(center);

                if is_inside && !was_inside {
                    self.needs_update[i] = 1;
                } else if !is_inside && was_inside {
                    if self.has_convolution {
                        self.energy_fields[i] = None;
                    }
                    if self.has_parametric {
                        self.reverbs[i] = Reverb::default();
                    }
                }
            }
        }
    }

    /// Upstream `serializedSize`: an estimate that counts the flags, the baked probes' energy fields and reverbs,
    /// without the FlatBuffers overhead.
    pub fn serialized_size(&self) -> u64 {
        let mut size = (self.needs_update.len() + 2) as u64;

        if self.has_convolution {
            for (field, &needs_update) in self.energy_fields.iter().zip(&self.needs_update) {
                if needs_update == 0 {
                    size += 1;
                    if let Some(field) = field {
                        size += field.serialized_size();
                    }
                }
            }
        }

        if self.has_parametric {
            for &needs_update in self.needs_update.iter().take(self.reverbs.len()) {
                if needs_update == 0 {
                    size += REVERB_SIZE as u64;
                }
            }
        }

        size
    }

    /// Adds the energy fields of the neighbourhood's entries, scaled by their weights, to `energy_field` (upstream
    /// `evaluateEnergyField`). `this_batch` is the index in `batches` of a batch holding this data.
    ///
    /// Upstream's quirk is kept: an entry is skipped only if its batch holds *other* data under this identifier, so
    /// an entry of a batch without such data is looked up here, by its probe index in its own batch.
    pub(crate) fn evaluate_energy_field<B: Borrow<ProbeBatch>>(
        &self,
        neighborhood: &ProbeNeighborhood,
        batches: &[B],
        energy_field: &mut EnergyField,
    ) {
        for i in 0..neighborhood.num_probes() {
            let Some(probe) = self.lookup_entry(neighborhood, batches, i) else {
                continue;
            };
            let Some(probe_energy_field) = self.energy_field(probe) else {
                continue;
            };
            EnergyField::scale_accumulate(
                probe_energy_field,
                neighborhood.weights[i],
                energy_field,
            );
        }
    }

    /// Adds the reverbs of the neighbourhood's entries, scaled by their weights, to `reverb` (upstream
    /// `evaluateReverb`), with the same quirk as [`evaluate_energy_field`](Self::evaluate_energy_field).
    pub(crate) fn evaluate_reverb<B: Borrow<ProbeBatch>>(
        &self,
        neighborhood: &ProbeNeighborhood,
        batches: &[B],
        reverb: &mut Reverb,
    ) {
        for i in 0..neighborhood.num_probes() {
            let Some(probe) = self.lookup_entry(neighborhood, batches, i) else {
                continue;
            };
            let Some(probe_reverb) = self.reverb(probe) else {
                continue;
            };
            for j in 0..NUM_BANDS {
                reverb.reverb_times[j] += neighborhood.weights[i] * probe_reverb.reverb_times[j];
            }
        }
    }

    /// The probe of entry `i` to look up in this data, if any.
    fn lookup_entry<B: Borrow<ProbeBatch>>(
        &self,
        neighborhood: &ProbeNeighborhood,
        batches: &[B],
        i: usize,
    ) -> Option<usize> {
        let batch = neighborhood.batches[i]?;
        let probe = neighborhood.probe_indices[i]?;
        let other_data = match batches[batch].borrow().data(&self.identifier) {
            Some(BakedData::Reflections(data)) => !std::ptr::eq(data, self),
            Some(BakedData::Pathing(_)) => true,
            None => false,
        };
        (!other_data).then_some(probe)
    }

    /// Upstream `serialize`.
    pub(crate) fn serialize_into(&self, builder: &mut Builder) -> Offset {
        let needs_update = builder.create_vector(self.needs_update.len(), 1, |bytes| {
            bytes.copy_from_slice(&self.needs_update);
        });

        let energy_fields = self.has_convolution.then(|| {
            let offsets: Vec<Option<Offset>> = self
                .energy_fields
                .iter()
                .zip(&self.needs_update)
                .map(|(field, &needs_update)| match field {
                    Some(field) if needs_update == 0 => Some(field.serialize_into(builder)),
                    _ => None,
                })
                .collect();
            builder.create_vector_of_optional_offsets(&offsets)
        });

        let reverbs = self.has_parametric.then(|| {
            builder.create_vector_of_structs(self.reverbs.len(), REVERB_SIZE, |bytes| {
                for (out, reverb) in bytes.chunks_exact_mut(REVERB_SIZE).zip(&self.reverbs) {
                    for (out, x) in out.chunks_exact_mut(4).zip(reverb.reverb_times) {
                        out.copy_from_slice(&x.to_le_bytes());
                    }
                }
            })
        });

        let start = builder.start_table();
        builder.add_offset(VT_NEEDS_UPDATE, Some(needs_update));
        builder.add_offset(VT_REVERBS, reverbs);
        builder.add_offset(VT_ENERGY_FIELDS, energy_fields);
        builder.end_table(start)
    }

    /// Upstream's deserializing constructor, for a batch of `num_probes` probes. Energy fields of probes that need
    /// baking are not read. Flags and reverbs missing at the end are 0 (upstream copies what there is into zeroed
    /// arrays); more of them than probes is an error (upstream overruns its arrays).
    pub(crate) fn from_table(
        identifier: BakedDataIdentifier,
        num_probes: usize,
        table: &Table<'_>,
    ) -> Result<Self> {
        let malformed = Error::MalformedSerializedData;

        let serialized_needs_update = table
            .vector(VT_NEEDS_UPDATE, 1)?
            .filter(|v| v.len() > 0)
            .ok_or(malformed("baked reflections without probe flags"))?;
        if serialized_needs_update.len() > num_probes {
            return Err(malformed(
                "baked reflections for more probes than the batch has",
            ));
        }
        let mut needs_update = vec![0; num_probes];
        for (i, flag) in needs_update
            .iter_mut()
            .enumerate()
            .take(serialized_needs_update.len())
        {
            *flag = serialized_needs_update.element(i)[0];
        }

        let serialized_energy_fields = table.vector(VT_ENERGY_FIELDS, 4)?;
        let serialized_reverbs = table.vector(VT_REVERBS, REVERB_SIZE)?;
        let has_convolution = serialized_energy_fields.is_some();
        let has_parametric = serialized_reverbs.is_some();

        let mut energy_fields = Vec::new();
        if let Some(serialized) = serialized_energy_fields {
            energy_fields = vec![None; num_probes];
            for (i, field) in energy_fields.iter_mut().enumerate() {
                if needs_update[i] == 0 && i < serialized.len() {
                    *field = serialized
                        .optional_table(i)?
                        .map(|table| EnergyField::from_table(&table))
                        .transpose()?;
                }
            }
        }

        let mut reverbs = Vec::new();
        if let Some(serialized) = serialized_reverbs {
            if serialized.len() > num_probes {
                return Err(malformed(
                    "baked reverbs for more probes than the batch has",
                ));
            }
            reverbs = vec![Reverb::default(); num_probes];
            for (i, reverb) in reverbs.iter_mut().enumerate().take(serialized.len()) {
                for (time, word) in reverb.reverb_times.iter_mut().zip(serialized.words(i)) {
                    *time = f32::from_le_bytes(word);
                }
            }
        }

        Ok(Self {
            identifier,
            has_convolution,
            has_parametric,
            energy_fields,
            reverbs,
            needs_update,
        })
    }
}
