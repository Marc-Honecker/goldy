use derive_builder::Builder;

use crate::{
    storage::{
        atom_type_store::AtomTypeStore,
        vector::{Forces, Positions, Velocities},
    },
    Real,
};

#[derive(Debug, Clone, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
/// Stores the atoms of a simulation and ensures, that everything remains in sync.
pub struct AtomStore<T: Real, const D: usize> {
    #[builder(setter(name = "positions"))]
    pub x: Positions<T, D>,
    #[builder(setter(name = "velocities"))]
    pub v: Velocities<T, D>,
    #[builder(setter(name = "forces"))]
    pub f: Forces<T, D>,
    pub atom_types: AtomTypeStore<T>,
}

impl<T: Real, const D: usize> AtomStore<T, D> {
    /// Returns the number of atoms in this `AtomStore`.
    pub fn number_of_atoms(&self) -> usize {
        self.x.data.len()
    }
}

impl<T: Real, const D: usize> AtomStoreBuilder<T, D> {
    fn validate(&self) -> Result<(), String> {
        let err_msg = "Please provide the same amount of atoms for all properties";

        if let Some(x) = &self.x {
            if let Some(v) = &self.v {
                if v.data.len() != x.data.len() {
                    return Err(err_msg.into());
                }
            }

            if let Some(f) = &self.f {
                if f.data.len() != x.data.len() {
                    return Err(err_msg.into());
                }
            }

            if let Some(atom_types) = &self.atom_types {
                if atom_types.data.len() != x.data.len() {
                    return Err(err_msg.into());
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::{atom_type::AtomTypeBuilder, atom_type_store::AtomTypeStoreBuilder};

    use super::*;

    #[test]
    fn test_atomstore_builder() {
        // Let's first test the easy case
        let num_atoms = 1_000;

        // Creates a bunch of properties.
        let x = Positions::<f64, 3>::zeros(num_atoms);
        let v = Velocities::<f64, 3>::zeros(num_atoms);
        let f = Forces::<f64, 3>::zeros(num_atoms);
        let ats = AtomTypeStoreBuilder::default()
            .add_many(
                AtomTypeBuilder::default()
                    .id(0)
                    .mass(1.0)
                    .damping(0.01)
                    .build()
                    .unwrap(),
                num_atoms,
            )
            .build();

        let atom_store = AtomStoreBuilder::default()
            .positions(x)
            .velocities(v)
            .forces(f)
            .atom_types(ats)
            .build()
            .unwrap();

        assert!(
            atom_store.x.data.len() == num_atoms
                && atom_store.v.data.len() == num_atoms
                && atom_store.f.data.len() == num_atoms
                && atom_store.atom_types.data.len() == num_atoms
        );

        // Now let's test when one element doesn't match in size
        let num_atoms = 1_000;

        // Creates a bunch of properties.
        let x = Positions::<f64, 3>::zeros(num_atoms);
        let v = Velocities::<f64, 3>::zeros(num_atoms);
        let f = Forces::<f64, 3>::zeros(num_atoms);
        // Here comes the bug.
        let ats = AtomTypeStoreBuilder::default()
            .add(
                AtomTypeBuilder::default()
                    .id(0)
                    .mass(1.0)
                    .damping(0.01)
                    .build()
                    .unwrap(),
            )
            .build();

        // This build should fail.
        assert!(AtomStoreBuilder::default()
            .positions(x)
            .velocities(v)
            .forces(f)
            .atom_types(ats)
            .build()
            .is_err());
    }
}
