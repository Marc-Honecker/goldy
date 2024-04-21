use derive_builder::Builder;
use goldy_core::Real;

use crate::{
    atom_type_store::AtomTypeStore,
    vector::{Forces, Positions, Velocities},
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
    /// Returns a reference to the positions.
    pub fn positions(&self) -> &Positions<T, D> {
        &self.x
    }

    /// Returns a mutable reference to the positions.
    pub fn positions_mut(&mut self) -> &mut Positions<T, D> {
        &mut self.x
    }

    /// Returns a reference to the velocities.
    pub fn velocities(&self) -> &Velocities<T, D> {
        &self.v
    }

    /// Returns a mutable reference to the velocities.
    pub fn velocities_mut(&mut self) -> &mut Velocities<T, D> {
        &mut self.v
    }

    /// Returns a reference to the forces.
    pub fn forces(&self) -> &Forces<T, D> {
        &self.f
    }

    /// Returns a mutable reference to the forces.
    pub fn forces_mut(&mut self) -> &mut Forces<T, D> {
        &mut self.f
    }

    pub fn atom_types(&self) -> &AtomTypeStore<T> {
        &self.atom_types
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
    use crate::{atom_type::AtomTypeBuilder, atom_type_store::AtomTypeStoreBuilder};

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
                    .mass(1.0)
                    .gamma(0.01)
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
                    .mass(1.0)
                    .gamma(0.01)
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
