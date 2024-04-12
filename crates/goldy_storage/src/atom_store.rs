use derive_builder::Builder;
use goldy_core::Real;

use crate::{
    atom_type_store::AtomTypeStore,
    vector::{Forces, Positions, Velocities},
};

#[derive(Debug, Clone, Builder)]
/// Stores the atoms of a simulation and ensures, that everything remains in sync.
pub struct AtomStore<T: Real, const D: usize> {
    x: Positions<T, D>,
    v: Velocities<T, D>,
    f: Forces<T, D>,
    atom_type_store: AtomTypeStore<T>,
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
        &self.atom_type_store
    }
}
