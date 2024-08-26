use crate::storage::atom_store::AtomStore;
use crate::Real;

/// This struct handles all observations.
pub struct Observer<T: Real> {
    kinetic_energy: T,
}

impl<T: Real> Observer<T> {
    pub fn observe_kinetic_energy<const D: usize>(&mut self, atoms: &AtomStore<T, D>) {
        self.kinetic_energy += T::from(0.5).unwrap()
            * atoms
                .v
                .iter()
                .zip(&atoms.atom_types)
                .fold(T::zero(), |acc, (&v, &t)| acc + t.mass() * v.dot(&v))
            / T::from(atoms.number_of_atoms()).unwrap();
    }

    pub fn get_kinetic_energy(&self) -> T {
        self.kinetic_energy
    }
}
