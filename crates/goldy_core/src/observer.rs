use nalgebra::SVector;

use crate::force_update::ForceUpdate;
use crate::neighbor_list::NeighborList;
use crate::storage::atom_store::AtomStore;
use crate::storage::vector::Iterable;
use crate::system::System;
use crate::Real;

/// This struct handles all observations.
#[derive(Default)]
pub struct Observer<T: Real> {
    kinetic_energies: Vec<T>,
    potential_energies: Vec<T>,
}

impl<T: Real> Observer<T> {
    /// Creates a new empty `EnergyObserver`.
    pub fn new() -> Self {
        Self {
            kinetic_energies: Vec::new(),
            potential_energies: Vec::new(),
        }
    }

    /// Observes the kinetic energy.
    pub fn observe_kinetic_energy<const D: usize>(&mut self, atoms: &AtomStore<T, D>) {
        self.kinetic_energies.push(
            T::from(0.5).unwrap()
                * atoms
                    .v
                    .iter()
                    .zip(&atoms.atom_types)
                    .fold(T::zero(), |acc, (&v, &t)| acc + t.mass() * v.dot(&v))
                / T::from(atoms.number_of_atoms()).unwrap(),
        );
    }

    /// Returns the average of all kinetic energy observations.
    pub fn get_mean_kinetic_energy(&self) -> T {
        self.kinetic_energies
            .iter()
            .fold(T::zero(), |acc, &e| acc + e)
            / T::from_usize(self.kinetic_energies.len()).unwrap()
    }

    /// Observes the potential energy.
    pub fn observe_potential_energy<const D: usize>(
        &mut self,
        system: &System<T, D>,
        updater: &ForceUpdate<T, D>,
        neighbor_list: &NeighborList<T, D>,
    ) {
        if let Some(pot) = updater.measure_energy(
            &system.atoms.x,
            neighbor_list,
            &system.sim_box,
            &system.atoms.atom_types,
        ) {
            self.potential_energies
                .push(pot / T::from_usize(system.number_of_atoms()).unwrap());
        }
    }

    /// Returns the average of all potential energy observations.
    #[inline]
    pub fn get_mean_potential_energy(&self) -> Option<T> {
        if self.potential_energies.is_empty() {
            None
        } else {
            Some(
                self.potential_energies
                    .iter()
                    .fold(T::zero(), |acc, &e| acc + e)
                    / T::from_usize(self.potential_energies.len())?,
            )
        }
    }

    /// Returns a reference to all kinetic energy observations.
    #[inline]
    pub fn get_kinetic_energy_observations(&self) -> &Vec<T> {
        &self.kinetic_energies
    }

    /// Returns a reference to all potential energy observations.
    #[inline]
    pub fn get_potential_energy_observations(&self) -> &Vec<T> {
        &self.potential_energies
    }

    #[inline]
    pub fn compute_mean_vector<I, const D: usize>(&self, observable: &I) -> SVector<T, D>
    where
        I: Iterable<Item = SVector<T, D>>,
    {
        observable.iter().fold(SVector::zeros(), |acc, x| acc + x)
            / T::from(observable.len()).unwrap()
    }

    #[inline]
    pub fn compute_mean<I, const D: usize>(&self, observable: &I) -> T
    where
        I: Iterable<Item = SVector<T, D>>,
    {
        self.compute_mean_vector(observable).sum() / T::from(D).unwrap()
    }

    #[inline]
    pub fn compute_nth_moment_vector<I, const D: usize>(
        &self,
        observable: &I,
        n: usize,
    ) -> SVector<T, D>
    where
        I: Iterable<Item = SVector<T, D>>,
    {
        assert!(n > 1, "Please use this method with n > 1");

        let mean_vector = self.compute_mean_vector(observable);

        observable.iter().fold(SVector::zeros(), |acc, x| {
            let mut diff = x - mean_vector;
            diff.apply(|d| *d = num_traits::Float::powi(*d, n as i32));

            acc + diff
        })
    }

    #[inline]
    pub fn compute_nth_moment<I, const D: usize>(&self, observable: &I, n: usize) -> T
    where
        I: Iterable<Item = SVector<T, D>>,
    {
        self.compute_nth_moment_vector(observable, n).sum() / T::from(D).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::storage::vector::Positions;
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_mean() {
        let pos: Positions<f64, 3> = Positions::zeros(1_000);
        let observer = Observer::new();

        let mean_vector = observer.compute_mean_vector(&pos);

        assert_approx_eq!(mean_vector.sum(), 0.0);
    }
}
