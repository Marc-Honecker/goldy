use goldy_core::Real;
use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, Normal};

use crate::{
    atom_type::AtomType,
    atom_type_store::{AtomTypeStore, AtomTypeStoreBuilder},
    vector::{Forces, Positions, Velocities},
};

#[derive(Debug, Clone)]
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

/// Builder-struct for the `AtomStore`.
#[allow(unused)]
pub struct AtomStoreBuilder<T: Real, const D: usize> {
    x: Positions<T, D>,
    v: Velocities<T, D>,
    f: Forces<T, D>,
    atom_type_store: AtomTypeStore<T>,
}

impl<T, const D: usize> AtomStoreBuilder<T, D>
where
    T: Real,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
{
    pub fn new_with_gaussian_positions(
        n: usize,
        mean: T,
        std_dev: T,
        at: AtomType<T>,
    ) -> AtomStore<T, D> {
        let mut rng = ChaChaRng::from_entropy();
        let gauss = Normal::new(mean, std_dev).unwrap();

        let x = (0..n)
            .map(|_| SVector::from_iterator(gauss.sample_iter(&mut rng)))
            .collect::<Positions<T, D>>();
        let v = (0..n)
            .map(|_| SVector::zeros())
            .collect::<Velocities<T, D>>();
        let f = (0..n).map(|_| SVector::zeros()).collect::<Forces<T, D>>();
        let atom_type_store = AtomTypeStoreBuilder::new().add_many(at, n).build();

        AtomStore {
            x,
            v,
            f,
            atom_type_store,
        }
    }
}
