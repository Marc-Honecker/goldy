use nalgebra::{ComplexField, SVector};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{uniform::SampleUniform, Distribution, Uniform};

use crate::storage::atom_store::AtomStore;
use crate::{
    thermo::Thermostat,
    Real,
};

pub struct Langevin<T>
where
    T: Real + SampleUniform,
{
    rng: ChaChaRng,
    distr: Uniform<T>,
}

impl<T> Langevin<T>
where
    T: Real + SampleUniform,
{
    /// Creates a new `Langevin`-thermostat.
    pub fn new() -> Self {
        Self {
            rng: ChaChaRng::from_entropy(),
            distr: Uniform::<T>::new_inclusive(-T::one(), T::one()),
        }
    }
}

impl<T, const D: usize> Thermostat<T, D> for Langevin<T>
where
    T: Real + SampleUniform,
{
    fn apply_thermostat(&mut self, atoms: &mut AtomStore<T, D>, temp: T, dt: T) {
        atoms
            .f
            .iter_mut()
            .zip(&atoms.v)
            .zip(&atoms.atom_types)
            .for_each(|((f, v), ty)| {
                // precomputing a constant
                let dp = ty.mass() * ty.damping() / dt * (T::one() + ty.damping());

                // adding the deterministic forces
                *f -= v * dp;

                // Drawing the random vector.
                let rand_vec =
                    SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

                // adding the random forces
                *f += rand_vec * ComplexField::sqrt(T::from(6).unwrap() * dp / dt * temp);
            });
    }
}

impl<T> Default for Langevin<T>
where
    T: Real + SampleUniform,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::atom_store::AtomStoreBuilder;
    use crate::storage::vector::Positions;
    use crate::storage::{
        atom_type::AtomTypeBuilder,
        atom_type_store::AtomTypeStoreBuilder,
        vector::{Forces, Velocities},
    };

    #[test]
    fn test_langevin() {
        // some parameters
        let num_atoms = 1_000;
        let temp = 1.0;
        let dt = 0.001;
        let mass = 39.95;
        let gamma = 0.01;

        // the test-object
        let mut langevin = Langevin::new();

        // the atoms
        let types = AtomTypeStoreBuilder::default()
            .add_many(
                AtomTypeBuilder::default()
                    .id(0)
                    .mass(mass)
                    .damping(gamma)
                    .build()
                    .unwrap(),
                num_atoms,
            )
            .build();

        let mut atoms = AtomStoreBuilder::default()
            .positions(Positions::<f32, 3>::zeros(num_atoms))
            .velocities(Velocities::zeros(num_atoms))
            .forces(Forces::zeros(num_atoms))
            .atom_types(types)
            .build()
            .unwrap();

        // performing one step in the thermostat
        langevin.apply_thermostat(&mut atoms, temp, dt);

        // Some directions need to be negative ...
        assert!(atoms.f.iter().any(|f| f.iter().any(|&x| x < 0.0)));
        // ... and some positive.
        // In fact, this test ensures, that the thermostat really
        // did something.
        assert!(atoms.f.iter().any(|f| f.iter().any(|&x| x > 0.0)));
    }
}
