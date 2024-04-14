use goldy_core::Real;
use nalgebra::{ComplexField, SVector};
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{uniform::SampleUniform, Distribution, Uniform};

use crate::ForceDrivenThermostat;

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

impl<T, const D: usize> ForceDrivenThermostat<T, D> for Langevin<T>
where
    T: Real + SampleUniform,
{
    fn thermo(
        &mut self,
        f: &mut goldy_storage::vector::Forces<T, D>,
        v: &goldy_storage::vector::Velocities<T, D>,
        types: &goldy_storage::atom_type_store::AtomTypeStore<T>,
        temp: T,
        dt: T,
    ) {
        f.iter_mut().zip(v).zip(types).for_each(|((f, v), ty)| {
            *f -= v * ty.gamma() / dt * ty.mass();

            let rand_vec = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));
            *f += rand_vec
                * ComplexField::sqrt(T::from(6).unwrap() * ty.mass() * ty.gamma() / dt * temp);
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
    use goldy_storage::{
        atom_type::AtomTypeBuilder,
        atom_type_store::AtomTypeStoreBuilder,
        vector::{Forces, Velocities},
    };

    use super::*;

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
        let v = Velocities::<f32, 3>::zeros(num_atoms);
        let mut f = Forces::zeros(num_atoms);
        let types = AtomTypeStoreBuilder::default()
            .add_many(
                AtomTypeBuilder::default()
                    .mass(mass)
                    .gamma(gamma)
                    .build()
                    .unwrap(),
                num_atoms,
            )
            .build();

        // performing one step in the thermostat
        langevin.thermo(&mut f, &v, &types, temp, dt);

        // all forces must be in [-sqrt(6*m*gamma*T), sqrt(6*m*gamma*T)]
        let bound = (6.0 * mass * gamma / dt * temp).sqrt();
        assert!(f
            .iter()
            .all(|f| f.iter().all(|x| (-bound..bound).contains(x))));

        // Some directions need to be negative ..
        assert!(f.iter().any(|f| f.iter().any(|&x| x < 0.0)));
        // ... and some positive.
        // In fact, this test ensures, that the thermostat really
        // did something.
        assert!(f.iter().any(|f| f.iter().any(|&x| x > 0.0)));
    }
}
