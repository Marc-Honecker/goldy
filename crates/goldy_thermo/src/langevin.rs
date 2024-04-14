use goldy_core::Real;
use goldy_storage::{
    atom_type_store::AtomTypeStore,
    vector::{Forces, Velocities},
};
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
        f: &mut Forces<T, D>,
        v: &Velocities<T, D>,
        types: &AtomTypeStore<T>,
        temp: T,
        dt: T,
    ) {
        f.iter_mut().zip(v).zip(types).for_each(|((f, v), ty)| {
            // rescaling the damping
            // safety: T must be a number, so it's save to simply unwrap at this point.
            let dp = ty.mass() * ty.gamma() / dt * (T::from(1).unwrap() + ty.gamma());

            // adding the non-iteracting forces
            *f -= v * dp;

            // Drawing the random vector.
            let rand_vec = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

            // adding the random forces
            // safety: the same as above
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
        let dp = mass * gamma / dt * (1.0 + gamma);
        let bound = (6.0 * dp / dt * temp).sqrt();
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
