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
            *f -= v * ty.damping() / dt * ty.mass();

            let rand_vec = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));
            *f += rand_vec
                * ComplexField::sqrt(T::from(6).unwrap() * ty.mass() * ty.damping() / dt * temp);
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
