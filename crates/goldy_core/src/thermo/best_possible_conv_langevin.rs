use std::marker::PhantomData;

use crate::storage::{atom_store::AtomStore, vector::Iterable};
use crate::Real;
use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, StandardNormal};

pub struct BestPossibleConvLangevin<T>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    rng: ChaChaRng,
    distr: StandardNormal,
    phantom_data: PhantomData<T>,
}

impl<T> BestPossibleConvLangevin<T>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    pub fn new() -> Self {
        let rng = ChaChaRng::from_os_rng();
        let distr = StandardNormal;

        Self {
            rng,
            distr,
            phantom_data: PhantomData,
        }
    }

    pub fn propagate<const D: usize>(&mut self, atom_store: &mut AtomStore<T, D>, dt: T, temp: T) {
        atom_store
            .x
            .iter_mut()
            .zip(&mut atom_store.v)
            .zip(&mut atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|(((x, v), f), at)| {
                let tau = at.mass() / at.damping();

                let c_vv = num_traits::Float::exp(-dt / tau);
                let c_vf = num_traits::Float::sqrt(
                    tau * dt * (T::one() - c_vv * c_vv) / T::from(2.0).unwrap(),
                ) / at.mass();
                let c_vg = num_traits::Float::sqrt(T::from(2.0).unwrap() * dt * temp / tau);
                let c_xv = num_traits::Float::sqrt(
                    T::from(2.0).unwrap() * tau * dt * (T::one() - c_vv) / (T::one() + c_vv),
                );

                let g = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

                *v = *v * c_vv + *f * c_vf + g * c_vg;
                *x = *x + *v * c_xv;
            });
    }
}
