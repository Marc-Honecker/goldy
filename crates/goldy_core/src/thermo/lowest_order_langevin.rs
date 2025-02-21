use std::marker::PhantomData;

use crate::storage::{atom_store::AtomStore, vector::Iterable};
use crate::Real;
use nalgebra::SVector;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, StandardNormal};

pub struct LowestOrderLangevin<T>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    rng: ChaChaRng,
    distr: StandardNormal,
    phantom_data: PhantomData<T>,
}

impl<T> LowestOrderLangevin<T>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    pub fn new() -> Self {
        let rng = ChaChaRng::from_entropy();
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

                let c_vv = T::one() - dt / tau;
                let c_vf = dt / at.mass();
                let c_vg = num_traits::Float::sqrt(T::from(2.0).unwrap() * temp * dt / tau);
                let c_xv = dt;

                let g = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

                *v = *v * c_vv + *f * c_vf + g * c_vg;
                *x = *x + *v * c_xv;
            });
    }
}
