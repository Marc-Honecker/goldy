use crate::Real;
use crate::storage::{atom_store::AtomStore, vector::Iterable};
use nalgebra::SVector;
use num_traits::Float;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, StandardNormal};

pub struct GronbechJensen<T, const D: usize>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    rng: ChaChaRng,
    distr: StandardNormal,
    g_old: Vec<SVector<T, D>>,
}

impl<T, const D: usize> GronbechJensen<T, D>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    pub fn new(num_atoms: usize) -> Self {
        let mut rng = ChaChaRng::from_os_rng();
        let distr = StandardNormal;

        let mut g_old = vec![SVector::<T, D>::zeros(); num_atoms];
        for g in &mut g_old {
            *g = SVector::<T, D>::from_iterator(distr.sample_iter(&mut rng));
        }

        Self { rng, distr, g_old }
    }

    pub fn propagate(&mut self, atom_store: &mut AtomStore<T, D>, dt: T, temp: T) {
        atom_store
            .x
            .iter_mut()
            .zip(&mut atom_store.v)
            .zip(&mut atom_store.f)
            .zip(&atom_store.atom_types)
            .zip(&mut self.g_old)
            .for_each(|((((x, v), f), at), g_old)| {
                // compute some "constants"
                let tau = at.mass() / at.damping();

                let c_vv = Float::exp(-dt / tau);
                let c_xv = Float::sqrt((T::one() - c_vv) * tau * dt);
                let c_vf = c_xv / at.mass();
                let c_vg =
                    Float::sqrt(temp / at.mass() * (T::one() - c_vv) / T::from(2.0).unwrap());

                // drawing a new random number
                let g_new =
                    SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

                // updating the velocity
                *v = *v * c_vv + *f * c_vf + (*g_old + g_new) * c_vg;
                // the new random number becomes the new g_old
                *g_old = g_new;

                // updating the position
                *x = *x + *v * c_xv;
            });
    }
}
