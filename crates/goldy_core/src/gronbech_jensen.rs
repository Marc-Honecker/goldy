use crate::storage::atom_store::AtomStore;
use crate::Real;
use nalgebra::SVector;
use rand_chacha::rand_core::SeedableRng;
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
        let mut rng = ChaChaRng::from_entropy();
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
            .for_each(|((((x, v), f), at), g)| {
                let tau = at.mass() / at.damping();

                let dt_half = dt / T::from(2.0).unwrap();
                let c3 =
                    num_traits::Float::exp(dt_half) * num_traits::Float::sinh(dt_half) / dt_half;
                let c3 = num_traits::Float::sqrt(c3);

                let c_rv = c3 * dt;
                let c_vv = num_traits::Float::exp(-dt / tau);
                let c_vf = c_rv / at.mass();
                let c_vg = num_traits::Float::sqrt(temp * dt / (T::from(2.0).unwrap() * at.mass()));
                let c_vg = c_vg * c3;

                *v *= c_vv;
                *v += *f * c_vf;

                let g_new =
                    SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));
                *v += (*g + g_new) * c_vg;
                *g = g_new;

                *x += *v * c_rv;
            });
    }
}
