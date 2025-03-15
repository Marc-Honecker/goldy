use std::marker::PhantomData;

use crate::Real;
use crate::force_update::ForceUpdate;
use crate::neighbor_list::NeighborList;
use crate::propagator::Propagator;
use crate::simulation_box::SimulationBox;
use crate::storage::{atom_store::AtomStore, vector::Iterable};
use nalgebra::SVector;
use num_traits::Float;
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

                let c_vv = Float::exp(-dt / tau);
                let c_xv = Float::sqrt(
                    T::from(2.0).unwrap() * tau * dt * (T::one() - c_vv) / (T::one() + c_vv),
                );
                let c_vf = Float::sqrt(
                    (T::one() - Float::powi(c_vv, 2)) * tau * dt / T::from(2.0).unwrap(),
                ) / at.mass();
                let c_vg = Float::sqrt(temp / at.mass() * (T::one() - Float::powi(c_vv, 2)));

                let g = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

                *v = *v * c_vv + *f * c_vf + g * c_vg;
                *x = *x + *v * c_xv;
            });
    }
}

impl<T, const D: usize> Propagator<T, D> for BestPossibleConvLangevin<T>
where
    T: Real,
    StandardNormal: Distribution<T>,
{
    fn integrate(
        &mut self,
        atom_store: &mut AtomStore<T, D>,
        neighbor_list: &NeighborList<T, D>,
        sim_box: &SimulationBox<T, D>,
        updater: &mut ForceUpdate<T, D>,
        dt: T,
        temp: T,
    ) {
        // setting the forces to zero
        atom_store.f.set_to_zero();

        // updating the forces with the potential
        updater.update_forces(atom_store, neighbor_list, sim_box, temp, dt);

        atom_store
            .x
            .iter_mut()
            .zip(&mut atom_store.v)
            .zip(&mut atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|(((x, v), f), at)| {
                let tau = at.mass() / at.damping();

                let c_vv = Float::exp(-dt / tau);
                let c_xv = Float::sqrt(
                    T::from(2.0).unwrap() * tau * dt * (T::one() - c_vv) / (T::one() + c_vv),
                );
                let c_vf = Float::sqrt(
                    (T::one() - Float::powi(c_vv, 2)) * tau * dt / T::from(2.0).unwrap(),
                ) / at.mass();
                let c_vg = Float::sqrt(temp / at.mass() * (T::one() - Float::powi(c_vv, 2)));

                let g = SVector::<T, D>::from_iterator((&self.distr).sample_iter(&mut self.rng));

                *v = *v * c_vv + *f * c_vf + g * c_vg;
                *x = *x + *v * c_xv;
            });
    }
}
