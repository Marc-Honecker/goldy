use crate::neighbor_list::NeighborList;
use crate::storage::vector::Iterable;
use crate::{
    Real, force_update::ForceUpdate, propagator::Propagator, simulation_box::SimulationBox,
    storage::atom_store::AtomStore,
};

#[derive(Debug, Clone, Copy, Default)]
pub struct VelocityVerlet;

impl VelocityVerlet {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Real, const D: usize> Propagator<T, D> for VelocityVerlet {
    fn integrate(
        &mut self,
        atom_store: &mut AtomStore<T, D>,
        neighbor_list: &NeighborList<T, D>,
        sim_box: &SimulationBox<T, D>,
        updater: &mut ForceUpdate<T, D>,
        dt: T,
        temp: T,
    ) {
        // First, we need to update the velocities half a timestep.
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|((v, &f), &t)| *v += f / t.mass() * T::from(0.5).unwrap() * dt);

        // Now we can update the positions.
        atom_store
            .x
            .iter_mut()
            .zip(&atom_store.v)
            .for_each(|(x, &v)| *x += v * dt);

        // updating the forces
        updater.update_forces(atom_store, neighbor_list, sim_box, temp, dt);

        // And now we can update the velocities the last half timestep.
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|((v, &f), t)| *v += f / t.mass() * T::from(0.5).unwrap() * dt);
    }
}
