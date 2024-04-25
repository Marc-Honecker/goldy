use crate::{
    force_update::ForceUpdate, propagator::Propagator, simulation_box::SimulationBox,
    storage::atom_store::AtomStore, Real,
};

pub struct VelocityVerlet;

impl Propagator for VelocityVerlet {
    fn integrate<T: Real, const D: usize>(
        atom_store: &mut AtomStore<T, D>,
        sim_box: &SimulationBox<T, D>,
        updater: &mut ForceUpdate<T, D>,
        dt: T,
        temp: T,
    ) -> Option<T> {
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
        let potential_energy = updater.update_forces(atom_store, sim_box, temp, dt);

        // And now we can update the velocities the last half timestep.
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|((v, &f), t)| *v += f / t.mass() * T::from(0.5).unwrap() * dt);

        potential_energy
    }
}
