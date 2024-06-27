use crate::{
    force_update::ForceUpdate, simulation_box::SimulationBox, storage::atom_store::AtomStore, Real,
};

pub mod euler;
pub mod velocity_verlet;

pub trait Propagator {
    /// Propagates a system in time and returns the accumulated potential-energy,
    /// if the potential is given.
    fn integrate<T: Real, const D: usize>(
        atom_store: &mut AtomStore<T, D>,
        neighbor_list: &[&[usize]],
        sim_box: &SimulationBox<T, D>,
        updater: &mut ForceUpdate<T, D>,
        dt: T,
        temp: T,
    );
}
