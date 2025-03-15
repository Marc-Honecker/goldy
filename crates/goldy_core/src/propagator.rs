use crate::neighbor_list::NeighborList;
use crate::{
    Real, force_update::ForceUpdate, simulation_box::SimulationBox, storage::atom_store::AtomStore,
};

pub mod best_possible_conv_langevin;
pub mod gronbech_jensen;
pub mod leapfrog_verlet;
pub mod lowest_order_langevin;
pub mod velocity_verlet;

pub trait Propagator<T: Real, const D: usize> {
    /// Propagates a system in time and returns the accumulated potential-energy,
    /// if the potential is given.
    fn integrate(
        &mut self,
        atom_store: &mut AtomStore<T, D>,
        neighbor_list: &NeighborList<T, D>,
        sim_box: &SimulationBox<T, D>,
        updater: &mut ForceUpdate<T, D>,
        dt: T,
        temp: T,
    );
}
