use crate::neighbor_list::NeighborList;
use crate::{
    simulation_box::SimulationBox,
    storage::{
        atom_type_store::AtomTypeStore,
        vector::{Forces, Positions},
    },
    Real,
};

pub mod harmonic_oscillator;
pub mod pair_potential;
pub mod pair_potential_collection;

pub trait Potential<T: Real, const D: usize> {
    /// Measures the potential energy.
    fn measure_energy(
        &mut self,
        x: &Positions<T, D>,
        neighbor_list: &NeighborList<T, D>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    ) -> T;

    /// Updates the forces.
    fn update_forces(
        &mut self,
        x: &Positions<T, D>,
        neighbor_list: &NeighborList<T, D>,
        f: &mut Forces<T, D>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    );
}
