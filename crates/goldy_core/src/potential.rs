use crate::{
    Real,
    simulation_box::SimulationBox,
    storage::{
        atom_type_store::AtomTypeStore,
        vector::{Forces, Positions},
    },
};

pub mod harmonic_oscillator;
pub mod pair_potential;
pub mod pair_potential_collection;

pub trait Potential<T: Real, const D: usize> {
    /// Measures the potential energy.
    fn measure_energy(
        &self,
        x: &Positions<T, D>,
        neighbor_list: &[&[usize]],
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    ) -> T;

    /// Updates the forces.
    fn update_forces(
        &self,
        x: &Positions<T, D>,
        neighbor_list: &[&[usize]],
        f: &mut Forces<T, D>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    );
}
