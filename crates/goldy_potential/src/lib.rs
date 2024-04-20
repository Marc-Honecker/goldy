use goldy_box::SimulationBox;
use goldy_core::Real;
use goldy_storage::{
    atom_type_store::AtomTypeStore,
    vector::{Forces, Positions},
};

pub mod harmonic_oscillator;

pub trait Potential<T: Real, const D: usize> {
    /// Evaluates the forces and energies.
    /// This method returns the accumulated energy and updates the forces accordingly.
    fn eval(
        &self,
        x: &Positions<T, D>,
        f: &mut Forces<T, D>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    ) -> T;
}
