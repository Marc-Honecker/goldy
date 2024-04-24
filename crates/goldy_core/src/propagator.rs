use crate::{
    potential::Potential, simulation_box::SimulationBox, storage::atom_store::AtomStore,
    thermo::ForceDrivenThermostat, Real,
};

pub mod euler;
pub mod velocity_verlet;

pub trait Propagator {
    /// Propagates a system in time and returns the accumulated potential-energy,
    /// if the potential is given.
    fn integrate<T, const D: usize, Pot, FDT>(
        atom_store: &mut AtomStore<T, D>,
        sim_box: &SimulationBox<T, D>,
        potential: Option<&Pot>,
        thermostat: Option<&mut FDT>,
        dt: T,
        temp: T,
    ) -> Option<T>
    where
        T: Real,
        Pot: Potential<T, D>,
        FDT: ForceDrivenThermostat<T, D>;
}
