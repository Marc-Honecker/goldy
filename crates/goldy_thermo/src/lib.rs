use goldy_core::Real;
use goldy_storage::{
    atom_type_store::AtomTypeStore,
    vector::{Forces, Velocities},
};

pub mod langevin;

pub trait ForceDrivenThermostat<T: Real, const D: usize> {
    fn thermo(
        &mut self,
        f: &mut Forces<T, D>,
        v: &Velocities<T, D>,
        types: &AtomTypeStore<T>,
        temp: T,
        dt: T,
    );
}
