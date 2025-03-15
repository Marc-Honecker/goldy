use crate::Real;
use crate::storage::atom_store::AtomStore;

pub mod langevin;

pub trait Thermostat<T: Real, const D: usize> {
    fn apply_thermostat(&mut self, atoms: &mut AtomStore<T, D>, temp: T, dt: T);
}
