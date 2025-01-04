use crate::storage::atom_store::AtomStore;
use crate::Real;

pub mod langevin;

pub trait Thermostat<T: Real, const D: usize> {
    fn thermostat(&mut self, atoms: &mut AtomStore<T, D>, temp: T, dt: T);
}
