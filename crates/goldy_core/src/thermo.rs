use crate::Real;
use crate::storage::atom_store::AtomStore;

pub mod best_possible_conv_langevin;
pub mod langevin;
pub mod lowest_order_langevin;

pub trait Thermostat<T: Real, const D: usize> {
    fn apply_thermostat(&mut self, atoms: &mut AtomStore<T, D>, temp: T, dt: T);
}
