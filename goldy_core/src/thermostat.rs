use crate::vector::{Force, Velocity};

pub trait ForceDrivenThermostat<const D: usize> {
    fn thermostat(&mut self, force: &mut [Force<D>], vel: &[Velocity<D>]);
}
