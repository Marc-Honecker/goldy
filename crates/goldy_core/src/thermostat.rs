use crate::{
    vector::{Force, Velocity},
    Real,
};

pub trait ForceDrivenThermostat<T, const D: usize>
where
    T: Real,
{
    fn thermostat(&mut self, force: &mut [Force<T, D>], vel: &[Velocity<T, D>]);
}
