use crate::vector::{Force, Velocity};

pub trait ImplicitThermostat {
    fn thermostat(&mut self, force: &mut [Force], vel: &[Velocity]);
}
