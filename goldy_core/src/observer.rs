use crate::{units::Energy, vector::Velocity};

#[allow(unused)]
#[derive(Debug, Default)]
pub struct Observer {
    kinetic_energy_mean: Energy,
    sq_kinetic_energy_mean: Energy,
    potential_energy_mean: Energy,
    sq_potential_energy_mean: Energy,
}

impl Observer {
    pub fn new() -> Self {
        Self::default()
    }

    // TODO: implement
    pub fn measure_kinetic_energy<const D: usize>(&mut self, _vel: &[Velocity<D>]) {
        unimplemented!()
    }

    // TODO: implement
    pub fn measure_potential_energy(&mut self) {
        unimplemented!()
    }

    // TODO: implement
    pub fn dump_results(&self) {
        unimplemented!()
    }
}
