use crate::{units::Energy, vector::Velocity, Real};

#[allow(unused)]
#[derive(Debug)]
pub struct Observer<T>
where
    T: Real,
{
    kinetic_energy_mean: Energy<T>,
    sq_kinetic_energy_mean: Energy<T>,
    potential_energy_mean: Energy<T>,
    sq_potential_energy_mean: Energy<T>,
}

impl<T> Observer<T>
where
    T: Real,
{
    pub fn new() -> Self {
        Self {
            kinetic_energy_mean: Energy::zero(),
            sq_kinetic_energy_mean: Energy::zero(),
            potential_energy_mean: Energy::zero(),
            sq_potential_energy_mean: Energy::zero(),
        }
    }

    // TODO: implement
    pub fn measure_kinetic_energy<const D: usize>(&mut self, _vel: &[Velocity<T, D>]) {
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

impl<T> Default for Observer<T>
where
    T: Real,
{
    fn default() -> Self {
        Self::new()
    }
}
