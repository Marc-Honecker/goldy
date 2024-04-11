use crate::{
    units::{Damping, Mass},
    vector::{Force, Position, Velocity},
    Real,
};

pub struct AtomStore<T, const D: usize>
where
    T: Real,
{
    pos: Vec<Position<T, D>>,
    vel: Vec<Velocity<T, D>>,
    force: Vec<Force<T, D>>,
    // id: Vec<usize>,
    mass: Mass<T>,
    damping: Damping<T>,
}

impl<T, const D: usize> AtomStore<T, D>
where
    T: Real,
{
    pub fn get_positions(&self) -> &Vec<Position<T, D>> {
        &self.pos
    }

    pub fn get_mut_positions(&mut self) -> &mut Vec<Position<T, D>> {
        &mut self.pos
    }

    pub fn get_velocities(&self) -> &Vec<Velocity<T, D>> {
        &self.vel
    }

    pub fn get_mut_velocities(&mut self) -> &mut Vec<Velocity<T, D>> {
        &mut self.vel
    }

    pub fn get_forces(&self) -> &Vec<Force<T, D>> {
        &self.force
    }

    pub fn get_mut_forces(&mut self) -> &mut Vec<Force<T, D>> {
        &mut self.force
    }

    pub fn get_mass(&self) -> &Mass<T> {
        &self.mass
    }

    pub fn get_damping(&self) -> &Damping<T> {
        &self.damping
    }
}
