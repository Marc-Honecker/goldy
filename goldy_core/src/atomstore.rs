use crate::{
    units::{Damping, Mass},
    vector::{Position, Velocity},
    Real,
};

#[allow(unused)]
pub struct AtomStore<T, const D: usize>
where
    T: Real,
{
    pos: Vec<Position<T, D>>,
    vel: Vec<Velocity<T, D>>,
    id: Vec<usize>,
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
}
