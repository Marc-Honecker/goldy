use crate::{
    units::{Damping, Mass},
    vector::{Position, Velocity},
};

#[allow(unused)]
pub struct AtomStore<const D: usize> {
    pos: Vec<Position<D>>,
    vel: Vec<Velocity<D>>,
    id: Vec<usize>,
    mass: Mass,
    damping: Damping,
}

impl<const D: usize> AtomStore<D> {
    pub fn get_positions(&self) -> &Vec<Position<D>> {
        &self.pos
    }

    pub fn get_mut_positions(&mut self) -> &mut Vec<Position<D>> {
        &mut self.pos
    }

    pub fn get_velocities(&self) -> &Vec<Velocity<D>> {
        &self.vel
    }

    pub fn get_mut_velocities(&mut self) -> &mut Vec<Velocity<D>> {
        &mut self.vel
    }
}
