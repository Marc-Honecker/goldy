use crate::{
    units::Energy,
    vector::{Force, Position},
    Float,
};

pub trait Potential<const D: usize> {
    /// Computes the potential energy.
    fn energy(&mut self, pos: &[Position<D>]) -> Vec<Energy>;
    /// Computes -dU/dr.
    fn force(&mut self, pos: &[Position<D>]) -> Vec<Force<D>>;
}

pub struct HarmonicOscillator {
    k: Float,
}

impl HarmonicOscillator {
    pub fn new(k: Float) -> Self {
        Self { k }
    }
}

impl<const D: usize> Potential<D> for HarmonicOscillator {
    fn force(&mut self, pos: &[Position<D>]) -> Vec<Force<D>> {
        pos.iter().map(|&pos| Force::new(-self.k * *pos)).collect()
    }

    fn energy(&mut self, pos: &[Position<D>]) -> Vec<Energy> {
        pos.iter()
            .map(|&pos| Energy::new(0.5 * self.k * pos.dot(&pos)))
            .collect()
    }
}
