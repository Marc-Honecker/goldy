use crate::{
    units::Energy,
    vector::{Force, Position},
};

pub trait Potential {
    /// Computes the potential energy.
    fn energy(&mut self, pos: &[Position]) -> Vec<Energy>;
    /// Computes -dU/dr.
    fn force(&mut self, pos: &[Position]) -> Vec<Force>;
}

pub struct HarmonicOscillator {
    k: f64,
}

impl HarmonicOscillator {
    pub fn new(k: f64) -> Self {
        Self { k }
    }
}

impl Potential for HarmonicOscillator {
    fn force(&mut self, pos: &[Position]) -> Vec<Force> {
        pos.iter().map(|&pos| Force::new(-self.k * *pos)).collect()
    }

    fn energy(&mut self, pos: &[Position]) -> Vec<Energy> {
        pos.iter()
            .map(|&pos| Energy::new(0.5 * self.k * pos.dot(&pos)))
            .collect()
    }
}
