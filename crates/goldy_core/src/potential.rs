use nalgebra::RealField;

use crate::{
    units::Energy,
    vector::{Force, Position},
    Real,
};

pub trait Potential<T, const D: usize>
where
    T: Real,
{
    /// Computes the potential energy.
    fn energy(&mut self, pos: &[Position<T, D>]) -> Vec<Energy<T>>;
    /// Computes -dU/dr.
    fn force(&mut self, pos: &[Position<T, D>]) -> Vec<Force<T, D>>;
}

pub struct HarmonicOscillator<T>
where
    T: RealField,
{
    k: T,
}

impl<T> HarmonicOscillator<T>
where
    T: RealField,
{
    pub fn new(k: T) -> Self {
        Self { k }
    }
}

impl<T, const D: usize> Potential<T, D> for HarmonicOscillator<T>
where
    T: Real,
{
    fn force(&mut self, pos: &[Position<T, D>]) -> Vec<Force<T, D>> {
        pos.iter().map(|pos| Force::new(**pos * -self.k)).collect()
    }

    fn energy(&mut self, pos: &[Position<T, D>]) -> Vec<Energy<T>> {
        pos.iter()
            .map(|pos| Energy::new(T::from_f64(0.5).unwrap() * self.k * pos.dot(pos)))
            .collect()
    }
}
