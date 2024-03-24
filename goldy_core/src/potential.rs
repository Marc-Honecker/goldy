use nalgebra::Vector3;

pub trait Potential {
    /// Computes the potential energy.
    fn energy(&mut self, pos: &[Vector3<f64>]) -> Vec<f64>;
    /// Computes -dU/dr.
    fn force(&mut self, pos: &[Vector3<f64>]) -> Vec<Vector3<f64>>;
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
    fn force(&mut self, pos: &[Vector3<f64>]) -> Vec<Vector3<f64>> {
        pos.iter().map(|&pos| -self.k * pos).collect()
    }

    fn energy(&mut self, pos: &[Vector3<f64>]) -> Vec<f64> {
        pos.iter()
            .map(|&pos| 0.5 * self.k * pos.dot(&pos))
            .collect()
    }
}
