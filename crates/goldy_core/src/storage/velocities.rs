use nalgebra::SVector;

use crate::{storage::vector::Velocities, Real};

impl<T: Real, const D: usize> Velocities<T, D> {
    pub fn zeros(n: usize) -> Self {
        Self {
            data: (0..n).map(|_| SVector::zeros()).collect(),
        }
    }
}
