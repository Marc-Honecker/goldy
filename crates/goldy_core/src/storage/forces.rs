use nalgebra::SVector;

use crate::{Real, storage::vector::Forces};

impl<T: Real, const D: usize> Forces<T, D> {
    pub fn zeros(n: usize) -> Self {
        Self {
            data: (0..n).map(|_| SVector::zeros()).collect(),
        }
    }

    /// Sets all forces to zero.
    pub fn set_to_zero(&mut self) {
        self.iter_mut().for_each(|f| *f = SVector::zeros());
    }
}
