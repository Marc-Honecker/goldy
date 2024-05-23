use nalgebra::SVector;

use crate::{Real, storage::vector::Forces};

impl<T: Real, const D: usize> Forces<T, D> {
    /// Sets all forces to zero.
    pub fn set_to_zero(&mut self) {
        self.iter_mut().for_each(|f| *f = SVector::zeros());
    }
}
