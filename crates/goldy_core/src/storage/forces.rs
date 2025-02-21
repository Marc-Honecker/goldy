use nalgebra::SVector;

use crate::{storage::vector::Forces, Real};

use super::vector::Iterable;

impl<T: Real, const D: usize> Forces<T, D> {
    /// Sets all forces to zero.
    pub fn set_to_zero(&mut self) {
        self.iter_mut().for_each(|f| *f = SVector::zeros());
    }
}
