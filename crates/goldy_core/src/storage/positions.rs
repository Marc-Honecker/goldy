use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, Normal};

use crate::{storage::vector::Positions, Real};

impl<T, const D: usize> Positions<T, D>
where
    T: Real,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
{
    pub fn new_gaussian(n: usize, mean: T, std_dev: T) -> Self {
        let mut rng = ChaChaRng::from_entropy();
        let gauss = Normal::new(mean, std_dev).unwrap();

        let x = (0..n)
            .map(|_| SVector::from_iterator(gauss.sample_iter(&mut rng)))
            .collect();

        Self { data: x }
    }
}

impl<T: Real, const D: usize> Positions<T, D> {
    pub fn zeros(n: usize) -> Self {
        Self {
            data: (0..n).map(|_| SVector::zeros()).collect(),
        }
    }
}
