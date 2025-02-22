use nalgebra::SVector;
use rand::{SeedableRng, distr::Uniform};
use rand_chacha::ChaChaRng;
use rand_distr::{Distribution, Normal};

use crate::{Real, storage::vector::Positions};

use super::vector::Iterable;

impl<T, const D: usize> Positions<T, D>
where
    T: Real,
    rand_distr::StandardNormal: Distribution<T>,
{
    pub fn new_gaussian(n: usize, mean: T, std_dev: T) -> Self {
        let mut rng = ChaChaRng::from_os_rng();
        let gauss = Normal::new(mean, std_dev).unwrap();

        let x = (0..n)
            .map(|_| SVector::from_iterator(gauss.sample_iter(&mut rng)))
            .collect();

        Self { data: x }
    }
}

impl<T, const D: usize> Positions<T, D>
where
    T: Real + rand_distr::uniform::SampleUniform + rand::distr::uniform::SampleUniform,
{
    pub fn new_uniform(lengths: &SVector<T, D>, n: usize) -> Self {
        use rand::distr::Distribution;

        // initializing the RNG
        let mut rng = ChaChaRng::from_os_rng();
        // initializing the uniform samplers
        let samplers: Vec<_> = lengths
            .iter()
            .map(|&x| {
                Uniform::new(
                    num_traits::Float::min(T::zero(), x),
                    num_traits::Float::max(T::zero(), x),
                )
                .unwrap()
            })
            .collect();

        // creating the positions
        let mut x = Self::zeros(n);

        x.iter_mut().for_each(|x| {
            x.iter_mut()
                .zip(&samplers)
                .for_each(|(direction, sampler)| {
                    *direction = sampler.sample(&mut rng);
                })
        });

        x
    }
}
