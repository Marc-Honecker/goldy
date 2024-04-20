use derive_builder::Builder;
use goldy_box::SimulationBox;
use goldy_core::Real;
use goldy_storage::{atom_type_store::AtomTypeStore, vector::Forces};

use crate::Potential;

#[derive(Debug, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct HarmonicOscillator<T: Real> {
    k: T,
}

impl<T: Real, const D: usize> Potential<T, D> for HarmonicOscillator<T> {
    fn eval(
        &self,
        x: &goldy_storage::vector::Positions<T, D>,
        f: &mut Forces<T, D>,
        _: &SimulationBox<T, D>,
        _: &AtomTypeStore<T>,
    ) -> T {
        // accumulator for the potential energy
        let mut pot_energy = T::zero();

        // main loop
        f.iter_mut().zip(x).for_each(|(f, &x)| {
            // f = -kx
            *f = -x * self.k;
            // u = 1/2 * k * x^2
            // safety: T::from(0.5) is guaranteed to exist by design
            pot_energy += T::from(0.5).unwrap() * self.k * x.dot(&x);
        });

        pot_energy
    }
}

impl<T: Real> HarmonicOscillatorBuilder<T> {
    fn validate(&self) -> Result<(), String> {
        if let Some(k) = self.k {
            if k < T::zero() {
                Err("Please provide a spring constant greater than zero.".into())
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}
