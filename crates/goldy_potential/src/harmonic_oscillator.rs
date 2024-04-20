use derive_builder::Builder;
use goldy_box::SimulationBox;
use goldy_core::Real;
use goldy_storage::{
    atom_type_store::AtomTypeStore,
    vector::{Forces, Positions},
};

use crate::Potential;

#[derive(Debug, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct HarmonicOscillator<T: Real> {
    k: T,
}

impl<T: Real, const D: usize> Potential<T, D> for HarmonicOscillator<T> {
    fn eval(
        &self,
        x: &Positions<T, D>,
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

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Matrix3;

    use goldy_box::{BoundaryTypes, SimulationBoxBuilder};
    use goldy_storage::{
        atom_type::AtomTypeBuilder, atom_type_store::AtomTypeStoreBuilder, vector::Velocities,
    };

    use super::*;

    #[test]
    fn test_harmonic_oscillator_with_one_atom() {
        // let's spawn one at [0,0,0]
        let mut x = Positions::<f32, 3>::zeros(1);
        let mut v = Velocities::<f32, 3>::zeros(1);
        // The SimulationBox and the atom-types don't matter here, but
        // we still need to define them.
        let sim_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Open)
            .build()
            .unwrap();
        let atom_types = AtomTypeStoreBuilder::default()
            .add(
                AtomTypeBuilder::default()
                    .mass(39.95)
                    .gamma(0.01)
                    .build()
                    .unwrap(),
            )
            .build();

        // defining the potential
        let potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

        // the potential energy
        let mut pot_energy = 0.0;

        // the md parameters
        let runs = 1000;
        let dt = 2.0 * PI / 40.0;

        for _ in 0..runs {
            // initializing the forces
            let mut f = Forces::<f32, 3>::zeros(1);
            // computing the Forces
            pot_energy += potential.eval(&x, &mut f, &sim_box, &atom_types);
            // stepping forward in time
            f.iter_mut()
                .zip(&atom_types)
                .for_each(|(f, at)| *f /= at.mass());
            v.iter_mut().zip(&f).for_each(|(v, &f)| *v += f * dt);
            x.iter_mut().zip(&v).for_each(|(x, &v)| *x += v * dt);
        }

        // Since everything was zero, the potential energy must be zero.
        assert_approx_eq!(pot_energy / runs as f32, 0.0);
    }

    #[test]
    fn test_harmonic_oscillator() {
        // This time, we spawn many atoms.
        let num_atoms = 100_000;

        // Let's spawn some atoms at random positions.
        let mut x = Positions::<f32, 3>::new_gaussian(num_atoms, 0.0, 1.0);
        let mut v = Velocities::<f32, 3>::zeros(num_atoms);
        // The SimulationBox and the atom-types don't matter here, but
        // we still need to define them.
        let sim_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Open)
            .build()
            .unwrap();
        let atom_types = AtomTypeStoreBuilder::default()
            .add(
                AtomTypeBuilder::default()
                    .mass(1.0)
                    .gamma(0.01)
                    .build()
                    .unwrap(),
            )
            .build();

        // defining the potential
        let potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

        // the potential energy
        let mut pot_energy = 0.0;

        // the md parameters
        let runs = 1_000;
        let dt = 2.0 * PI / 40.0;

        for _ in 0..runs {
            // initializing the forces
            let mut f = Forces::<f32, 3>::zeros(1);
            // computing the Forces
            pot_energy += potential.eval(&x, &mut f, &sim_box, &atom_types);
            // stepping forward in time
            f.iter_mut()
                .zip(&atom_types)
                .for_each(|(f, at)| *f /= at.mass());
            v.iter_mut().zip(&f).for_each(|(v, &f)| *v += f * dt);
            x.iter_mut().zip(&v).for_each(|(x, &v)| *x += v * dt);
        }

        // Since the atoms were zero centered, the first moment of the potential energy must be zero.
        assert_approx_eq!(pot_energy / (runs * num_atoms) as f32, 0.0, 1e-4);
    }
}
