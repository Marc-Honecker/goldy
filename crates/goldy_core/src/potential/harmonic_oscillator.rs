use derive_builder::Builder;

use crate::neighbor_list::NeighborList;
use crate::storage::vector::Iterable;
use crate::{
    potential::Potential,
    simulation_box::SimulationBox,
    storage::{
        atom_type_store::AtomTypeStore,
        vector::{Forces, Positions},
    },
    Real,
};

#[derive(Debug, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct HarmonicOscillator<T: Real> {
    k: T,
}

impl<T: Real, const D: usize> Potential<T, D> for HarmonicOscillator<T> {
    fn measure_energy(
        &mut self,
        x: &Positions<T, D>,
        _: &NeighborList<T, D>,
        _: &SimulationBox<T, D>,
        _: &AtomTypeStore<T>,
    ) -> T {
        x.iter().fold(T::zero(), |e, x| {
            e + T::from(0.5).unwrap() * self.k * x.dot(x)
        })
    }

    fn update_forces(
        &mut self,
        x: &Positions<T, D>,
        _: &NeighborList<T, D>,
        f: &mut Forces<T, D>,
        _: &SimulationBox<T, D>,
        _: &AtomTypeStore<T>,
    ) {
        f.iter_mut().zip(x).for_each(|(f, x)| *f -= x * self.k)
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

    use super::*;
    use crate::{
        simulation_box::{BoundaryTypes, SimulationBoxBuilder},
        storage::{
            atom_type::AtomTypeBuilder, atom_type_store::AtomTypeStoreBuilder, vector::Velocities,
        },
    };

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
                    .id(0)
                    .mass(39.95)
                    .damping(0.01)
                    .build()
                    .unwrap(),
            )
            .build();

        // defining the potential
        let mut potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

        // the potential energy
        let mut pot_energy = 0.0;

        // the md parameters
        let runs = 1000;
        let dt = 2.0 * PI / 40.0;

        for _ in 0..runs {
            // initializing the forces
            let mut f = Forces::<f32, 3>::zeros(1);
            // computing the Forces
            potential.update_forces(
                &x,
                &NeighborList::new_empty(),
                &mut f,
                &sim_box,
                &atom_types,
            );
            pot_energy +=
                potential.measure_energy(&x, &NeighborList::new_empty(), &sim_box, &atom_types);
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
        let num_atoms = 10_000;

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
            .add_many(
                AtomTypeBuilder::default()
                    .id(0)
                    .mass(1.0)
                    .damping(0.01)
                    .build()
                    .unwrap(),
                num_atoms,
            )
            .build();

        // defining the potential
        let mut potential = HarmonicOscillatorBuilder::default().k(1.0).build().unwrap();

        // the potential energy
        let mut pot_energy = 0.0;

        // the md parameters
        let runs = 1_000;
        let dt = 2.0 * PI / 40.0;

        for _ in 0..runs {
            // initializing the forces
            let mut f = Forces::<f32, 3>::zeros(num_atoms);
            // computing the Forces
            potential.update_forces(
                &x,
                &NeighborList::new_empty(),
                &mut f,
                &sim_box,
                &atom_types,
            );
            pot_energy +=
                potential.measure_energy(&x, &NeighborList::new_empty(), &sim_box, &atom_types);
            // stepping forward in time
            f.iter_mut()
                .zip(&atom_types)
                .for_each(|(f, at)| *f /= at.mass());
            v.iter_mut().zip(&f).for_each(|(v, &f)| *v += f * dt);
            x.iter_mut().zip(&v).for_each(|(x, &v)| *x += v * dt);
        }

        assert_approx_eq!(pot_energy / (runs * num_atoms) as f32, 0.75, 5e-2);
    }
}
