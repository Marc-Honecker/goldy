use goldy_core::Real;

use crate::Propagator;

#[derive(Debug, Clone, Copy)]
pub struct Euler;

impl Propagator for Euler {
    fn integrate<T, const D: usize, Pot, FDT>(
        atom_store: &mut goldy_storage::atom_store::AtomStore<T, D>,
        sim_box: &goldy_box::SimulationBox<T, D>,
        potential: Option<&Pot>,
        thermostat: Option<&mut FDT>,
        dt: T,
        temp: T,
    ) -> Option<T>
    where
        T: Real,
        Pot: goldy_potential::Potential<T, D>,
        FDT: goldy_thermo::ForceDrivenThermostat<T, D>,
    {
        // initalizing the forces
        atom_store.f.set_to_zero();

        // evaluating the potential if present
        let potential_energy = match potential {
            Some(pot) => Some(pot.eval(
                &atom_store.x,
                &mut atom_store.f,
                sim_box,
                &atom_store.atom_types,
            )),
            None => None,
        };

        // evaluating the thermostat if present.
        if let Some(thermostat) = thermostat {
            thermostat.thermo(
                &mut atom_store.f,
                &atom_store.v,
                &atom_store.atom_types,
                temp,
                dt,
            );
        }

        // The force computation is completed and we can update the rest.
        // Updating the velocities.
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|((v, &f), &t)| {
                *v += f / t.mass() * dt;
            });

        // Updating the positions.
        atom_store
            .x
            .iter_mut()
            .zip(&atom_store.v)
            .for_each(|(x, &v)| *x += v * dt);

        potential_energy
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use goldy_box::SimulationBoxBuilder;
    use goldy_potential::harmonic_oscillator::HarmonicOscillator;
    use goldy_storage::{
        atom_store::{AtomStore, AtomStoreBuilder},
        atom_type::AtomTypeBuilder,
        atom_type_store::AtomTypeStoreBuilder,
        vector::{Forces, Positions, Velocities},
    };
    use goldy_thermo::langevin::Langevin;
    use nalgebra::Matrix3;

    use super::*;

    #[test]
    fn test_euler_without_everything() {
        // defining our big AtomStore.
        let num_atoms = 1;
        let mut atom_store: AtomStore<f32, 3> = AtomStoreBuilder::default()
            .positions(Positions::zeros(num_atoms))
            .velocities(Velocities::zeros(num_atoms))
            .forces(Forces::zeros(num_atoms))
            .atom_types(
                AtomTypeStoreBuilder::default()
                    .add(
                        AtomTypeBuilder::default()
                            .mass(39.95)
                            .damping(0.01)
                            .build()
                            .unwrap(),
                    )
                    .build(),
            )
            .build()
            .unwrap();

        // doesn't do anything, but we need it
        let sim_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal_element(10.0))
            .boundary_type(goldy_box::BoundaryTypes::Open)
            .build()
            .unwrap();

        // propating one step
        let potential_energy = Euler::integrate::<f32, 3, HarmonicOscillator<f32>, Langevin<f32>>(
            &mut atom_store,
            &sim_box,
            None,
            None,
            0.01,
            1.0,
        );

        // potential_energy must be None, because we didn't provide any Potential nor Thermostat.
        assert_eq!(potential_energy, None);

        // testing, if everything stayed zero
        atom_store
            .x
            .iter()
            .for_each(|x| x.iter().for_each(|x| assert_approx_eq!(x, 0.0)));
        atom_store
            .v
            .iter()
            .for_each(|v| v.iter().for_each(|x| assert_approx_eq!(x, 0.0)));
        atom_store
            .f
            .iter()
            .for_each(|f| f.iter().for_each(|x| assert_approx_eq!(x, 0.0)));
    }
}
