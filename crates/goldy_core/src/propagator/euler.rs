use crate::{force_update::ForceUpdate, propagator::Propagator, Real};

#[derive(Debug, Clone, Copy)]
pub struct Euler;

impl Propagator for Euler {
    fn integrate<T: Real, const D: usize>(
        atom_store: &mut crate::storage::atom_store::AtomStore<T, D>,
        sim_box: &crate::simulation_box::SimulationBox<T, D>,
        updater: &mut ForceUpdate<T, D>,
        dt: T,
        temp: T,
    ) -> Option<T> {
        // updating the forces
        let potential_energy = updater.update_forces(atom_store, sim_box, temp, dt);

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
    use crate::{
        simulation_box::{BoundaryTypes, SimulationBoxBuilder},
        storage::{
            atom_store::{AtomStore, AtomStoreBuilder},
            atom_type::AtomTypeBuilder,
            atom_type_store::AtomTypeStoreBuilder,
            vector::{Forces, Positions, Velocities},
        },
    };

    use assert_approx_eq::assert_approx_eq;
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
            .boundary_type(BoundaryTypes::Open)
            .build()
            .unwrap();

        // propating one step
        let potential_energy = Euler::integrate::<f32, 3>(
            &mut atom_store,
            &sim_box,
            &mut ForceUpdate::new(),
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
