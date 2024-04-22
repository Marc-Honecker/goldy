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
