use crate::{
    potential::Potential, propagator::Propagator, simulation_box::SimulationBox,
    storage::atom_store::AtomStore, thermo::ForceDrivenThermostat, Real,
};

pub struct VelocityVerlet;

impl Propagator for VelocityVerlet {
    fn integrate<T, const D: usize, Pot, FDT>(
        atom_store: &mut AtomStore<T, D>,
        sim_box: &SimulationBox<T, D>,
        potential: Option<&Pot>,
        thermostat: Option<&mut FDT>,
        dt: T,
        temp: T,
    ) -> Option<T>
    where
        T: Real,
        Pot: Potential<T, D>,
        FDT: ForceDrivenThermostat<T, D>,
    {
        // First, we need to update the velocities half a timestep.
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|((v, &f), &t)| *v += f / t.mass() * T::from(0.5).unwrap() * dt);

        // Now we can update the positions.
        atom_store
            .x
            .iter_mut()
            .zip(&atom_store.v)
            .for_each(|(x, &v)| *x += v * dt);

        // Now we need to compute the new forces.
        atom_store.f.set_to_zero();

        let potential_energy = match potential {
            Some(pot) => Some(pot.eval(
                &atom_store.x,
                &mut atom_store.f,
                sim_box,
                &atom_store.atom_types,
            )),
            None => None,
        };

        if let Some(thermo) = thermostat {
            thermo.thermo(
                &mut atom_store.f,
                &atom_store.v,
                &atom_store.atom_types,
                temp,
                dt,
            );
        }

        // And now we can update the velocities the last half timestep.
        atom_store
            .v
            .iter_mut()
            .zip(&atom_store.f)
            .zip(&atom_store.atom_types)
            .for_each(|((v, &f), t)| *v += f / t.mass() * T::from(0.5).unwrap() * dt);

        potential_energy
    }
}
