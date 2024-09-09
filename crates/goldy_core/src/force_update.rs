use crate::storage::atom_type_store::AtomTypeStore;
use crate::storage::vector::Positions;
use crate::{
    potential::Potential, simulation_box::SimulationBox, storage::atom_store::AtomStore,
    thermo::ForceDrivenThermostat, Real,
};

#[derive(Default)]
pub struct ForceUpdate<T: Real, const D: usize> {
    thermostat: Option<Box<dyn ForceDrivenThermostat<T, D>>>,
    potential: Option<Box<dyn Potential<T, D>>>,
}

impl<T: Real, const D: usize> ForceUpdate<T, D> {
    /// Returns an empty `ForceUpdate`
    pub fn new() -> Self {
        Self {
            thermostat: None,
            potential: None,
        }
    }

    pub fn get_potential(&self) -> &Option<Box<dyn Potential<T, D>>> {
        &self.potential
    }

    /// Updates the forces according to the defined `ForceDrivenThermostat` and `Potential`.
    pub fn update_forces(
        &mut self,
        atom_store: &mut AtomStore<T, D>,
        neighbor_list: &Vec<Vec<usize>>,
        sim_box: &SimulationBox<T, D>,
        temp: T,
        dt: T,
    ) {
        // First, the forces need to be set to zero.
        atom_store.f.set_to_zero();

        // Now we can apply the thermostat, if present.
        if let Some(thermostat) = &mut self.thermostat {
            thermostat.thermo(
                &mut atom_store.f,
                &atom_store.v,
                &atom_store.atom_types,
                temp,
                dt,
            );
        }

        // And now the potential.
        if let Some(potential) = &self.potential {
            potential.update_forces(
                &atom_store.x,
                neighbor_list,
                &mut atom_store.f,
                sim_box,
                &atom_store.atom_types,
            );
        }
    }

    pub fn measure_energy(
        &self,
        x: &Positions<T, D>,
        neighbor_list: &Vec<Vec<usize>>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    ) -> Option<T> {
        self.potential
            .as_ref()
            .map(|potential| potential.measure_energy(x, neighbor_list, sim_box, atom_types))
    }
}

#[derive(Default)]
pub struct ForceUpdateBuilder<T, const D: usize>
where
    T: Real,
{
    thermostat: Option<Box<dyn ForceDrivenThermostat<T, D>>>,
    potential: Option<Box<dyn Potential<T, D>>>,
}

impl<T, const D: usize> ForceUpdateBuilder<T, D>
where
    T: Real,
{
    pub fn thermostat(self, thermostat: Box<dyn ForceDrivenThermostat<T, D>>) -> Self {
        let mut new = self;
        new.thermostat = Some(thermostat);

        new
    }

    pub fn potential(self, potential: Box<dyn Potential<T, D>>) -> Self {
        let mut new = self;

        new.potential = Some(potential);

        new
    }

    pub fn build(self) -> ForceUpdate<T, D> {
        ForceUpdate {
            thermostat: self.thermostat,
            potential: self.potential,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        potential::harmonic_oscillator::HarmonicOscillatorBuilder, thermo::langevin::Langevin,
    };

    use super::*;

    #[test]
    fn test_force_update_builder() {
        // Defining the potential and thermostat.
        let potential = HarmonicOscillatorBuilder::<f32>::default()
            .k(1.0)
            .build()
            .unwrap();
        let thermostat = Langevin::<f32>::new();

        // Let's define a "full" updater.
        let updater = ForceUpdateBuilder::<f32, 3>::default()
            .thermostat(Box::new(thermostat))
            .potential(Box::new(potential))
            .build();

        // Since everything was set, both members should be present.
        assert!(updater.thermostat.is_some());
        assert!(updater.potential.is_some());

        // Now we need to test if one is absent.
        let thermostat = Langevin::<f32>::new();

        // Let's define a "full" updater.
        let updater = ForceUpdateBuilder::<f32, 3>::default()
            .thermostat(Box::new(thermostat))
            .build();

        // The thermostat should be present and the potential not.
        assert!(updater.thermostat.is_some());
        assert!(updater.potential.is_none());

        // And now we can build an empty one.
        let updater = ForceUpdateBuilder::<f32, 3>::default().build();

        // Now everything should be missing.
        assert!(updater.thermostat.is_none());
        assert!(updater.potential.is_none());
    }

    #[test]
    fn test_new() {
        let empty_updater = ForceUpdate::<f64, 3>::new();

        assert!(empty_updater.potential.is_none());
        assert!(empty_updater.thermostat.is_none());
    }
}
