use derive_builder::Builder;

use crate::force_update::ForceUpdate;
use crate::propagator::Propagator;
use crate::simulation_box::SimulationBox;
use crate::storage::atom_store::AtomStore;
use crate::Real;

pub struct Simulation<T: Real, const D: usize, P: Propagator> {
    atoms: AtomStore<T, D>,
    simulation_box: SimulationBox<T, D>,
    force_updater: ForceUpdate<T, D>,
    propagator: P,
    simulation_parameters: SimulationParameters<T>,
}

#[derive(Builder)]
#[builder(build_fn(validate = "Self::validate"))]
/// Holds all global simulation related parameters.
pub struct SimulationParameters<T: Real> {
    #[builder(default = "T::from(0.005).unwrap()")]
    time_step: T,
    #[builder(setter(strip_option), default)]
    temperature: Option<T>,
}

impl<T: Real> SimulationParameters<T> {
    pub fn time_step(&self) -> T {
        self.time_step
    }
    pub fn temperature(&self) -> Option<T> {
        self.temperature
    }
}

impl<T: Real> SimulationParametersBuilder<T> {
    fn validate(&self) -> Result<(), String> {
        if let Some(ref time_step) = self.time_step {
            if !time_step.is_positive() {
                // a negative time step doesn't make any sense
                return Err("Please provide a time step greater than zero.".into());
            }
        }

        if let Some(Some(temp)) = self.temperature {
            if temp.is_negative() {
                return Err("The unit of temperature is Kelvin, so please provide temperatures greater or equal to 0.".into());
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    #[test]
    fn test_simulation_parameters_builder() {
        // Let's test the default case
        let default_sim_params = SimulationParametersBuilder::<f64>::default()
            .build()
            .unwrap();

        assert_eq!(default_sim_params.temperature(), None);
        assert_approx_eq!(default_sim_params.time_step(), 0.005);

        // testing the easy case
        let sim_params = SimulationParametersBuilder::default()
            .temperature(100.0)
            .time_step(0.01)
            .build()
            .unwrap();

        assert_approx_eq!(sim_params.temperature().unwrap(), 100f64);
        assert_approx_eq!(sim_params.time_step(), 0.01);

        // And now with a wrong time step.
        let wrong_time_step = SimulationParametersBuilder::default()
            .time_step(-0.01)
            .build();

        assert!(wrong_time_step
            .is_err_and(|err| err.to_string() == "Please provide a time step greater than zero."));

        // And now the same for the temperature.
        let wrong_temperature = SimulationParametersBuilder::default()
            .temperature(-1.0)
            .build();

        assert!(wrong_temperature.is_err_and(|err| err.to_string() == "The unit of temperature is Kelvin, so please provide temperatures greater or equal to 0."));
    }
}
