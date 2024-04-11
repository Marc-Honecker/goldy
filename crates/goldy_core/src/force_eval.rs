use crate::{
    potential::Potential,
    thermostat::ForceDrivenThermostat,
    vector::{Force, Position, Velocity},
    Real, Result,
};

pub struct ForceEval<T, const D: usize>
where
    T: Real,
{
    thermostat: Option<Box<dyn ForceDrivenThermostat<T, D>>>,
    potential: Option<Box<dyn Potential<T, D>>>,
}

impl<T, const D: usize> ForceEval<T, D>
where
    T: Real,
{
    pub fn new<F, P>(thermostat: Option<F>, potential: Option<P>) -> Result<Self>
    where
        F: ForceDrivenThermostat<T, D> + 'static,
        P: Potential<T, D> + 'static,
    {
        match thermostat {
            Some(t) => {
                if let Some(p) = potential {
                    Ok(Self {
                        thermostat: Some(Box::new(t)),
                        potential: Some(Box::new(p)),
                    })
                } else {
                    Ok(Self {
                        thermostat: Some(Box::new(t)),
                        potential: None,
                    })
                }
            }

            None => {
                if let Some(p) = potential {
                    Ok(Self {
                        thermostat: None,
                        potential: Some(Box::new(p)),
                    })
                } else {
                    Err(crate::error::GoldyError::CatchAll {
                        msg: "You need to provide at least one potential or thermostat."
                            .to_string(),
                    })
                }
            }
        }
    }

    pub fn compute_new_forces(
        &mut self,
        pos: &[Position<T, D>],
        vel: &[Velocity<T, D>],
    ) -> Vec<Force<T, D>> {
        let mut force = match &mut self.potential {
            Some(potential) => potential.force(pos),
            None => vec![Force::zeros(); pos.len()],
        };

        if let Some(thermostat) = &mut self.thermostat {
            thermostat.thermostat(&mut force, vel);
        };

        force
    }
}
