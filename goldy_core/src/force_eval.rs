use crate::{
    potential::Potential,
    thermostat::ForceDrivenThermostat,
    vector::{Force, Position, Velocity},
    Result,
};

pub struct ForceEval<const D: usize> {
    thermostat: Option<Box<dyn ForceDrivenThermostat<D>>>,
    potential: Option<Box<dyn Potential<D>>>,
}

impl<const D: usize> ForceEval<D> {
    pub fn new<T, P>(thermostat: Option<T>, potential: Option<P>) -> Result<Self>
    where
        T: ForceDrivenThermostat<D> + 'static,
        P: Potential<D> + 'static,
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
        pos: &[Position<D>],
        vel: &[Velocity<D>],
    ) -> Vec<Force<D>> {
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
