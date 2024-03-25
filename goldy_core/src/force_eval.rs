use crate::{
    potential::Potential,
    thermostat::ImplicitThermostat,
    vector::{Force, Position, Velocity},
    Result,
};

pub struct ForceEval {
    thermostat: Option<Box<dyn ImplicitThermostat>>,
    potential: Option<Box<dyn Potential>>,
}

impl ForceEval {
    pub fn new<T, P>(thermostat: Option<T>, potential: Option<P>) -> Result<Self>
    where
        T: ImplicitThermostat + 'static,
        P: Potential + 'static,
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

    pub fn compute_new_forces(&mut self, pos: &[Position], vel: &[Velocity]) -> Vec<Force> {
        let mut force = match &mut self.potential {
            Some(potential) => potential.force(pos),
            None => vec![Force::default(); pos.len()],
        };

        if let Some(thermostat) = &mut self.thermostat {
            thermostat.thermostat(&mut force, vel);
        };

        force
    }
}
