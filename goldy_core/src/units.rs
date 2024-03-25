#![allow(unused)]

use crate::{error::GoldyError, Result};

#[derive(Debug, Clone, Copy)]
pub struct Mass {
    value: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Energy {
    value: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeStep {
    value: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Temperature {
    value: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Damping {
    value: f64,
}

macro_rules! generate_guarded_new {
    ($struct_type: ident) => {
        impl $struct_type {
            pub fn new(value: f64) -> Result<Self> {
                if value < 0.0 {
                    Err(GoldyError::NegativValue {
                        found: value.to_string(),
                    })
                } else {
                    Ok(Self { value })
                }
            }
        }
    };
}

generate_guarded_new!(Mass);
generate_guarded_new!(TimeStep);
generate_guarded_new!(Temperature);
generate_guarded_new!(Damping);
