use std::{fmt::Display, ops::Deref};

use crate::{error::GoldyError, Result};

#[derive(Debug, Clone, Copy)]
pub struct Mass {
    value: f64,
}

#[derive(Debug, Clone, Copy, Default)]
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
                if value <= 0.0 {
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

impl Damping {
    // Damping must be divided by TimeStep, so we need an extra implementation.
    pub fn new(value: f64, dt: &TimeStep) -> Result<Self> {
        if value <= 0.0 {
            Err(GoldyError::NegativValue {
                found: value.to_string(),
            })
        } else {
            Ok(Self {
                value: value / **dt,
            })
        }
    }

    pub fn new_raw(value: f64) -> Result<Self> {
        if value <= 0.0 {
            Err(GoldyError::NegativValue {
                found: value.to_string(),
            })
        } else {
            Ok(Self { value })
        }
    }
}

impl Energy {
    // Energy can be positive and negative, so we can simply return it.
    pub fn new(value: f64) -> Self {
        Self { value }
    }
}

macro_rules! generate_deref {
    ($struct_type: ident) => {
        impl Deref for $struct_type {
            type Target = f64;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }
    };
}

generate_deref!(Mass);
generate_deref!(TimeStep);
generate_deref!(Energy);
generate_deref!(Damping);
generate_deref!(Temperature);

macro_rules! generate_as_ref {
    ($struct_type: ident) => {
        impl AsRef<f64> for $struct_type {
            #[inline]
            fn as_ref(&self) -> &f64 {
                &self.value
            }
        }
    };
}

generate_as_ref!(Mass);
generate_as_ref!(TimeStep);
generate_as_ref!(Energy);
generate_as_ref!(Damping);
generate_as_ref!(Temperature);

macro_rules! generate_display {
    ($struct_type: ident ) => {
        impl Display for $struct_type {
            #[inline]
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.value, f)
            }
        }
    };
}

generate_display!(Mass);
generate_display!(TimeStep);
generate_display!(Energy);
generate_display!(Damping);
generate_display!(Temperature);
