use std::{fmt::Display, ops::Deref};

use crate::{error::GoldyError, Real, Result};

#[derive(Debug, Clone, Copy)]
pub struct Mass<T>
where
    T: Real,
{
    value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Energy<T>
where
    T: Real,
{
    value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct TimeStep<T>
where
    T: Real,
{
    value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Temperature<T>
where
    T: Real,
{
    value: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Damping<T>
where
    T: Real,
{
    value: T,
}

macro_rules! generate_guarded_new {
    ($struct_type: ident) => {
        impl<T> $struct_type<T>
        where
            T: Real,
        {
            pub fn new(value: T) -> Result<Self> {
                if value <= T::zero() {
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

impl<T> Damping<T>
where
    T: Real,
{
    // Damping must be divided by TimeStep, so we need an extra implementation.
    pub fn new(value: T, dt: &TimeStep<T>) -> Result<Self> {
        if value <= T::zero() {
            Err(GoldyError::NegativValue {
                found: value.to_string(),
            })
        } else {
            Ok(Self {
                value: value / **dt,
            })
        }
    }

    pub fn new_raw(value: T) -> Result<Self> {
        if value <= T::zero() {
            Err(GoldyError::NegativValue {
                found: value.to_string(),
            })
        } else {
            Ok(Self { value })
        }
    }
}

impl<T> Energy<T>
where
    T: Real,
{
    // Energy can be positive and negative, so we can simply return it.
    pub fn new(value: T) -> Self {
        Self { value }
    }

    pub fn zero() -> Self {
        Self { value: T::zero() }
    }
}

macro_rules! generate_deref {
    ($struct_type: ident) => {
        impl<T> Deref for $struct_type<T>
        where
            T: Real,
        {
            type Target = T;

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
        impl<T> AsRef<T> for $struct_type<T>
        where
            T: Real,
        {
            #[inline]
            fn as_ref(&self) -> &T {
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
        impl<T> Display for $struct_type<T>
        where
            T: Real,
        {
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
