use nalgebra::{RealField, Scalar};
use num_traits::Float;

pub mod potential;
pub mod thermostat;
pub mod util;

/// Trait bound for all types.
pub trait Real: RealField + Float + Clone + Scalar {}

impl<T> Real for T where T: RealField + Float + Scalar + Clone {}
