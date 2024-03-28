use error::GoldyError;
use nalgebra::RealField;
use num_traits::Float;

pub mod atomstore;
pub mod error;
pub mod force_eval;
pub mod observer;
pub mod potential;
pub mod thermostat;
pub mod units;
pub mod vector;

pub type Result<T> = std::result::Result<T, GoldyError>;

pub trait Real: RealField + Float {}
impl<T> Real for T where T: RealField + Float {}
