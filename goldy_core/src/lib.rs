use error::GoldyError;

pub mod error;
pub mod force_eval;
pub mod observer;
pub mod potential;
pub mod thermostat;
pub mod units;
pub mod vector;

pub type Result<T> = std::result::Result<T, GoldyError>;

pub type Float = f64;
