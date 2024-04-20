use error::GoldyError;
use nalgebra::RealField;
use num_traits::Float;

pub mod error;

pub type Result<T> = std::result::Result<T, GoldyError>;

pub trait Real: RealField + Float {}
impl<T> Real for T where T: RealField + Float {}
