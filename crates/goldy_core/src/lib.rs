use nalgebra::RealField;
use num_traits::Float;

use error::GoldyError;

pub mod observer;
pub mod error;
pub mod force_update;
pub mod gronbech_jensen;
pub mod neighbor_list;
pub mod potential;
pub mod propagator;
pub mod rdf;
pub mod simulation;
pub mod simulation_box;
pub mod storage;
pub mod system;
pub mod thermo;

pub type Result<T> = std::result::Result<T, GoldyError>;

pub trait Real: RealField + Float {}
impl<T> Real for T where T: RealField + Float {}
