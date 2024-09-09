use nalgebra::RealField;
use num_traits::Float;

use crate::simulation_box::SimulationBox;
use crate::storage::vector::Positions;
use error::GoldyError;

pub mod error;
pub mod force_update;
pub mod observe;
pub mod potential;
pub mod propagator;
pub mod simulation;
pub mod simulation_box;
pub mod storage;
pub mod system;
pub mod thermo;

pub type Result<T> = std::result::Result<T, GoldyError>;

pub trait Real: RealField + Float {}
impl<T> Real for T where T: RealField + Float {}

pub fn compute_neighbor_list<T: Real, const D: usize>(
    x: &Positions<T, D>,
    simulation_box: &SimulationBox<T, D>,
    max_cutoff: T,
) -> Vec<Vec<usize>> {
    let mut neighbor_list = vec![Vec::new(); x.len()];
    let sq_max_cutoff = max_cutoff * max_cutoff;

    for (nl, x1) in neighbor_list.iter_mut().zip(x) {
        for (idx, x2) in x.iter().enumerate() {
            let sq_dist = simulation_box.sq_distance(x1, x2);

            if sq_dist <= sq_max_cutoff && sq_dist > T::zero() {
                nl.push(idx);
            }
        }
    }

    neighbor_list
}
