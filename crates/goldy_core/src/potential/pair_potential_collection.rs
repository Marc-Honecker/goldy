use crate::potential::pair_potential::PairPotential;
use crate::Real;

pub struct PairPotentialCollection<T: Real> {
    atom_type_ids: Vec<usize>,
    pair_potentials: Vec<PairPotential<T>>,
}
