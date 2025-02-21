use std::vec;

use derive_builder::Builder;
use nalgebra::SVector;

use crate::neighbor_list::NeighborList;
use crate::potential::Potential;
use crate::simulation_box::SimulationBox;
use crate::storage::atom_type_store::AtomTypeStore;
use crate::storage::vector::{Forces, Iterable, Positions};
use crate::{potential::pair_potential::PairPotential, storage::atom_type::AtomType, Real};

#[derive(Builder, Clone)]
#[builder(build_fn(validate = "Self::validate"))]
pub struct PairPotentialCollection<T: Real> {
    #[builder(setter(custom))]
    atom_type_ids: Vec<(usize, usize)>,
    #[builder(setter(custom))]
    pair_potentials: Vec<PairPotential<T>>,
}

// TODO: use new neighbor list implementation
impl<T: Real, const D: usize> Potential<T, D> for PairPotentialCollection<T> {
    fn measure_energy(
        &self,
        x: &Positions<T, D>,
        neighbor_list: &NeighborList<T, D>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    ) -> T {
        // Iterating over all given atom-types, positions and the neighbor-list.
        atom_types
            .iter()
            .zip(x)
            .zip(&neighbor_list.neighbor_lists)
            .fold(
                // Initializing the accumulator to zero.
                T::zero(),
                |acc, ((curr_atom_type, curr_pos), neighbors)| {
                    // computing the energies of the curr_pos/neighbor pairs and updating the energy
                    acc + neighbors.iter().fold(T::zero(), |acc, &neighbor_idx| {
                        // retrieving the neighbor position
                        let neighbor_pos = x.get_by_idx(neighbor_idx);
                        // retrieving the neighbor type
                        let neighbor_type = atom_types.get_by_idx(neighbor_idx);

                        // evaluating the potential energy
                        let curr_energy = self
                            .get_pair_potential(curr_atom_type, neighbor_type)
                            .expect("Please provide a proper amount of `AtomType`s.")
                            .energy(sim_box.sq_distance(curr_pos, neighbor_pos));

                        acc + curr_energy
                    })
                },
            )
    }

    fn update_forces(
        &self,
        x: &Positions<T, D>,
        neighbor_list: &NeighborList<T, D>,
        f: &mut Forces<T, D>,
        sim_box: &SimulationBox<T, D>,
        atom_types: &AtomTypeStore<T>,
    ) {
        let mut pseudo_forces = vec![SVector::zeros(); x.len()];

        for (id, neighbors) in neighbor_list.neighbor_lists.iter().enumerate() {
            let curr_pos = x.get_by_idx(id);
            let curr_type = atom_types.get_by_idx(id);

            for &neighbor_id in neighbors {
                let neighbor_pos = x.get_by_idx(neighbor_id);
                let neighbor_type = atom_types.get_by_idx(neighbor_id);

                // evaluating the pseudo-force
                let pseudo_force = self
                    .get_pair_potential(curr_type, neighbor_type)
                    .expect("Please provide a proper amount of `AtomType`s.")
                    .pseudo_force(sim_box.sq_distance(curr_pos, neighbor_pos));
                let pseudo_force_vector = sim_box.difference(curr_pos, neighbor_pos) * pseudo_force;

                pseudo_forces[id] += pseudo_force_vector;
                pseudo_forces[neighbor_id] -= pseudo_force_vector;
            }
        }

        f.iter_mut()
            .zip(pseudo_forces)
            .for_each(|(f, pseudo_force)| {
                *f -= pseudo_force;
            });
    }
}

impl<T: Real> PairPotentialCollection<T> {
    pub(crate) fn get_pair_potential(
        &self,
        at1: &AtomType<T>,
        at2: &AtomType<T>,
    ) -> Option<&PairPotential<T>> {
        for (idx, &(id1, id2)) in self.atom_type_ids.iter().enumerate() {
            // respects symmetry
            if id1 == at1.id() && id2 == at2.id() || id1 == at2.id() && id2 == at1.id() {
                // We found our `PairPotential`, so we can return it.
                return Some(&self.pair_potentials[idx]);
            }
        }

        None
    }

    pub(crate) fn get_cutoffs(&self) -> Vec<T> {
        self.pair_potentials
            .iter()
            .map(|pp| pp.get_cutoff())
            .collect()
    }

    pub(crate) fn get_cutoff_by_atom_type(
        &self,
        at1: &AtomType<T>,
        at2: &AtomType<T>,
    ) -> Option<T> {
        let pp = self.get_pair_potential(at1, at2);

        pp.map(|pp| pp.get_cutoff())
    }
}

impl<T: Real> PairPotentialCollectionBuilder<T> {
    /// Adds a new `PairPotential` to the collection.
    pub fn add_potential(
        &mut self,
        at1: &AtomType<T>,
        at2: &AtomType<T>,
        pair_potential: PairPotential<T>,
    ) -> &mut Self {
        // Saving the IDs of the `AtomType`s.
        let ids = (at1.id(), at2.id());

        // inserting the new ids into our Vector
        match &mut self.atom_type_ids {
            None => {
                self.atom_type_ids = Some(vec![ids]);
            }
            Some(pp_ids) => {
                pp_ids.push(ids);
            }
        }

        // inserting the new `PairPotential`
        match &mut self.pair_potentials {
            None => {
                self.pair_potentials = Some(vec![pair_potential]);
            }
            Some(pps) => {
                pps.push(pair_potential);
            }
        }

        self
    }

    fn validate(&self) -> Result<(), String> {
        // Ensures, that at least one `PairPotential` was provided and the number
        // of ID-pairs is equal to the number of potentials.
        match &self.atom_type_ids {
            None => Err("Please provide at least one pair of `PairPotential`s.".into()),
            Some(ids) => {
                match &self.pair_potentials {
                    None => {
                        unreachable!()
                    }
                    Some(pps) => {
                        if ids.len() != pps.len() {
                            Err("Please provide the same amount of `PairPotential`s as `AtomType`s.".into())
                        } else {
                            Ok(())
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::potential::pair_potential::PairPotential;
    use crate::potential::pair_potential_collection::PairPotentialCollectionBuilder;
    use crate::storage::atom_type::AtomTypeBuilder;

    #[test]
    fn test_get_pair_potential() {
        // Let's define two valid `AtomType`s.
        let at1 = AtomTypeBuilder::default()
            .id(0)
            .mass(1.0)
            .damping(0.05)
            .build()
            .unwrap();
        let at2 = AtomTypeBuilder::default()
            .id(1)
            .mass(39.95)
            .damping(0.01)
            .build()
            .unwrap();

        // And now the `PairPotentialCollection`.
        let ppc = PairPotentialCollectionBuilder::default()
            .add_potential(
                &at1,
                &at1,
                PairPotential::new_lennard_jones(120.0, 3.5, 7.9),
            )
            .add_potential(&at1, &at2, PairPotential::new_mie(10, 5, 100.0, 2.4, 10.0))
            .add_potential(&at2, &at2, PairPotential::new_morse(14, 7, 150.0, 3.0, 8.6))
            .build()
            .unwrap();

        // easiest test possible
        assert!(ppc.get_pair_potential(&at1, &at1).is_some());
        assert!(ppc.get_pair_potential(&at1, &at2).is_some());
        assert!(ppc.get_pair_potential(&at2, &at2).is_some());
    }
}
