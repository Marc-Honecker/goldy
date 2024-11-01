use crate::neighbor_list::NeighborList;
use crate::potential::pair_potential_collection::PairPotentialCollection;
use crate::simulation_box::SimulationBox;
use crate::storage::atom_store::AtomStore;
use crate::storage::atom_type::AtomType;
use crate::Real;
use rand::distributions::uniform::SampleUniform;

/// Implements the radial distribution function.
pub struct RDF<T: Real> {
    histogram: Vec<Vec<T>>,
    counter: Vec<Vec<T>>,
    num_random_atoms: usize,
    num_intervals: usize,
    pair_ids: Vec<(usize, usize)>,
    cutoffs: Vec<T>,
    r_max: T,
    pre_fac: T,
    // rng: ChaChaRng,
}

impl<T> RDF<T>
where
    T: Real + SampleUniform,
{
    pub fn new(
        num_random_atoms: usize,
        num_intervals: usize,
        r_max: T,
        pair_potential_collection: &PairPotentialCollection<T>,
    ) -> Self {
        let pair_ids = pair_potential_collection.get_id_pairs();
        let cutoffs = pair_potential_collection.get_cutoffs();

        assert!(
            cutoffs.iter().all(|&cutoff| cutoff <= r_max),
            "Please choose r_max large enough."
        );

        let histogram = vec![vec![T::zero(); num_intervals]; pair_ids.len()];
        let counter = vec![vec![T::zero(); num_intervals]; pair_ids.len()];

        Self {
            histogram,
            counter,
            num_random_atoms,
            num_intervals,
            pair_ids,
            cutoffs,
            r_max,
            pre_fac: T::from(4.0).unwrap() / T::from(3.0).unwrap() * T::pi(),
            // rng: ChaChaRng::from_entropy(),
        }
    }

    /// Measures the radial distribution function.
    pub fn measure(
        &mut self,
        atoms: &AtomStore<T, 3>,
        simulation_box: &SimulationBox<T, 3>,
        neighbor_list: &NeighborList<T, 3>,
    ) {
        // measure vicinity first
        atoms
            .x
            .iter()
            .zip(&atoms.atom_types)
            .zip(&neighbor_list.neighbor_lists)
            .for_each(|((pos1, at1), neighbors)| {
                for &neighbor in neighbors {
                    // properties for atom2
                    let at2 = atoms.atom_types.get_by_idx(neighbor);
                    let pos2 = atoms.x.get_by_idx(neighbor);

                    // computing the distance
                    let dist = simulation_box.distance(pos1, pos2);

                    // computing the index for the rdf
                    let idx = self
                        .get_idx(at1, at2)
                        .expect("Please provide a proper amount of `AtomType`s.");

                    // if the distance is greater than the cutoff, we are done
                    if dist < self.cutoffs[idx] {
                        // mutable reference to the proper rdf
                        let rdf = &mut self.histogram[idx];
                        // computing the index of our rdf
                        let update_idx = T::from(self.num_intervals).unwrap() * dist / self.r_max;
                        // updating the rdf
                        rdf[update_idx.to_usize().unwrap()] += T::one();
                    }
                }
            });

        let prefac_loc = self.pre_fac * simulation_box.compute_volume();
    }
}

impl<T: Real> RDF<T> {
    fn get_idx(&self, at1: &AtomType<T>, at2: &AtomType<T>) -> Option<usize> {
        for (idx, &(id1, id2)) in self.pair_ids.iter().enumerate() {
            if id1 == at1.id() && id2 == at2.id() || id1 == at2.id() && id2 == at1.id() {
                return Some(idx);
            }
        }

        None
    }
}
