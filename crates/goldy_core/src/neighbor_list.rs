use crate::potential::pair_potential_collection::PairPotentialCollection;
use crate::simulation_box::SimulationBox;
use crate::storage::atom_type_store::AtomTypeStore;
use crate::storage::vector::Positions;
use crate::Real;
use nalgebra::SVector;

/// Struct representing a neighbor list for atoms in a simulation box.
pub struct NeighborList<T: Real, const D: usize> {
    pub neighbor_lists: Vec<Vec<usize>>,
    old_x: Positions<T, D>,
    skin: T,
    offsets: [usize; D],
    num_bins: [usize; D],
}

impl<T: Real, const D: usize> NeighborList<T, D> {
    /// Creates a new `NeighborList`.
    ///
    /// # Arguments
    ///
    /// * `x` - Positions of the atoms.
    /// * `atom_types` - Store of atom types.
    /// * `sim_box` - The simulation box.
    /// * `ppc` - Pair potential collection.
    pub fn new(
        x: &Positions<T, D>,
        atom_types: &AtomTypeStore<T>,
        sim_box: &SimulationBox<T, D>,
        ppc: PairPotentialCollection<T>,
    ) -> Self {
        let max_cutoff = ppc.get_cutoffs().iter().fold(T::zero(), |acc, &cutoff| {
            num_traits::Float::max(acc, cutoff + T::from(0.75).unwrap())
        });

        let num_bins_float = sim_box.get_max_values() / max_cutoff;
        let mut num_bins = [0; D];
        let mut offsets = [0; D];

        for (idx, bin_size) in num_bins_float.iter().enumerate() {
            num_bins[idx] = bin_size.to_usize().unwrap();
            offsets[idx] = 2.min(num_bins[idx]);
        }

        let mut nl = Self {
            neighbor_lists: Vec::new(),
            old_x: Positions::zeros(0),
            skin: T::from(0.5).unwrap(),
            offsets,
            num_bins,
        };

        nl.update(x, atom_types, sim_box, ppc);

        nl
    }

    /// Updates the neighbor list if needed.
    ///
    /// # Arguments
    ///
    /// * `x` - Positions of the atoms.
    /// * `atom_types` - Store of atom types.
    /// * `sim_box` - The simulation box.
    /// * `ppc` - Pair potential collection.
    pub fn update(
        &mut self,
        x: &Positions<T, D>,
        atom_types: &AtomTypeStore<T>,
        sim_box: &SimulationBox<T, D>,
        ppc: PairPotentialCollection<T>,
    ) {
        if self.update_needed(x, sim_box) {
            self.old_x = x.clone();

            let mut new_neighbor_lists = vec![Vec::new(); x.len()];

            // Building the binning list
            let mut bin_list = vec![Vec::new(); self.num_bins.iter().product()];

            x.iter().enumerate().for_each(|(id, x)| {
                let idx = self.get_idx_from_conceptual(self.map_to_conceptual(x, sim_box));

                bin_list[idx].push((id, *x));
            });

            // Building the actual neighbor list
            let neighbor_bin_indices = self.create_neighbor_bin_indices();

            for bin in bin_list.iter() {
                for neighbor_bin_idx in neighbor_bin_indices.iter() {
                    neighbor_bin_idx.iter().for_each(|&idx| {
                        let neighbor_bin = &bin_list[idx];

                        for (id1, atom1) in bin.iter() {
                            for (id2, atom2) in neighbor_bin.iter() {
                                if id1 != id2 {
                                    let at1 = atom_types.get_by_idx(*id1);
                                    let at2 = atom_types.get_by_idx(*id2);

                                    let cutoff = ppc
                                        .get_cutoff_by_atom_type(at1, at2)
                                        .expect("Please provide a proper amount of `AtomType`s.");

                                    if sim_box.distance(atom1, atom2) + self.skin < cutoff {
                                        new_neighbor_lists[*id1].push(*id2);
                                        new_neighbor_lists[*id2].push(*id1);
                                    }
                                }
                            }
                        }
                    })
                }
            }

            self.neighbor_lists = new_neighbor_lists;
        }
    }

    /// Creates indices for neighboring bins.
    #[inline]
    fn create_neighbor_bin_indices(&self) -> Vec<Vec<usize>> {
        let relative_conceptual_indices = generate_conceptual_indices(&self.offsets);
        let conceptual_bin_indices = generate_conceptual_indices(&self.num_bins);

        conceptual_bin_indices
            .iter()
            .map(|&conceptual_bin_idx| {
                relative_conceptual_indices
                    .iter()
                    .map(|&relative_conceptual_idx| {
                        let mut new_idx =
                            add_conceptuals(conceptual_bin_idx, relative_conceptual_idx);

                        new_idx
                            .iter_mut()
                            .zip(&self.num_bins)
                            .for_each(|(idx, bin_size)| *idx %= bin_size);

                        self.get_idx_from_conceptual(new_idx)
                    })
                    .collect()
            })
            .collect()
    }

    /// Checks if an update is needed based on the positions and simulation box.
    #[inline]
    fn update_needed(&self, x: &Positions<T, D>, sim_box: &SimulationBox<T, D>) -> bool {
        self.old_x.len() != x.len()
            || self
                .old_x
                .iter()
                .zip(x)
                .any(|(x1, x2)| sim_box.distance(x1, x2) >= T::from(0.5).unwrap() * self.skin)
    }

    /// Maps a position to a conceptual bin index.
    ///
    /// # Arguments
    ///
    /// * `pos` - Position vector.
    /// * `sim_box` - The simulation box.
    #[inline]
    fn map_to_conceptual(&self, pos: &SVector<T, D>, sim_box: &SimulationBox<T, D>) -> [usize; D] {
        let mut conceptual_idx = [0; D];
        let rel_pos = sim_box.to_relative(*pos);

        for (idx, &d) in rel_pos.iter().enumerate() {
            conceptual_idx[idx] = (d * T::from(self.num_bins[idx]).unwrap())
                .to_usize()
                .unwrap()
                % self.num_bins[idx];
        }

        conceptual_idx
    }

    /// Converts a conceptual bin index to a linear index.
    ///
    /// # Arguments
    ///
    /// * `conceptual_idx` - Conceptual bin index.
    #[inline]
    fn get_idx_from_conceptual(&self, conceptual_idx: [usize; D]) -> usize {
        self.get_idx_rec(conceptual_idx, 0, D)
    }

    /// Recursive helper function to convert a conceptual bin index to a linear index.
    ///
    /// # Arguments
    ///
    /// * `conceptual_idx` - Conceptual bin index.
    /// * `idx_acc` - Accumulated index.
    /// * `curr_dim` - Current dimension.
    #[inline]
    fn get_idx_rec(&self, conceptual_idx: [usize; D], idx_acc: usize, curr_dim: usize) -> usize {
        match curr_dim {
            0 => idx_acc,
            _ => self.get_idx_rec(
                conceptual_idx,
                idx_acc
                    + conceptual_idx[curr_dim - 1]
                        * self.num_bins[0..curr_dim - 1].iter().product::<usize>(),
                curr_dim - 1,
            ),
        }
    }
}

/// Adds two conceptual indices.
///
/// # Arguments
///
/// * `a` - First conceptual index.
/// * `b` - Second conceptual index.
#[inline]
fn add_conceptuals<const D: usize>(mut a: [usize; D], b: [usize; D]) -> [usize; D] {
    a.iter_mut().zip(&b).for_each(|(a, &b)| *a += b);

    a
}

/// Generates all possible conceptual indices for a given array of sizes.
///
/// # Arguments
///
/// * `arr` - Array of sizes.
#[inline]
fn generate_conceptual_indices<const D: usize>(arr: &[usize; D]) -> Vec<[usize; D]> {
    let mut indices = Vec::new();
    generate_conceptual_indices_rec(arr, 0, [0; D], &mut indices);
    indices
}

/// Recursive helper function to generate all possible conceptual indices.
///
/// # Arguments
///
/// * `arr` - Array of sizes.
/// * `curr_dim` - Current dimension.
/// * `curr_idx` - Current index.
/// * `indices` - Vector to store generated indices.
#[inline]
fn generate_conceptual_indices_rec<const D: usize>(
    arr: &[usize; D],
    curr_dim: usize,
    curr_idx: [usize; D],
    indices: &mut Vec<[usize; D]>,
) {
    if curr_dim == D {
        indices.push(curr_idx);
    } else {
        let mut new_curr_idx = curr_idx;

        for i in 0..arr[curr_dim] {
            new_curr_idx[curr_dim] = i;
            generate_conceptual_indices_rec(arr, curr_dim + 1, new_curr_idx, indices);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::neighbor_list::{generate_conceptual_indices, NeighborList};
    use crate::simulation_box::{BoundaryTypes, SimulationBoxBuilder};
    use crate::storage::vector::Positions;
    use nalgebra::{Matrix3, Vector3};

    #[test]
    fn test_get_idx() {
        let neighbor_list: NeighborList<f64, 2> = NeighborList {
            neighbor_lists: Vec::new(),
            old_x: Positions::zeros(0),
            skin: 0.0,
            offsets: [0, 0],
            num_bins: [5, 10],
        };

        assert_eq!(neighbor_list.get_idx_from_conceptual([0, 0]), 0);
        assert_eq!(neighbor_list.get_idx_from_conceptual([1, 0]), 1);
        assert_eq!(neighbor_list.get_idx_from_conceptual([0, 1]), 5);
        assert_eq!(neighbor_list.get_idx_from_conceptual([3, 4]), 23);
        assert_eq!(neighbor_list.get_idx_from_conceptual([4, 9]), 49);

        let neighbor_list: NeighborList<f64, 3> = NeighborList {
            neighbor_lists: Vec::new(),
            old_x: Positions::zeros(0),
            skin: 0.0,
            offsets: [0, 0, 0],
            num_bins: [4, 5, 6],
        };

        assert_eq!(neighbor_list.get_idx_from_conceptual([0, 0, 0]), 0);
        assert_eq!(neighbor_list.get_idx_from_conceptual([3, 4, 5]), 119);
        assert_eq!(neighbor_list.get_idx_from_conceptual([1, 1, 1]), 25);
    }

    #[test]
    fn test_map_to_conceptual() {
        let sim_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Periodic)
            .build()
            .unwrap();

        let nl = NeighborList {
            neighbor_lists: Vec::new(),
            old_x: Positions::zeros(0),
            skin: 0.0,
            offsets: [0, 0, 0],
            num_bins: [10, 10, 10],
        };

        assert_eq!(
            nl.map_to_conceptual(&Vector3::new(0.0, 0.0, 0.0), &sim_box),
            [0, 0, 0]
        );

        assert_eq!(
            nl.map_to_conceptual(&Vector3::new(1.0, 2.0, 3.0), &sim_box),
            [1, 2, 3]
        );
    }

    #[test]
    fn test_generate_indices() {
        assert_eq!(
            vec![
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            generate_conceptual_indices(&[2, 2, 2])
        );
    }
}
