use crate::Real;
use crate::neighbor_list::NeighborList;
use crate::potential::pair_potential_collection::PairPotentialCollection;
use crate::simulation_box::SimulationBox;
use crate::storage::atom_store::AtomStore;
use crate::storage::atom_type::AtomType;
use crate::storage::vector::Iterable;
use std::io::Write;

/// Implements the radial distribution function.
pub struct RDF<T: Real> {
    atom_type: AtomType<T>,
    rdf: Vec<T>,
    rdf_counter: Vec<T>,
    // num_random_atoms: usize,
    num_intervals: usize,
    cutoff: T,
    r_max: T,
    dr: T,
    pre_fac: T,
    // rng: ChaChaRng,
}

impl<T: Real> RDF<T> {
    pub fn new(
        atom_type: &AtomType<T>,
        atoms: &AtomStore<T, 3>,
        num_intervals: usize,
        ppc: &PairPotentialCollection<T>,
        r_max: T,
    ) -> Self {
        let num_atoms = atoms.get_number_of_atoms(atom_type);
        let pre_fac = T::from(4.0).unwrap() / T::from(3.0).unwrap()
            * T::pi()
            * T::from(num_atoms).unwrap()
            * T::from(num_atoms).unwrap();

        Self {
            atom_type: *atom_type,
            rdf: vec![T::zero(); num_intervals],
            rdf_counter: vec![T::zero(); num_intervals],
            num_intervals,
            cutoff: ppc.get_cutoff_by_atom_type(atom_type, atom_type).unwrap(),
            r_max,
            dr: r_max / T::from_usize(num_intervals).unwrap(),
            pre_fac,
        }
    }

    pub fn measure(
        &mut self,
        atoms: &AtomStore<T, 3>,
        neighbor_list: &NeighborList<T, 3>,
        simulation_box: &SimulationBox<T, 3>,
    ) {
        // measure vicinity first
        atoms
            .x
            .iter()
            .zip(&atoms.atom_types)
            .zip(&neighbor_list.neighbor_lists)
            .filter(|((_, at), _)| at.id() == self.atom_type.id())
            .for_each(|((pos1, _at1), neighbors)| {
                for &neighbor in neighbors {
                    // properties for atom2
                    let at2 = atoms.atom_types.get_by_idx(neighbor);
                    let pos2 = atoms.x.get_by_idx(neighbor);

                    if at2.id() != self.atom_type.id() {
                        continue;
                    }

                    // computing the distance
                    let dist = simulation_box.distance(pos1, pos2);

                    // if the distance is greater than the cutoff, we are done
                    if dist < self.cutoff {
                        // computing the index of our rdf
                        let update_idx = T::from(self.num_intervals).unwrap() * dist / self.r_max;
                        // updating the rdf
                        self.rdf[update_idx.to_usize().unwrap()] += T::one();
                    }
                }
            });

        let prefac_loc = self.pre_fac / simulation_box.compute_volume();
        let mut old_volume = T::zero();

        for i in 0..self.num_intervals {
            let r_new =
                num_traits::Float::min((T::from(i).unwrap() + T::one()) * self.dr, self.cutoff);

            let new_volume = num_traits::Float::powi(r_new, 3);
            self.rdf_counter[i] += (new_volume - old_volume) * prefac_loc;

            if r_new >= self.cutoff {
                break;
            }

            old_volume = new_volume;
        }
    }

    pub fn write(&self, filename: &str) {
        let mut file = std::fs::File::create(filename).expect("Unable to create file");

        let mut contents = String::new();
        for i in 0..self.num_intervals {
            if self.rdf_counter[i] > T::zero() {
                let line = format!(
                    "{:.5}\t{:.5}\t{:.5}\n",
                    (T::from(i).unwrap() + T::from(0.5).unwrap()) * self.dr,
                    self.rdf[i] / self.rdf_counter[i],
                    self.rdf_counter[i]
                );
                contents.push_str(&line);
            }
        }

        file.write_all(contents.as_bytes())
            .expect("Unable to write to file");
    }
}
