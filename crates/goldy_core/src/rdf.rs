use nalgebra::SVector;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Uniform};

use crate::Real;
use crate::neighbor_list::NeighborList;
use crate::potential::pair_potential_collection::PairPotentialCollection;
use crate::simulation_box::SimulationBox;
use crate::storage::atom_store::AtomStore;
use crate::storage::atom_type::AtomType;
use crate::storage::vector::Iterable;
use std::io::Write;

/// Implements the radial distribution function.
pub struct RDF<T>
where
    T: Real + SampleUniform,
{
    atom_type: AtomType<T>,
    rdf: Vec<T>,
    rdf_counter: Vec<T>,
    num_random_atoms: usize,
    num_intervals: usize,
    cutoff: T,
    r_max: T,
    dr: T,
    pre_fac: T,
    rng: ChaChaRng,
    int_distr: Uniform<usize>,
    float_distr: Uniform<T>,
}

impl<T> RDF<T>
where
    T: Real + SampleUniform,
{
    pub fn new(
        atom_type: &AtomType<T>,
        atoms: &AtomStore<T, 3>,
        num_intervals: usize,
        ppc: &PairPotentialCollection<T>,
        num_random_atoms: usize,
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
            num_random_atoms,
            num_intervals,
            cutoff: ppc.get_cutoff_by_atom_type(atom_type, atom_type).unwrap(),
            r_max,
            dr: r_max / T::from_usize(num_intervals).unwrap(),
            pre_fac,
            rng: ChaChaRng::from_os_rng(),
            int_distr: Uniform::new(0, num_atoms).unwrap(),
            float_distr: Uniform::new(T::from(-0.5).unwrap(), T::from(0.5).unwrap()).unwrap(),
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

        let non_local_weight = T::from(0.01).unwrap();
        for id in 0..atoms.number_of_atoms() {
            let curr_pos = atoms.x.get_by_idx(id);

            for _ in 0..self.num_random_atoms {
                // draw random atom
                let random_id = (&self.int_distr).sample(&mut self.rng);

                if random_id == id || *atoms.atom_types.get_by_idx(random_id) != self.atom_type {
                    continue;
                }

                let neighbor_pos = atoms.x.get_by_idx(random_id);
                let dist = simulation_box.distance(curr_pos, neighbor_pos);

                if dist < self.r_max {
                    // computing the index of our rdf
                    let update_idx = T::from(self.num_intervals).unwrap() * dist / self.r_max;
                    // updating the rdf
                    self.rdf[update_idx.to_usize().unwrap()] += non_local_weight;
                }

                // draw reference atom from ideal-gas distribution
                let rand_atom =
                    SVector::<T, 3>::from_iterator((&self.float_distr).sample_iter(&mut self.rng));
                let rand_atom = simulation_box.to_real(rand_atom);
                let dist = rand_atom.norm();

                if dist < self.r_max {
                    // computing the index of our rdf
                    let update_idx = T::from(self.num_intervals).unwrap() * dist / self.r_max;
                    // updating the rdf
                    self.rdf_counter[update_idx.to_usize().unwrap()] += non_local_weight;
                }
            }
        }
    }

    pub fn write(&self, filename: &str) {
        let mut file = std::fs::File::create(filename).expect("Unable to create file");

        let mut contents = String::new();

        for i in 0..self.num_intervals {
            let normed_rdf = if (self.rdf[i] / self.rdf_counter[i]).is_finite() {
                self.rdf[i] / self.rdf_counter[i]
            } else {
                T::zero()
            };

            let line = format!(
                "{:.5}\t{:.5}\t{:.5}\n",
                (T::from(i).unwrap() + T::from(0.5).unwrap()) * self.dr,
                normed_rdf,
                self.rdf_counter[i]
            );
            contents.push_str(&line);
        }

        file.write_all(contents.as_bytes())
            .expect("Unable to write to file");
    }
}
