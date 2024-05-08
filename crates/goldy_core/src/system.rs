use nalgebra::{SMatrix, SVector};

use crate::{
    simulation_box::{BoundaryTypes, SimulationBox, SimulationBoxBuilder},
    storage::{
        atom_store::{AtomStore, AtomStoreBuilder},
        atom_type::AtomType,
        atom_type_store::AtomTypeStoreBuilder,
        vector::{Forces, Positions, Velocities},
    },
    Real,
};

pub struct System<T: Real, const D: usize> {
    pub atoms: AtomStore<T, D>,
    pub sim_box: SimulationBox<T, D>,
}

impl<T: Real, const D: usize> System<T, D> {
    /// Creates a 2D-`System` based on the given number of crystals, the lattice constant, boundary
    /// type and `AtomType`.
    pub fn new_cubic(
        crystals: SVector<usize, D>,
        lattice_constant: T,
        boundary_type: BoundaryTypes,
        atom_type: AtomType<T>,
    ) -> Self {
        // creating the SimulationBox
        let sim_box = SimulationBoxBuilder::default()
            .hmatrix(SMatrix::from_diagonal(&SVector::from_iterator(
                crystals.iter().map(|&x| {
                    lattice_constant * T::from_usize(x).unwrap()
                        + lattice_constant / T::from(2.0).unwrap()
                }),
            )))
            .boundary_type(boundary_type)
            .build()
            .expect("Building the SimulationBox should have worked");

        // computing the number of atoms
        let num_atoms = crystals.iter().map(|x| x + 1).product::<usize>();

        // creating the positions
        let crystals = crystals.data.0[0];
        let mut x = vec![SVector::zeros(); num_atoms];
        Self::rec_cubic(&mut x, lattice_constant, SVector::zeros(), crystals, D);
        let x = Positions::from_iter(x);

        // creating the remaining properties
        let v = Velocities::zeros(num_atoms);
        let f = Forces::zeros(num_atoms);
        let atom_types = AtomTypeStoreBuilder::new()
            .add_many(atom_type, num_atoms)
            .build();

        // building the store
        let atoms = AtomStoreBuilder::default()
            .positions(x)
            .velocities(v)
            .forces(f)
            .atom_types(atom_types)
            .build()
            .expect("Could not build AtomStore");

        Self { sim_box, atoms }
    }

    fn rec_cubic(
        x: &mut [SVector<T, D>],
        lattice_constant: T,
        shift_vector: SVector<T, D>,
        crystals: [usize; D],
        dim: usize,
    ) {
        // For setting the current offset.
        let mut arr = [T::zero(); D];

        if dim == 1 {
            // we reached the base case
            x.iter_mut().enumerate().for_each(|(i, pos)| {
                // computing the x-coordinate
                arr[0] = T::from_usize(i).unwrap() * lattice_constant;
                // setting the shift_vector at x
                let shift_vector = shift_vector + SVector::from_vec(arr.to_vec());
                println!("{shift_vector}");

                // setting the position
                *pos += shift_vector;
            });
        } else {
            // the recursive step
            // Dividing the positions in N / (c[dim] + 1) many chunks
            x.chunks_mut(x.len() / (crystals[dim - 1] + 1))
                // Iterating over each chunk...
                .enumerate()
                .for_each(|(i, x_chunk)| {
                    // ...updating the new offset...
                    arr[dim - 1] = T::from_usize(i).unwrap() * lattice_constant;
                    let shift_vector = shift_vector + SVector::from_vec(arr.to_vec());

                    // ... and going into recursion.
                    Self::rec_cubic(x_chunk, lattice_constant, shift_vector, crystals, dim - 1);
                });
        }
    }
}

#[cfg(test)]
mod tests {

    use assert_approx_eq::assert_approx_eq;
    use num_traits::Zero;

    use super::*;

    const ONE_D: usize = 1;
    const TWO_D: usize = 2;
    const THREE_D: usize = 3;

    #[test]
    fn test_rec_cubic() {
        // testing the 1D case
        let mut x = vec![SVector::<f64, ONE_D>::zero(); 10];
        let lattice_constant = 2.0;
        let crystals = [9];

        System::rec_cubic(
            &mut x,
            lattice_constant,
            SVector::<f64, ONE_D>::zero(),
            crystals,
            ONE_D,
        );

        x.iter().enumerate().for_each(|(i, x)| {
            x.iter()
                .for_each(|&x| assert_approx_eq!(x, lattice_constant * i as f64))
        });

        // testing the 2D case
        let crystals = [10, 5];
        let mut x = vec![SVector::<f64, TWO_D>::zero(); crystals.iter().map(|c| c + 1).product()];
        let mut x_ref = x.clone();
        let lattice_constant = 0.5;

        System::rec_cubic(
            &mut x,
            lattice_constant,
            SVector::<f64, TWO_D>::zero(),
            crystals,
            TWO_D,
        );

        // setting up the reference
        for y in 0..=crystals[1] {
            for x in 0..=crystals[0] {
                let idx = x + y * (crystals[0] + 1);

                x_ref[idx] = SVector::<f64, TWO_D>::new(
                    x as f64 * lattice_constant,
                    y as f64 * lattice_constant,
                );
            }
        }

        x.iter().zip(&x_ref).for_each(|(x, x_ref)| {
            x.iter()
                .zip(x_ref)
                .for_each(|(&x, &x_ref)| assert_approx_eq!(x, x_ref))
        });

        // testing the 3D case
        let crystals = [10, 5, 8];
        let mut x = vec![SVector::<f64, THREE_D>::zero(); crystals.iter().map(|c| c + 1).product()];
        let mut x_ref = x.clone();
        let lattice_constant = 1.23;

        System::rec_cubic(
            &mut x,
            lattice_constant,
            SVector::<f64, THREE_D>::zero(),
            crystals,
            THREE_D,
        );

        // setting up the reference
        let mut idx = 0;
        for z in 0..=crystals[2] {
            for y in 0..=crystals[1] {
                for x in 0..=crystals[0] {
                    x_ref[idx] = SVector::<f64, THREE_D>::new(
                        x as f64 * lattice_constant,
                        y as f64 * lattice_constant,
                        z as f64 * lattice_constant,
                    );

                    idx += 1;
                }
            }
        }

        x.iter().zip(&x_ref).for_each(|(x, x_ref)| {
            x.iter()
                .zip(x_ref)
                .for_each(|(&x, &x_ref)| assert_approx_eq!(x, x_ref))
        });
    }
}
