use std::fmt::Display;

use derive_builder::Builder;
use nalgebra as na;
use nalgebra::{SMatrix, SVector};

use crate::{Real, storage::vector::Positions};

#[derive(Debug, Builder, PartialEq, Eq)]
pub struct SimulationBox<T: Real, const D: usize> {
    hmatrix: SMatrix<T, D, D>,
    #[builder(setter(skip), default = "self.inv_hmatrix()?")]
    inv_hmatrix: SMatrix<T, D, D>,
    #[builder(setter)]
    boundary_type: BoundaryTypes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryTypes {
    Open,
    Periodic,
}

impl<T: Real, const D: usize> SimulationBox<T, D> {
    #[inline]
    /// Returns the squared distance between two `SVector`s. This
    /// method obeys the defined boundary conditions.
    pub fn sq_distance(&self, x1: &SVector<T, D>, x2: &SVector<T, D>) -> T {
        match self.boundary_type {
            // Periodic boundaries are currently the only special cases.
            BoundaryTypes::Periodic => {
                /*                // making two local copies
                let mut x1 = *x1;
                let mut x2 = *x2;

                // setting them back
                self.set_back_to_cell(&mut x1);
                self.set_back_to_cell(&mut x2);

                // and computing the squared norm
                (x1 - x2).norm_squared()*/
                self.difference(x1, x2).norm_squared()
            }
            // Here, everything is normal.
            BoundaryTypes::Open => (x1 - x2).norm_squared(),
        }
    }

    /// Returns the distance between two `SVector`s. This
    /// method obeys the defined boundary conditions.
    #[inline]
    pub fn distance(&self, x1: &SVector<T, D>, x2: &SVector<T, D>) -> T {
        na::ComplexField::sqrt(self.sq_distance(x1, x2))
    }

    /// Applies the specified boundary conditions.
    #[inline]
    pub fn apply_boundary_conditions(&self, x: &mut Positions<T, D>) {
        match self.boundary_type {
            BoundaryTypes::Periodic => {
                x.iter_mut().for_each(|x| self.set_back_to_cell(x));
            }
            BoundaryTypes::Open => {
                // nothing to do!
            }
        }
    }

    /// Tests, if a given position lies in the `SimulationBox`.
    pub fn contains(&self, x: &SVector<T, D>) -> bool {
        match self.boundary_type {
            BoundaryTypes::Periodic => {
                let x = self.to_relative(*x);

                x.iter().all(|&x| T::zero() <= x && x <= T::one())
            }
            // This case is trivial
            BoundaryTypes::Open => true,
        }
    }

    /// Computes the difference of the two given `SVector`s w.r.t. to the boundary conditions.
    fn difference(&self, x1: &SVector<T, D>, x2: &SVector<T, D>) -> SVector<T, D> {
        let mut d = self.to_relative(x1 - x2);

        d.iter_mut().for_each(|x| {
            *x -= na::ComplexField::round(*x);
        });

        self.to_real(d)
    }

    /// Returns a relative `SVector`.
    #[inline]
    fn to_relative(&self, x: SVector<T, D>) -> SVector<T, D> {
        self.inv_hmatrix * x
    }

    /// Returns a real `SVector`.
    #[inline]
    fn to_real(&self, x: SVector<T, D>) -> SVector<T, D> {
        self.hmatrix * x
    }

    /// Sets a `SVector` back into the simulation cell
    #[inline]
    fn set_back_to_cell(&self, x: &mut SVector<T, D>) {
        *x = self.to_relative(*x);

        x.iter_mut().for_each(|x| {
            if *x > T::one() || *x < T::zero() {
                *x -= na::ComplexField::floor(*x);
            }
        });

        *x = self.to_real(*x);
    }
}

impl<T: Real + Display> SimulationBox<T, 3> {
    /// Returns a string representation of the `SimulationBox` boundaries.
    pub(crate) fn convert_to_string(&self) -> String {
        let (p1, p2) = (
            // the 'lower left' point of the simulation cell in reduced coordinates
            SVector::<T, 3>::zeros(),
            // the 'upper right' point of the simulation cell in reduced coordinates
            SVector::<T, 3>::from_element(T::one()),
        );

        // converting them to real space
        let (p1, p2) = (self.to_real(p1), self.to_real(p2));

        let mut contents = String::new();

        let names = ["xlo xhi", "ylo yhi", "zlo zhi"];
        p1.iter()
            .zip(&p2)
            .zip(&names)
            .for_each(|((&p1, &p2), &name)| {
                contents.push_str(format!("{p1: <14.5} {p2:14.5} {name}").as_str());
                contents.push('\n');
            });

        contents
    }
}

impl<T: Real, const D: usize> SimulationBoxBuilder<T, D> {
    fn inv_hmatrix(&self) -> Result<SMatrix<T, D, D>, String> {
        match self.hmatrix {
            Some(hmatrix) => hmatrix
                .try_inverse()
                .ok_or_else(|| "Please provide an invertible hmatrix.".into()),

            None => Err("Please provide an invertible hmatrix.".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use assert_approx_eq::assert_approx_eq;
    use na::ComplexField;
    use nalgebra::{Matrix3, Vector3};

    use super::*;

    #[test]
    fn test_builder() {
        // Let's create a simulation box with...
        let sim_box = SimulationBoxBuilder::<f32, 3>::default()
            // ... an appropriate hmatrix and
            .hmatrix(SMatrix::from_diagonal_element(10.0))
            // ... open boundaries.
            .boundary_type(BoundaryTypes::Open)
            // If we build this,
            .build()
            // it should work well.
            .unwrap();

        assert_eq!(
            sim_box,
            SimulationBox {
                hmatrix: SMatrix::from_diagonal_element(10.0),
                inv_hmatrix: SMatrix::from_diagonal_element(0.1),
                boundary_type: BoundaryTypes::Open,
            }
        );

        // Now let's create one with ...
        assert!(SimulationBoxBuilder::<f32, 2>::default()
            // ... an inappropriate hmatrix.
            .hmatrix(SMatrix::zeros())
            // The boundaries must be still okay.
            .boundary_type(BoundaryTypes::Periodic)
            // If we call build this time, ...
            .build()
            // ... it must fail.
            .is_err());
    }

    #[test]
    fn test_distance() {
        // Let's create a simulation box with open boundaries...
        let open_sim_box = SimulationBoxBuilder::<f64, 4>::default()
            .hmatrix(SMatrix::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Open)
            .build()
            .unwrap();

        // ... and test it with positions inside the box.
        let x1 = SVector::<f64, 4>::from_iterator([1.0, 1.0, 1.0, 1.0]);
        let x2 = SVector::<f64, 4>::from_iterator([9.0, 9.0, 9.0, 9.0]);

        assert_approx_eq!(open_sim_box.sq_distance(&x1, &x2), 256.0);
        assert_approx_eq!(open_sim_box.sq_distance(&x2, &x1), 256.0);
        assert_approx_eq!(open_sim_box.distance(&x1, &x2), 16.0);
        assert_approx_eq!(open_sim_box.distance(&x2, &x1), 16.0);

        assert_approx_eq!(open_sim_box.sq_distance(&-x1, &(x2 * 2.0)), 1444.0);
        assert_approx_eq!(open_sim_box.sq_distance(&(x2 * 2.0), &-x1), 1444.0);
        assert_approx_eq!(open_sim_box.distance(&-x1, &(x2 * 2.0)), 38.0);
        assert_approx_eq!(open_sim_box.distance(&(x2 * 2.0), &-x1), 38.0);

        // Now let's run the tests with periodic boundaries.
        let periodic_sim_box = SimulationBoxBuilder::<f64, 2>::default()
            .hmatrix(SMatrix::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Periodic)
            .build()
            .unwrap();

        // Setting up the positions (without wrapping around).
        let x1 = SVector::<f64, 2>::zeros();
        let x2 = SVector::<f64, 2>::from_iterator([5.0, 5.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 50.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 50.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 50.0.sqrt());
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 50.0.sqrt());

        // testing exactly at the boundaries
        let x1 = SVector::<f64, 2>::from_iterator([0.0, 10.0]);
        let x2 = SVector::<f64, 2>::from_iterator([10.0, 0.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 0.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 0.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 0.0);
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 0.0);

        // Setting up positions, s.t. periodic boundaries kick in.
        let x1 = SVector::<f64, 2>::from_iterator([1.0, 1.0]);
        let x2 = SVector::<f64, 2>::from_iterator([9.0, 9.0]);
        let x3 = SVector::<f64, 2>::from_iterator([-1.0, -1.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 8.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 8.0);
        assert_approx_eq!(
            periodic_sim_box.sq_distance(&x1, &x2),
            periodic_sim_box.sq_distance(&x1, &x3)
        );
        assert_approx_eq!(
            periodic_sim_box.sq_distance(&x2, &x1),
            periodic_sim_box.sq_distance(&x1, &x3)
        );
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 8.0.sqrt());
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 8.0.sqrt());

        // The implementation should also work for positions which are far outside.
        let x1 = SVector::<f64, 2>::from_iterator([-39.0, -30.0]);
        let x2 = SVector::<f64, 2>::from_iterator([40.0, 41.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 2.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 2.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 2.0.sqrt());
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 2.0.sqrt());
    }

    #[test]
    fn test_set_back_to_cell() {
        // defining a 10x10x10 simulation box with periodic boundaries
        let simulation_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Periodic)
            .build()
            .unwrap();

        // wrap_around shouldn't do anything on these cases ...
        let mut x1 = Vector3::new(1.0, 1.0, 1.0);
        let mut x2 = Vector3::new(0.0, 0.0, 0.0);
        let mut x3 = Vector3::new(10.0, 10.0, 10.0);

        simulation_box.set_back_to_cell(&mut x1);
        simulation_box.set_back_to_cell(&mut x2);
        simulation_box.set_back_to_cell(&mut x3);

        // ... so these assertions must hold.
        assert_eq!(round_vector(&x1), Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(round_vector(&x2), Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(round_vector(&x3), Vector3::new(10.0, 10.0, 10.0));

        // adding cases on the "right" side of the box
        let mut x4 = Vector3::new(11.0, 9.0, 16.0);
        let mut x5 = Vector3::new(20.0, 23.0, 37.0);

        simulation_box.set_back_to_cell(&mut x4);
        simulation_box.set_back_to_cell(&mut x5);

        assert_eq!(round_vector(&x4), Vector3::new(1.0, 9.0, 6.0));
        assert_eq!(round_vector(&x5), Vector3::new(0.0, 3.0, 7.0));

        // adding cases on the "left" side of the box
        let mut x6 = Vector3::new(-3.0, -7.0, -10.0);
        let mut x7 = Vector3::new(-21.0, -14.0, -39.0);
        let mut x8 = Vector3::new(-5.0, -5.0, -5.0);

        simulation_box.set_back_to_cell(&mut x6);
        simulation_box.set_back_to_cell(&mut x7);
        simulation_box.set_back_to_cell(&mut x8);

        assert_eq!(round_vector(&x6), Vector3::new(7.0, 3.0, 0.0));
        assert_eq!(round_vector(&x7), Vector3::new(9.0, 6.0, 1.0));
        assert_eq!(round_vector(&x8), Vector3::new(5.0, 5.0, 5.0));
    }

    #[test]
    fn test_conversion() {
        // defining a 10x10x10 simulation box with periodic boundaries
        let simulation_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal_element(10.0))
            .boundary_type(BoundaryTypes::Periodic)
            .build()
            .unwrap();

        // defining some points in the simulation box ...
        let x1 = Vector3::new(1.0, 1.0, 1.0);
        let x2 = Vector3::new(0.0, 0.0, 0.0);
        let x3 = Vector3::new(9.0, 9.0, 9.0);
        // ... and on the outside
        let x4 = Vector3::new(-10.0, -10.0, -10.0);
        let x5 = Vector3::new(20.0, 20.0, 20.0);
        let x6 = Vector3::new(1.0, 20.0, -10.0);

        // converting them to relative coordinates
        let x1_rel = simulation_box.to_relative(x1);
        let x2_rel = simulation_box.to_relative(x2);
        let x3_rel = simulation_box.to_relative(x3);
        let x4_rel = simulation_box.to_relative(x4);
        let x5_rel = simulation_box.to_relative(x5);
        let x6_rel = simulation_box.to_relative(x6);

        // and converting them back
        let x1_back = simulation_box.to_real(x1_rel);
        let x2_back = simulation_box.to_real(x2_rel);
        let x3_back = simulation_box.to_real(x3_rel);
        let x4_back = simulation_box.to_real(x4_rel);
        let x5_back = simulation_box.to_real(x5_rel);
        let x6_back = simulation_box.to_real(x6_rel);

        // so the original ones must equal the back-transformed ones
        assert_eq!(x1, round_vector(&x1_back));
        assert_eq!(x2, round_vector(&x2_back));
        assert_eq!(x3, round_vector(&x3_back));
        assert_eq!(x4, round_vector(&x4_back));
        assert_eq!(x5, round_vector(&x5_back));
        assert_eq!(x6, round_vector(&x6_back));
    }

    fn round_vector<T: Real, const D: usize>(x: &SVector<T, D>) -> SVector<T, D> {
        let mut x = *x;

        x.iter_mut().for_each(|x| *x = na::ComplexField::round(*x));

        x
    }

    #[test]
    fn test_convert_to_string() {
        let simulation_box = SimulationBoxBuilder::default()
            .hmatrix(Matrix3::from_diagonal(&Vector3::new(10.0, 20.0, 30.0)))
            .boundary_type(BoundaryTypes::Periodic)
            .build()
            .unwrap();

        assert_eq!(
            simulation_box.convert_to_string(),
            r"0.00000              10.00000 xlo xhi
0.00000              20.00000 ylo yhi
0.00000              30.00000 zlo zhi
"
        );
    }
}
