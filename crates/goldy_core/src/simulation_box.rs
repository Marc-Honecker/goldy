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
                // making two local copies
                let mut x1 = *x1;
                let mut x2 = *x2;

                // wrapping them around
                self.wrap_around(&mut x1);
                self.wrap_around(&mut x2);

                // and computing the squared norm
                (x1 - x2).norm_squared()
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
                x.iter_mut().for_each(|x| self.wrap_around(x));
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

    /// Wraps a `SVector` s.t. it obeys PBC.
    #[inline]
    fn wrap_around(&self, x: &mut SVector<T, D>) {
        *x = self.to_relative(*x);

        x.iter_mut().for_each(|x| {
            if *x > T::one() || *x < T::zero() {
                *x -= na::ComplexField::floor(*x);
            }
        });

        *x = self.to_real(*x);
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
            .hmatrix(SMatrix::from_diagonal_element(15.0))
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

        // Setting up positions, s.t. periodic boundaries kick in.
        let x1 = SVector::<f64, 2>::from_iterator([1.0, 18.0]);
        let x2 = SVector::<f64, 2>::from_iterator([18.0, 1.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 8.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 8.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 8.0.sqrt());
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 8.0.sqrt());

        // The implementation should also work for positions which are far outside.
        let x1 = SVector::<f64, 2>::from_iterator([-44.0, -45.0]);
        let x2 = SVector::<f64, 2>::from_iterator([45.0, 46.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 2.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 2.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 2.0.sqrt());
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 2.0.sqrt());
    }

    #[test]
    fn test_wrap_around() {
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

        simulation_box.wrap_around(&mut x1);
        simulation_box.wrap_around(&mut x2);
        simulation_box.wrap_around(&mut x3);

        // ... so these assertions must hold.
        assert_eq!(round_vector(&x1), Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(round_vector(&x2), Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(round_vector(&x3), Vector3::new(10.0, 10.0, 10.0));

        // adding cases on the "right" side of the box
        let mut x4 = Vector3::new(11.0, 9.0, 16.0);
        let mut x5 = Vector3::new(20.0, 23.0, 37.0);

        simulation_box.wrap_around(&mut x4);
        simulation_box.wrap_around(&mut x5);

        assert_eq!(round_vector(&x4), Vector3::new(1.0, 9.0, 6.0));
        assert_eq!(round_vector(&x5), Vector3::new(0.0, 3.0, 7.0));

        // adding cases on the "left" side of the box
        let mut x6 = Vector3::new(-3.0, -7.0, -10.0);
        let mut x7 = Vector3::new(-21.0, -14.0, -39.0);
        let mut x8 = Vector3::new(-5.0, -5.0, -5.0);

        simulation_box.wrap_around(&mut x6);
        simulation_box.wrap_around(&mut x7);
        simulation_box.wrap_around(&mut x8);

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
}
