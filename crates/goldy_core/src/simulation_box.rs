use derive_builder::Builder;
use nalgebra as na;
use nalgebra::{SMatrix, SVector};

use crate::{storage::vector::Positions, Real};

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
    /// method obeyes the defined boundary conditions.
    pub fn sq_distance(&self, x1: &SVector<T, D>, x2: &SVector<T, D>) -> T {
        match self.boundary_type {
            // Periodic boundaries are currently the only special cases.
            BoundaryTypes::Periodic => {
                // computing the difference
                let mut d = x1 - x2;

                // computing the wrapped d
                self.wrap_around(&mut d);

                // we are done and return the squared norm
                d.norm_squared()
            }
            // Here, everything is normal.
            BoundaryTypes::Open => (x1 - x2).norm_squared(),
        }
    }

    /// Returns the distance between two `SVector`s. This
    /// method obeyes the defined boundary conditions.
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

    /// Wraps a `SVector` s.t. it obeyes PBC.
    #[inline]
    fn wrap_around(&self, x: &mut SVector<T, D>) {
        *x = self.to_relative(*x);
        x.iter_mut().for_each(|x| {
            *x -= na::ComplexField::round(*x);
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
    use super::*;

    use assert_approx_eq::assert_approx_eq;
    use na::ComplexField;

    #[test]
    fn test_builder() {
        // Let's create a simulation box with...
        let sim_box = SimulationBoxBuilder::<f32, 3>::default()
            // ... a appropiate hmatrix and
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
            // ... an inappropiate hmatrix.
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

        // ... and test it with positions inside of the box.
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
        let x1 = SVector::<f64, 2>::from_iterator([1.0, 14.0]);
        let x2 = SVector::<f64, 2>::from_iterator([14.0, 1.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 8.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 8.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 8.0.sqrt());
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 8.0.sqrt());

        // The implementation should also work for positions which are far outside.
        let x1 = SVector::<f64, 2>::from_iterator([-45.0, -45.0]);
        let x2 = SVector::<f64, 2>::from_iterator([45.0, 45.0]);

        assert_approx_eq!(periodic_sim_box.sq_distance(&x1, &x2), 0.0);
        assert_approx_eq!(periodic_sim_box.sq_distance(&x2, &x1), 0.0);
        assert_approx_eq!(periodic_sim_box.distance(&x1, &x2), 0.0);
        assert_approx_eq!(periodic_sim_box.distance(&x2, &x1), 0.0);
    }
}
