use derive_builder::Builder;
use nalgebra as na;

use goldy_core::Real;
use goldy_storage::vector::Positions;
use nalgebra::{SMatrix, SVector};

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

                // computing the "real" d
                self.wrap_around(&mut d);

                // we are done and return the squared norm
                d.norm_squared()
            }
            // Here, everything is normal.
            _ => (x1 - x2).norm_squared(),
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
        x.iter_mut().for_each(|x| *x -= na::ComplexField::floor(*x));
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
}
