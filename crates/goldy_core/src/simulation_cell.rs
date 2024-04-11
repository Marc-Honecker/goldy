use nalgebra::SMatrix;

use crate::Real;

pub struct Cell<T, const D: usize>
where
    T: Real,
{
    hmatrix: SMatrix<T, D, D>,
    inv_hmatrix: SMatrix<T, D, D>,
}

impl<T, const D: usize> Cell<T, D>
where
    T: Real,
{
    pub fn from_matrix(mat: SMatrix<T, D, D>) -> Self {
        Self {
            hmatrix: mat,
            inv_hmatrix: mat.try_inverse().unwrap(),
        }
    }

    pub fn get_hmatrix(&self) -> &SMatrix<T, D, D> {
        &self.hmatrix
    }

    pub fn get_inv_hmatrix(&self) -> &SMatrix<T, D, D> {
        &self.inv_hmatrix
    }
}
