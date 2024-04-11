use nalgebra::SVector;

use crate::iterator::{Iter, IterMut};

/// Holds all positions of the atoms.
pub struct Positions<T, const D: usize> {
    data: Vec<SVector<T, D>>,
}

/// Holds all rescaled positions of the atoms.
pub struct ScaledPositions<T, const D: usize> {
    data: Vec<SVector<T, D>>,
}

/// Holds all velocities of the atoms.
pub struct Velocities<T, const D: usize> {
    data: Vec<SVector<T, D>>,
}

/// Holds all forces of the atoms.
pub struct Forces<T, const D: usize> {
    data: Vec<SVector<T, D>>,
}

/// Holds all accelerations of the atoms.
pub struct Accelerations<T, const D: usize> {
    data: Vec<SVector<T, D>>,
}

impl<T, const D: usize> Positions<T, D> {}

macro_rules! generate_iterators {
    ($type_name: ident) => {
        impl<T, const D: usize> $type_name<T, D> {
            /// Returns an iterator over its data.
            pub fn iter(&self) -> Iter<SVector<T, D>> {
                Iter::new(self.data.iter())
            }

            /// Returns a mutable iterator over its data.
            pub fn iter_mut(&mut self) -> IterMut<SVector<T, D>> {
                IterMut::new(self.data.iter_mut())
            }
        }

        impl<T, const D: usize> IntoIterator for $type_name<T, D> {
            type Item = SVector<T, D>;
            type IntoIter = std::vec::IntoIter<Self::Item>;

            /// Returns a iterator over its data.
            fn into_iter(self) -> Self::IntoIter {
                self.data.into_iter()
            }
        }
    };
}

generate_iterators!(Positions);
generate_iterators!(ScaledPositions);
generate_iterators!(Velocities);
generate_iterators!(Forces);
generate_iterators!(Accelerations);
