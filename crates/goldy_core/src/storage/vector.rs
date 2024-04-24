use nalgebra::SVector;

use crate::storage::iterator::{Iter, IterMut};

/// Holds all positions of the atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Positions<T, const D: usize> {
    pub(super) data: Vec<SVector<T, D>>,
}

/// Holds all rescaled positions of the atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScaledPositions<T, const D: usize> {
    pub(super) data: Vec<SVector<T, D>>,
}

/// Holds all velocities of the atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Velocities<T, const D: usize> {
    pub(super) data: Vec<SVector<T, D>>,
}

/// Holds all forces of the atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Forces<T, const D: usize> {
    pub(super) data: Vec<SVector<T, D>>,
}

/// Holds all accelerations of the atoms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Accelerations<T, const D: usize> {
    pub(super) data: Vec<SVector<T, D>>,
}

impl<T, const D: usize> Positions<T, D> {}

macro_rules! generate_iterators {
    ($type_name: ident) => {
        impl<T, const D: usize> $type_name<T, D> {
            /// Returns an iterator over the data.
            pub fn iter(&self) -> Iter<SVector<T, D>> {
                Iter::new(self.data.iter())
            }

            /// Returns a mutable iterator over the data.
            pub fn iter_mut(&mut self) -> IterMut<SVector<T, D>> {
                IterMut::new(self.data.iter_mut())
            }
        }

        impl<T, const D: usize> IntoIterator for $type_name<T, D> {
            type Item = SVector<T, D>;
            type IntoIter = std::vec::IntoIter<Self::Item>;

            /// Returns a iterator over the data.
            fn into_iter(self) -> Self::IntoIter {
                self.data.into_iter()
            }
        }

        impl<'a, T, const D: usize> IntoIterator for &'a $type_name<T, D> {
            type Item = &'a SVector<T, D>;
            type IntoIter = crate::storage::iterator::Iter<'a, SVector<T, D>>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter()
            }
        }

        impl<'a, T, const D: usize> IntoIterator for &'a mut $type_name<T, D> {
            type Item = &'a mut SVector<T, D>;
            type IntoIter = crate::storage::iterator::IterMut<'a, SVector<T, D>>;

            fn into_iter(self) -> Self::IntoIter {
                self.iter_mut()
            }
        }

        impl<A, const D: usize> FromIterator<SVector<A, D>> for $type_name<A, D> {
            fn from_iter<T: IntoIterator<Item = SVector<A, D>>>(iter: T) -> Self {
                Self {
                    data: Vec::from_iter(iter),
                }
            }
        }
    };
}

generate_iterators!(Positions);
generate_iterators!(ScaledPositions);
generate_iterators!(Velocities);
generate_iterators!(Forces);
generate_iterators!(Accelerations);

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use super::*;

    #[test]
    fn test_iter() {
        let pos = Positions {
            data: vec![
                Vector3::new(0, 0, 0),
                Vector3::new(1, 1, 1),
                Vector3::new(2, 2, 2),
            ],
        };
        pos.iter().enumerate().for_each(|(i, &p)| {
            assert_eq!(p, Vector3::new(i, i, i));
        });

        let vel = pos.iter().map(|p| p / 1).collect::<Velocities<usize, 3>>();
        vel.iter()
            .enumerate()
            .for_each(|(i, &v)| assert_eq!(v, Vector3::new(i, i, i)));

        let f = vel.iter().map(|v| v / 2).collect::<Forces<usize, 3>>();
        f.iter()
            .enumerate()
            .for_each(|(i, &f)| assert_eq!(f, Vector3::new(i / 2, i / 2, i / 2)));
    }

    #[test]
    fn test_iter_mut() {
        let mut pos = Positions {
            data: vec![
                Vector3::new(1, 2, 3),
                Vector3::new(4, 5, 6),
                Vector3::new(7, 8, 9),
            ],
        };

        pos.iter_mut().for_each(|p| *p *= 100);

        assert_eq!(
            pos.data,
            vec![
                Vector3::new(100, 200, 300),
                Vector3::new(400, 500, 600),
                Vector3::new(700, 800, 900),
            ]
        );

        // simulating a time-step
        let dt = 0.5;
        let mut vel = Velocities {
            data: vec![
                Vector3::new(1.0, 1.0, 2.0),
                Vector3::new(2.0, 1.0, 1.0),
                Vector3::new(1.0, 2.0, 1.0),
            ],
        };
        let force = Forces {
            data: vec![Vector3::zeros(), Vector3::zeros(), Vector3::zeros()],
        };

        vel.iter_mut().zip(&force).for_each(|(v, f)| *v += f * dt);
    }

    #[test]
    fn test_into_iter() {
        let dt = 0.5;
        let s_pos = ScaledPositions {
            data: vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.5, 0.5, 0.5),
            ],
        };
        let reference = s_pos.clone();

        let vel = s_pos
            .into_iter()
            .map(|p| p / dt)
            .collect::<Velocities<f64, 3>>();

        assert_eq!(
            vel,
            reference
                .iter()
                .map(|r| r * 2.0)
                .collect::<Velocities<f64, 3>>()
        );
    }
}
