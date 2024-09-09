use std::fmt::Display;

use nalgebra::SVector;

use crate::storage::atom_type_store::AtomTypeStore;
use crate::storage::iterator::{Iter, IterMut};
use crate::Real;

macro_rules! generate_structs {
    ($type_name: ident) => {
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct $type_name<T, const D: usize> {
            pub(super) data: Vec<SVector<T, D>>,
        }

        impl<T, const D: usize> $type_name<T, D> {
            /// Returns an iterator over the data.
            pub fn iter(&self) -> Iter<SVector<T, D>> {
                Iter::new(self.data.iter())
            }

            /// Returns a mutable iterator over the data.
            pub fn iter_mut(&mut self) -> IterMut<SVector<T, D>> {
                IterMut::new(self.data.iter_mut())
            }

            /// Returns the length of the data
            pub fn len(&self) -> usize {
                self.data.len()
            }

            /// Tests, if the data is empty.
            pub fn is_empty(&self) -> bool {
                self.data.is_empty()
            }

            /// Returns the element at the given index.
            pub fn get_by_idx(&self, idx: usize) -> &SVector<T, D> {
                &self.data[idx]
            }
        }

        impl<T: Real, const D: usize> $type_name<T, D> {
            /// Creates `n` entries, all set to the zero vector.
            pub fn zeros(n: usize) -> Self {
                Self {
                    data: (0..n).map(|_| SVector::zeros()).collect(),
                }
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

        impl<T: Real + Display, const D: usize> $type_name<T, D> {
            /// Converts the data into a string representation.
            #[allow(unused)]
            pub(crate) fn convert_to_string(&self, atom_types: &AtomTypeStore<T>) -> String {
                let mut contents = String::new();

                self.data
                    .iter()
                    .zip(atom_types)
                    .enumerate()
                    .for_each(|(id, (x, at))| {
                        contents.push_str(format!("{: >10}{: >10}", id + 1, at.id()).as_str());

                        x.iter()
                            .for_each(|x| contents.push_str(format!("{x: >50.6}").as_str()));
                        contents.push('\n');
                    });

                contents
            }
        }
    };
}

generate_structs!(Positions);
generate_structs!(Velocities);
generate_structs!(Forces);

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;

    use crate::storage::atom_type::AtomTypeBuilder;
    use crate::storage::atom_type_store::AtomTypeStoreBuilder;

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
        let pos = Positions {
            data: vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.5, 0.5, 0.5),
            ],
        };
        let reference = pos.clone();

        let vel = pos
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

    #[test]
    fn test_get_by_index() {
        let pos = Positions {
            data: vec![
                Vector3::new(1.0, 1.0, 1.0),
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(0.5, 0.5, 0.5),
            ],
        };

        assert_eq!(pos.get_by_idx(0), &Vector3::new(1.0, 1.0, 1.0));
        assert_eq!(pos.get_by_idx(1), &Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(pos.get_by_idx(2), &Vector3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_convert_to_string() {
        let pos = Positions {
            data: vec![
                Vector3::new(1.0, 2.0, 3.0),
                Vector3::new(4.0, 5.0, 6.0),
                Vector3::new(7.0, 8.0, 9.0),
            ],
        };

        let atom_types = AtomTypeStoreBuilder::default()
            .add_many(
                AtomTypeBuilder::default()
                    .id(1)
                    .mass(39.95)
                    .damping(0.01)
                    .build()
                    .unwrap(),
                3,
            )
            .build();

        assert_eq!(
            pos.convert_to_string(&atom_types),
            r"         1         1            1.000000            2.000000            3.000000
         2         1            4.000000            5.000000            6.000000
         3         1            7.000000            8.000000            9.000000
"
        );
    }
}
