use std::ops::{Deref, DerefMut};

use nalgebra::{SVector, Vector3};

use crate::units::Mass;

#[derive(Debug, Clone, Copy, Default)]
pub struct Position {
    value: Vector3<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Velocity {
    value: Vector3<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Force {
    value: Vector3<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Accelaration {
    value: Vector3<f64>,
}

impl Force {
    #[inline]
    pub fn as_accelaration(&self, mass: &Mass) -> Accelaration {
        Accelaration::new(self.value / **mass)
    }

    #[inline]
    pub fn to_accelaration(self, mass: &Mass) -> Accelaration {
        Accelaration::new(self.value / **mass)
    }
}

impl Accelaration {
    #[inline]
    pub fn as_force(&self, mass: &Mass) -> Force {
        Force::new(self.value * **mass)
    }

    #[inline]
    pub fn to_force(self, mass: &Mass) -> Force {
        Force::new(self.value * **mass)
    }
}

macro_rules! generate_new_from_iterator {
    ($struct_type: ident) => {
        impl $struct_type {
            #[inline]
            pub fn new(value: Vector3<f64>) -> Self {
                Self { value }
            }

            #[inline]
            pub fn from_iterator<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = f64>,
            {
                Self {
                    value: SVector::from_iterator(iter),
                }
            }
        }
    };
}

generate_new_from_iterator!(Position);
generate_new_from_iterator!(Velocity);
generate_new_from_iterator!(Force);
generate_new_from_iterator!(Accelaration);

macro_rules! generate_deref_deref_mut {
    ($type_name: ident) => {
        impl Deref for $type_name {
            type Target = Vector3<f64>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl DerefMut for $type_name {
            #[inline]
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.value
            }
        }
    };
}

generate_deref_deref_mut!(Position);
generate_deref_deref_mut!(Velocity);
generate_deref_deref_mut!(Force);
generate_deref_deref_mut!(Accelaration);

macro_rules! generate_as_ref_as_mut {
    ($struct_type: ident) => {
        impl AsRef<Vector3<f64>> for $struct_type {
            #[inline]
            fn as_ref(&self) -> &Vector3<f64> {
                &self.value
            }
        }

        impl AsMut<Vector3<f64>> for $struct_type {
            #[inline]
            fn as_mut(&mut self) -> &mut Vector3<f64> {
                &mut self.value
            }
        }
    };
}

generate_as_ref_as_mut!(Position);
generate_as_ref_as_mut!(Velocity);
generate_as_ref_as_mut!(Force);
generate_as_ref_as_mut!(Accelaration);

macro_rules! generate_display {
    ($struct_type: ident) => {
        impl std::fmt::Display for $struct_type {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                std::fmt::Display::fmt(&self.value, f)
            }
        }
    };
}

generate_display!(Position);
generate_display!(Velocity);
generate_display!(Force);
generate_display!(Accelaration);

macro_rules! generate_from {
    ($struct_type: ident) => {
        impl From<Vector3<f64>> for $struct_type {
            fn from(value: Vector3<f64>) -> Self {
                $struct_type::new(value)
            }
        }
    };
}

generate_from!(Position);
generate_from!(Velocity);
generate_from!(Force);
generate_from!(Accelaration);
