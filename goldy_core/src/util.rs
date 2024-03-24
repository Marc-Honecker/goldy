use std::ops::{Deref, DerefMut};

use nalgebra::{RealField, SVector};

#[derive(Debug, Clone, Copy)]
pub struct Position<T: RealField, const D: usize> {
    value: SVector<T, D>,
}

#[derive(Debug, Clone, Copy)]
pub struct Velocity<T: RealField, const D: usize> {
    value: SVector<T, D>,
}

#[derive(Debug, Clone, Copy)]
pub struct Force<T: RealField, const D: usize> {
    value: SVector<T, D>,
}

#[derive(Debug, Clone, Copy)]
pub struct Accelaration<T: RealField, const D: usize> {
    value: SVector<T, D>,
}

impl<T: RealField, const D: usize> Force<T, D> {
    #[inline]
    pub fn as_accelaration(&self, mass: T) -> Accelaration<T, D> {
        Accelaration::new(self.value.clone() / mass)
    }

    #[inline]
    pub fn to_accelaration(self, mass: T) -> Accelaration<T, D> {
        Accelaration::new(self.value / mass)
    }
}

impl<T: RealField, const D: usize> Accelaration<T, D> {
    #[inline]
    pub fn as_force(&self, mass: T) -> Force<T, D> {
        Force::new(self.value.clone() * mass)
    }

    #[inline]
    pub fn to_force(self, mass: T) -> Force<T, D> {
        Force::new(self.value * mass)
    }
}

macro_rules! generate_new_from_iterator {
    ($struct_type: ident) => {
        impl<T: RealField, const D: usize> $struct_type<T, D> {
            #[inline]
            pub fn new(value: SVector<T, D>) -> Self {
                Self { value }
            }

            #[inline]
            pub fn from_iterator<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = T>,
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
        impl<T: RealField, const D: usize> Deref for $type_name<T, D> {
            type Target = SVector<T, D>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl<T: RealField, const D: usize> DerefMut for $type_name<T, D> {
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
        impl<T: RealField, const D: usize> AsRef<SVector<T, D>> for $struct_type<T, D> {
            #[inline]
            fn as_ref(&self) -> &SVector<T, D> {
                &self.value
            }
        }

        impl<T: RealField, const D: usize> AsMut<SVector<T, D>> for $struct_type<T, D> {
            #[inline]
            fn as_mut(&mut self) -> &mut SVector<T, D> {
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
        impl<T: RealField, const D: usize> std::fmt::Display for $struct_type<T, D> {
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
        impl<T: RealField, const D: usize> From<SVector<T, D>> for $struct_type<T, D> {
            fn from(value: SVector<T, D>) -> Self {
                $struct_type::new(value)
            }
        }
    };
}

generate_from!(Position);
generate_from!(Velocity);
generate_from!(Force);
generate_from!(Accelaration);
