use std::ops::{Deref, DerefMut};

use nalgebra::SVector;

use crate::{units::Mass, Float};

#[derive(Debug, Clone, Copy)]
pub struct Position<const D: usize> {
    value: SVector<Float, D>,
}

#[derive(Debug, Clone, Copy)]
pub struct Velocity<const D: usize> {
    value: SVector<Float, D>,
}

#[derive(Debug, Clone, Copy)]
pub struct Force<const D: usize> {
    value: SVector<Float, D>,
}

#[derive(Debug, Clone, Copy)]
pub struct Accelaration<const D: usize> {
    value: SVector<Float, D>,
}

impl<const D: usize> Force<D> {
    #[inline]
    pub fn as_accelaration(&self, mass: &Mass) -> Accelaration<D> {
        Accelaration::new(self.value / **mass)
    }

    #[inline]
    pub fn to_accelaration(self, mass: &Mass) -> Accelaration<D> {
        Accelaration::new(self.value / **mass)
    }
}

impl<const D: usize> Accelaration<D> {
    #[inline]
    pub fn as_force(&self, mass: &Mass) -> Force<D> {
        Force::new(self.value * **mass)
    }

    #[inline]
    pub fn to_force(self, mass: &Mass) -> Force<D> {
        Force::new(self.value * **mass)
    }
}

macro_rules! generate_new_from_iterator {
    ($struct_type: ident) => {
        impl<const D: usize> $struct_type<D> {
            #[inline]
            pub fn new(value: SVector<Float, D>) -> Self {
                Self { value }
            }

            #[inline]
            pub fn from_iterator<I>(iter: I) -> Self
            where
                I: IntoIterator<Item = Float>,
            {
                Self {
                    value: SVector::from_iterator(iter),
                }
            }

            #[inline]
            pub fn zeros() -> Self {
                Self {
                    value: SVector::zeros(),
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
        impl<const D: usize> Deref for $type_name<D> {
            type Target = SVector<Float, D>;

            #[inline]
            fn deref(&self) -> &Self::Target {
                &self.value
            }
        }

        impl<const D: usize> DerefMut for $type_name<D> {
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
        impl<const D: usize> AsRef<SVector<Float, D>> for $struct_type<D> {
            #[inline]
            fn as_ref(&self) -> &SVector<Float, D> {
                &self.value
            }
        }

        impl<const D: usize> AsMut<SVector<Float, D>> for $struct_type<D> {
            #[inline]
            fn as_mut(&mut self) -> &mut SVector<Float, D> {
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
        impl<const D: usize> std::fmt::Display for $struct_type<D> {
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
        impl<const D: usize> From<SVector<Float, D>> for $struct_type<D> {
            fn from(value: SVector<Float, D>) -> Self {
                $struct_type::new(value)
            }
        }
    };
}

generate_from!(Position);
generate_from!(Velocity);
generate_from!(Force);
generate_from!(Accelaration);
