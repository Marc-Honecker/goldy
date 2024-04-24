use derive_builder::Builder;
use uuid::Uuid;

use crate::Real;

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, Builder)]
#[builder(build_fn(validate = "Self::validate"))]
/// Determines the type of the atom and stores per atom data like
/// the mass and the damping.
pub struct AtomType<T: Real> {
    #[builder(setter(skip), default = "Uuid::new_v4()")]
    id: Uuid,
    mass: T,
    damping: T,
}

impl<T: Real> AtomType<T> {
    /// Returns the mass of this atom-type.
    pub fn mass(&self) -> T {
        self.mass
    }

    /// Returns the damping of this atom-type.
    /// Please note, that the damping is *not* rescaled by the time-step.
    pub fn damping(&self) -> T {
        self.damping
    }

    /// Returns the ID of this atom-type.
    pub fn id(&self) -> Uuid {
        self.id
    }
}

impl<T: Real> AtomTypeBuilder<T> {
    fn validate(&self) -> Result<(), String> {
        if let Some(mass) = self.mass {
            if mass < T::zero() {
                return Err(format!("Expected mass greater than zero, but got {mass}"));
            }
        }

        if let Some(damping) = self.damping {
            if damping < T::zero() {
                return Err(format!(
                    "Expected damping greater than zero, but got {damping}"
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomtype_builder() {
        // easy case
        let at = AtomTypeBuilder::default()
            .damping(0.01)
            .mass(39.95)
            .build()
            .unwrap();

        // everything should have worked well, so we can compare
        // with an new entity (but with the same id).
        assert_eq!(
            at,
            AtomType {
                id: at.id,
                mass: 39.95,
                damping: 0.01
            }
        );

        // with an inappropriate mass
        let failed_at = AtomTypeBuilder::default()
            .damping(0.01)
            .mass(-1.0)
            .build()
            .unwrap_err();

        // the mass was wrong, so we should get an error message
        // stating it was wrong
        assert_eq!(
            &failed_at.to_string(),
            "Expected mass greater than zero, but got -1"
        );

        // with an inappropriate damping
        let other_failed = AtomTypeBuilder::default()
            .damping(-2.0)
            .mass(39.95)
            .build()
            .unwrap_err();

        // and here was the damping wrong, so the error message
        // should tell it that
        assert_eq!(
            &other_failed.to_string(),
            "Expected damping greater than zero, but got -2"
        );
    }

    #[test]
    fn test_atom_type_getter() {
        // Building a valide atom-type.
        let at = AtomTypeBuilder::default()
            .mass(39.95)
            .damping(0.01)
            .build()
            .unwrap();

        // The mass and damping have to match, but the id is random,
        // so testing is difficult.
        assert_eq!(at.mass(), 39.95);
        assert_eq!(at.damping(), 0.01);
    }
}
