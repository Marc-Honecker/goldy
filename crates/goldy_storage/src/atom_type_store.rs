use goldy_core::Real;

use crate::{atom_type::AtomType, iterator::Iter};

#[derive(Debug, Default, Clone)]
pub struct AtomTypeStore<T: Real> {
    // The data-layout will most likely change!
    pub(crate) data: Vec<AtomType<T>>,
}

impl<T: Real> AtomTypeStore<T> {
    /// Returns an iterator over the `AtomType`s.
    pub fn iter(&self) -> Iter<AtomType<T>> {
        Iter::new(self.data.iter())
    }
}

impl<T: Real> IntoIterator for AtomTypeStore<T> {
    type Item = AtomType<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T: Real> IntoIterator for &'a AtomTypeStore<T> {
    type Item = &'a AtomType<T>;
    type IntoIter = crate::iterator::Iter<'a, AtomType<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Default)]
pub struct AtomTypeStoreBuilder<T: Real> {
    data: Vec<AtomType<T>>,
}

impl<T: Real> AtomTypeStoreBuilder<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Adds a single `AtomType` at a time.
    pub fn add(&mut self, atom_type: AtomType<T>) -> &mut Self {
        self.data.push(atom_type);

        self
    }

    /// Adds a bunch of `AtomType`s all at once.
    pub fn add_many(&mut self, atom_type: AtomType<T>, n: usize) -> &mut Self {
        // Propably a bit inefficient, but that can be changed later on.
        (0..n).for_each(|_| self.data.push(atom_type));

        self
    }

    /// Builds the resulting `AtomTypeStore`.
    pub fn build(&mut self) -> AtomTypeStore<T> {
        AtomTypeStore {
            data: Clone::clone(&self.data),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::atom_type::AtomTypeBuilder;

    use super::*;

    #[test]
    fn test_atom_type_store() {
        // First, we create a proper Argon atom.
        let argon = AtomTypeBuilder::default()
            .mass(39.95)
            .damping(0.01)
            .build()
            .unwrap();

        // Then, we create a Hydrogen atom.
        let hydrogen = AtomTypeBuilder::default()
            .mass(1.0)
            .damping(0.005)
            .build()
            .unwrap();

        // Now let's create a AtomTypeStore with only one Hydrogen.
        let little_store = AtomTypeStoreBuilder::new().add(hydrogen).build();

        // The store should have exactly one AtomType.
        assert_eq!(little_store.data.len(), 1);

        // Further, the id, mass and damping need to match with the original one.
        little_store.iter().for_each(|&at| {
            assert_eq!(at.id(), hydrogen.id());
            assert_eq!(at.mass(), hydrogen.mass());
            assert_eq!(at.damping(), hydrogen.damping());
        });

        // Now let's create a bigger AtomTypeStore with a bunch of Argon atoms.
        let big_store = AtomTypeStoreBuilder::new().add_many(argon, 1000).build();

        // This store should have now 1000 AtomTypes.
        assert_eq!(big_store.data.len(), 1000);

        // And again, the inner data has to match with our original Argon atom.
        // Since we implemented `IntoIterator` for `&AtomTypeStore`, let's try it out.
        for at in &big_store {
            assert_eq!(at.id(), argon.id());
            assert_eq!(at.mass(), argon.mass());
            assert_eq!(at.damping(), argon.damping());
        }
    }
}
