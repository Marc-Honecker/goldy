use crate::{
    Real,
    storage::{atom_type::AtomType, iterator::Iter},
};

#[derive(Debug, Default, Clone)]
pub struct AtomTypeStore<T: Real> {
    // The data-layout will most likely change!
    data: Vec<AtomType<T>>,
}

impl<T: Real> AtomTypeStore<T> {
    /// Returns the number of types.
    pub fn number_types(&self) -> usize {
        let mut seen_ids = Vec::new();

        self.data.iter().for_each(|at| {
            if !seen_ids.contains(&at.id()) {
                seen_ids.push(at.id())
            }
        });

        seen_ids.len()
    }

    /// Returns all Pairs of `AtomType`.id() and the masses.
    /// For efficiency, if you need the number of types and the
    /// pairs of ids, masses, consider using this method, since
    /// self.number_types() == self.get_masses().len()
    pub fn get_masses(&self) -> Vec<(u32, T)> {
        let mut seen_pairs = Vec::new();

        self.data.iter().for_each(|at| {
            if !seen_pairs.contains(&(at.id(), at.mass())) {
                seen_pairs.push((at.id(), at.mass()))
            }
        });

        seen_pairs
    }

    /// Returns an iterator over the `AtomType`s.
    pub fn iter(&self) -> Iter<AtomType<T>> {
        Iter::new(self.data.iter())
    }

    /// Returns the length of the data
    pub(super) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn get_by_idx(&self, idx: usize) -> &AtomType<T> {
        &self.data[idx]
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
    type IntoIter = Iter<'a, AtomType<T>>;

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
        // Probably a bit inefficient, but that can be changed later on.
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
    use crate::storage::atom_type::AtomTypeBuilder;

    use super::*;

    #[test]
    fn test_atom_type_store() {
        // First, we create a proper Argon atom.
        let argon = AtomTypeBuilder::default()
            .id(0)
            .mass(39.95)
            .damping(0.01)
            .build()
            .unwrap();

        // Then, we create a Hydrogen atom.
        let hydrogen = AtomTypeBuilder::default()
            .id(1)
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

        // We must also be able to get the proper `AtomType` by index.
        assert_eq!(1, little_store.get_by_idx(0).id());
        assert_eq!(1.0, little_store.get_by_idx(0).mass());
        assert_eq!(0.005, little_store.get_by_idx(0).damping());

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

    #[test]
    fn test_number_types() {
        // creating an `AtomTypeStore` with only one type and one atom
        let ats1 = AtomTypeStoreBuilder::default()
            .add(
                AtomTypeBuilder::default()
                    .mass(1.0)
                    .damping(0.01)
                    .id(0)
                    .build()
                    .unwrap(),
            )
            .build();

        assert_eq!(ats1.number_types(), 1);

        // now we create an `AtomTypeStore` with 500 atoms, but still only
        // one type
        let ats2 = AtomTypeStoreBuilder::default()
            .add_many(
                AtomTypeBuilder::default()
                    .mass(39.95)
                    .damping(0.01)
                    .id(1)
                    .build()
                    .unwrap(),
                500,
            )
            .build();

        assert_eq!(ats2.number_types(), 1);

        // finally, we create an `AtomTypeStore` with multiple atoms and
        // multiple types
        let ats3 = AtomTypeStoreBuilder::default()
            .add(
                AtomTypeBuilder::default()
                    .id(0)
                    .mass(1.0)
                    .damping(0.01)
                    .build()
                    .unwrap(),
            )
            .add_many(
                AtomTypeBuilder::default()
                    .id(1)
                    .mass(39.95)
                    .damping(0.01)
                    .build()
                    .unwrap(),
                500,
            )
            .add_many(
                AtomTypeBuilder::default()
                    .id(2)
                    .mass(4.0)
                    .damping(0.01)
                    .build()
                    .unwrap(),
                100,
            )
            .build();

        // here we should get exactly 3 different types
        assert_eq!(ats3.number_types(), 3);
    }
}
