pub struct Iter<'a, T: 'a> {
    data: core::slice::Iter<'a, T>,
}

impl<'a, T: 'a> Iter<'a, T> {
    pub fn new(data: core::slice::Iter<'a, T>) -> Self {
        Self { data }
    }
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.next()
    }
}

pub struct IterMut<'a, T: 'a> {
    data: core::slice::IterMut<'a, T>,
}

impl<'a, T: 'a> IterMut<'a, T> {
    pub fn new(data: core::slice::IterMut<'a, T>) -> Self {
        Self { data }
    }
}

impl<'a, T: 'a> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.data.next()
    }
}

#[cfg(test)]
mod tests {
    use std::ptr;

    use super::*;

    #[test]
    fn test_iter() {
        let vec1 = vec![1, 2, 3, 4];
        let vec2 = vec1.clone();

        let vec1_iter = Iter::new(vec1.iter());

        vec1_iter.zip(&vec2).for_each(|(v1, v2)| {
            assert_eq!(*v1, *v2);
            assert!(!ptr::eq(v1, v2));
        });
    }

    #[test]
    fn test_iter_mut() {
        let mut vec1 = [1, 2, 3, 4];
        let vec2 = vec![2, 4, 6, 8];

        let vec1_iter_mut = IterMut::new(vec1.iter_mut());

        vec1_iter_mut.zip(&vec2).for_each(|(v1, v2)| {
            *v1 *= 2;

            assert_eq!(*v1, *v2);
            assert!(!ptr::eq(v1, v2));
        });
    }
}
