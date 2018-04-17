
use std::ops;
use std::slice;

#[derive(Debug)]
pub struct Vec2D<T> {
    data: Vec<T>,
    n_columns: usize,
}

impl<T> Vec2D<T> {
    pub fn new() -> Vec2D<T> {
        Vec2D {
            data: Vec::new(),
            n_columns: 0,
        }
    }

    #[inline(always)]
    pub fn n_cols(&self) -> usize {
        self.n_columns
    }

    #[inline(always)]
    pub fn n_rows(&self) -> usize {
        self.data.len() / self.n_columns
    }

    pub fn iter<'a>(&'a self) -> slice::Chunks<'a, T> {
        self.data.chunks(self.n_columns)
    }
}

impl<T: Clone> Vec2D<T> {
    pub fn from_slice(x: &[T], n_columns: usize) -> Vec2D<T> {
        assert_eq!(0, x.len() % n_columns);
        Vec2D {
            data: x.into(),
            n_columns,
        }
    }
}

impl<'a, T> IntoIterator for &'a Vec2D<T> {
    type Item = &'a [T];
    type IntoIter = slice::Chunks<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        if self.data.len() == 0 {
        }
        self.data.chunks(self.n_columns)
    }
}

impl<T> ops::Index<usize> for Vec2D<T> {
    type Output = [T];

    fn index(&self, idx: usize) -> &Self::Output {
        let a = idx * self.n_columns;
        let b = a + self.n_columns;
        &self.data[a..b]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iteration() {
        let x: Vec2D<i32> = Vec2D::from_slice(&vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 4);
        assert_eq!(x.n_rows(), 3);

        let mut i = x.into_iter();
        assert_eq!(i.next(), Some([1, 2, 3, 4].as_ref()));
        assert_eq!(i.next(), Some([5, 6, 7, 8].as_ref()));
        assert_eq!(i.next(), Some([9, 10, 11, 12].as_ref()));
        assert_eq!(i.next(), None);
        assert_eq!(x.into_iter().next(), Some([1, 2, 3, 4].as_ref()));
    }

    #[test]
    fn indexing() {
        let x: Vec2D<i32> = Vec2D::from_slice(&vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 3);
        assert_eq!(x.n_rows(), 4);

        assert_eq!(x[0], [1, 2, 3]);
        assert_eq!(x[1], [4, 5, 6]);
        assert_eq!(x[2], [7, 8, 9]);
        assert_eq!(x[3], [10, 11, 12]);
    }
}
