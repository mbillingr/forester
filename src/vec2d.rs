//! A row-major contiguous two-dimensional array type, written `Vec2D<T>`.
//!
//! # Examples
//!
//! Create an empty [`Vec2D<T>`] with [`new`]:
//!
//! ```
//! # use forester::vec2d::Vec2D;
//! let x: Vec2D<i32> = Vec2D::new();
//! ```
//!
//! Or by taking ownership of a `Vec<T>` and specifying the number of columns:
//!
//! ```
//! # use forester::vec2d::Vec2D;
//! let x = Vec2D::from_vec(vec![11, 12, 13, 24, 25, 26], 2);
//! ```
//!
//! Or by copying data from a slice:
//!
//! ```
//! # use forester::vec2d::Vec2D;
//! let x = Vec2D::from_slice(&[11, 12, 21, 22, 31, 32], 3);
//! ```
//!
//! Note: The length of the vector/slice must be an integer multiple of the number of columns.

use std::ops;
use std::slice;

/// A row-major contiguous two-dimensional array.
#[derive(Debug)]
pub struct Vec2D<T> {
    data: Vec<T>,
    n_columns: usize,
}

impl<T> Vec2D<T> {
    /// Construct a new empty Vec2D
    pub fn new() -> Self {
        Vec2D {
            data: Vec::new(),
            n_columns: 0,
        }
    }

    /// Take ownership of a `Vec<T>` and reinterpret as two-dimensional data.
    ///
    /// Panics if `data.len()` is not an integer multiple of `n_columns`.
    pub fn from_vec(data: Vec<T>, n_columns: usize) -> Self {
        assert_eq!(0, data.len() % n_columns);
        Vec2D { data, n_columns }
    }

    /// Return number of columns
    #[inline(always)]
    pub fn n_cols(&self) -> usize {
        self.n_columns
    }

    /// Return number of rows
    #[inline(always)]
    pub fn n_rows(&self) -> usize {
        if self.data.is_empty() {
            0
        } else {
            self.data.len() / self.n_columns
        }
    }

    /// Iterate over rows
    pub fn iter<'a>(&'a self) -> slice::Chunks<'a, T> {
        self.data.chunks(self.n_columns)
    }
}

impl<T: Clone> Vec2D<T> {
    /// Copy data from a slice and reinterpret as two-dimensional.
    ///
    /// Panics if `x.len()` is not an integer multiple of `n_columns`.
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

impl<T> ops::Index<(usize, usize)> for Vec2D<T> {
    type Output = T;

    fn index(&self, (r, c): (usize, usize)) -> &Self::Output {
        &self.data[r * self.n_columns + c]
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let x: Vec2D<f64> = Vec2D::new();
        assert_eq!(x.n_cols(), 0);
        assert_eq!(x.n_rows(), 0);

        let y = Vec2D::from_slice(&[1, 2, 3, 4], 1);
        assert_eq!(y.n_cols(), 1);
        assert_eq!(y.n_rows(), 4);

        let y = Vec2D::from_slice(&[1, 2, 3, 4], 2);
        assert_eq!(y.n_cols(), 2);
        assert_eq!(y.n_rows(), 2);

        let y = Vec2D::from_slice(&[1, 2, 3, 4], 4);
        assert_eq!(y.n_cols(), 4);
        assert_eq!(y.n_rows(), 1);

        let z = Vec2D::from_vec(vec![1, 2, 3, 4], 2);
        assert_eq!(z.n_cols(), 2);
        assert_eq!(z.n_rows(), 2);
        assert_eq!(z[(1, 0)], 3);
    }

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
    #[should_panic]
    fn empty_iter() {
        let x: Vec2D<u32> = Vec2D::new();
        x.iter();
    }

    #[test]
    fn indexing() {
        let x: Vec2D<i32> = Vec2D::from_slice(&vec!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 3);
        assert_eq!(x.n_rows(), 4);

        assert_eq!(x[0], [1, 2, 3]);
        assert_eq!(x[1], [4, 5, 6]);
        assert_eq!(x[2], [7, 8, 9]);
        assert_eq!(x[3], [10, 11, 12]);

        assert_eq!(x[(0, 0)], 1);
        assert_eq!(x[(1, 1)], 5);
        assert_eq!(x[(2, 2)], 9);
        assert_eq!(x[(0, 2)], 3);
        assert_eq!(x[(3, 0)], 10);
    }
}
