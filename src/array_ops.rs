//! Various operations on arrays, slices, and vectors.

use std::iter;
use std::mem;
use std::ops;

/// Trait that defines a dot product
pub trait Dot<B: ?Sized> {
    type Output;
    fn dot(&self, other: &B) -> Self::Output;
}

impl<T> Dot<[T]> for [T]
    where T: ops::Mul + Clone,
          T::Output: iter::Sum
{
    type Output = T::Output;

    fn dot(&self, other: &[T]) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
    }
}

impl<T> Dot<[T]> for Vec<T>
    where T: ops::Mul + Clone,
          T::Output: iter::Sum
{
    type Output = T::Output;

    fn dot(&self, other: &[T]) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
    }
}

macro_rules! impl_dot_array_for_array {
    ( $( $size:expr ),* ) => {
        $(
            impl<T> Dot<[T; $size]> for [T; $size]
                where T: ops::Mul + Clone,
                      T::Output: iter::Sum
            {
                type Output = T::Output;
                fn dot(&self, other: &Self) -> Self::Output {
                    self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
                }
            }
        )*
    };
}

impl_dot_array_for_array!{ 1,  2,  3,  4,  5,  6,  7,  8}
impl_dot_array_for_array!{ 9, 10, 11, 12, 13, 14, 15, 16}
impl_dot_array_for_array!{17, 18, 19, 20, 21, 22, 23, 24}
impl_dot_array_for_array!{25, 26, 27, 28, 29, 30, 31, 32}

macro_rules! impl_dot_slice_for_array {
    ( $( $size:expr ),* ) => {
        $(
            impl<T> Dot<[T]> for [T; $size]
                where T: ops::Mul + Clone,
                      T::Output: iter::Sum
            {
                type Output = T::Output;
                fn dot(&self, other: &[T]) -> Self::Output {
                    assert_eq!($size, other.len());
                    self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
                }
            }
        )*
    };
}

impl_dot_slice_for_array!{ 1,  2,  3,  4,  5,  6,  7,  8}
impl_dot_slice_for_array!{ 9, 10, 11, 12, 13, 14, 15, 16}
impl_dot_slice_for_array!{17, 18, 19, 20, 21, 22, 23, 24}
impl_dot_slice_for_array!{25, 26, 27, 28, 29, 30, 31, 32}

macro_rules! impl_dot_array_for_slice {
    ( $( $size:expr ),* ) => {
        $(
            impl<T> Dot<[T; $size]> for [T]
                where T: ops::Mul + Clone,
                      T::Output: iter::Sum
            {
                type Output = T::Output;
                fn dot(&self, other: &[T; $size]) -> Self::Output {
                    assert_eq!($size, self.len());
                    self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
                }
            }
        )*
    };
}

impl_dot_array_for_slice!{ 1,  2,  3,  4,  5,  6,  7,  8}
impl_dot_array_for_slice!{ 9, 10, 11, 12, 13, 14, 15, 16}
impl_dot_array_for_slice!{17, 18, 19, 20, 21, 22, 23, 24}
impl_dot_array_for_slice!{25, 26, 27, 28, 29, 30, 31, 32}

/// In-place partitioning
pub trait Partition<T> {
    /// Partition `self` in place so that all elements for which the predicate holds are placed in
    /// the beginning. Return index of first element for which the predicate does not hold.
    fn partition<F: FnMut(&T) -> bool>(&mut self, predicate: F) -> usize;
}

impl<T> Partition<T> for [T]
{
    /// Partition a slice in place in O(n).
    fn partition<F: FnMut(&T) -> bool>(&mut self, mut predicate: F) -> usize {
        // Using unsafe pointer arithmetic is 50% faster than safe indexing
        unsafe {
            let start = &mut self[0] as *mut T;
            let end = start.offset(self.len() as isize);
            let mut target = &mut self[0] as *mut T;
            let mut current = target;
            let mut count = 0;
            while current < end {
                if predicate(&*current) {
                    mem::swap(&mut *target, &mut *current);
                    target = target.offset(1);
                    count += 1;
                }
                current = current.offset(1);
            }
            count
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_vec() {
        let a = vec!(1.0, 2.0, 3.0);
        let b = vec!(2.0, 1.0, 1.0);
        assert_eq!(a.dot(&b), 7.0);

        let a = vec!(1, 2, 1);
        let b = vec!(2, 0, -3);
        assert_eq!(a.dot(&b), -1);

        let a = vec!(1, 2, -1);
        let b = vec!(2, 1, 4);
        assert_eq!(a.dot(&b), 0);
    }

    #[test]
    fn dot_array() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 1.0, 1.0];
        assert_eq!(a.dot(&b), 7.0);

        let a = [1, 2, 1];
        let b = [2, 0, -3];
        assert_eq!(a.dot(&b), -1);

        let a = [1, 2, -1];
        let b = [2, 1, 4];
        assert_eq!(a.dot(&b), 0);
    }

    #[test]
    fn dot_slice() {
        let a: &[_] = &[1.0, 2.0, 3.0];
        let b: &[_] = &[2.0, 1.0, 1.0];
        assert_eq!(a.dot(b), 7.0);

        let a: &[_] = &[1, 2, 1];
        let b: &[_] = &[2, 0, -3];
        assert_eq!(a.dot(b), -1);

        let a: &[_] = &[1, 2, -1];
        let b: &[_] = &[2, 1, 4];
        assert_eq!(a.dot(b), 0);
    }

    #[test]
    fn partition() {
        let mut x = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        let i = x.partition(|&xi| xi < 5);
        assert_eq!(i, 5);
        assert!(x[..i].iter().all(|&xi| xi < 5));
        assert!(x[i..].iter().all(|&xi| xi >= 5));

        // correctly partitioned sequence stays unchanged
        let mut x = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let i = x.partition(|&xi| xi <= 3);
        assert_eq!(i, 4);
        assert_eq!(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
