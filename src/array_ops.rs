use std::iter;
use std::marker::PhantomData;
use std::ops;

use predictors::CategoricalProbabilities;

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

// Todo implement for all kinds of array sizes (macro!)

impl<T> Dot<[T]> for [T;1]
    where T: ops::Mul + Clone,
          T::Output: iter::Sum
{
    type Output = T::Output;

    fn dot(&self, other: &[T]) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
    }
}

impl<T> Dot<[T;1]> for [T;1]
    where T: ops::Mul + Clone,
          T::Output: iter::Sum
{
    type Output = T::Output;

    fn dot(&self, other: &[T;1]) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
    }
}

impl<T> Dot<[T]> for [T;2]
    where T: ops::Mul + Clone,
          T::Output: iter::Sum
{
    type Output = T::Output;

    fn dot(&self, other: &[T]) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
    }
}

impl<T> Dot<[T;2]> for [T;2]
    where T: ops::Mul + Clone,
          T::Output: iter::Sum
{
    type Output = T::Output;

    fn dot(&self, other: &[T;2]) -> Self::Output {
        assert_eq!(self.len(), other.len());
        self.iter().zip(other.iter()).map(|(a, b)| a.clone() * b.clone()).sum()
    }
}

/// In-place partitioning
pub trait Partition<T> {
    /// Partition `self` in place so that all elements for which the predicate holds are placed in
    /// the beginning. Return index of first element for which the predicate does not hold.
    fn partition<F: FnMut(&T) -> bool>(&mut self, predicate: F) -> usize;
}

impl<T> Partition<T> for [T]
{
    /// Partition a slice in place in O(n).
    // TODO: possible speed up with unsafe code? (less bounds checks, presumably)
    fn partition<F: FnMut(&T) -> bool>(&mut self, mut predicate: F) -> usize {
        let mut target_index = 0;
        for i in 0..self.len() {
            if predicate(&self[i]) {
                self.swap(target_index, i);
                target_index += 1;
            }
        }
        target_index
    }
}


pub trait IterMean<A = Self> {
    fn mean<I: Iterator<Item=A>>(iter: I) -> Self;
}

impl IterMean<f64> for f64 {
    fn mean<I: Iterator<Item=f64>>(iter: I) -> Self {
        let mut sum = 0.0;
        let mut n = 0;
        for i in iter {
            sum += i;
            n += 1
        }
        sum / n as f64
    }
}

impl IterMean for CategoricalProbabilities {
    fn mean<I: Iterator<Item=CategoricalProbabilities>>(iter: I) -> Self {
        let mut total = CategoricalProbabilities::new();
        for i in iter {
            total.add(&i);
        }
        total
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
    }

    #[test]
    fn dot_array() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 1.0, 1.0];
        assert_eq!(a.dot(&b), 7.0);

        let a = [1, 2, 1];
        let b = [2, 0, -3];
        assert_eq!(a.dot(&b), -1);
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

    #[test]
    fn mean() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let m: f64 = IterMean::mean(x.into_iter());
        assert_eq!(m, 2.5);
    }
}
