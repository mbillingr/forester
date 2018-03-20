use std::iter;
use std::ops;

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
}
