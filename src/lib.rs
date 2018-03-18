extern crate rand;

use std::cmp;
use std::f64;
use std::iter;
use std::marker::PhantomData;
use std::ops;
use std::slice;

use rand::Rng;

mod criteria;
mod features;
mod predictors;
mod vec2d;

/// The side of a split
enum Side {
    Left,
    Right,
}

/// Type has a length
trait FixedLength {
    fn len(&self) -> usize;
}

impl<'a, T:Sized> FixedLength for &'a [T] {
    fn len(&self) -> usize {
        (self as &[T]).len()
    }
}

/// For comparing splits
trait SplitCriterion<'a> {
    type Y: ?Sized;
    type C: ?Sized + cmp::PartialOrd + Copy;
    fn calc_presplit(y: &'a Self::Y) -> Self::C;
    fn calc_postsplit(yl: &'a Self::Y, yr: &'a Self::Y) -> Self::C;
}

/// Prediction of the final Leaf value.
trait LeafPredictor<'a, T: 'a>
    where &'a Self::X: IntoIterator,
          &'a Self::Y: IntoIterator<Item=&'a T>
{
    type X: 'a + ? Sized;
    type Y: 'a + ? Sized;

    /// predicted value
    fn predict(&self, x: <&'a Self::X as IntoIterator>::Item) -> T;

    /// fit predictor to data
    fn fit(x: &'a Self::X, y: &'a Self::Y) -> Self;
}

/// The probabilistic leaf predictor models uncertainty in the prediction.
trait ProbabilisticLeafPredictor<'a, T: 'a>: LeafPredictor<'a, T>
    where &'a Self::X: IntoIterator,
          &'a Self::Y: IntoIterator<Item=&'a T>
{
    /// probability of given output `p(y|x)`
    fn prob(&self, x: <&'a Self::X as IntoIterator>::Item, y: <&'a Self::Y as IntoIterator>::Item) -> f64;
}

/// Extract feature from sample.
trait FeatureExtractor {
    type Xi: ? Sized;
    type Fi: ? Sized;
    fn new_random<R: Rng>(x: &Self::Xi, rng: &mut R) -> Self;
    fn extract(&self, x: &Self::Xi) -> Self::Fi;
}

/// Splits data at a tree node. This is a marker trait, shared by more specialized Splitters.
trait Splitter {
    type X: ? Sized;
    type Xi: ? Sized;
    fn new_random(x: &Self::X) -> Self;
}

/// Assigns a sample to either side of the split.
trait DeterministicSplitter: Splitter {
    fn split(&self, x: &Self::Xi) -> Side;
}

/// Assigns a sample to both sides of the split with some probability each.
trait ProbabilisticSplitter: Splitter {
    /// Probability that the sample belongs to the left side of the split
    fn p_left(&self, x: &Self::Xi) -> f64;

    /// Probability that the sample belongs to the right side of the split
    fn p_right(&self, x: &Self::Xi) -> f64 { 1.0 - self.p_left(x) }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trig() {
    }
}
