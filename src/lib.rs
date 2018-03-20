extern crate rand;

use std::cmp;
use std::f64;

pub mod array_ops;
pub mod criteria;
pub mod datasets;
pub mod features;
pub mod get_item;
pub mod predictors;
pub mod splitters;
pub mod tree;
//pub mod vec2d;

use array_ops::Partition;

type Real = f64;

trait RealConstants {
    #[inline(always)] fn pi() -> Real;
    #[inline(always)] fn zero() -> Real {0.0}
    #[inline(always)] fn one() -> Real {1.0}
}

impl RealConstants for Real {
    #[inline(always)] fn pi() -> Real {f64::consts::PI}
}

/// The side of a split
#[derive(Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

pub trait Feature<X> {
    type Theta;
    type F: cmp::PartialOrd + Clone;
    fn get_feature(x: &X, theta: &Self::Theta) -> Self::F;
    fn random(x: &X) -> Self::Theta;
}

pub trait Sample {
    type Theta;
    type F: cmp::PartialOrd + Clone;
    type FX: Feature<Self::X, Theta=Self::Theta, F=Self::F>;
    type X;
    type Y;
    fn get_x(&self) -> Self::X;
    fn get_y(&self) -> Self::Y;
}

pub trait DataSet {
    type Theta;
    type F: cmp::PartialOrd + Clone;
    type FX: Feature<Self::X, Theta=Self::Theta, F=Self::F>;
    type X;
    type Y;
    type Item: Sample<Theta=Self::Theta, F=Self::F, FX=Self::FX, X=Self::X, Y=Self::Y>;

    fn n_samples(&self) -> usize;
    fn partition_by_split<S: DeterministicSplitter<D=Self>>(&mut self, s: &S) -> usize;

    fn subsets(&mut self, i: usize) -> (&mut Self, &mut Self);

    fn random_feature(&self) -> Self::Theta;

    fn reduce_feature<B, F: FnMut(B, Self::F) -> B>(&self, theta: &Self::Theta, init: B, f: F) -> B;
}

impl<S> DataSet for [S]
where S: Sample,
      //S::F: fmt::Debug,
{
    type Theta = S::Theta;
    type F = S::F;
    type FX = S::FX;
    type X = S::X;
    type Y = S::Y;
    type Item = S;

    fn n_samples(&self) -> usize {
        self.len()
    }

    fn partition_by_split<SP: DeterministicSplitter<D=Self>>(&mut self, split: &SP) -> usize {
        self.partition(|sample| split.split(&sample.get_x()) == Side::Left)
    }

    fn subsets(&mut self, i: usize) -> (&mut Self, &mut Self) {
        self.split_at_mut(i)
    }

    fn random_feature(&self) -> Self::Theta {
        S::FX::random(&self[0].get_x())
    }

    fn reduce_feature<B, F: FnMut(B, Self::F) -> B>(&self, theta: &Self::Theta, init: B, f: F) -> B {
        self.iter().map(|s| Self::FX::get_feature(&s.get_x(), theta)).fold(init, f)
    }
}

/*pub trait FeatureSet {
    type Item: ?Sized + Sample;
    fn n_samples(&self) -> usize;
    fn get_sample(&self, n: usize) -> &Self::Item;
    fn random_feature<R: Rng>(&self, rng: &mut R) -> <Self::Item as Sample>::Theta;
    fn minmax(&self, theta: &<Self::Item as Sample>::Theta) -> Option<(<Self::Item as Sample>::Feature, <Self::Item as Sample>::Feature)>;

    fn for_each_mut<F: FnMut(&Self::Item)>(&self, f: F);
    #[inline] fn for_each<F: Fn(&Self::Item)>(&self, f: F) { self.for_each_mut(f) }
}

pub trait OutcomeVariable {
    type Item: ?Sized;
    fn n_samples(&self) -> usize;
    fn for_each_mut<F: FnMut(&Self::Item)>(&self, f: F);
    #[inline] fn for_each<F: Fn(&Self::Item)>(&self, f: F) { self.for_each_mut(f) }
}*/
/*
/// Type has a length
pub trait FixedLength {
    fn len(&self) -> usize;
}

pub trait Shape2D {
    fn n_rows(&self) -> usize;
    fn n_cols(&self) -> usize;
}

impl<'a, T> FixedLength for &'a [T] {
    fn len(&self) -> usize {
        (self as &[T]).len()
    }
}

impl<'a, T> FixedLength for [T] {
    fn len(&self) -> usize {
        self.len()
    }
}
*/
/// For comparing splits
pub trait SplitCriterion {
    type D: ?Sized + DataSet;
    type C: ?Sized + cmp::PartialOrd + Copy;
    fn calc_presplit(y: &Self::D) -> Self::C;
    fn calc_postsplit(yl: &Self::D, yr: &Self::D) -> Self::C;
}

/// Prediction of the final Leaf value.
pub trait LeafPredictor
{
    type S: Sample;
    type D: ?Sized + DataSet;

    /// predicted value
    fn predict(&self, s: &<Self::S as Sample>::X) -> <Self::S as Sample>::Y;

    /// fit predictor to data
    fn fit(data: &Self::D) -> Self;
}

/// The probabilistic leaf predictor models uncertainty in the prediction.
pub trait ProbabilisticLeafPredictor: LeafPredictor
{
    /// probability of given output `p(y|x)`
    fn prob(&self, s: &Self::S) -> Real;
}

/// Splits data at a tree node. This is a marker trait, shared by more specialized Splitters.
pub trait Splitter {
    type D: ?Sized + DataSet;
    fn theta(&self) -> &<Self::D as DataSet>::Theta;
}

/// Assigns a sample to either side of the split.
pub trait DeterministicSplitter: Splitter {
    //fn split(&self, f: &<Self::F as FeatureSet>::Sample::Output) -> Side;
    fn split(&self, x: &<Self::D as DataSet>::X) -> Side;
}

/// Assigns a sample to both sides of the split with some probability each.
pub trait ProbabilisticSplitter: Splitter {
    /// Probability that the sample belongs to the left side of the split
    fn p_left(&self, x: &<Self::D as DataSet>::X) -> Real;

    /// Probability that the sample belongs to the right side of the split
    fn p_right(&self, x: &<Self::D as DataSet>::X) -> Real { 1.0 - self.p_left(x) }
}

trait RandomSplit<S: Splitter> {
    fn new_random(data: &S::D) -> S;
}

/// Find split
trait SplitFitter {
    type D: ?Sized + DataSet;
    type Split: Splitter<D=Self::D>;
    type Criterion: SplitCriterion<D=Self::D>;
    fn find_split(&self, data: &mut Self::D) -> Option<Self::Split>;
}


#[cfg(test)]
mod tests {
}
