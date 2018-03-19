extern crate rand;

use std::cmp;
use std::f64;
use std::iter;
use std::marker::PhantomData;
use std::ops;
use std::slice;

use rand::Rng;

pub mod array_ops;
pub mod criteria;
pub mod datasets;
pub mod features;
pub mod get_item;
pub mod predictors;
pub mod splitters;
pub mod tree;
pub mod vec2d;

/// The side of a split
pub enum Side {
    Left,
    Right,
}

pub trait Float {
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(v: usize) -> Self;
}

impl Float for f32 {
    fn zero() -> Self {0.0}
    fn one() -> Self {0.0}
    fn from_usize(v: usize) -> Self {
        v as f32
    }
}

impl Float for f64 {
    fn zero() -> Self {0.0}
    fn one() -> Self {0.0}
    fn from_usize(v: usize) -> Self {
        v as f64
    }
}

pub trait Feature<X> {
    type Theta;
    type F: cmp::PartialOrd;
    fn get_feature(x: &X, theta: &Self::Theta) -> Self::F;
}

pub trait Sample {
    type Theta;
    type F: cmp::PartialOrd;
    type FX: Feature<Self::X, Theta=Self::Theta, F=Self::F>;
    type X;
    type Y;
    fn get_x(&self) -> Self::X;
    fn get_y(&self) -> Self::Y;
}

pub trait DataSet {
    type Theta;
    type F: cmp::PartialOrd;
    type FX: Feature<Self::X, Theta=Self::Theta, F=Self::F>;
    type X;
    type Y;
    type Item: Sample<Theta=Self::Theta, F=Self::F, FX=Self::FX, X=Self::X, Y=Self::Y>;

    fn n_samples(&self) -> usize;
    fn sort_by_feature(&mut self, theta: &Self::Theta);
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

    fn sort_by_feature(&mut self, theta: &Self::Theta) {
        self.sort_unstable_by(|sa, sb| {
            let fa = S::FX::get_feature(&sa.get_x(), theta);
            let fb = S::FX::get_feature(&sb.get_x(), theta);
            // TODO: We probably don't want to panic on e.g. NaN values -- but how shall we treat them?
            match fa.partial_cmp(&fb) {
                Some(o) => o,
                None => panic!("unable to compare features: ")//{:?} < {:?}", fa, fb)
            }
        });
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
    fn prob(&self, s: &Self::S) -> f64;
}

/// Splits data at a tree node. This is a marker trait, shared by more specialized Splitters.
pub trait Splitter {
    type D: ?Sized + DataSet;
    fn new_random<R: Rng>(x: &Self::D, rng: &mut R) -> Self;
}

/// Assigns a sample to either side of the split.
pub trait DeterministicSplitter: Splitter {
    //fn split(&self, f: &<Self::F as FeatureSet>::Sample::Output) -> Side;
    fn split(&self, s: &<Self::D as DataSet>::Item) -> Side;
}

/// Assigns a sample to both sides of the split with some probability each.
pub trait ProbabilisticSplitter: Splitter {
    /// Probability that the sample belongs to the left side of the split
    fn p_left(&self, s: &<Self::D as DataSet>::Item) -> f64;

    /// Probability that the sample belongs to the right side of the split
    fn p_right(&self, s: &<Self::D as DataSet>::Item) -> f64 { 1.0 - self.p_left(s) }
}


#[cfg(test)]
mod tests {
}
